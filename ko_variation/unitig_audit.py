"""Annotate the complete DNA unitig catalogue and audit AMR-pattern phylogenetic D.

Standalone downstream analysis: no MI selection, association refit or locus removal.
Exact matches on both strands are retained in every supplied cohort assembly.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import numpy as np

from . import __version__
from .amr_report import candidate_status, label_tokens, read_tsv, write_tsv
from .phenotype import binary_fasta, filter_cohort, json_write
from .workflow_cache import stage_identity, cache_matches, save_cache


def log(message):
    print(f"[unitig audit] {message}", flush=True)


def tsv_gz(path, rows, columns):
    temporary = Path(str(path) + ".partial")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=1) as handle:
        writer = csv.DictWriter(handle, columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def load_unitigs(path):
    """IDs are explicit original zero-based locus IDs, never inferred from row order."""
    sequences = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) != 2:
                raise ValueError(f"Unitig table line {number}: expected ID and DNA sequence")
            locus, sequence = int(fields[0]), fields[1].upper()
            if locus < 0 or locus in sequences or not sequence or set(sequence) - set("ACGT"):
                raise ValueError(f"Invalid/duplicate unitig ID or non-ACGT sequence at line {number}")
            sequences[locus] = sequence
    if not sequences:
        raise ValueError("Empty unitig catalogue")
    return sequences


def prepare_cohort(args):
    """Keep explicit identities, remove metadata non-target species, root a copy only."""
    from Bio import Phylo
    samples, X = binary_fasta(args.fasta)
    loci = list(range(X.shape[1]))
    if args.locus_map:
        rows = read_tsv(args.locus_map, ["fasta_column", "locus"])
        mapping = {int(r["fasta_column"]): int(r["locus"]) for r in rows}
        if len(mapping) != len(rows) or set(mapping) != set(loci):
            raise ValueError("Locus map must cover every FASTA column exactly once")
        loci = [mapping[i] for i in range(X.shape[1])]
    if len(set(loci)) != len(loci) or min(loci) < 0:
        raise ValueError("Locus IDs must be unique and nonnegative")
    metadata = read_tsv(args.metadata, ["id", "isolate", "species"])
    if len({row["id"] for row in metadata}) != len(metadata):
        raise ValueError("Metadata IDs must be unique")
    if args.sample_map:
        sample_map = read_tsv(args.sample_map, ["metadata_id", "sample"])
        known = {row["sample"] for row in sample_map if row["metadata_id"] in {r["id"] for r in metadata}}
    else:
        known = {r["id"]+"_"+r["isolate"].replace(".", "_") for r in metadata}
    if unmatched := set(samples)-known-set(args.exclude_sample):
        raise ValueError(f"{len(unmatched)} FASTA samples lack matched metadata: {sorted(unmatched)[:8]}. Supply --sample-map rather than guessing species")
    original_samples = list(samples)
    samples, X, excluded = filter_cohort(args, samples, X, args.out)
    tree = Phylo.read(args.tree, "newick")
    tips = [tip.name for tip in tree.get_terminals()]
    if None in tips or len(set(tips)) != len(tips):
        raise ValueError("Tree requires nonempty, unique sample IDs")
    missing = set(samples) - set(tips)
    if missing and args.tree_missing_samples == "error":
        raise ValueError(f"Tree missing {len(missing)} FASTA samples: {sorted(missing)[:8]}")
    if missing:
        keep = [i for i, name in enumerate(samples) if name not in missing]
        samples, X = [samples[i] for i in keep], X[keep]
    if len(samples) < 4:
        raise ValueError("At least four matched isolates are required for this audit")
    for node in tree.find_clades():
        if node is not tree.root and (node.branch_length is None or
                not math.isfinite(node.branch_length) or node.branch_length < 0):
            raise ValueError("Tree must have finite nonnegative branch lengths")
    for name in sorted(set(tips) - set(samples)):
        tree.prune(name)
    if args.rooting == "midpoint":
        tree.root_at_midpoint()
    elif not tree.rooted:
        raise ValueError("--rooting as-is requires an explicitly rooted Newick tree")
    tree.root.branch_length = 0.0
    tree.rooted = True
    Phylo.write(tree, args.out / "analysis.treefile", "newick", format_branch_length="%.17g")
    write_tsv(args.out / "sample_inclusion.tsv",
        (dict(sample=name, status="included" if name in samples else "excluded",
              reason="non_target_species_or_explicit" if name in excluded else
                     "tree_missing" if name in missing else "") for name in original_samples),
        ["sample", "status", "reason"])
    write_tsv(args.out / "tree_only_samples.tsv",
        (dict(sample=name, reason="outside_analysis_cohort") for name in sorted(set(tips)-set(samples))),
        ["sample", "reason"])
    log(f"cohort: {len(samples)} isolates; {len(loci):,} binary columns")
    return samples, X, loci


def pattern_tables(args, samples, X, loci):
    """Deduplicate exact bit vectors, not biological sequences or genomic locations."""
    packed = np.packbits(X, axis=0, bitorder="little")
    by_bits, locus_pattern, patterns = {}, {}, {}
    columns = ["pattern_id", "n_present", "n_absent", "maf", "entropy_bits", "passes_maf"]
    for column, locus in enumerate(loci):
        bits = packed[:, column].tobytes()
        if bits not in by_bits:
            pattern_id = f"pattern_{len(by_bits)+1:07d}"
            by_bits[bits] = pattern_id
            n = int(X[:, column].sum())
            p = n / len(samples)
            entropy = -p*math.log2(p)-(1-p)*math.log2(1-p) if 0 < p < 1 else 0.
            patterns[pattern_id] = dict(pattern_id=pattern_id, n_present=n,
                n_absent=len(samples)-n, maf=min(p, 1-p), entropy_bits=entropy,
                passes_maf=int(min(n, len(samples)-n) > args.maf*len(samples)),
                bits=bits, n_unitigs=0)
        pattern_id = by_bits[bits]
        patterns[pattern_id]["n_unitigs"] += 1
        locus_pattern[locus] = pattern_id
        if (column+1) % 50000 == 0:
            log(f"patterns: {column+1:,}/{len(loci):,} columns; {len(patterns):,} unique")
    write_tsv(args.out / "pattern_statistics.tsv", patterns.values(), columns+["n_unitigs"])
    write_tsv(args.out / "column_mapping.tsv",
        (dict(fasta_column=i, locus=locus, pattern_id=locus_pattern[locus]) for i, locus in enumerate(loci)),
        ["fasta_column", "locus", "pattern_id"])
    if not args.no_matrix_tsv:
        def matrix_rows():
            for record in patterns.values():
                states = np.unpackbits(np.frombuffer(record["bits"], dtype=np.uint8), bitorder="little")[:len(samples)]
                yield dict(pattern_id=record["pattern_id"], **dict(zip(samples, states.tolist())))
        tsv_gz(args.out / "pattern_presence_absence.tsv.gz", matrix_rows(), ["pattern_id"]+samples)
    log(f"{len(patterns):,} exact patterns; no MAF or D filtering applied")
    return locus_pattern, patterns


def panaroo_members(path, samples):
    metadata, members = {}, {sample: {} for sample in samples}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "Gene" not in (reader.fieldnames or []):
            raise ValueError("Panaroo CSV lacks Gene column; use the Roary-style CSV")
        available = [s for s in samples if s in reader.fieldnames]
        if not available:
            raise ValueError("No Panaroo sample columns match the analysis FASTA IDs")
        for row in reader:
            cluster = row["Gene"]
            metadata[cluster] = dict(label=row.get("Non-unique Gene name", ""), product=row.get("Annotation", ""))
            for sample in available:
                for feature in re.split(r"[;\t]", row.get(sample, "") or ""):
                    if feature:
                        members[sample].setdefault(feature, set()).add(cluster)
    return metadata, members


GENE_COLUMNS = ["reference", "contig", "feature_id", "gene_id", "label", "product", "gene_start", "gene_end", "strand"]
HIT_COLUMNS = ["locus", "reference", "contig", "hit_start", "hit_end", "hit_strand", "feature_id", "gene_id",
               "label", "product", "gene_start", "gene_end", "strand", "annotation_class", "distance_bp"]


def gene_geometry(gff, sample, membership, metadata):
    """Use Bakta labels even for CDS absent from Panaroo; retain all memberships."""
    contigs = defaultdict(list)
    with gff.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "CDS":
                continue
            attrs = {key: unquote(value) for key, value in
                     (item.split("=", 1) for item in fields[8].split(";") if "=" in item)}
            feature = attrs.get("ID", "")
            if not feature:
                raise ValueError(f"CDS without ID in {gff}")
            clusters = membership.get(feature, set())
            if not clusters:
                clusters = set().union(*(membership.get(feature+suffix, set())
                              for suffix in ("_len", "_pseudo", "_len_pseudo")))
            for cluster in sorted(clusters) or [""]:
                meta = metadata.get(cluster, {})
                names = [x.strip() for x in re.split(r";|~~~", meta.get("label", ""))
                         if x.strip() and x.strip() != cluster and not re.fullmatch(r"group_\d+", x.strip(), re.I)]
                if attrs.get("gene"):
                    names.append(attrs["gene"])
                start, end = int(fields[3]), int(fields[4])
                if not 1 <= start <= end:
                    raise ValueError(f"Invalid CDS coordinates in {gff}: {feature}")
                contigs[fields[0]].append(dict(reference=sample, contig=fields[0], feature_id=feature,
                    gene_id=cluster or f"{sample}:{feature}", label="; ".join(sorted(set(names))) or "unannotated",
                    product=attrs.get("product", "") or meta.get("product", ""),
                    gene_start=start, gene_end=end, strand=fields[6]))
    geometry = {}
    for contig, rows in contigs.items():
        rows.sort(key=lambda r: r["gene_start"])
        geometry[contig] = (rows, [r["gene_start"] for r in rows],
                            np.maximum.accumulate([r["gene_end"] for r in rows]).tolist())
    return geometry


def build_matcher(sequences):
    import ahocorasick
    words = defaultdict(list)
    trans = str.maketrans("ACGT", "TGCA")
    for locus, sequence in sequences.items():
        reverse = sequence.translate(trans)[::-1]
        words[sequence].append((locus, "both" if sequence == reverse else "+"))
        if reverse != sequence:
            words[reverse].append((locus, "-"))
    automaton = ahocorasick.Automaton()
    for sequence, values in words.items():
        automaton.add_word(sequence, (len(sequence), tuple(values)))
    automaton.make_automaton()
    log(f"exact matcher: {len(sequences):,} unitigs; {len(automaton):,} strand-specific sequences")
    return automaton


def annotate_reference(sample, fna, gff, directory, matcher, membership, metadata, nearby):
    """Write one independently resumable assembly; no first-hit or repeat truncation."""
    from Bio import SeqIO
    geometry = gene_geometry(gff, sample, membership, metadata)
    genes = [row for rows, _, _ in geometry.values() for row in rows]
    tsv_gz(directory / "genes.tsv.gz", genes, GENE_COLUMNS)
    seen, count, matched = set(), 0, set()
    def rows():
        nonlocal count
        for record in SeqIO.parse(str(fna), "fasta"):
            if record.id in seen:
                raise ValueError(f"Duplicate contig in {fna}: {record.id}")
            seen.add(record.id)
            sequence = str(record.seq).upper()
            cds, starts, prefix_ends = geometry.get(record.id, ([], [], []))
            if any(row["gene_end"] > len(sequence) for row in cds):
                raise ValueError(f"GFF coordinates exceed contig length: {sample}/{record.id}")
            for endpoint, (length, queries) in matcher.iter(sequence):
                start, end = endpoint-length+2, endpoint+1  # 1-based inclusive
                lo, hi = bisect_left(prefix_ends, start-nearby), bisect_right(starts, end+nearby)
                candidates = [r for r in cds[lo:hi] if r["gene_end"] >= start-nearby]
                for locus, strand in queries:
                    matched.add(locus)
                    base = dict(locus=locus, reference=sample, contig=record.id,
                                hit_start=start, hit_end=end, hit_strand=strand)
                    if not candidates:
                        count += 1
                        yield dict(base, annotation_class="intergenic_outside_window", label="unannotated")
                    for gene in candidates:
                        if gene["gene_start"] <= end and gene["gene_end"] >= start:
                            category, distance = "coding", 0
                        elif gene["gene_start"] > end:
                            category = "upstream" if gene["strand"] == "+" else "downstream" if gene["strand"] == "-" else "unoriented_nearby"
                            distance = gene["gene_start"]-end
                        else:
                            category = "downstream" if gene["strand"] == "+" else "upstream" if gene["strand"] == "-" else "unoriented_nearby"
                            distance = start-gene["gene_end"]
                        count += 1
                        yield dict(base, **{k: v for k, v in gene.items() if k not in ("reference", "contig")},
                                   annotation_class=category, distance_bp=distance)
    tsv_gz(directory / "hits.tsv.gz", rows(), HIT_COLUMNS)
    if set(geometry)-seen:
        raise ValueError(f"GFF contigs missing from {fna}: {sorted(set(geometry)-seen)[:5]}")
    return dict(n_evidence_rows=count, n_matched_unitigs=len(matched), n_cds=len(genes))


def target_rules(args):
    source = args.targets or Path(__file__).with_name("data") / "amr_determinants.tsv"
    rules = {}
    for row in read_tsv(source, ["determinant", "aliases"]):
        if row.get("species") and row["species"].strip().casefold() != args.species.casefold():
            continue
        name = row["determinant"]
        if name in rules:
            # The bundled catalogue repeats determinants across antimicrobial rows.
            for key in ("product_require", "product_exclude"):
                if rules[name].get(key, "") != row.get(key, ""):
                    raise ValueError(f"Conflicting target rules for {name}; provide one consistent rule")
            row["aliases"] = ";".join(sorted(set(row["aliases"].split(";")) | set(rules[name]["aliases"].split(";"))))
            row["determinant_type"] = ";".join(sorted(set(filter(None, [row.get("determinant_type", ""), rules[name].get("determinant_type", "")]))))
        rules[name] = row
    # graR is an explicitly requested regulatory target, not a gene-presence resistance call.
    additions = (["graR"] if args.species.casefold() == "staphylococcus aureus" else []) + (args.target or [])
    for name in additions:
        rules.setdefault(name, dict(determinant=name, aliases=name, determinant_type="requested_gene_context"))
    if not args.targets and args.species.casefold() == "staphylococcus aureus":
        rules.setdefault("mecC", dict(determinant="mecC", aliases="mecC", determinant_type="requested_gene_context",
            product_require="PBP2|penicillin.binding|transpeptidase"))
        rules.setdefault("mprF", dict(determinant="mprF", aliases="mprF;fmtC", determinant_type="mutation_deferred"))
    if not rules:
        raise ValueError("No targets for this species; supply --targets or --target")
    return source, rules


def aggregate(args, sequences, directories, rules, locus_pattern, patterns):
    """Retain all gene contexts and rejected homonyms, not one preferred annotation."""
    index = defaultdict(set)
    for name, rule in rules.items():
        for token in label_tokens(rule["aliases"]):
            index[token].add(name)
    contexts, matched = defaultdict(set), set()
    genes_for_target = {name: set() for name in rules}
    target_records = {}
    for i, directory in enumerate(directories, 1):
        for gene in read_gz(directory / "genes.tsv.gz"):
            for name in set().union(*(index[token] for token in label_tokens(gene["label"]))):
                if candidate_status(gene, rules[name]) == "annotation_match":
                    genes_for_target[name].add((gene["reference"], gene["feature_id"]))
        for hit in read_gz(directory / "hits.tsv.gz"):
            locus = int(hit["locus"])
            matched.add(locus)
            if hit["gene_id"]:
                contexts[locus].add((hit["gene_id"], hit["label"], hit["annotation_class"]))
            names = set().union(*(index[token] for token in label_tokens(hit["label"])))
            for name in names:
                status = candidate_status(hit, rules[name])
                key = (name, locus, hit["gene_id"], hit["label"], hit["product"], hit["annotation_class"], status)
                if key not in target_records:
                    target_records[key] = dict(target=name, locus=locus, gene_id=hit["gene_id"], label=hit["label"],
                        product=hit["product"], annotation_class=hit["annotation_class"], match_status=status,
                        determinant_type=rules[name].get("determinant_type", ""), evidence_rows=0,
                        references=set(), distance_min_bp=int(hit["distance_bp"]))
                row = target_records[key]
                row["evidence_rows"] += 1
                row["references"].add(hit["reference"])
                row["distance_min_bp"] = min(row["distance_min_bp"], int(hit["distance_bp"]))
        log(f"collect annotations: {i}/{len(directories)} assemblies")
    def annotation_rows():
        for locus in sorted(sequences):
            evidence = contexts[locus]
            gene_ids = {g for g, _, _ in evidence}
            yield dict(locus=locus, sequence_length=len(sequences[locus]), pattern_id=locus_pattern.get(locus, ""),
                binary_status="available" if locus in locus_pattern else "binary_unavailable",
                annotation_status="multiple_gene_contexts" if len(gene_ids)>1 else "gene_context_found" if gene_ids
                    else "intergenic_outside_window" if locus in matched else "no_exact_match_in_supplied_references",
                labels="; ".join(sorted({label for _, label, _ in evidence})),
                gene_ids="; ".join(sorted(gene_ids)), annotation_classes="; ".join(sorted({c for _, _, c in evidence})))
    tsv_gz(args.out / "unitig_annotation.tsv.gz", annotation_rows(),
        ["locus", "sequence_length", "pattern_id", "binary_status", "annotation_status", "labels", "gene_ids", "annotation_classes"])
    rows = list(target_records.values())
    for row in rows:
        row["n_references"] = len(row.pop("references"))
        row["pattern_id"] = locus_pattern.get(row["locus"], "")
    # Include uncertain products for review, but never a known rejected homonym in D selection.
    selected = sorted({r["pattern_id"] for r in rows if r["pattern_id"] and r["match_status"] != "rejected_product"})
    write_tsv(args.out / "target_patterns.tsv", (
        dict(pattern_id=pid, **dict(zip(args.samples, np.unpackbits(
            np.frombuffer(patterns[pid]["bits"], dtype=np.uint8), bitorder="little")[:len(args.samples)].tolist())))
        for pid in selected), ["pattern_id"]+args.samples)
    return rows, genes_for_target, selected


def d_calculation(args, selected):
    script = Path(__file__).with_name("scripts") / "unitig_phylo_d.R"
    if args.stage == "annotate":
        return {}, dict(status="not_run_annotation_only")
    identity = stage_identity("unitig_D_1", [args.out/"analysis.treefile", args.out/"target_patterns.tsv", script],
        dict(permutations=args.permutations, seed=args.seed, zero_tip_policy=args.zero_tip_policy,
             tip_epsilon=args.tip_epsilon, rscript=args.rscript, versions=args.r_versions))
    # Output files are rewritten deterministically upstream. Hash their content, not mtime,
    # so restarting annotation/reporting does not discard completed D-pattern checkpoints.
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    stable = dict(tree=digest(args.out/"analysis.treefile"), patterns=digest(args.out/"target_patterns.tsv"),
                  script=digest(script), settings=identity["settings"])
    namespace = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:20]
    cache = args.out / "d_checkpoints" / namespace
    cache.mkdir(parents=True, exist_ok=True)
    log(f"D: {len(selected)} target patterns, {args.permutations} simulations/reference; checkpoint {namespace}")
    command = [args.rscript, "--vanilla", str(script), "--tree", str(args.out/"analysis.treefile"),
               "--patterns", str(args.out/"target_patterns.tsv"), "--out", str(cache),
               "--permutations", str(args.permutations), "--seed", str(args.seed),
               "--zero-tip-policy", args.zero_tip_policy, "--tip-epsilon", str(args.tip_epsilon)]
    if args.resume:
        command.append("--resume")
    subprocess.run(command, check=True)
    result = {r["pattern_id"]: r for r in read_tsv(cache/"D.tsv", ["pattern_id", "D", "D_status"])}
    if set(result) != set(selected):
        raise ValueError("D output pattern identities differ from requested patterns")
    errors = sum(r["D_status"] == "error" for r in result.values())
    if errors:
        log(f"D warning: {errors} patterns failed; inspect D_error; --resume retries these patterns")
    return result, dict(status="completed_with_pattern_errors" if errors else "completed", versions=args.r_versions, command=command,
                        checkpoint=str(cache), settings=stable)


def reports(args, records, genes_for_target, rules, patterns, d_results):
    d_keys = ["D", "D_status", "p_random_lower", "p_brownian_upper", "normalization_denominator", "D_error"]
    stat_keys = ["n_present", "n_absent", "maf", "entropy_bits", "passes_maf"]
    for row in records:
        pattern = patterns.get(row["pattern_id"], {})
        row.update({key: pattern.get(key, "") for key in stat_keys})
        d = d_results.get(row["pattern_id"], {})
        row.update({key: d.get(key, "") for key in d_keys})
        if not row["pattern_id"]:
            row["D_status"] = "binary_unavailable"
        elif not d:
            row["D_status"] = "rejected_product_not_selected" if row["match_status"] == "rejected_product" else "not_run"
        value = float(row["D"]) if row["D"] not in ("", "NA") else math.nan
        row["passes_D_nonnegative"] = int(value >= 0) if math.isfinite(value) else ""
        row["passes_both"] = int(bool(row["passes_maf"]) and value >= 0) if math.isfinite(value) else ""
    columns = ["target", "locus", "pattern_id", "gene_id", "label", "product", "determinant_type", "annotation_class",
               "distance_min_bp", "match_status", "evidence_rows", "n_references"]+stat_keys+d_keys+["passes_D_nonnegative", "passes_both"]
    write_tsv(args.out/"amr_unitig_audit.tsv", sorted(records, key=lambda r: (r["target"], r["locus"], r["gene_id"], r["annotation_class"])), columns)
    summaries = []
    for target in rules:
        accepted = [r for r in records if r["target"] == target and r["match_status"] == "annotation_match"]
        coding = [r for r in accepted if r["annotation_class"] == "coding"]
        values = [float(r["D"]) for r in coding if r["D_status"] == "ok"]
        summaries.append(dict(target=target, n_annotated_CDS=genes_for_target[target] if isinstance(genes_for_target[target], int) else len(genes_for_target[target]),
            n_coding_unitigs=len({r["locus"] for r in coding}), n_nearby_unitigs=len({r["locus"] for r in accepted if r["annotation_class"] != "coding"}),
            n_coding_patterns=len({r["pattern_id"] for r in coding if r["pattern_id"]}),
            n_coding_binary_unavailable=len({r["locus"] for r in coding if not r["pattern_id"]}),
            n_coding_patterns_pass_maf=len({r["pattern_id"] for r in coding if r["passes_maf"] == 1}),
            n_coding_patterns_D_ok=len({r["pattern_id"] for r in coding if r["D_status"] == "ok"}),
            n_coding_patterns_D_unavailable=len({r["pattern_id"] for r in coding if r["pattern_id"] and r["D_status"] != "ok"}),
            n_coding_patterns_D_negative=len({r["pattern_id"] for r in coding if r["passes_D_nonnegative"] == 0}),
            n_coding_patterns_D_nonnegative=len({r["pattern_id"] for r in coding if r["passes_D_nonnegative"] == 1}),
            n_coding_patterns_pass_both=len({r["pattern_id"] for r in coding if r["passes_both"] == 1}),
            n_product_unverified_unitigs=len({r["locus"] for r in records if r["target"] == target and r["match_status"] == "product_unverified"}),
            D_min=min(values) if values else "", D_max=max(values) if values else "",
            status="coding_unitigs_found" if coding else "nearby_only" if accepted else
                   "annotated_gene_without_matched_unitig" if genes_for_target[target] else "no_verified_annotation_match"))
    write_tsv(args.out/"amr_gene_summary.tsv", summaries, list(summaries[0]))
    write_tsv(args.out/"target_pattern_statistics.tsv", (
        {**{k: v for k, v in patterns[pid].items() if k != "bits"}, **d_results.get(pid, dict(D_status="not_run"))}
        for pid in sorted({r["pattern_id"] for r in records if r["pattern_id"]})),
        ["pattern_id"]+stat_keys+["n_unitigs"]+d_keys)


def probe_r(args):
    """Check only the runtime needed for D, before expensive upstream work."""
    probe = subprocess.run([args.rscript, "--vanilla", "-e",
        'stopifnot(requireNamespace("ape",quietly=TRUE),requireNamespace("caper",quietly=TRUE)); '
        'cat(R.version.string,as.character(packageVersion("ape")),as.character(packageVersion("caper")),sep="\\n")'],
        check=False, capture_output=True, text=True)
    if probe.returncode:
        raise ValueError(f"R dependency check failed: {probe.stderr.strip()}. Install ape/caper in this R environment, or use --stage annotate")
    args.r_versions = probe.stdout.strip()


def save_annotation_state(args, records, genes_for_target, rules, patterns, provenance, counts):
    """Freeze small prepared inputs independently of mutable final D reports."""
    wanted = {r["pattern_id"] for r in records if r["pattern_id"]}
    state = dict(schema=1, records=records,
        genes_for_target={k: v if isinstance(v, int) else len(v) for k, v in genes_for_target.items()},
        rules=rules, patterns={pid: {k: v for k, v in patterns[pid].items() if k != "bits"} for pid in sorted(wanted)},
        provenance=provenance, counts=counts)
    json_write(args.out/"annotation_state.json", state)
    outputs = [args.out/name for name in ("annotation_state.json", "analysis.treefile", "target_patterns.tsv")]
    save_cache(args.out/"annotation_ready.json", dict(stage="annotation_ready_1"), outputs)
    return state


def legacy_annotation_state(args):
    """Import old completed tables once, without reading assembly evidence or gzip.

    Older failed-D runs did not save source provenance. This explicit D-only
    restart adopts the existing prepared tree/patterns, not today's raw inputs.
    """
    required = ["amr_unitig_audit.tsv", "amr_gene_summary.tsv", "pattern_statistics.tsv",
                "analysis.treefile", "target_patterns.tsv"]
    for name in required:
        if not (args.out/name).is_file():
            raise ValueError(f"D-only restart missing {name}. Finish --stage annotate first")
    records = read_tsv(args.out/"amr_unitig_audit.tsv", ["target", "locus", "pattern_id", "match_status"])
    for row in records:
        for key in ("locus", "distance_min_bp", "evidence_rows", "n_references"):
            row[key] = int(row[key])
    summary = read_tsv(args.out/"amr_gene_summary.tsv", ["target", "n_annotated_CDS"])
    rules = {r["target"]: dict(determinant=r["target"]) for r in summary}
    if len(rules) != len(summary) or not {r["target"] for r in records}.issubset(rules):
        raise ValueError("Saved target summary and unitig report do not agree")
    wanted = {r["pattern_id"] for r in records if r["pattern_id"]}
    patterns, n_patterns, n_binary = {}, 0, 0
    with (args.out/"pattern_statistics.tsv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            n_patterns += 1
            n_binary += int(row["n_unitigs"])
            if row["pattern_id"] in wanted:
                if row["pattern_id"] in patterns:
                    raise ValueError("Duplicate pattern statistics")
                for key in ("n_present", "n_absent", "n_unitigs", "passes_maf"):
                    row[key] = int(row[key])
                for key in ("maf", "entropy_bits"):
                    row[key] = float(row[key])
                patterns[row["pattern_id"]] = row
    if set(patterns) != wanted:
        raise ValueError("Saved pattern statistics do not cover all reported unitigs")
    log("importing legacy prepared tables; original source provenance was not saved by that run")
    log("D-only mode uses the saved tree and patterns; raw input options are not reapplied")
    return save_annotation_state(args, records, {r["target"]: int(r["n_annotated_CDS"]) for r in summary},
        rules, patterns, dict(origin="legacy_prepared_tables", raw_source_identity_verified=False),
        dict(n_patterns=n_patterns, n_binary_loci=n_binary))


def restart_d(args):
    """Skip FASTA loading, reference annotation, collection and all gzip exports."""
    from Bio import Phylo
    ready = args.out/"annotation_ready.json"
    if ready.exists():
        outputs = [args.out/name for name in ("annotation_state.json", "analysis.treefile", "target_patterns.tsv")]
        if not cache_matches(ready, dict(stage="annotation_ready_1"), outputs):
            raise ValueError("Prepared annotation/tree/pattern checkpoint changed or is incomplete; restore its matching files or rerun annotation")
        state = json.loads((args.out/"annotation_state.json").read_text(encoding="utf-8"))
    else:
        state = legacy_annotation_state(args)
    if state.get("schema") != 1:
        raise ValueError("Unsupported prepared annotation schema")
    records, patterns = state["records"], state["patterns"]
    selected = []
    with (args.out/"target_patterns.tsv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = reader.fieldnames or []
        if not header or header[0] != "pattern_id" or len(set(header)) != len(header):
            raise ValueError("Invalid prepared pattern-matrix header")
        samples = header[1:]
        tips = [t.name for t in Phylo.read(args.out/"analysis.treefile", "newick").get_terminals()]
        if len(tips) != len(samples) or set(tips) != set(samples):
            raise ValueError("Saved pattern-matrix samples and tree tips differ")
        for row in reader:
            pid = row["pattern_id"]
            if pid not in patterns or any(row.get(s) not in ("0", "1") for s in samples):
                raise ValueError(f"Invalid prepared binary pattern {pid}")
            n_present = sum(row[s] == "1" for s in samples)
            if n_present != patterns[pid]["n_present"] or len(samples)-n_present != patterns[pid]["n_absent"]:
                raise ValueError(f"Prepared pattern counts disagree for {pid}")
            selected.append(pid)
    expected = {r["pattern_id"] for r in records if r["pattern_id"] and r["match_status"] != "rejected_product"}
    if len(set(selected)) != len(selected) or set(selected) != expected:
        raise ValueError("Saved AMR records and selected patterns do not agree")
    for row in patterns.values():
        n = row["n_present"]+row["n_absent"]
        row["passes_maf"] = int(min(row["n_present"], row["n_absent"]) > args.maf*n)
    log("D-only restart: skipping binary-matrix export, assembly scans, annotation collection and gzipping")
    d_results, d_info = d_calculation(args, selected)
    reports(args, records, state["genes_for_target"], state["rules"], patterns, d_results)
    previous = args.out/"run_manifest.json"
    payload = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
    payload.update(state["counts"])
    payload.update(status="complete", stage="d", n_samples=len(samples), n_target_patterns=len(selected),
        D=d_info, annotation_provenance=state["provenance"],
        restart="Prepared outputs reused; no reference annotation, annotation collection or gzip export")
    json_write(previous, payload)
    log(f"D-only complete: {args.out/'amr_gene_summary.tsv'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("unitigs", "fasta", "tree", "metadata", "panaroo", "bakta", "out"):
        parser.add_argument("--"+name, required=(name == "out"), type=Path)
    contract = parser.add_mutually_exclusive_group()
    contract.add_argument("--locus-map", type=Path, help="TSV with fasta_column,locus; both zero-based")
    contract.add_argument("--fasta-columns-are-locus-ids", action="store_true", help="Explicitly assert original column i is unitig ID i")
    parser.add_argument("--sample-map", type=Path)
    parser.add_argument("--species", default="Staphylococcus aureus")
    parser.add_argument("--exclude-sample", action="append", default=[])
    parser.add_argument("--tree-missing-samples", choices=["error", "drop"], default="error")
    parser.add_argument("--rooting", choices=["midpoint", "as-is"], default="midpoint")
    parser.add_argument("--targets", type=Path, help="Curated determinant,aliases TSV; optional species/product rules")
    parser.add_argument("--target", action="append", help="Additional exact gene label; repeatable")
    parser.add_argument("--nearby-bp", type=int, default=500)
    parser.add_argument("--maf", type=float, default=.05, help="Report strict MAF > threshold; do not discard")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--zero-tip-policy", choices=["error", "epsilon"], default="error")
    parser.add_argument("--tip-epsilon", type=float, default=1e-12)
    parser.add_argument("--stage", choices=["annotate", "all", "d"], default="all",
                        help="d reuses prepared outputs and skips annotation collection and compression")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-missing-references", action="store_true", help="Explicit incomplete annotation; missing assemblies are reported")
    parser.add_argument("--no-matrix-tsv", action="store_true", help="Skip large unique-pattern matrix text export")
    args = parser.parse_args(argv)
    if not 0 <= args.maf < .5 or args.nearby_bp < 0 or args.permutations < 2 or args.seed < 0 or not math.isfinite(args.tip_epsilon) or args.tip_epsilon <= 0:
        parser.error("Invalid MAF, nearby distance, permutations, seed or tip epsilon")
    args.out = args.out.resolve()
    if args.stage == "d":
        if not args.out.is_dir():
            parser.error("D-only restart requires an existing annotation output directory")
        probe_r(args)
        return restart_d(args)
    for name in ("unitigs", "fasta", "tree", "metadata", "panaroo", "bakta"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required for --stage {args.stage}")
    if not args.locus_map and not args.fasta_columns_are_locus_ids:
        parser.error("Supply --locus-map or --fasta-columns-are-locus-ids for annotation")
    sources = [args.unitigs, args.fasta, args.tree, args.metadata, args.panaroo]
    sources += [p for p in (args.locus_map, args.sample_map, args.targets) if p]
    for path in sources:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
        if path.resolve().is_relative_to(args.out):
            parser.error("Output must be separate from source inputs")
    if not args.bakta.is_dir() or args.out.is_relative_to(args.bakta.resolve()):
        parser.error("Bakta must be a directory, separate from output")
    import ahocorasick  # Early dependency check before large data loading.
    args.r_versions = "not_required_annotation_only"
    if args.stage == "all":
        probe_r(args)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out/"run_manifest.json"
    json_write(manifest, dict(status="running", interpretation="Annotation and distribution audit; no locus removal or KOVAR rerun"))
    samples, X, loci = prepare_cohort(args)
    args.samples = samples
    sequences = load_unitigs(args.unitigs)
    if set(loci)-set(sequences):
        raise ValueError("Binary locus IDs absent from DNA catalogue; verify the locus-column mapping")
    locus_pattern, patterns = pattern_tables(args, samples, X, loci)
    del X
    source_targets, rules = target_rules(args)
    metadata, members = panaroo_members(args.panaroo, samples)
    directories, references, missing = [], [], []
    for sample in samples:
        fna, gff = args.bakta/sample/f"{sample}.fna", args.bakta/sample/f"{sample}.gff3"
        if not fna.is_file() or not gff.is_file():
            missing.append(dict(sample=sample, fna=str(fna), gff=str(gff)))
        else:
            references.append((sample, fna, gff))
    write_tsv(args.out/"missing_references.tsv", missing, ["sample", "fna", "gff"])
    if missing and not args.allow_missing_references:
        raise ValueError(f"{len(missing)} retained isolates lack Bakta inputs; see missing_references.tsv. Use --allow-missing-references only deliberately")
    if not references:
        raise ValueError("No matching annotated assemblies")
    matcher = None
    for i, (sample, fna, gff) in enumerate(references, 1):
        directory = args.out/"reference_annotations"/hashlib.sha256(sample.encode()).hexdigest()[:20]
        directory.mkdir(parents=True, exist_ok=True)
        identity = stage_identity("all_unitig_exact_annotation_1", [args.unitigs, args.panaroo, fna, gff, Path(__file__)],
                                  dict(sample=sample, nearby_bp=args.nearby_bp,
                                       matcher_version=importlib.metadata.version("pyahocorasick")))
        outputs = [directory/"hits.tsv.gz", directory/"genes.tsv.gz"]
        checkpoint = directory/"complete.json"
        if args.resume and cache_matches(checkpoint, identity, outputs):
            log(f"annotation {i}/{len(references)}: reuse {sample}")
        else:
            checkpoint.unlink(missing_ok=True)
            if matcher is None:
                matcher = build_matcher(sequences)
            log(f"annotation {i}/{len(references)}: scanning {sample}")
            details = annotate_reference(sample, fna, gff, directory, matcher, members[sample], metadata, args.nearby_bp)
            save_cache(checkpoint, identity, outputs, **details)
            log(f"annotation {i}/{len(references)}: {details['n_matched_unitigs']:,} matching unitigs; {details['n_evidence_rows']:,} evidence rows")
        directories.append(directory)
    del matcher, members
    records, genes_for_target, selected = aggregate(args, sequences, directories, rules, locus_pattern, patterns)
    # Publish inspectable annotation reports even if later R calculations fail.
    reports(args, records, genes_for_target, rules, patterns, {})
    # Merge indexed per-assembly evidence without loading it into memory.
    tsv_gz(args.out/"unitig_gene_hits.tsv.gz", (r for d in directories for r in read_gz(d/"hits.tsv.gz")), HIT_COLUMNS)
    inventory = stage_identity("unitig_audit_1", sources+[source_targets, Path(__file__)],
        dict(species=args.species, nearby_bp=args.nearby_bp, maf=args.maf, rooting=args.rooting,
             samples=samples, targets=rules, stage=args.stage), directories=[args.bakta])
    save_annotation_state(args, records, genes_for_target, rules, patterns, inventory,
        dict(n_unitigs=len(sequences), n_binary_loci=len(loci), n_patterns=len(patterns),
             n_samples=len(samples), n_references=len(references), n_missing_references=len(missing)))
    d_results, d_info = d_calculation(args, selected)
    reports(args, records, genes_for_target, rules, patterns, d_results)
    json_write(manifest, dict(status="complete", provenance=inventory, n_unitigs=len(sequences), n_binary_loci=len(loci),
        n_patterns=len(patterns), n_target_patterns=len(selected), n_samples=len(samples),
        n_references=len(references), n_missing_references=len(missing), D=d_info,
        versions=dict(kovar=__version__, python=sys.version, numpy=np.__version__, biopython=importlib.metadata.version("biopython"),
                      pyahocorasick=importlib.metadata.version("pyahocorasick")),
        annotation="All full-length exact matches, both strands, all supplied cohort assemblies; all overlapping and bounded nearby CDS retained. No mismatch/partial mapping or resistance calling.",
        interpretation="MAF/D are report flags only. Unitig overlap does not establish a resistance mutation or phenotype effect. Original scan and network unchanged."))
    log(f"complete: {args.out/'amr_gene_summary.tsv'}; detailed unitigs: amr_unitig_audit.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

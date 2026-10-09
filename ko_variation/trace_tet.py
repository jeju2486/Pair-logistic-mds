"""Read-only tet/unitig tracing before network selection; no model fitting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote

import numpy as np
import pandas as pd

from .workflow_cache import cache_matches, save_cache, stage_identity


def announce(message):
    print(f"[tet trace] {message}", flush=True)


def fasta_records(path):
    name, chunks = None, []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks).upper()
                name, chunks = line[1:].split()[0], []
            else:
                if name is None:
                    raise ValueError(f"FASTA sequence without header: {path}")
                chunks.append(line)
        if name is not None:
            yield name, "".join(chunks).upper()


def tet_names(text):
    # Explicit determinant names; TetR-family regulators are not default targets.
    return {"tet" + suffix.upper() for suffix in re.findall(
        r"(?i)(?<![a-z0-9])tet\s*\(?\s*(38|[a-z])\s*\)?(?![a-z0-9])", text)}


def signature(paths, settings):
    # Include tracer source as well as scientific settings in stage reuse.
    return stage_identity("tet_trace", paths, dict(settings=settings,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def table(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare(args, out):
    """Extract all requested raw Bakta CDS regions, independently of Panaroo labels."""
    names, lengths = [], set()
    for name, sequence in fasta_records(args.fasta):
        names.append(name)
        lengths.add(len(sequence))
    if len(lengths) != 1 or not names:
        raise ValueError("Scanner FASTA must have equal-length records")
    n_loci = next(iter(lengths))
    if len(names) != len(set(names)):
        raise ValueError("Duplicate scanner FASTA sample IDs")
    n_input = len(names)
    if args.sample_inclusion:
        with args.sample_inclusion.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if any(not {"sample", "status"}.issubset(row) for row in rows):
            raise ValueError("Sample inclusion TSV requires sample/status")
        if len(rows) != len({x['sample'] for x in rows}) or {x['sample'] for x in rows} != set(names):
            raise ValueError("Sample inclusion table does not match original FASTA IDs")
        retained = {x['sample'] for x in rows if x['status'] == 'included'}
        names = [name for name in names if name in retained]
        if not names:
            raise ValueError("No included scanner samples")
    sources, missing = [], []
    for name in names:
        gff = args.bakta / name / f"{name}.gff3"
        assembly = args.bakta / name / f"{name}.fna"
        if gff.is_file() and assembly.is_file():
            sources.append((name, gff, assembly))
        else:
            missing.append(name)
    if not sources:
        raise ValueError("No matching Bakta GFF/assembly files for scanner sample IDs")
    if missing and not args.allow_missing_assemblies:
        raise ValueError(f"Missing Bakta inputs for {len(missing)} isolates (e.g. {missing[:3]}); use --allow-missing-assemblies only for an explicitly partial trace")
    paths = [args.fasta, args.unitigs] + [p for _, g, f in sources for p in (g, f)]
    if args.panaroo:
        paths.append(args.panaroo)
    if args.sample_inclusion:
        paths.append(args.sample_inclusion)
    identity = signature(paths, dict(genes=args.genes, nearby_bp=args.nearby_bp,
                                    missing_assemblies=missing, coordinate_system="one-based inclusive"))
    outputs = [out/"queries.fna", out/"regions.fna", out/"regions.json", out/"tet_targets.tsv"]
    manifest = out/"prepare.cache.json"
    if args.resume and cache_matches(manifest, identity, outputs):
        announce("reusing prepared unitig queries and tet reference regions")
        return json.loads(outputs[2].read_text())
    announce("reading all unitig sequences; retaining original zero-based locus IDs")
    maximum, minimum, unitig_count, seen = 0, math.inf, 0, set()
    with outputs[0].open("w") as queries, args.unitigs.open() as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 2:
                raise ValueError("Expected unitig file columns: zero-based locus ID, DNA sequence")
            locus, dna = int(fields[0]), fields[1].upper()
            if not 0 <= locus < n_loci or locus in seen or not dna or set(dna) - set("ACGT"):
                raise ValueError(f"Invalid/duplicate unitig ID or DNA: {locus}")
            seen.add(locus)
            maximum = max(maximum, len(dna))
            minimum = min(minimum, len(dna))
            unitig_count += 1
            queries.write(f">{locus}\n{dna}\n")
    if not maximum:
        raise ValueError("No unitigs supplied")
    wanted = set(args.genes)
    windows, targets = {}, []
    announce(f"finding raw Bakta tet CDS features across {len(sources)} assemblies")
    with outputs[1].open("w") as reference:
        for index, (sample, gff, assembly) in enumerate(sources, 1):
            cds = defaultdict(list)
            with gff.open() as handle:
                for line in handle:
                    if line.startswith("##FASTA"):
                        break
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 9 or fields[2] != "CDS":
                        continue
                    attrs = {k: unquote(v) for k, v in (x.split("=", 1) for x in fields[8].split(";") if "=" in x)}
                    feature = dict(sample=sample, contig=fields[0], start=int(fields[3]), end=int(fields[4]),
                                   strand=fields[6], feature_id=attrs.get("ID", ""),
                                   raw_gene=attrs.get("gene", ""), raw_name=attrs.get("Name", ""), product=attrs.get("product", ""))
                    if not 1 <= feature["start"] <= feature["end"]:
                        raise ValueError(f"Invalid GFF coordinates: {sample}/{feature['feature_id']}")
                    feature["targets"] = sorted(tet_names(" ".join([feature["raw_gene"], feature["product"], attrs.get("Name", "")])) & wanted)
                    cds[fields[0]].append(feature)
            target_contigs = {c for c, features in cds.items() if any(x["targets"] for x in features)}
            found_contigs = set()
            for contig, sequence in fasta_records(assembly):
                if contig not in target_contigs:
                    continue
                found_contigs.add(contig)
                features = cds[contig]
                focal = [x for x in features if x["targets"]]
                if any(x["end"] > len(sequence) for x in features):
                    raise ValueError(f"GFF extends beyond assembly: {sample}/{contig}")
                targets.extend(focal)
                # This margin guarantees a complete match for any supplied unitig
                # that overlaps or lies within nearby_bp of a target CDS.
                margin = args.nearby_bp + maximum - 1
                intervals = sorted((max(1, x["start"]-margin), min(len(sequence), x["end"]+margin)) for x in focal)
                merged = []
                for start, end in intervals:
                    if merged and start <= merged[-1][1] + 1:
                        merged[-1][1] = max(end, merged[-1][1])
                    else:
                        merged.append([start, end])
                for start, end in merged:
                    key = f"tetregion{len(windows)}"
                    windows[key] = dict(sample=sample, contig=contig, start=start, end=end,
                                        features=[x for x in features if x["start"] <= end and x["end"] >= start])
                    reference.write(f">{key}\n{sequence[start-1:end]}\n")
            if target_contigs - found_contigs:
                raise ValueError(f"Target GFF contigs missing from assembly: {sample}")
            if index == 1 or index % 50 == 0:
                announce(f"examined {index}/{len(sources)} assemblies; {len(targets)} tet CDS copies")
    membership = defaultdict(set)
    if args.panaroo:
        target_keys = {(x["sample"], x["feature_id"]) for x in targets}
        with args.panaroo.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "Gene" not in (reader.fieldnames or []):
                raise ValueError("Panaroo CSV requires the Gene column")
            samples = set(x["sample"] for x in targets) & set(reader.fieldnames)
            for row in reader:
                for sample in samples:
                    for feature in re.split(r"[;\t]", row.get(sample, "") or ""):
                        original = re.sub(r"(?:_len_pseudo|_pseudo|_len)$", "", feature)
                        if (sample, feature) in target_keys:
                            membership[(sample, feature)].add(row["Gene"])
                        elif (sample, original) in target_keys:
                            membership[(sample, original)].add(row["Gene"])
    for window in windows.values():
        for feature in window["features"]:
            feature["panaroo_clusters"] = sorted(membership[(feature["sample"], feature["feature_id"])])
    for target in targets:
        target["panaroo_clusters"] = sorted(membership[(target["sample"], target["feature_id"])])
    data = dict(windows=windows, unitig_count=unitig_count, maximum_unitig_length=maximum,
                minimum_unitig_length=minimum,
                n_input_samples=n_input, n_scanner_samples=len(names), sample_names=names, n_binary_loci=n_loci,
                n_assemblies=len(sources), missing_assemblies=missing,
                n_target_cds=len(targets), target_genes_found=sorted({g for x in targets for g in x["targets"]}))
    outputs[2].write_text(json.dumps(data))
    table(outputs[3], targets, ["sample", "contig", "start", "end", "strand", "feature_id", "raw_gene", "raw_name", "product", "targets", "panaroo_clusters"])
    save_cache(manifest, identity, outputs)
    return data


def map_unitigs(args, out, prepared):
    outputs = [out/"tet_unitig_hits.tsv", out/"mapping_summary.json", out/"repeat_limited_unitigs.tsv"]
    identity = signature([out/"queries.fna", out/"regions.fna", out/"regions.json"],
                         dict(max_hits=args.max_hits, bwa=args.bwa, nearby_bp=args.nearby_bp))
    manifest = out/"mapping.cache.json"
    if args.resume and cache_matches(manifest, identity, outputs):
        announce("reusing completed exact unitig mapping")
        return pd.read_csv(outputs[0], sep="\t"), json.loads(outputs[1].read_text())
    windows = prepared["windows"]
    commands = []
    if windows:
        bwa = shutil.which(args.bwa)
        if not bwa:
            raise ValueError(f"BWA not found: {args.bwa}")
        reference = str((out/"regions.fna").resolve())
        commands = [[bwa, "index", reference], [bwa, "fastmap", "-l", str(prepared["minimum_unitig_length"]),
                    "-w", str(args.max_hits), reference, str((out/"queries.fna").resolve())]]
        announce("indexing extracted tet-containing regions (source assemblies unchanged)")
        with (out/"bwa_index.log").open("w") as log:
            subprocess.run(commands[0], stdout=log, stderr=log, check=True)
        announce("mapping every unitig with full-length exact matches; this BWA stage is single-process")
        with (out/"fastmap.txt").open("w") as hits, (out/"bwa_fastmap.log").open("w") as log:
            subprocess.run(commands[1], stdout=hits, stderr=log, check=True)
    else:
        (out/"fastmap.txt").write_text("")
    columns = ["locus", "target_gene", "sample", "contig", "hit_start", "hit_end", "unitig_strand",
               "annotation_class", "distance_bp", "target_feature_id", "target_start", "target_end", "target_strand",
               "target_raw_gene", "target_product", "panaroo_clusters", "overlapping_cds"]
    repeat_loci, count, current, length, query_count = set(), 0, None, None, 0
    with (out/"fastmap.txt").open() as handle, outputs[0].open("w", newline="", encoding="utf-8") as result:
        writer = csv.DictWriter(result, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        seen = set()
        for line in handle:
            fields = line.rstrip().split("\t")
            if fields[0] == "SQ":
                current, length = int(fields[1]), int(fields[2])
                seen.clear()
                query_count += 1
                if query_count % 100000 == 0:
                    announce(f"parsed {query_count:,}/{prepared['unitig_count']:,} unitigs; {count:,} target-region hits")
            elif fields[0] == "EM" and fields[1:3] == ["0", str(length)]:
                for hit in fields[4:]:
                    if hit == "*":
                        repeat_loci.add(current)
                        continue
                    region, signed_position = hit.rsplit(":", 1)
                    window = windows[region]
                    local = int(signed_position[1:])
                    if not 1 <= local <= local + length - 1 <= window["end"]-window["start"]+1:
                        raise ValueError("BWA mapping outside extracted window")
                    start, end = window["start"] + local - 1, window["start"] + local + length - 2
                    overlapping = [f for f in window["features"] if f["start"] <= end and f["end"] >= start]
                    for target in window["features"]:
                        if not target["targets"]:
                            continue
                        distance = max(target["start"]-end, start-target["end"], 0)
                        if distance > args.nearby_bp:
                            continue
                        category = ("coding" if distance == 0 else
                                    "upstream" if (end < target["start"] and target["strand"] == "+") or
                                                  (start > target["end"] and target["strand"] == "-") else
                                    "downstream" if target["strand"] in {"+", "-"} else "unoriented_nearby")
                        for gene in target["targets"]:
                            key = (current, gene, window["sample"], window["contig"], start, end, target["feature_id"])
                            if key in seen:
                                continue
                            seen.add(key)
                            writer.writerow(dict(locus=current, target_gene=gene, sample=window["sample"], contig=window["contig"],
                                hit_start=start, hit_end=end, unitig_strand=signed_position[0], annotation_class=category,
                                distance_bp=distance, target_feature_id=target["feature_id"], target_start=target["start"], target_end=target["end"],
                                target_strand=target["strand"], target_raw_gene=target["raw_gene"], target_product=target["product"],
                                panaroo_clusters=json.dumps(target["panaroo_clusters"]),
                                overlapping_cds=json.dumps([{k:f[k] for k in ("feature_id", "raw_gene", "product", "targets")} for f in overlapping])))
                            count += 1
    repeat_path = out/"repeat_limited_unitigs.tsv"
    table(repeat_path, [dict(locus=x) for x in sorted(repeat_loci)], ["locus"])
    summary = dict(commands=commands, n_query_records_parsed=query_count, n_target_hits=count,
                   repeat_limited_unitigs=len(repeat_loci), repeat_limit=args.max_hits,
                   mapping="full-length exact, both strands, all supplied target-positive assemblies",
                   absence_caveat="repeat-limited queries, missing assemblies, absent/incorrect Bakta tet labels and linear contig boundaries can cause missed mappings")
    if windows and query_count != prepared["unitig_count"]:
        raise ValueError("BWA output lacks some unitig query records")
    outputs[1].write_text(json.dumps(summary, indent=2))
    save_cache(manifest, identity, outputs)
    return pd.read_csv(outputs[0], sep="\t"), summary


def pair_chunks(path, chunk_rows, headerless=False):
    if headerless:
        with Path(path).open() as handle:
            first = next((x for x in handle if x.strip() and not x.startswith("#")), "")
        fields = first.replace(",", " ").split()
        if not fields:
            raise ValueError(f"Empty pair file: {path}")
        try:
            int(fields[0]); int(fields[1]); no_header = True
        except ValueError:
            no_header = False
        defaults = ["u", "v", "distance", "ARACNE", "MI", "count", "M2", "min_distance", "max_distance"]
        names = defaults[:len(fields)] + [f"extra_{i}" for i in range(len(defaults), len(fields))]
        reader = pd.read_csv(path, sep=r"\s+|,", engine="python", comment="#", chunksize=chunk_rows,
                             header=None if no_header else 0, names=names if no_header else None)
    else:
        reader = pd.read_csv(path, sep="\t", chunksize=chunk_rows)
    for chunk in reader:
        if "u" not in chunk or "v" not in chunk:
            if not headerless:
                raise ValueError("KOVAR results require u/v columns")
            chunk = chunk.rename(columns={chunk.columns[0]: "u", chunk.columns[1]: "v"})
        for side in ("u", "v"):
            values = pd.to_numeric(chunk[side], errors="raise")
            if values.isna().any() or (values < 0).any() or (values != np.floor(values)).any():
                raise ValueError("Pairs must use nonnegative integer zero-based locus IDs")
            chunk[side] = values.astype(np.int64)
        yield chunk


def trace_pairs(args, out, hits, prepared, mapping):
    paths = [args.results, args.fasta, out/"tet_unitig_hits.tsv", out/"regions.json"]
    paths.extend(p for p in [args.pairs, args.annotation_status, args.sample_inclusion, args.run_summary] if p)
    identity = signature(paths, dict(alpha=args.alpha, distance_column=args.distance_column,
                                    ld_distance=args.ld_distance, cross_contig=args.cross_contig,
                                    numpy=np.__version__, pandas=pd.__version__))
    outputs = [out/"tet_locus_trace.tsv", out/"tet_pair_trace.tsv", out/"trace_summary.json"]
    manifest = out/"trace.cache.json"
    if args.resume and cache_matches(manifest, identity, outputs):
        announce("reusing completed pair tracing and summary reports")
        return json.loads(outputs[2].read_text())
    loci = set(hits.locus.astype(int))
    genes = hits.groupby("locus").target_gene.agg(lambda x: "; ".join(sorted(set(x)))).to_dict()
    sample_set = set(prepared["sample_names"])
    presence, inconsistent, sample_count = Counter(), Counter(), 0
    hit_samples = hits.groupby("locus")['sample'].agg(set).to_dict()
    ordered = np.array(sorted(loci), dtype=np.int64)
    if loci:
        announce("counting target-unitig presence in the scanner cohort")
        for name, sequence in fasta_records(args.fasta):
            if name not in sample_set:
                continue
            if set(sequence) - {'A', 'C'}:
                raise ValueError("Frequency tracing expects scanner encoding A=absence, C=presence")
            sample_count += 1
            present = np.frombuffer(sequence.encode("ascii"), dtype="S1")[ordered] == b'C'
            presence.update(int(x) for x in ordered[present])
            for locus, is_present in zip(ordered, present):
                if not is_present and name in hit_samples[locus]:
                    inconsistent[int(locus)] += 1
        if sample_count != len(sample_set):
            raise ValueError("Scanner sample IDs missing from original FASTA")
    counts, statuses = defaultdict(Counter), Counter()
    candidate_rows = 0
    if args.pairs and loci:
        announce("tracing tet-associated unitigs through the supplied PAN-GWES candidate file")
        for i, chunk in enumerate(pair_chunks(args.pairs, args.chunk_rows, True), 1):
            selected = chunk.loc[chunk.u.isin(loci) | chunk.v.isin(loci)]
            candidate_rows += len(selected)
            for side in ("u", "v"):
                for locus, value in selected.loc[selected[side].isin(loci), side].value_counts().items():
                    counts[int(locus)]["candidate_rows"] += int(value)
            if i % 100 == 0:
                announce(f"candidate chunks {i}; {candidate_rows:,} tet-associated rows")
    original_tests, declared, input_rows, target_rows = 0, None, 0, 0
    raw = out/"tet_pairs.raw.tsv"
    announce("streaming the full KOVAR table, including failed/ineligible pairs")
    with raw.open("w", newline="") as handle:
        first = True
        for i, chunk in enumerate(pair_chunks(args.results, args.chunk_rows), 1):
            required = {"status", "p_primary", args.distance_column}
            if required - set(chunk):
                raise ValueError(f"Missing KOVAR columns: {sorted(required-set(chunk))}")
            p = pd.to_numeric(chunk.p_primary, errors="raise")
            if ((p.dropna() < 0) | (p.dropna() > 1)).any():
                raise ValueError("Invalid primary P-values")
            original_tests += int(p.notna().sum())
            if "n_tests" in chunk and len(chunk):
                values = pd.to_numeric(chunk.n_tests, errors="raise")
                if values.isna().any() or values.nunique() != 1 or values.iloc[0] != int(values.iloc[0]):
                    raise ValueError("Invalid original n_tests")
                n = int(values.iloc[0])
                if declared is not None and n != declared:
                    raise ValueError("Inconsistent original n_tests")
                declared = n
            selected = chunk.loc[chunk.u.isin(loci) | chunk.v.isin(loci)]
            selected.to_csv(handle, sep="\t", index=False, header=first)
            first = False
            target_rows += len(selected)
            input_rows += len(chunk)
            if i == 1 or i % 100 == 0:
                announce(f"KOVAR rows {input_rows:,}; traced rows {target_rows:,}")
    n_tests = declared if declared is not None else original_tests
    if n_tests < original_tests:
        raise ValueError("n_tests is smaller than the full nonmissing P-value count")
    run_settings = {}
    if args.run_summary:
        with args.run_summary.open() as handle:
            run_settings = dict(line.rstrip("\n").split("\t", 1) for line in handle if "\t" in line)
        if "n_samples" in run_settings and int(run_settings["n_samples"]) != len(sample_set):
            raise ValueError("FASTA cohort differs from run_summary; supply the scanner sample_inclusion.tsv")
        if "n_pair_rows" in run_settings and int(run_settings["n_pair_rows"]) != input_rows:
            raise ValueError("Result row count differs from run_summary; supply the full original KOVAR table")
        if "n_tests" in run_settings and int(run_settings["n_tests"]) != n_tests:
            raise ValueError("Original test count differs from run_summary")
    cutoff = args.alpha/n_tests if n_tests else 0
    stages = Counter()
    final = out/"tet_pair_trace.tsv"
    with final.open("w", newline="") as handle:
        first = True
        for chunk in pd.read_csv(raw, sep="\t", chunksize=args.chunk_rows):
            p = pd.to_numeric(chunk.p_primary, errors="raise")
            distance = pd.to_numeric(chunk[args.distance_column], errors="raise")
            if ((distance < 0) & distance.ne(-1)).any():
                raise ValueError("Unexpected negative distance; only PAN-GWES -1 is supported")
            accepted = chunk.status.isin(["OK", "OK_SPA_FAILED"])
            significant = accepted & p.notna() & p.le(cutoff)
            distal = np.isfinite(distance) & distance.gt(args.ld_distance)
            if {"u_contig", "v_contig"}.issubset(chunk.columns):
                cross = chunk.u_contig.notna() & chunk.v_contig.notna() & chunk.u_contig.astype(str).ne(chunk.v_contig.astype(str))
                distal &= ~cross
                if args.cross_contig == "distal":
                    distal |= cross
            chunk["trace_stage"] = np.select([~accepted, accepted & p.isna(), accepted & ~significant,
                                               significant & ~distal],
                                             ["ineligible_or_failed", "missing_primary_p", "not_significant", "excluded_distance"],
                                             default="selected_distal")
            chunk["trace_bonferroni_cutoff"] = cutoff
            chunk["u_target_genes"] = chunk.u.map(genes).fillna("")
            chunk["v_target_genes"] = chunk.v.map(genes).fillna("")
            statuses.update(chunk.status)
            stages.update(chunk.trace_stage)
            for side in ("u", "v"):
                for locus, block in chunk.loc[chunk[side].isin(loci)].groupby(side):
                    counts[int(locus)].update({"kovar_rows": len(block), **block.trace_stage.value_counts().to_dict()})
            chunk.to_csv(handle, sep="\t", index=False, header=first)
            first = False
    # An empty original scan still needs an explicit trace-table header.
    if final.stat().st_size == 0:
        table(final, [], ["u", "v", "status", "p_primary", args.distance_column, "trace_stage"])
    audit = {}
    if args.annotation_status:
        with args.annotation_status.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"locus", "status"}.issubset(reader.fieldnames or []):
                raise ValueError("Annotation audit must contain locus/status")
            for row in reader:
                if int(row["locus"]) in loci:
                    audit[int(row["locus"])] = row
    locus_rows = []
    for locus, block in hits.groupby("locus", sort=True):
        record = dict(locus=int(locus), target_genes=genes[locus],
            annotation_classes="; ".join(sorted(set(block.annotation_class))),
            target_positive_assemblies=block['sample'].nunique(), n_exact_hits=len(block),
            minimum_gene_distance_bp=int(block.distance_bp.min()),
            n_present=presence[locus], n_scanner_samples=len(sample_set),
            prevalence=presence[locus]/len(sample_set),
            minor_state_frequency=min(presence[locus], len(sample_set)-presence[locus])/len(sample_set),
            n_exact_hit_samples_binary_absent=inconsistent[locus],
            n_hits_overlapping_multiple_cds=sum(len(json.loads(value)) > 1 for value in block.overlapping_cds),
            n_candidate_rows=counts[locus]["candidate_rows"] if args.pairs else "not_checked",
            n_kovar_rows=counts[locus]["kovar_rows"],
            original_annotation_status=audit.get(locus, {}).get("status", "NOT_IN_SUPPLIED_AUDIT"),
            original_annotation_gene=audit.get(locus, {}).get("gene", ""),
            original_candidate_evidence=audit.get(locus, {}).get("candidate_evidence", ""))
        for stage in ["ineligible_or_failed", "missing_primary_p", "not_significant", "excluded_distance", "selected_distal"]:
            record["n_"+stage] = counts[locus][stage]
        locus_rows.append(record)
    columns = ["locus", "target_genes", "annotation_classes", "target_positive_assemblies", "n_exact_hits", "minimum_gene_distance_bp",
               "n_present", "n_scanner_samples", "prevalence", "minor_state_frequency",
               "n_exact_hit_samples_binary_absent", "n_hits_overlapping_multiple_cds",
               "n_candidate_rows", "n_kovar_rows", "original_annotation_status", "original_annotation_gene", "original_candidate_evidence",
               "n_ineligible_or_failed", "n_missing_primary_p", "n_not_significant", "n_excluded_distance", "n_selected_distal"]
    table(out/"tet_locus_trace.tsv", locus_rows, columns)
    summary = dict(requested_genes=args.genes, preparation={k:v for k,v in prepared.items() if k not in {"windows", "sample_names"}}, mapping=mapping,
        scanner_run_settings=run_settings,
        target_unitigs=len(loci), supplied_candidate_file=str(args.pairs) if args.pairs else None,
        target_candidate_rows=candidate_rows if args.pairs else None, full_kovar_rows=input_rows,
        target_kovar_rows=target_rows, original_n_tests=n_tests, bonferroni_alpha=args.alpha,
        bonferroni_cutoff=cutoff, distance_column=args.distance_column, strict_distal_cutoff_bp=args.ld_distance,
        cross_contig_policy=args.cross_contig,
        unitigs_with_exact_hit_binary_mismatch=sum(x["n_exact_hit_samples_binary_absent"] > 0 for x in locus_rows),
        target_pair_status_counts=dict(statuses), target_pair_stage_counts=dict(stages),
        target_unitigs_without_kovar_rows=sum(x["n_kovar_rows"] == 0 for x in locus_rows),
        phenotype_interpretation="Genomic association only. A reported MIC modifier need not covary with tet in this cohort.",
        input_results=str(args.results.resolve()), diagnostic_only=True, refitted=False,
        caveat="Target-region annotation is reference-dependent; identical unitigs can map elsewhere. No genes or pairs are added to the scientific network. Distances use existing pair metadata; unknown cross-contig identity cannot be inferred from a distance-only table.")
    (out/"trace_summary.json").write_text(json.dumps(summary, indent=2))
    save_cache(manifest, identity, outputs)
    announce(f"{len(loci)} target-associated unitigs; pair stages: {dict(stages)}")
    announce(f"reports: {out/'trace_summary.json'} and {out/'tet_locus_trace.tsv'}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unitigs", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path, help="Original scanner binary FASTA (defines assembly sample IDs)")
    parser.add_argument("--bakta", required=True, type=Path, help="Directory containing SAMPLE/SAMPLE.gff3 and SAMPLE.fna")
    parser.add_argument("--results", required=True, type=Path, help="Full original ko_variation.tsv, not selected distal results")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--panaroo", type=Path, help="Optional gene_presence_absence CSV for raw CDS-to-cluster diagnostics")
    parser.add_argument("--pairs", type=Path, help="Optional exact zero-based PAN-GWES candidate file supplied to KOVAR")
    parser.add_argument("--annotation-status", type=Path)
    parser.add_argument("--sample-inclusion", type=Path, help="Scanner sample_inclusion.tsv; required if the tree excluded FASTA isolates")
    parser.add_argument("--run-summary", type=Path, help="Scanner run_summary.txt for filter provenance and full-result checks")
    parser.add_argument("--genes", nargs="+", default=["tetK", "tetM"])
    parser.add_argument("--nearby-bp", type=int, default=500)
    parser.add_argument("--ld-distance", type=float, default=10000)
    parser.add_argument("--distance-column", default="min_distance")
    parser.add_argument("--cross-contig", choices=["exclude", "distal"], default="exclude")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--chunk-rows", type=int, default=50000)
    parser.add_argument("--bwa", default="bwa")
    parser.add_argument("--max-hits", type=int, default=10000, help="BWA exact-hit limit; over-repeated queries are reported separately")
    parser.add_argument("--resume", action="store_true", help="Reuse matching completed preparation and mapping stages")
    parser.add_argument("--allow-missing-assemblies", action="store_true")
    args = parser.parse_args(argv)
    names = [tet_names(x) for x in args.genes]
    if any(len(x) != 1 for x in names):
        parser.error("--genes expects explicit tet names, e.g. tetK tetM tet38")
    args.genes = sorted({next(iter(x)) for x in names})
    if args.nearby_bp < 0 or args.chunk_rows < 1 or args.max_hits < 1 or not math.isfinite(args.ld_distance) or args.ld_distance < 0 or not 0 <= args.alpha <= 1:
        parser.error("Invalid distance, alpha, chunk-size or hit-limit setting")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    # Output directory is separate from scanner and annotation inputs.
    inputs = [args.unitigs, args.fasta, args.results, args.panaroo, args.pairs, args.annotation_status, args.sample_inclusion, args.run_summary]
    if any(p and p.resolve().is_relative_to(args.out) for p in inputs):
        parser.error("Place diagnostic output in a separate directory from input files")
    prepared = prepare(args, args.out)
    hits, mapping = map_unitigs(args, args.out, prepared)
    trace_pairs(args, args.out, hits, prepared, mapping)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

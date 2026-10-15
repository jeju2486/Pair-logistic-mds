"""Run official R fastbaps on the cleaned analysis cohort's nucleotide alignment.

This is a downstream grouping stage. It uses sample identity/species metadata,
never MIC values or target-unitig states, and does not change the KOVAR scanner.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import numpy as np

from .amr_report import read_tsv, write_tsv
from .phenotype import filter_cohort, log, sha256
from .workflow_cache import stage_identity, cache_matches, save_cache


def subset_alignment(source, samples, out):
    """Stream genuine aligned DNA; remove non-cohort sequences before SNP import."""
    from Bio import SeqIO
    wanted, found, seen, omitted = set(samples), set(), set(), []
    length = None
    target = out / "cohort_alignment.fasta"
    temporary = target.with_suffix(".partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in SeqIO.parse(str(source), "fasta"):
                if record.id in seen:
                    raise ValueError(f"Duplicate core-alignment ID: {record.id}")
                seen.add(record.id)
                if record.id not in wanted:
                    omitted.append(dict(sample=record.id, reason="outside_analysis_cohort"))
                    continue
                if length is None:
                    length = len(record.seq)
                if not length or len(record.seq) != length:
                    raise ValueError("Core alignment contains empty or unequal retained sequence lengths")
                if set(str(record.seq).upper()) - set("ACGTRYSWKMBDHVN-?"):
                    raise ValueError(f"Invalid nucleotide alignment sequence: {record.id}")
                record.description = ""
                SeqIO.write(record, handle, "fasta")
                found.add(record.id)
        if wanted - found:
            raise ValueError(f"Core alignment missing {len(wanted-found)} retained samples: {sorted(wanted-found)[:8]}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    write_tsv(out / "omitted_alignment_samples.tsv", omitted, ["sample", "reason"])
    return target, length


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("alignment", "fasta", "metadata", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--species", default="Staphylococcus aureus")
    parser.add_argument("--sample-map", type=Path)
    parser.add_argument("--exclude-sample", action="append", default=[])
    parser.add_argument("--prior", choices=["baps", "symmetric", "optimise.symmetric", "optimise.baps"], default="optimise.symmetric")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.seed < 0:
        parser.error("threads must be positive and seed nonnegative")
    if args.alignment.resolve() == args.fasta.resolve():
        parser.error("--alignment must be real aligned DNA, not the A/C unitig presence FASTA")
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = [args.alignment, args.fasta, args.metadata, Path(__file__),
              Path(__file__).with_name("phenotype.py"),
              Path(__file__).with_name("scripts") / "fastbaps_clusters.R"]
    if args.sample_map:
        inputs.append(args.sample_map)
    for path in inputs:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
        if path.resolve().is_relative_to(args.out.resolve()):
            parser.error("Output directory must be separate from source inputs")
    # Exact FASTA identities establish the analysis cohort without loading its matrix.
    with args.fasta.open(encoding="utf-8") as handle:
        samples = [line[1:].split()[0] for line in handle if line.startswith(">")]
    if not samples or len(set(samples)) != len(samples):
        raise ValueError("Unitig FASTA requires unique sample IDs")
    samples, _, excluded = filter_cohort(args, samples, np.empty((len(samples), 0)), args.out)
    if len(samples) < 3:
        raise ValueError("Fewer than three isolates remain for population clustering")
    probe = subprocess.run([args.rscript, "--vanilla", "-e",
        'stopifnot(requireNamespace("fastbaps", quietly=TRUE), requireNamespace("ape", quietly=TRUE)); '
        'cat(R.version.string, as.character(utils::packageVersion("fastbaps")), '
        'as.character(utils::packageVersion("ape")), '
        'as.character(utils::packageVersion("Matrix")), '
        'utils::packageDescription("fastbaps")$RemoteSha, sep="\\n")'],
        check=True, capture_output=True, text=True)
    settings = dict(prior=args.prior, seed=args.seed, threads=args.threads,
                    rscript=args.rscript, versions=probe.stdout.strip(), samples=samples)
    identity = stage_identity("fastbaps_cohort_1", inputs, settings)
    outputs = [args.out/name for name in ("clusters.tsv", "cluster_sizes.tsv", "hierarchy.rds",
                                         "sessionInfo.txt", "fastbaps_run.tsv", "omitted_alignment_samples.tsv")]
    manifest = args.out / "fastbaps.json"
    if args.resume and cache_matches(manifest, identity, outputs):
        log(f"reuse completed fastBAPS partition: {args.out / 'clusters.tsv'}")
        return 0
    manifest.unlink(missing_ok=True)
    alignment, length = subset_alignment(args.alignment, samples, args.out)
    script = Path(__file__).with_name("scripts") / "fastbaps_clusters.R"
    log(f"fastBAPS: {len(samples)} isolates, prior={args.prior}; cluster count inferred")
    command = [args.rscript, "--vanilla", str(script), "--alignment", str(alignment),
               "--out", str(args.out), "--prior", args.prior,
               "--threads", str(args.threads), "--seed", str(args.seed)]
    subprocess.run(command, check=True)
    rows = read_tsv(args.out / "clusters.tsv", ["sample", "cluster"])
    if len(rows) != len(samples) or {r["sample"] for r in rows} != set(samples):
        raise ValueError("fastBAPS output sample identities do not match the retained cohort")
    n_clusters = len({r["cluster"] for r in rows})
    save_cache(manifest, identity, outputs, n_samples=len(samples), n_clusters=n_clusters,
               excluded_samples=sorted(excluded), alignment_columns=length, command=command,
               cluster_sha256=sha256(args.out / "clusters.tsv"),
               interpretation="Alignment-defined population groups; no MIC or target-feature selection; no fixed final K.")
    log(f"fastBAPS complete: {n_clusters} clusters; {args.out / 'clusters.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

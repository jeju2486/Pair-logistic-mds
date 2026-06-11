from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import pandas as pd

from . import __version__
from .io_utils import read_fake_fasta, read_pairs, validate_pairs, read_labelled_distance_matrix
from .structure import binary_hamming_distance, classical_mds, tree_patristic_distance
from .scan import ScanConfig, scan_pairs


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "MDS-corrected bidirectional logistic tester for PAN-GWES/SpydrPick candidate pairs. "
            "No GRM/GLMM is used in this version."
        )
    )
    p.add_argument("--fasta", required=True, help="Fake FASTA matrix; A=0, C=1 by default.")
    p.add_argument("--pairs", required=True, help="PAN-GWES/SpydrPick candidate pair file.")
    p.add_argument("--out", required=True, help="Output directory.")

    p.add_argument("--tree", default=None, help="Optional Newick tree. If supplied, tree patristic MDS is used.")
    p.add_argument("--distance-matrix", default=None, help="Optional labelled sample distance matrix TSV/CSV.")
    p.add_argument("--n-mds", type=int, default=10, help="Number of MDS axes [10].")
    p.add_argument("--max-structure-loci", type=int, default=20000, help="Max loci used for FASTA-derived distance when no tree/matrix is supplied [20000].")
    p.add_argument("--distance-chunk-size", type=int, default=64, help="Sample chunk size for FASTA-distance calculation [64].")
    p.add_argument("--seed", type=int, default=1, help="Random seed for structure-locus subsampling [1].")

    p.add_argument("--presence-char", default="C", help="Presence character in fake FASTA [C].")
    p.add_argument("--absence-char", default="A", help="Absence character in fake FASTA [A].")
    p.add_argument("--drop-invalid-pairs", action="store_true", help="Drop invalid/self/out-of-range pairs instead of failing.")
    p.add_argument("--max-pairs", type=int, default=0, help="Optional debugging limit on input pairs [0=all].")

    p.add_argument("--min-count", type=int, default=3, help="Minimum count in all four 2x2 cells [3].")
    p.add_argument("--min-count-auto", action="store_true", help="Use max(--min-count, ceil(0.0025*N)).")

    p.add_argument("--engine", choices=["score", "exact"], default="score", help="Association engine: fast score test or exact logistic LRT [score].")
    p.add_argument("--spa", action="store_true", help="Use SPA p-value for the score test. Falls back to chi-square if SPA fails.")
    p.add_argument("--no-firth", action="store_true", help="Disable Firth fallback for --engine exact.")
    p.add_argument("--high-se-threshold", type=float, default=3.0, help="Exact-mode SE threshold that triggers Firth fallback [3.0].")
    p.add_argument("--max-iter", type=int, default=100, help="Maximum iterations for null/exact logistic fits [100].")

    p.add_argument("--no-progress", action="store_true", help="Suppress internal progress messages.")
    p.add_argument("--progress-every-rows", type=int, default=10000, help="Print scan progress at least every N rows [10000].")
    p.add_argument("--progress-every-pct", type=float, default=5.0, help="Print scan progress at least every X percent [5].")

    p.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory.")
    p.add_argument("--version", action="version", version=f"pair-mds-gwes {__version__}")
    return p.parse_args(argv)


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _step(t0: float, label: str, message: str = "") -> None:
    elapsed = time.time() - t0
    msg = f"[{_ts()}] [pair-mds-gwes] step={label} elapsed={elapsed:.1f}s"
    if message:
        msg += f" {message}"
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def main(argv=None):
    t0 = time.time()
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory exists and is not empty: {out}. Use --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    progress = not args.no_progress

    sys.stderr.write("[pair-mds-gwes]\n")
    sys.stderr.write(f"  version={__version__}\n")
    sys.stderr.write(f"  fasta={args.fasta}\n")
    sys.stderr.write(f"  pairs={args.pairs}\n")
    sys.stderr.write(f"  out={args.out}\n")
    sys.stderr.write(f"  engine={args.engine}\n")
    sys.stderr.write(f"  spa={bool(args.spa)}\n")
    sys.stderr.flush()

    _step(t0, "1/7_read_fasta", "start")
    fm = read_fake_fasta(args.fasta, presence_char=args.presence_char, absence_char=args.absence_char)
    sample_names = fm.sample_names
    X = fm.X
    n_samples, n_loci = X.shape
    _step(t0, "1/7_read_fasta", f"done matrix={n_samples}x{n_loci}")

    _step(t0, "2/7_read_pairs", "start")
    pairs = read_pairs(args.pairs)
    pairs = validate_pairs(pairs, n_loci=n_loci, drop_invalid=args.drop_invalid_pairs)
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs.iloc[:args.max_pairs].copy()
    _step(t0, "2/7_read_pairs", f"done pairs_tested={len(pairs)}")

    _step(t0, "3/7_structure_distance", "start")
    if args.tree:
        D = tree_patristic_distance(args.tree, sample_names)
        structure_source = "tree_mds"
    elif args.distance_matrix:
        D = read_labelled_distance_matrix(args.distance_matrix, sample_names)
        structure_source = "distance_matrix_mds"
    else:
        D = binary_hamming_distance(
            X,
            max_loci=args.max_structure_loci,
            seed=args.seed,
            chunk_size=args.distance_chunk_size,
        )
        structure_source = "fasta_hamming_mds"
    _step(t0, "3/7_structure_distance", f"done source={structure_source} n={D.shape[0]}")

    _step(t0, "4/7_mds", "start")
    cov, mds_summary = classical_mds(D, k=args.n_mds)
    _step(t0, "4/7_mds", f"done n_mds={cov.shape[1]}")

    _step(t0, "5/7_write_mds", "start")
    mds_df = pd.DataFrame({"sample": sample_names})
    for i in range(cov.shape[1]):
        mds_df[f"MDS{i+1}"] = cov[:, i]
    mds_df.to_csv(out / "mds_covariates.tsv", sep="\t", index=False)
    mds_summary.to_csv(out / "mds_summary.tsv", sep="\t", index=False)
    _step(t0, "5/7_write_mds", "done")

    min_count = args.min_count
    if args.min_count_auto:
        min_count = max(min_count, int(math.ceil(0.0025 * n_samples)))

    config = ScanConfig(
        min_count=min_count,
        engine=args.engine,
        spa=bool(args.spa),
        firth_fallback=not args.no_firth,
        max_iter=args.max_iter,
        high_se_threshold=args.high_se_threshold,
        progress=progress,
        progress_every_rows=args.progress_every_rows,
        progress_every_pct=args.progress_every_pct,
    )

    _step(t0, "6/7_scan_pairs", f"start pairs={len(pairs)} min_count={min_count}")
    results, null_summary = scan_pairs(pairs, X, cov, config)
    _step(t0, "6/7_scan_pairs", f"done result_rows={len(results)} null_rows={len(null_summary)}")

    _step(t0, "7/7_write_results", "start")
    results.to_csv(out / "pair_mds_logistic.tsv", sep="\t", index=False, na_rep="NA")
    null_summary.to_csv(out / "null_logistic_summary.tsv", sep="\t", index=False, na_rep="NA")

    with (out / "run_summary.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"version\t{__version__}\n")
        fh.write(f"n_samples\t{n_samples}\n")
        fh.write(f"n_loci\t{n_loci}\n")
        fh.write(f"n_pairs\t{len(pairs)}\n")
        fh.write(f"structure_source\t{structure_source}\n")
        fh.write(f"n_mds\t{cov.shape[1]}\n")
        fh.write(f"engine\t{args.engine}\n")
        fh.write(f"spa\t{bool(args.spa)}\n")
        fh.write(f"min_count\t{min_count}\n")

    if "status" in results.columns:
        sys.stderr.write("[pair-mds-gwes] status_counts\n")
        sys.stderr.write(results["status"].fillna("NA").value_counts().to_string() + "\n")
    if "bidirectional_score" in results.columns:
        s = pd.to_numeric(results["bidirectional_score"], errors="coerce")
        sys.stderr.write("[pair-mds-gwes] bidirectional_score_summary\n")
        sys.stderr.write(s.describe().to_string() + "\n")

    _step(t0, "7/7_write_results", f"done wrote={out / 'pair_mds_logistic.tsv'}")
    _step(t0, "complete", f"elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

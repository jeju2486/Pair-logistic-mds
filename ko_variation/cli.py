from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# Set these before importing NumPy/SciPy so worker processes do not multiply a
# dense response-level process pool by a second BLAS thread pool.
for _blas_variable in (
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_variable, "1")

import numpy as np

from . import __version__
from .glmm import prepare_kinship
from .io_utils import read_fake_fasta, read_pairs, validate_pairs
from .kinship import build_background_grm, build_tree_covariance
from .scan import ScanConfig, scan_pairs_glmm


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "KOVAR: experimental directional covariation scanning with a "
            "phylogeny-adjusted logistic mixed model. Each candidate pair is "
            "tested as predictor->response; this direction is predictive, not causal."
        )
    )
    parser.add_argument("--fasta", required=True, help="Binary fake FASTA; A=0 and C=1 by default.")
    parser.add_argument("--pairs", required=True, help="Zero-based candidate-locus pairs (u and v columns).")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--tree", help="Preferred: rooted Newick tree with branch lengths.")
    parser.add_argument(
        "--tree-missing-length",
        choices=["error", "one", "zero"],
        default="error",
        help="Handling for missing tree branch lengths [error].",
    )

    input_group = parser.add_argument_group("input and filtering")
    input_group.add_argument("--presence-char", default="C", help="Presence state in fake FASTA [C].")
    input_group.add_argument("--absence-char", default="A", help="Absence state in fake FASTA [A].")
    input_group.add_argument(
        "--drop-invalid-pairs",
        action="store_true",
        help="Drop self/out-of-range pairs instead of stopping.",
    )
    input_group.add_argument("--max-pairs", type=int, default=0, help="Debugging limit; 0 scans all pairs [0].")
    input_group.add_argument(
        "--min-maf",
        type=float,
        default=0.05,
        help="Minimum minor-state frequency for predictor and response [0.05].",
    )
    input_group.add_argument(
        "--min-cell-count",
        type=int,
        default=5,
        help="Minimum count in each oriented 2x2 table cell [5].",
    )
    input_group.add_argument(
        "--near-redundant-mismatch",
        type=float,
        default=0.02,
        help="Same/complement mismatch rate used to annotate near-redundant pairs [0.02].",
    )
    input_group.add_argument(
        "--exclude-near-redundant",
        action="store_true",
        help="Filter annotated near-copy or near-complement pairs.",
    )

    model_group = parser.add_argument_group("directional logistic mixed model")
    model_group.add_argument(
        "--direction-mode",
        choices=["both", "input"],
        default="both",
        help="Test both directions or only u_predicts_v [both].",
    )
    model_group.add_argument(
        "--spa-mode",
        choices=["off", "auto", "always"],
        default="off",
        help="Experimental saddlepoint calibration [off]. Not a SAIGE reproduction.",
    )
    model_group.add_argument(
        "--full-refit-p",
        type=float,
        default=0.05,
        help="Refit a full mixed model for effects when primary p <= threshold; 0 disables [0.05].",
    )
    model_group.add_argument("--null-max-iter", type=int, default=100, help="Maximum PQL iterations [100].")
    model_group.add_argument("--null-tolerance", type=float, default=1e-7, help="PQL convergence tolerance [1e-7].")
    model_group.add_argument("--threads", type=int, default=1, help="Response-wise worker processes [1].")
    model_group.add_argument("--worker-chunk-size", type=int, default=1, help="Worker scheduling chunk size [1].")
    model_group.add_argument(
        "--predictor-batch-size",
        type=int,
        default=256,
        help="Predictors sharing a response scored per matrix block [256].",
    )

    grm_group = parser.add_argument_group("background GRM fallback (used only without --tree)")
    grm_group.add_argument("--kinship-min-mac", type=int, default=2, help="Background-locus minimum minor count [2].")
    grm_group.add_argument("--kinship-chunk-size", type=int, default=4096, help="GRM construction locus chunk [4096].")
    grm_group.add_argument("--kinship-dtype", choices=["float64", "float32"], default="float64", help="Stored covariance dtype [float64].")
    grm_group.add_argument(
        "--grm-proxy-r2",
        type=float,
        default=0.8,
        help="Exclude background loci with raw r2 to any tested locus >= threshold [0.8].",
    )
    grm_group.add_argument(
        "--grm-proxy-mismatch",
        type=float,
        default=0.02,
        help="Exclude same/complement tested-locus proxies at this mismatch rate [0.02].",
    )
    grm_group.add_argument("--grm-target-chunk-size", type=int, default=256, help="Proxy-mask target chunk [256].")
    grm_group.add_argument("--grm-locus-chunk-size", type=int, default=2048, help="Proxy-mask background chunk [2048].")

    output_group = parser.add_argument_group("execution and output")
    output_group.add_argument("--no-progress", action="store_true", help="Suppress progress messages.")
    output_group.add_argument("--overwrite", action="store_true", help="Write into a non-empty output directory.")
    output_group.add_argument("--version", action="version", version=f"KO-Variation {__version__}")
    args = parser.parse_args(argv)

    if not 0.0 <= args.min_maf <= 0.5:
        parser.error("--min-maf must be between 0 and 0.5")
    if args.min_cell_count < 0:
        parser.error("--min-cell-count must be non-negative")
    if args.threads < 1 or args.worker_chunk_size < 1 or args.predictor_batch_size < 1:
        parser.error("--threads, --worker-chunk-size and --predictor-batch-size must be positive")
    if args.null_max_iter < 2 or args.null_tolerance <= 0:
        parser.error("--null-max-iter must be at least 2 and --null-tolerance must be positive")
    if not 0.0 <= args.full_refit_p <= 1.0:
        parser.error("--full-refit-p must be between 0 and 1")
    if not 0.0 <= args.grm_proxy_r2 <= 1.0:
        parser.error("--grm-proxy-r2 must be between 0 and 1")
    if not 0.0 <= args.grm_proxy_mismatch <= 1.0:
        parser.error("--grm-proxy-mismatch must be between 0 and 1")
    return args


def _step(started: float, label: str, message: str = "") -> None:
    suffix = f" {message}" if message else ""
    sys.stderr.write(f"[KOVAR] step={label} elapsed={time.time() - started:.1f}s{suffix}\n")
    sys.stderr.flush()


def main(argv=None):
    started = time.time()
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {out}. Use --overwrite to continue.")
    out.mkdir(parents=True, exist_ok=True)
    progress = not args.no_progress

    _step(started, "read_fasta")
    fasta = read_fake_fasta(
        args.fasta,
        presence_char=args.presence_char,
        absence_char=args.absence_char,
    )
    X = fasta.X
    n_samples, n_loci = X.shape
    _step(started, "read_fasta", f"samples={n_samples} loci={n_loci}")

    _step(started, "read_pairs")
    pairs = validate_pairs(
        read_pairs(args.pairs),
        n_loci=n_loci,
        drop_invalid=args.drop_invalid_pairs,
    )
    if args.max_pairs > 0:
        pairs = pairs.iloc[:args.max_pairs].copy().reset_index(drop=True)
    if pairs.empty:
        raise SystemExit("No candidate pairs remain after input validation")
    _step(started, "read_pairs", f"pairs={len(pairs)}")

    targets = np.unique(pairs[["u", "v"]].to_numpy(dtype=np.int64).reshape(-1))
    if args.tree:
        _step(started, "build_covariance", "source=tree")
        kinship = build_tree_covariance(
            args.tree,
            fasta.sample_names,
            dtype=args.kinship_dtype,
            missing_length=args.tree_missing_length,
        )
    else:
        _step(started, "build_covariance", "source=background_grm")
        kinship = build_background_grm(
            X,
            targets,
            min_mac=args.kinship_min_mac,
            chunk_size=args.kinship_chunk_size,
            dtype=args.kinship_dtype,
            progress=progress,
            mask_r2=args.grm_proxy_r2,
            mask_mismatch=args.grm_proxy_mismatch,
            mask_target_chunk_size=args.grm_target_chunk_size,
            mask_grm_chunk_size=args.grm_locus_chunk_size,
        )
    K, kdiag = prepare_kinship(kinship.K)
    del kinship.K
    _step(
        started,
        "build_covariance",
        f"source={kinship.source} rank={kdiag.rank}/{n_samples} "
        f"eig_min={kdiag.eigen_min:.3g} eig_max={kdiag.eigen_max:.3g}",
    )

    config = ScanConfig(
        min_maf=args.min_maf,
        min_cell_count=args.min_cell_count,
        direction_mode=args.direction_mode,
        spa_mode=args.spa_mode,
        full_refit_p=args.full_refit_p,
        threads=args.threads,
        worker_chunk_size=args.worker_chunk_size,
        predictor_batch_size=args.predictor_batch_size,
        null_max_iter=args.null_max_iter,
        null_tolerance=args.null_tolerance,
        progress=progress,
        near_redundant_mismatch=args.near_redundant_mismatch,
        exclude_near_redundant=args.exclude_near_redundant,
    )
    _step(
        started,
        "directional_scan",
        f"direction_mode={args.direction_mode} min_maf={args.min_maf:g} "
        f"min_cell_count={args.min_cell_count} spa={args.spa_mode}",
    )
    results, response_models = scan_pairs_glmm(pairs, X, K, config)

    shared_diagnostics = {
        "kinship_source": kinship.source,
        "kinship_rank": kdiag.rank,
        "kinship_eigen_min": kdiag.eigen_min,
        "kinship_eigen_max": kdiag.eigen_max,
        "kinship_roundoff_correction": kdiag.roundoff_correction,
    }
    for column, value in shared_diagnostics.items():
        results[column] = value
        response_models[column] = value

    _step(started, "write_results")
    result_path = out / "ko_variation.tsv"
    model_path = out / "response_models.tsv"
    results.to_csv(result_path, sep="\t", index=False, na_rep="NA")
    response_models.to_csv(model_path, sep="\t", index=False, na_rep="NA")
    tested = int(np.isfinite(results["p_primary"].to_numpy(dtype=np.float64)).sum())
    with (out / "run_summary.txt").open("w", encoding="utf-8") as summary:
        summary.write(f"version\t{__version__}\n")
        summary.write("tool\tKO-Variation\n")
        summary.write("acronym\tKOVAR\n")
        summary.write("release_status\texperimental\n")
        summary.write("model\tdirectional_logistic_mixed_model_pql_score\n")
        summary.write("interpretation\tdirectional_covariation_not_causal_direction\n")
        summary.write(f"n_samples\t{n_samples}\n")
        summary.write(f"n_loci\t{n_loci}\n")
        summary.write(f"n_input_pairs\t{len(pairs)}\n")
        summary.write(f"n_directional_rows\t{len(results)}\n")
        summary.write(f"n_directional_tests\t{tested}\n")
        summary.write(f"direction_mode\t{args.direction_mode}\n")
        summary.write(f"min_maf\t{args.min_maf}\n")
        summary.write(f"min_cell_count\t{args.min_cell_count}\n")
        summary.write(f"spa_mode\t{args.spa_mode}\n")
        summary.write(f"full_refit_p\t{args.full_refit_p}\n")
        summary.write(f"kinship_source\t{kinship.source}\n")
        summary.write(f"tree\t{args.tree or 'NA'}\n")
        summary.write(f"kinship_rank\t{kdiag.rank}\n")
        summary.write(f"kinship_eigen_min\t{kdiag.eigen_min}\n")
        summary.write(f"kinship_eigen_max\t{kdiag.eigen_max}\n")
        summary.write(f"kinship_roundoff_correction\t{kdiag.roundoff_correction}\n")
        summary.write(f"kinship_mean_diag_before_norm\t{kinship.mean_diag_before_norm}\n")
        summary.write(f"kinship_loci_used\t{kinship.n_loci_used}\n")
        summary.write(f"null_max_iter\t{args.null_max_iter}\n")
        summary.write(f"null_tolerance\t{args.null_tolerance}\n")
        summary.write(f"threads\t{args.threads}\n")
        summary.write(f"worker_chunk_size\t{args.worker_chunk_size}\n")
        summary.write(f"predictor_batch_size\t{args.predictor_batch_size}\n")
        summary.write(f"near_redundant_mismatch\t{args.near_redundant_mismatch}\n")
        summary.write(f"exclude_near_redundant\t{int(args.exclude_near_redundant)}\n")
        for key in sorted(kinship.details):
            summary.write(f"kinship_detail_{key}\t{kinship.details[key]}\n")

    sys.stderr.write("[KOVAR] result_status_counts\n")
    sys.stderr.write(results["status"].fillna("NA").value_counts().to_string() + "\n")
    _step(started, "complete", f"results={result_path} response_models={model_path}")


if __name__ == "__main__":
    main()

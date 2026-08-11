from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# Set conservative defaults before importing NumPy/SciPy so response-level
# worker processes do not silently multiply a second native thread pool. User
# settings remain authoritative and are inspected and reported at runtime.
from .diagnostics import (
    collect_parallel_runtime,
    estimate_scan_memory,
    guard_blas_environment,
)

guard_blas_environment(default_threads=1)

import numpy as np
import pandas as pd
import scipy

from . import __version__
from .checkpoint import CheckpointError, CheckpointStore, fingerprint_inputs
from .glmm import prepare_kinship
from .io_utils import read_fake_fasta, read_pairs, validate_pairs
from .kinship import build_background_grm, build_tree_covariance
from .scan import ScanConfig, ScanMetrics, scan_pairs_glmm


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
    output_group.add_argument(
        "--checkpoint-dir",
        help="Checkpoint directory [OUT/.kovar_checkpoint].",
    )
    output_group.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Atomically checkpoint after this many completed tasks [100].",
    )
    output_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume an exactly matching checkpoint.",
    )
    output_group.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint writing for this run.",
    )
    output_group.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep checkpoint shards after final outputs are written.",
    )
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
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.resume and args.no_checkpoint:
        parser.error("--resume cannot be combined with --no-checkpoint")
    if args.keep_checkpoints and args.no_checkpoint:
        parser.error("--keep-checkpoints cannot be combined with --no-checkpoint")
    return args


def _step(started: float, label: str, message: str = "") -> None:
    suffix = f" {message}" if message else ""
    sys.stderr.write(f"[KOVAR] step={label} elapsed={time.time() - started:.1f}s{suffix}\n")
    sys.stderr.flush()


def _atomic_dataframe_write(frame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, sep="\t", index=False, na_rep="NA")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_identity(
    args,
    *,
    input_fingerprints: dict[str, object],
    n_samples: int,
    n_loci: int,
    n_pairs: int,
    blas_libraries: str,
) -> dict[str, object]:
    """Scientific inputs/settings that must match before work can be reused."""

    return {
        "tool": "KO-Variation",
        "version": __version__,
        "inputs": input_fingerprints,
        "dimensions": {
            "n_samples": n_samples,
            "n_loci": n_loci,
            "n_pairs": n_pairs,
        },
        "numerical_environment": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "blas_libraries": blas_libraries,
        },
        "settings": {
            "presence_char": args.presence_char,
            "absence_char": args.absence_char,
            "drop_invalid_pairs": bool(args.drop_invalid_pairs),
            "max_pairs": args.max_pairs,
            "tree_missing_length": args.tree_missing_length,
            "min_maf": args.min_maf,
            "min_cell_count": args.min_cell_count,
            "near_redundant_mismatch": args.near_redundant_mismatch,
            "exclude_near_redundant": bool(args.exclude_near_redundant),
            "direction_mode": args.direction_mode,
            "spa_mode": args.spa_mode,
            "full_refit_p": args.full_refit_p,
            "null_max_iter": args.null_max_iter,
            "null_tolerance": args.null_tolerance,
            "predictor_batch_size": args.predictor_batch_size,
            "kinship_min_mac": args.kinship_min_mac,
            "kinship_chunk_size": args.kinship_chunk_size,
            "kinship_dtype": args.kinship_dtype,
            "grm_proxy_r2": args.grm_proxy_r2,
            "grm_proxy_mismatch": args.grm_proxy_mismatch,
            "grm_target_chunk_size": args.grm_target_chunk_size,
            "grm_locus_chunk_size": args.grm_locus_chunk_size,
        },
    }


def _metadata_rows(
    category: str,
    fields: dict[str, object],
    *,
    unit: str = "",
) -> list[dict[str, object]]:
    return [
        {"category": category, "metric": key, "value": value, "unit": unit}
        for key, value in sorted(fields.items())
    ]


def main(argv=None):
    started = time.time()
    args = parse_args(argv)
    runtime = collect_parallel_runtime(args.threads)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite and not args.resume:
        raise SystemExit(
            f"Output directory is not empty: {out}. Use --overwrite for a new run "
            "or --resume for an exactly matching checkpoint."
        )
    out.mkdir(parents=True, exist_ok=True)
    progress = not args.no_progress

    _step(
        started,
        "runtime",
        f"workers={runtime.worker_processes} blas_threads={runtime.effective_blas_threads} "
        f"estimated_native_threads={runtime.estimated_native_threads} "
        f"logical_cpus={runtime.logical_cpus}",
    )
    for warning in runtime.warnings:
        sys.stderr.write(
            f"[KOVAR] warning={warning} workers={runtime.worker_processes} "
            f"blas_threads={runtime.effective_blas_threads}; "
            "prefer one BLAS thread per response worker unless benchmarked otherwise\n"
        )
    sys.stderr.flush()

    stage_seconds: dict[str, float] = {}

    stage_started = time.monotonic()
    _step(started, "read_fasta")
    fasta = read_fake_fasta(
        args.fasta,
        presence_char=args.presence_char,
        absence_char=args.absence_char,
    )
    X = fasta.X
    n_samples, n_loci = X.shape
    stage_seconds["read_fasta"] = time.monotonic() - stage_started
    _step(started, "read_fasta", f"samples={n_samples} loci={n_loci}")

    stage_started = time.monotonic()
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
    stage_seconds["read_pairs"] = time.monotonic() - stage_started
    _step(started, "read_pairs", f"pairs={len(pairs)}")

    targets = np.unique(pairs[["u", "v"]].to_numpy(dtype=np.int64).reshape(-1))
    stage_started = time.monotonic()
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
    stage_seconds["build_covariance"] = time.monotonic() - stage_started
    _step(
        started,
        "build_covariance",
        f"source={kinship.source} rank={kdiag.rank}/{n_samples} "
        f"eig_min={kdiag.eigen_min:.3g} eig_max={kdiag.eigen_max:.3g}",
    )

    memory_estimate = estimate_scan_memory(
        n_samples=n_samples,
        n_loci=n_loci,
        worker_processes=args.threads,
        predictor_batch_size=args.predictor_batch_size,
        logical_cpus=runtime.logical_cpus,
    )
    _step(
        started,
        "memory",
        f"available_gib={memory_estimate.available_bytes / 2**30:.2f} "
        f"private_worker_mib={memory_estimate.private_bytes_per_worker / 2**20:.1f} "
        f"requested_private_gib={memory_estimate.requested_private_worker_bytes / 2**30:.2f} "
        f"recommended_max_workers={memory_estimate.recommended_max_workers_by_memory} "
        f"start_method={memory_estimate.start_method}",
    )
    for warning in memory_estimate.warnings:
        sys.stderr.write(
            f"[KOVAR] warning={warning} requested_workers={args.threads} "
            f"memory_recommended_max={memory_estimate.recommended_max_workers_by_memory}\n"
        )
    sys.stderr.flush()

    checkpoint: CheckpointStore | None = None
    if not args.no_checkpoint:
        stage_started = time.monotonic()
        _step(started, "checkpoint_fingerprint")
        fingerprint_sources: list[tuple[str, str]] = [
            ("fasta", args.fasta),
            ("pairs", args.pairs),
        ]
        if args.tree:
            fingerprint_sources.append(("tree", args.tree))
        try:
            input_fingerprints = fingerprint_inputs(fingerprint_sources)
            identity = _checkpoint_identity(
                args,
                input_fingerprints=input_fingerprints,
                n_samples=n_samples,
                n_loci=n_loci,
                n_pairs=len(pairs),
                blas_libraries=str(
                    runtime.summary_fields().get("runtime_blas_libraries", "unresolved")
                ),
            )
            checkpoint_path = (
                Path(args.checkpoint_dir)
                if args.checkpoint_dir
                else out / ".kovar_checkpoint"
            )
            resolved_checkpoint = checkpoint_path.resolve()
            resolved_out = out.resolve()
            if resolved_checkpoint == resolved_out or resolved_out.is_relative_to(
                resolved_checkpoint
            ):
                raise CheckpointError(
                    "Checkpoint directory must not be the output directory or one of its parents"
                )
            checkpoint = CheckpointStore(
                checkpoint_path,
                identity,
                tool_version=__version__,
                every=args.checkpoint_every,
                resume=args.resume,
            )
        except CheckpointError as exc:
            raise SystemExit(f"Checkpoint error: {exc}") from exc
        stage_seconds["checkpoint_fingerprint"] = time.monotonic() - stage_started
        _step(
            started,
            "checkpoint_fingerprint",
            f"resume={int(args.resume)} directory={checkpoint.root}",
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
    scan_metrics = ScanMetrics()
    stage_started = time.monotonic()
    try:
        results, response_models = scan_pairs_glmm(
            pairs,
            X,
            K,
            config,
            checkpoint=checkpoint,
            metrics=scan_metrics,
        )
    except CheckpointError as exc:
        raise SystemExit(f"Checkpoint error: {exc}") from exc
    stage_seconds["directional_scan"] = time.monotonic() - stage_started

    cache_diagnostics: dict[str, object] = {
        "response_pattern_cache": "exact_identical_complement",
        "n_response_loci": len(response_models),
    }
    required_cache_columns = {
        "response_pattern_id",
        "response_pattern_flipped",
        "response_pattern_size",
        "null_fit_reused",
    }
    if required_cache_columns.issubset(response_models.columns):
        reused = response_models["null_fit_reused"].fillna(False).astype(bool)
        flipped = response_models["response_pattern_flipped"].fillna(False).astype(bool)
        cache_diagnostics.update(
            {
                "n_unique_response_patterns": int(response_models["response_pattern_id"].nunique()),
                "n_null_glmm_fits": int((~reused).sum()),
                "n_null_glmm_reused": int(reused.sum()),
                "n_complement_response_loci": int(flipped.sum()),
                "max_response_pattern_size": (
                    int(response_models["response_pattern_size"].max()) if len(response_models) else 0
                ),
                "response_pattern_reuse_fraction": float(reused.mean()) if len(reused) else 0.0,
            }
        )
        _step(
            started,
            "directional_scan",
            f"duration={stage_seconds['directional_scan']:.1f}s "
            f"responses={len(response_models)} "
            f"unique_patterns={cache_diagnostics['n_unique_response_patterns']} "
            f"null_fits_reused={cache_diagnostics['n_null_glmm_reused']}",
        )
    else:
        cache_diagnostics["response_pattern_cache"] = "unavailable"
        _step(started, "directional_scan", f"duration={stage_seconds['directional_scan']:.1f}s")

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

    stage_started = time.monotonic()
    _step(started, "write_results")
    result_path = out / "ko_variation.tsv"
    model_path = out / "response_models.tsv"
    summary_path = out / "run_summary.txt"
    metadata_path = out / "execution_metadata.tsv"
    _atomic_dataframe_write(results, result_path)
    _atomic_dataframe_write(response_models, model_path)
    stage_seconds["write_primary_outputs"] = time.monotonic() - stage_started
    tested = int(np.isfinite(results["p_primary"].to_numpy(dtype=np.float64)).sum())

    checkpoint_enabled = checkpoint is not None
    summary_lines = [
        f"version\t{__version__}",
        "tool\tKO-Variation",
        "acronym\tKOVAR",
        "release_status\texperimental",
        "model\tdirectional_logistic_mixed_model_pql_score",
        "interpretation\tdirectional_covariation_not_causal_direction",
        f"n_samples\t{n_samples}",
        f"n_loci\t{n_loci}",
        f"n_input_pairs\t{len(pairs)}",
        f"n_directional_rows\t{len(results)}",
        f"n_directional_tests\t{tested}",
        f"direction_mode\t{args.direction_mode}",
        f"min_maf\t{args.min_maf}",
        f"min_cell_count\t{args.min_cell_count}",
        f"spa_mode\t{args.spa_mode}",
        f"full_refit_p\t{args.full_refit_p}",
        f"kinship_source\t{kinship.source}",
        f"tree\t{args.tree or 'NA'}",
        f"threads\t{args.threads}",
        "solver_backend\tdense_weighted_eigen_pql",
        "tau_profile_backend\texact_spectral_coordinates",
        f"checkpoint_enabled\t{int(checkpoint_enabled)}",
        f"checkpoint_resumed\t{int(bool(checkpoint and checkpoint.statistics.resumed))}",
        f"checkpoint_retained\t{int(bool(checkpoint and args.keep_checkpoints))}",
        f"execution_metadata\t{metadata_path.name}",
    ]
    _atomic_text_write(summary_path, "\n".join(summary_lines) + "\n")

    metadata_rows: list[dict[str, object]] = []
    metadata_rows.extend(_metadata_rows("configuration", {
        "direction_mode": args.direction_mode,
        "min_maf": args.min_maf,
        "min_cell_count": args.min_cell_count,
        "near_redundant_mismatch": args.near_redundant_mismatch,
        "exclude_near_redundant": int(args.exclude_near_redundant),
        "spa_mode": args.spa_mode,
        "full_refit_p": args.full_refit_p,
        "null_max_iter": args.null_max_iter,
        "null_tolerance": args.null_tolerance,
        "threads": args.threads,
        "worker_chunk_size": args.worker_chunk_size,
        "predictor_batch_size": args.predictor_batch_size,
        "checkpoint_every": args.checkpoint_every,
    }))
    metadata_rows.extend(_metadata_rows("runtime", runtime.summary_fields()))
    metadata_rows.extend(_metadata_rows("memory", memory_estimate.summary_fields()))
    metadata_rows.extend(_metadata_rows("response_cache", cache_diagnostics))
    metadata_rows.extend(_metadata_rows("stage_timing", stage_seconds, unit="seconds"))
    metadata_rows.extend(
        _metadata_rows("scan_timing", scan_metrics.seconds, unit="seconds")
    )
    metadata_rows.extend(_metadata_rows("scan_counts", scan_metrics.counts, unit="count"))
    metadata_rows.extend(_metadata_rows("kinship", {
        "rank": kdiag.rank,
        "eigen_min": kdiag.eigen_min,
        "eigen_max": kdiag.eigen_max,
        "roundoff_correction": kdiag.roundoff_correction,
        "mean_diag_before_norm": kinship.mean_diag_before_norm,
        "loci_used": kinship.n_loci_used,
        **{f"detail_{key}": value for key, value in kinship.details.items()},
    }))
    if checkpoint is None:
        metadata_rows.extend(_metadata_rows("checkpoint", {"enabled": 0}))
    else:
        metadata_rows.extend(_metadata_rows("checkpoint", checkpoint.metadata_fields()))
    _atomic_dataframe_write(
        pd.DataFrame(metadata_rows),
        metadata_path,
    )

    sys.stderr.write("[KOVAR] result_status_counts\n")
    sys.stderr.write(results["status"].fillna("NA").value_counts().to_string() + "\n")
    if checkpoint is not None:
        checkpoint_finalized = checkpoint.complete(cleanup=not args.keep_checkpoints)
        if not checkpoint_finalized:
            sys.stderr.write(
                f"[KOVAR] warning=checkpoint_cleanup_failed directory={checkpoint.root}; "
                "final outputs are complete and the checkpoint is marked complete\n"
            )
            sys.stderr.flush()
    _step(
        started,
        "complete",
        f"results={result_path} response_models={model_path} metadata={metadata_path}",
    )


if __name__ == "__main__":
    main()

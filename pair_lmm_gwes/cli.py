from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import __version__
from .io_utils import read_fake_fasta, read_pairs, validate_pairs
from .kinship import build_pangenome_grm, build_tree_covariance, build_pangenome_grm_with_tested_locus_mask, build_response_masked_grm_workspace
from .lmm_core import eigen_decompose_kinship, apply_eigen_rank_limit
from .scan import ScanConfig, scan_pairs_lmm, _apply_posthoc_tree_counts, _apply_posthoc_ctmc_counts
from .tree_events import Tree11EventCounter
from .nj_tree import build_nj_tree_from_binary_matrix


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Standalone pyseer-like LMM pairwise GWES for PAN-GWES/SpydrPick candidate pairs. "
            "Similarity is built from a Newick tree when --tree is supplied, otherwise internally from fake FASTA; no MDS, no SAIGE, no external pyseer dependency. v0.6.3 keeps the low-rank h2-capped LMM branch, preserves original PAN-GWES pair metadata columns in the result table, and leaves CTMC/tree post-hoc diagnostics optional. If a tree is needed and no event tree is supplied, v0.5.4 can build a midpoint-rooted NJ event tree from the fake FASTA."
        )
    )
    p.add_argument("--fasta", required=True, help="Fake FASTA matrix; A=0, C=1 by default.")
    p.add_argument("--pairs", required=True, help="Candidate pair file. First two columns are u/v if no header.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--threads", type=int, default=1, help="Response-wise multiprocessing workers [1].")
    p.add_argument("--tree", default=None, help="Optional Newick tree. If supplied, build pyseer-style LMM covariance K_ij=root-to-MRCA length; otherwise build K from fake FASTA.")
    p.add_argument("--tree-missing-length", choices=["error", "one", "zero"], default="error", help="How to handle missing branch lengths in --tree [error].")

    p.add_argument("--presence-char", default="C", help="Presence character in fake FASTA [C].")
    p.add_argument("--absence-char", default="A", help="Absence character in fake FASTA [A].")
    p.add_argument("--drop-invalid-pairs", action="store_true", help="Drop invalid/self/out-of-range pairs instead of failing.")
    p.add_argument("--max-pairs", type=int, default=0, help="Optional debugging limit on input pairs [0=all].")

    p.add_argument("--min-count", type=int, default=3, help="Minimum count in all four 2x2 cells [3].")
    p.add_argument("--min-count-auto", action="store_true", help="Use max(--min-count, ceil(0.0025*N)).")

    p.add_argument("--kinship-min-mac", type=int, default=2, help="Minimum minor allele count for loci used to build pangenome GRM [2].")
    p.add_argument("--kinship-chunk-size", type=int, default=4096, help="Locus chunk size for GRM construction [4096].")
    p.add_argument("--kinship-dtype", choices=["float64", "float32"], default="float64", help="Stored GRM dtype before eigendecomposition [float64].")
    p.add_argument("--grm-mask-mode", choices=["response", "tested", "none"], default="response",
                   help="For fake-FASTA GRM: response=build response-specific K excluding only response-proxy loci [default]; tested=v0.4.0 global candidate/proxy mask; none=v0.3.1 unmasked GRM.")
    p.add_argument("--grm-mask-r2", type=float, default=0.8,
                   help="Proxy masking threshold. In response mode, mask GRM loci with raw r^2 >= this to the response locus; in tested mode, to any tested locus [0.8].")
    p.add_argument("--grm-mask-mismatch", type=float, default=0.02,
                   help="Proxy masking threshold. In response mode, mask GRM loci with same/complement mismatch rate <= this to the response locus; in tested mode, to any tested locus [0.02].")
    p.add_argument("--grm-mask-target-chunk-size", type=int, default=256,
                   help="Target-locus chunk size for proxy masking [256].")
    p.add_argument("--grm-mask-grm-chunk-size", type=int, default=2048,
                   help="GRM-locus chunk size for proxy masking [2048].")
    p.add_argument("--grm-shrinkage", type=float, default=0.1,
                   help="Shrinkage toward identity after GRM masking: K=(1-lambda)K+lambda*I [0.1].")

    p.add_argument("--event-tree", default=None,
                   help="Rooted Newick tree used only for tree-based independent 11-origin counting. If omitted and --tree is supplied, --tree is reused. If neither is supplied, an NJ event tree is built automatically unless --no-auto-event-tree is set.")
    p.add_argument("--no-auto-event-tree", action="store_true",
                   help="Disable automatic midpoint-rooted NJ event-tree construction when tree 11-origin counting is requested and neither --event-tree nor --tree is supplied.")
    p.add_argument("--tree-11-mode", choices=["off", "prefilter", "posthoc"], default="off",
                   help="How to use tree-based 11-origin counting. off=disable; prefilter=apply before LMM after cheap filters; posthoc=compute only for LMM hits over --tree-11-posthoc-score-threshold [off].")
    p.add_argument("--nj-tree-out", default=None,
                   help="Output path for automatically built NJ event tree [OUT/auto_nj_event_tree.nwk].")
    p.add_argument("--nj-min-mac", type=int, default=2,
                   help="Minimum minor allele count for loci used in automatic NJ event-tree construction [2].")
    p.add_argument("--nj-max-loci", type=int, default=50000,
                   help="Maximum loci used for automatic NJ event-tree construction; 0 means all eligible loci [50000].")
    p.add_argument("--nj-seed", type=int, default=1,
                   help="Random seed used when --nj-max-loci subsamples eligible loci [1].")
    p.add_argument("--nj-chunk-size", type=int, default=2048,
                   help="Locus chunk size for automatic NJ Hamming-distance construction [2048].")
    p.add_argument("--tree-11-gain-min", type=int, default=3,
                   help="Minimum tree 11-gain count used by --tree-11-mode prefilter, and PASS/LOW_TREE_11_GAIN annotation threshold in posthoc mode [3].")
    p.add_argument("--tree-11-loss-cost", type=float, default=1.0,
                   help="Sankoff loss cost for 1->0 while counting tree 11 gains; gain cost is fixed to 1 [1].")
    p.add_argument("--tree-11-min-count", type=int, default=0,
                   help="Absolute minimum n11 count for the tree-origin filter [0].")
    p.add_argument("--tree-11-min-count-frac", type=float, default=0.01,
                   help="Population-fraction minimum for n11 before tree-origin testing [0.01]. Effective threshold is max(--tree-11-min-count, ceil(frac*N)).")
    p.add_argument("--tree-11-posthoc-score-threshold", type=float, default=5.0,
                   help="In posthoc tree/CTMC modes, evaluate only status=OK pairs with score_lmm >= this threshold [5.0].")
    p.add_argument("--tree-11-posthoc-distance-min", type=float, default=0.0,
                   help="In posthoc tree/CTMC modes, additionally require distance >= this value if a distance column exists [0=no distance restriction].")
    p.add_argument("--ctmc-11-mode", choices=["off", "posthoc"], default="off",
                   help="Binary CTMC effective 11-gain diagnostic. Uses reconstructed-edge likelihood with fixed second-order Taylor approximation and the same posthoc score/distance thresholds as tree-11 [posthoc].")
    p.add_argument("--ctmc-11-gain-min", type=float, default=1.5,
                   help="PASS/LOW_CTMC_11_GAIN annotation threshold for ctmc_11_gain_expected [1.5].")
    # Deprecated v0.4.2 MST event arguments are accepted for command-line compatibility but ignored.
    p.add_argument("--effective-event-min", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--effective-event-clusters", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--effective-event-source", default=None, help=argparse.SUPPRESS)

    p.add_argument("--predictor-batch-size", type=int, default=8192, help="Predictor block size per response LMM [8192]. Larger values reduce Python overhead at higher memory use.")
    p.add_argument("--worker-chunk-size", type=int, default=8, help="Multiprocessing scheduling chunk size in response-locus tasks [8].")
    p.add_argument("--h2-grid-size", type=int, default=21, help="Initial h2 grid points before scalar optimisation [21].")
    p.add_argument("--h2-max", type=float, default=0.9, help="Maximum h2 allowed in the null LMM optimisation [0.9].")
    p.add_argument("--grm-rank", type=int, default=20, help="Low-rank GRM covariance rank. Keeps the residual sample subspace but zeros all except the top rank kernel eigenvalues; 0=full rank [20].")

    p.add_argument("--pair-test", choices=["symmetric", "bidirectional"], default="symmetric",
                   help="Pair test mode. symmetric tests one orientation per unordered pair and writes p_lmm/score_lmm [symmetric]. bidirectional retains v0.3.0 two-direction tests.")
    p.add_argument("--h2-boundary-warn", type=float, default=0.899999,
                   help="Flag response models with h2 >= this value. Default is just below the v0.6.1 h2 cap [0.899999].")
    p.add_argument("--near-redundant-mismatch", type=float, default=0.02,
                   help="Mismatch-rate threshold for near-copy or near-complement annotation [0.02].")
    p.add_argument("--exclude-near-redundant", action="store_true",
                   help="Do not test pairs flagged near-copy or near-complement by --near-redundant-mismatch. Default: annotate only.")

    p.add_argument("--write-diagnostics", action="store_true", help="Write response and direction-level LMM diagnostic files.")
    p.add_argument("--no-progress", action="store_true", help="Suppress progress messages.")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into existing non-empty output directory.")
    p.add_argument("--version", action="version", version=f"pair-lmm-gwes {__version__}")
    return p.parse_args(argv)


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _step(t0: float, label: str, message: str = "") -> None:
    elapsed = time.time() - t0
    msg = f"[{_ts()}] [pair-lmm-gwes] step={label} elapsed={elapsed:.1f}s"
    if message:
        msg += f" {message}"
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def main(argv=None):
    # Reduce BLAS oversubscription with response-wise multiprocessing.
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    t0 = time.time()
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory exists and is not empty: {out}. Use --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    progress = not args.no_progress

    sys.stderr.write("[pair-lmm-gwes]\n")
    sys.stderr.write(f"  version={__version__}\n")
    sys.stderr.write(f"  fasta={args.fasta}\n")
    sys.stderr.write(f"  pairs={args.pairs}\n")
    sys.stderr.write(f"  out={args.out}\n")
    sys.stderr.write(f"  threads={args.threads}\n")
    sys.stderr.write(f"  tree={args.tree if args.tree else 'NA'}\n")
    sys.stderr.write(f"  pair_test={args.pair_test}\n")
    sys.stderr.write(f"  grm_mask_mode={args.grm_mask_mode if not args.tree else 'NA_tree_mode'}\n")
    sys.stderr.write(f"  grm_mask_r2={args.grm_mask_r2}\n")
    sys.stderr.write(f"  grm_mask_mismatch={args.grm_mask_mismatch}\n")
    sys.stderr.write(f"  grm_shrinkage={args.grm_shrinkage}\n")
    needs_event_tree = args.tree_11_mode != "off"
    event_tree_label = args.event_tree if args.event_tree else (args.tree if args.tree else ("AUTO_NJ_PREFILTER" if args.tree_11_mode == "prefilter" and not args.no_auto_event_tree else "AUTO_NJ_IF_POSTHOC_HITS" if args.tree_11_mode == "posthoc" and not args.no_auto_event_tree else "NA"))
    sys.stderr.write(f"  event_tree={event_tree_label}\n")
    sys.stderr.write(f"  tree_11_mode={args.tree_11_mode}\n")
    sys.stderr.write(f"  tree_11_gain_min={args.tree_11_gain_min}\n")
    sys.stderr.write(f"  tree_11_loss_cost={args.tree_11_loss_cost}\n")
    sys.stderr.write(f"  tree_11_min_count_frac={args.tree_11_min_count_frac}\n")
    sys.stderr.write(f"  tree_11_posthoc_score_threshold={args.tree_11_posthoc_score_threshold}\n")
    sys.stderr.write(f"  tree_11_posthoc_distance_min={args.tree_11_posthoc_distance_min}\n")
    sys.stderr.write(f"  ctmc_11_mode={args.ctmc_11_mode}\n")
    sys.stderr.write(f"  ctmc_11_gain_min={args.ctmc_11_gain_min}\n")
    sys.stderr.write(f"  grm_rank={args.grm_rank}\n")
    sys.stderr.write(f"  h2_max={args.h2_max}\n")
    sys.stderr.write(f"  predictor_batch_size={args.predictor_batch_size}\n")
    sys.stderr.write(f"  worker_chunk_size={args.worker_chunk_size}\n")
    sys.stderr.flush()

    _step(t0, "1/6_read_fasta")
    fm = read_fake_fasta(args.fasta, presence_char=args.presence_char, absence_char=args.absence_char)
    X = fm.X
    n_samples, n_loci = X.shape
    _step(t0, "1/6_read_fasta", f"samples={n_samples} loci={n_loci}")

    _step(t0, "2/6_read_pairs")
    pairs = read_pairs(args.pairs)
    pairs = validate_pairs(pairs, n_loci=n_loci, drop_invalid=args.drop_invalid_pairs)
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs.iloc[:args.max_pairs].copy().reset_index(drop=True)
    _step(t0, "2/6_read_pairs", f"pairs={len(pairs)}")

    min_count = int(args.min_count)
    if args.min_count_auto:
        import math
        min_count = max(min_count, int(math.ceil(0.0025 * n_samples)))
    response_grm_workspace = None
    kr_summary = None
    U = None
    S = None
    tree_event_counter = None
    tree_event_source = "none"
    tree_event_details = {}

    if args.tree:
        _step(t0, "3/6_build_kinship", f"source=tree tree={args.tree}")
        kr = build_tree_covariance(
            args.tree,
            sample_names=fm.sample_names,
            dtype=args.kinship_dtype,
            missing_length=args.tree_missing_length,
        )
        kr_summary = kr
        _step(t0, "3/6_build_kinship", f"source={kr_summary.source} n_tree_tips={kr.details.get('n_tree_tips', 'NA')}")
        _step(t0, "4/6_eigendecompose_kinship", f"N={n_samples}")
        U, S = eigen_decompose_kinship(kr.K)
        U, S = apply_eigen_rank_limit(U, S, int(args.grm_rank))
        del kr.K
        _step(t0, "4/6_eigendecompose_kinship", f"eig_min={float(np.min(S)):.3g} eig_max={float(np.max(S)):.3g} rank={args.grm_rank}")
    else:
        _step(
            t0,
            "3/6_build_kinship",
            f"source=fake_fasta_grm kinship_min_mac={args.kinship_min_mac} mask_mode={args.grm_mask_mode}",
        )
        target_loci = np.unique(pairs[["u", "v"]].to_numpy(dtype=np.int64).reshape(-1))
        if args.grm_mask_mode == "response":
            response_grm_workspace = build_response_masked_grm_workspace(
                X,
                min_mac=args.kinship_min_mac,
                chunk_size=args.kinship_chunk_size,
                dtype=args.kinship_dtype,
                progress=progress,
                mask_r2=args.grm_mask_r2,
                mask_mismatch=args.grm_mask_mismatch,
                mask_grm_chunk_size=args.grm_mask_grm_chunk_size,
                shrinkage=args.grm_shrinkage,
            )
            kr_summary = response_grm_workspace.summary_result()
            _step(
                t0,
                "3/6_build_kinship",
                f"source={kr_summary.source} valid_loci={kr_summary.n_loci_used} filtered_loci={kr_summary.n_loci_filtered} response_specific_K=1",
            )
            _step(t0, "4/6_eigendecompose_kinship", "response-specific; done inside each response worker")
        else:
            kr = build_pangenome_grm_with_tested_locus_mask(
                X,
                target_loci=target_loci,
                min_mac=args.kinship_min_mac,
                chunk_size=args.kinship_chunk_size,
                dtype=args.kinship_dtype,
                progress=progress,
                mask_mode=args.grm_mask_mode,
                mask_r2=args.grm_mask_r2,
                mask_mismatch=args.grm_mask_mismatch,
                mask_target_chunk_size=args.grm_mask_target_chunk_size,
                mask_grm_chunk_size=args.grm_mask_grm_chunk_size,
                shrinkage=args.grm_shrinkage,
            )
            kr_summary = kr
            _step(
                t0,
                "3/6_build_kinship",
                f"source={kr_summary.source} used_loci={kr_summary.n_loci_used} filtered_loci={kr_summary.n_loci_filtered} "
                f"masked={kr.details.get('n_loci_removed_by_mask', 'NA')}",
            )
            _step(t0, "4/6_eigendecompose_kinship", f"N={n_samples}")
            U, S = eigen_decompose_kinship(kr.K)
            U, S = apply_eigen_rank_limit(U, S, int(args.grm_rank))
            del kr.K
            _step(t0, "4/6_eigendecompose_kinship", f"eig_min={float(np.min(S)):.3g} eig_max={float(np.max(S)):.3g} rank={args.grm_rank}")

    event_tree_path = args.event_tree if args.event_tree else args.tree
    auto_nj_result = None
    if args.tree_11_mode == "prefilter" and not event_tree_path and not args.no_auto_event_tree:
        event_tree_path = args.nj_tree_out if args.nj_tree_out else str(out / "auto_nj_event_tree.nwk")
        _step(
            t0,
            "4a/6_auto_nj_event_tree",
            f"out={event_tree_path} min_mac={args.nj_min_mac} max_loci={args.nj_max_loci}",
        )
        auto_nj_result = build_nj_tree_from_binary_matrix(
            X,
            fm.sample_names,
            event_tree_path,
            min_mac=args.nj_min_mac,
            max_loci=args.nj_max_loci,
            seed=args.nj_seed,
            chunk_size=args.nj_chunk_size,
            progress=progress,
        )
        tree_event_source = "auto_nj_tree_sankoff_11_gain"
        tree_event_details.update(auto_nj_result.details())
        _step(t0, "4a/6_auto_nj_event_tree", f"wrote={event_tree_path} loci_used={auto_nj_result.n_loci_used}")

    if args.tree_11_mode == "prefilter" and event_tree_path:
        _step(t0, "4b/6_tree_11_event_counter", f"tree={event_tree_path} gain_min={args.tree_11_gain_min}")
        tree_event_counter = Tree11EventCounter.from_newick(event_tree_path, fm.sample_names, missing_length="zero")
        if tree_event_source == "none":
            tree_event_source = "tree_sankoff_11_gain"
        tree_event_details.update(tree_event_counter.summary_details())
        _step(t0, "4b/6_tree_11_event_counter", f"nodes={tree_event_counter.n_nodes} samples={tree_event_counter.n_samples}")
    elif args.tree_11_mode == "prefilter":
        sys.stderr.write("[pair-lmm-gwes] warning: --tree-11-mode is not off but no --event-tree/--tree was supplied and automatic NJ was disabled; tree 11-origin counting disabled.\n")
        sys.stderr.flush()

    _step(t0, "5/6_scan_pairs", f"min_count={min_count}")
    config = ScanConfig(
        min_count=min_count,
        threads=args.threads,
        predictor_batch_size=args.predictor_batch_size,
        worker_chunk_size=args.worker_chunk_size,
        h2_grid_size=args.h2_grid_size,
        h2_max=args.h2_max,
        grm_rank=args.grm_rank,
        progress=progress,
        write_diagnostics=args.write_diagnostics,
        pair_test=args.pair_test,
        h2_boundary_warn=args.h2_boundary_warn,
        near_redundant_mismatch=args.near_redundant_mismatch,
        exclude_near_redundant=args.exclude_near_redundant,
        tree_11_mode=args.tree_11_mode,
        tree_11_gain_min=args.tree_11_gain_min,
        tree_11_loss_cost=args.tree_11_loss_cost,
        tree_11_min_count=args.tree_11_min_count,
        tree_11_min_count_frac=args.tree_11_min_count_frac,
        tree_11_posthoc_score_threshold=args.tree_11_posthoc_score_threshold,
        tree_11_posthoc_distance_min=args.tree_11_posthoc_distance_min,
        tree_event_counter=tree_event_counter,
        tree_event_source=tree_event_source,
        ctmc_11_mode=args.ctmc_11_mode,
        ctmc_11_gain_min=args.ctmc_11_gain_min,
    )
    result, null_df, diag_df = scan_pairs_lmm(pairs, X, U, S, config, response_grm_workspace=response_grm_workspace)

    # v0.5.4 lazy post-hoc tree/CTMC counting: build/read event tree only if over-threshold LMM hits need it.
    need_posthoc_tree = args.tree_11_mode == "posthoc" or args.ctmc_11_mode == "posthoc"
    if need_posthoc_tree:
        score = result["score_lmm"].astype(float) if "score_lmm" in result.columns else np.full(len(result), np.nan)
        posthoc_mask = score >= float(args.tree_11_posthoc_score_threshold)
        if "status" in result.columns:
            posthoc_mask &= result["status"].astype(str).eq("OK")
        if float(args.tree_11_posthoc_distance_min) > 0 and "distance" in result.columns:
            posthoc_mask &= result["distance"].astype(float) >= float(args.tree_11_posthoc_distance_min)
        n_posthoc = int(np.count_nonzero(posthoc_mask))
        _step(t0, "5b/6_posthoc_tree_ctmc", f"candidates={n_posthoc} score_threshold={args.tree_11_posthoc_score_threshold} distance_min={args.tree_11_posthoc_distance_min}")
        if n_posthoc > 0:
            if not event_tree_path and not args.no_auto_event_tree:
                event_tree_path = args.nj_tree_out if args.nj_tree_out else str(out / "auto_nj_event_tree.nwk")
                _step(t0, "5b/6_auto_nj_event_tree", f"out={event_tree_path} min_mac={args.nj_min_mac} max_loci={args.nj_max_loci}")
                auto_nj_result = build_nj_tree_from_binary_matrix(
                    X, fm.sample_names, event_tree_path, min_mac=args.nj_min_mac, max_loci=args.nj_max_loci,
                    seed=args.nj_seed, chunk_size=args.nj_chunk_size, progress=progress,
                )
                tree_event_source = "auto_nj_tree_posthoc"
                tree_event_details.update(auto_nj_result.details())
                _step(t0, "5b/6_auto_nj_event_tree", f"wrote={event_tree_path} loci_used={auto_nj_result.n_loci_used}")
            if event_tree_path:
                _step(t0, "5b/6_tree_event_counter", f"tree={event_tree_path}")
                tree_event_counter = Tree11EventCounter.from_newick(event_tree_path, fm.sample_names, missing_length="zero")
                if tree_event_source == "none":
                    tree_event_source = "tree_posthoc"
                tree_event_details.update(tree_event_counter.summary_details())
                config.tree_event_counter = tree_event_counter
                config.tree_event_source = tree_event_source
                if args.tree_11_mode == "posthoc":
                    result = _apply_posthoc_tree_counts(result, X, config)
                if args.ctmc_11_mode == "posthoc":
                    result = _apply_posthoc_ctmc_counts(result, X, config)
                _step(t0, "5b/6_posthoc_tree_ctmc", "complete")
            elif args.no_auto_event_tree:
                sys.stderr.write("[pair-lmm-gwes] warning: posthoc tree/CTMC requested but no event tree is available and automatic NJ was disabled.\n")
                sys.stderr.flush()

    _step(t0, "6/6_write_results")
    result_path = out / "pair_lmm_gwes.tsv"
    result.to_csv(result_path, sep="\t", index=False, na_rep="NA")
    if args.write_diagnostics:
        null_df.to_csv(out / "lmm_response_summary.tsv", sep="\t", index=False, na_rep="NA")
        diag_df.to_csv(out / "lmm_direction_diagnostics.tsv", sep="\t", index=False, na_rep="NA")
    with (out / "run_summary.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"version\t{__version__}\n")
        fh.write(f"model\tpyseer_like_lowrank_lmm_h2cap\n")
        fh.write(f"n_samples\t{n_samples}\n")
        fh.write(f"n_loci\t{n_loci}\n")
        fh.write(f"n_pairs\t{len(pairs)}\n")
        fh.write(f"kinship_source\t{kr_summary.source}\n")
        fh.write(f"tree\t{args.tree if args.tree else 'NA'}\n")
        fh.write(f"event_tree\t{event_tree_path if event_tree_path else 'NA'}\n")
        fh.write(f"kinship_loci_used\t{kr_summary.n_loci_used}\n")
        fh.write(f"kinship_loci_filtered\t{kr_summary.n_loci_filtered}\n")
        fh.write(f"kinship_mean_diag_before_norm\t{kr_summary.mean_diag_before_norm}\n")
        for key in sorted(kr_summary.details):
            fh.write(f"kinship_detail_{key}\t{kr_summary.details[key]}\n")
        fh.write(f"min_count\t{min_count}\n")
        fh.write(f"grm_rank\t{args.grm_rank}\n")
        fh.write(f"h2_max\t{args.h2_max}\n")
        fh.write(f"predictor_batch_size\t{args.predictor_batch_size}\n")
        fh.write(f"worker_chunk_size\t{args.worker_chunk_size}\n")
        fh.write(f"tree_11_gain_min\t{args.tree_11_gain_min}\n")
        fh.write(f"tree_11_loss_cost\t{args.tree_11_loss_cost}\n")
        fh.write(f"tree_11_min_count\t{args.tree_11_min_count}\n")
        fh.write(f"tree_11_min_count_frac\t{args.tree_11_min_count_frac}\n")
        fh.write(f"tree_11_event_source\t{tree_event_source}\n")
        fh.write(f"ctmc_11_mode\t{args.ctmc_11_mode}\n")
        fh.write(f"ctmc_11_gain_min\t{args.ctmc_11_gain_min}\n")
        fh.write("ctmc_11_exp_method\treconstructed_edge_taylor2\n")
        for key in sorted(tree_event_details):
            fh.write(f"{key}\t{tree_event_details[key]}\n")
        fh.write(f"threads\t{args.threads}\n")
        fh.write("notes\tstandalone_pyseer_like_lowrank_lmm_h2cap_rank20_no_mds_no_saige_ctmc_off_by_default\n")
    if "status" in result.columns:
        sys.stderr.write("[pair-lmm-gwes] status_counts\n")
        sys.stderr.write(result["status"].fillna("NA").value_counts().to_string() + "\n")
    if "score_lmm" in result.columns:
        sc = result["score_lmm"].astype(float)
        sys.stderr.write("[pair-lmm-gwes] score_lmm_summary\n")
        sys.stderr.write(sc.describe().to_string() + "\n")
    if "h2_boundary" in result.columns:
        sys.stderr.write("[pair-lmm-gwes] h2_boundary_counts\n")
        sys.stderr.write(result["h2_boundary"].fillna(0).astype(int).value_counts().to_string() + "\n")
    _step(t0, "complete", f"wrote={result_path}")


if __name__ == "__main__":
    main()

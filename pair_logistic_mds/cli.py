#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from pair_logistic_mds.io_utils import (
    read_fake_fasta,
    read_pairs,
    align_matrix_to_tips,
    validate_pairs,
)
from pair_logistic_mds.tree_mds import (
    read_tree,
    get_tip_names,
    compute_tree_distance_matrix,
    classical_mds,
)
from pair_logistic_mds.pair_scan import ScanConfig, scan_pairs
from pair_logistic_mds.score_scan import ScoreScanConfig, score_scan_pairs


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Pairwise SEER/pyseer-style locus-locus association using "
            "tree-MDS population-structure covariates."
        )
    )

    # Required inputs
    p.add_argument("--tree", required=True, help="Newick tree with branch lengths")
    p.add_argument("--fasta", required=True, help="Fake FASTA presence/absence matrix")
    p.add_argument(
        "--pairs",
        required=True,
        help="Whitespace pair file: u v distance [extra columns ignored]",
    )
    p.add_argument("--out", required=True, help="Main output TSV")

    # Main workflow mode
    p.add_argument(
        "--mode",
        choices=["score", "exact", "score-then-exact"],
        default="score-then-exact",
        help=(
            "Analysis mode. 'score' = fast MDS-adjusted score test only; "
            "'exact' = exact logistic/Firth on all pairs; "
            "'score-then-exact' = score scan all pairs, then exact refit top pairs "
            "[score-then-exact]"
        ),
    )

    # Optional extra outputs
    p.add_argument(
        "--score-out",
        default=None,
        help=(
            "Optional TSV for full fast score-scan results. "
            "Useful in --mode score-then-exact for plotting all candidate pairs."
        ),
    )
    p.add_argument(
        "--top-score-out",
        default=None,
        help=(
            "Optional TSV for selected top score pairs before exact refit. "
            "Only used in --mode score-then-exact."
        ),
    )

    # MDS / population structure
    p.add_argument("--mds-k", type=int, default=10, help="Number of MDS axes to use [10]")
    p.add_argument("--write-mds", default=None, help="Optional MDS covariates TSV")
    p.add_argument("--write-eigen", default=None, help="Optional retained MDS eigenvalues TSV")

    # FASTA encoding
    p.add_argument("--presence-char", default="C", help="Presence character in fake FASTA [C]")
    p.add_argument("--absence-char", default="A", help="Absence character in fake FASTA [A]")

    # Sample/pair handling
    p.add_argument(
        "--allow-intersection",
        action="store_true",
        help="Use only shared tree/FASTA samples instead of requiring exact match",
    )
    p.add_argument(
        "--write-dropped-samples",
        default=None,
        help="Optional file of dropped samples when using --allow-intersection",
    )
    p.add_argument(
        "--drop-invalid-pairs",
        action="store_true",
        help="Drop invalid/self/out-of-range pairs instead of failing",
    )

    # Filtering / model stability
    p.add_argument(
        "--min-maf",
        type=float,
        default=0.01,
        help="Minimum minor allele/state frequency for both loci [0.01]",
    )
    p.add_argument(
        "--low-count-threshold",
        "--min-count",
        dest="low_count_threshold",
        type=int,
        default=1,
        help="Flag pairs with min cell count <= threshold [1]",
    )
    p.add_argument(
        "--reduce-mds-by-minor-count",
        action="store_true",
        help=(
            "For each directional exact logistic model, reduce the number of MDS axes "
            "based on the response minor-state count."
        ),
    )

    # Exact/Firth fitting options
    p.add_argument(
        "--no-firth",
        action="store_true",
        help="Disable Firth fallback for separated/unstable exact logistic fits",
    )
    p.add_argument(
        "--high-bse-threshold",
        type=float,
        default=3.0,
        help="Fallback to Firth if beta SE exceeds this [3.0]",
    )

    # Score-then-exact options
    p.add_argument(
        "--top-frac",
        type=float,
        default=0.01,
        help=(
            "Fraction of score-scan pairs to refit exactly in --mode score-then-exact [0.01]"
        ),
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help=(
            "Number of score-scan pairs to refit exactly. "
            "Overrides --top-frac if supplied."
        ),
    )
    p.add_argument(
        "--score-p-col",
        default="score_p_pair_max",
        help=(
            "Column used to rank score-scan pairs for exact refit "
            "[score_p_pair_max]"
        ),
    )
    
    p.add_argument(
        "--distance-bin-size",
        type=int,
        default=10000,
        help=(
            "Distance bin size for distance-stratified top-pair selection "
            "in --mode score-then-exact [10000]"
        ),
    )

    p.add_argument(
        "--min-per-bin",
        type=int,
        default=100,
        help=(
            "Minimum number of pairs selected per distance bin in "
            "--mode score-then-exact [100]"
        ),
    )

    # Performance
    p.add_argument("--threads", type=int, default=1, help="Parallel processes for exact pair scan [1]")

    return p.parse_args()


def _write_optional_tsv(df: pd.DataFrame, path: str | None, label: str) -> None:
    if path is None:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False, na_rep="NA")
    sys.stderr.write(f"  wrote_{label}={out_path}\n")


def _select_top_score_pairs(
    score_df: pd.DataFrame,
    top_frac: float,
    top_n: int | None,
    score_p_col: str,
    distance_bin_size: int = 10000,
    min_per_bin: int = 100,
) -> pd.DataFrame:
    """
    Select top score pairs for exact refit.

    If top_n is supplied, use global top-N selection.

    Otherwise, use distance-stratified selection:
        for each distance bin, keep top max(min_per_bin, ceil(top_frac * n_bin))

    Default:
        10 kb bins, top 1% per bin, minimum 100 per bin.
    """
    if score_p_col not in score_df.columns:
        raise ValueError(
            f"Cannot select top score pairs: column '{score_p_col}' not found. "
            f"Available columns: {', '.join(score_df.columns)}"
        )

    required = {"u", "v", "distance"}
    missing = required - set(score_df.columns)
    if missing:
        raise ValueError(
            f"Cannot select top score pairs: missing columns: {', '.join(sorted(missing))}"
        )

    if top_n is not None:
        if top_n <= 0:
            raise ValueError("--top-n must be > 0")

        finite = score_df[
            np.isfinite(score_df[score_p_col].to_numpy(dtype=float, na_value=np.nan))
        ].copy()

        if len(finite) > 0:
            return finite.sort_values(score_p_col, ascending=True).head(top_n).copy()
        return score_df.head(top_n).copy()

    if not (0.0 < top_frac <= 1.0):
        raise ValueError("--top-frac must be in the range (0, 1]")

    if distance_bin_size <= 0:
        raise ValueError("--distance-bin-size must be > 0")

    if min_per_bin < 0:
        raise ValueError("--min-per-bin must be >= 0")

    df = score_df.copy()
    df["_p_rank"] = pd.to_numeric(df[score_p_col], errors="coerce")
    df["_distance_num"] = pd.to_numeric(df["distance"], errors="coerce")

    # Only finite p-values can be ranked.
    df = df[np.isfinite(df["_p_rank"].to_numpy())].copy()

    if len(df) == 0:
        return score_df.head(0).copy()

    # Put negative or missing distances into a separate bin.
    # This handles distance = -1 cases without discarding them.
    dist = df["_distance_num"].to_numpy()
    finite_nonnegative = np.isfinite(dist) & (dist >= 0)

    bin_id = np.full(len(df), -1, dtype=np.int64)
    bin_id[finite_nonnegative] = (
        np.floor(dist[finite_nonnegative] / distance_bin_size).astype(np.int64)
    )

    df["_distance_bin"] = bin_id

    selected = []

    for _, g in df.groupby("_distance_bin", sort=True):
        n_bin = len(g)
        n_keep = max(min_per_bin, int(np.ceil(top_frac * n_bin)))
        n_keep = min(n_keep, n_bin)

        selected.append(
            g.sort_values("_p_rank", ascending=True).head(n_keep)
        )

    out = pd.concat(selected, axis=0)

    # Remove helper columns.
    out = out.drop(columns=["_p_rank", "_distance_num", "_distance_bin"])

    # Stable final ordering: strongest p first.
    out = out.sort_values(score_p_col, ascending=True).copy()

    return out

def main():
    args = parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sys.stderr.write("[1/5] Reading tree and FASTA\n")
    tree = read_tree(args.tree)
    tip_names = get_tip_names(tree)

    fm = read_fake_fasta(
        args.fasta,
        presence_char=args.presence_char,
        absence_char=args.absence_char,
    )

    aligned_names, Y, dropped = align_matrix_to_tips(
        fm.sample_names,
        fm.Y,
        tip_names,
        allow_intersection=args.allow_intersection,
    )

    if args.write_dropped_samples is not None:
        dropped_path = Path(args.write_dropped_samples)
        dropped_path.parent.mkdir(parents=True, exist_ok=True)
        dropped_path.write_text("\n".join(dropped) + ("\n" if dropped else ""))

    sys.stderr.write(f"  samples={Y.shape[0]} loci={Y.shape[1]}\n")

    sys.stderr.write("[2/5] Reading and validating pairs\n")
    pairs = read_pairs(args.pairs)
    pairs = validate_pairs(
        pairs,
        n_loci=Y.shape[1],
        drop_invalid=args.drop_invalid_pairs,
    )
    sys.stderr.write(f"  pairs={len(pairs)}\n")

    sys.stderr.write("[3/5] Computing tree distances and MDS covariates\n")
    _, D = compute_tree_distance_matrix(tree, aligned_names)
    MDS, evals = classical_mds(D, k=args.mds_k)
    sys.stderr.write(f"  retained_mds_axes={MDS.shape[1]}\n")

    if args.write_mds is not None:
        mds_df = pd.DataFrame(MDS, columns=[f"MDS{i + 1}" for i in range(MDS.shape[1])])
        mds_df.insert(0, "sample", aligned_names)
        _write_optional_tsv(mds_df, args.write_mds, "mds")

    if args.write_eigen is not None:
        eig_df = pd.DataFrame(
            {
                "axis": np.arange(1, len(evals) + 1),
                "eigenvalue": evals,
            }
        )
        if len(evals) > 0 and float(np.sum(evals)) > 0:
            eig_df["fraction_retained"] = eig_df["eigenvalue"] / eig_df["eigenvalue"].sum()
        _write_optional_tsv(eig_df, args.write_eigen, "eigen")

    config = ScanConfig(
        use_firth=not args.no_firth,
        high_bse_threshold=args.high_bse_threshold,
        low_count_threshold=args.low_count_threshold,
        min_maf=args.min_maf,
        reduce_mds_by_minor_count=args.reduce_mds_by_minor_count,
    )

    if args.mode == "score":
        sys.stderr.write("[4/5] Fast MDS-adjusted score scan\n")
        score_config = ScoreScanConfig(
            min_maf=args.min_maf,
            low_count_threshold=args.low_count_threshold,
        )
        score_df = score_scan_pairs(
            pairs=pairs,
            Y=Y,
            covariates=MDS,
            config=score_config,
        )

        # Main output is the score output in score mode.
        sys.stderr.write("[5/5] Writing output\n")
        score_df.to_csv(out_path, sep="\t", index=False, na_rep="NA")
        sys.stderr.write(f"Done: {out_path}\n")
        return

    if args.mode == "exact":
        sys.stderr.write("[4/5] Exact logistic/Firth models for all pairs\n")
        exact_df = scan_pairs(
            pairs=pairs,
            Y=Y,
            covariates=MDS,
            config=config,
            threads=args.threads,
        )

        sys.stderr.write("[5/5] Writing output\n")
        exact_df.to_csv(out_path, sep="\t", index=False, na_rep="NA")
        sys.stderr.write(f"Done: {out_path}\n")
        return

    if args.mode == "score-then-exact":
        sys.stderr.write("[4/5] Fast MDS-adjusted score scan\n")
        score_config = ScoreScanConfig(
            min_maf=args.min_maf,
            low_count_threshold=args.low_count_threshold,
        )
        score_df = score_scan_pairs(
            pairs=pairs,
            Y=Y,
            covariates=MDS,
            config=score_config,
        )

        _write_optional_tsv(score_df, args.score_out, "score_scan")

        top_score_df = _select_top_score_pairs(
            score_df=score_df,
            top_frac=args.top_frac,
            top_n=args.top_n,
            score_p_col=args.score_p_col,
            distance_bin_size=args.distance_bin_size,
            min_per_bin=args.min_per_bin,
        )
        _write_optional_tsv(top_score_df, args.top_score_out, "top_score_pairs")

        top_pairs = top_score_df[["u", "v", "distance"]].copy()

        sys.stderr.write(
            f"  selected_top_pairs={len(top_pairs)} "
            f"from_total_pairs={len(score_df)} "
            f"rank_col={args.score_p_col}\n"
        )

        sys.stderr.write("[5/5] Exact logistic/Firth refit for top score pairs\n")
        exact_df = scan_pairs(
            pairs=top_pairs,
            Y=Y,
            covariates=MDS,
            config=config,
            threads=args.threads,
        )

        # Attach score-scan columns to exact-refit output for plotting/comparison.
        score_cols_to_merge = [
            c for c in top_score_df.columns
            if c not in {"n11", "n10", "n01", "n00", "freq_u", "freq_v", "status"}
        ]

        merged = exact_df.merge(
            top_score_df[score_cols_to_merge],
            on=["u", "v", "distance"],
            how="left",
            suffixes=("", "_score"),
        )

        merged.to_csv(out_path, sep="\t", index=False, na_rep="NA")
        sys.stderr.write(f"Done: {out_path}\n")
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
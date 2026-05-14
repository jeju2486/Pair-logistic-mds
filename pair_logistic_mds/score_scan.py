from __future__ import annotations

from dataclasses import dataclass
import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning


@dataclass
class ScoreScanConfig:
    min_maf: float = 0.01
    low_count_threshold: int = 1
    reduce_mds_by_minor_count: bool = False
    batch_size: int = 4096


def _status_from_counts(n10: int, n01: int, min_cell: int, low_count_threshold: int) -> str:
    notes = ["SCORE"]
    if min_cell <= low_count_threshold:
        notes.append("LOW_CELL")
    if n10 == 0 or n01 == 0:
        notes.append("SEPARATION_WARNING")
    return ";".join(notes)


def _null_covariates_for_response(y: np.ndarray, covariates: np.ndarray, reduce: bool) -> np.ndarray:
    y = np.asarray(y, dtype=np.uint8)
    k_global = covariates.shape[1]
    if reduce:
        n1 = int(np.sum(y == 1))
        n0 = int(np.sum(y == 0))
        minor_count = min(n1, n0)
        k_eff = min(k_global, max(0, minor_count - 2))
    else:
        k_eff = k_global
    if k_eff <= 0:
        return np.ones((y.size, 1), dtype=np.float64)
    return np.column_stack([np.ones(y.size, dtype=np.float64), covariates[:, :k_eff].astype(np.float64, copy=False)])


def _fit_null_logit(y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Fit y ~ Z and return p, W, residual, inv(Z'WZ), status."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if np.unique(y).size < 2:
        raise ValueError("LOW_VARIATION")
    mean = (float(y.sum()) + 0.5) / (len(y) + 1.0)
    mean = min(max(mean, 1e-8), 1.0 - 1e-8)
    start = np.zeros(Z.shape[1], dtype=np.float64)
    start[0] = np.log(mean / (1.0 - mean))
    mod = sm.Logit(y, Z)
    mod.raise_on_perfect_prediction = True
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        warnings.filterwarnings("error", category=PerfectSeparationWarning)
        warnings.filterwarnings("error", category=ConvergenceWarning)
        res = mod.fit(start_params=start, method="newton", disp=False, maxiter=100)
    if not bool(res.mle_retvals.get("converged", False)):
        raise RuntimeError("NULL_NOT_CONVERGED")
    eta = np.clip(Z @ np.asarray(res.params, dtype=np.float64), -35.0, 35.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    W = p * (1.0 - p)
    r = y - p
    ZWZ = Z.T @ (W[:, None] * Z)
    inv_ZWZ = np.linalg.pinv(ZWZ)
    return p, W, r, inv_ZWZ, "NULL_OK"


def _score_predictors_for_response(
    response_locus: int,
    predictor_loci: np.ndarray,
    Y: np.ndarray,
    covariates: np.ndarray,
    config: ScoreScanConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Score-test many predictors for one response locus.

    Returns beta_approx, p_values, n11_values, status_base.
    """
    y = Y[:, response_locus].astype(np.float64, copy=False)
    Z = _null_covariates_for_response(y, covariates, config.reduce_mds_by_minor_count)
    p_out = np.full(predictor_loci.shape[0], np.nan, dtype=np.float64)
    beta_out = np.full(predictor_loci.shape[0], np.nan, dtype=np.float64)
    n11_out = np.zeros(predictor_loci.shape[0], dtype=np.int64)

    try:
        _, W, r, inv_ZWZ, status_base = _fit_null_logit(y, Z)
    except Exception as e:
        return beta_out, p_out, n11_out, f"NULL_FAILED:{str(e)}"

    WZ = W[:, None] * Z
    y_int = Y[:, response_locus].astype(np.int64, copy=False)

    for start in range(0, predictor_loci.shape[0], config.batch_size):
        end = min(start + config.batch_size, predictor_loci.shape[0])
        loci = predictor_loci[start:end]
        X = Y[:, loci].astype(np.float64, copy=False)  # N x batch
        U = X.T @ r
        XTWX = X.T @ W  # binary x => x^T W x = x^T W
        XTWZ = X.T @ WZ
        # Efficient variance: x'Wx - x'WZ (Z'WZ)^-1 Z'Wx
        middle = XTWZ @ inv_ZWZ
        V = XTWX - np.sum(middle * XTWZ, axis=1)
        valid = np.isfinite(V) & (V > 1e-12)
        stat = np.full(V.shape[0], np.nan, dtype=np.float64)
        stat[valid] = (U[valid] * U[valid]) / V[valid]
        p_out[start:end] = stats.chi2.sf(stat, 1)
        beta = np.full(V.shape[0], np.nan, dtype=np.float64)
        beta[valid] = U[valid] / V[valid]
        beta_out[start:end] = beta
        n11_out[start:end] = y_int @ Y[:, loci].astype(np.int64, copy=False)

    return beta_out, p_out, n11_out, status_base


def score_scan_pairs(
    pairs: pd.DataFrame,
    Y: np.ndarray,
    covariates: np.ndarray,
    config: ScoreScanConfig,
) -> pd.DataFrame:
    """Fast pyseer/SEER-style logistic score scan for all pairs.

    This fits y ~ MDS once per response locus and score-tests each partner locus.
    It computes both directions and returns approximate score-test p-values.

    MAF filtering is applied before score testing:
        min(freq_locus, 1 - freq_locus) >= config.min_maf
    """
    t0 = time.time()
    n_pairs = len(pairs)
    n_samples = Y.shape[0]

    if n_pairs == 0:
        return pd.DataFrame(columns=_score_columns())

    u = pairs["u"].to_numpy(dtype=np.int64)
    v = pairs["v"].to_numpy(dtype=np.int64)
    distance = pairs["distance"].to_numpy()

    # ------------------------------------------------------------
    # Per-locus frequencies and MAF
    # ------------------------------------------------------------
    locus_counts = Y.sum(axis=0).astype(np.float64)
    locus_freq = locus_counts / float(n_samples)
    locus_maf = np.minimum(locus_freq, 1.0 - locus_freq)

    valid_maf = (locus_maf[u] >= config.min_maf) & (locus_maf[v] >= config.min_maf)

    # ------------------------------------------------------------
    # Allocate output arrays
    # ------------------------------------------------------------
    n11 = np.zeros(n_pairs, dtype=np.int64)
    beta_u_to_v = np.full(n_pairs, np.nan, dtype=np.float64)
    p_u_to_v = np.full(n_pairs, np.nan, dtype=np.float64)
    beta_v_to_u = np.full(n_pairs, np.nan, dtype=np.float64)
    p_v_to_u = np.full(n_pairs, np.nan, dtype=np.float64)

    status = np.array(["LOW_MAF"] * n_pairs, dtype=object)

    # ------------------------------------------------------------
    # Directional score-test runner
    # ------------------------------------------------------------
    def run_direction(
        response_arr: np.ndarray,
        predictor_arr: np.ndarray,
        p_out: np.ndarray,
        beta_out: np.ndarray,
        fill_counts: bool,
        direction_label: str,
    ) -> None:
        valid_idx = np.where(valid_maf)[0]

        if valid_idx.size == 0:
            sys.stderr.write(f"  {direction_label}: no pairs passed min_maf={config.min_maf}\n")
            return

        resp_valid = response_arr[valid_idx]
        unique_resp = np.unique(resp_valid)

        total_groups = unique_resp.size
        done_pairs = 0
        last_report = 0
        progress_every = max(1, valid_idx.size // 100)

        for gi, resp in enumerate(unique_resp, start=1):
            pair_idx = valid_idx[resp_valid == resp]
            predictors = predictor_arr[pair_idx].astype(np.int64, copy=False)

            b_beta, b_p, b_n11, base_status = _score_predictors_for_response(
                int(resp),
                predictors,
                Y,
                covariates,
                config,
            )

            beta_out[pair_idx] = b_beta
            p_out[pair_idx] = b_p

            if fill_counts:
                n11[pair_idx] = b_n11

            fail_mask = ~np.isfinite(b_p)
            if np.any(fail_mask):
                for idx in pair_idx[fail_mask]:
                    status[idx] = base_status

            done_pairs += pair_idx.size

            if (
                gi == total_groups
                or done_pairs - last_report >= progress_every
            ):
                elapsed = time.time() - t0
                rate = done_pairs / elapsed if elapsed > 0 else 0.0
                pct = 100.0 * done_pairs / max(1, valid_idx.size)

                sys.stderr.write(
                    f"\r  {direction_label}_score_progress="
                    f"{done_pairs}/{valid_idx.size} ({pct:5.1f}%) "
                    f"groups={gi}/{total_groups} "
                    f"rate={rate:8.1f} dir-pairs/s"
                )
                sys.stderr.flush()
                last_report = done_pairs

        sys.stderr.write("\n")
        sys.stderr.flush()

    # ------------------------------------------------------------
    # Direction u -> v: response is v, predictor is u
    # ------------------------------------------------------------
    run_direction(
        response_arr=v,
        predictor_arr=u,
        p_out=p_u_to_v,
        beta_out=beta_u_to_v,
        fill_counts=True,
        direction_label="u_to_v",
    )

    # ------------------------------------------------------------
    # Direction v -> u: response is u, predictor is v
    # ------------------------------------------------------------
    run_direction(
        response_arr=u,
        predictor_arr=v,
        p_out=p_v_to_u,
        beta_out=beta_v_to_u,
        fill_counts=False,
        direction_label="v_to_u",
    )

    # ------------------------------------------------------------
    # Finish contingency counts
    # ------------------------------------------------------------
    count_u = locus_counts[u].astype(np.int64)
    count_v = locus_counts[v].astype(np.int64)

    n10 = count_u - n11
    n01 = count_v - n11
    n00 = n_samples - n11 - n10 - n01

    # ------------------------------------------------------------
    # Pair-level score summary
    # ------------------------------------------------------------
    finite_both = np.isfinite(p_u_to_v) & np.isfinite(p_v_to_u)

    p_pair_max = np.full(n_pairs, np.nan, dtype=np.float64)
    p_pair_max[finite_both] = np.maximum(
        p_u_to_v[finite_both],
        p_v_to_u[finite_both],
    )

    # ------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------
    valid_idx = np.where(valid_maf)[0]

    for i in valid_idx:
        if finite_both[i]:
            status[i] = _status_from_counts(
                int(n10[i]),
                int(n01[i]),
                int(min(n11[i], n10[i], n01[i], n00[i])),
                config.low_count_threshold,
            )
        elif status[i] == "LOW_MAF":
            status[i] = "SCORE_FAILED"

    # LOW_MAF rows stay LOW_MAF but still report counts/frequencies.

    return pd.DataFrame(
        {
            "u": u,
            "v": v,
            "distance": distance,
            "n11": n11,
            "n10": n10,
            "n01": n01,
            "n00": n00,
            "freq_u": locus_freq[u],
            "freq_v": locus_freq[v],
            "score_beta_u_to_v": beta_u_to_v,
            "score_p_u_to_v": p_u_to_v,
            "score_beta_v_to_u": beta_v_to_u,
            "score_p_v_to_u": p_v_to_u,
            "score_p_pair_max": p_pair_max,
            "score_status": status,
        }
    )


def select_top_pairs(score_df: pd.DataFrame, top_frac: float = 0.01, top_n: int | None = None) -> pd.DataFrame:
    finite = score_df[np.isfinite(score_df["score_p_pair_max"].to_numpy(dtype=np.float64))].copy()
    if finite.empty:
        return finite
    if top_n is None:
        if top_frac <= 0:
            raise ValueError("top_frac must be positive when top_n is not set")
        top_n = max(1, int(np.ceil(len(finite) * top_frac)))
    top_n = min(int(top_n), len(finite))
    return finite.nsmallest(top_n, "score_p_pair_max").copy()


def _score_columns() -> list[str]:
    return [
        "u", "v", "distance", "n11", "n10", "n01", "n00", "freq_u", "freq_v",
        "score_beta_u_to_v", "score_p_u_to_v", "score_beta_v_to_u", "score_p_v_to_u",
        "score_p_pair_max", "score_status",
    ]

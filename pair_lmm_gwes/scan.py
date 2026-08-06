from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .lmm_core import fit_null_lmm, test_predictor_block, neglog10, eigen_decompose_kinship, apply_eigen_rank_limit


@dataclass
class ScanConfig:
    min_count: int = 3
    threads: int = 1
    predictor_batch_size: int = 8192
    worker_chunk_size: int = 8
    h2_grid_size: int = 21
    h2_max: float = 0.9
    grm_rank: int = 20
    progress: bool = True
    progress_every_responses: int = 50
    write_diagnostics: bool = False
    pair_test: str = "symmetric"  # symmetric or bidirectional
    h2_boundary_warn: float = 0.99
    near_redundant_mismatch: float = 0.02
    exclude_near_redundant: bool = False
    # v0.5.1 tree-based independent 11-origin handling.
    # mode=off: no tree counting.
    # mode=prefilter: apply the tree 11-gain filter before LMM, but only after cheap filters pass.
    # mode=posthoc: run LMM first and compute tree 11-gain counts only for pairs over the posthoc score threshold.
    tree_11_mode: str = "posthoc"  # off, prefilter, posthoc
    tree_11_gain_min: int = 3
    tree_11_loss_cost: float = 1.0
    tree_11_min_count: int = 0
    tree_11_min_count_frac: float = 0.01
    tree_11_posthoc_score_threshold: float = 5.0
    tree_11_posthoc_distance_min: float = 0.0
    tree_event_counter: Any | None = None
    tree_event_source: str = "none"
    # v0.5.4 binary CTMC effective 11-gain count. Uses a reconstructed-edge
    # likelihood with fixed second-order Taylor approximation. Computed post-hoc
    # only, using the same score/distance thresholds as tree_11 posthoc.
    ctmc_11_mode: str = "posthoc"  # off, posthoc
    ctmc_11_gain_min: float = 1.5


_GLOBAL_X: np.ndarray | None = None
_GLOBAL_U: np.ndarray | None = None
_GLOBAL_S: np.ndarray | None = None
_GLOBAL_CONFIG: ScanConfig | None = None
_GLOBAL_RESPONSE_GRM_WORKSPACE = None


def _init_worker(X: np.ndarray, U: np.ndarray | None, S: np.ndarray | None, config: ScanConfig, response_grm_workspace=None) -> None:
    global _GLOBAL_X, _GLOBAL_U, _GLOBAL_S, _GLOBAL_CONFIG, _GLOBAL_RESPONSE_GRM_WORKSPACE
    _GLOBAL_X = X
    _GLOBAL_U = U
    _GLOBAL_S = S
    _GLOBAL_CONFIG = config
    _GLOBAL_RESPONSE_GRM_WORKSPACE = response_grm_workspace


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def pair_counts_fast(X: np.ndarray, u: int, v: int, col_sums: np.ndarray) -> tuple[int, int, int, int]:
    xu = X[:, u].astype(bool, copy=False)
    xv = X[:, v].astype(bool, copy=False)
    n11 = int(np.count_nonzero(xu & xv))
    su = int(col_sums[u])
    sv = int(col_sums[v])
    n10 = su - n11
    n01 = sv - n11
    n00 = X.shape[0] - n11 - n10 - n01
    return n11, n10, n01, n00


def _meta_cols(pairs: pd.DataFrame) -> list[str]:
    """Columns from the candidate pair file copied into pair_lmm_gwes.tsv.

    v0.6.3 preserves original PAN-GWES/SpydrPick metadata, not only distance.
    For a no-header PAN-GWES pair file the columns are normally:
      u, v, distance, ARACNE, MI, count, M2, min_distance, max_distance.
    Any additional columns are also retained.
    """
    return [c for c in pairs.columns if str(c) not in {"u", "v"}]


def _safe_r2_from_counts(n11: int, n10: int, n01: int, n00: int) -> float:
    n = n11 + n10 + n01 + n00
    a = n11 + n10
    b = n01 + n00
    c = n11 + n01
    d = n10 + n00
    denom = a * b * c * d
    if n <= 0 or denom <= 0:
        return float("nan")
    phi = (n11 * n00 - n10 * n01) / math.sqrt(denom)
    return float(phi * phi)


def _safe_phi_from_counts(n11: int, n10: int, n01: int, n00: int) -> float:
    a = n11 + n10
    b = n01 + n00
    c = n11 + n01
    d = n10 + n00
    denom = a * b * c * d
    if denom <= 0:
        return float("nan")
    return float((n11 * n00 - n10 * n01) / math.sqrt(denom))


def _or_05pc(n11: int, n10: int, n01: int, n00: int) -> float:
    return float(((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5)))


def _empty_ctmc_fields() -> dict[str, Any]:
    return {
        "ctmc_11_gain_expected": np.nan,
        "ctmc_11_loss_expected": np.nan,
        "ctmc_11_q01": np.nan,
        "ctmc_11_q10": np.nan,
        "ctmc_11_loglik": np.nan,
        "ctmc_11_status": "NOT_EVALUATED",
        "ctmc_11_mode": "off",
        "ctmc_11_posthoc_evaluated": 0,
        "ctmc_11_posthoc_status": "NOT_EVALUATED",
    }


def _empty_tree_fields() -> dict[str, Any]:
    return {
        "tree_11_gain_count": np.nan,
        "tree_11_gain_count_max": np.nan,
        "tree_11_loss_count": np.nan,
        "tree_11_parsimony_score": np.nan,
        "tree_11_largest_origin_n11": np.nan,
        "tree_11_largest_origin_fraction": np.nan,
        "tree_11_singleton_origin_count": np.nan,
        "tree_11_origin_sizes": "NA",
    }


def _summarize_tree_cached(
    tree_event_counter: Any,
    X: np.ndarray,
    u: int,
    v: int,
    tree_11_loss_cost: float,
    tree_event_cache: dict[bytes, Any],
) -> dict[str, Any]:
    xu = X[:, u].astype(bool, copy=False)
    xv = X[:, v].astype(bool, copy=False)
    z11 = xu & xv
    key = np.packbits(z11.astype(np.uint8, copy=False)).tobytes()
    summ = tree_event_cache.get(key)
    if summ is None:
        summ = tree_event_counter.summarize(z11, gain_cost=1.0, loss_cost=float(tree_11_loss_cost))
        if len(tree_event_cache) < 200000:
            tree_event_cache[key] = summ
    return {
        "tree_11_gain_count": summ.tree_11_gain_count,
        "tree_11_gain_count_max": summ.tree_11_gain_count_max,
        "tree_11_loss_count": summ.tree_11_loss_count,
        "tree_11_parsimony_score": summ.tree_11_parsimony_score,
        "tree_11_largest_origin_n11": summ.tree_11_largest_origin_n11,
        "tree_11_largest_origin_fraction": summ.tree_11_largest_origin_fraction,
        "tree_11_singleton_origin_count": summ.tree_11_singleton_origin_count,
        "tree_11_origin_sizes": summ.tree_11_origin_sizes,
    }




def _summarize_ctmc_cached(
    tree_event_counter: Any,
    X: np.ndarray,
    u: int,
    v: int,
    ctmc_cache: dict[bytes, Any],
) -> dict[str, Any]:
    xu = X[:, u].astype(bool, copy=False)
    xv = X[:, v].astype(bool, copy=False)
    z11 = xu & xv
    key = np.packbits(z11.astype(np.uint8, copy=False)).tobytes()
    summ = ctmc_cache.get(key)
    if summ is None:
        summ = tree_event_counter.summarize_ctmc_taylor2(z11)
        if len(ctmc_cache) < 200000:
            ctmc_cache[key] = summ
    return {
        "ctmc_11_gain_expected": summ.ctmc_11_gain_expected,
        "ctmc_11_loss_expected": summ.ctmc_11_loss_expected,
        "ctmc_11_q01": summ.ctmc_11_q01,
        "ctmc_11_q10": summ.ctmc_11_q10,
        "ctmc_11_loglik": summ.ctmc_11_loglik,
        "ctmc_11_status": summ.ctmc_11_status,
    }


def _compute_raw_counts(
    pairs: pd.DataFrame,
    X: np.ndarray,
    min_count: int,
    progress: bool,
    near_redundant_mismatch: float,
    exclude_near_redundant: bool,
    tree_event_counter: Any | None = None,
    tree_11_mode: str = "posthoc",
    tree_11_gain_min: int = 0,
    tree_11_loss_cost: float = 1.0,
    tree_11_min_count: int = 0,
    tree_11_min_count_frac: float = 0.0,
    tree_event_source: str = "none",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_pairs = len(pairs)
    n_samples = X.shape[0]
    col_sums = X.sum(axis=0).astype(np.int64)
    rows: list[dict[str, Any]] = []
    valid = np.zeros(n_pairs, dtype=bool)
    excluded_redundant = np.zeros(n_pairs, dtype=bool)
    low_tree_11_gain = np.zeros(n_pairs, dtype=bool)
    low_tree_11_count = np.zeros(n_pairs, dtype=bool)
    mode = str(tree_11_mode or "off").lower()
    use_tree_prefilter = tree_event_counter is not None and mode == "prefilter" and int(tree_11_gain_min) > 0
    tree_event_cache: dict[bytes, Any] = {}
    n11_count_min = max(int(tree_11_min_count), int(math.ceil(float(tree_11_min_count_frac) * n_samples)))
    meta_cols = _meta_cols(pairs)
    t0 = time.time()
    every = max(1, min(50000, int(math.ceil(n_pairs * 0.05))))
    n_tree_evaluated = 0
    n_tree_skipped_cheap = 0
    for idx, row in pairs.iterrows():
        u = int(row["u"])
        v = int(row["v"])
        n11, n10, n01, n00 = pair_counts_fast(X, u, v, col_sums)
        min_cell = min(n11, n10, n01, n00)
        mismatch_same = n10 + n01
        mismatch_complement = n11 + n00
        same_mismatch_rate = mismatch_same / max(1, n_samples)
        complement_mismatch_rate = mismatch_complement / max(1, n_samples)
        near_copy = same_mismatch_rate <= near_redundant_mismatch
        near_complement = complement_mismatch_rate <= near_redundant_mismatch
        excluded = exclude_near_redundant and (near_copy or near_complement)
        failed_count = min_cell < min_count
        failed_tree_count = bool(use_tree_prefilter and n11_count_min > 0 and n11 < n11_count_min)
        cheap_pass_for_tree = (not failed_count) and (not excluded) and (not failed_tree_count)
        tree_fields = _empty_tree_fields()
        if use_tree_prefilter:
            if cheap_pass_for_tree:
                tree_fields = _summarize_tree_cached(tree_event_counter, X, u, v, tree_11_loss_cost, tree_event_cache)
                n_tree_evaluated += 1
            else:
                n_tree_skipped_cheap += 1
        failed_tree_gain = bool(
            use_tree_prefilter
            and cheap_pass_for_tree
            and not (float(tree_fields["tree_11_gain_count"]) >= float(tree_11_gain_min))
        )
        out = {"u": u, "v": v}
        for c in meta_cols:
            out[c] = row[c]
        out.update({
            "n11": n11,
            "n10": n10,
            "n01": n01,
            "n00": n00,
            "min_cell": min_cell,
            "mismatch_count": mismatch_same,
            "same_mismatch_rate": same_mismatch_rate,
            "complement_mismatch_rate": complement_mismatch_rate,
            "r2": _safe_r2_from_counts(n11, n10, n01, n00),
            "phi": _safe_phi_from_counts(n11, n10, n01, n00),
            "or_0.5pc": _or_05pc(n11, n10, n01, n00),
            "near_copy": int(near_copy),
            "near_complement": int(near_complement),
            "near_redundant": int(near_copy or near_complement),
            **tree_fields,
            **_empty_ctmc_fields(),
            "tree_11_mode": mode,
            "tree_11_gain_min": int(tree_11_gain_min) if mode != "off" else 0,
            "tree_11_loss_cost": float(tree_11_loss_cost) if mode != "off" else 0.0,
            "tree_11_min_count": int(n11_count_min) if mode != "off" else 0,
            "tree_11_event_source": tree_event_source if tree_event_counter is not None and mode != "off" else "none",
            "tree_11_prefilter_evaluated": int(use_tree_prefilter and cheap_pass_for_tree),
            "tree_11_posthoc_evaluated": 0,
            "tree_11_posthoc_status": "NOT_EVALUATED",
        })
        rows.append(out)
        valid[idx] = (not failed_count) and not excluded and not failed_tree_count and not failed_tree_gain
        excluded_redundant[idx] = bool(excluded)
        low_tree_11_gain[idx] = bool(failed_tree_gain)
        low_tree_11_count[idx] = bool(failed_tree_count)
        done = idx + 1
        if progress and (done == 1 or done == n_pairs or done % every == 0):
            elapsed = time.time() - t0
            sys.stderr.write(
                f"[{_now()}] raw_counts_progress={done}/{n_pairs} ({100*done/n_pairs:5.1f}%) "
                f"elapsed={elapsed:.1f}s tree_prefilter_eval={n_tree_evaluated} tree_skipped_cheap={n_tree_skipped_cheap}\n"
            )
            sys.stderr.flush()
    return pd.DataFrame(rows), valid, excluded_redundant, low_tree_11_gain, low_tree_11_count


def _worker_response(task: tuple[int, list[tuple[int, int, str]]]) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    """Fit one response-locus null LMM and test requested predictors.

    task entries: (row_idx, predictor_locus, direction_name)
    In symmetric mode, direction_name is 'symmetric' and only one direction is tested.
    """
    response, entries = task
    assert _GLOBAL_X is not None and _GLOBAL_CONFIG is not None
    X = _GLOBAL_X
    config = _GLOBAL_CONFIG

    # v0.4.1 optional response-specific GRM. If a workspace is supplied, build
    # K_response by removing only direct proxies of this response locus, then
    # eigendecompose that K. Otherwise use the single global U/S from tree,
    # unmasked GRM, or v0.4.0-style global tested mask.
    kinship_info: dict[str, Any] = {
        "kinship_source": "global",
        "n_grm_loci_used": np.nan,
        "n_grm_loci_masked": np.nan,
        "n_response_proxy_masked_beyond_direct": np.nan,
    }
    if _GLOBAL_RESPONSE_GRM_WORKSPACE is not None:
        kr = _GLOBAL_RESPONSE_GRM_WORKSPACE.build_for_response(int(response))
        U, S = eigen_decompose_kinship(kr.K)
        U, S = apply_eigen_rank_limit(U, S, int(config.grm_rank))
        kinship_info = {
            "kinship_source": kr.source,
            "n_grm_loci_used": kr.n_loci_used,
            "n_grm_loci_masked": kr.details.get("n_loci_removed_by_response_mask", np.nan),
            "n_response_proxy_masked_beyond_direct": kr.details.get("n_response_proxy_masked_beyond_direct", np.nan),
        }
        del kr.K
    else:
        assert _GLOBAL_U is not None and _GLOBAL_S is not None
        U = _GLOBAL_U
        S = _GLOBAL_S

    y = X[:, response].astype(np.float64, copy=False)
    null, cache = fit_null_lmm(response, y, U, S, covariates=None, h2_grid_size=config.h2_grid_size, h2_max=config.h2_max)
    h2_boundary_cut = min(float(config.h2_boundary_warn), max(0.0, float(config.h2_max) - 1e-6))
    h2_boundary = bool(np.isfinite(null.h2) and null.h2 >= h2_boundary_cut)
    null_row = {
        "response": response,
        "status": null.status,
        "h2": null.h2,
        "h2_boundary": int(h2_boundary),
        "h2_max": float(config.h2_max),
        "grm_rank": int(config.grm_rank),
        "sigma2": null.sigma2,
        "n_present": null.n_present,
        "mac": null.mac,
        "null_nll": null.null_nll,
        **kinship_info,
    }
    results: list[dict[str, Any]] = []
    if null.status != "OK":
        for row_idx, predictor, direction in entries:
            results.append({
                "row_idx": row_idx,
                "direction": direction,
                "response": response,
                "predictor": predictor,
                "p": np.nan,
                "score": np.nan,
                "beta": np.nan,
                "se": np.nan,
                "F": np.nan,
                "xKx": np.nan,
                "xKy": np.nan,
                "h2": null.h2,
                "h2_boundary": int(h2_boundary),
                "h2_max": float(config.h2_max),
                "grm_rank": int(config.grm_rank),
                "status": f"NULL_{null.status}",
                **kinship_info,
            })
        return response, null_row, results

    batch = max(1, int(config.predictor_batch_size))
    for start in range(0, len(entries), batch):
        sub = entries[start:start+batch]
        predictors = [int(e[1]) for e in sub]
        Xb = X[:, predictors]
        br = test_predictor_block(Xb, U, cache)
        for j, (row_idx, predictor, direction) in enumerate(sub):
            p = float(br.p[j]) if np.isfinite(br.p[j]) else np.nan
            results.append({
                "row_idx": row_idx,
                "direction": direction,
                "response": response,
                "predictor": predictor,
                "p": p,
                "score": neglog10(p),
                "beta": float(br.beta[j]) if np.isfinite(br.beta[j]) else np.nan,
                "se": float(br.se[j]) if np.isfinite(br.se[j]) else np.nan,
                "F": float(br.F[j]) if br.F is not None and np.isfinite(br.F[j]) else np.nan,
                "xKx": float(br.xKx[j]) if br.xKx is not None and np.isfinite(br.xKx[j]) else np.nan,
                "xKy": float(br.xKy[j]) if br.xKy is not None and np.isfinite(br.xKy[j]) else np.nan,
                "h2": null.h2,
                "h2_boundary": int(h2_boundary),
                "h2_max": float(config.h2_max),
                "grm_rank": int(config.grm_rank),
                "status": br.status[j],
                **kinship_info,
            })
    return response, null_row, results


def _tasks_symmetric(pairs: pd.DataFrame, valid_mask: np.ndarray) -> dict[int, list[tuple[int, int, str]]]:
    """One LMM test per unordered pair. Use v as response and u as predictor.

    For a linear mixed model with common K and intercept-only fixed effects, pair
    association is essentially a weighted residual correlation. The statistic is
    theoretically symmetric when h2 is identical or near-identical, so one tested
    orientation is sufficient for the main score and halves runtime.
    """
    tasks: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for row_idx, row in pairs.loc[valid_mask].iterrows():
        u = int(row["u"])
        v = int(row["v"])
        tasks[v].append((int(row_idx), u, "symmetric"))
    return dict(tasks)


def _tasks_bidirectional(pairs: pd.DataFrame, valid_mask: np.ndarray) -> dict[int, list[tuple[int, int, str]]]:
    tasks: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for row_idx, row in pairs.loc[valid_mask].iterrows():
        u = int(row["u"])
        v = int(row["v"])
        tasks[v].append((int(row_idx), u, "u_to_v"))
        tasks[u].append((int(row_idx), v, "v_to_u"))
    return dict(tasks)



def _apply_posthoc_tree_counts(result: pd.DataFrame, X: np.ndarray, config: ScanConfig) -> pd.DataFrame:
    """Compute tree 11-gain summaries after LMM only for over-threshold pairs.

    This is v0.5.1's fast/default mode: tree counting is diagnostic/post-hoc
    and does not decide the main LMM status. It is only evaluated for rows that
    passed LMM (status=OK), have score above --tree-11-posthoc-score-threshold,
    and optionally pass --tree-11-posthoc-distance-min.
    """
    mode = str(config.tree_11_mode or "off").lower()
    if mode != "posthoc" or config.tree_event_counter is None:
        return result
    if "score_lmm" not in result.columns:
        return result
    score = pd.to_numeric(result["score_lmm"], errors="coerce")
    mask = score >= float(config.tree_11_posthoc_score_threshold)
    if "status" in result.columns:
        mask &= result["status"].astype(str).eq("OK")
    if float(config.tree_11_posthoc_distance_min) > 0 and "distance" in result.columns:
        mask &= pd.to_numeric(result["distance"], errors="coerce") >= float(config.tree_11_posthoc_distance_min)
    idxs = np.flatnonzero(mask.to_numpy())
    if len(idxs) == 0:
        return result

    n_samples = X.shape[0]
    n11_count_min = max(int(config.tree_11_min_count), int(math.ceil(float(config.tree_11_min_count_frac) * n_samples)))
    cache: dict[bytes, Any] = {}
    t0 = time.time()
    every = max(1, min(10000, int(math.ceil(len(idxs) * 0.10))))
    for done, idx in enumerate(idxs, start=1):
        u = int(result.at[idx, "u"])
        v = int(result.at[idx, "v"])
        n11 = int(result.at[idx, "n11"]) if "n11" in result.columns and pd.notna(result.at[idx, "n11"]) else 0
        result.at[idx, "tree_11_event_source"] = config.tree_event_source
        result.at[idx, "tree_11_mode"] = mode
        result.at[idx, "tree_11_gain_min"] = int(config.tree_11_gain_min)
        result.at[idx, "tree_11_loss_cost"] = float(config.tree_11_loss_cost)
        result.at[idx, "tree_11_min_count"] = int(n11_count_min)
        if n11_count_min > 0 and n11 < n11_count_min:
            result.at[idx, "tree_11_posthoc_status"] = "LOW_TREE_11_COUNT"
        else:
            fields = _summarize_tree_cached(config.tree_event_counter, X, u, v, config.tree_11_loss_cost, cache)
            for k, val in fields.items():
                result.at[idx, k] = val
            gain = float(fields["tree_11_gain_count"])
            if gain >= float(config.tree_11_gain_min):
                result.at[idx, "tree_11_posthoc_status"] = "PASS"
            else:
                result.at[idx, "tree_11_posthoc_status"] = "LOW_TREE_11_GAIN"
        result.at[idx, "tree_11_posthoc_evaluated"] = 1
        if config.progress and (done == 1 or done == len(idxs) or done % every == 0):
            elapsed = time.time() - t0
            sys.stderr.write(
                f"[{_now()}] tree_posthoc_progress={done}/{len(idxs)} "
                f"({100*done/max(1,len(idxs)):5.1f}%) elapsed={elapsed:.1f}s\n"
            )
            sys.stderr.flush()
    return result




def _apply_posthoc_ctmc_counts(result: pd.DataFrame, X: np.ndarray, config: ScanConfig) -> pd.DataFrame:
    """Compute binary CTMC effective 11-gain counts for over-threshold LMM hits.

    v0.5.4 keeps this deliberately minimal: CTMC is post-hoc only, uses the same
    score/distance thresholds as tree_11 posthoc, and uses a reconstructed-edge
    likelihood with fixed second-order Taylor approximation inside Tree11EventCounter.
    """
    mode = str(config.ctmc_11_mode or "off").lower()
    if mode != "posthoc" or config.tree_event_counter is None:
        return result
    if "score_lmm" not in result.columns:
        return result
    score = pd.to_numeric(result["score_lmm"], errors="coerce")
    mask = score >= float(config.tree_11_posthoc_score_threshold)
    if "status" in result.columns:
        mask &= result["status"].astype(str).eq("OK")
    if float(config.tree_11_posthoc_distance_min) > 0 and "distance" in result.columns:
        mask &= pd.to_numeric(result["distance"], errors="coerce") >= float(config.tree_11_posthoc_distance_min)
    idxs = np.flatnonzero(mask.to_numpy())
    if len(idxs) == 0:
        return result

    cache: dict[bytes, Any] = {}
    t0 = time.time()
    every = max(1, min(1000, int(math.ceil(len(idxs) * 0.10))))
    for done, idx in enumerate(idxs, start=1):
        u = int(result.at[idx, "u"])
        v = int(result.at[idx, "v"])
        fields = _summarize_ctmc_cached(config.tree_event_counter, X, u, v, cache)
        for k, val in fields.items():
            result.at[idx, k] = val
        result.at[idx, "ctmc_11_mode"] = mode
        result.at[idx, "ctmc_11_posthoc_evaluated"] = 1
        gain = float(fields.get("ctmc_11_gain_expected", np.nan))
        status = str(fields.get("ctmc_11_status", "NA"))
        if status != "OK":
            result.at[idx, "ctmc_11_posthoc_status"] = status
        elif np.isfinite(gain) and gain >= float(config.ctmc_11_gain_min):
            result.at[idx, "ctmc_11_posthoc_status"] = "PASS"
        else:
            result.at[idx, "ctmc_11_posthoc_status"] = "LOW_CTMC_11_GAIN"
        if config.progress and (done == 1 or done == len(idxs) or done % every == 0):
            elapsed = time.time() - t0
            sys.stderr.write(
                f"[{_now()}] ctmc11_posthoc_progress={done}/{len(idxs)} "
                f"({100*done/max(1,len(idxs)):5.1f}%) elapsed={elapsed:.1f}s\n"
            )
            sys.stderr.flush()
    return result


def scan_pairs_lmm(pairs: pd.DataFrame, X: np.ndarray, U: np.ndarray | None, S: np.ndarray | None, config: ScanConfig, response_grm_workspace=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run pyseer-like LMM tests for candidate pairs."""
    t0 = time.time()
    n_pairs = len(pairs)
    if config.pair_test not in {"symmetric", "bidirectional"}:
        raise ValueError("pair_test must be 'symmetric' or 'bidirectional'")
    raw_df, valid_mask, excluded_redundant, low_tree_11_gain, low_tree_11_count = _compute_raw_counts(
        pairs,
        X,
        config.min_count,
        config.progress,
        config.near_redundant_mismatch,
        config.exclude_near_redundant,
        tree_event_counter=config.tree_event_counter,
        tree_11_mode=config.tree_11_mode,
        tree_11_gain_min=config.tree_11_gain_min,
        tree_11_loss_cost=config.tree_11_loss_cost,
        tree_11_min_count=config.tree_11_min_count,
        tree_11_min_count_frac=config.tree_11_min_count_frac,
        tree_event_source=config.tree_event_source,
    )
    if config.pair_test == "bidirectional":
        tasks_by_response = _tasks_bidirectional(pairs, valid_mask)
    else:
        tasks_by_response = _tasks_symmetric(pairs, valid_mask)
    tasks = sorted(tasks_by_response.items(), key=lambda kv: kv[0])
    if config.progress:
        sys.stderr.write(
            f"[{_now()}] lmm_scan_start n_pairs={n_pairs} valid_pairs={int(valid_mask.sum())} "
            f"pair_test={config.pair_test} n_responses={len(tasks)} threads={config.threads}\n"
        )
        sys.stderr.flush()

    null_rows: list[dict[str, Any]] = []
    dir_rows: list[dict[str, Any]] = []
    threads = max(1, int(config.threads))
    done = 0
    if threads == 1:
        _init_worker(X, U, S, config, response_grm_workspace)
        for task in tasks:
            _, null_row, res = _worker_response(task)
            null_rows.append(null_row)
            dir_rows.extend(res)
            done += 1
            if config.progress and (done == 1 or done == len(tasks) or done % max(1, config.progress_every_responses) == 0):
                elapsed = time.time() - t0
                sys.stderr.write(f"[{_now()}] lmm_response_progress={done}/{len(tasks)} ({100*done/max(1,len(tasks)):5.1f}%) elapsed={elapsed:.1f}s\n")
                sys.stderr.flush()
    else:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ctx.Pool(processes=threads, initializer=_init_worker, initargs=(X, U, S, config, response_grm_workspace)) as pool:
            for _, null_row, res in pool.imap_unordered(_worker_response, tasks, chunksize=max(1, int(config.worker_chunk_size))):
                null_rows.append(null_row)
                dir_rows.extend(res)
                done += 1
                if config.progress and (done == 1 or done == len(tasks) or done % max(1, config.progress_every_responses) == 0):
                    elapsed = time.time() - t0
                    sys.stderr.write(f"[{_now()}] lmm_response_progress={done}/{len(tasks)} ({100*done/max(1,len(tasks)):5.1f}%) elapsed={elapsed:.1f}s\n")
                    sys.stderr.flush()

    diag_df = pd.DataFrame(dir_rows)
    result = raw_df.copy()
    result["pair_test"] = config.pair_test

    # Main v0.3.1 symmetric outputs.
    p_lmm = np.full(n_pairs, np.nan, dtype=np.float64)
    score_lmm = np.full(n_pairs, np.nan, dtype=np.float64)
    beta_lmm = np.full(n_pairs, np.nan, dtype=np.float64)
    se_lmm = np.full(n_pairs, np.nan, dtype=np.float64)
    h2_lmm = np.full(n_pairs, np.nan, dtype=np.float64)
    h2_boundary = np.zeros(n_pairs, dtype=np.int8)
    lmm_status = np.array(["NOT_TESTED"] * n_pairs, dtype=object)
    tested_direction = np.array(["NA"] * n_pairs, dtype=object)

    # Backward-compatible directional columns.
    p_u_to_v = np.full(n_pairs, np.nan, dtype=np.float64)
    p_v_to_u = np.full(n_pairs, np.nan, dtype=np.float64)
    status_u = np.array(["NOT_TESTED"] * n_pairs, dtype=object)
    status_v = np.array(["NOT_TESTED"] * n_pairs, dtype=object)
    if not diag_df.empty:
        for r in diag_df.itertuples(index=False):
            idx = int(r.row_idx)
            if r.direction == "u_to_v":
                p_u_to_v[idx] = r.p
                status_u[idx] = r.status
            elif r.direction == "v_to_u":
                p_v_to_u[idx] = r.p
                status_v[idx] = r.status
            else:
                p_lmm[idx] = r.p
                score_lmm[idx] = r.score
                beta_lmm[idx] = r.beta
                se_lmm[idx] = r.se
                h2_lmm[idx] = r.h2
                h2_boundary[idx] = int(r.h2_boundary)
                lmm_status[idx] = r.status
                tested_direction[idx] = "v~u"
                # Compatibility: in symmetric mode, expose the single score in the
                # historical directional fields, but mark pair_test=symmetric.
                p_u_to_v[idx] = r.p
                p_v_to_u[idx] = r.p
                status_u[idx] = r.status
                status_v[idx] = r.status

    if config.pair_test == "bidirectional" and not diag_df.empty:
        for i in range(n_pairs):
            if not valid_mask[i]:
                continue
            s1 = neglog10(p_u_to_v[i])
            s2 = neglog10(p_v_to_u[i])
            if np.isfinite(s1) and np.isfinite(s2):
                score_lmm[i] = min(s1, s2)
                p_lmm[i] = max(p_u_to_v[i], p_v_to_u[i])  # same as weaker/min-score direction
            if status_u[i] == "OK" and status_v[i] == "OK":
                lmm_status[i] = "OK"
            else:
                lmm_status[i] = f"{status_u[i]}|{status_v[i]}"
            tested_direction[i] = "bidirectional_min"
        # Pull h2 from one of the directions if available; boundary if either direction boundary.
        if not diag_df.empty:
            for r in diag_df.itertuples(index=False):
                idx = int(r.row_idx)
                if not np.isfinite(h2_lmm[idx]):
                    h2_lmm[idx] = r.h2
                h2_boundary[idx] = max(int(h2_boundary[idx]), int(r.h2_boundary))

    final_status = []
    bidirectional_score = []
    direction_delta = []
    for i in range(n_pairs):
        if excluded_redundant[i]:
            final_status.append("NEAR_REDUNDANT")
            bidirectional_score.append(np.nan)
            direction_delta.append(np.nan)
            continue
        if low_tree_11_count[i]:
            final_status.append("LOW_TREE_11_COUNT")
            bidirectional_score.append(np.nan)
            direction_delta.append(np.nan)
            continue
        if low_tree_11_gain[i]:
            final_status.append("LOW_TREE_11_GAIN")
            bidirectional_score.append(np.nan)
            direction_delta.append(np.nan)
            continue
        if not valid_mask[i]:
            final_status.append("LOW_COUNT")
            bidirectional_score.append(np.nan)
            direction_delta.append(np.nan)
            continue
        if config.pair_test == "bidirectional":
            s1 = neglog10(p_u_to_v[i])
            s2 = neglog10(p_v_to_u[i])
            bidirectional_score.append(min(s1, s2) if np.isfinite(s1) and np.isfinite(s2) else np.nan)
            direction_delta.append(abs(s1 - s2) if np.isfinite(s1) and np.isfinite(s2) else np.nan)
            final_status.append("OK" if status_u[i] == "OK" and status_v[i] == "OK" else f"{status_u[i]}|{status_v[i]}")
        else:
            bidirectional_score.append(score_lmm[i])
            direction_delta.append(0.0 if np.isfinite(score_lmm[i]) else np.nan)
            final_status.append("OK" if lmm_status[i] == "OK" else str(lmm_status[i]))

    result["p_lmm"] = p_lmm
    result["score_lmm"] = score_lmm
    result["beta_lmm"] = beta_lmm
    result["se_lmm"] = se_lmm
    result["h2_lmm"] = h2_lmm
    result["h2_boundary"] = h2_boundary
    result["h2_boundary_warn"] = float(config.h2_boundary_warn)
    result["h2_max"] = float(config.h2_max)
    result["grm_rank"] = int(config.grm_rank)
    if not diag_df.empty and "n_grm_loci_masked" in diag_df.columns:
        response_masked = np.full(n_pairs, np.nan, dtype=np.float64)
        response_grm_used = np.full(n_pairs, np.nan, dtype=np.float64)
        response_proxy_extra = np.full(n_pairs, np.nan, dtype=np.float64)
        for r in diag_df.itertuples(index=False):
            idx = int(r.row_idx)
            if hasattr(r, "n_grm_loci_masked"):
                response_masked[idx] = r.n_grm_loci_masked
            if hasattr(r, "n_grm_loci_used"):
                response_grm_used[idx] = r.n_grm_loci_used
            if hasattr(r, "n_response_proxy_masked_beyond_direct"):
                response_proxy_extra[idx] = r.n_response_proxy_masked_beyond_direct
        result["response_grm_loci_masked"] = response_masked
        result["response_grm_loci_used"] = response_grm_used
        result["response_proxy_masked_beyond_direct"] = response_proxy_extra
    result["tested_direction"] = tested_direction
    result["lmm_status"] = lmm_status
    # Backward compatibility with old plotting scripts.
    result["p_u_to_v"] = p_u_to_v
    result["p_v_to_u"] = p_v_to_u
    result["direction_delta_score"] = direction_delta
    result["bidirectional_score"] = bidirectional_score
    result["status"] = final_status

    # v0.5.1/v0.5.4 default: tree/CTMC 11-origin diagnostics can be applied post-hoc only to over-threshold LMM hits.
    result = _apply_posthoc_tree_counts(result, X, config)
    result = _apply_posthoc_ctmc_counts(result, X, config)

    null_df = pd.DataFrame(null_rows).sort_values("response") if null_rows else pd.DataFrame()
    if config.progress:
        elapsed = time.time() - t0
        sys.stderr.write(f"[{_now()}] lmm_scan_complete elapsed={elapsed:.1f}s\n")
        sys.stderr.flush()
    return result, null_df, diag_df

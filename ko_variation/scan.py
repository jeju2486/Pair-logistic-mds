from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import multiprocessing as mp
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from .glmm import (
    fit_full_glmm,
    fit_null_glmm,
    neglog10,
    prepare_kinship_eigensystem,
    score_predictor_block,
)
from .spa import spa_pvalue


@dataclass
class ScanConfig:
    min_maf: float = 0.05
    min_cell_count: int = 5
    direction_mode: str = "both"  # both or input
    spa_mode: str = "off"  # off, auto or always
    full_refit_p: float = 0.05
    threads: int = 1
    worker_chunk_size: int = 1
    predictor_batch_size: int = 256
    null_max_iter: int = 100
    null_tolerance: float = 1e-7
    progress: bool = True
    progress_every_responses: int = 25
    near_redundant_mismatch: float = 0.02
    exclude_near_redundant: bool = False


_GLOBAL_X: np.ndarray | None = None
_GLOBAL_K: np.ndarray | None = None
_GLOBAL_CONFIG: ScanConfig | None = None
_GLOBAL_KINSHIP_EIGENSYSTEM: Any | None = None


@dataclass(frozen=True)
class _ResponseEntry:
    output_index: int
    predictor_locus: int
    response_locus: int
    response_pattern_flipped: bool
    row_context: dict[str, Any]


@dataclass(frozen=True)
class _ResponsePatternTask:
    pattern_id: int
    canonical_response_locus: int
    response_members: tuple[tuple[int, bool], ...]
    entries: tuple[_ResponseEntry, ...]

_RESERVED_METADATA_COLUMNS = {
    "pair_id", "pair_u", "pair_v", "predictor_locus", "response_locus",
    "direction", "direction_order", "predictor_prevalence",
    "response_prevalence", "predictor_maf", "response_maf", "n11", "n10",
    "n01", "n00", "min_cell", "phi", "r2", "odds_ratio_0.5pc",
    "same_mismatch_rate", "complement_mismatch_rate", "near_copy",
    "near_complement", "near_redundant", "status", "eligible",
    "tau_phylogenetic", "latent_phylogenetic_fraction", "score_u",
    "score_variance", "score_z", "p_score", "beta_score", "se_score",
    "spa_applied", "spa_status", "spa_variance_ratio", "p_spa", "p_primary",
    "primary_method", "score_primary", "full_refit_attempted",
    "beta_log_odds", "se_log_odds", "odds_ratio", "odds_ratio_ci_low",
    "odds_ratio_ci_high", "tau_alt", "p_wald", "full_refit_status", "q_bh",
    "bonferroni_significant", "n_directional_tests", "kinship_source",
    "kinship_rank", "kinship_eigen_min", "kinship_eigen_max",
    "kinship_roundoff_correction",
    "canonical_response_locus", "response_pattern_id",
    "response_pattern_flipped", "response_pattern_size", "null_fit_reused",
}


def _init_worker(
    X: np.ndarray,
    K: np.ndarray,
    config: ScanConfig,
    kinship_eigensystem: Any | None = None,
) -> None:
    global _GLOBAL_X, _GLOBAL_K, _GLOBAL_CONFIG, _GLOBAL_KINSHIP_EIGENSYSTEM
    _GLOBAL_X = X
    _GLOBAL_K = K
    _GLOBAL_CONFIG = config
    _GLOBAL_KINSHIP_EIGENSYSTEM = kinship_eigensystem


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _canonical_response_pattern(response: np.ndarray) -> tuple[bytes, bool]:
    """Return an exact packed response/complement key and its orientation.

    The packed bytes themselves are the dictionary key, rather than a digest.
    Python dictionaries resolve hash collisions with byte-for-byte equality, so
    distinct response patterns cannot be merged by a hash collision.
    """
    values = np.asarray(response).reshape(-1)
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("Response patterns must contain only binary 0/1 values")
    bits = values.astype(np.uint8, copy=False)
    packed = np.packbits(bits, bitorder="little").tobytes()
    complemented = np.packbits(1 - bits, bitorder="little").tobytes()
    if complemented < packed:
        return complemented, True
    return packed, False


def pair_counts_fast(
    X: np.ndarray,
    predictor: int,
    response: int,
    column_sums: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return n11,n10,n01,n00, oriented predictor first and response second."""
    xp = X[:, predictor].astype(bool, copy=False)
    yr = X[:, response].astype(bool, copy=False)
    n11 = int(np.count_nonzero(xp & yr))
    n10 = int(column_sums[predictor]) - n11
    n01 = int(column_sums[response]) - n11
    n00 = X.shape[0] - n11 - n10 - n01
    return n11, n10, n01, n00


def _safe_phi(n11: int, n10: int, n01: int, n00: int) -> float:
    denominator = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if denominator <= 0:
        return np.nan
    return float((n11 * n00 - n10 * n01) / math.sqrt(denominator))


def _or_half_correction(n11: int, n10: int, n01: int, n00: int) -> float:
    return float(((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5)))


def _validate_unique_pairs(pairs: pd.DataFrame, direction_mode: str) -> None:
    if direction_mode == "both":
        keys = [tuple(sorted((int(u), int(v)))) for u, v in pairs[["u", "v"]].itertuples(index=False, name=None)]
    else:
        keys = [(int(u), int(v)) for u, v in pairs[["u", "v"]].itertuples(index=False, name=None)]
    if len(keys) != len(set(keys)):
        kind = "unordered" if direction_mode == "both" else "directed"
        raise ValueError(f"Pair file contains duplicate {kind} pairs; deduplicate before scanning")


def _directional_rows(pairs: pd.DataFrame, X: np.ndarray, config: ScanConfig) -> pd.DataFrame:
    if config.direction_mode not in {"both", "input"}:
        raise ValueError("direction_mode must be 'both' or 'input'")
    if config.spa_mode not in {"off", "auto", "always"}:
        raise ValueError("spa_mode must be 'off', 'auto' or 'always'")
    if not (0.0 <= float(config.min_maf) <= 0.5):
        raise ValueError("min_maf must be between 0 and 0.5")
    if int(config.min_cell_count) < 0:
        raise ValueError("min_cell_count must be non-negative")
    _validate_unique_pairs(pairs, config.direction_mode)

    n = X.shape[0]
    sums = X.sum(axis=0).astype(np.int64)
    prevalence = sums.astype(np.float64) / float(n)
    maf = np.minimum(prevalence, 1.0 - prevalence)
    metadata_columns = [column for column in pairs.columns if str(column) not in {"u", "v"}]
    collisions = sorted({str(column) for column in metadata_columns} & _RESERVED_METADATA_COLUMNS)
    if collisions:
        raise ValueError(
            "Pair metadata uses reserved KOVAR output columns: "
            + ", ".join(collisions)
            + ". Rename these input columns before scanning."
        )
    rows: list[dict[str, Any]] = []

    for pair_id, pair in pairs.iterrows():
        u = int(pair["u"])
        v = int(pair["v"])
        base_counts = pair_counts_fast(X, u, v, sums)
        n11, n10_uv, n01_uv, n00 = base_counts
        same_mismatch = n10_uv + n01_uv
        complement_mismatch = n11 + n00
        same_rate = same_mismatch / float(n)
        complement_rate = complement_mismatch / float(n)
        near_copy = same_rate <= float(config.near_redundant_mismatch)
        near_complement = complement_rate <= float(config.near_redundant_mismatch)
        directions = [(u, v, "u_predicts_v", 0)]
        if config.direction_mode == "both":
            directions.append((v, u, "v_predicts_u", 1))

        for predictor, response, direction, direction_order in directions:
            if predictor == u:
                counts = (n11, n10_uv, n01_uv, n00)
            else:
                counts = (n11, n01_uv, n10_uv, n00)
            d11, d10, d01, d00 = counts
            row: dict[str, Any] = {
                "pair_id": int(pair_id),
                "u": u,
                "v": v,
                "pair_u": u,
                "pair_v": v,
                "predictor_locus": int(predictor),
                "response_locus": int(response),
                "direction": direction,
                "direction_order": direction_order,
            }
            for column in metadata_columns:
                row[column] = pair[column]
            row.update({
                "predictor_prevalence": float(prevalence[predictor]),
                "response_prevalence": float(prevalence[response]),
                "predictor_maf": float(maf[predictor]),
                "response_maf": float(maf[response]),
                "n11": d11,
                "n10": d10,
                "n01": d01,
                "n00": d00,
                "min_cell": int(min(counts)),
                "phi": _safe_phi(d11, d10, d01, d00),
                "r2": _safe_phi(d11, d10, d01, d00) ** 2,
                "odds_ratio_0.5pc": _or_half_correction(d11, d10, d01, d00),
                "same_mismatch_rate": same_rate,
                "complement_mismatch_rate": complement_rate,
                "near_copy": int(near_copy),
                "near_complement": int(near_complement),
                "near_redundant": int(near_copy or near_complement),
            })
            if maf[predictor] + 1e-12 < float(config.min_maf):
                status = "LOW_PREDICTOR_MAF"
            elif maf[response] + 1e-12 < float(config.min_maf):
                status = "LOW_RESPONSE_MAF"
            elif min(counts) < int(config.min_cell_count):
                status = "LOW_CELL_COUNT"
            elif config.exclude_near_redundant and (near_copy or near_complement):
                status = "NEAR_REDUNDANT"
            else:
                status = "ELIGIBLE"
            row["status"] = status
            row["eligible"] = int(status == "ELIGIBLE")
            rows.append(row)
    return pd.DataFrame(rows)


def _response_pattern_tasks(
    result: pd.DataFrame,
    X: np.ndarray,
) -> tuple[list[_ResponsePatternTask], int]:
    """Group eligible response loci by exact pattern, including complements."""
    eligible_indices = result.index[result["eligible"] == 1].to_numpy(dtype=np.int64)
    response_loci = sorted(
        {int(value) for value in result.loc[eligible_indices, "response_locus"].tolist()}
    )
    pattern_for_response: dict[int, tuple[bytes, bool]] = {}
    members_by_pattern: dict[bytes, list[tuple[int, bool]]] = defaultdict(list)
    for response_locus in response_loci:
        pattern, flipped = _canonical_response_pattern(X[:, response_locus])
        pattern_for_response[response_locus] = (pattern, flipped)
        members_by_pattern[pattern].append((response_locus, flipped))

    entries_by_pattern: dict[bytes, list[_ResponseEntry]] = defaultdict(list)
    for index in eligible_indices:
        row = result.loc[index]
        response_locus = int(row["response_locus"])
        pattern, flipped = pattern_for_response[response_locus]
        entries_by_pattern[pattern].append(
            _ResponseEntry(
                output_index=int(index),
                predictor_locus=int(row["predictor_locus"]),
                response_locus=response_locus,
                response_pattern_flipped=flipped,
                row_context={
                    "response_maf": row["response_maf"],
                    "predictor_maf": row["predictor_maf"],
                    "min_cell": row["min_cell"],
                },
            )
        )

    ordered_patterns = sorted(
        members_by_pattern,
        key=lambda pattern: members_by_pattern[pattern][0][0],
    )
    tasks: list[_ResponsePatternTask] = []
    for pattern_id, pattern in enumerate(ordered_patterns):
        canonical_members = tuple(members_by_pattern[pattern])
        representative_locus, representative_canonical_flip = canonical_members[0]
        members = tuple(
            (response_locus, canonical_flip != representative_canonical_flip)
            for response_locus, canonical_flip in canonical_members
        )
        entries = tuple(
            _ResponseEntry(
                output_index=entry.output_index,
                predictor_locus=entry.predictor_locus,
                response_locus=entry.response_locus,
                response_pattern_flipped=(
                    entry.response_pattern_flipped != representative_canonical_flip
                ),
                row_context=entry.row_context,
            )
            for entry in entries_by_pattern[pattern]
        )
        tasks.append(
            _ResponsePatternTask(
                pattern_id=pattern_id,
                canonical_response_locus=representative_locus,
                response_members=members,
                entries=entries,
            )
        )
    return tasks, len(response_loci)


def _should_apply_spa(row: dict[str, Any], config: ScanConfig) -> bool:
    if config.spa_mode == "always":
        return True
    if config.spa_mode == "off":
        return False
    return bool(
        float(row["p_score"]) <= 0.05
        and (
            float(row["response_maf"]) <= 0.10
            or float(row["predictor_maf"]) <= 0.10
            or int(row["min_cell"]) < 10
        )
    )


def _empty_inference() -> dict[str, Any]:
    return {
        "tau_phylogenetic": np.nan,
        "latent_phylogenetic_fraction": np.nan,
        "score_u": np.nan,
        "score_variance": np.nan,
        "score_z": np.nan,
        "p_score": np.nan,
        "beta_score": np.nan,
        "se_score": np.nan,
        "spa_applied": 0,
        "spa_status": "NOT_APPLIED",
        "spa_variance_ratio": np.nan,
        "p_spa": np.nan,
        "p_primary": np.nan,
        "primary_method": "NONE",
        "score_primary": np.nan,
        "full_refit_attempted": 0,
        "beta_log_odds": np.nan,
        "se_log_odds": np.nan,
        "odds_ratio": np.nan,
        "odds_ratio_ci_low": np.nan,
        "odds_ratio_ci_high": np.nan,
        "tau_alt": np.nan,
        "p_wald": np.nan,
        "full_refit_status": "NOT_ATTEMPTED",
    }


def _worker_response(
    task: _ResponsePatternTask,
) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, Any]]]]:
    assert _GLOBAL_X is not None and _GLOBAL_K is not None and _GLOBAL_CONFIG is not None
    X = _GLOBAL_X
    K = _GLOBAL_K
    config = _GLOBAL_CONFIG
    representative = int(task.canonical_response_locus)
    y = X[:, representative].astype(np.float64, copy=False)
    fit, cache = fit_null_glmm(
        representative,
        y,
        K,
        max_iter=config.null_max_iter,
        tolerance=config.null_tolerance,
        kinship_eigensystem=_GLOBAL_KINSHIP_EIGENSYSTEM,
    )
    null_rows: list[dict[str, Any]] = []
    for response_locus, flipped in task.response_members:
        original_y = X[:, response_locus]
        n_present = int(np.sum(original_y))
        null_rows.append({
            "response_locus": int(response_locus),
            "response_prevalence": float(np.mean(original_y)),
            "response_maf": float(min(np.mean(original_y), 1.0 - np.mean(original_y))),
            "n_present": n_present,
            "minor_count": int(min(n_present, y.size - n_present)),
            "tau_phylogenetic": fit.tau,
            "latent_phylogenetic_fraction": fit.latent_phylogenetic_fraction,
            "pql_iterations": fit.iterations,
            "pql_converged": int(fit.converged),
            "pql_objective": fit.objective,
            "max_eta_change": fit.max_eta_change,
            "min_working_weight": fit.min_weight,
            "n_weights_clipped": fit.n_weights_clipped,
            "fixed_effect_rank": int(fit.beta.size),
            "tau_boundary": int(fit.tau_boundary),
            "status": fit.status,
            "message": fit.message,
            "canonical_response_locus": representative,
            "response_pattern_id": int(task.pattern_id),
            "response_pattern_flipped": int(flipped),
            "response_pattern_size": len(task.response_members),
            "null_fit_reused": int(response_locus != representative),
        })
    updates: list[tuple[int, dict[str, Any]]] = []
    scores_by_predictor: dict[int, Any] = {}
    if fit.status == "OK":
        batch_size = max(1, int(config.predictor_batch_size))
        predictor_loci = list(dict.fromkeys(entry.predictor_locus for entry in task.entries))
        for start in range(0, len(predictor_loci), batch_size):
            batch = predictor_loci[start:start + batch_size]
            predictor_block = X[:, batch]
            block_scores = score_predictor_block(
                predictor_block,
                cache,
                compute_spa_adjustment=config.spa_mode != "off",
            )
            for predictor, score in zip(batch, block_scores):
                scores_by_predictor[int(predictor)] = score
    for entry in task.entries:
        output_index = int(entry.output_index)
        predictor = int(entry.predictor_locus)
        flipped = bool(entry.response_pattern_flipped)
        score_sign = -1.0 if flipped else 1.0
        out = _empty_inference()
        out["tau_phylogenetic"] = fit.tau
        out["latent_phylogenetic_fraction"] = fit.latent_phylogenetic_fraction
        if fit.status != "OK":
            out["status"] = f"NULL_{fit.status}"
            updates.append((output_index, out))
            continue
        score = scores_by_predictor[predictor]
        out.update({
            "score_u": score_sign * score.score_u,
            "score_variance": score.score_variance,
            "score_z": score_sign * score.score_z,
            "p_score": score.p_score,
            "beta_score": score_sign * score.beta_score,
            "se_score": score.se_score,
        })
        if score.status != "OK":
            out["status"] = score.status
            updates.append((output_index, out))
            continue
        out["status"] = "OK"
        out["p_primary"] = score.p_score
        out["primary_method"] = "score_normal"

        context = dict(entry.row_context)
        context["p_score"] = score.p_score
        if _should_apply_spa(context, config):
            out["spa_applied"] = 1
            if score.adjusted_predictor is None:
                raise RuntimeError("SPA adjustment was not prepared for an SPA-selected score")
            fitted_probability = fit.fitted_probability
            if flipped:
                fitted_probability = 1.0 - fitted_probability
            spa = spa_pvalue(
                score_sign * score.score_u,
                score.score_variance,
                score.adjusted_predictor,
                fitted_probability,
            )
            out["spa_status"] = spa.status
            out["spa_variance_ratio"] = spa.variance_ratio
            out["p_spa"] = spa.p_value
            if spa.status in {"OK", "NORMAL_NEAR_MEAN"} and np.isfinite(spa.p_value):
                out["p_primary"] = spa.p_value
                out["primary_method"] = "score_spa"
            else:
                out["primary_method"] = "score_normal_spa_failed"
                out["status"] = "OK_SPA_FAILED"

        out["score_primary"] = neglog10(float(out["p_primary"]))
        if float(config.full_refit_p) > 0 and float(out["p_primary"]) <= float(config.full_refit_p):
            out["full_refit_attempted"] = 1
            original_y = X[:, entry.response_locus].astype(np.float64, copy=False)
            full = fit_full_glmm(
                original_y,
                X[:, predictor],
                K,
                max_iter=config.null_max_iter,
                tolerance=config.null_tolerance,
                kinship_eigensystem=_GLOBAL_KINSHIP_EIGENSYSTEM,
            )
            out.update({
                "beta_log_odds": full.beta,
                "se_log_odds": full.se,
                "odds_ratio": full.odds_ratio,
                "odds_ratio_ci_low": full.ci_low,
                "odds_ratio_ci_high": full.ci_high,
                "tau_alt": full.tau,
                "p_wald": full.wald_p,
                "full_refit_status": full.status,
            })
            if full.status != "OK" and out["status"] == "OK":
                out["status"] = "OK_REFIT_FAILED"
        updates.append((output_index, out))
    return null_rows, updates


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    m = ranked.size
    adjusted = ranked * m / np.arange(1, m + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(m, dtype=np.float64)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def scan_pairs_glmm(
    pairs: pd.DataFrame,
    X: np.ndarray,
    K: np.ndarray,
    config: ScanConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run directional PQL logistic mixed-model tests for candidate pairs."""
    started = time.time()
    result = _directional_rows(pairs, X, config)
    for key, value in _empty_inference().items():
        result[key] = value
    eligible_indices = result.index[result["eligible"] == 1].to_numpy(dtype=np.int64)
    tasks, original_response_count = _response_pattern_tasks(result, X)
    kinship_eigensystem = prepare_kinship_eigensystem(K) if tasks else None
    if config.progress:
        sys.stderr.write(
            f"[{_now()}] glmm_scan_start input_pairs={len(pairs)} directional_rows={len(result)} "
            f"eligible={len(eligible_indices)} responses={original_response_count} "
            f"canonical_response_patterns={len(tasks)} "
            f"reused_response_models={original_response_count-len(tasks)} "
            f"threads={max(1, int(config.threads))}\n"
        )
        sys.stderr.flush()

    null_rows: list[dict[str, Any]] = []
    done = 0
    represented_responses = 0
    if max(1, int(config.threads)) == 1:
        _init_worker(X, K, config, kinship_eigensystem)
        iterator = (_worker_response(task) for task in tasks)
        pool = None
    else:
        context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        pool = context.Pool(
            processes=max(1, int(config.threads)),
            initializer=_init_worker,
            initargs=(X, K, config, kinship_eigensystem),
        )
        iterator = pool.imap_unordered(
            _worker_response,
            tasks,
            chunksize=max(1, int(config.worker_chunk_size)),
        )
    try:
        for task_null_rows, updates in iterator:
            null_rows.extend(task_null_rows)
            for index, update in updates:
                for key, value in update.items():
                    result.at[index, key] = value
            done += 1
            represented_responses += len(task_null_rows)
            if config.progress and (
                done == 1
                or done == len(tasks)
                or done % max(1, int(config.progress_every_responses)) == 0
            ):
                elapsed = time.time() - started
                sys.stderr.write(
                    f"[{_now()}] glmm_response_progress={represented_responses}/{original_response_count} "
                    f"canonical_patterns={done}/{len(tasks)} elapsed={elapsed:.1f}s\n"
                )
                sys.stderr.flush()
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()

    finite = np.isfinite(pd.to_numeric(result["p_primary"], errors="coerce").to_numpy(dtype=np.float64))
    q_values = np.full(len(result), np.nan, dtype=np.float64)
    bonferroni = np.zeros(len(result), dtype=np.int8)
    n_tested = int(np.count_nonzero(finite))
    if n_tested:
        primary = result.loc[finite, "p_primary"].to_numpy(dtype=np.float64)
        q_values[finite] = _bh_adjust(primary)
        bonferroni[finite] = (primary <= 0.05 / n_tested).astype(np.int8)
    result["q_bh"] = q_values
    result["bonferroni_significant"] = bonferroni
    result["n_directional_tests"] = n_tested
    result = result.sort_values(["pair_id", "direction_order"], kind="stable").reset_index(drop=True)
    result = result.drop(columns=["direction_order"])

    response_models = pd.DataFrame(null_rows)
    if not response_models.empty:
        response_models = response_models.sort_values("response_locus", kind="stable").reset_index(drop=True)
    if config.progress:
        sys.stderr.write(
            f"[{_now()}] glmm_scan_complete tested={n_tested} elapsed={time.time()-started:.1f}s\n"
        )
        sys.stderr.flush()
    return result, response_models

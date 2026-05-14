from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from typing import Any

import numpy as np
import pandas as pd

from .model_fit import fit_logistic_mds


@dataclass
class ScanConfig:
    use_firth: bool = True
    high_bse_threshold: float = 3.0
    low_count_threshold: int = 1
    chunksize: int = 100
    min_maf: float = 0.01
    reduce_mds_by_minor_count: bool = False


def contingency_counts(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int, int]:
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    n11 = int(np.sum((a == 1) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    return n11, n10, n01, n00


def _select_covariates_for_response(
    y: np.ndarray,
    covariates: np.ndarray,
    reduce_mds_by_minor_count: bool,
) -> np.ndarray:
    if not reduce_mds_by_minor_count:
        return covariates
    y = np.asarray(y, dtype=np.uint8)
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    minor_count = min(n1, n0)
    k_global = covariates.shape[1]
    # Model has intercept + tested locus + MDS axes.
    k_eff = min(k_global, max(0, minor_count - 3))
    return covariates[:, :k_eff]


def _combine_status(
    s1: str,
    s2: str,
    counts: tuple[int, int, int, int],
    low_count_threshold: int,
) -> str:
    notes: list[str] = []
    if s1 == "OK" and s2 == "OK":
        notes.append("OK")
    else:
        notes.append(f"{s1}|{s2}")
    if min(counts) <= low_count_threshold:
        notes.append("LOW_CELL")
    if counts[1] == 0 or counts[2] == 0:
        notes.append("SEPARATION_WARNING")
    return ";".join(notes)


def _low_maf_row(u: int, v: int, distance: Any, counts, freq_u, freq_v, config: ScanConfig) -> dict[str, Any]:
    n11, n10, n01, n00 = counts
    status_notes = ["LOW_MAF"]
    if min(counts) <= config.low_count_threshold:
        status_notes.append("LOW_CELL")
    if n10 == 0 or n01 == 0:
        status_notes.append("SEPARATION_WARNING")
    return {
        "u": int(u), "v": int(v), "distance": distance,
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "freq_u": freq_u, "freq_v": freq_v,
        "beta_u_to_v": np.nan, "p_u_to_v": np.nan,
        "beta_v_to_u": np.nan, "p_v_to_u": np.nan,
        "p_pair_max": np.nan,
        "status": ";".join(status_notes),
    }


def scan_pair(
    u: int,
    v: int,
    distance: Any,
    Y: np.ndarray,
    covariates: np.ndarray,
    config: ScanConfig,
) -> dict[str, Any]:
    a = Y[:, u]
    b = Y[:, v]
    n = len(a)
    counts = contingency_counts(a, b)
    n11, n10, n01, n00 = counts
    freq_u = (n11 + n10) / n
    freq_v = (n11 + n01) / n
    maf_u = min(freq_u, 1.0 - freq_u)
    maf_v = min(freq_v, 1.0 - freq_v)

    if maf_u < config.min_maf or maf_v < config.min_maf:
        return _low_maf_row(u, v, distance, counts, freq_u, freq_v, config)

    cov_u_to_v = _select_covariates_for_response(
        y=b,
        covariates=covariates,
        reduce_mds_by_minor_count=config.reduce_mds_by_minor_count,
    )
    cov_v_to_u = _select_covariates_for_response(
        y=a,
        covariates=covariates,
        reduce_mds_by_minor_count=config.reduce_mds_by_minor_count,
    )

    fit_u_to_v = fit_logistic_mds(
        y=b,
        x=a,
        covariates=cov_u_to_v,
        use_firth=config.use_firth,
        high_bse_threshold=config.high_bse_threshold,
    )
    fit_v_to_u = fit_logistic_mds(
        y=a,
        x=b,
        covariates=cov_v_to_u,
        use_firth=config.use_firth,
        high_bse_threshold=config.high_bse_threshold,
    )

    p_values = [fit_u_to_v.p, fit_v_to_u.p]
    p_pair_max = float(max(p_values)) if all(np.isfinite(p) for p in p_values) else np.nan

    return {
        "u": int(u), "v": int(v), "distance": distance,
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "freq_u": freq_u, "freq_v": freq_v,
        "beta_u_to_v": fit_u_to_v.beta,
        "p_u_to_v": fit_u_to_v.p,
        "beta_v_to_u": fit_v_to_u.beta,
        "p_v_to_u": fit_v_to_u.p,
        "p_pair_max": p_pair_max,
        "status": _combine_status(
            fit_u_to_v.status,
            fit_v_to_u.status,
            counts,
            config.low_count_threshold,
        ),
    }


_WORKER_Y: np.ndarray | None = None
_WORKER_COVARIATES: np.ndarray | None = None
_WORKER_CONFIG: ScanConfig | None = None


def _init_worker(Y: np.ndarray, covariates: np.ndarray, config: ScanConfig) -> None:
    global _WORKER_Y, _WORKER_COVARIATES, _WORKER_CONFIG
    _WORKER_Y = Y
    _WORKER_COVARIATES = covariates
    _WORKER_CONFIG = config


def _scan_pair_worker(record: tuple[int, int, int, Any]) -> tuple[int, dict[str, Any]]:
    idx, u, v, distance = record
    if _WORKER_Y is None or _WORKER_COVARIATES is None or _WORKER_CONFIG is None:
        raise RuntimeError("Worker was not initialized correctly.")
    row = scan_pair(u, v, distance, _WORKER_Y, _WORKER_COVARIATES, _WORKER_CONFIG)
    return idx, row


def _result_columns() -> list[str]:
    return [
        "u", "v", "distance", "n11", "n10", "n01", "n00", "freq_u", "freq_v",
        "beta_u_to_v", "p_u_to_v", "beta_v_to_u", "p_v_to_u", "p_pair_max", "status",
    ]


def scan_pairs(
    pairs: pd.DataFrame,
    Y: np.ndarray,
    covariates: np.ndarray,
    config: ScanConfig,
    threads: int = 1,
) -> pd.DataFrame:
    import sys
    import time

    records = [
        (idx, int(u), int(v), distance)
        for idx, (u, v, distance) in enumerate(
            pairs[["u", "v", "distance"]].itertuples(index=False, name=None)
        )
    ]
    total = len(records)
    if total == 0:
        return pd.DataFrame(columns=_result_columns())

    progress_every = max(1, total // 100)
    start_time = time.time()

    def report_progress(done: int) -> None:
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        pct = 100.0 * done / total
        if rate > 0:
            remaining = (total - done) / rate
            msg = f"\r  exact_progress={done}/{total} ({pct:5.1f}%) rate={rate:6.1f} pairs/s eta={remaining:7.1f}s"
        else:
            msg = f"\r  exact_progress={done}/{total} ({pct:5.1f}%)"
        sys.stderr.write(msg)
        sys.stderr.flush()

    rows_with_idx: list[tuple[int, dict[str, Any]]] = []
    done = 0
    report_progress(done)

    if threads <= 1:
        for idx, u, v, distance in records:
            row = scan_pair(u, v, distance, Y, covariates, config)
            rows_with_idx.append((idx, row))
            done += 1
            if done % progress_every == 0 or done == total:
                report_progress(done)
    else:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
        with ctx.Pool(processes=threads, initializer=_init_worker, initargs=(Y, covariates, config)) as pool:
            iterator = pool.imap_unordered(_scan_pair_worker, records, chunksize=config.chunksize)
            for item in iterator:
                rows_with_idx.append(item)
                done += 1
                if done % progress_every == 0 or done == total:
                    report_progress(done)

    sys.stderr.write("\n")
    rows_with_idx.sort(key=lambda x: x[0])
    rows = [row for _, row in rows_with_idx]
    return pd.DataFrame(rows, columns=_result_columns())

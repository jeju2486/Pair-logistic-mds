from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import time

import numpy as np
import pandas as pd

from .logit import fit_null_logit, efficient_score, score_p_chisq, score_p_spa, fit_exact_lrt


@dataclass
class ScanConfig:
    min_count: int = 3
    engine: str = "score"  # score or exact
    spa: bool = False
    firth_fallback: bool = True
    max_iter: int = 100
    high_se_threshold: float = 3.0
    batch_size: int = 4096

    # Console progress tracking, similar to the previous pair_logistic_mds code.
    progress: bool = True
    progress_every_rows: int = 10000
    progress_every_pct: float = 5.0


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _progress_enabled(config: ScanConfig) -> bool:
    return bool(getattr(config, "progress", True))


def _progress_interval(total: int, config: ScanConfig) -> int:
    by_rows = max(1, int(getattr(config, "progress_every_rows", 10000)))
    pct = max(0.1, float(getattr(config, "progress_every_pct", 5.0)))
    by_pct = max(1, int(math.ceil(total * pct / 100.0)))
    return max(1, min(by_rows, by_pct))


def _emit_progress(label: str, done: int, total: int, t0: float, config: ScanConfig, extra: str = "") -> None:
    if not _progress_enabled(config):
        return
    elapsed = max(time.time() - t0, 1e-9)
    rate = done / elapsed
    pct = 100.0 * done / total if total else 100.0
    msg = (
        f"[{_now()}] {label}_progress={done}/{total} "
        f"({pct:5.1f}%) elapsed={elapsed:8.1f}s rate={rate:8.1f}/s"
    )
    if extra:
        msg += f" {extra}"
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def neglog10(p: float) -> float:
    if p is None or not np.isfinite(p) or p <= 0:
        return np.nan
    return -math.log10(max(float(p), 1e-300))


def pair_counts(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int, int]:
    aa = a.astype(bool)
    bb = b.astype(bool)
    n11 = int(np.sum(aa & bb))
    n10 = int(np.sum(aa & ~bb))
    n01 = int(np.sum(~aa & bb))
    n00 = int(np.sum(~aa & ~bb))
    return n11, n10, n01, n00


def raw_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    n11, n10, n01, n00 = pair_counts(a, b)
    denom = math.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    phi = ((n11*n00 - n10*n01) / denom) if denom > 0 else np.nan
    r2 = phi * phi if np.isfinite(phi) else np.nan
    or_pc = ((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5))
    return {
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "min_cell": min(n11, n10, n01, n00),
        "r2": r2,
        "or": or_pc,
    }


def _meta_cols(pairs: pd.DataFrame) -> list[str]:
    return [c for c in ["distance", "min_distance", "MI"] if c in pairs.columns]


def _compact_row_base(row: pd.Series, meta_cols: list[str]) -> dict:
    out = {"u": int(row["u"]), "v": int(row["v"])}
    for c in meta_cols:
        out[c] = row[c]
    return out


def _fit_direction_score(response: int, predictor: int, X: np.ndarray, null_cache: dict[int, object], spa: bool) -> tuple[float, float, str]:
    null = null_cache[response]
    if getattr(null, "status") != "OK":
        return np.nan, np.nan, f"NULL_{getattr(null, 'status')}"
    x = X[:, predictor].astype(np.float64, copy=False)
    if np.unique(x).size < 2:
        return np.nan, np.nan, "NO_VARIATION_PREDICTOR"
    U, V, beta, xt = efficient_score(x, null)
    p = score_p_spa(U, V, xt, null.mu) if spa else score_p_chisq(U, V)
    if not np.isfinite(p):
        return beta, np.nan, "SCORE_FAILED"
    return beta, p, "OK"


def _fit_direction_exact(response: int, predictor: int, X: np.ndarray, covariates: np.ndarray, firth: bool, high_se_threshold: float) -> tuple[float, float, str]:
    y = X[:, response].astype(np.float64, copy=False)
    x = X[:, predictor].astype(np.float64, copy=False)
    fit = fit_exact_lrt(y, x, covariates, use_firth=firth, high_bse_threshold=high_se_threshold)
    return fit.beta, fit.p, fit.status


def scan_pairs(pairs: pd.DataFrame, X: np.ndarray, covariates: np.ndarray, config: ScanConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run bidirectional MDS-corrected pair tests.

    Output is intentionally compact. Null fits are cached per response locus.
    Progress is printed to stderr so a wrapper script can save it with tee.
    """
    t_all = time.time()
    n_samples = X.shape[0]
    n_pairs = len(pairs)
    meta_cols = _meta_cols(pairs)
    responses = sorted(set(pairs["u"].astype(int).tolist()) | set(pairs["v"].astype(int).tolist()))

    if _progress_enabled(config):
        sys.stderr.write(
            f"[{_now()}] scan_start n_pairs={n_pairs} "
            f"n_responses={len(responses)} n_samples={n_samples} "
            f"engine={config.engine} spa={config.spa} min_count={config.min_count}\n"
        )
        sys.stderr.flush()

    # Step 1: response-wise null models.
    null_cache = {}
    null_rows = []
    t_null = time.time()
    null_interval = _progress_interval(len(responses), config)
    null_ok = 0
    null_fail = 0

    for i, loc in enumerate(responses, 1):
        y = X[:, loc].astype(np.float64, copy=False)
        mac = int(min(y.sum(), n_samples - y.sum()))
        maf = mac / n_samples
        nf = fit_null_logit(y, covariates, max_iter=config.max_iter)
        null_cache[loc] = nf
        if nf.status == "OK":
            null_ok += 1
        else:
            null_fail += 1
        null_rows.append({
            "response": loc,
            "n_present": int(y.sum()),
            "mac": mac,
            "maf": maf,
            "status": nf.status,
            "null_loglik": nf.loglik,
        })
        if i == 1 or i == len(responses) or (i % null_interval == 0):
            _emit_progress(
                "null_models",
                i,
                len(responses),
                t_null,
                config,
                extra=f"ok={null_ok} fail={null_fail}",
            )

    # Step 2: pair scan.
    rows = []
    t_scan = time.time()
    pair_interval = _progress_interval(n_pairs, config)
    n_ok = 0
    n_low = 0
    n_fail = 0

    for done, (_, row) in enumerate(pairs.iterrows(), 1):
        u = int(row["u"])
        v = int(row["v"])
        xu = X[:, u]
        xv = X[:, v]
        st = raw_stats(xu, xv)
        out = _compact_row_base(row, meta_cols)
        out.update({
            "n11": st["n11"],
            "n10": st["n10"],
            "n01": st["n01"],
            "n00": st["n00"],
            "r2": st["r2"],
            "or": st["or"],
        })

        if st["min_cell"] < config.min_count:
            out.update({
                "p_u_to_v": np.nan,
                "p_v_to_u": np.nan,
                "bidirectional_score": np.nan,
                "status": "LOW_CELL",
            })
            n_low += 1
        else:
            if config.engine == "exact":
                beta_uv, p_uv, s_uv = _fit_direction_exact(v, u, X, covariates, config.firth_fallback, config.high_se_threshold)
                beta_vu, p_vu, s_vu = _fit_direction_exact(u, v, X, covariates, config.firth_fallback, config.high_se_threshold)
            else:
                beta_uv, p_uv, s_uv = _fit_direction_score(v, u, X, null_cache, config.spa)
                beta_vu, p_vu, s_vu = _fit_direction_score(u, v, X, null_cache, config.spa)

            nlu = neglog10(p_uv)
            nlv = neglog10(p_vu)
            bidir = min(nlu, nlv) if np.isfinite(nlu) and np.isfinite(nlv) else np.nan
            status = "OK" if s_uv == "OK" and s_vu == "OK" else f"{s_uv};{s_vu}"
            if status == "OK":
                n_ok += 1
            else:
                n_fail += 1
            out.update({
                "p_u_to_v": p_uv,
                "p_v_to_u": p_vu,
                "bidirectional_score": bidir,
                "status": status,
            })
        rows.append(out)

        if done == 1 or done == n_pairs or (done % pair_interval == 0):
            _emit_progress(
                "pair_scan",
                done,
                n_pairs,
                t_scan,
                config,
                extra=f"ok={n_ok} low_cell={n_low} fail={n_fail}",
            )

    if _progress_enabled(config):
        elapsed = time.time() - t_all
        sys.stderr.write(
            f"[{_now()}] scan_done n_pairs={n_pairs} ok={n_ok} "
            f"low_cell={n_low} fail={n_fail} elapsed={elapsed:.1f}s\n"
        )
        sys.stderr.flush()

    return pd.DataFrame(rows), pd.DataFrame(null_rows)

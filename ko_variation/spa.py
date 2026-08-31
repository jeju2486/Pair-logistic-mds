from __future__ import annotations

"""Optional saddlepoint calibration for KOVAR score tests.

This module implements a Lugannani-Rice approximation for a weighted centred
Bernoulli score.  When the mixed-model variance differs from the conditional
Bernoulli variance, the observed score is adjusted by their variance ratio in
the same broad spirit as SAIGE.  It is deliberately off by default and is not
claimed to reproduce SAIGE's full variance-ratio machinery.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy import optimize, stats
from scipy.special import expit


@dataclass
class SPAResult:
    p_value: float = np.nan
    status: str = "NOT_APPLIED"
    saddlepoint: float = np.nan
    variance_ratio: float = np.nan
    root_residual: float = np.nan


def cumulants(t: float, weights: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float]:
    """Return K(t), K'(t), K''(t) for sum a_i(Y_i-mu_i)."""
    a = np.asarray(weights, dtype=np.float64).reshape(-1)
    mu = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if a.size != mu.size:
        raise ValueError("SPA weights and probabilities must have equal length")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(mu)):
        raise ValueError("SPA inputs contain non-finite values")
    if np.any((mu <= 0.0) | (mu >= 1.0)):
        raise ValueError("SPA probabilities must be strictly between zero and one")
    logit_mu = np.log(mu) - np.log1p(-mu)
    tilted = expit(logit_mu + float(t) * a)
    log_normaliser = np.logaddexp(np.log1p(-mu), np.log(mu) + float(t) * a)
    K = float(np.sum(log_normaliser - float(t) * a * mu))
    K1 = float(np.sum(a * (tilted - mu)))
    K2 = float(np.sum(a * a * tilted * (1.0 - tilted)))
    return K, K1, K2


def _solve_saddlepoint(score: float, weights: np.ndarray, probabilities: np.ndarray) -> float:
    if abs(score) <= 1e-14:
        return 0.0

    def root(t: float) -> float:
        return cumulants(t, weights, probabilities)[1] - score

    if score > 0:
        lo, hi = 0.0, 1.0
        while root(hi) < 0.0 and hi < 1e6:
            hi *= 2.0
    else:
        lo, hi = -1.0, 0.0
        while root(lo) > 0.0 and abs(lo) < 1e6:
            lo *= 2.0
    flo = root(lo)
    fhi = root(hi)
    if not np.isfinite(flo) or not np.isfinite(fhi) or flo * fhi > 0:
        raise ValueError("Observed score is outside the SPA support or could not be bracketed")
    solved = optimize.root_scalar(root, bracket=(lo, hi), method="brentq", xtol=1e-10, rtol=1e-10)
    if not solved.converged:
        raise ValueError("SPA saddlepoint root did not converge")
    return float(solved.root)


def _lugannani_rice_cdf(
    threshold: float,
    weights: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float, float]:
    """Approximate P(score <= threshold) and return root diagnostics."""
    centered_zero = -weights * probabilities
    centered_one = weights * (1.0 - probabilities)
    support_min = float(np.sum(np.minimum(centered_zero, centered_one)))
    support_max = float(np.sum(np.maximum(centered_zero, centered_one)))
    if threshold < support_min:
        return 0.0, np.nan, 0.0
    if threshold > support_max:
        return 1.0, np.nan, 0.0
    saddle = _solve_saddlepoint(threshold, weights, probabilities)
    K, K1, K2 = cumulants(saddle, weights, probabilities)
    radicand = 2.0 * (saddle * threshold - K)
    if radicand <= 0 or K2 <= 0:
        raise ValueError("Invalid Lugannani-Rice quantities")
    w = math.copysign(math.sqrt(radicand), saddle)
    v = saddle * math.sqrt(K2)
    if abs(w) <= 1e-10 or abs(v) <= 1e-10:
        raise ValueError("Saddlepoint is too close to the score mean")
    cdf = float(stats.norm.cdf(w) + stats.norm.pdf(w) * (1.0 / w - 1.0 / v))
    return min(max(cdf, 0.0), 1.0), saddle, float(K1 - threshold)


def spa_pvalue(
    score: float,
    target_variance: float,
    weights: np.ndarray,
    probabilities: np.ndarray,
) -> SPAResult:
    """Calculate a two-sided SPA p-value for a mixed-model score statistic."""
    a = np.asarray(weights, dtype=np.float64).reshape(-1)
    mu = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not np.isfinite(score) or not np.isfinite(target_variance) or target_variance <= 0:
        return SPAResult(status="BAD_SCORE_OR_VARIANCE")
    if a.size != mu.size or a.size == 0:
        return SPAResult(status="BAD_INPUT_SHAPE")
    conditional_variance = float(np.sum(a * a * mu * (1.0 - mu)))
    if not np.isfinite(conditional_variance) or conditional_variance <= 1e-14:
        return SPAResult(status="DEGENERATE_CONDITIONAL_VARIANCE")
    ratio = float(target_variance / conditional_variance)
    if not np.isfinite(ratio) or ratio <= 0:
        return SPAResult(status="BAD_VARIANCE_RATIO")

    # Approximate the mixed score by sqrt(ratio) times the conditional
    # Bernoulli score, so the saddlepoint is evaluated at U/sqrt(ratio).
    adjusted_score = float(score / math.sqrt(ratio))
    normal_z = adjusted_score / math.sqrt(conditional_variance)
    if abs(normal_z) < 2.0:
        return SPAResult(
            p_value=float(2.0 * stats.norm.sf(abs(normal_z))),
            status="NORMAL_NEAR_MEAN",
            saddlepoint=0.0,
            variance_ratio=ratio,
            root_residual=0.0,
        )
    try:
        magnitude = abs(adjusted_score)
        cdf_positive, saddle_positive, residual_positive = _lugannani_rice_cdf(
            magnitude, a, mu
        )
        cdf_negative, saddle_negative, residual_negative = _lugannani_rice_cdf(
            -magnitude, a, mu
        )
        # A two-sided skewed score distribution requires both saddlepoints:
        # P(S >= |s|) + P(S <= -|s|). Doubling the smaller of one CDF and its
        # complement is only valid for a symmetric distribution.
        p = min(1.0, max(0.0, (1.0 - cdf_positive) + cdf_negative))
        if not np.isfinite(p) or p <= 0.0:
            p = float(max(np.finfo(float).tiny, 2.0 * stats.norm.sf(abs(normal_z))))
        if adjusted_score >= 0:
            saddle, residual = saddle_positive, residual_positive
        else:
            saddle, residual = saddle_negative, residual_negative
        return SPAResult(
            p_value=p,
            status="OK",
            saddlepoint=saddle,
            variance_ratio=ratio,
            root_residual=residual,
        )
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return SPAResult(status=f"SPA_FAILED:{exc}", variance_ratio=ratio)

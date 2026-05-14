from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationError,
    PerfectSeparationWarning,
    ConvergenceWarning,
)


@dataclass
class FitResult:
    beta: float
    p: float
    status: str


def _bad_fit(beta: float, bse: float | None = None, max_abs_beta: float = 30.0) -> bool:
    """Detect numerically unstable logistic fits."""
    if not np.isfinite(beta):
        return True
    if abs(beta) > max_abs_beta:
        return True
    if bse is not None:
        if not np.isfinite(bse):
            return True
        if bse <= 0:
            return True
    return False


def _safe_logit_start(y: np.ndarray, p: int) -> np.ndarray:
    # Add small empirical correction to avoid infinite intercept for rare labels.
    mean = (float(y.sum()) + 0.5) / (len(y) + 1.0)
    mean = min(max(mean, 1e-8), 1.0 - 1e-8)
    start = np.zeros(p, dtype=np.float64)
    start[0] = math.log(mean / (1.0 - mean))
    return start


def _design_matrix(y: np.ndarray, x: np.ndarray, covariates: np.ndarray | None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    if covariates is None or covariates.size == 0:
        return np.column_stack([np.ones(y.shape[0]), x])
    cov = np.asarray(covariates, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    return np.concatenate([np.ones((y.shape[0], 1)), x, cov], axis=1)


def _null_matrix(y: np.ndarray, covariates: np.ndarray | None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if covariates is None or covariates.size == 0:
        return np.ones((y.shape[0], 1), dtype=np.float64)
    cov = np.asarray(covariates, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    return np.concatenate([np.ones((y.shape[0], 1)), cov], axis=1)


def _fit_logit_ll(y: np.ndarray, X: np.ndarray):
    mod = sm.Logit(y, X)
    mod.raise_on_perfect_prediction = True
    start = _safe_logit_start(y, X.shape[1])

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        warnings.filterwarnings("error", category=PerfectSeparationWarning)
        warnings.filterwarnings("error", category=ConvergenceWarning)
        res = mod.fit(start_params=start, method="newton", disp=False, maxiter=100)

    if not bool(res.mle_retvals.get("converged", False)):
        raise ConvergenceWarning("Logit did not converge")
    if not np.isfinite(res.llf):
        raise FloatingPointError("non-finite log-likelihood")
    return mod, res


def _firth_likelihood(beta: np.ndarray, mod: sm.Logit) -> float:
    """Negative penalized likelihood for minimization/step-halving."""
    X = mod.exog
    y = mod.endog
    eta = X @ beta
    eta = np.clip(eta, -35.0, 35.0)
    loglike = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    pi = 1.0 / (1.0 + np.exp(-eta))
    W = pi * (1.0 - pi)
    H = X.T @ (W[:, None] * X)
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0 or not np.isfinite(logdet):
        return np.inf
    return -(loglike + 0.5 * logdet)


def _fit_firth(y: np.ndarray, X: np.ndarray, max_iter: int = 1000, tol: float = 1e-5):
    """Firth logistic regression fallback for separable binary traits."""
    mod = sm.Logit(y, X)
    beta = _safe_logit_start(y, X.shape[1])
    last = beta.copy()

    for _ in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -35.0, 35.0)
        pi = 1.0 / (1.0 + np.exp(-eta))
        W = pi * (1.0 - pi)
        H = X.T @ (W[:, None] * X)
        try:
            vcov = np.linalg.pinv(H)
        except np.linalg.LinAlgError:
            return None

        # Diagonal of hat matrix: diag(W^1/2 X (X'WX)^-1 X' W^1/2)
        Xv = X @ vcov
        hat_diag = W * np.sum(Xv * X, axis=1)
        U = X.T @ (y - pi + hat_diag * (0.5 - pi))
        step = vcov @ U
        new_beta = beta + step

        old_obj = _firth_likelihood(beta, mod)
        for _ in range(50):
            new_obj = _firth_likelihood(new_beta, mod)
            if np.isfinite(new_obj) and new_obj <= old_obj:
                break
            new_beta = beta + 0.5 * (new_beta - beta)
        else:
            return None

        last = beta
        beta = new_beta
        if np.linalg.norm(beta - last) < tol:
            fitll = -_firth_likelihood(beta, mod)
            return beta, fitll
    return None


def fit_logistic_mds(
    y: np.ndarray,
    x: np.ndarray,
    covariates: np.ndarray | None,
    use_firth: bool = True,
    high_bse_threshold: float = 3.0,
) -> FitResult:
    """Fit y ~ x + covariates with binary logit link.

    p-value is an LRT comparing y ~ x + covariates against y ~ covariates.
    If ordinary logistic regression has separation, singularity, or unstable
    estimates, Firth logistic regression is attempted.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if y.size != x.size:
        raise ValueError("y and x must have same length")
    if np.unique(y).size < 2:
        return FitResult(np.nan, np.nan, "LOW_VARIATION")
    if np.unique(x).size < 2:
        return FitResult(np.nan, np.nan, "LOW_VARIATION")

    X_full = _design_matrix(y, x, covariates)
    X_null = _null_matrix(y, covariates)

    try:
        _, null_res = _fit_logit_ll(y, X_null)
        _, full_res = _fit_logit_ll(y, X_full)
        beta = float(full_res.params[1])
        bse = float(full_res.bse[1])
        if _bad_fit(beta, bse):
            raise PerfectSeparationError("unstable beta/bse fallback")
        if bse > high_bse_threshold and use_firth:
            raise PerfectSeparationError("high-bse fallback")
        lrstat = 2.0 * (float(full_res.llf) - float(null_res.llf))
        if not np.isfinite(lrstat):
            raise FloatingPointError("non-finite LRT statistic")
        p = stats.chi2.sf(max(lrstat, 0.0), 1)
        if not np.isfinite(p):
            raise FloatingPointError("non-finite p-value")
        return FitResult(beta, float(p), "OK")
    except Exception:
        if not use_firth:
            return FitResult(np.nan, np.nan, "FAILED")

    try:
        full_fit = _fit_firth(y, X_full)
        null_fit = _fit_firth(y, X_null)
        if full_fit is None or null_fit is None:
            return FitResult(np.nan, np.nan, "FAILED")
        full_beta, full_ll = full_fit
        _, null_ll = null_fit
        beta = float(full_beta[1])
        if _bad_fit(beta, None, max_abs_beta=50.0):
            return FitResult(np.nan, np.nan, "FAILED")
        lrstat = 2.0 * (float(full_ll) - float(null_ll))
        if not np.isfinite(lrstat):
            return FitResult(np.nan, np.nan, "FAILED")
        p = stats.chi2.sf(max(lrstat, 0.0), 1)
        if not np.isfinite(p):
            return FitResult(np.nan, np.nan, "FAILED")
        return FitResult(beta, float(p), "FIRTH")
    except Exception:
        return FitResult(np.nan, np.nan, "FAILED")

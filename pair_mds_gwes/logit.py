from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy import stats, optimize
from scipy.special import expit, ndtr
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError, PerfectSeparationWarning


@dataclass
class NullFit:
    mu: np.ndarray
    W: np.ndarray
    resid: np.ndarray
    Z: np.ndarray
    inv_info: np.ndarray
    loglik: float
    status: str
    converged: bool


@dataclass
class ExactFit:
    beta: float
    se: float
    p: float
    loglik_alt: float
    status: str


def make_null_design(n: int, covariates: np.ndarray | None) -> np.ndarray:
    if covariates is None or covariates.size == 0:
        return np.ones((n, 1), dtype=np.float64)
    cov = np.asarray(covariates, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    return np.column_stack([np.ones(n, dtype=np.float64), cov])


def make_full_design(x: np.ndarray, covariates: np.ndarray | None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if covariates is None or covariates.size == 0:
        return np.column_stack([np.ones(n), x])
    cov = np.asarray(covariates, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    return np.column_stack([np.ones(n), x, cov])


def logistic_loglik(y: np.ndarray, eta: np.ndarray) -> float:
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def _safe_logit_start(y: np.ndarray, p: int) -> np.ndarray:
    mean = (float(y.sum()) + 0.5) / (len(y) + 1.0)
    mean = min(max(mean, 1e-8), 1.0 - 1e-8)
    beta = np.zeros(p, dtype=np.float64)
    beta[0] = math.log(mean / (1 - mean))
    return beta



def _fit_logit_irls_ridge(y: np.ndarray, X: np.ndarray, max_iter: int = 100, tol: float = 1e-7, ridge: float = 1e-8):
    """Small ridge IRLS fallback used only to keep null score fitting robust."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    p = X.shape[1]
    beta = _safe_logit_start(y, p)
    last_ll = -np.inf
    for it in range(max_iter):
        eta = np.clip(X @ beta, -35.0, 35.0)
        mu = expit(eta)
        W = np.clip(mu * (1.0 - mu), 1e-10, None)
        grad = X.T @ (y - mu) - ridge * beta
        H = (X.T * W) @ X
        H.flat[::p+1] += ridge
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        step_factor = 1.0
        ll0 = logistic_loglik(y, np.clip(X @ beta, -35.0, 35.0)) - 0.5 * ridge * float(beta @ beta)
        for _ in range(30):
            nb = beta + step_factor * step
            ll1 = logistic_loglik(y, np.clip(X @ nb, -35.0, 35.0)) - 0.5 * ridge * float(nb @ nb)
            if ll1 >= ll0 - 1e-10:
                beta = nb
                break
            step_factor *= 0.5
        ll = logistic_loglik(y, np.clip(X @ beta, -35.0, 35.0))
        if np.max(np.abs(step_factor * step)) < tol or abs(ll - last_ll) < tol:
            return beta, ll, True
        last_ll = ll
    return beta, logistic_loglik(y, np.clip(X @ beta, -35.0, 35.0)), False

def fit_null_logit(y: np.ndarray, covariates: np.ndarray | None, max_iter: int = 100) -> NullFit:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    Z = make_null_design(n, covariates)
    if np.unique(y).size < 2:
        return NullFit(np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), Z, np.full((Z.shape[1], Z.shape[1]), np.nan), np.nan, "NO_VARIATION_RESPONSE", False)
    try:
        mod = sm.Logit(y, Z)
        mod.raise_on_perfect_prediction = True
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            warnings.filterwarnings("error", category=PerfectSeparationWarning)
            warnings.filterwarnings("error", category=ConvergenceWarning)
            res = mod.fit(start_params=_safe_logit_start(y, Z.shape[1]), method="newton", disp=False, maxiter=max_iter)
        if not bool(res.mle_retvals.get("converged", False)):
            raise RuntimeError("NULL_NOT_CONVERGED")
        eta = np.clip(Z @ np.asarray(res.params), -35.0, 35.0)
        mu = expit(eta)
        W = np.clip(mu * (1 - mu), 1e-12, None)
        info = Z.T @ (W[:, None] * Z)
        inv = np.linalg.pinv(info)
        return NullFit(mu, W, y - mu, Z, inv, float(res.llf), "OK", True)
    except Exception as e:
        # Fallback to a tiny-ridge IRLS null. This keeps score testing usable when
        # statsmodels is over-strict on small or nearly separated null fits.
        try:
            beta, ll, conv = _fit_logit_irls_ridge(y, Z, max_iter=max_iter)
            eta = np.clip(Z @ beta, -35.0, 35.0)
            mu = expit(eta)
            W = np.clip(mu * (1 - mu), 1e-12, None)
            info = Z.T @ (W[:, None] * Z)
            inv = np.linalg.pinv(info)
            status = "OK" if conv else "OK_RIDGE_MAXITER"
            return NullFit(mu, W, y - mu, Z, inv, float(ll), status, bool(conv))
        except Exception:
            return NullFit(np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), Z, np.full((Z.shape[1], Z.shape[1]), np.nan), np.nan, f"NULL_FAIL:{type(e).__name__}", False)


def efficient_score(x: np.ndarray, null: NullFit) -> tuple[float, float, float, np.ndarray]:
    """Return score U, variance V, beta approximation, and weighted-residualized x.

    x_tilde = x - projection of x onto null covariates under W inner product.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    Z = null.Z
    W = null.W
    # weighted projection coefficients: (Z'WZ)^-1 Z'W x
    a = null.inv_info @ (Z.T @ (W * x))
    xt = x - Z @ a
    U = float(xt @ null.resid)
    V = float(np.sum(W * xt * xt))
    beta = U / V if V > 1e-12 and np.isfinite(V) else np.nan
    return U, V, beta, xt


def score_p_chisq(U: float, V: float) -> float:
    if not np.isfinite(U) or not np.isfinite(V) or V <= 1e-12:
        return np.nan
    stat = U * U / V
    return float(stats.chi2.sf(stat, 1))


def _spa_tail_positive(q: float, weights: np.ndarray, mu: np.ndarray) -> float:
    """Approximate P(S >= q) for q > 0 using Lugannani-Rice.

    S = sum_i weights_i * (Y_i - mu_i), Y_i ~ Bernoulli(mu_i).
    Returns np.nan if SPA bracketing fails.
    """
    if q <= 0:
        return 0.5
    w = np.asarray(weights, dtype=np.float64)
    p = np.asarray(mu, dtype=np.float64)
    keep = np.isfinite(w) & np.isfinite(p) & (np.abs(w) > 1e-14) & (p > 0) & (p < 1)
    w = w[keep]
    p = p[keep]
    if w.size == 0:
        return np.nan

    def K(t: float) -> float:
        tw = np.clip(t * w, -700, 700)
        return float(np.sum(np.log1p(p * np.expm1(tw)) - t * p * w))

    def Kp(t: float) -> float:
        tw = np.clip(t * w, -700, 700)
        etw = np.exp(tw)
        pt = p * etw / (1.0 - p + p * etw)
        return float(np.sum(w * (pt - p)))

    def Kpp(t: float) -> float:
        tw = np.clip(t * w, -700, 700)
        etw = np.exp(tw)
        pt = p * etw / (1.0 - p + p * etw)
        return float(np.sum(w * w * pt * (1.0 - pt)))

    hi = 1.0
    try:
        while Kp(hi) < q and hi < 100.0:
            hi *= 2.0
        if Kp(hi) < q:
            return np.nan
        t = optimize.brentq(lambda z: Kp(z) - q, 0.0, hi, maxiter=100)
        kt = K(t)
        kpp = Kpp(t)
        root_arg = 2.0 * (t * q - kt)
        if root_arg <= 0 or kpp <= 0:
            return np.nan
        r = math.copysign(math.sqrt(root_arg), t)
        s = t * math.sqrt(kpp)
        if abs(r) < 1e-10 or abs(s) < 1e-10:
            return np.nan
        phi = math.exp(-0.5 * r * r) / math.sqrt(2.0 * math.pi)
        upper = (1.0 - ndtr(r)) + phi * (1.0 / s - 1.0 / r)
        return float(min(max(upper, 0.0), 1.0))
    except Exception:
        return np.nan


def score_p_spa(U: float, V: float, weights: np.ndarray, mu: np.ndarray) -> float:
    """Two-sided SPA p-value for an efficient logistic score.

    Falls back to chi-square if SPA fails.
    """
    if not np.isfinite(U) or not np.isfinite(V) or V <= 1e-12:
        return np.nan
    if abs(U) < 1e-12:
        return 1.0
    if U > 0:
        tail = _spa_tail_positive(U, weights, mu)
    else:
        tail = _spa_tail_positive(-U, -weights, mu)
    if not np.isfinite(tail):
        return score_p_chisq(U, V)
    return float(min(1.0, max(0.0, 2.0 * tail)))


# Exact logistic/Firth code adapted from the previous pair_logistic_mds fixed-MDS path.
def _fit_logit_ll(y: np.ndarray, X: np.ndarray):
    mod = sm.Logit(y, X)
    mod.raise_on_perfect_prediction = True
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        warnings.filterwarnings("error", category=PerfectSeparationWarning)
        warnings.filterwarnings("error", category=ConvergenceWarning)
        res = mod.fit(start_params=_safe_logit_start(y, X.shape[1]), method="newton", disp=False, maxiter=100)
    if not bool(res.mle_retvals.get("converged", False)):
        raise ConvergenceWarning("Logit did not converge")
    if not np.isfinite(res.llf):
        raise FloatingPointError("non-finite log-likelihood")
    return res


def _firth_likelihood(beta: np.ndarray, y: np.ndarray, X: np.ndarray) -> float:
    eta = np.clip(X @ beta, -35.0, 35.0)
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    pi = expit(eta)
    W = pi * (1 - pi)
    H = X.T @ (W[:, None] * X)
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0 or not np.isfinite(logdet):
        return np.inf
    return -(ll + 0.5 * logdet)


def _fit_firth(y: np.ndarray, X: np.ndarray, max_iter: int = 1000, tol: float = 1e-5):
    beta = _safe_logit_start(y, X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -35.0, 35.0)
        pi = expit(eta)
        W = pi * (1.0 - pi)
        H = X.T @ (W[:, None] * X)
        try:
            vcov = np.linalg.pinv(H)
        except np.linalg.LinAlgError:
            return None
        Xv = X @ vcov
        h = W * np.sum(Xv * X, axis=1)
        U = X.T @ (y - pi + h * (0.5 - pi))
        step = vcov @ U
        new_beta = beta + step
        old_obj = _firth_likelihood(beta, y, X)
        for _ in range(50):
            new_obj = _firth_likelihood(new_beta, y, X)
            if np.isfinite(new_obj) and new_obj <= old_obj:
                break
            new_beta = beta + 0.5 * (new_beta - beta)
        else:
            return None
        if np.linalg.norm(new_beta - beta) < tol:
            return new_beta, -_firth_likelihood(new_beta, y, X)
        beta = new_beta
    return None


def fit_exact_lrt(y: np.ndarray, x: np.ndarray, covariates: np.ndarray | None, use_firth: bool = True, high_bse_threshold: float = 3.0) -> ExactFit:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if y.size != x.size:
        raise ValueError("y and x must have same length")
    if np.unique(y).size < 2 or np.unique(x).size < 2:
        return ExactFit(np.nan, np.nan, np.nan, np.nan, "LOW_VARIATION")
    X_null = make_null_design(y.size, covariates)
    X_full = make_full_design(x, covariates)
    try:
        null_res = _fit_logit_ll(y, X_null)
        full_res = _fit_logit_ll(y, X_full)
        beta = float(full_res.params[1])
        se = float(full_res.bse[1])
        if (not np.isfinite(beta)) or (abs(beta) > 30) or (not np.isfinite(se)) or se <= 0:
            raise PerfectSeparationError("unstable beta/se")
        if se > high_bse_threshold and use_firth:
            raise PerfectSeparationError("high se")
        lr = max(0.0, 2.0 * (float(full_res.llf) - float(null_res.llf)))
        p = float(stats.chi2.sf(lr, 1))
        return ExactFit(beta, se, p, float(full_res.llf), "OK")
    except Exception:
        if not use_firth:
            return ExactFit(np.nan, np.nan, np.nan, np.nan, "FAILED")
    try:
        full = _fit_firth(y, X_full)
        null = _fit_firth(y, X_null)
        if full is None or null is None:
            return ExactFit(np.nan, np.nan, np.nan, np.nan, "FAILED")
        beta_full, ll_full = full
        _, ll_null = null
        beta = float(beta_full[1])
        lr = max(0.0, 2.0 * (float(ll_full) - float(ll_null)))
        p = float(stats.chi2.sf(lr, 1))
        return ExactFit(beta, np.nan, p, float(ll_full), "FIRTH")
    except Exception:
        return ExactFit(np.nan, np.nan, np.nan, np.nan, "FAILED")

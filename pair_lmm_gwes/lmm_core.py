from __future__ import annotations

"""
Pyseer/FaST-LMM-style linear mixed-model core for pair-lmm-gwes.

This module intentionally follows the calculation pattern used by pyseer LMM:
  - regress covariates out of the kernel and tested variables;
  - eigendecompose the covariate-residualised kernel using the same K+I trick;
  - estimate h2 by minimising the transformed-space negative log-likelihood;
  - test predictors using beta^2 / variance_beta, where variance_beta is based
    on the SNP-included residual variance, not the null-model residual variance.

It is standalone and does not import pyseer at runtime.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy import optimize, stats


@dataclass
class NullLMM:
    response: int
    status: str
    h2: float = np.nan
    sigma2: float = np.nan
    n: int = 0
    covar_rank: int = 0
    n_present: int = 0
    mac: int = 0
    null_nll: float = np.nan
    message: str = ""


@dataclass
class BlockLMMResult:
    p: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    status: list[str]
    xKx: np.ndarray | None = None
    xKy: np.ndarray | None = None
    F: np.ndarray | None = None


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _covariate_q(covariates: Optional[np.ndarray], n: int) -> tuple[np.ndarray, int]:
    """Return an orthonormal basis Q for fixed covariates.

    pyseer passes an intercept covariate for LMM. For this standalone tool, if no
    covariates are supplied, use an intercept-only model. Rank-deficient columns
    are dropped through QR with pivot-free diagonal screening; this is sufficient
    for the intercept/default path and avoids scipy.linalg dependency here.
    """
    if covariates is None:
        C = np.ones((n, 1), dtype=np.float64)
    else:
        C = _as_2d(covariates)
        if C.shape[0] != n:
            raise ValueError("covariates row count does not match sample count")
        # pyseer expects the caller to supply the intercept. For robustness in this
        # package, append one only if no near-constant covariate exists.
        if C.shape[1] == 0 or not np.any(np.nanstd(C, axis=0) < 1e-12):
            C = np.c_[C, np.ones((n, 1), dtype=np.float64)]
    # Replace non-finite covariates with column means; fail if impossible.
    if not np.all(np.isfinite(C)):
        C = C.copy()
        for j in range(C.shape[1]):
            col = C[:, j]
            good = np.isfinite(col)
            if not np.any(good):
                raise ValueError("covariate column is entirely non-finite")
            col[~good] = float(np.mean(col[good]))
            C[:, j] = col
    Q, R = np.linalg.qr(C, mode="reduced")
    diag = np.abs(np.diag(R)) if R.size else np.array([], dtype=float)
    keep = diag > 1e-10
    if keep.sum() == 0:
        C = np.ones((n, 1), dtype=np.float64)
        Q, _ = np.linalg.qr(C, mode="reduced")
        return Q[:, :1], 1
    Q = Q[:, keep]
    return Q, int(Q.shape[1])


def _residualise(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    A = _as_2d(A)
    return A - Q @ (Q.T @ A)


def eigen_decompose_kinship(K: np.ndarray, covariates: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """pyseer/FaST-LMM-style eigendecomposition of sample covariance matrix.

    This replaces the earlier raw eigendecomposition. It follows the same key steps
    as pyseer's lmm_cov.setSU_fromK(): add I, residualise K with respect to fixed
    covariates on both sides, eigendecompose, drop the covariate dimensions, then
    subtract 1 from eigenvalues.

    Returns
    -------
    U : ndarray, shape (N, N-D)
        Eigenvectors in the covariate-residualised sample subspace.
    S : ndarray, shape (N-D,)
        Kernel eigenvalues after subtracting the identity contribution.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square sample x sample matrix")
    n = K.shape[0]
    K = 0.5 * (K + K.T)
    Q, D = _covariate_q(covariates, n)

    # pyseer modifies K by adding I before residualising, then subtracts 1 from
    # eigenvalues after dropping covariate axes. This avoids rank problems in the
    # residualised subspace while preserving the original kernel eigenvalues.
    K_work = K.copy()
    K_work.flat[:: n + 1] += 1.0
    P_K = _residualise(K_work, Q)
    P_K = _residualise(P_K.T, Q).T
    P_K = 0.5 * (P_K + P_K.T)
    eigvals, eigvecs = np.linalg.eigh(P_K)
    if D >= n:
        raise ValueError("covariate rank is >= number of samples")
    eigvals = eigvals[D:]
    eigvecs = eigvecs[:, D:]
    S = eigvals - 1.0
    # Tiny negative values can occur from numerical roundoff after residualisation.
    # Keep real negative values as diagnostic-significant only if severe; otherwise
    # clip to zero because h2 < 1 keeps covariance positive.
    S = np.where(S < 0.0, np.maximum(S, -1e-8), S)
    S = np.clip(S, 0.0, None)
    return eigvecs, S


def apply_eigen_rank_limit(U: np.ndarray, S: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the full residual sample basis but zero all except the top `rank` kernel eigenvalues.

    This implements a low-rank covariance component without dropping the residual
    subspace. Dropping columns of U would incorrectly remove residual directions
    from the likelihood/test. Keeping U full with zero eigenvalues means

        V = h2 * K_rank + (1-h2) * I

    where K_rank contains only the largest `rank` eigencomponents.
    """
    U = np.asarray(U, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64).reshape(-1)
    r = int(rank) if rank is not None else 0
    if r <= 0 or r >= S.shape[0]:
        return U, S
    keep = np.argsort(S)[-r:]
    S2 = np.zeros_like(S)
    S2[keep] = S[keep]
    return U, S2


def _rotate_residualised(A: np.ndarray, U: np.ndarray, Q: np.ndarray) -> np.ndarray:
    R = _residualise(A, Q)
    # pyseer zeroes columns that are explained by covariates after residualisation.
    std = np.std(R, axis=0)
    if R.ndim == 2:
        R[:, std <= 1e-10] = 0.0
    return U.T @ R


def _nll_from_rotated(h2: float, UY: np.ndarray, S: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Return nLL, sigma2, yKy and weights for a fixed h2.

    This is the transformed-space full-rank single-kernel likelihood used for h2
    search after covariate residualisation. It uses the residual subspace dimension
    n_eff = N-D, matching pyseer's covariate-regressed LMM representation.
    """
    if not (0.0 <= h2 <= 0.99999):
        return np.inf, np.nan, np.nan, np.array([], dtype=float)
    d = h2 * S + (1.0 - h2)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return np.inf, np.nan, np.nan, np.array([], dtype=float)
    w = 1.0 / d
    y = UY.reshape(-1)
    n_eff = y.shape[0]
    yKy = float(np.dot(y, w * y))
    if yKy <= 0 or not np.isfinite(yKy):
        return np.inf, np.nan, yKy, w
    sigma2 = yKy / n_eff
    nll = 0.5 * (n_eff * (math.log(2.0 * math.pi * sigma2) + 1.0) + float(np.sum(np.log(d))))
    return float(nll), float(sigma2), float(yKy), w


def fit_null_lmm(
    response: int,
    y: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    h2_grid_size: int = 21,
    h2_max: float = 0.99999,
) -> tuple[NullLMM, dict[str, np.ndarray | float | int]]:
    """Fit response-wise pyseer-like LMM null by h2 likelihood search."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.shape[0]
    if U.shape[0] != n:
        raise ValueError("U row count does not match y")
    n_present = int(np.sum(y))
    mac = int(min(n_present, n - n_present))
    if mac == 0:
        return NullLMM(response=response, status="MONOMORPHIC_RESPONSE", n=n, n_present=n_present, mac=mac), {}

    Q, D = _covariate_q(covariates, n)
    UY = _rotate_residualised(y, U, Q).reshape(-1)
    n_eff = int(UY.shape[0])
    if n_eff <= 2:
        return NullLMM(response=response, status="TOO_FEW_DF", n=n, covar_rank=D, n_present=n_present, mac=mac), {}

    h2_upper = float(h2_max)
    if not np.isfinite(h2_upper):
        h2_upper = 0.99999
    h2_upper = min(max(h2_upper, 0.0), 0.99999)
    grid = np.linspace(0.0, h2_upper, max(5, int(h2_grid_size)))
    vals = np.array([_nll_from_rotated(float(h), UY, S)[0] for h in grid])
    if not np.any(np.isfinite(vals)):
        return NullLMM(response=response, status="H2_OPT_FAILED", n=n, covar_rank=D, n_present=n_present, mac=mac), {}
    best_i = int(np.nanargmin(vals))
    candidates: list[tuple[float, float]] = [(float(grid[best_i]), float(vals[best_i]))]

    # Search within local grid brackets, including boundary brackets if optimum is at boundary.
    lo = float(grid[max(0, best_i - 1)])
    hi = float(grid[min(len(grid) - 1, best_i + 1)])
    if hi > lo:
        try:
            opt = optimize.minimize_scalar(
                lambda h: _nll_from_rotated(float(h), UY, S)[0],
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-7, "maxiter": 100},
            )
            if opt.success and np.isfinite(opt.fun):
                candidates.append((float(opt.x), float(opt.fun)))
        except Exception:
            pass
    h2, nll = min(candidates, key=lambda z: z[1])
    nll2, sigma2, yKy, w = _nll_from_rotated(h2, UY, S)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return NullLMM(response=response, status="BAD_SIGMA2", h2=h2, n=n, covar_rank=D, n_present=n_present, mac=mac, null_nll=nll2), {}

    null = NullLMM(
        response=response,
        status="OK",
        h2=float(h2),
        sigma2=float(sigma2),
        n=n,
        covar_rank=D,
        n_present=n_present,
        mac=mac,
        null_nll=float(nll2),
    )
    cache: dict[str, np.ndarray | float | int] = {
        "Q": Q,
        "UY": UY,
        "w": w,
        "h2": float(h2),
        "h2_max": float(h2_upper),
        "sigma2_null": float(sigma2),
        "yKy_null": float(yKy),
        "n_eff": n_eff,
        "covar_rank": D,
        "dof_test": max(1, n - (D + 1)),
    }
    # Backwards-ish diagnostic aliases. These are not old REML quantities, but make
    # basic diagnosis easier if downstream code inspects cache keys.
    cache["y_t"] = UY
    cache["resid"] = UY
    cache["sigma2"] = float(sigma2)
    cache["dof"] = max(1, n - D)
    return null, cache


def test_predictor_block(
    X_block: np.ndarray,
    U: np.ndarray,
    cache: dict[str, np.ndarray | float | int],
) -> BlockLMMResult:
    """Test predictors using pyseer/FaST-LMM-style nLLeval quantities.

    The key correction versus v0.2.x is that var_beta is based on the
    predictor-included residual variance:

        sigma2_alt = (yKy - xKy^2/xKx) / n_eff
        var_beta   = sigma2_alt / xKx

    not on the null-model sigma2. This matches the calculation pattern used by
    pyseer's LMM block tests.
    """
    Xb = _as_2d(X_block)
    n, k = Xb.shape
    Q = np.asarray(cache["Q"])
    UY = np.asarray(cache["UY"]).reshape(-1)
    w = np.asarray(cache["w"]).reshape(-1)
    n_eff = int(cache["n_eff"])
    dof_test = int(cache["dof_test"])

    UX = _rotate_residualised(Xb, U, Q)
    xKx = np.sum(UX * (w[:, None] * UX), axis=0)
    xKy = UX.T @ (w * UY)
    yKy = float(np.dot(UY, w * UY))

    beta = np.full(k, np.nan, dtype=np.float64)
    se = np.full(k, np.nan, dtype=np.float64)
    p = np.full(k, np.nan, dtype=np.float64)
    F_arr = np.full(k, np.nan, dtype=np.float64)
    status: list[str] = ["OK"] * k

    for j in range(k):
        col = Xb[:, j]
        if np.min(col) == np.max(col):
            status[j] = "NO_VARIATION_PREDICTOR"
            continue
        if not np.isfinite(xKx[j]) or xKx[j] <= 1e-12:
            status[j] = "SINGULAR_PREDICTOR"
            continue
        b = float(xKy[j] / xKx[j])
        # pyseer/FaST-LMM's nLLeval returns variance_beta from the model that
        # includes the tested SNP; this is essential for matching pyseer.
        resid_quad = float(yKy - (xKy[j] * xKy[j]) / xKx[j])
        if not np.isfinite(resid_quad) or resid_quad <= 0:
            status[j] = "BAD_ALT_SIGMA2"
            continue
        sigma2_alt = resid_quad / max(1, n_eff)
        var_beta = sigma2_alt / xKx[j]
        if not np.isfinite(var_beta) or var_beta <= 0:
            status[j] = "BAD_VAR_BETA"
            continue
        beta[j] = b
        se[j] = math.sqrt(var_beta)
        F = (b * b) / var_beta
        F_arr[j] = float(F)
        p[j] = float(stats.f.sf(F, 1, max(1, dof_test)))
        if not np.isfinite(p[j]):
            status[j] = "BAD_PVALUE"
    return BlockLMMResult(p=p, beta=beta, se=se, status=status, xKx=xKx, xKy=xKy, F=F_arr)


def neglog10(p: float) -> float:
    if p is None or not np.isfinite(p):
        return np.nan
    if p <= 0:
        return 300.0
    return -math.log10(max(float(p), 1e-300))

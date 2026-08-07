from __future__ import annotations

"""Directional binary mixed-model core for KOVAR.

The implementation uses a one-variance-component penalised quasi-likelihood
(PQL) fit and the prospective score-test construction used by GMMAT.  It is a
standalone NumPy/SciPy implementation, not a wrapper around or exact numerical
reproduction of GMMAT.
"""

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np
from scipy import linalg, optimize, stats
from scipy.special import expit


_TAU_GRID_MIN = -10.0
_TAU_GRID_MAX = 8.0
_TAU_ABSOLUTE_MAX = 1.0e10


@dataclass
class KinshipDiagnostics:
    rank: int
    eigen_min: float
    eigen_max: float
    roundoff_correction: float
    mean_diagonal: float


@dataclass
class GLMMFit:
    status: str
    converged: bool
    beta: np.ndarray
    tau: float
    latent_phylogenetic_fraction: float
    fitted_probability: np.ndarray
    random_effect: np.ndarray
    iterations: int
    objective: float
    max_eta_change: float
    min_weight: float
    n_weights_clipped: int
    tau_boundary: bool
    message: str = ""


@dataclass
class ScoreResult:
    score_u: float = np.nan
    score_variance: float = np.nan
    score_z: float = np.nan
    p_score: float = np.nan
    beta_score: float = np.nan
    se_score: float = np.nan
    status: str = "NOT_TESTED"
    adjusted_predictor: np.ndarray | None = None


@dataclass
class FullFitResult:
    status: str
    attempted: bool = True
    beta: float = np.nan
    se: float = np.nan
    odds_ratio: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    tau: float = np.nan
    wald_p: float = np.nan
    iterations: int = 0
    converged: bool = False


def prepare_kinship(K: np.ndarray, symmetry_tolerance: float = 1e-8) -> tuple[np.ndarray, KinshipDiagnostics]:
    """Validate, normalise and roundoff-correct a sample covariance matrix.

    Singular positive-semidefinite matrices are valid.  Only negative
    eigenvalues compatible with floating-point roundoff are clipped; a
    materially indefinite matrix is rejected rather than scientifically
    altered with shrinkage.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("Kinship matrix must be square")
    if K.shape[0] < 2:
        raise ValueError("Kinship matrix must contain at least two samples")
    if not np.all(np.isfinite(K)):
        raise ValueError("Kinship matrix contains non-finite values")
    scale = max(1.0, float(np.max(np.abs(K))))
    asymmetry = float(np.max(np.abs(K - K.T)))
    if asymmetry > symmetry_tolerance * scale:
        raise ValueError(f"Kinship matrix is materially asymmetric (max difference {asymmetry:.3g})")
    K = 0.5 * (K + K.T)
    if float(np.max(np.abs(K))) <= 1e-14:
        return K, KinshipDiagnostics(
            rank=0,
            eigen_min=0.0,
            eigen_max=0.0,
            roundoff_correction=0.0,
            mean_diagonal=0.0,
        )
    mean_diag = float(np.mean(np.diag(K)))
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        raise ValueError("Kinship matrix must have a positive finite mean diagonal")
    K = K / mean_diag

    eigvals, eigvecs = np.linalg.eigh(K)
    eig_scale = max(1.0, float(np.max(np.abs(eigvals))))
    reject_floor = -1e-7 * eig_scale
    if float(eigvals[0]) < reject_floor:
        raise ValueError(
            f"Kinship matrix is not positive semidefinite (minimum eigenvalue {eigvals[0]:.6g})"
        )
    correction = float(max(0.0, -float(eigvals[0])))
    if correction > 0:
        eigvals = np.clip(eigvals, 0.0, None)
        K = (eigvecs * eigvals) @ eigvecs.T
        K = 0.5 * (K + K.T)
    rank_tol = max(1e-10, float(np.max(eigvals)) * 1e-10)
    diagnostics = KinshipDiagnostics(
        rank=int(np.count_nonzero(eigvals > rank_tol)),
        eigen_min=float(np.min(eigvals)),
        eigen_max=float(np.max(eigvals)),
        roundoff_correction=correction,
        mean_diagonal=mean_diag,
    )
    return K, diagnostics


def _fixed_design(n: int, covariates: Optional[np.ndarray]) -> np.ndarray:
    if covariates is None:
        C = np.ones((n, 1), dtype=np.float64)
    else:
        A = np.asarray(covariates, dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        if A.ndim != 2 or A.shape[0] != n:
            raise ValueError("Fixed-effect covariates must have one row per sample")
        if not np.all(np.isfinite(A)):
            raise ValueError("Fixed-effect covariates contain non-finite values")
        if A.shape[1] == 0 or not np.any(np.std(A, axis=0) <= 1e-12):
            C = np.column_stack([np.ones(n, dtype=np.float64), A])
        else:
            C = A
    if np.linalg.matrix_rank(C, tol=1e-10) != C.shape[1]:
        raise ValueError("Fixed-effect design matrix is rank deficient")
    return C


@dataclass
class _WeightedEigen:
    sqrt_w: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    logdet_w_inverse: float

    def solve(self, A: np.ndarray, tau: float) -> np.ndarray:
        arr = np.asarray(A, dtype=np.float64)
        was_vector = arr.ndim == 1
        if was_vector:
            arr = arr.reshape(-1, 1)
        weighted = self.sqrt_w[:, None] * arr
        rotated = self.eigenvectors.T @ weighted
        denom = 1.0 + float(tau) * self.eigenvalues
        solved = self.sqrt_w[:, None] * (self.eigenvectors @ (rotated / denom[:, None]))
        return solved[:, 0] if was_vector else solved

    def logdet(self, tau: float) -> float:
        return float(self.logdet_w_inverse + np.sum(np.log1p(float(tau) * self.eigenvalues)))


def _weighted_eigen(K: np.ndarray, weights: np.ndarray) -> _WeightedEigen:
    sqrt_w = np.sqrt(weights)
    B = (sqrt_w[:, None] * K) * sqrt_w[None, :]
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = np.linalg.eigh(B)
    tolerance = max(1e-12, float(np.max(np.abs(eigvals))) * 1e-10)
    if float(eigvals[0]) < -tolerance:
        raise np.linalg.LinAlgError(
            f"Weighted kinship became indefinite (minimum eigenvalue {eigvals[0]:.6g})"
        )
    eigvals = np.clip(eigvals, 0.0, None)
    return _WeightedEigen(
        sqrt_w=sqrt_w,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        logdet_w_inverse=float(-np.sum(np.log(weights))),
    )


def _working_solution(
    decomp: _WeightedEigen,
    z: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    tau: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Sinv_C = decomp.solve(C, tau)
    M = C.T @ Sinv_C
    sign, logdet_m = np.linalg.slogdet(M)
    if sign <= 0 or not np.isfinite(logdet_m):
        raise np.linalg.LinAlgError("Fixed-effect information matrix is singular")
    factor = linalg.cho_factor(M, lower=True, check_finite=False)
    M_inv = linalg.cho_solve(
        factor,
        np.eye(M.shape[0], dtype=np.float64),
        check_finite=False,
    )
    Sinv_z = decomp.solve(z, tau)
    beta = linalg.cho_solve(factor, C.T @ Sinv_z, check_finite=False)
    residual = z - C @ beta
    Sinv_residual = decomp.solve(residual, tau)
    quad = float(residual @ Sinv_residual)
    objective = float(decomp.logdet(tau) + logdet_m + quad)
    random_effect = float(tau) * (K @ Sinv_residual)
    return objective, beta, random_effect, Sinv_C, M_inv


def _profile_tau(
    decomp: _WeightedEigen,
    z: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
) -> tuple[float, bool, tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    shared = K - np.diag(np.diag(K))
    if (
        float(np.max(np.abs(K))) <= 1e-14
        or float(np.max(np.abs(shared))) <= 1e-14
        or float(np.max(decomp.eigenvalues)) <= 1e-14
    ):
        # A diagonal-only kernel contains no covariance between samples. With
        # one Bernoulli observation per sample, its variance is not a
        # phylogenetic/relatedness component and is not separately identifiable
        # from individual-level logistic variation.
        solution = _working_solution(decomp, z, C, K, 0.0)
        return 0.0, False, solution

    cache: dict[float, tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def evaluate(tau: float):
        key = float(tau)
        if key not in cache:
            cache[key] = _working_solution(decomp, z, C, K, key)
        return cache[key]

    candidates = [0.0] + list(np.logspace(_TAU_GRID_MIN, _TAU_GRID_MAX, 25))
    values = []
    for tau in candidates:
        try:
            values.append(evaluate(float(tau))[0])
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            values.append(np.inf)
    best = int(np.argmin(values))
    boundary = False
    while best == len(candidates) - 1 and candidates[-1] < _TAU_ABSOLUTE_MAX:
        next_tau = min(_TAU_ABSOLUTE_MAX, candidates[-1] * 100.0)
        candidates.append(float(next_tau))
        try:
            values.append(evaluate(float(next_tau))[0])
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            values.append(np.inf)
        best = int(np.argmin(values))
        if next_tau >= _TAU_ABSOLUTE_MAX:
            boundary = best == len(candidates) - 1
            break

    best_tau = float(candidates[best])
    best_value = float(values[best])
    if best > 0:
        lo = max(1e-14, float(candidates[best - 1]))
        hi = float(candidates[min(best + 1, len(candidates) - 1)])
        if hi > lo:
            try:
                opt = optimize.minimize_scalar(
                    lambda log_tau: evaluate(float(math.exp(log_tau)))[0],
                    bounds=(math.log(lo), math.log(hi)),
                    method="bounded",
                    options={"xatol": 1e-6, "maxiter": 100},
                )
                if opt.success and np.isfinite(opt.fun) and float(opt.fun) < best_value:
                    best_tau = float(math.exp(float(opt.x)))
                    best_value = float(opt.fun)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                pass
    solution = evaluate(best_tau)
    return best_tau, boundary, solution


def fit_logistic_mixed(
    y: np.ndarray,
    K: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tolerance: float = 1e-7,
    weight_floor: float = 1e-8,
    damping: float = 0.8,
) -> tuple[GLMMFit, dict[str, Any]]:
    """Fit a one-kernel logistic mixed model by PQL/pseudo-REML profiling."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    n = y.size
    if K.shape != (n, n):
        raise ValueError("Kinship dimensions do not match the response")
    if not np.all(np.isfinite(y)) or not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("Response must contain only finite binary 0/1 values")
    if int(max_iter) < 2:
        raise ValueError("max_iter must be at least two")
    if not np.isfinite(tolerance) or float(tolerance) <= 0:
        raise ValueError("tolerance must be positive and finite")
    if not np.isfinite(weight_floor) or not 0 < float(weight_floor) < 0.25:
        raise ValueError("weight_floor must be between zero and 0.25")
    if not np.isfinite(damping) or not 0 < float(damping) <= 1:
        raise ValueError("damping must be in (0, 1]")
    C = _fixed_design(n, covariates)
    n_present = int(np.sum(y))
    if n_present == 0 or n_present == n:
        fit = GLMMFit(
            status="MONOMORPHIC_RESPONSE",
            converged=False,
            beta=np.full(C.shape[1], np.nan),
            tau=np.nan,
            latent_phylogenetic_fraction=np.nan,
            fitted_probability=np.full(n, np.nan),
            random_effect=np.full(n, np.nan),
            iterations=0,
            objective=np.nan,
            max_eta_change=np.nan,
            min_weight=np.nan,
            n_weights_clipped=0,
            tau_boundary=False,
        )
        return fit, {}

    prevalence = (n_present + 0.5) / (n + 1.0)
    eta = np.full(n, math.log(prevalence / (1.0 - prevalence)), dtype=np.float64)
    tau_previous = 0.0
    consecutive = 0
    last: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    last_decomp: _WeightedEigen | None = None
    clipped_count = 0
    max_change = np.inf
    tau_boundary = False
    status = "PQL_NOT_CONVERGED"
    message = "Maximum PQL iterations reached"

    try:
        for iteration in range(1, max(2, int(max_iter)) + 1):
            mu = np.clip(expit(eta), 1e-12, 1.0 - 1e-12)
            raw_w = mu * (1.0 - mu)
            clipped_count = int(np.count_nonzero(raw_w < weight_floor))
            weights = np.maximum(raw_w, float(weight_floor))
            z = eta + (y - mu) / weights
            decomp = _weighted_eigen(K, weights)
            tau, boundary, solution = _profile_tau(decomp, z, C, K)
            objective, beta, random_effect, Sinv_C, M_inv = solution
            eta_target = C @ beta + random_effect
            max_change = float(np.max(np.abs(eta_target - eta)))
            delta = np.clip(eta_target - eta, -5.0, 5.0)
            eta_new = eta + float(damping) * delta
            tau_change = abs(math.log1p(tau) - math.log1p(tau_previous))
            tau_boundary = bool(boundary)
            last = solution
            last_decomp = decomp
            if max_change <= tolerance and tau_change <= math.sqrt(tolerance):
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= 2:
                status = "OK"
                message = ""
                break
            eta = eta_new
            tau_previous = tau
        else:
            iteration = max(2, int(max_iter))

        if status == "OK" and boundary:
            status = "TAU_SEARCH_BOUNDARY"
            message = "Phylogenetic variance reached the numerical search boundary"
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        fit = GLMMFit(
            status="PQL_NUMERICAL_FAILURE",
            converged=False,
            beta=np.full(C.shape[1], np.nan),
            tau=np.nan,
            latent_phylogenetic_fraction=np.nan,
            fitted_probability=np.full(n, np.nan),
            random_effect=np.full(n, np.nan),
            iterations=iteration if "iteration" in locals() else 0,
            objective=np.nan,
            max_eta_change=np.nan,
            min_weight=np.nan,
            n_weights_clipped=clipped_count,
            tau_boundary=False,
            message=str(exc),
        )
        return fit, {}

    converged = status == "OK"
    fit = GLMMFit(
        status=status,
        converged=converged,
        beta=np.asarray(beta, dtype=np.float64),
        tau=float(tau),
        latent_phylogenetic_fraction=float(tau / (tau + math.pi * math.pi / 3.0)),
        fitted_probability=mu,
        random_effect=np.asarray(random_effect, dtype=np.float64),
        iterations=int(iteration),
        objective=float(objective),
        max_eta_change=max_change,
        min_weight=float(np.min(weights)),
        n_weights_clipped=clipped_count,
        tau_boundary=bool(boundary),
        message=message,
    )
    cache: dict[str, Any] = {
        "y": y,
        "K": K,
        "C": C,
        "mu": mu,
        "weights": weights,
        "tau": float(tau),
        "decomp": last_decomp,
        "Sinv_C": Sinv_C,
        "M_inv": M_inv,
    }
    return fit, cache


def fit_null_glmm(
    response: int,
    y: np.ndarray,
    K: np.ndarray,
    **kwargs,
) -> tuple[GLMMFit, dict[str, Any]]:
    fit, cache = fit_logistic_mixed(y, K, covariates=None, **kwargs)
    if cache:
        cache["response"] = int(response)
    return fit, cache


def score_predictor_block(
    predictors: np.ndarray,
    cache: dict[str, Any],
) -> list[ScoreResult]:
    """Score a sample-by-predictor block with shared mixed-model solves."""
    if not cache:
        array = np.asarray(predictors)
        width = 1 if array.ndim == 1 else array.shape[1]
        return [ScoreResult(status="NULL_MODEL_UNAVAILABLE") for _ in range(width)]
    X = np.asarray(predictors, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("Predictor block must be one- or two-dimensional")
    y = np.asarray(cache["y"])
    if X.shape[0] != y.size:
        raise ValueError("Predictor block row count does not match response")
    results = [ScoreResult() for _ in range(X.shape[1])]
    finite = np.all(np.isfinite(X), axis=0)
    variable = finite & (np.ptp(X, axis=0) > 0)
    for index in np.flatnonzero(~finite):
        results[int(index)] = ScoreResult(status="NONFINITE_PREDICTOR")
    for index in np.flatnonzero(finite & ~variable):
        results[int(index)] = ScoreResult(status="MONOMORPHIC_PREDICTOR")
    valid_indices = np.flatnonzero(variable)
    if not valid_indices.size:
        return results
    block = X[:, valid_indices]

    decomp: _WeightedEigen = cache["decomp"]
    tau = float(cache["tau"])
    C = np.asarray(cache["C"])
    Sinv_C = np.asarray(cache["Sinv_C"])
    M_inv = np.asarray(cache["M_inv"])
    Sinv_X = decomp.solve(block, tau)
    coefficient = M_inv @ (C.T @ Sinv_X)
    PX = Sinv_X - Sinv_C @ coefficient
    variances = np.sum(block * PX, axis=0)
    residual = y - np.asarray(cache["mu"])
    scores = block.T @ residual
    weights = np.asarray(cache["weights"], dtype=np.float64)
    weighted_information = C.T @ (weights[:, None] * C)
    weighted_factor = linalg.cho_factor(
        weighted_information, lower=True, check_finite=False
    )
    weighted_coefficient = linalg.cho_solve(
        weighted_factor,
        C.T @ (weights[:, None] * block),
        check_finite=False,
    )
    spa_block = block - C @ weighted_coefficient

    for local, original in enumerate(valid_indices):
        variance = float(variances[local])
        if not np.isfinite(variance) or variance <= 1e-12:
            results[int(original)] = ScoreResult(
                status="SINGULAR_PREDICTOR_AFTER_KINSHIP"
            )
            continue
        score = float(scores[local])
        z = float(score / math.sqrt(variance))
        p = float(stats.chi2.sf(z * z, 1))
        if not np.isfinite(p):
            results[int(original)] = ScoreResult(status="BAD_SCORE_PVALUE")
            continue
        results[int(original)] = ScoreResult(
            score_u=score,
            score_variance=variance,
            score_z=z,
            p_score=p,
            beta_score=float(score / variance),
            se_score=float(1.0 / math.sqrt(variance)),
            status="OK",
            # SPA conditions on independent Bernoulli working means. Its
            # cumulant weights use W-residualization; the mixed-model target
            # variance is reconciled through SPA's variance ratio.
            adjusted_predictor=spa_block[:, local],
        )
    return results


def score_predictor(predictor: np.ndarray, cache: dict[str, Any]) -> ScoreResult:
    """Score one predictor; retained as a small public convenience wrapper."""
    return score_predictor_block(np.asarray(predictor).reshape(-1, 1), cache)[0]


def fit_full_glmm(
    y: np.ndarray,
    predictor: np.ndarray,
    K: np.ndarray,
    **kwargs,
) -> FullFitResult:
    x = np.asarray(predictor, dtype=np.float64).reshape(-1)
    try:
        fit, cache = fit_logistic_mixed(y, K, covariates=x, **kwargs)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return FullFitResult(status=f"FULL_FIT_ERROR:{exc}")
    if fit.status != "OK" or not cache:
        return FullFitResult(
            status=f"FULL_{fit.status}",
            tau=fit.tau,
            iterations=fit.iterations,
            converged=fit.converged,
        )
    beta = float(fit.beta[-1])
    covariance = np.asarray(cache["M_inv"])
    variance = float(covariance[-1, -1])
    if not np.isfinite(variance) or variance <= 0:
        return FullFitResult(status="FULL_BAD_INFORMATION", tau=fit.tau, iterations=fit.iterations, converged=True)
    se = math.sqrt(variance)
    if not np.isfinite(beta) or abs(beta) > 20.0 or not np.isfinite(se) or se > 20.0:
        return FullFitResult(
            status="FULL_SEPARATION_OR_UNSTABLE",
            beta=beta,
            se=se,
            tau=fit.tau,
            iterations=fit.iterations,
            converged=True,
        )
    z = beta / se
    p = float(2.0 * stats.norm.sf(abs(z)))
    low = beta - 1.959963984540054 * se
    high = beta + 1.959963984540054 * se
    return FullFitResult(
        status="OK",
        beta=beta,
        se=se,
        odds_ratio=float(math.exp(beta)),
        ci_low=float(math.exp(low)),
        ci_high=float(math.exp(high)),
        tau=fit.tau,
        wald_p=p,
        iterations=fit.iterations,
        converged=True,
    )


def neglog10(p: float) -> float:
    if not np.isfinite(p):
        return np.nan
    return float(-math.log10(max(float(p), 1e-300)))

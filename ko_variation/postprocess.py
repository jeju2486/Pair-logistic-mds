"""Bonferroni/distance selection and selected-pair alternative PQL fits."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm
from .glmm import fit_logistic_mixed, prepare_kinship_eigensystem


@dataclass(frozen=True)
class SelectionConfig:
    significance_threshold: float = 0.05
    distance_column: str = "distance"
    ld_distance: float = 0
    cross_contig: str = "exclude"


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")


def _numeric(frame: pd.DataFrame, columns: list[str], *, integer=False) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must contain finite nonnegative values")
        if integer and (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain integers")
        frame[column] = values


def validate_annotation(annotation: pd.DataFrame) -> pd.DataFrame:
    """One mapping per zero-based locus; gene IDs must identify unique genes."""
    _require(annotation, ["locus", "gene"])
    ann = annotation.copy()
    _numeric(ann, ["locus"], integer=True)
    if ann.locus.duplicated().any():
        raise ValueError("Annotation contains duplicate locus IDs")
    if ann.gene.isna().any() or ann.gene.astype(str).str.strip().eq("").any():
        raise ValueError("Annotation gene IDs must be nonempty")
    ann["gene"] = ann.gene.astype(str)
    return ann


def annotate_pairs(pairs: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    ann = validate_annotation(annotation).set_index("locus")
    result = pairs.copy()
    for side in ("u", "v"):
        for column in ann.columns:
            result[f"{side}_{column}"] = result[side].map(ann[column])
    return result


def select_distal_signals(results: pd.DataFrame, config: SelectionConfig | None = None,
                          annotation: pd.DataFrame | None = None) -> pd.DataFrame:
    """Select p_primary <= Bonferroni alpha/n_tests and distance > cutoff.

    Distance must be strictly greater than the cutoff; significance is inclusive.
    A distance column takes priority; otherwise use annotation contig/position.
    Missing distances are excluded. Distances are physical units, not measured LD.
    """
    cfg = config or SelectionConfig()
    if not np.isfinite(cfg.significance_threshold) or not 0 <= cfg.significance_threshold <= 1:
        raise ValueError("Significance threshold must lie in [0, 1]")
    if not np.isfinite(cfg.ld_distance) or cfg.ld_distance < 0:
        raise ValueError("LD distance must be finite and nonnegative")
    if cfg.cross_contig not in {"exclude", "distal"}:
        raise ValueError("cross_contig must be exclude or distal")
    _require(results, ["u", "v", "status", "p_primary", "n11", "n10", "n01", "n00"])
    data = results.copy()
    _numeric(data, ["u", "v", "n11", "n10", "n01", "n00"], integer=True)
    if (data.u >= data.v).any() or data.duplicated(["u", "v"]).any():
        raise ValueError("Expected unique canonical unordered pairs u < v")
    counts = data[["n11", "n10", "n01", "n00"]].to_numpy(float)
    if (counts.sum(axis=1) == 0).any():
        raise ValueError("Joint tables must contain observations")
    significance = pd.to_numeric(data["p_primary"], errors="raise")
    if ((significance.dropna() < 0) | (significance.dropna() > 1)).any():
        raise ValueError("Significance values must lie in [0, 1] or be missing")
    n_tested = int(significance.notna().sum())
    if "n_tests" in data and len(data):
        _numeric(data, ["n_tests"], integer=True)
        if data.n_tests.nunique() != 1 or int(data.n_tests.iloc[0]) < n_tested:
            raise ValueError("n_tests must consistently describe the original full scan")
        n_tested = int(data.n_tests.iloc[0])
    threshold = cfg.significance_threshold / n_tested if n_tested else 0.0
    if annotation is not None:
        data = annotate_pairs(data, annotation)
    cross = pd.Series(False, index=data.index)
    if {"u_contig", "v_contig"} <= set(data):
        known = data.u_contig.notna() & data.v_contig.notna()
        cross = known & data.u_contig.ne(data.v_contig)
    if cfg.distance_column in data:
        distance = pd.to_numeric(data[cfg.distance_column], errors="raise")
        source = cfg.distance_column
    else:
        _require(data, ["u_contig", "v_contig", "u_position", "v_position"])
        for side in ("u", "v"):
            position = pd.to_numeric(data[f"{side}_position"], errors="raise")
            if (position.dropna() < 0).any() or not np.isfinite(position.dropna()).all():
                raise ValueError("Positions must be finite and nonnegative")
            data[f"{side}_position"] = position
        distance = (data.u_position - data.v_position).abs()
        distance = distance.where(data.u_contig.notna() & data.v_contig.notna() & ~cross)
        source = "annotation_linear_positions"
    if (distance.dropna() < 0).any() or not np.isfinite(distance.dropna()).all():
        raise ValueError("Distance must be finite nonnegative or missing")
    distal = distance.gt(cfg.ld_distance) & ~cross
    if cfg.cross_contig == "distal":
        distal |= cross
    keep = data.status.isin(["OK", "OK_SPA_FAILED"]) & significance.le(threshold) & distal
    data["physical_distance"] = distance.where(~cross)
    data["distance_source"] = source
    data["distance_class"] = np.where(cross, "cross_contig", "distal")
    data["selection_significance_column"] = "p_primary"
    data["selection_bonferroni_alpha"] = cfg.significance_threshold
    data["selection_n_tests"] = n_tested
    data["selection_threshold"] = threshold
    data["ld_distance_cutoff"] = cfg.ld_distance
    return data.loc[keep].reset_index(drop=True)


def fit_selected_effects(signals: pd.DataFrame, X: np.ndarray, K: np.ndarray,
                         *, confidence: float = 0.95) -> pd.DataFrame:
    """Fit u -> v only for selected rows; SE is final working-PQL covariance.

    K must be the prepared covariance, aligned to X as in the scanner. The Wald
    interval treats fitted tau as fixed and does not correct post-selection bias.
    """
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie in (0, 1)")
    _require(signals, ["u", "v", "n11", "n10", "n01", "n00"])
    if X.ndim != 2 or not np.isin(X, [0, 1]).all() or (len(signals) and K.shape != (len(X), len(X))):
        raise ValueError("Expected binary sample-by-locus matrix and aligned sample covariance")
    result = signals.copy()
    _numeric(result, ["u", "v"], integer=True)
    if (result.u >= result.v).any() or (result.v >= X.shape[1]).any():
        raise ValueError("Selected locus IDs do not match genotype columns")
    if "n_samples" in result and not result.n_samples.eq(len(X)).all():
        raise ValueError("Sample count differs from original scan")
    # Check every selected table before performing expensive fits.
    tables = []
    for row in result.itertuples():
        x, y = X[:, int(row.u)], X[:, int(row.v)]
        table = [int(np.count_nonzero((x == a) & (y == b))) for a, b in [(1, 1), (1, 0), (0, 1), (0, 0)]]
        if table != [row.n11, row.n10, row.n01, row.n00]:
            raise ValueError(f"Genotype joint counts differ from scan for pair {row.u}, {row.v}")
        tables.append(table)
    for column in ("adjusted_beta", "adjusted_beta_se", "adjusted_odds_ratio", "adjusted_or_ci_low", "adjusted_or_ci_high", "alternative_tau"):
        result[column] = np.nan
    result["effect_status"] = "NOT_FITTED"
    result["effect_iterations"] = 0
    result["effect_message"] = ""
    result["effect_method"] = "alternative_logistic_mixed_pql"
    result["effect_confidence"] = confidence
    if result.empty:
        return result
    eigensystem = prepare_kinship_eigensystem(K)
    z = norm.ppf((1 + confidence) / 2)
    for index, row in enumerate(result.itertuples()):
        if min(tables[index]) == 0:
            result.loc[result.index[index], "effect_status"] = "SEPARATION_OR_MONOMORPHIC"
            continue
        fit, cache = fit_logistic_mixed(X[:, int(row.v)], K,
                                        covariates=X[:, int(row.u)].reshape(-1, 1),
                                        kinship_eigensystem=eigensystem)
        idx = result.index[index]
        result.loc[idx, ["effect_status", "effect_iterations", "effect_message", "alternative_tau"]] = [fit.status, fit.iterations, fit.message, fit.tau]
        covariance = cache.get("M_inv")
        del cache
        if not fit.converged:
            continue
        beta = float(fit.beta[1])
        variance = float(covariance[1, 1])
        if fit.n_weights_clipped or not np.isfinite([beta, variance]).all() or variance <= 0:
            result.loc[idx, "effect_status"] = "UNSTABLE_EFFECT"
            continue
        se = np.sqrt(variance)
        with np.errstate(over="ignore", under="ignore"):
            odds = np.exp([beta, beta - z * se, beta + z * se])
        if not np.isfinite(odds).all() or (odds == 0).any():
            result.loc[idx, "effect_status"] = "UNSTABLE_EFFECT"
            continue
        result.loc[idx, ["adjusted_beta", "adjusted_beta_se", "adjusted_odds_ratio", "adjusted_or_ci_low", "adjusted_or_ci_high"]] = [beta, se, *odds]
    return result

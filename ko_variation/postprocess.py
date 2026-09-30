"""Downstream selection and descriptive effects; never modifies scanner output."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class SelectionConfig:
    significance_column: str = "q_bh"
    significance_threshold: float = 0.05
    distance_column: str = "distance"
    ld_distance: float = 10000
    cross_contig: str = "exclude"
    zero_cell_correction: float = 0.5
    confidence: float = 0.95


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
    """Select successful significant rows and append RAW table ORs and Wald CIs.

    Distance must be strictly greater than the cutoff; significance is inclusive.
    A distance column takes priority; otherwise use annotation contig/position.
    Missing distances are excluded. Distances are physical units, not measured LD.
    """
    cfg = config or SelectionConfig()
    if cfg.significance_column not in {"p_primary", "q_bh", "p_score", "p_spa"}:
        raise ValueError("Choose p_primary, q_bh, p_score or p_spa")
    if not np.isfinite(cfg.significance_threshold) or not 0 <= cfg.significance_threshold <= 1:
        raise ValueError("Significance threshold must lie in [0, 1]")
    if not np.isfinite(cfg.ld_distance) or cfg.ld_distance < 0:
        raise ValueError("LD distance must be finite and nonnegative")
    if cfg.cross_contig not in {"exclude", "distal"}:
        raise ValueError("cross_contig must be exclude or distal")
    if not np.isfinite(cfg.zero_cell_correction) or cfg.zero_cell_correction < 0:
        raise ValueError("Zero-cell correction must be finite and nonnegative")
    if not 0 < cfg.confidence < 1:
        raise ValueError("Confidence must lie in (0, 1)")
    _require(results, ["u", "v", "status", cfg.significance_column, "n11", "n10", "n01", "n00"])
    data = results.copy()
    _numeric(data, ["u", "v", "n11", "n10", "n01", "n00"], integer=True)
    if (data.u >= data.v).any() or data.duplicated(["u", "v"]).any():
        raise ValueError("Expected unique canonical unordered pairs u < v")
    counts = data[["n11", "n10", "n01", "n00"]].to_numpy(float)
    if (counts.sum(axis=1) == 0).any():
        raise ValueError("Joint tables must contain observations")
    significance = pd.to_numeric(data[cfg.significance_column], errors="raise")
    if ((significance.dropna() < 0) | (significance.dropna() > 1)).any():
        raise ValueError("Significance values must lie in [0, 1] or be missing")
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
    keep = data.status.isin(["OK", "OK_SPA_FAILED"]) & significance.le(cfg.significance_threshold) & distal
    data["physical_distance"] = distance.where(~cross)
    data["distance_source"] = source
    data["distance_class"] = np.where(cross, "cross_contig", "distal")
    data["selection_significance_column"] = cfg.significance_column
    data["selection_threshold"] = cfg.significance_threshold
    data["ld_distance_cutoff"] = cfg.ld_distance
    data = data.loc[keep].copy()
    cells = data[["n11", "n10", "n01", "n00"]].to_numpy(float, copy=True)
    corrected = (cells == 0).any(axis=1) & (cfg.zero_cell_correction > 0)
    cells += corrected[:, None] * cfg.zero_cell_correction
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_or = np.log(cells[:, 0]) + np.log(cells[:, 3]) - np.log(cells[:, 1]) - np.log(cells[:, 2])
        se = np.sqrt((1 / cells).sum(axis=1))
        z = norm.ppf((1 + cfg.confidence) / 2)
        data["raw_odds_ratio"] = np.exp(log_or)
        data["raw_log_odds_ratio"] = log_or
        data["raw_log_or_se"] = np.where(np.isfinite(se), se, np.nan)
        data["raw_or_ci_low"] = np.where(np.isfinite(se), np.exp(log_or - z * se), np.nan)
        data["raw_or_ci_high"] = np.where(np.isfinite(se), np.exp(log_or + z * se), np.nan)
    data["raw_or_correction"] = corrected * cfg.zero_cell_correction
    data["raw_or_confidence"] = cfg.confidence
    data["effect_method"] = "raw_contingency_table_unadjusted"
    return data.reset_index(drop=True)

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class KinshipResult:
    """A normalized sample covariance and provenance for its construction."""

    K: np.ndarray
    n_loci_used: int = 0
    n_loci_filtered: int = 0
    mean_diag_before_norm: float = float("nan")
    source: str = "unknown"
    details: dict[str, str | int | float] = field(default_factory=dict)


@dataclass
class _Node:
    name: Optional[str] = None
    length: Optional[float] = None
    children: list["_Node"] = field(default_factory=list)
    parent: Optional["_Node"] = None
    depth: float = 0.0
    tip_index: Optional[int] = None
    desc_tips: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return not self.children


class _NewickParser:
    """Small standalone Newick parser for branch-length covariance trees."""

    def __init__(self, text: str):
        self.text = text.strip()
        self.i = 0
        self.n = len(self.text)

    def parse(self) -> _Node:
        self._skip_ws()
        root = self._parse_subtree()
        self._skip_ws()
        if self.i < self.n and self.text[self.i] == ";":
            self.i += 1
        self._skip_ws()
        if self.i != self.n:
            raise ValueError(
                f"Unexpected trailing Newick text near position {self.i}: "
                f"{self.text[self.i:self.i + 40]!r}"
            )
        root.parent = None
        return root

    def _skip_ws(self) -> None:
        while self.i < self.n and self.text[self.i].isspace():
            self.i += 1

    def _parse_subtree(self) -> _Node:
        self._skip_ws()
        if self.i >= self.n:
            raise ValueError("Unexpected end of Newick string")
        if self.text[self.i] != "(":
            return _Node(
                name=self._parse_label_required(),
                length=self._parse_length_optional(),
            )

        self.i += 1
        children: list[_Node] = []
        while True:
            children.append(self._parse_subtree())
            self._skip_ws()
            if self.i >= self.n:
                raise ValueError("Unclosed internal node in Newick string")
            if self.text[self.i] == ",":
                self.i += 1
                continue
            if self.text[self.i] == ")":
                self.i += 1
                break
            raise ValueError(
                f"Expected ',' or ')' at position {self.i}, "
                f"found {self.text[self.i]!r}"
            )
        node = _Node(
            name=self._parse_label_optional(),
            length=self._parse_length_optional(),
            children=children,
        )
        for child in children:
            child.parent = node
        return node

    def _parse_label_required(self) -> str:
        label = self._parse_label_optional()
        if not label:
            raise ValueError(f"Expected leaf label at position {self.i}")
        return label

    def _parse_label_optional(self) -> Optional[str]:
        self._skip_ws()
        if self.i >= self.n or self.text[self.i] in ":,();":
            return None
        if self.text[self.i] in "'\"":
            quote = self.text[self.i]
            self.i += 1
            out: list[str] = []
            while self.i < self.n:
                ch = self.text[self.i]
                self.i += 1
                if ch == quote:
                    if self.i < self.n and self.text[self.i] == quote:
                        out.append(quote)
                        self.i += 1
                        continue
                    return "".join(out)
                out.append(ch)
            raise ValueError("Unclosed quoted label in Newick string")
        start = self.i
        while self.i < self.n and self.text[self.i] not in ":,();":
            self.i += 1
        return self.text[start:self.i].strip()

    def _parse_length_optional(self) -> Optional[float]:
        self._skip_ws()
        if self.i >= self.n or self.text[self.i] != ":":
            return None
        self.i += 1
        self._skip_ws()
        start = self.i
        while self.i < self.n and self.text[self.i] not in ",();":
            self.i += 1
        raw = self.text[start:self.i].strip()
        if not raw:
            raise ValueError("Empty branch length in Newick string")
        try:
            length = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid branch length {raw!r}") from exc
        if not np.isfinite(length) or length < 0:
            raise ValueError(f"Branch lengths must be finite and nonnegative: {raw!r}")
        return length


def _walk_preorder(node: _Node):
    yield node
    for child in node.children:
        yield from _walk_preorder(child)


def _walk_postorder(node: _Node):
    for child in node.children:
        yield from _walk_postorder(child)
    yield node


def _assign_depths(root: _Node, missing_length: str) -> None:
    if missing_length not in {"error", "one", "zero"}:
        raise ValueError("missing_length must be one of: error, one, zero")

    def edge_length(node: _Node) -> float:
        if node.parent is None:
            return 0.0
        if node.length is None:
            if missing_length == "error":
                raise ValueError(
                    f"Missing branch length for node {node.name or '<internal>'!r}. "
                    "KOVAR requires branch lengths unless an explicit fallback is chosen."
                )
            return 1.0 if missing_length == "one" else 0.0
        return float(node.length)

    root.depth = 0.0
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            child.depth = node.depth + edge_length(child)
            stack.append(child)


def build_tree_covariance(
    tree_path: str | Path,
    sample_names: list[str],
    dtype: str = "float64",
    missing_length: str = "error",
) -> KinshipResult:
    """Build root-to-MRCA covariance in FASTA sample order.

    This follows the tree-similarity construction used by pyseer's
    ``phylogeny_distance.py --lmm``: an off-diagonal element is the root-to-MRCA
    branch length, and a diagonal element is the root-to-tip length. The matrix
    is normalized to mean diagonal one before GLMM fitting.
    """
    path = Path(tree_path)
    root = _NewickParser(path.read_text(encoding="utf-8").strip()).parse()
    _assign_depths(root, missing_length)
    tips = [node for node in _walk_preorder(root) if node.is_leaf]
    if not tips:
        raise ValueError(f"No tips found in tree: {path}")
    tip_by_name: dict[str, _Node] = {}
    for tip in tips:
        if not tip.name:
            raise ValueError("Tree contains an unnamed tip")
        if tip.name in tip_by_name:
            raise ValueError(f"Duplicate tip label in tree: {tip.name}")
        tip_by_name[tip.name] = tip

    sample_set = set(sample_names)
    missing = [sample for sample in sample_names if sample not in tip_by_name]
    if missing:
        raise ValueError(
            f"Tree is missing {len(missing)} FASTA samples; examples: {missing[:5]}"
        )
    extra = [name for name in tip_by_name if name not in sample_set]
    if extra:
        sys.stderr.write(
            f"[KOVAR] warning: ignoring {len(extra)} tree tips absent from FASTA; "
            f"examples: {extra[:5]}\n"
        )

    for index, sample in enumerate(sample_names):
        tip_by_name[sample].tip_index = index
    for node in _walk_postorder(root):
        if node.is_leaf:
            node.desc_tips = (
                np.empty(0, dtype=np.int64)
                if node.tip_index is None
                else np.array([node.tip_index], dtype=np.int64)
            )
        else:
            arrays = [
                child.desc_tips
                for child in node.children
                if child.desc_tips is not None and child.desc_tips.size
            ]
            node.desc_tips = (
                np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
            )

    n = len(sample_names)
    K = np.zeros((n, n), dtype=np.float64)
    for sample in sample_names:
        tip = tip_by_name[sample]
        assert tip.tip_index is not None
        K[tip.tip_index, tip.tip_index] = tip.depth
    for node in _walk_preorder(root):
        if node.is_leaf:
            continue
        groups = [
            child.desc_tips
            for child in node.children
            if child.desc_tips is not None and child.desc_tips.size
        ]
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                K[np.ix_(left, right)] = node.depth
                K[np.ix_(right, left)] = node.depth

    K = 0.5 * (K + K.T)
    mean_diag = float(np.mean(np.diag(K)))
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        raise ValueError("Invalid tree covariance scale; check rooting and branch lengths")
    K /= mean_diag
    out_dtype = np.float64 if dtype == "float64" else np.float32
    return KinshipResult(
        K=K.astype(out_dtype, copy=False),
        mean_diag_before_norm=mean_diag,
        source="tree_mrca_covariance",
        details={"tree": str(path), "n_tree_tips": len(tips), "n_samples": n},
    )


def _valid_grm_loci(X: np.ndarray, min_mac: int) -> np.ndarray:
    n = X.shape[0]
    counts = X.sum(axis=0).astype(np.float64)
    mac = np.minimum(counts, n - counts)
    return np.flatnonzero((mac >= min_mac) & (mac > 0)).astype(np.int64)


def identify_proxy_loci_for_targets(
    X: np.ndarray,
    valid_loci: np.ndarray,
    target_loci: np.ndarray,
    r2_threshold: float = 0.8,
    mismatch_threshold: float = 0.02,
    target_chunk_size: int = 256,
    grm_chunk_size: int = 2048,
    progress: bool = True,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Mask candidate loci and close same/complement sample-pattern proxies."""
    n, m = X.shape
    valid_loci = np.asarray(valid_loci, dtype=np.int64)
    targets = np.asarray(
        sorted({int(x) for x in np.asarray(target_loci).reshape(-1) if 0 <= int(x) < m}),
        dtype=np.int64,
    )
    valid_position = {int(locus): i for i, locus in enumerate(valid_loci)}
    target_valid = np.asarray(
        [int(x) for x in targets if int(x) in valid_position], dtype=np.int64
    )
    mask = np.zeros(valid_loci.size, dtype=bool)
    direct = [valid_position[int(x)] for x in target_valid]
    if direct:
        mask[np.asarray(direct, dtype=np.int64)] = True

    sums_valid = X[:, valid_loci].sum(axis=0).astype(np.float64)
    scanned = 0
    started = time.time()
    for target_start in range(0, target_valid.size, target_chunk_size):
        target_block = target_valid[target_start:target_start + target_chunk_size]
        T = X[:, target_block].astype(np.float64, copy=False)
        sum_t = T.sum(axis=0)
        denom_t = sum_t * (n - sum_t)
        for grm_start in range(0, valid_loci.size, grm_chunk_size):
            grm_loci = valid_loci[grm_start:grm_start + grm_chunk_size]
            G = X[:, grm_loci].astype(np.float64, copy=False)
            sum_g = sums_valid[grm_start:grm_start + grm_loci.size]
            n11 = T.T @ G
            n10 = sum_t[:, None] - n11
            n01 = sum_g[None, :] - n11
            n00 = n - n11 - n10 - n01
            proxy = (
                (n10 + n01 <= mismatch_threshold * n)
                | (n11 + n00 <= mismatch_threshold * n)
            )
            if r2_threshold > 0:
                denom = denom_t[:, None] * (sum_g * (n - sum_g))[None, :]
                with np.errstate(invalid="ignore", divide="ignore"):
                    r2 = ((n11 * n00 - n10 * n01) ** 2) / denom
                proxy |= np.isfinite(r2) & (r2 >= r2_threshold)
            mask[grm_start:grm_start + grm_loci.size] |= np.any(proxy, axis=0)
            scanned += int(target_block.size * grm_loci.size)
        if progress:
            sys.stderr.write(
                "[KOVAR] GRM proxy mask "
                f"targets={min(target_start + target_chunk_size, target_valid.size)}/"
                f"{target_valid.size} masked={int(mask.sum())}/{valid_loci.size} "
                f"elapsed={time.time() - started:.1f}s\n"
            )

    return mask, {
        "n_valid_loci_before_mask": int(valid_loci.size),
        "n_target_loci_total": int(targets.size),
        "n_target_loci_valid": int(target_valid.size),
        "n_direct_masked": int(len(direct)),
        "n_proxy_masked_total": int(mask.sum()),
        "n_proxy_masked_beyond_direct": int(mask.sum() - len(set(direct))),
        "n_target_x_valid_pairs_scanned": int(scanned),
        "r2_threshold": float(r2_threshold),
        "mismatch_threshold": float(mismatch_threshold),
    }


def build_background_grm(
    X: np.ndarray,
    target_loci: np.ndarray,
    min_mac: int = 2,
    chunk_size: int = 4096,
    dtype: str = "float64",
    progress: bool = True,
    mask_r2: float = 0.8,
    mask_mismatch: float = 0.02,
    mask_target_chunk_size: int = 256,
    mask_grm_chunk_size: int = 2048,
) -> KinshipResult:
    """Build a background-locus GRM without changing its covariance spectrum.

    All tested candidate loci and their close pattern proxies are excluded to
    reduce proximal contamination. No shrinkage or rank truncation is applied;
    the GLMM accepts a singular positive-semidefinite covariance.
    """
    n, m = X.shape
    if n < 2:
        raise ValueError("At least two samples are required for a GRM")
    if min_mac < 1:
        raise ValueError("min_mac must be at least one")
    valid = _valid_grm_loci(X, int(min_mac))
    mask, details = identify_proxy_loci_for_targets(
        X,
        valid,
        target_loci,
        r2_threshold=float(mask_r2),
        mismatch_threshold=float(mask_mismatch),
        target_chunk_size=int(mask_target_chunk_size),
        grm_chunk_size=int(mask_grm_chunk_size),
        progress=progress,
    )
    used_loci = valid[~mask]
    if not used_loci.size:
        raise ValueError(
            "No background loci remain after candidate/proxy masking. Supply a "
            "branch-length tree, more background loci, or less stringent proxy thresholds."
        )

    K = np.zeros((n, n), dtype=np.float64)
    started = time.time()
    for start in range(0, used_loci.size, int(chunk_size)):
        loci = used_loci[start:start + int(chunk_size)]
        B = X[:, loci].astype(np.float64, copy=False)
        frequencies = B.mean(axis=0)
        Z = (B - frequencies) / np.sqrt(frequencies * (1.0 - frequencies))
        K += Z @ Z.T
        if progress:
            sys.stderr.write(
                f"[KOVAR] GRM loci={min(start + int(chunk_size), used_loci.size)}/"
                f"{used_loci.size} elapsed={time.time() - started:.1f}s\n"
            )
    K /= float(used_loci.size)
    K = 0.5 * (K + K.T)
    mean_diag = float(np.mean(np.diag(K)))
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        raise ValueError("Invalid background GRM diagonal scale")
    K /= mean_diag

    details.update(
        {
            "n_samples": int(n),
            "n_loci": int(m),
            "n_loci_filtered_by_mac": int(m - valid.size),
            "n_loci_removed_by_candidate_or_proxy_mask": int(valid.size - used_loci.size),
            "n_loci_used_after_mask": int(used_loci.size),
        }
    )
    out_dtype = np.float64 if dtype == "float64" else np.float32
    return KinshipResult(
        K=K.astype(out_dtype, copy=False),
        n_loci_used=int(used_loci.size),
        n_loci_filtered=int(m - used_loci.size),
        mean_diag_before_norm=mean_diag,
        source="fake_fasta_background_grm_candidate_proxy_masked",
        details=details,
    )

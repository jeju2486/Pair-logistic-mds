from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MDSResult:
    covariates: np.ndarray
    summary: pd.DataFrame
    source: str


def binary_hamming_distance(X: np.ndarray, max_loci: int = 20000, seed: int = 1, chunk_size: int = 64) -> np.ndarray:
    """Compute sample-sample binary Hamming distance from a samples x loci matrix.

    For large locus sets, subsample loci for structure estimation. This is only used
    when no tree/precomputed distance matrix is supplied.
    """
    n, m = X.shape
    rng = np.random.default_rng(seed)
    if max_loci > 0 and m > max_loci:
        cols = rng.choice(m, size=max_loci, replace=False)
        cols.sort()
        Xs = X[:, cols]
    else:
        Xs = X
    packed = np.packbits(Xs.astype(np.uint8), axis=1)
    lookup = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    D_count = np.zeros((n, n), dtype=np.float64)
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        xor = np.bitwise_xor(packed[s:e, None, :], packed[None, :, :])
        D_count[s:e, :] = lookup[xor].sum(axis=2)
    D = D_count / float(Xs.shape[1])
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def classical_mds(D: np.ndarray, k: int = 10, eps: float = 1e-12) -> tuple[np.ndarray, pd.DataFrame]:
    """Classical MDS/PCoA from a distance matrix."""
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix")
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    B = -0.5 * (J @ (D ** 2) @ J)
    B = 0.5 * (B + B.T)
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    pos = evals > eps
    evals_pos = evals[pos]
    evecs_pos = evecs[:, pos]
    keep = min(k, len(evals_pos))
    if keep == 0:
        coords = np.zeros((n, k), dtype=np.float64)
    else:
        coords = evecs_pos[:, :keep] * np.sqrt(evals_pos[:keep])
        sd = coords.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        coords = (coords - coords.mean(axis=0)) / sd
        if keep < k:
            coords = np.column_stack([coords, np.zeros((n, k - keep), dtype=np.float64)])
    total_pos = float(evals_pos.sum()) if len(evals_pos) else np.nan
    rows = []
    for i, val in enumerate(evals[:max(k, min(20, len(evals)))]):
        frac = (float(val) / total_pos) if np.isfinite(total_pos) and total_pos > 0 and val > 0 else np.nan
        rows.append({"axis": i + 1, "eigenvalue": float(val), "variance_fraction": frac})
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["cumulative_variance_fraction"] = summary["variance_fraction"].fillna(0).cumsum()
    return coords[:, :k], summary


# Optional tree support. Biopython is preferred; a minimal parser is included as fallback.
def tree_patristic_distance(tree_path: str | Path, sample_names: list[str]) -> np.ndarray:
    try:
        from Bio import Phylo  # type: ignore
        tree = Phylo.read(str(tree_path), "newick")
        tips = tree.get_terminals()
        by_name = {t.name: t for t in tips}
        missing = [s for s in sample_names if s not in by_name]
        if missing:
            raise ValueError(f"Tree is missing FASTA samples; first missing: {missing[:5]}")
        ordered = [by_name[s] for s in sample_names]
        n = len(ordered)
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(tree.distance(ordered[i], ordered[j]))
                D[i, j] = D[j, i] = d
        return D
    except ImportError:
        return _tree_patristic_distance_simple(tree_path, sample_names)


class _Node:
    def __init__(self, name: str = "", length: float = 0.0):
        self.name = name
        self.length = length
        self.children: list[_Node] = []
        self.parent: Optional[_Node] = None


def _parse_newick_simple(text: str) -> _Node:
    text = text.strip()
    if text.endswith(";"):
        text = text[:-1]
    idx = 0

    def parse_subtree() -> _Node:
        nonlocal idx
        node = _Node()
        if idx < len(text) and text[idx] == "(":
            idx += 1
            while True:
                child = parse_subtree()
                child.parent = node
                node.children.append(child)
                if idx < len(text) and text[idx] == ",":
                    idx += 1
                    continue
                if idx < len(text) and text[idx] == ")":
                    idx += 1
                    break
        chars = []
        while idx < len(text) and text[idx] not in ":,()":
            chars.append(text[idx]); idx += 1
        node.name = "".join(chars).strip().strip("'\"")
        if idx < len(text) and text[idx] == ":":
            idx += 1
            lchars = []
            while idx < len(text) and text[idx] not in ",()":
                lchars.append(text[idx]); idx += 1
            try:
                node.length = float("".join(lchars))
            except Exception:
                node.length = 0.0
        return node
    return parse_subtree()


def _tree_patristic_distance_simple(tree_path: str | Path, sample_names: list[str]) -> np.ndarray:
    root = _parse_newick_simple(Path(tree_path).read_text())
    paths: dict[str, list[tuple[_Node, float]]] = {}
    def walk(node: _Node, dist: float, trail: list[tuple[_Node, float]]):
        cur = trail + [(node, dist)]
        if not node.children:
            paths[node.name] = cur
        else:
            for child in node.children:
                walk(child, dist + child.length, cur)
    walk(root, 0.0, [])
    missing = [s for s in sample_names if s not in paths]
    if missing:
        raise ValueError(f"Tree is missing FASTA samples; first missing: {missing[:5]}")
    amap = {s: {id(n): d for n, d in paths[s]} for s in sample_names}
    total = {s: paths[s][-1][1] for s in sample_names}
    n = len(sample_names)
    D = np.zeros((n, n), dtype=np.float64)
    for i, si in enumerate(sample_names):
        ai = amap[si]
        for j in range(i + 1, n):
            sj = sample_names[j]
            common = set(ai).intersection(amap[sj])
            mrca = max(ai[x] for x in common) if common else 0.0
            d = total[si] + total[sj] - 2.0 * mrca
            D[i, j] = D[j, i] = d
    return D

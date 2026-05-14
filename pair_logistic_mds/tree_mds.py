from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio import Phylo


def read_tree(path: str | Path):
    return Phylo.read(str(path), "newick")


def get_tip_names(tree) -> list[str]:
    tips = [t.name for t in tree.get_terminals()]
    if any(t is None or t == "" for t in tips):
        raise ValueError("All tree tips must have non-empty names")
    if len(set(tips)) != len(tips):
        raise ValueError("Tree contains duplicated tip names")
    return tips


def _parent_depth_rootdist(tree):
    parent = {}
    depth = {tree.root: 0}
    rootdist = {tree.root: 0.0}
    stack = [tree.root]
    while stack:
        node = stack.pop()
        for child in node.clades:
            parent[child] = node
            depth[child] = depth[node] + 1
            rootdist[child] = rootdist[node] + float(child.branch_length or 0.0)
            stack.append(child)
    return parent, depth, rootdist


def compute_tree_distance_matrix(tree, tip_names: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Compute patristic distance matrix in tree tip order.

    Uses root distances plus lowest common ancestor lookup. This is intended for
    moderate N. For very large N, computing/storing N x N distances may dominate.
    """
    all_tips = tree.get_terminals()
    name_to_tip = {t.name: t for t in all_tips}
    if tip_names is None:
        tip_names = [t.name for t in all_tips]
    tips = [name_to_tip[name] for name in tip_names]

    parent, depth, rootdist = _parent_depth_rootdist(tree)

    ancestor_sets = []
    for tip in tips:
        s = set()
        node = tip
        while True:
            s.add(node)
            if node is tree.root:
                break
            node = parent[node]
        ancestor_sets.append(s)

    n = len(tips)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ti = tips[i]
        ai = ancestor_sets[i]
        for j in range(i + 1, n):
            node = tips[j]
            # Walk from j towards root until reaching an ancestor of i.
            while node not in ai:
                node = parent[node]
            lca = node
            d = rootdist[ti] + rootdist[tips[j]] - 2.0 * rootdist[lca]
            D[i, j] = D[j, i] = max(d, 0.0)
    return tip_names, D


def classical_mds(D: np.ndarray, k: int = 10, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Classical MDS from a distance matrix.

    Returns coordinates N x min(k, n_positive_eigenvalues) and the retained
    positive eigenvalues in descending order.
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix")
    n = D.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples for MDS")
    D2 = D.astype(np.float64) ** 2
    J = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    B = -0.5 * (J @ D2 @ J)
    # Numerical symmetry
    B = 0.5 * (B + B.T)
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    pos = evals > eps
    evals_pos = evals[pos]
    evecs_pos = evecs[:, pos]
    keep = min(k, evals_pos.shape[0])
    if keep == 0:
        return np.zeros((n, 0), dtype=np.float64), np.array([], dtype=np.float64)
    coords = evecs_pos[:, :keep] * np.sqrt(evals_pos[:keep])
    # Standardize axes for numerically stable regression.
    sd = coords.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    coords = (coords - coords.mean(axis=0)) / sd
    return coords, evals_pos[:keep]

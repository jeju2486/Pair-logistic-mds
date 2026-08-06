from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import sys

import numpy as np


@dataclass
class NJBuildResult:
    tree_path: str
    n_samples: int
    n_loci_available: int
    n_loci_used: int
    min_mac: int
    max_loci: int
    distance: str
    midpoint_rooted: int
    nj_negative_branches_clipped: int
    nj_min_distance: float
    nj_max_distance: float
    nj_mean_distance: float

    def details(self) -> dict[str, Any]:
        return {
            "auto_nj_tree_path": self.tree_path,
            "auto_nj_n_samples": self.n_samples,
            "auto_nj_n_loci_available": self.n_loci_available,
            "auto_nj_n_loci_used": self.n_loci_used,
            "auto_nj_min_mac": self.min_mac,
            "auto_nj_max_loci": self.max_loci,
            "auto_nj_distance": self.distance,
            "auto_nj_midpoint_rooted": self.midpoint_rooted,
            "auto_nj_negative_branches_clipped": self.nj_negative_branches_clipped,
            "auto_nj_min_distance": self.nj_min_distance,
            "auto_nj_max_distance": self.nj_max_distance,
            "auto_nj_mean_distance": self.nj_mean_distance,
        }


def _quote_newick_label(label: str) -> str:
    # Keep common FASTA/sample labels unquoted. Quote labels with spaces or Newick punctuation.
    if label and all(ch not in " \t\n\r:,();'\"[]" for ch in label):
        return label
    return "'" + str(label).replace("'", "''") + "'"


def _fmt_len(x: float) -> str:
    if not np.isfinite(x):
        x = 0.0
    if abs(x) < 5e-15:
        x = 0.0
    return f"{float(x):.10g}"


def select_nj_loci(
    X: np.ndarray,
    min_mac: int = 2,
    max_loci: int = 50000,
    seed: int = 1,
) -> tuple[np.ndarray, int]:
    """Return locus indices used for automatic NJ tree construction.

    The default keeps polymorphic loci with minor allele count >= min_mac. If more
    than max_loci pass and max_loci > 0, loci are reproducibly subsampled. This
    prevents accidental very large O(N^2*L) distance builds while still using a
    broad genome-wide background.
    """
    Xb = np.asarray(X)
    n = int(Xb.shape[0])
    counts = np.asarray(Xb.sum(axis=0)).reshape(-1)
    mac = np.minimum(counts, n - counts)
    valid = np.flatnonzero(mac >= int(min_mac))
    n_available = int(valid.size)
    if max_loci is not None and int(max_loci) > 0 and valid.size > int(max_loci):
        rng = np.random.default_rng(int(seed))
        valid = np.sort(rng.choice(valid, size=int(max_loci), replace=False))
    return valid.astype(np.int64), n_available


def hamming_distance_matrix_from_binary(
    X: np.ndarray,
    loci: np.ndarray,
    chunk_size: int = 2048,
    progress: bool = True,
) -> np.ndarray:
    """Compute normalized pairwise Hamming distances between samples.

    X is samples x loci and binary-coded. The result is N x N, symmetric, with
    diagonal zero. Computation is chunked over loci.
    """
    Xb = np.asarray(X)
    loci = np.asarray(loci, dtype=np.int64)
    n = int(Xb.shape[0])
    if loci.size == 0:
        raise ValueError("No loci available for automatic NJ tree construction after filtering")
    D = np.zeros((n, n), dtype=np.float64)
    chunk_size = max(1, int(chunk_size))
    total = int(loci.size)
    t_next = 0
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        B = Xb[:, loci[start:end]].astype(np.float32, copy=False)
        row = np.asarray(B.sum(axis=1), dtype=np.float64)
        shared = np.asarray(B @ B.T, dtype=np.float64)
        D += row[:, None] + row[None, :] - 2.0 * shared
        if progress and start >= t_next:
            sys.stderr.write(f"[auto-nj] hamming chunks {end}/{total}\n")
            sys.stderr.flush()
            t_next = start + max(chunk_size * 10, 1)
    D /= float(total)
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    # Numerical roundoff can make very small negative distances.
    D[D < 0] = 0.0
    return D


def _build_nj_adjacency(
    D_input: np.ndarray,
    sample_names: list[str],
    clip_negative: bool = True,
    progress: bool = True,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, str], int, int]:
    """Build an unrooted NJ tree as an adjacency list.

    Returns adjacency, leaf name map, last root node id, and number of negative
    branch lengths clipped to zero.
    """
    D = np.asarray(D_input, dtype=np.float64).copy()
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("Distance matrix must be square")
    if n != len(sample_names):
        raise ValueError("Distance matrix/sample_names size mismatch")
    if n < 2:
        raise ValueError("Need at least two samples to build an NJ tree")
    active = list(range(n))
    next_id = n
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    names = {i: str(sample_names[i]) for i in range(n)}
    neg_clipped = 0

    def add_edge(a: int, b: int, length: float) -> None:
        nonlocal neg_clipped
        length = float(length)
        if clip_negative and length < 0:
            neg_clipped += 1
            length = 0.0
        adj.setdefault(a, []).append((b, length))
        adj.setdefault(b, []).append((a, length))

    last_report = n + 1
    while len(active) > 2:
        m = len(active)
        r = D.sum(axis=1)
        Q = (m - 2.0) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        flat = int(np.argmin(Q))
        i, j = divmod(flat, m)
        if j < i:
            i, j = j, i
        dij = float(D[i, j])
        delta = (float(r[i]) - float(r[j])) / float(m - 2)
        limb_i = 0.5 * dij + 0.5 * delta
        limb_j = dij - limb_i
        new_id = next_id
        next_id += 1
        adj.setdefault(new_id, [])
        add_edge(new_id, active[i], limb_i)
        add_edge(new_id, active[j], limb_j)

        keep = [k for k in range(m) if k not in (i, j)]
        new_d = np.array([(D[i, k] + D[j, k] - dij) * 0.5 for k in keep], dtype=np.float64)
        new_d[new_d < 0] = 0.0
        D_keep = D[np.ix_(keep, keep)]
        D2 = np.empty((m - 1, m - 1), dtype=np.float64)
        D2[:-1, :-1] = D_keep
        D2[:-1, -1] = new_d
        D2[-1, :-1] = new_d
        D2[-1, -1] = 0.0
        D = D2
        active = [active[k] for k in keep] + [new_id]
        if progress and (m <= 20 or m <= last_report - 100):
            sys.stderr.write(f"[auto-nj] NJ active_nodes={m}\n")
            sys.stderr.flush()
            last_report = m

    root_id = next_id
    adj.setdefault(root_id, [])
    final_d = float(D[0, 1]) if len(active) == 2 else 0.0
    add_edge(root_id, active[0], final_d * 0.5)
    add_edge(root_id, active[1], final_d * 0.5)
    return adj, names, root_id, neg_clipped


def _farthest_leaf(adj: dict[int, list[tuple[int, float]]], names: dict[int, str], start: int) -> tuple[int, float, dict[int, int]]:
    parent: dict[int, int] = {start: -1}
    dist: dict[int, float] = {start: 0.0}
    stack = [start]
    while stack:
        u = stack.pop()
        for v, w in adj[u]:
            if v == parent.get(u, -2):
                continue
            parent[v] = u
            dist[v] = dist[u] + float(w)
            stack.append(v)
    leaf_dists = [(node, d) for node, d in dist.items() if node in names]
    if not leaf_dists:
        raise ValueError("No leaves found while midpoint-rooting NJ tree")
    node, d = max(leaf_dists, key=lambda x: x[1])
    return int(node), float(d), parent


def _midpoint_root_adjacency(adj: dict[int, list[tuple[int, float]]], names: dict[int, str]) -> int:
    # Find weighted diameter among leaves, then split the diameter path at the midpoint.
    any_leaf = next(iter(names))
    a, _, _ = _farthest_leaf(adj, names, any_leaf)
    b, diam, parent = _farthest_leaf(adj, names, a)
    if diam <= 0:
        # Degenerate distances. Use the previous NJ final root if possible, otherwise any internal node.
        return max(adj)

    # Reconstruct path from b back to a, then reverse to a->b.
    path = [b]
    x = b
    while x != a:
        x = parent[x]
        if x < 0:
            raise ValueError("Internal error reconstructing NJ midpoint path")
        path.append(x)
    path = list(reversed(path))
    half = diam * 0.5
    acc = 0.0

    def edge_len(u: int, v: int) -> float:
        for z, w in adj[u]:
            if z == v:
                return float(w)
        raise KeyError((u, v))

    eps = 1e-12
    for idx in range(len(path) - 1):
        u, v = path[idx], path[idx + 1]
        w = edge_len(u, v)
        if abs(acc - half) <= eps:
            return u
        if acc + w >= half - eps:
            offset = half - acc
            if offset <= eps:
                return u
            if w - offset <= eps:
                return v
            root = max(adj) + 1
            # Remove u-v and replace by u-root-v.
            adj[u] = [(z, l) for z, l in adj[u] if z != v]
            adj[v] = [(z, l) for z, l in adj[v] if z != u]
            adj[root] = []
            adj[u].append((root, offset))
            adj[root].append((u, offset))
            adj[v].append((root, w - offset))
            adj[root].append((v, w - offset))
            return root
        acc += w
    return path[-1]


def _to_newick(adj: dict[int, list[tuple[int, float]]], names: dict[int, str], root: int) -> str:
    sys.setrecursionlimit(max(10000, len(adj) * 3))

    def rec(u: int, parent: int | None) -> str:
        children = [(v, w) for v, w in adj[u] if v != parent]
        if u in names and not children:
            label = _quote_newick_label(names[u])
        else:
            label = "(" + ",".join(rec(v, u) + ":" + _fmt_len(w) for v, w in children) + ")"
            if u in names:
                label += _quote_newick_label(names[u])
        return label

    return rec(root, None) + ";\n"


def build_nj_tree_from_binary_matrix(
    X: np.ndarray,
    sample_names: list[str],
    out_path: str | Path,
    min_mac: int = 2,
    max_loci: int = 50000,
    seed: int = 1,
    chunk_size: int = 2048,
    progress: bool = True,
) -> NJBuildResult:
    """Build a midpoint-rooted neighbour-joining tree from binary fake-FASTA data.

    The distance is normalized Hamming distance over polymorphic loci with MAC >=
    min_mac. If max_loci > 0, at most max_loci loci are used, chosen with a fixed
    random seed. The written tree is rooted by midpoint and suitable for the
    v0.5.0 tree_11_gain_count diagnostic/filter.
    """
    Xb = np.asarray(X)
    sample_names = list(sample_names)
    if Xb.shape[0] != len(sample_names):
        raise ValueError("X/sample_names size mismatch for NJ tree construction")
    loci, n_available = select_nj_loci(Xb, min_mac=min_mac, max_loci=max_loci, seed=seed)
    if progress:
        sys.stderr.write(
            f"[auto-nj] selected_loci={len(loci)} available={n_available} min_mac={min_mac} max_loci={max_loci}\n"
        )
        sys.stderr.flush()
    D = hamming_distance_matrix_from_binary(Xb, loci, chunk_size=chunk_size, progress=progress)
    tri = D[np.triu_indices(D.shape[0], k=1)]
    min_d = float(np.min(tri)) if tri.size else 0.0
    max_d = float(np.max(tri)) if tri.size else 0.0
    mean_d = float(np.mean(tri)) if tri.size else 0.0
    if progress:
        sys.stderr.write(f"[auto-nj] distance min={min_d:.4g} mean={mean_d:.4g} max={max_d:.4g}\n")
        sys.stderr.flush()
    adj, names, _final_root, neg_clipped = _build_nj_adjacency(D, sample_names, clip_negative=True, progress=progress)
    root = _midpoint_root_adjacency(adj, names)
    newick = _to_newick(adj, names, root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(newick, encoding="utf-8")
    return NJBuildResult(
        tree_path=str(out_path),
        n_samples=int(Xb.shape[0]),
        n_loci_available=int(n_available),
        n_loci_used=int(loci.size),
        min_mac=int(min_mac),
        max_loci=int(max_loci),
        distance="normalized_hamming_mac_filtered_binary_fasta",
        midpoint_rooted=1,
        nj_negative_branches_clipped=int(neg_clipped),
        nj_min_distance=min_d,
        nj_max_distance=max_d,
        nj_mean_distance=mean_d,
    )

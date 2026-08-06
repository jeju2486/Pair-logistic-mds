from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EffectiveEventClusterResult:
    labels: np.ndarray
    n_clusters_requested: int
    n_clusters_observed: int
    source: str
    details: dict[str, int | float | str]


def build_sample_clusters_from_kinship(
    K: np.ndarray,
    n_clusters: int = 50,
    source: str = "kinship_mst",
) -> EffectiveEventClusterResult:
    """Build sample clusters for approximate independent-event counting.

    The clusters are derived from the minimum spanning tree of the sample-sample
    distance induced by a covariance/similarity matrix K:
        d_ij^2 = K_ii + K_jj - 2 K_ij.

    Cutting the largest MST edges gives a simple, deterministic approximation to
    broad sample/lineage partitions. For a pair, effective_event_11 is computed
    from how its 11 carriers are distributed across these clusters.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")
    n = int(K.shape[0])
    if n == 0:
        raise ValueError("K has zero samples")
    k = int(n_clusters)
    if k <= 1 or n == 1:
        return EffectiveEventClusterResult(
            labels=np.zeros(n, dtype=np.int64),
            n_clusters_requested=max(1, k),
            n_clusters_observed=1,
            source=source,
            details={"n_samples": n, "n_edges_cut": 0},
        )
    k = min(k, n)

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

    diag = np.diag(K).astype(np.float64)
    D2 = diag[:, None] + diag[None, :] - 2.0 * K
    D2[~np.isfinite(D2)] = 0.0
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)

    mst = minimum_spanning_tree(D).tocoo()
    if mst.nnz == 0:
        return EffectiveEventClusterResult(
            labels=np.zeros(n, dtype=np.int64),
            n_clusters_requested=k,
            n_clusters_observed=1,
            source=source,
            details={"n_samples": n, "n_edges_cut": 0, "mst_edges": 0},
        )

    rows = mst.row.astype(np.int64)
    cols = mst.col.astype(np.int64)
    weights = mst.data.astype(np.float64)
    # Remove largest k-1 edges from the MST.
    n_cut = min(k - 1, weights.size)
    order_desc = np.argsort(weights)[::-1]
    keep = np.ones(weights.size, dtype=bool)
    if n_cut > 0:
        keep[order_desc[:n_cut]] = False

    rr = np.concatenate([rows[keep], cols[keep]])
    cc = np.concatenate([cols[keep], rows[keep]])
    data = np.ones(rr.size, dtype=np.int8)
    adj = coo_matrix((data, (rr, cc)), shape=(n, n)).tocsr()
    n_comp, labels = connected_components(adj, directed=False, return_labels=True)
    return EffectiveEventClusterResult(
        labels=labels.astype(np.int64, copy=False),
        n_clusters_requested=k,
        n_clusters_observed=int(n_comp),
        source=source,
        details={
            "n_samples": n,
            "mst_edges": int(weights.size),
            "n_edges_cut": int(n_cut),
            "mst_edge_weight_median": float(np.median(weights)) if weights.size else float("nan"),
            "mst_edge_weight_max": float(np.max(weights)) if weights.size else float("nan"),
        },
    )


def effective_event_stats_from_mask(z: np.ndarray, labels: np.ndarray, n_clusters: int | None = None) -> tuple[int, float, float]:
    """Return event_count, inverse-Simpson effective event number, largest fraction.

    z is a boolean vector for samples carrying the pair state of interest, here 11.
    labels assigns each sample to a broad sample/lineage cluster.
    """
    z = np.asarray(z, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)
    n11 = int(np.count_nonzero(z))
    if n11 <= 0:
        return 0, 0.0, 1.0
    if n_clusters is None:
        n_clusters = int(labels.max()) + 1 if labels.size else 0
    if n_clusters <= 0:
        return 0, 0.0, 1.0
    counts = np.bincount(labels[z], minlength=int(n_clusters)).astype(np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0, 0.0, 1.0
    frac = counts / float(n11)
    eff = 1.0 / float(np.sum(frac * frac))
    largest = float(np.max(frac))
    return int(counts.size), float(eff), largest

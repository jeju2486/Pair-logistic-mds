from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import numpy as np
from scipy.optimize import minimize

from .kinship import _NewickParser, _assign_depths, _walk_postorder, _walk_preorder


@dataclass
class TreeEventSummary:
    tree_11_gain_count: int
    tree_11_gain_count_max: int
    tree_11_loss_count: int
    tree_11_parsimony_score: float
    tree_11_largest_origin_n11: int
    tree_11_largest_origin_fraction: float
    tree_11_singleton_origin_count: int
    tree_11_origin_sizes: str


@dataclass
class CTMC11Summary:
    ctmc_11_gain_expected: float
    ctmc_11_loss_expected: float
    ctmc_11_q01: float
    ctmc_11_q10: float
    ctmc_11_loglik: float
    ctmc_11_status: str


class Tree11EventCounter:
    """Tree diagnostics for the binary pair state z=1[u=1 and v=1].

    Two related diagnostics are implemented:
    - Sankoff parsimony gain count (older v0.5.x tree_11_* columns).
    - Binary CTMC reconstructed-edge effective 0->1 gain count with fixed
      second-order Taylor transition approximation (v0.5.4 ctmc_11_* columns).

    Both use a rooted tree and fix the root state to z=0.
    """

    def __init__(self, tree_path: str | Path, sample_names: list[str], missing_length: str = "zero"):
        self.tree_path = str(tree_path)
        text = Path(tree_path).read_text(encoding="utf-8").strip()
        root = _NewickParser(text).parse()
        _assign_depths(root, missing_length=missing_length)
        self.root_node = root
        self.sample_names = list(sample_names)
        self.sample_to_index = {s: i for i, s in enumerate(sample_names)}
        if len(self.sample_to_index) != len(sample_names):
            raise ValueError("Duplicate sample names in FASTA")

        tips = [node for node in _walk_preorder(root) if node.is_leaf]
        tip_names = [t.name for t in tips if t.name]
        tip_set = set(tip_names)
        missing = [s for s in sample_names if s not in tip_set]
        if missing:
            raise ValueError(f"Event tree is missing {len(missing)} FASTA sample names. Examples: {missing[:5]}")

        self.nodes = list(_walk_postorder(root))
        self.node_index = {id(node): i for i, node in enumerate(self.nodes)}
        self.root = self.node_index[id(root)]
        self.children: list[list[int]] = []
        self.parent = np.full(len(self.nodes), -1, dtype=np.int64)
        self.tip_index = np.full(len(self.nodes), -1, dtype=np.int64)
        self.branch_length = np.zeros(len(self.nodes), dtype=np.float64)
        self.desc_tips: list[np.ndarray] = []

        for i, node in enumerate(self.nodes):
            child_idx = [self.node_index[id(c)] for c in node.children]
            self.children.append(child_idx)
            for c in child_idx:
                self.parent[c] = i
            if node.parent is not None:
                # Depths have already resolved missing lengths. This is safer than
                # reading node.length directly after midpoint/NJ rooting.
                self.branch_length[i] = max(0.0, float(node.depth - node.parent.depth))
            if node.is_leaf and node.name in self.sample_to_index:
                self.tip_index[i] = self.sample_to_index[node.name]

        for i, node in enumerate(self.nodes):
            ti = self.tip_index[i]
            if ti >= 0:
                arr = np.array([int(ti)], dtype=np.int64)
            else:
                arrs = [self.desc_tips[c] for c in self.children[i] if self.desc_tips[c].size > 0]
                arr = np.concatenate(arrs) if arrs else np.empty(0, dtype=np.int64)
            self.desc_tips.append(arr)

        self.n_nodes = len(self.nodes)
        self.n_samples = len(sample_names)
        self.n_tree_tips = len(tips)
        self.n_extra_tree_tips = int(sum(1 for name in tip_names if name not in self.sample_to_index))
        self.preorder_indices = [self.node_index[id(node)] for node in _walk_preorder(root)]

    @classmethod
    def from_newick(cls, tree_path: str | Path, sample_names: list[str], missing_length: str = "zero") -> "Tree11EventCounter":
        return cls(tree_path, sample_names, missing_length=missing_length)

    def summary_details(self) -> dict[str, Any]:
        finite_bl = self.branch_length[np.isfinite(self.branch_length)]
        positive = finite_bl[finite_bl > 0]
        return {
            "tree_11_event_tree": self.tree_path,
            "tree_11_event_n_nodes": self.n_nodes,
            "tree_11_event_n_samples": self.n_samples,
            "tree_11_event_n_tree_tips": self.n_tree_tips,
            "tree_11_event_n_extra_tree_tips": self.n_extra_tree_tips,
            "tree_11_event_root_fixed": 0,
            "tree_11_event_branch_length_median": float(np.median(positive)) if positive.size else 0.0,
            "tree_11_event_branch_length_max": float(np.max(positive)) if positive.size else 0.0,
        }

    def summarize(self, z11: np.ndarray, gain_cost: float = 1.0, loss_cost: float = 1.0) -> TreeEventSummary:
        z = np.asarray(z11, dtype=bool)
        if z.shape[0] != self.n_samples:
            raise ValueError(f"z11 length {z.shape[0]} does not match tree sample count {self.n_samples}")
        n11 = int(np.count_nonzero(z))
        if n11 <= 0:
            return TreeEventSummary(0, 0, 0, 0.0, 0, 0.0, 0, "")

        n = self.n_nodes
        inf = 1e100
        cost = np.full((n, 2), inf, dtype=np.float64)
        gmin = np.zeros((n, 2), dtype=np.int64)
        lmin = np.zeros((n, 2), dtype=np.int64)
        gmax = np.zeros((n, 2), dtype=np.int64)

        for i in range(n):
            ti = int(self.tip_index[i])
            if len(self.children[i]) == 0:
                if ti >= 0:
                    state = 1 if z[ti] else 0
                    cost[i, state] = 0.0
                    cost[i, 1 - state] = inf
                else:
                    cost[i, 0] = 0.0
                    cost[i, 1] = 0.0
                continue
            for ps in (0, 1):
                total_cost = 0.0
                total_gmin = 0
                total_lmin = 0
                total_gmax = 0
                for c in self.children[i]:
                    cand = []
                    for cs in (0, 1):
                        tc = 0.0
                        tg = 0
                        tl = 0
                        if ps == 0 and cs == 1:
                            tc = gain_cost
                            tg = 1
                        elif ps == 1 and cs == 0:
                            tc = loss_cost
                            tl = 1
                        cand.append((cost[c, cs] + tc, int(gmin[c, cs]) + tg, int(lmin[c, cs]) + tl, int(gmax[c, cs]) + tg, cs))
                    best_min = min(cand, key=lambda x: (x[0], x[1], x[2], x[4]))
                    min_cost = min(x[0] for x in cand)
                    best_max = max([x for x in cand if abs(x[0] - min_cost) <= 1e-9], key=lambda x: (x[3], -x[2], -x[4]))
                    total_cost += best_min[0]
                    total_gmin += best_min[1]
                    total_lmin += best_min[2]
                    total_gmax += best_max[3]
                cost[i, ps] = total_cost
                gmin[i, ps] = total_gmin
                lmin[i, ps] = total_lmin
                gmax[i, ps] = total_gmax

        root_state = 0
        states = np.full(n, -1, dtype=np.int8)
        states[self.root] = root_state
        for i in self.preorder_indices:
            ps = int(states[i])
            if ps < 0:
                continue
            for c in self.children[i]:
                cand = []
                for cs in (0, 1):
                    tc = 0.0
                    tg = 0
                    tl = 0
                    if ps == 0 and cs == 1:
                        tc = gain_cost
                        tg = 1
                    elif ps == 1 and cs == 0:
                        tc = loss_cost
                        tl = 1
                    cand.append((cost[c, cs] + tc, int(gmin[c, cs]) + tg, int(lmin[c, cs]) + tl, cs))
                best = min(cand, key=lambda x: (x[0], x[1], x[2], x[3]))
                states[c] = int(best[3])

        origin_sizes: list[int] = []
        losses = 0
        for i in range(n):
            p = int(self.parent[i])
            if p < 0:
                continue
            if states[p] == 0 and states[i] == 1:
                desc = self.desc_tips[i]
                origin_sizes.append(int(np.count_nonzero(z[desc])) if desc.size else 0)
            elif states[p] == 1 and states[i] == 0:
                losses += 1
        origin_sizes = [s for s in origin_sizes if s > 0]
        origin_sizes.sort(reverse=True)
        largest = int(origin_sizes[0]) if origin_sizes else 0
        largest_frac = float(largest / n11) if n11 > 0 else 0.0
        singleton = int(sum(1 for s in origin_sizes if s == 1))
        return TreeEventSummary(
            tree_11_gain_count=int(gmin[self.root, root_state]),
            tree_11_gain_count_max=int(gmax[self.root, root_state]),
            tree_11_loss_count=int(losses),
            tree_11_parsimony_score=float(cost[self.root, root_state]),
            tree_11_largest_origin_n11=largest,
            tree_11_largest_origin_fraction=largest_frac,
            tree_11_singleton_origin_count=singleton,
            tree_11_origin_sizes=",".join(map(str, origin_sizes[:50])),
        )

    @staticmethod
    def _P_taylor2(q01: float, q10: float, t: float) -> np.ndarray:
        """Second-order Taylor approximation of exp(Qt), projected to a stochastic matrix."""
        t = max(0.0, float(t))
        a = max(1e-12, float(q01))
        b = max(1e-12, float(q10))
        q = np.array([[-a, a], [b, -b]], dtype=np.float64)
        p = np.eye(2, dtype=np.float64) + q * t + (q @ q) * (0.5 * t * t)
        # Taylor2 can overshoot for long branches/rates. Project rows back to a
        # valid transition matrix; this keeps the approximation stable and fast.
        p = np.clip(p, 1e-300, np.inf)
        p /= p.sum(axis=1, keepdims=True)
        return p

    @staticmethod
    def _N_taylor2(q01: float, q10: float, t: float, event: str) -> np.ndarray:
        """Second-order expected-count numerator matrix for one branch.

        N_event[i,j] approximates E[#event and endpoint j | start i] * P(i->j)
        using the integral of first-order transition expansions. Dividing by
        P[i,j] gives the conditional endpoint-specific count; in pruning we use
        the numerator directly.
        """
        t = max(0.0, float(t))
        a = max(1e-12, float(q01))
        b = max(1e-12, float(q10))
        n = np.zeros((2, 2), dtype=np.float64)
        if t == 0.0:
            return n
        if event == "01":
            rate = a
            src, dst = 0, 1
        else:
            rate = b
            src, dst = 1, 0
        q = np.array([[-a, a], [b, -b]], dtype=np.float64)
        for i in (0, 1):
            for j in (0, 1):
                val = 0.0
                if i == src and j == dst:
                    val += t
                if i == src:
                    val += q[dst, j] * (0.5 * t * t)
                if j == dst:
                    val += q[i, src] * (0.5 * t * t)
                n[i, j] = max(0.0, rate * val)
        return n

    def _reconstruct_binary_states(self, z: np.ndarray, gain_cost: float = 1.0, loss_cost: float = 1.0) -> tuple[np.ndarray, float, int, int]:
        """Reconstruct one minimum-cost binary history with root fixed to 0.

        This is the fast v0.5.4 CTMC backend: once internal states are fixed,
        rate fitting reduces to an observed edge-transition likelihood. The
        tie-break is conservative for origin counting: minimum total cost, then
        minimum number of 0->1 gains, then minimum losses.
        """
        n = self.n_nodes
        inf = 1e100
        cost = np.full((n, 2), inf, dtype=np.float64)
        gmin = np.zeros((n, 2), dtype=np.int64)
        lmin = np.zeros((n, 2), dtype=np.int64)

        for i in range(n):
            ti = int(self.tip_index[i])
            if len(self.children[i]) == 0:
                if ti >= 0:
                    state = 1 if z[ti] else 0
                    cost[i, state] = 0.0
                    cost[i, 1 - state] = inf
                else:
                    cost[i, 0] = 0.0
                    cost[i, 1] = 0.0
                continue
            for ps in (0, 1):
                total_cost = 0.0
                total_g = 0
                total_l = 0
                for c in self.children[i]:
                    cand = []
                    for cs in (0, 1):
                        tc = 0.0
                        tg = 0
                        tl = 0
                        if ps == 0 and cs == 1:
                            tc = gain_cost
                            tg = 1
                        elif ps == 1 and cs == 0:
                            tc = loss_cost
                            tl = 1
                        cand.append((cost[c, cs] + tc, int(gmin[c, cs]) + tg, int(lmin[c, cs]) + tl, cs))
                    best = min(cand, key=lambda x: (x[0], x[1], x[2], x[3]))
                    total_cost += best[0]
                    total_g += best[1]
                    total_l += best[2]
                cost[i, ps] = total_cost
                gmin[i, ps] = total_g
                lmin[i, ps] = total_l

        root_state = 0
        states = np.full(n, -1, dtype=np.int8)
        states[self.root] = root_state
        for i in self.preorder_indices:
            ps = int(states[i])
            if ps < 0:
                continue
            for c in self.children[i]:
                cand = []
                for cs in (0, 1):
                    tc = 0.0
                    tg = 0
                    tl = 0
                    if ps == 0 and cs == 1:
                        tc = gain_cost
                        tg = 1
                    elif ps == 1 and cs == 0:
                        tc = loss_cost
                        tl = 1
                    cand.append((cost[c, cs] + tc, int(gmin[c, cs]) + tg, int(lmin[c, cs]) + tl, cs))
                best = min(cand, key=lambda x: (x[0], x[1], x[2], x[3]))
                states[c] = int(best[3])
        return states, float(cost[self.root, root_state]), int(gmin[self.root, root_state]), int(lmin[self.root, root_state])

    def _edge_transition_arrays(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return parent states, child states, and branch lengths for all non-root edges."""
        edge_nodes = np.nonzero(self.parent >= 0)[0].astype(np.int64)
        pstate = states[self.parent[edge_nodes]].astype(np.int8, copy=False)
        cstate = states[edge_nodes].astype(np.int8, copy=False)
        blen = self.branch_length[edge_nodes].astype(np.float64, copy=False)
        return pstate, cstate, blen

    @staticmethod
    def _edge_P_entries_taylor2(q01: float, q10: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised second-order Taylor transition entries for all edges."""
        tt = np.maximum(0.0, np.asarray(t, dtype=np.float64))
        a = max(1e-12, float(q01))
        b = max(1e-12, float(q10))
        t2 = tt * tt
        ab = a + b
        p00 = 1.0 - a * tt + 0.5 * a * ab * t2
        p01 = a * tt - 0.5 * a * ab * t2
        p10 = b * tt - 0.5 * b * ab * t2
        p11 = 1.0 - b * tt + 0.5 * b * ab * t2
        p00 = np.maximum(p00, 1e-300)
        p01 = np.maximum(p01, 1e-300)
        p10 = np.maximum(p10, 1e-300)
        p11 = np.maximum(p11, 1e-300)
        row0 = p00 + p01
        row1 = p10 + p11
        p00 = p00 / row0
        p01 = p01 / row0
        p10 = p10 / row1
        p11 = p11 / row1
        return p00, p01, p10, p11

    @classmethod
    def _edge_loglik_taylor2(cls, pstate: np.ndarray, cstate: np.ndarray, blen: np.ndarray, q01: float, q10: float) -> float:
        p00, p01, p10, p11 = cls._edge_P_entries_taylor2(q01, q10, blen)
        probs = np.where(
            pstate == 0,
            np.where(cstate == 0, p00, p01),
            np.where(cstate == 0, p10, p11),
        )
        ll = float(np.sum(np.log(np.maximum(probs, 1e-300))))
        return ll

    def _ctmc_fit_rates_on_edges(self, pstate: np.ndarray, cstate: np.ndarray, blen: np.ndarray) -> tuple[float, float, float, str]:
        gains = int(np.count_nonzero((pstate == 0) & (cstate == 1)))
        losses = int(np.count_nonzero((pstate == 1) & (cstate == 0)))
        len0 = float(np.sum(blen[pstate == 0]))
        len1 = float(np.sum(blen[pstate == 1]))
        q01_0 = max(1e-5, (gains + 0.5) / max(len0, 1e-6))
        q10_0 = max(1e-5, (losses + 0.5) / max(len1, 1e-6))
        bounds = [(-12.0, 4.0), (-12.0, 4.0)]

        def obj(theta: np.ndarray) -> float:
            q01 = float(math.exp(theta[0]))
            q10 = float(math.exp(theta[1]))
            ll = self._edge_loglik_taylor2(pstate, cstate, blen, q01, q10)
            if not np.isfinite(ll):
                return 1e100
            return -ll

        x0 = np.array([math.log(min(max(q01_0, math.exp(bounds[0][0])), math.exp(bounds[0][1]))),
                       math.log(min(max(q10_0, math.exp(bounds[1][0])), math.exp(bounds[1][1])))], dtype=np.float64)
        try:
            res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 40, "ftol": 1e-7})
            theta = res.x if np.all(np.isfinite(res.x)) else x0
            q01 = float(math.exp(theta[0]))
            q10 = float(math.exp(theta[1]))
            ll = -float(res.fun) if np.isfinite(res.fun) else self._edge_loglik_taylor2(pstate, cstate, blen, q01, q10)
            status = "OK" if res.success else "OPT_WARN"
        except Exception as e:
            q01, q10 = q01_0, q10_0
            ll = self._edge_loglik_taylor2(pstate, cstate, blen, q01, q10)
            status = f"OPT_FAIL:{type(e).__name__}"
        return q01, q10, ll, status

    @staticmethod
    def _edge_effective_counts_taylor2(pstate: np.ndarray, cstate: np.ndarray, blen: np.ndarray, q01: float, q10: float) -> tuple[float, float]:
        """Expected gain/loss counts conditional on reconstructed edge endpoints.

        Uses the same Taylor2 endpoint transition approximation as the likelihood.
        The result is close to the reconstructed transition count, with small
        hidden gain-loss/loss-gain contributions on 0->0 and 1->1 branches.
        """
        p00, p01, p10, p11 = Tree11EventCounter._edge_P_entries_taylor2(q01, q10, blen)
        t = np.maximum(0.0, np.asarray(blen, dtype=np.float64))
        t2 = t * t
        a = max(1e-12, float(q01))
        b = max(1e-12, float(q10))
        ab = a * b
        # Taylor2 expected-count numerator matrices for endpoints.
        n01_00 = 0.5 * ab * t2
        n01_01 = np.maximum(0.0, a * (t - 0.5 * (a + b) * t2))
        n01_10 = np.zeros_like(t)
        n01_11 = 0.5 * ab * t2
        n10_00 = 0.5 * ab * t2
        n10_01 = np.zeros_like(t)
        n10_10 = np.maximum(0.0, b * (t - 0.5 * (a + b) * t2))
        n10_11 = 0.5 * ab * t2
        p = np.where(
            pstate == 0,
            np.where(cstate == 0, p00, p01),
            np.where(cstate == 0, p10, p11),
        )
        n01 = np.where(
            pstate == 0,
            np.where(cstate == 0, n01_00, n01_01),
            np.where(cstate == 0, n01_10, n01_11),
        )
        n10 = np.where(
            pstate == 0,
            np.where(cstate == 0, n10_00, n10_01),
            np.where(cstate == 0, n10_10, n10_11),
        )
        denom = np.maximum(p, 1e-300)
        gain = float(np.sum(n01 / denom))
        loss = float(np.sum(n10 / denom))
        return gain, loss

    def summarize_ctmc_taylor2(self, z11: np.ndarray) -> CTMC11Summary:
        """Fast binary CTMC using reconstructed edge transitions (v0.5.4).

        This is intentionally approximate: it first reconstructs one binary
        internal-state history with root fixed to 0, then fits q01/q10 from the
        resulting observed edge transitions. ctmc_11_gain_expected is the
        Taylor2 effective 0->1 count conditional on that reconstructed history.
        """
        z = np.asarray(z11, dtype=bool)
        if z.shape[0] != self.n_samples:
            raise ValueError(f"z11 length {z.shape[0]} does not match tree sample count {self.n_samples}")
        n11 = int(np.count_nonzero(z))
        if n11 <= 0:
            return CTMC11Summary(0.0, 0.0, 1e-12, 1e-12, 0.0, "ALL_ZERO")

        try:
            states, _score, _g, _l = self._reconstruct_binary_states(z, gain_cost=1.0, loss_cost=1.0)
            pstate, cstate, blen = self._edge_transition_arrays(states)
            q01, q10, ll, status = self._ctmc_fit_rates_on_edges(pstate, cstate, blen)
            gain_exp, loss_exp = self._edge_effective_counts_taylor2(pstate, cstate, blen, q01, q10)
            if not np.isfinite(gain_exp) or not np.isfinite(loss_exp):
                status = "BAD_EXPECTATION" if status == "OK" else status + ";BAD_EXPECTATION"
                gain_exp = np.nan
                loss_exp = np.nan
        except Exception as e:
            return CTMC11Summary(np.nan, np.nan, np.nan, np.nan, np.nan, f"EDGE_CTMC_FAIL:{type(e).__name__}")

        return CTMC11Summary(
            ctmc_11_gain_expected=float(gain_exp),
            ctmc_11_loss_expected=float(loss_exp),
            ctmc_11_q01=float(q01),
            ctmc_11_q10=float(q10),
            ctmc_11_loglik=float(ll),
            ctmc_11_status=status,
        )

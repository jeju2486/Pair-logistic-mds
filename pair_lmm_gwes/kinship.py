from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class KinshipResult:
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
        return len(self.children) == 0


class _NewickParser:
    """Small Newick parser sufficient for pyseer-style tree covariance.

    The goal here is not to replace DendroPy. It is to keep pair-lmm-gwes
    standalone while reproducing pyseer's `phylogeny_distance.py --lmm` logic:
    K_ij = distance from root to MRCA(i, j), using branch lengths.
    """

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
            raise ValueError(f"Unexpected trailing Newick text near position {self.i}: {self.text[self.i:self.i+40]!r}")
        root.parent = None
        return root

    def _skip_ws(self) -> None:
        while self.i < self.n and self.text[self.i].isspace():
            self.i += 1

    def _parse_subtree(self) -> _Node:
        self._skip_ws()
        if self.i >= self.n:
            raise ValueError("Unexpected end of Newick string")
        if self.text[self.i] == "(":
            self.i += 1
            children: list[_Node] = []
            while True:
                child = self._parse_subtree()
                children.append(child)
                self._skip_ws()
                if self.i >= self.n:
                    raise ValueError("Unclosed internal node in Newick string")
                ch = self.text[self.i]
                if ch == ",":
                    self.i += 1
                    continue
                if ch == ")":
                    self.i += 1
                    break
                raise ValueError(f"Expected ',' or ')' at position {self.i}, found {ch!r}")
            name = self._parse_label_optional()
            length = self._parse_length_optional()
            node = _Node(name=name, length=length, children=children)
            for c in children:
                c.parent = node
            return node
        else:
            name = self._parse_label_required()
            length = self._parse_length_optional()
            return _Node(name=name, length=length)

    def _parse_label_required(self) -> str:
        label = self._parse_label_optional()
        if label is None or label == "":
            raise ValueError(f"Expected leaf label at position {self.i}")
        return label

    def _parse_label_optional(self) -> Optional[str]:
        self._skip_ws()
        if self.i >= self.n or self.text[self.i] in ":,();":
            return None
        if self.text[self.i] in "'\"":
            quote = self.text[self.i]
            self.i += 1
            out = []
            while self.i < self.n:
                ch = self.text[self.i]
                self.i += 1
                if ch == quote:
                    # Newick quoted labels may represent escaped single quotes by doubling.
                    if self.i < self.n and self.text[self.i] == quote:
                        out.append(quote)
                        self.i += 1
                        continue
                    break
                out.append(ch)
            else:
                raise ValueError("Unclosed quoted label in Newick string")
            return "".join(out)
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
        if raw == "":
            raise ValueError("Empty branch length in Newick string")
        try:
            return float(raw)
        except ValueError as e:
            raise ValueError(f"Invalid branch length {raw!r}") from e


def _walk_preorder(node: _Node):
    yield node
    for c in node.children:
        yield from _walk_preorder(c)


def _walk_postorder(node: _Node):
    for c in node.children:
        yield from _walk_postorder(c)
    yield node


def _assign_depths(root: _Node, missing_length: str = "error") -> None:
    if missing_length not in {"error", "one", "zero"}:
        raise ValueError("missing_length must be one of: error, one, zero")

    def length_for(node: _Node) -> float:
        if node.parent is None:
            # Root branch length, if present, is ignored for covariance-from-root.
            return 0.0
        if node.length is None:
            if missing_length == "error":
                label = node.name or "<internal>"
                raise ValueError(
                    f"Missing branch length for node {label!r}. "
                    "pyseer --lmm uses branch lengths; provide a branch-length tree."
                )
            return 1.0 if missing_length == "one" else 0.0
        return float(node.length)

    root.depth = 0.0
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            child.depth = node.depth + length_for(child)
            stack.append(child)


def build_tree_covariance(
    tree_path: str | Path,
    sample_names: list[str],
    dtype: str = "float64",
    missing_length: str = "error",
) -> KinshipResult:
    """Build pyseer-style LMM covariance C from a Newick tree.

    This follows the logic of pyseer's `scripts/phylogeny_distance.py --lmm`:
    for tips i and j, K[i, j] is the branch length from the root to their MRCA.
    Diagonal entries are root-to-tip depths. The matrix is then scaled so
    mean(diag(K)) = 1, matching pyseer's similarity-matrix normalisation.
    """
    path = Path(tree_path)
    text = path.read_text(encoding="utf-8").strip()
    root = _NewickParser(text).parse()
    _assign_depths(root, missing_length=missing_length)

    tips = [node for node in _walk_preorder(root) if node.is_leaf]
    if len(tips) == 0:
        raise ValueError(f"No tips found in tree: {path}")
    tip_by_name: dict[str, _Node] = {}
    for tip in tips:
        if not tip.name:
            raise ValueError("Tree contains unnamed tip")
        if tip.name in tip_by_name:
            raise ValueError(f"Duplicate tip label in tree: {tip.name}")
        tip_by_name[tip.name] = tip

    missing = [s for s in sample_names if s not in tip_by_name]
    extra = [name for name in tip_by_name if name not in set(sample_names)]
    if missing:
        raise ValueError(
            f"Tree is missing {len(missing)} FASTA sample names. "
            f"Examples: {missing[:5]}"
        )
    if extra:
        sys.stderr.write(
            f"[pair-lmm-gwes] warning: tree has {len(extra)} extra tips not in FASTA; they will be ignored. "
            f"Examples: {extra[:5]}\n"
        )
        sys.stderr.flush()

    n = len(sample_names)
    for idx, sample in enumerate(sample_names):
        tip_by_name[sample].tip_index = idx

    # Postorder descendant tip arrays restricted to FASTA samples.
    for node in _walk_postorder(root):
        if node.is_leaf:
            if node.tip_index is None:
                node.desc_tips = np.empty(0, dtype=np.int64)
            else:
                node.desc_tips = np.array([node.tip_index], dtype=np.int64)
        else:
            arrs = [c.desc_tips for c in node.children if c.desc_tips is not None and c.desc_tips.size > 0]
            node.desc_tips = np.concatenate(arrs) if arrs else np.empty(0, dtype=np.int64)

    K = np.zeros((n, n), dtype=np.float64)
    # Diagonal: root-to-tip depth.
    for sample in sample_names:
        tip = tip_by_name[sample]
        idx = tip.tip_index
        assert idx is not None
        K[idx, idx] = tip.depth

    # Off-diagonal: depth of MRCA. Assign each pair once at the node where
    # descendant tips are split between different child subtrees.
    for node in _walk_preorder(root):
        if node.is_leaf or len(node.children) < 2:
            continue
        child_tips = [c.desc_tips for c in node.children if c.desc_tips is not None and c.desc_tips.size > 0]
        if len(child_tips) < 2:
            continue
        d = float(node.depth)
        for a in range(len(child_tips)):
            ia = child_tips[a]
            for b in range(a + 1, len(child_tips)):
                ib = child_tips[b]
                K[np.ix_(ia, ib)] = d
                K[np.ix_(ib, ia)] = d

    K = 0.5 * (K + K.T)
    mean_diag = float(np.mean(np.diag(K)))
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        raise ValueError(
            "Invalid tree covariance diagonal scale. Check tree branch lengths and rooting."
        )
    K /= mean_diag
    out_dtype = np.float64 if dtype == "float64" else np.float32
    return KinshipResult(
        K=K.astype(out_dtype, copy=False),
        n_loci_used=0,
        n_loci_filtered=0,
        mean_diag_before_norm=mean_diag,
        source="tree_mrca_covariance",
        details={"tree": str(path), "n_tree_tips": len(tips), "n_samples": n},
    )


def build_pangenome_grm(
    X: np.ndarray,
    min_mac: int = 2,
    chunk_size: int = 4096,
    dtype: str = "float64",
    progress: bool = True,
) -> KinshipResult:
    """Build a pangenome GRM from binary fake-FASTA matrix.

    K = ZZ^T / M, where Z columns are frequency-standardised binary loci.
    Monomorphic and low-MAC loci are omitted from K construction.
    K is normalised so mean(diag(K)) = 1, following pyseer-style scaling.
    """
    n, m = X.shape
    out_dtype = np.float64 if dtype == "float64" else np.float32
    K = np.zeros((n, n), dtype=np.float64)  # accumulate in float64 for stability
    used = 0
    filtered = 0
    t0 = time.time()
    for start in range(0, m, chunk_size):
        end = min(m, start + chunk_size)
        B = X[:, start:end].astype(np.float64, copy=False)
        sums = B.sum(axis=0)
        mac = np.minimum(sums, n - sums)
        valid = mac >= min_mac
        if not np.any(valid):
            filtered += B.shape[1]
            continue
        B = B[:, valid]
        p = B.mean(axis=0)
        denom = np.sqrt(p * (1.0 - p))
        valid2 = np.isfinite(denom) & (denom > 0)
        if not np.any(valid2):
            filtered += int(valid.sum())
            continue
        B = B[:, valid2]
        p = p[valid2]
        denom = denom[valid2]
        Z = (B - p) / denom
        K += Z @ Z.T
        used += Z.shape[1]
        filtered += (end - start) - Z.shape[1]
        if progress and (start == 0 or end == m or ((start // chunk_size) % 10 == 0)):
            elapsed = time.time() - t0
            sys.stderr.write(f"[pair-lmm-gwes] grm_progress loci={end}/{m} used={used} elapsed={elapsed:.1f}s\n")
            sys.stderr.flush()
    if used == 0:
        raise ValueError("No loci passed min_mac filter for GRM construction")
    K /= float(used)
    K = 0.5 * (K + K.T)
    mean_diag = float(np.mean(np.diag(K)))
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        raise ValueError("Invalid GRM diagonal scale")
    K /= mean_diag
    return KinshipResult(
        K=K.astype(out_dtype, copy=False),
        n_loci_used=used,
        n_loci_filtered=filtered,
        mean_diag_before_norm=mean_diag,
        source="fake_fasta_pangenome_grm",
        details={"n_samples": n, "n_loci": m},
    )


def _valid_grm_loci(X: np.ndarray, min_mac: int = 2) -> np.ndarray:
    """Return loci that pass the same polymorphism/MAC filter used for fake-FASTA GRM."""
    n, m = X.shape
    sums = X.sum(axis=0).astype(np.float64)
    mac = np.minimum(sums, n - sums)
    valid = mac >= min_mac
    p = sums / float(n)
    denom = np.sqrt(p * (1.0 - p))
    valid &= np.isfinite(denom) & (denom > 0)
    return np.flatnonzero(valid).astype(np.int64)


def _accumulate_grm_from_loci(
    X: np.ndarray,
    loci: np.ndarray,
    chunk_size: int = 4096,
    dtype: str = "float64",
    progress: bool = True,
    progress_label: str = "grm_masked_progress",
) -> tuple[np.ndarray, int, float]:
    """Build normalised GRM from an explicit list of loci."""
    n = X.shape[0]
    out_dtype = np.float64 if dtype == "float64" else np.float32
    K = np.zeros((n, n), dtype=np.float64)
    used = 0
    t0 = time.time()
    loci = np.asarray(loci, dtype=np.int64)
    total = int(loci.size)
    for start in range(0, total, chunk_size):
        sub = loci[start:start + chunk_size]
        B = X[:, sub].astype(np.float64, copy=False)
        p = B.mean(axis=0)
        denom = np.sqrt(p * (1.0 - p))
        ok = np.isfinite(denom) & (denom > 0)
        if not np.any(ok):
            continue
        B = B[:, ok]
        p = p[ok]
        denom = denom[ok]
        Z = (B - p) / denom
        K += Z @ Z.T
        used += int(Z.shape[1])
        if progress and (start == 0 or start + chunk_size >= total or ((start // max(1, chunk_size)) % 10 == 0)):
            elapsed = time.time() - t0
            sys.stderr.write(f"[pair-lmm-gwes] {progress_label} loci={min(start+chunk_size,total)}/{total} used={used} elapsed={elapsed:.1f}s\n")
            sys.stderr.flush()
    if used == 0:
        raise ValueError("No loci available for GRM construction after masking")
    K /= float(used)
    K = 0.5 * (K + K.T)
    mean_diag = float(np.mean(np.diag(K)))
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        raise ValueError("Invalid masked GRM diagonal scale")
    K /= mean_diag
    return K.astype(out_dtype, copy=False), used, mean_diag


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
    """Identify valid GRM loci that are proxies for tested/target loci.

    A valid GRM locus is masked if it is directly a target locus or if any valid
    target locus satisfies at least one proxy condition:
      - raw r^2 >= r2_threshold
      - same_mismatch_rate <= mismatch_threshold
      - complement_mismatch_rate <= mismatch_threshold

    This is a global tested-locus-proxy mask. It is intended to prevent the GRM
    from containing the tested marker pattern or close sample-pattern proxies.
    It is computationally cheaper than rebuilding a fully response-specific GRM.
    """
    n, m = X.shape
    valid_loci = np.asarray(valid_loci, dtype=np.int64)
    target_loci = np.asarray(sorted(set(int(x) for x in target_loci if 0 <= int(x) < m)), dtype=np.int64)
    valid_pos = {int(locus): i for i, locus in enumerate(valid_loci)}
    target_valid = np.array([int(x) for x in target_loci if int(x) in valid_pos], dtype=np.int64)

    mask = np.zeros(valid_loci.size, dtype=bool)
    direct_positions = [valid_pos[int(x)] for x in target_valid]
    if direct_positions:
        mask[np.asarray(direct_positions, dtype=np.int64)] = True

    if target_valid.size == 0:
        return mask, {
            "n_valid_loci": int(valid_loci.size),
            "n_target_loci_total": int(target_loci.size),
            "n_target_loci_valid": 0,
            "n_direct_masked": 0,
            "n_proxy_masked_total": int(mask.sum()),
            "r2_threshold": float(r2_threshold),
            "mismatch_threshold": float(mismatch_threshold),
        }

    # Work only with valid target loci so all denominators are defined.
    # Boolean/int8 matrices are cast to float64 inside matmul for stable counts.
    sums_valid = X[:, valid_loci].sum(axis=0).astype(np.float64)
    t0 = time.time()
    n_pairs_scanned = 0
    for ts in range(0, target_valid.size, target_chunk_size):
        t_loci = target_valid[ts:ts + target_chunk_size]
        T = X[:, t_loci].astype(np.float64, copy=False)
        sum_t = T.sum(axis=0).astype(np.float64)
        denom_t = sum_t * (n - sum_t)
        for gs in range(0, valid_loci.size, grm_chunk_size):
            g_loci = valid_loci[gs:gs + grm_chunk_size]
            G = X[:, g_loci].astype(np.float64, copy=False)
            sum_g = sums_valid[gs:gs + g_loci.size]
            denom_g = sum_g * (n - sum_g)
            n11 = T.T @ G
            n10 = sum_t[:, None] - n11
            n01 = sum_g[None, :] - n11
            n00 = n - n11 - n10 - n01

            mismatch_same = n10 + n01
            mismatch_complement = n11 + n00
            proxy = (mismatch_same <= mismatch_threshold * n) | (mismatch_complement <= mismatch_threshold * n)

            if r2_threshold is not None and r2_threshold > 0:
                denom = denom_t[:, None] * denom_g[None, :]
                with np.errstate(invalid="ignore", divide="ignore"):
                    phi = (n11 * n00 - n10 * n01) / np.sqrt(denom)
                    r2 = phi * phi
                proxy |= np.isfinite(r2) & (r2 >= float(r2_threshold))

            cols = np.any(proxy, axis=0)
            if np.any(cols):
                mask[gs:gs + g_loci.size] |= cols
            n_pairs_scanned += int(t_loci.size * g_loci.size)
        if progress:
            elapsed = time.time() - t0
            sys.stderr.write(
                f"[pair-lmm-gwes] grm_proxy_mask_progress targets={min(ts+target_chunk_size,target_valid.size)}/{target_valid.size} "
                f"masked={int(mask.sum())}/{valid_loci.size} elapsed={elapsed:.1f}s\n"
            )
            sys.stderr.flush()

    return mask, {
        "n_valid_loci": int(valid_loci.size),
        "n_target_loci_total": int(target_loci.size),
        "n_target_loci_valid": int(target_valid.size),
        "n_direct_masked": int(len(direct_positions)),
        "n_proxy_masked_total": int(mask.sum()),
        "n_proxy_masked_beyond_direct": int(mask.sum() - len(set(direct_positions))),
        "n_target_valid_x_valid_pairs_scanned": int(n_pairs_scanned),
        "r2_threshold": float(r2_threshold),
        "mismatch_threshold": float(mismatch_threshold),
    }


def build_pangenome_grm_with_tested_locus_mask(
    X: np.ndarray,
    target_loci: np.ndarray,
    min_mac: int = 2,
    chunk_size: int = 4096,
    dtype: str = "float64",
    progress: bool = True,
    mask_mode: str = "tested",
    mask_r2: float = 0.8,
    mask_mismatch: float = 0.02,
    mask_target_chunk_size: int = 256,
    mask_grm_chunk_size: int = 2048,
    shrinkage: float = 0.0,
) -> KinshipResult:
    """Build fake-FASTA GRM with optional tested-locus/proxy masking.

    mask_mode="none" reproduces build_pangenome_grm except for extra details.
    mask_mode="tested" excludes all candidate-pair loci that pass the GRM MAC
    filter, plus valid GRM loci that are high-r2/near-copy/near-complement proxies
    of those tested loci.

    This is a practical global approximation to response-proxy masking. It prevents
    direct tested-marker leakage into K without rebuilding/eigendecomposing a new
    K for every response locus.
    """
    if mask_mode not in {"none", "tested"}:
        raise ValueError("mask_mode must be 'none' or 'tested'")
    if not (0.0 <= float(shrinkage) < 1.0):
        raise ValueError("shrinkage must be in [0, 1)")

    valid_loci = _valid_grm_loci(X, min_mac=min_mac)
    n_total = int(X.shape[1])
    details: dict[str, str | int | float] = {
        "n_samples": int(X.shape[0]),
        "n_loci": n_total,
        "mask_mode": mask_mode,
        "mask_r2": float(mask_r2),
        "mask_mismatch": float(mask_mismatch),
        "grm_shrinkage": float(shrinkage),
        "n_valid_loci_before_mask": int(valid_loci.size),
    }

    if mask_mode == "tested":
        mask, mask_details = identify_proxy_loci_for_targets(
            X,
            valid_loci=valid_loci,
            target_loci=np.asarray(target_loci, dtype=np.int64),
            r2_threshold=mask_r2,
            mismatch_threshold=mask_mismatch,
            target_chunk_size=mask_target_chunk_size,
            grm_chunk_size=mask_grm_chunk_size,
            progress=progress,
        )
        details.update(mask_details)
        used_loci = valid_loci[~mask]
    else:
        used_loci = valid_loci
        details.update({
            "n_target_loci_total": int(len(set(map(int, np.asarray(target_loci).reshape(-1)))) if target_loci is not None else 0),
            "n_target_loci_valid": 0,
            "n_direct_masked": 0,
            "n_proxy_masked_total": 0,
            "n_proxy_masked_beyond_direct": 0,
        })

    if used_loci.size == 0:
        raise ValueError(
            "No loci remain for GRM after tested-locus/proxy masking. "
            "Use --grm-mask-mode none or relax --grm-mask-r2/--grm-mask-mismatch."
        )

    K, used, mean_diag = _accumulate_grm_from_loci(
        X,
        used_loci,
        chunk_size=chunk_size,
        dtype=dtype,
        progress=progress,
        progress_label="grm_masked_progress" if mask_mode != "none" else "grm_progress",
    )
    if shrinkage > 0.0:
        # K is already mean-diagonal normalised. Shrinking toward I stabilises
        # singular/near-singular fake-FASTA GRMs while preserving mean diagonal.
        K = (1.0 - float(shrinkage)) * K + float(shrinkage) * np.eye(K.shape[0], dtype=K.dtype)
        K = 0.5 * (K + K.T)
        md = float(np.mean(np.diag(K)))
        if np.isfinite(md) and md > 0:
            K /= md
    details["n_loci_used_after_mask"] = int(used)
    details["n_loci_filtered_mac_or_monomorphic"] = int(n_total - valid_loci.size)
    details["n_loci_removed_by_mask"] = int(valid_loci.size - used)

    source = "fake_fasta_pangenome_grm" if mask_mode == "none" else "fake_fasta_pangenome_grm_tested_proxy_masked"
    return KinshipResult(
        K=K,
        n_loci_used=int(used),
        n_loci_filtered=int(n_total - used),
        mean_diag_before_norm=float(mean_diag),
        source=source,
        details=details,
    )


@dataclass
class ResponseMaskedGRMWorkspace:
    """Workspace for response-locus-specific fake-FASTA GRM masking.

    The baseline GRM contribution is stored as an unnormalised sum of standardised
    marker outer products. For each response locus, only direct proxies of that
    response are subtracted before the response-specific K is normalised and
    eigendecomposed by the scanner.
    """

    X: np.ndarray
    valid_loci: np.ndarray
    Z_valid: np.ndarray
    K_sum: np.ndarray
    sums_valid: np.ndarray
    min_mac: int = 2
    dtype: str = "float64"
    mask_r2: float = 0.8
    mask_mismatch: float = 0.02
    grm_chunk_size: int = 2048
    shrinkage: float = 0.0
    n_total_loci: int = 0
    mean_diag_all_before_norm: float = float("nan")

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_valid(self) -> int:
        return int(self.valid_loci.size)

    def proxy_mask_for_response(self, response: int) -> tuple[np.ndarray, dict[str, int | float]]:
        """Return valid-GRM-locus mask for loci directly proxying response."""
        n = self.n_samples
        response = int(response)
        valid_loci = self.valid_loci
        mask = np.zeros(valid_loci.size, dtype=bool)
        valid_pos = {int(l): i for i, l in enumerate(valid_loci)}
        direct_masked = 0
        if response in valid_pos:
            mask[valid_pos[response]] = True
            direct_masked = 1

        x = self.X[:, response].astype(np.float64, copy=False)
        sx = float(x.sum())
        denom_x = sx * (n - sx)
        n_pairs_scanned = 0
        if denom_x > 0:
            for gs in range(0, valid_loci.size, self.grm_chunk_size):
                g_loci = valid_loci[gs:gs + self.grm_chunk_size]
                G = self.X[:, g_loci].astype(np.float64, copy=False)
                sum_g = self.sums_valid[gs:gs + g_loci.size]
                n11 = x @ G
                n10 = sx - n11
                n01 = sum_g - n11
                n00 = n - n11 - n10 - n01
                mismatch_same = n10 + n01
                mismatch_complement = n11 + n00
                proxy = (mismatch_same <= float(self.mask_mismatch) * n) | (mismatch_complement <= float(self.mask_mismatch) * n)

                if self.mask_r2 is not None and float(self.mask_r2) > 0:
                    denom = denom_x * (sum_g * (n - sum_g))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        phi = (n11 * n00 - n10 * n01) / np.sqrt(denom)
                        r2 = phi * phi
                    proxy |= np.isfinite(r2) & (r2 >= float(self.mask_r2))

                if np.any(proxy):
                    mask[gs:gs + g_loci.size] |= proxy
                n_pairs_scanned += int(g_loci.size)

        return mask, {
            "response": int(response),
            "n_valid_loci_before_response_mask": int(valid_loci.size),
            "n_response_direct_masked": int(direct_masked),
            "n_response_proxy_masked_total": int(mask.sum()),
            "n_response_proxy_masked_beyond_direct": int(mask.sum() - direct_masked),
            "n_response_proxy_pairs_scanned": int(n_pairs_scanned),
            "response_mask_r2": float(self.mask_r2),
            "response_mask_mismatch": float(self.mask_mismatch),
        }

    def build_for_response(self, response: int) -> KinshipResult:
        """Build response-specific K by subtracting response-proxy marker contributions."""
        mask, details = self.proxy_mask_for_response(int(response))
        used = int(self.valid_loci.size - int(mask.sum()))
        if used <= 0:
            raise ValueError(
                f"No GRM loci remain for response {response} after response-proxy masking. "
                "Relax --grm-mask-r2/--grm-mask-mismatch or use --grm-mask-mode none."
            )
        K_sum = self.K_sum.copy()
        if np.any(mask):
            Zm = self.Z_valid[:, mask].astype(np.float64, copy=False)
            K_sum -= Zm @ Zm.T
        K = K_sum / float(used)
        K = 0.5 * (K + K.T)
        mean_diag = float(np.mean(np.diag(K)))
        if not np.isfinite(mean_diag) or mean_diag <= 0:
            raise ValueError(f"Invalid response-masked GRM diagonal scale for response {response}")
        K /= mean_diag
        if self.shrinkage > 0.0:
            K = (1.0 - float(self.shrinkage)) * K + float(self.shrinkage) * np.eye(K.shape[0], dtype=K.dtype)
            K = 0.5 * (K + K.T)
            md = float(np.mean(np.diag(K)))
            if np.isfinite(md) and md > 0:
                K /= md
        details.update({
            "n_samples": int(self.X.shape[0]),
            "n_loci": int(self.n_total_loci),
            "n_loci_filtered_mac_or_monomorphic": int(self.n_total_loci - self.valid_loci.size),
            "n_loci_used_after_response_mask": int(used),
            "n_loci_removed_by_response_mask": int(mask.sum()),
            "grm_shrinkage": float(self.shrinkage),
        })
        out_dtype = np.float64 if self.dtype == "float64" else np.float32
        return KinshipResult(
            K=K.astype(out_dtype, copy=False),
            n_loci_used=int(used),
            n_loci_filtered=int(self.n_total_loci - used),
            mean_diag_before_norm=mean_diag,
            source="fake_fasta_pangenome_grm_response_proxy_masked",
            details=details,
        )

    def summary_result(self) -> KinshipResult:
        details = {
            "n_samples": int(self.X.shape[0]),
            "n_loci": int(self.n_total_loci),
            "mask_mode": "response",
            "mask_r2": float(self.mask_r2),
            "mask_mismatch": float(self.mask_mismatch),
            "grm_shrinkage": float(self.shrinkage),
            "n_valid_loci_before_response_mask": int(self.valid_loci.size),
            "n_loci_filtered_mac_or_monomorphic": int(self.n_total_loci - self.valid_loci.size),
            "response_specific_K": 1,
        }
        return KinshipResult(
            K=np.zeros((0, 0), dtype=np.float64),
            n_loci_used=int(self.valid_loci.size),
            n_loci_filtered=int(self.n_total_loci - self.valid_loci.size),
            mean_diag_before_norm=float(self.mean_diag_all_before_norm),
            source="fake_fasta_pangenome_grm_response_proxy_masked",
            details=details,
        )


def build_response_masked_grm_workspace(
    X: np.ndarray,
    min_mac: int = 2,
    chunk_size: int = 4096,
    dtype: str = "float64",
    progress: bool = True,
    mask_r2: float = 0.8,
    mask_mismatch: float = 0.02,
    mask_grm_chunk_size: int = 2048,
    shrinkage: float = 0.0,
) -> ResponseMaskedGRMWorkspace:
    """Precompute workspace for response-locus-specific proxy-masked GRMs.

    This keeps all valid background loci available globally, but for each response
    subtracts only loci that are direct high-r2/near-copy/near-complement proxies
    of that response. This avoids the v0.4.0 behaviour of globally removing every
    candidate-pair locus, which weakened population-structure correction.
    """
    if not (0.0 <= float(shrinkage) < 1.0):
        raise ValueError("shrinkage must be in [0, 1)")
    n, m = X.shape
    valid_loci = _valid_grm_loci(X, min_mac=min_mac)
    if valid_loci.size == 0:
        raise ValueError("No loci passed min_mac filter for response-masked GRM workspace")
    z_dtype = np.float64 if dtype == "float64" else np.float32
    Z_valid = np.empty((n, valid_loci.size), dtype=z_dtype)
    K_sum = np.zeros((n, n), dtype=np.float64)
    sums_valid = np.empty(valid_loci.size, dtype=np.float64)
    t0 = time.time()
    filled = 0
    for start in range(0, valid_loci.size, chunk_size):
        sub = valid_loci[start:start + chunk_size]
        B = X[:, sub].astype(np.float64, copy=False)
        p = B.mean(axis=0)
        denom = np.sqrt(p * (1.0 - p))
        # valid_loci should already pass this, but keep robust screening.
        ok = np.isfinite(denom) & (denom > 0)
        if not np.all(ok):
            sub = sub[ok]
            B = B[:, ok]
            p = p[ok]
            denom = denom[ok]
        Z = (B - p) / denom
        end = filled + Z.shape[1]
        Z_valid[:, filled:end] = Z.astype(z_dtype, copy=False)
        K_sum += Z @ Z.T
        sums_valid[filled:end] = B.sum(axis=0)
        filled = end
        if progress and (start == 0 or start + chunk_size >= valid_loci.size or ((start // max(1, chunk_size)) % 10 == 0)):
            elapsed = time.time() - t0
            sys.stderr.write(
                f"[pair-lmm-gwes] response_mask_workspace_progress valid_loci={min(start+chunk_size,valid_loci.size)}/{valid_loci.size} "
                f"elapsed={elapsed:.1f}s\n"
            )
            sys.stderr.flush()
    if filled != valid_loci.size:
        valid_loci = valid_loci[:filled]
        Z_valid = Z_valid[:, :filled]
        sums_valid = sums_valid[:filled]
    mean_diag = float(np.mean(np.diag(K_sum / float(valid_loci.size))))
    return ResponseMaskedGRMWorkspace(
        X=X,
        valid_loci=valid_loci.astype(np.int64, copy=False),
        Z_valid=Z_valid,
        K_sum=K_sum,
        sums_valid=sums_valid,
        min_mac=int(min_mac),
        dtype=dtype,
        mask_r2=float(mask_r2),
        mask_mismatch=float(mask_mismatch),
        grm_chunk_size=int(mask_grm_chunk_size),
        shrinkage=float(shrinkage),
        n_total_loci=int(m),
        mean_diag_all_before_norm=mean_diag,
    )

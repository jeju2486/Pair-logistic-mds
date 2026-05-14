from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class FastaMatrix:
    sample_names: list[str]
    Y: np.ndarray  # N x L uint8


def read_fake_fasta(path: str | Path, presence_char: str = "C", absence_char: str = "A") -> FastaMatrix:
    """Read a fake FASTA presence/absence matrix.

    Each sequence must have the same length. Presence and absence are encoded by
    `presence_char` and `absence_char`; any other character raises an error.
    Returns samples in FASTA order and a uint8 matrix with 1=presence, 0=absence.
    """
    path = Path(path)
    presence_char = presence_char.upper()
    absence_char = absence_char.upper()
    if len(presence_char) != 1 or len(absence_char) != 1:
        raise ValueError("presence_char and absence_char must be single characters")
    if presence_char == absence_char:
        raise ValueError("presence_char and absence_char must differ")

    names: list[str] = []
    seqs: list[str] = []
    current_name: str | None = None
    current_chunks: list[str] = []

    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    names.append(current_name)
                    seqs.append("".join(current_chunks).upper())
                current_name = line[1:].split()[0]
                current_chunks = []
            else:
                if current_name is None:
                    raise ValueError(f"FASTA sequence found before header in {path}")
                current_chunks.append(line)
        if current_name is not None:
            names.append(current_name)
            seqs.append("".join(current_chunks).upper())

    if not names:
        raise ValueError(f"No FASTA records found in {path}")
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"Sequences have unequal lengths: {sorted(lengths)[:10]}")

    n = len(seqs)
    L = len(seqs[0])
    Y = np.empty((n, L), dtype=np.uint8)
    valid = {presence_char, absence_char}
    for i, seq in enumerate(seqs):
        bad = set(seq) - valid
        if bad:
            raise ValueError(
                f"Unexpected characters in sample {names[i]}: {sorted(bad)}. "
                f"Expected only {absence_char}/{presence_char}."
            )
        # Vectorized conversion through bytes is faster than per-character loops.
        arr = np.frombuffer(seq.encode("ascii"), dtype="S1")
        Y[i, :] = (arr == presence_char.encode("ascii")).astype(np.uint8)

    return FastaMatrix(names, Y)


def read_pairs(path: str | Path) -> pd.DataFrame:
    """Read whitespace-delimited pair file.

    Uses first three columns as u, v, distance. Extra columns are ignored.
    The locus indices are assumed to be 0-based.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", usecols=[0, 1, 2])
    df.columns = ["u", "v", "distance"]
    df["u"] = df["u"].astype(int)
    df["v"] = df["v"].astype(int)
    return df


def align_matrix_to_tips(
    sample_names: list[str],
    Y: np.ndarray,
    tip_names: list[str],
    allow_intersection: bool = False,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Reorder FASTA matrix to match tree tip order.

    If allow_intersection is False, all tree tips must be present in the FASTA and
    all FASTA samples must be present in the tree. If True, only shared samples
    are retained in tree order.

    Returns (aligned_names, Y_aligned, dropped_names).
    """
    sample_to_idx = {s: i for i, s in enumerate(sample_names)}
    tip_set = set(tip_names)
    sample_set = set(sample_names)

    if allow_intersection:
        keep_tips = [t for t in tip_names if t in sample_to_idx]
        dropped = sorted((tip_set ^ sample_set) | (sample_set - set(keep_tips)))
        if len(keep_tips) < 2:
            raise ValueError("Fewer than two shared samples between tree and FASTA")
    else:
        missing_in_fasta = sorted(tip_set - sample_set)
        missing_in_tree = sorted(sample_set - tip_set)
        if missing_in_fasta or missing_in_tree:
            msg = []
            if missing_in_fasta:
                msg.append(f"Tree tips missing in FASTA: {missing_in_fasta[:10]}")
            if missing_in_tree:
                msg.append(f"FASTA samples missing in tree: {missing_in_tree[:10]}")
            raise ValueError("; ".join(msg))
        keep_tips = tip_names
        dropped = []

    idx = [sample_to_idx[t] for t in keep_tips]
    return keep_tips, Y[idx, :], dropped


def validate_pairs(pairs: pd.DataFrame, n_loci: int, drop_invalid: bool = False) -> pd.DataFrame:
    valid = (
        (pairs["u"] >= 0)
        & (pairs["v"] >= 0)
        & (pairs["u"] < n_loci)
        & (pairs["v"] < n_loci)
        & (pairs["u"] != pairs["v"])
    )
    if not valid.all():
        n_bad = int((~valid).sum())
        if drop_invalid:
            pairs = pairs.loc[valid].copy()
        else:
            raise ValueError(f"Pair file contains {n_bad} invalid/self/out-of-range pairs")
    return pairs.reset_index(drop=True)

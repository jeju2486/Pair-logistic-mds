from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FastaMatrix:
    sample_names: list[str]
    X: np.ndarray  # samples x loci, uint8, 1=presence, 0=absence


def read_fake_fasta(path: str | Path, presence_char: str = "C", absence_char: str = "A") -> FastaMatrix:
    """Read fake FASTA as a binary samples x loci matrix.

    Default encoding follows PAN-GWES-style fake FASTA: A=absent, C=present.
    """
    path = Path(path)
    pchar = presence_char.upper()
    achar = absence_char.upper()
    if len(pchar) != 1 or len(achar) != 1 or pchar == achar:
        raise ValueError("presence_char and absence_char must be different single characters")

    names: list[str] = []
    seqs: list[str] = []
    name: Optional[str] = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    names.append(name)
                    seqs.append("".join(chunks).upper())
                name = line[1:].split()[0]
                chunks = []
            else:
                if name is None:
                    raise ValueError(f"FASTA sequence encountered before header in {path}")
                chunks.append(line)
        if name is not None:
            names.append(name)
            seqs.append("".join(chunks).upper())

    if not names:
        raise ValueError(f"No FASTA records found in {path}")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate sample names found in FASTA")
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"FASTA records have unequal lengths: {sorted(lengths)[:10]}")

    n = len(seqs)
    L = len(seqs[0])
    X = np.empty((n, L), dtype=np.uint8)
    valid = {pchar, achar}
    pbyte = pchar.encode("ascii")
    for i, seq in enumerate(seqs):
        bad = set(seq) - valid
        if bad:
            raise ValueError(
                f"Unexpected characters in {names[i]}: {sorted(bad)}. "
                f"Expected only {achar}/{pchar}."
            )
        arr = np.frombuffer(seq.encode("ascii"), dtype="S1")
        X[i, :] = (arr == pbyte).astype(np.uint8)
    return FastaMatrix(names, X)


def read_pairs(path: str | Path) -> pd.DataFrame:
    """Read PAN-GWES/SpydrPick candidate pair file.

    If no header is detected, columns are interpreted as:
    u, v, distance, ARACNE, MI, count, M2, min_distance, max_distance, ...

    If a header is detected, u/v columns are inferred from common names or from the
    first two columns.
    """
    path = Path(path)
    first = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                first = line
                break
    if first is None:
        raise ValueError(f"No pair rows found in {path}")

    toks = first.replace(",", "\t").split()
    try:
        int(toks[0]); int(toks[1])
        no_header = True
    except Exception:
        no_header = False

    if no_header:
        default_cols = ["u", "v", "distance", "ARACNE", "MI", "count", "M2", "min_distance", "max_distance"]
        df = pd.read_csv(path, sep=r"\s+|,", engine="python", header=None, comment="#")
        if df.shape[1] <= len(default_cols):
            df.columns = default_cols[:df.shape[1]]
        else:
            df.columns = default_cols + [f"extra_{i+1}" for i in range(df.shape[1] - len(default_cols))]
    else:
        df = pd.read_csv(path, sep=r"\s+|,", engine="python", comment="#")
        lower = {c.lower(): c for c in df.columns}
        u_names = ["u", "unitig_i", "i", "locus_i", "marker_i", "variant_i"]
        v_names = ["v", "unitig_j", "j", "locus_j", "marker_j", "variant_j"]
        u_col = next((lower[x] for x in u_names if x in lower), df.columns[0])
        v_col = next((lower[x] for x in v_names if x in lower), df.columns[1])
        df = df.rename(columns={u_col: "u", v_col: "v"})

    df["u"] = df["u"].astype(int)
    df["v"] = df["v"].astype(int)
    return df


def validate_pairs(pairs: pd.DataFrame, n_loci: int, drop_invalid: bool = False) -> pd.DataFrame:
    valid = (
        (pairs["u"] >= 0) & (pairs["v"] >= 0) &
        (pairs["u"] < n_loci) & (pairs["v"] < n_loci) &
        (pairs["u"] != pairs["v"])
    )
    if not bool(valid.all()):
        n_bad = int((~valid).sum())
        if not drop_invalid:
            raise ValueError(f"Pair file contains {n_bad} invalid/self/out-of-range pairs")
        pairs = pairs.loc[valid].copy()
    return pairs.reset_index(drop=True)


def read_labelled_distance_matrix(path: str | Path, sample_names: list[str]) -> np.ndarray:
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+|,", engine="python", index_col=0)
    missing_rows = [s for s in sample_names if s not in df.index]
    missing_cols = [s for s in sample_names if s not in df.columns]
    if missing_rows or missing_cols:
        raise ValueError(
            f"Distance matrix does not contain all FASTA samples. "
            f"Missing rows: {missing_rows[:5]}; missing columns: {missing_cols[:5]}"
        )
    D = df.loc[sample_names, sample_names].to_numpy(dtype=float)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D

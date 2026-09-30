"""Create synthetic scanner output, then run both downstream helpers.

Run from the repository root: python examples/downstream/reproduce.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from ko_variation.scan import ScanConfig, scan_pairs_glmm
from ko_variation.annotation_cli import main


def reproduce():
    root = Path(__file__).resolve().parent / "output"
    root.mkdir(parents=True, exist_ok=True)
    # Each quartet includes all four cells; synthetic columns repeat patterns.
    matrix = np.array([[1, 1]] * 30 + [[1, 0]] * 10 + [[0, 1]] * 10 + [[0, 0]] * 30, dtype=np.uint8)
    matrix = np.column_stack([matrix, matrix[:, 0], 1 - matrix[:, 1]])
    pairs = pd.DataFrame(dict(u=[0, 0, 2], v=[1, 3, 3], distance=[20000, 50000, 30000]))
    # Zero covariance provides an exact no-relatedness reproducibility oracle.
    results, _ = scan_pairs_glmm(pairs, matrix, np.zeros((80, 80)), ScanConfig(progress=False, spa_mode="off"))
    results.to_csv(root / "ko_variation.tsv", sep="\t", index=False)
    pd.DataFrame(dict(locus=[0, 1, 2, 3], gene=["gene_A", "gene_B", "gene_A", "gene_C"],
                      label=["Gene A", "Gene B", "Gene A", "Gene C"], group=["Group 1", "Group 2", "Group 1", "Group 2"],
                      product=["Synthetic annotation"] * 4)).to_csv(root / "annotation.tsv", sep="\t", index=False)
    return main(["--results", str(root / "ko_variation.tsv"), "--annotation", str(root / "annotation.tsv"),
                 "--out", str(root / "network"), "--network"])


if __name__ == "__main__":
    raise SystemExit(reproduce())

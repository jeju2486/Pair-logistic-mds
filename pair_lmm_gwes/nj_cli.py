from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_fake_fasta
from .nj_tree import build_nj_tree_from_binary_matrix


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build a midpoint-rooted neighbour-joining event tree from a pair-lmm-gwes fake FASTA."
    )
    p.add_argument("--fasta", required=True, help="Fake FASTA matrix; A=0, C=1 by default.")
    p.add_argument("--out", required=True, help="Output Newick tree path.")
    p.add_argument("--presence-char", default="C")
    p.add_argument("--absence-char", default="A")
    p.add_argument("--nj-min-mac", type=int, default=2)
    p.add_argument("--nj-max-loci", type=int, default=50000, help="0 means all eligible loci.")
    p.add_argument("--nj-seed", type=int, default=1)
    p.add_argument("--nj-chunk-size", type=int, default=2048)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    fm = read_fake_fasta(args.fasta, presence_char=args.presence_char, absence_char=args.absence_char)
    res = build_nj_tree_from_binary_matrix(
        fm.X,
        fm.sample_names,
        Path(args.out),
        min_mac=args.nj_min_mac,
        max_loci=args.nj_max_loci,
        seed=args.nj_seed,
        chunk_size=args.nj_chunk_size,
        progress=not args.no_progress,
    )
    print(f"wrote\t{res.tree_path}")
    for k, v in res.details().items():
        print(f"{k}\t{v}")


if __name__ == "__main__":
    main()

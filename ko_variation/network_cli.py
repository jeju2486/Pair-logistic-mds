"""Redraw an existing selected-pair effect table without annotation or refitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
from .network import build_gene_network, export_gene_network
from .network_display import add_network_arguments, network_options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", required=True, help="Completed native OUT.distal.tsv, including gene mappings and fitted effects")
    parser.add_argument("--out", required=True)
    parser.add_argument("--missing-genes", choices=["drop", "error"], default="drop")
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="")
    add_network_arguments(parser)
    args = parser.parse_args(argv)
    outputs = [Path(args.out + suffix) for suffix in (".png", ".svg", ".html", ".nodes.tsv", ".edges.tsv", ".layout.json", ".network.json")]
    if any(path.resolve() == Path(args.effects).resolve() for path in outputs):
        parser.error("Network outputs must not overwrite the effect table")
    try:
        print(f"[network] reading saved effects: {args.effects}; no refitting", flush=True)
        signals = pd.read_csv(args.effects, sep="\t", dtype={"u_gene": str, "v_gene": str})
        nodes, edges = build_gene_network(signals, missing_genes=args.missing_genes, include_self=args.include_self)
        paths = export_gene_network(nodes, edges, args.out, seed=args.seed, dpi=args.dpi, title=args.title, **network_options(args))
        metadata = dict(source_effects=str(Path(args.effects).resolve()), selected_pairs=len(signals),
                        nodes=len(nodes), edges=len(edges), refitted=False, display=network_options(args),
                        outputs={key: str(path) for key, path in paths.items()})
        Path(args.out + ".network.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[network] exported {len(nodes)} genes and {len(edges)} edges: {args.out}", flush=True)
    except (ValueError, OSError, ImportError) as exc:
        sys.stderr.write(f"Network error: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

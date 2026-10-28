"""Small-panel production benchmark; never enumerate a full genome by default.

Supply a small pseudo-FASTA panel with unique sample IDs and the matching tree.
This compares the release code paths, not a separate scoring implementation.
"""
import argparse
import csv
import json
from pathlib import Path
import sqlite3

from ko_variation.discovery import main
from ko_variation.discovery_store import records


def benchmark():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta",type=Path,required=True)
    p.add_argument("--tree",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--memory-gb",type=float,default=4)
    p.add_argument("--min-cell-count",type=int,default=20)
    p.add_argument("--max-loci",type=int,default=2000)
    args=p.parse_args()
    _,sequence=next(records(args.fasta))
    if len(sequence)>args.max_loci:
        p.error("Input exceeds --max-loci; use a frozen small panel, not the full genome")
    args.out.mkdir(parents=True,exist_ok=True)
    common=["--fasta",str(args.fasta),"--tree",str(args.tree),"--threads",str(args.threads),
            "--memory-gb",str(args.memory_gb),"--min-cell-count",str(args.min_cell_count),"--screen-p","0.01"]
    results=[]
    reference=set()
    for name,options in (("exact",["--search","exhaustive","--exact"]),
                         ("bounds",["--search","exhaustive"]),("hierarchy_bounds",[])):
        out=args.out/name
        command=common+["--out",str(out)]+options
        if (out/".kovar_work"/"run_identity.json").exists():
            command += ["--resume"]
        if name!="exact":
            command += ["--reuse-nulls-from",str(args.out/"exact")]
        main(command)
        summary=json.loads((out/"run_summary.json").read_text())
        db=sqlite3.connect(out/".kovar_work"/"work.sqlite")
        hits=set(db.execute("SELECT u,v FROM decisions WHERE final=1"))
        db.close()
        if name=="exact":
            reference=hits
        results.append(dict(method=name,seconds=summary["seconds_this_invocation"],
                            queries=summary["queried_pairs"],reference_hits=len(reference),hits=len(hits),
                            missed=len(reference-hits),extra=len(hits-reference),
                            recall=len(reference&hits)/len(reference) if reference else "NA"))
    with (args.out/"comparison.tsv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]),delimiter="\t",lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


if __name__=="__main__":
    benchmark()

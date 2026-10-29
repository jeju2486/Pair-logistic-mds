"""Small-panel production benchmark; never enumerate a full genome by default.

Supply a small pseudo-FASTA panel with unique sample IDs and the matching tree.
This compares the release code paths, not a separate scoring implementation.
"""
import argparse
import csv
import json
from pathlib import Path
import sqlite3

from ko_variation.discovery import main,identity,parser as discovery_parser
from ko_variation.discovery_store import records
from ko_variation.memory_budget import detect_budget


def benchmark():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta",type=Path,required=True)
    p.add_argument("--tree",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--memory-gb",type=float,default=None)
    p.add_argument("--min-cell-count",type=int,default=20)
    p.add_argument("--max-loci",type=int,default=2000)
    p.add_argument("--screen-p",type=float,default=.001)
    p.add_argument("--bonferroni-tests",type=int,help="Freeze the full study's hypothesis family for sampled panels")
    args=p.parse_args()
    _,sequence=next(records(args.fasta))
    if len(sequence)>args.max_loci:
        p.error("Input exceeds --max-loci; use a frozen small panel, not the full genome")
    args.out.mkdir(parents=True,exist_ok=True)
    common=["--fasta",str(args.fasta),"--tree",str(args.tree),"--threads",str(args.threads),
            "--min-cell-count",str(args.min_cell_count),"--screen-p",str(args.screen_p)]
    if args.memory_gb is not None:
        common += ['--memory-gb',str(args.memory_gb)]
    if args.bonferroni_tests is not None:
        common += ['--bonferroni-tests',str(args.bonferroni_tests)]
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
        saved=out/'run_summary.json'
        if saved.exists():
            summary=json.loads(saved.read_text())
            old_identity=json.loads((out/'.kovar_work'/'run_identity.json').read_text())
            current=identity(discovery_parser().parse_args(command))
            budget=detect_budget(args.memory_gb,args.threads)['working_gb']
            if old_identity!=current or summary['threads']!=args.threads or abs(summary['memory_gb']-budget)>1e-6:
                raise ValueError('Completed benchmark uses different code/settings/resources; choose a new --out')
            # Do not replace a completed method's measured time with the near-
            # zero time of simply reopening and exporting its checkpoint.
        else:
            main(command)
            summary=json.loads(saved.read_text())
        db=sqlite3.connect(out/".kovar_work"/"work.sqlite")
        hits=set(db.execute("SELECT u,v FROM decisions WHERE final=1"))
        db.close()
        if name=="exact":
            reference=hits
        metrics=summary.get('runtime_totals',{})
        results.append(dict(method=name,seconds=summary["seconds_this_invocation"],
                            queries=summary["queried_pairs"],reference_hits=len(reference),hits=len(hits),
                            missed=len(reference-hits),extra=len(hits-reference),
                            recall=len(reference&hits)/len(reference) if reference else "NA",
                            null_fit_worker_seconds=metrics.get('null_fit_worker_seconds',0),
                            cache_load_worker_seconds=metrics.get('cache_load_worker_seconds',0),
                            bounds_worker_seconds=metrics.get('bounds_worker_seconds',0),
                            exact_worker_seconds=metrics.get('exact_worker_seconds',0),
                            coordinator_db_read_seconds=metrics.get('driver_db_read_seconds',0),
                            coordinator_db_write_seconds=metrics.get('driver_db_write_seconds',0),
                            jobs=metrics.get('jobs',0),
                            mean_pairs_per_job=metrics.get('pairs',0)/max(1,metrics.get('jobs',0))))
    with (args.out/"comparison.tsv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]),delimiter="\t",lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


if __name__=="__main__":
    benchmark()

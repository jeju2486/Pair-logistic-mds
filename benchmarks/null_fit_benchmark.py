#!/usr/bin/env python3
"""Read-only whole-null-fit comparison of NumPy and SciPy LAPACK drivers.

Run against an existing .kovar_work directory. Results do not overwrite fitted
models or select a production solver automatically. Each worker uses one BLAS
thread. Compare this on the actual deployment host before changing a driver.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
from scipy import linalg

from ko_variation import discovery_score as kernels
from ko_variation.discovery_execution import bounded_map
from ko_variation.discovery_store import atomic_json
from ko_variation.glmm import fit_null_glmm
from ko_variation.memory_budget import detect_budget,plan_workers


def compare(pattern):
    state=kernels._STATE
    y=kernels.bits([pattern])[0].astype(float)
    results=[]
    reference=None
    for driver in ('numpy','evd','evr'):
        timing={}
        def solver(matrix,UPLO='L'):
            return linalg.eigh(np.array(matrix,order='F',copy=True),driver=driver,
                               lower=UPLO=='L',check_finite=False,overwrite_a=True)
        started=time.perf_counter()
        if driver=='numpy':
            fit,cache=fit_null_glmm(pattern,y,state['K'],kinship_eigensystem=state['eigen'],timing=timing)
        else:
            # Isolated process, no parallel threads or persistent monkey patch.
            with patch('ko_variation.glmm.np.linalg.eigh',new=solver):
                fit,cache=fit_null_glmm(pattern,y,state['K'],kinship_eigensystem=state['eigen'],timing=timing)
        seconds=time.perf_counter()-started
        if reference is None: reference=fit
        probability_difference=float(np.max(np.abs(fit.fitted_probability-reference.fitted_probability)))
        tau_difference=float(abs(fit.tau-reference.tau)/max(1,abs(reference.tau)))
        results.append(dict(pattern=pattern,driver=driver,status=fit.status,iterations=fit.iterations,
            seconds=seconds,weighted_eigendecomposition=timing.get('weighted_eigendecomposition',0),
            tau_profiling=timing.get('tau_profiling',0),tau=fit.tau,
            status_matches_reference=fit.status==reference.status,
            max_probability_difference=probability_difference if np.isfinite(probability_difference) else None,
            tau_relative_difference=tau_difference if np.isfinite(tau_difference) else None))
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True,help='New report prefix; .tsv and .json')
    p.add_argument('--patterns',type=int,default=12)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--memory-gb',type=float)
    args=p.parse_args(argv)
    if min(args.patterns,args.threads)<1: raise ValueError('Counts must be positive')
    info=json.loads((args.work/'prepared.json').read_text())
    budget=detect_budget(args.memory_gb,args.threads)
    plan_workers(budget,info['n'],args.threads,256)
    ids=sorted(np.random.default_rng(args.seed).choice(info['patterns'],
               size=min(args.patterns,info['patterns']),replace=False).tolist())
    rows=[]
    with ProcessPoolExecutor(max_workers=args.threads,mp_context=multiprocessing.get_context('spawn'),
            initializer=kernels.initialize,initargs=(str(args.work),info,dict(cache_bytes_per_worker=0))) as pool:
        for result in bounded_map(pool,compare,ids,args.threads,ordered=False):
            rows.extend(result)
            print(f"[null benchmark] {len(rows)//3}/{len(ids)} patterns",flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.with_suffix('.tsv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(sorted(rows,key=lambda r:(r['pattern'],r['driver'])))
    medians={d:statistics.median(r['seconds'] for r in rows if r['driver']==d) for d in ('numpy','evd','evr')}
    atomic_json(args.out.with_suffix('.json'),dict(patterns=ids,seed=args.seed,threads=args.threads,
        medians=medians,speedup_vs_numpy={d:medians['numpy']/s for d,s in medians.items()},
        all_statuses_match=all(r['status_matches_reference'] for r in rows),
        max_probability_difference=max((r['max_probability_difference'] for r in rows
                                       if r['max_probability_difference'] is not None),default=None),
        max_tau_relative_difference=max((r['tau_relative_difference'] for r in rows
                                        if r['tau_relative_difference'] is not None),default=None),
        production_solver_changed=False,note='Whole-fit local timings; drivers may alter convergence and require review.'))


if __name__=='__main__': main()

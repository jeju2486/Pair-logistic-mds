"""Resumable, bounded pre-fitting of the complete unique response pool."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
import time

import numpy as np

from . import discovery_score as kernels
from .discovery_execution import bounded_map
from .discovery_store import atomic_json


def publish_aliases(work, aliases):
    path=work/'null_canonical.npy'
    temporary=work/'null_canonical.partial.npy'
    np.save(temporary,aliases)
    os.replace(temporary,path)


def aliases_for_pool(db, work, info, *, reuse_complements=True):
    """Preserve native saved fits; alias only responses without their own fit."""
    db.execute('CREATE TABLE IF NOT EXISTS null_aliases(pid INTEGER PRIMARY KEY,canonical INTEGER)')
    db.execute('CREATE TABLE IF NOT EXISTS null_stage(pid INTEGER PRIMARY KEY,status TEXT,mode TEXT)')
    if db.execute('SELECT count(*) FROM null_aliases').fetchone()[0]==info['patterns']:
        aliases=np.fromiter((r[0] for r in db.execute('SELECT canonical FROM null_aliases ORDER BY pid')),
                            dtype=np.int64,count=info['patterns'])
        publish_aliases(work,aliases)
        return aliases
    existing=np.zeros(info['patterns'],dtype=bool)
    for path in (work/'null_models').glob('*/null_*.json'):
        pid=int(path.stem.split('_')[1])
        if 0<=pid<len(existing): existing[pid]=True
    for pid,path in db.execute('SELECT pid,path FROM external_nulls'):
        if os.path.isfile(path): existing[pid]=True
    aliases=np.arange(info['patterns'],dtype=np.int64)
    last=time.monotonic()
    if reuse_complements:
        for pid,raw in db.execute('SELECT id,bits FROM patterns ORDER BY id'):
            complement=bytearray(value^255 for value in raw)
            if info['n']%8:
                complement[-1]&=(1<<(info['n']%8))-1
            found=db.execute('SELECT id FROM patterns WHERE bits=?',(bytes(complement),)).fetchone()
            if found and found[0]>pid:
                other=found[0]
                if not existing[pid]: aliases[pid]=other if existing[other] else pid
                if not existing[other]: aliases[other]=pid
            if time.monotonic()-last>25:
                print(f'[KOVAR] null inventory: {pid+1:,}/{len(aliases):,} patterns',flush=True)
                last=time.monotonic()
    with db:
        db.execute('DELETE FROM null_aliases')
        db.executemany('INSERT INTO null_aliases VALUES (?,?)',enumerate(map(int,aliases)))
    publish_aliases(work,aliases)
    return aliases


def prefit(db, work, info, args, settings, get_meta, set_meta, merge_metrics):
    aliases=aliases_for_pool(db,work,info,reuse_complements=args.prefit_nulls=='all')
    if args.prefit_nulls!='all':
        return
    started=time.monotonic()
    summary={'stage':'null_fitting','response_patterns':info['patterns'],'completed':0}
    with db:
        set_meta(db,'active_stage','null_fitting')
    def run(ids):
        pending=[]
        totals={}
        states={}
        completed=0
        last=time.monotonic()
        def flush():
            if not pending: return
            with db:
                db.executemany('INSERT OR REPLACE INTO null_stage VALUES (?,?,?)',pending)
                saved=get_meta(db,'prefit_runtime_totals',{})
                merge_metrics(saved,totals)
                set_meta(db,'prefit_runtime_totals',saved)
                set_meta(db,'worker_state',states)
                set_meta(db,'null_stage_progress',dict(checked=completed,total=len(ids),
                         seconds=time.monotonic()-started))
            pending.clear();totals.clear()
        def jobs():
            for pid in ids:
                row=db.execute('SELECT path FROM external_nulls WHERE pid=?',(int(pid),)).fetchone()
                yield int(pid),row[0] if row else None
        # Dedicated dynamic process queue for independent fits. Scoring later
        # uses response affinity; no nested BLAS threads exceed --threads.
        with ProcessPoolExecutor(max_workers=args.threads,mp_context=multiprocessing.get_context('spawn'),
                initializer=kernels.initialize,initargs=(str(work),info,settings)) as pool:
            for payload in bounded_map(pool,kernels.prefit_profiled,jobs(),args.threads,ordered=False):
                completed+=1
                pending.append((payload['pattern'],payload['status'],payload['mode']))
                merge_metrics(totals,payload['metrics'])
                states[str(payload['worker_state']['pid'])]=payload['worker_state']
                if len(pending)>=256 or time.monotonic()-last>25:
                    flush()
                    print(f'[KOVAR] null fitting: {completed:,}/{len(ids):,} canonical patterns checked; '
                          f'{time.monotonic()-started:.0f}s this stage',flush=True)
                    last=time.monotonic()
            flush()
    run(np.flatnonzero(aliases==np.arange(len(aliases))))
    # A failed canonical fit must not suppress a potentially successful native
    # complement fit. Restore its own response and fit it independently once.
    failures={pid for pid,status in db.execute('SELECT pid,status FROM null_stage') if status!='OK'}
    retry=[pid for pid,canonical in enumerate(aliases) if pid!=canonical and int(canonical) in failures]
    if retry:
        for pid in retry: aliases[pid]=pid
        with db:
            db.executemany('UPDATE null_aliases SET canonical=pid WHERE pid=?',((p,) for p in retry))
        publish_aliases(work,aliases)
        run(retry)
    counts=dict(db.execute('SELECT status,count(*) FROM null_stage GROUP BY status'))
    modes=dict(db.execute('SELECT mode,count(*) FROM null_stage GROUP BY mode'))
    summary.update(stage='null_fitting_complete',completed=sum(counts.values()),
        canonical_models=int(np.count_nonzero(aliases==np.arange(len(aliases)))),
        complement_aliases=int(np.count_nonzero(aliases!=np.arange(len(aliases)))),
        statuses=counts,cache_modes=modes,seconds_this_invocation=time.monotonic()-started)
    with db:
        set_meta(db,'null_stage_summary',summary)
    atomic_json(args.out/'null_stage_summary.json',summary)
    print(f"[KOVAR] null stage complete: {summary['completed']:,} models; "
          f"{summary['complement_aliases']:,} complement aliases; statuses={counts}",flush=True)

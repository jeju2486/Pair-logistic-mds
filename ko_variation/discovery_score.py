"""Response-major worker kernels: saved nulls, spectral bounds and distal beta.

Workers never write the search database. The coordinator commits complete
batches, so a killed worker cannot turn a partial calculation into a result.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .discovery_store import atomic_json, digest, open_patterns
from .glmm import (KinshipEigensystem, _WeightedEigen, fit_logistic_mixed,
                   fit_null_glmm, score_predictor_block)
from .memory_budget import ByteLRU, working_usage, process_rss, MIB

_STATE = {}
_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype="u1")


def initialize(work, info, settings):
    # One native numerical thread per worker: --threads is a TOTAL CPU budget.
    threadpool_limits(limits=1)
    work = Path(work)
    _STATE.clear()
    cache_limit=settings.get("cache_bytes_per_worker",0)
    _STATE.update(work=work, info=info, settings=settings,
                  packed=open_patterns(work, info), nulls=ByteLRU(cache_limit),
                  cache_limit=cache_limit,last_pressure=0.,verified=OrderedDict(),metrics={},last_log=0.,
                  K=np.load(work/"K.npy", mmap_mode="r"),
                  eigen=KinshipEigensystem(np.load(work/"K_values.npy", mmap_mode="r"),
                      np.load(work/"K_vectors.npy", mmap_mode="r"), info["shared_covariance"]))


def metric(key,value=1):
    values=_STATE.get("metrics")
    if values is not None:
        values[key]=values.get(key,0)+value


def pressure_guard():
    s=_STATE
    now=time.monotonic()
    if now-s["last_pressure"]<1:
        return
    s["last_pressure"]=now
    settings=s["settings"]
    usage=working_usage(settings.get("monitor"))
    target=settings.get("total_working_bytes")
    if usage is None:
        usage=process_rss()
        target=settings.get("worker_target_bytes")
    if usage is None or not target:
        return
    cache=s["nulls"]
    if usage>=target:
        cache.trim(int(cache.limit*.75))
        metric("memory_pressure_events")
    elif usage<target*.85 and cache.limit<s["cache_limit"]:
        cache.limit=min(s["cache_limit"],cache.limit+max(16*MIB,int(.1*s["cache_limit"])))


def cache_charge(cache):
    if cache is None:
        return 1024
    seen=set()
    size=4096
    arrays=[v for k,v in cache.items() if k!="K" and isinstance(v,np.ndarray)]
    arrays.extend(v for v in vars(cache["decomp"]).values() if isinstance(v,np.ndarray))
    for array in arrays:
        key=(array.__array_interface__["data"][0],array.nbytes)
        if key not in seen:
            seen.add(key)
            size+=array.nbytes
    return size


def verify_product(product,expected):
    # Rehash only when the immutable file's size/mtime or expected digest changed.
    # Validation is never skipped on a worker's first access to a saved model.
    stamp=product.stat()
    signature=(stamp.st_size,stamp.st_mtime_ns,expected)
    verified=_STATE["verified"]
    key=str(product)
    if verified.get(key)!=signature:
        started=time.perf_counter()
        if digest(product)!=expected:
            raise ValueError(f"Corrupted null cache: {product}")
        metric("cache_hash_worker_seconds",time.perf_counter()-started)
        verified[key]=signature
        # Small verification metadata has a separate bounded allowance.
        while len(verified)*512>max(MIB,int(.01*_STATE["cache_limit"])):
            verified.popitem(last=False)
    verified.move_to_end(key)


def bits(ids):
    return np.unpackbits(_STATE["packed"][np.asarray(ids, dtype=int)], axis=1,
                         bitorder="little")[:, :_STATE["info"]["n"]]


def eligibility_summary(job):
    """Packed Hamming radius and necessary marginal eligibility; no null fit."""
    node,center,ids,minimum = job
    packed = _STATE['packed'][np.asarray(ids,dtype=int)]
    radius = int(_POPCOUNT[np.bitwise_xor(packed,_STATE['packed'][center])].sum(axis=1).max())
    counts = _POPCOUNT[packed].sum(axis=1)
    n = _STATE['info']['n']
    eligible = int(np.count_nonzero((counts>=2*minimum) & (counts<=n-2*minimum)))
    return node,radius,eligible


def similarity(job):
    """Bounded binary-phi comparisons; no dense all-pattern matrix."""
    representative, ids, threshold = job
    x = _STATE["packed"][representative]
    X = _STATE["packed"][ids]
    n, k = _STATE["info"]["n"], float(_POPCOUNT[x].sum())
    counts = _POPCOUNT[X].sum(axis=1).astype(float)
    overlap = _POPCOUNT[np.bitwise_and(X,x)].sum(axis=1).astype(float)
    rho = (n*overlap-counts*k)/np.sqrt(counts*(n-counts)*k*(n-k))
    return [int(i) for i, r in zip(ids, rho) if r > 0 and r*r+1e-12 >= threshold]


def farthest(job):
    ids, representatives = job
    X, R = bits(ids).astype(float), bits(representatives).astype(float)
    n = X.shape[1]
    k, m = X.sum(axis=1), R.sum(axis=1)
    rho = (n*(X @ R.T)-k[:, None]*m)/np.sqrt((k*(n-k))[:, None]*(m*(n-m)))
    nearest = rho.max(axis=1)
    order = np.lexsort((ids, nearest))
    for index in order:
        if ids[index] not in representatives:
            return float(nearest[index]), int(ids[index])
    return float("inf"), -1


def standardized_sum(ids):
    X=bits(ids).astype(float)
    n=X.shape[1]
    k=X.sum(axis=1)
    return ((X-k[:,None]/n)/np.sqrt(k*(n-k)/n)[:,None]).sum(axis=0)


def central(job):
    ids,total=job
    X=bits(ids).astype(float)
    n=X.shape[1]
    k=X.sum(axis=1)
    values=((X-k[:,None]/n)/np.sqrt(k*(n-k)/n)[:,None]) @ total
    order=np.lexsort((ids,-values))
    return -float(values[order[0]]),int(ids[order[0]])


def get_null(pattern, external=None):
    s = _STATE
    pressure_guard()
    if s.get('active_null') is not None and s['active_null'][0]==pattern:
        metric('null_job_reuse_hits')
        return s['active_null'][1]
    if pattern in s["nulls"]:
        metric("null_cache_hits")
        value=s['nulls'].get(pattern)
        if 'active_null' in s:
            s['active_null']=(pattern,value)
        return value
    load_started=time.perf_counter()
    folder = s["work"]/"null_models"/f"{pattern//1000:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder/f"null_{pattern:09d}.json"
    array = meta.with_suffix(".npz")
    source = meta if meta.exists() else Path(external) if external else None
    y = bits([pattern])[0].astype(float)
    if source is not None:
        metric("null_cache_disk_loads")
        record = json.loads(source.read_text())
        cache = None
        if record["array"] is not None:
            product = source.parent/record["array"]["name"]
            verify_product(product,record["array"]["sha256"])
            with np.load(product, allow_pickle=False) as saved:
                cache = {key: saved[key] for key in ("y", "mu", "weights", "C", "Sinv_C", "M_inv")}
                cache["tau"] = float(saved["tau"])
                cache["decomp"] = _WeightedEigen(saved["sqrt_w"], saved["eigenvalues"],
                    saved["eigenvectors"], float(saved["logdet_w_inverse"]))
            if not np.array_equal(cache["y"], y):
                raise ValueError("Cached null response differs")
            verify_product(product,record["array"]["sha256"])
        if record["status"] == "OK" and cache is None:
            raise ValueError("Successful null cache has no arrays")
        metric("cache_load_worker_seconds",time.perf_counter()-load_started)
    else:
        if s["settings"]["require_cached_nulls"]:
            raise ValueError(f"No cached null for pattern {pattern}")
        started = time.monotonic()
        if started-s["last_log"] > 25:
            print(f"[KOVAR worker {os.getpid()}] fitting response pattern {pattern}",flush=True)
            s["last_log"]=started
        fit, cache = fit_null_glmm(pattern, y, s["K"], kinship_eigensystem=s["eigen"])
        metric("new_null_fits")
        metric("null_fit_worker_seconds",time.monotonic()-started)
        record = dict(pattern=pattern, status=fit.status,
                      fit_seconds=time.monotonic()-started, array=None)
        save_started=time.perf_counter()
        if fit.status == "OK" and cache:
            decomp = cache["decomp"]
            order = np.argsort(decomp.eigenvalues)[::-1]
            decomp.eigenvalues = decomp.eigenvalues[order]
            decomp.eigenvectors = np.ascontiguousarray(decomp.eigenvectors[:, order])
            temporary = array.with_suffix(f".partial.{os.getpid()}")
            with temporary.open("wb") as stream:
                np.savez(stream, **{k: cache[k] for k in ("y", "mu", "weights", "tau", "C", "Sinv_C", "M_inv")},
                    sqrt_w=decomp.sqrt_w, eigenvalues=decomp.eigenvalues,
                    eigenvectors=decomp.eigenvectors, logdet_w_inverse=decomp.logdet_w_inverse)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, array)
            record["array"] = dict(name=array.name, sha256=digest(array))
        else:
            cache = None
        atomic_json(meta, record)
        metric("cache_save_worker_seconds",time.perf_counter()-save_started)
    if cache is not None:
        cache["_inverse_spectrum"]=1/(1+cache["tau"]*cache["decomp"].eigenvalues)
    value = (cache, record["status"])
    s["nulls"].put(pattern,value,cache_charge(cache))
    if 'active_null' in s:
        s['active_null']=(pattern,value)
    return value


def adaptive_statistics(X, cache, criticals, ranks=(0, 8, 16, 32, 64, 128)):
    """Bound ordinary score statistics, requiring all requested cutoffs resolved.

    Outward roundoff margins prevent near-boundary decisions from being
    accepted prematurely. Singular/invalid cases use the existing exact core.
    """
    C, decomp = cache["C"], cache["decomp"]
    A = decomp.sqrt_w[:, None]*(X-C @ (cache["M_inv"] @ (cache["Sinv_C"].T @ X)))
    energy = np.sum(A*A, axis=0)
    U = X.T @ (cache["y"]-cache["mu"])
    d = cache.get("_inverse_spectrum")
    if d is None:
        d = 1/(1+cache["tau"]*decomp.eigenvalues)
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("Invalid inverse covariance spectrum")
    width = len(U)
    lo, hi = np.zeros(width), np.full(width, np.inf)
    captured, projected = np.zeros(width), np.zeros(width)
    active, previous = np.ones(width, dtype=bool), 0
    ranks = sorted({min(len(d), int(rank)) for rank in ranks})
    for rank in ranks:
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        if rank > previous:
            z = decomp.eigenvectors[:, previous:rank].T @ A[:, ids]
            captured[ids] += np.sum(d[previous:rank, None]*z*z, axis=0)
            projected[ids] += np.sum(z*z, axis=0)
        guard = 1e-12 + 1e-10*np.maximum(energy[ids], 1)
        remaining = energy[ids]-projected[ids]
        if np.any(remaining < -10*guard):
            raise ArithmeticError("Spectral energy discrepancy exceeds tolerance")
        remaining = np.maximum(remaining, 0)
        lower = captured[ids].copy()
        upper = lower.copy()
        if rank < len(d):
            lower += np.min(d[rank:])*remaining
            upper += np.max(d[rank:])*remaining
        lower = np.maximum(0, lower-guard)
        upper = np.maximum(0, upper+guard)
        lo[ids] = np.divide(U[ids]**2, upper, out=np.zeros(len(ids)), where=upper > 0)
        hi[ids] = np.divide(U[ids]**2, lower, out=np.full(len(ids), np.inf), where=lower > 0)
        resolved = lower > 1e-12
        for cutoff in criticals:
            target = np.asarray(cutoff)[ids] if np.ndim(cutoff) else cutoff
            resolved &= (hi[ids] < target) | (lo[ids] >= target)
        active[ids[resolved]] = False
        previous = rank
    exact_p = [None]*width
    statuses = ["BOUNDED"]*width
    pending = np.flatnonzero(active)
    if len(pending):
        started=time.perf_counter()
        for index, result in zip(pending, score_predictor_block(X[:, pending], cache, compute_spa_adjustment=False)):
            statuses[index] = result.status
            if result.status == "OK" and np.isfinite(result.score_z):
                lo[index] = hi[index] = result.score_z**2
                exact_p[index] = float(result.p_score)
        metric("exact_worker_seconds",time.perf_counter()-started)
        metric("exact_pairs",len(pending))
    return lo, hi, exact_p, statuses


def _score_chunk(job):
    response, predictors, external, screening = job
    s, n = _STATE, _STATE["info"]["n"]
    X, y = bits(predictors).T.astype(float), bits([response])[0].astype(float)
    overlap, counts = X.T @ y, X.sum(axis=0)
    cell = np.minimum.reduce([overlap, counts-overlap, y.sum()-overlap, n-counts-y.sum()+overlap])
    eligible = cell >= s["settings"]["min_cell_count"]
    if not screening and not eligible.all():
        # Final-only leaf work cannot produce a discovery from ineligible
        # pairs. Save the eligibility decision without fitting a useless null.
        result=[(int(u),response,0,-2,None,None,None,"INELIGIBLE_NOT_SCREENED")
                for u,ok in zip(predictors,eligible) if not ok]
        selected=[u for u,ok in zip(predictors,eligible) if ok]
        if selected:
            result.extend(_score_chunk((response,selected,external,False)))
        return result
    cache, status = get_null(response, external)
    if cache is None:
        return [(int(u), response, 0, 0 if ok else -2, None, None, None, status)
                for u, ok in zip(predictors, eligible)]
    gate, final = s["settings"]["gate_critical"], s["settings"]["final_critical"]
    # Ineligible pairs route the hierarchy, but can never become discoveries.
    cutoffs = [gate, np.where(eligible, final, gate)]
    if s["settings"]["exact"]:
        started=time.perf_counter()
        lo, hi, p, statuses = np.zeros(len(predictors)), np.full(len(predictors), np.inf), [], []
        for i, result in enumerate(score_predictor_block(X, cache, compute_spa_adjustment=False)):
            statuses.append(result.status)
            p.append(float(result.p_score) if result.status == "OK" else None)
            if result.status == "OK":
                lo[i] = hi[i] = result.score_z**2
        metric("exact_worker_seconds",time.perf_counter()-started)
        metric("exact_pairs",len(predictors))
    else:
        started=time.perf_counter()
        before=s["metrics"].get("exact_worker_seconds",0.)
        lo, hi, p, statuses = adaptive_statistics(X, cache, cutoffs)
        metric("bounds_worker_seconds",max(0.,time.perf_counter()-started-(s["metrics"].get("exact_worker_seconds",0.)-before)))
        metric("bound_pairs",len(predictors))
        metric("bound_decided_pairs",sum(status=='BOUNDED' for status in statuses))
    def decision(low, high, cutoff):
        return 1 if low >= cutoff else -1 if high < cutoff else 0
    return [(int(u), response, decision(l, h, gate), decision(l, h, final) if ok else -2,
             float(l), float(h) if np.isfinite(h) else None, value, status)
            for u, ok, l, h, value, status in zip(predictors, eligible, lo, hi, p, statuses)]


def score(job):
    response, predictors, external, screening = job
    width = _STATE["settings"]["batch_size"]
    result = []
    _STATE['active_null']=None
    try:
        for start in range(0,len(predictors),width):
            result.extend(_score_chunk((response,predictors[start:start+width],external,screening)))
    finally:
        _STATE.pop('active_null',None)
    return result


def score_profiled(job):
    started=time.perf_counter()
    before=dict(_STATE.get("metrics",{}))
    old_evictions=_STATE["nulls"].evictions if "nulls" in _STATE else 0
    rows=score(job)
    metrics={key:value-before.get(key,0) for key,value in _STATE.get("metrics",{}).items()}
    metrics.update(worker_job_seconds=time.perf_counter()-started,jobs=1,pairs=len(rows))
    if "nulls" in _STATE:
        metrics["null_cache_evictions"]=_STATE["nulls"].evictions-old_evictions
        metrics["cache_peak_bytes_worker"]=_STATE["nulls"].peak_bytes
    return dict(rows=rows,metrics=metrics)


def beta(job):
    u, v = job
    if time.monotonic()-_STATE["last_log"]>25:
        print(f"[KOVAR worker {os.getpid()}] beta for pattern pair {u},{v}",flush=True)
        _STATE["last_log"]=time.monotonic()
    X = bits([u, v]).T.astype(float)
    overlap = np.sum(X[:, 0]*X[:, 1])
    a, b = X.sum(axis=0)
    if min(overlap, a-overlap, b-overlap, len(X)-a-b+overlap) == 0:
        return u, v, None, "SEPARATION_OR_MONOMORPHIC"
    fit, cache = fit_logistic_mixed(X[:, 1], _STATE["K"], covariates=X[:, :1],
                                  kinship_eigensystem=_STATE["eigen"])
    if not fit.converged or fit.status != "OK":
        return u, v, None, fit.status
    variance = cache["M_inv"][1, 1]
    value = float(fit.beta[1])
    if fit.n_weights_clipped or not np.isfinite([value, variance]).all() or variance <= 0:
        return u, v, None, "UNSTABLE_EFFECT"
    return u, v, value, "OK"

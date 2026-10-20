#!/usr/bin/env python3
"""Profile bounded real TRAIN native queries; never certify or train a Corpus.

Restore actual account snapshots from a previously verified Corpus, transport
the same complete state/history deltas as NativeLabelPool and call its original
worker primitive. Independent chunking here is diagnostic only: production
account/label scheduling is unchanged. cProfile time is not a speed benchmark.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import cProfile
from collections import defaultdict
import functools
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import pstats
import subprocess
import time

import numpy as np


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(*args):
    from gym_trading_env.research.fg_s3_quality_label_pool import _initialize
    _initialize(*args[:-1])
    global _timing_only, _timings
    _timing_only=args[-1]; _timings={}
    if _timing_only:
        from gym_trading_env.utils import snapshot_json
        from gym_trading_env.utils.account_fingerprint import AccountFingerprint
        from gym_trading_env.utils.market_prefix import MarketPrefix
        from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
        from gym_trading_env.research import fg_s3_quality_teacher_v2 as teacher
        import pandas as pd
        targets=[('snapshot_copy',snapshot_json,'copy_account_snapshot'),
            ('account_hash',AccountFingerprint,'digest'),('market_prefix_hash',MarketPrefix,'digest'),
            ('env_import',FixedDirectionEntryEnv,'import_state'),
            ('env_export',FixedDirectionEntryEnv,'export_state'),
            ('native_trace',teacher,'trace_native_trade'),('risk_facts',teacher,'trade_facts'),
            ('financial_step',FixedDirectionEntryEnv,'step'),('market_equals',pd.DataFrame,'equals')]
        if hasattr(AccountFingerprint, 'capture'):
            targets.append(('account_hash',AccountFingerprint,'capture'))
        for name,owner,attribute in targets:
            original=getattr(owner,attribute)
            def measured(*a,_name=name,_original=original,**kw):
                started=time.thread_time()
                try:return _original(*a,**kw)
                finally:
                    elapsed=time.thread_time()-started
                    value=_timings.setdefault(_name,dict(calls=0,cpu_seconds=0.))
                    value['calls']+=1;value['cpu_seconds']+=elapsed
            setattr(owner,attribute,functools.wraps(original)(measured))


def profile_job(job):
    from gym_trading_env.research.fg_s3_quality_label_pool import _begin_stream, _query
    from benchmark_s3_corpus import canonical
    _begin_stream()
    _timings.clear()
    profiler = cProfile.Profile()
    started = time.perf_counter()
    labels = []
    cpu_started=time.thread_time()
    if not _timing_only:profiler.enable()
    try:
        for state, delta, mask, expected, count, _ in job['queries']:
            labels.append(_query(state, delta, mask, expected, count))
    finally:
        if not _timing_only:
            profiler.disable()
            profiler.dump_stats(job['profile_path'])
    query_cpu=time.thread_time()-cpu_started
    for label, query in zip(labels, job['queries'], strict=True):
        facts = label['L1']
        assert all(facts['layers'][layer]['source'] == facts['teacher_identity']
                   for layer in ('L1', 'L2'))
        if canonical(label) != canonical(query[-1]):
            raise ValueError('Profiled native query differs from original complete teacher label')
    from gym_trading_env.research import fg_s3_quality_label_pool as pool
    pool._teacher._check_source_stats()
    return dict(policy=job['policy'], queries=len(labels),
        native_trades=sum(x['L1']['immediate_outcome_verified'] for x in labels),
        strict_valid=sum(x['L2']['teacher_valid'] for x in labels),
        labels='EXACT_SEMANTIC_COMPARISON', seconds=time.perf_counter()-started,
        profile_path=None if _timing_only else job['profile_path'],query_cpu_seconds=query_cpu,
        function_cpu={k:dict(v) for k,v in _timings.items()})


def select_rows(rows, count):
    """Deterministic spread of real flat and held states across the interval."""
    selected = set()
    for candidates in ([i for i, r in enumerate(rows) if r['teacher']['L1']['immediate_outcome_verified']],
                       [i for i, r in enumerate(rows) if not r['teacher']['L1']['immediate_outcome_verified']]):
        if candidates:
            selected.update(candidates[int(i)] for i in np.linspace(
                0, len(candidates)-1, min(count//2, len(candidates)), dtype=int))
    # Include a real chunk continuation and its neighbors if present.
    for i, row in enumerate(rows):
        if row['status']=='CHUNK_END':
            selected.update(range(max(0, i-1), min(len(rows), i+2)))
    return sorted(selected)


def run(corpus, output, workers, queries_per_policy, timing_only=False):
    from gym_trading_env.research import fg_s3_e_quality_corpus_cli as cli
    from gym_trading_env.research.fg_replay import (
        read_manifest, iter_sidecar, execute_observed, check_files)
    from gym_trading_env.utils.account_fingerprint import account_fingerprint
    from e_data_v1.provider import ETrainAdmission

    corpus=Path(corpus).resolve(); output=Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest=read_manifest(corpus/'corpus_manifest.json')
    if manifest['fold']!='TRAIN' or manifest['execution_mode']!='data':
        raise ValueError('Profile requires an untruncated TRAIN DATA Corpus')
    cli._inventory(corpus, manifest)
    saved=read_manifest(corpus/'source_lock.json')
    profile=ETrainAdmission.from_train_pool(saved['source_pool']['path'],
        corpus/'e_source', corpus/'selected_train_pool.json', contracts=[saved['contract']])
    row=copy.deepcopy(saved['row'])
    row['env_config']=str(corpus/'inputs'/saved['contract']/'env.yaml')
    factory=cli._factory(row, profile)
    initial=json.loads((corpus/'initial_account.json').read_text())
    env=factory(); env.import_state(initial)
    source=copy.deepcopy(manifest['source_binding'])
    # Phase-2 restores the original per-query file-change guard.
    source['source_guard_interval']=1
    # Bind profiling to actual runtime bytes as well as the original data.
    code=cli._code_lock()
    source['files'].update(code['files'])
    strict=cli.QualityStrictTeacher(env, source=source, start_cursor=saved['start_cursor'])
    strict.compile_value_store(output/'strict_value_cache', pair_end_cursor=saved['end_cursor'])
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    prototype=NativeQualityTeacher(env, factory, fold='TRAIN', source=source)
    identity=prototype.identity_hash; prototype.close();env.close()
    jobs=[]; restores=0
    for name in manifest['behaviors']:
        rows=list(iter_sidecar(corpus/'policies'/name))
        chosen=set(select_rows(rows, queries_per_policy))
        snapshots=[];env=factory(); env.import_state(initial); previous=None
        for ordinal, saved_row in enumerate(rows):
            action=None if previous is None or previous['is_last'] else previous['action']
            _, actual=execute_observed(env, action, seed=0 if ordinal==0 else None,
                resume_state=initial if ordinal==0 else None)
            if any(actual[k]!=saved_row[k] for k in actual):
                raise ValueError('Real account reconstruction differs from original trace')
            if ordinal in chosen:
                snapshot=env.export_state()
                if account_fingerprint(env,snapshot)!=saved_row['account_snapshot_hash']:
                    raise ValueError('Real account snapshot identity differs')
                snapshots.append((snapshot,env._published_mask.copy(),saved_row['account_snapshot_hash'],saved_row['teacher']))
            if actual['status']=='CHUNK_END':
                state=env.export_state();env.reset(options={'resume':copy.deepcopy(state)})
                if env.export_state()!=state:raise ValueError('Real account cold restore differs')
                restores+=1
            previous=saved_row
            if ordinal==max(chosen):break
        env.close()
        # Several bounded profiles per account expose both cold and warm work.
        for part in range(4):
            subset=snapshots[part*len(snapshots)//4:(part+1)*len(snapshots)//4]
            if not subset:continue
            count=len(initial['equity_history']);queries=[]
            for state,mask,expected,label in subset:
                history=state.pop('equity_history')
                queries.append((state,history[count:],mask,expected,count,label))
                count=len(history)
            jobs.append(dict(policy=name,queries=queries,
                profile_path=str(output/f'{name}_{part}.pstats')))
    limits=cli._worker_limits(len(jobs),requested_workers=workers)
    process_count=min(len(jobs),limits['total_worker_processes'])
    results=[];began=time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=process_count,
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
                initargs=(row,profile,initial,source,strict,identity,timing_only)) as executor:
            futures=[executor.submit(profile_job, job) for job in jobs]
            results=[future.result() for future in as_completed(futures)]
    finally:strict._value_store.close()
    check_files(source['files'])
    entries=[]
    totals={}
    if not timing_only:
        stats=pstats.Stats(*[x['profile_path'] for x in results])
        for (file,line,function),(primitive,calls,self_time,cumulative,_) in stats.stats.items():
            entries.append(dict(file=file,line=line,function=function,calls=calls,
                primitive_calls=primitive,self_seconds=self_time,cumulative_seconds=cumulative))
    for job in results:
        for key,value in job['function_cpu'].items():
            item=totals.setdefault(key,dict(calls=0,cpu_seconds=0.))
            item['calls']+=value['calls'];item['cpu_seconds']+=value['cpu_seconds']
    report=dict(protocol='FG_S3_DATA_PERF_NATIVE_QUERY_PROFILE_V1',status='PASS_PROFILE_ONLY',
        created_utc=datetime.now(timezone.utc).isoformat(),corpus=str(corpus),
        corpus_identity=manifest['content_hash'],runtime=code['repositories'] if 'repositories' in code else code.get('repos'),
        code_sha256=code['files'],source_unchanged=True,workers=process_count,
        queries=sum(x['queries'] for x in results),native_trades=sum(x['native_trades'] for x in results),
        strict_valid=sum(x['strict_valid'] for x in results),actual_account_restores=restores,
        profiled_worker_wall_seconds=time.perf_counter()-began,
        profile_self_seconds=sum(x['self_seconds'] for x in entries),
        query_cpu_seconds=sum(x['query_cpu_seconds'] for x in results),
        timing_only=timing_only,function_cpu=totals,
        timing_notes='Function CPU times are inclusive/nested; do not sum them as disjoint shares',
        top_self=sorted(entries,key=lambda x:x['self_seconds'],reverse=True)[:40],
        top_cumulative=sorted(entries,key=lambda x:x['cumulative_seconds'],reverse=True)[:40],
        jobs=results,optimizer_updates=0,training_admitted=False,
        native_g3='NOT_RUN_PROFILE_IS_NOT_CERTIFICATION',
        profile_overhead=('Outer function thread CPU clocks only; no per-row tracing' if timing_only else
            'cProfile instrumentation; not an end-to-end speed measurement'))
    write(output/'QUERY_PROFILE.json',report)
    print(json.dumps({k:report[k] for k in ('status','queries','native_trades','strict_valid','workers','actual_account_restores','profile_self_seconds')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--workers',type=int,default=20)
    parser.add_argument('--queries-per-policy',type=int,default=16)
    parser.add_argument('--timing-only',action='store_true')
    args=parser.parse_args()
    if args.queries_per_policy<4:parser.error('At least four representative queries per policy required')
    run(args.corpus,args.out,args.workers,args.queries_per_policy,args.timing_only)

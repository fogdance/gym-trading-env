"""Bounded Corpus orchestration. No Agent, trainer, Prepared or Gate mutation."""
from __future__ import annotations
import argparse
import copy
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from dreamerv3.configuration import read_config_yaml
from dreamerv3 import fg_s3_e_quality_course_contract as course
from e_data_v1.core import MarketData
from e_data_v1.schema import Decision,EDataConfig
from e_data_v1.sources import read_fg_source
from e_data_v1.io import stable_identity,independent_output,write_bundle,write_collection_index,verify_collection
from e_data_v1.provider import ETrainAdmission
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.envs.entry_market_cache import EntryMarketCache
from .fg_replay import read_manifest,signed,write_json,check_files,digest,iter_sidecar,reconcile_trace,E_QUALITY_SCHEMA
from .fg_s3_e_observation import observation_contract
from .fg_s3_e_quality_corpus import behavior_specs,generate_behavior,coverage,SCHEMA,TEACHER_SCHEMA,l1_labeler
from .fg_s3_quality_teacher_v2 import NativeQualityTeacher
from .fg_s3_e_quality_strict import QualityStrictTeacher
from .fg_s3_quality_experiment_v2 import _derive_rows
from .fg_s3_e_quality_course_preflight import _local_path,preflight

PROJECT=Path(__file__).resolve().parents[3]
ARTIFACTS=PROJECT/'artifacts/fg_s3_v2/e_quality_wp02'
# Reviewed original TRAIN partition -> native Env receipt bytes. Integrity
# hashes alone cannot make an arbitrary self-signed source a trusted source.
TRUSTED_TRAIN_ROOTS={'ffc45db584106b7c348b7f3aee95096acbcb560c31a7d67fa340f30d2e10009a':
    'cf8fbf93953142b743f718effe2f377a80eba8a0cddf196a901d5344cf3d19ad'}


def _date_selection(start_day, end_day):
    if start_day is None and end_day is None:
        return None
    if start_day is None or end_day is None:
        raise ValueError('Explicit official start/end days must be supplied together')
    for value in (start_day,end_day):
        if not isinstance(value,str) or date.fromisoformat(value).isoformat()!=value:
            raise ValueError('Explicit official days require YYYY-MM-DD')
    if start_day>end_day:
        raise ValueError('Explicit official start day exceeds end day')
    return dict(start_day=start_day,end_day=end_day)


def _sample_bounds(td, start_day=None, end_day=None):
    selection=_date_selection(start_day,end_day)
    available=list(td.unique())
    if len(available)<2:raise ValueError('Two official days unavailable in TRAIN source')
    if selection is None:
        days=available[-2:];scope='last_two_official_days_of_main_qualification'
    else:
        if start_day not in available or end_day not in available:
            raise ValueError('Explicit official days must exist in the selected TRAIN contract')
        days=[day for day in available if start_day<=day<=end_day]
        scope='explicit_official_day_range_with_full_train_suffix'
    cursors=np.flatnonzero(td.isin(days))
    start,end=int(cursors[0]),int(cursors[-1])
    if start>=end:raise ValueError('Explicit official range needs at least two market cursors')
    return days,start,end,scope


def source_inputs(pool_path,env_receipt,contract,*,start_day=None,end_day=None):
    """Whitelist before ANY selected price-file access. Content hashes are
    integrity links; native verification below supplies the behavioral proof.
    """
    selection=_date_selection(start_day,end_day)
    pool_path=_local_path(pool_path);eroot=_local_path(env_receipt)
    pool=read_manifest(pool_path)
    if set(pool)-{'product','files','source_training_hash','content_hash'} or pool.get('product')!='FG':
        raise ValueError('Unknown source pool or forged approval claims')
    members=[r for r in pool['files'] if r.get('contract')==contract]
    if len(members)!=1:raise ValueError('Exactly one explicit TRAIN contract required')
    row=members[0]
    if (row.get('role')!='train' or row.get('eligible_for_training') is not True or
            row.get('complete_lifecycle') is not True or row.get('calendar_complete') is not True):
        raise ValueError('Selected source is not complete eligible TRAIN; TEST/DEV prices not opened')
    em=read_manifest(eroot/'manifest.json')
    if TRUSTED_TRAIN_ROOTS.get(pool.get('source_training_hash'))!=digest(eroot/'manifest.json'):
        raise ValueError('Unreviewed TRAIN trust root; a self-signed approval is not authentication')
    ipath=eroot/'inputs/inputs.json'
    if em.get('stage')!='E' or em.get('status')!='PASS' or em.get('clock_protocol')!='S3_MARKET_MINUTE':
        raise ValueError('Source requires the original native market-minute Env receipt')
    check_files({str(ipath):em['files']['inputs/inputs.json']})
    inputs=read_manifest(ipath)
    if pool.get('source_training_hash')!=inputs['original_content_hash']:
        raise ValueError('TRAIN source pool escaped the original frozen source identity')
    partition=_local_path(em['config']['training_manifest'])
    check_files({str(partition):em['frozen_inputs'][str(partition)]})
    partition_data=read_manifest(partition)
    if partition_data['content_hash']!=inputs['original_content_hash']:
        raise ValueError('Original TRAIN partition identity differs')
    original=[r for r in inputs['views']['C1']['rows'] if r.get('contract')==contract]
    if len(original)!=1 or original[0]!=row:
        raise ValueError('TRAIN member differs from the native source whitelist')
    for k in ('path','env_config','contract_config'):_local_path(row[k])
    pool_record=stable_identity(pool_path)
    source=read_fg_source(row,pool_record)
    bars=source.bars
    times=pd.DatetimeIndex(pd.to_datetime(bars.Date))
    times=times.tz_localize(source.schedule['timezone']) if times.tz is None else times.tz_convert(source.schedule['timezone'])
    td=pd.to_datetime(bars.trading_date.astype(str),format='mixed').dt.strftime('%Y-%m-%d')
    if (times.has_duplicates or not times.is_monotonic_increasing or
            not td.is_monotonic_increasing or str(times[0])!=row['first_timestamp'] or str(times[-1])!=row['last_timestamp']):
        raise ValueError('TRAIN time bounds/calendar identity differs')
    actual_days=set(td.str.replace('-','').astype(int))
    roles=partition_data['roles']
    if not actual_days<=set(roles['train']) or actual_days&(set(roles['validation'])|set(roles['test'])):
        raise ValueError('Actual selected market rows escaped TRAIN partition')
    config=read_config_yaml(row['env_config']);terms=read_config_yaml(row['contract_config'])
    boundary=pd.Timestamp(config['entry']['main_exit_at'])
    boundary=boundary.tz_localize('Asia/Shanghai') if boundary.tz is None else boundary
    if terms.get('expiry_basis')!='main_qualification' or boundary!=times[-1]:
        raise ValueError('Trusted TRAIN source must end at the authentic main-qualification exit')
    days,start,end,scope=_sample_bounds(td,start_day,end_day)
    # The explicit dated calendar, not a row/step limit, certifies the boundary.
    market=MarketData(source.bars,source.schedule,EDataConfig())
    if any(e['type']=='MINUTE_REJECTED' for e in market.events):raise ValueError('Invalid trusted TRAIN minute data')
    expected=market.calendar[(market.calendar>=times[start]) & (market.calendar<=times[end])]
    observed=times[start:end+1]
    if not observed.equals(expected):raise ValueError('Selected official days have missing or extra calendar minutes')
    lock=dict(protocol='FG_S3_E_QUALITY_SOURCE_LOCK_V2',fold='TRAIN',contract=contract,
        source_pool=dict(path=str(pool_path),sha256=pool_record['sha256']),
        env_receipt=dict(path=str(eroot),sha256=digest(eroot/'manifest.json')),
        inputs_sha256=digest(ipath),partition_manifest=dict(path=str(partition),sha256=digest(partition)),row=row,provenance=source.provenance,
        official_days=days,start_cursor=start,end_cursor=end,
        sample_scope=scope,
        suffix_end_cursor=len(bars)-1,suffix_boundary_time=str(boundary),
        first_time=str(times[start]),last_time=str(times[end]),
        mathematical_schema=market.config.schema(__import__('e_data_v1.reference',fromlist=['load_legacy']).load_legacy()[1]),
        no_test_prices_opened=True,training_admitted=False)
    # The selected Replay interval and the complete TRAIN teacher suffix are
    # distinct. Never fabricate a financial terminal at an observation cutoff.
    if selection is not None:lock['selection']=selection
    return signed(lock),source,market


def _collection(lock,source,market,out):
    folder=out/'e_source';folder.mkdir()
    times=market.frame.index[lock['start_cursor']:lock['end_cursor']+1]
    bundle=market.build([Decision(source.contract,t) for t in times])
    bundle.provenance.update(source.provenance)
    target=write_bundle(bundle,folder/source.contract,inputs=[lock['source_pool']['path']])
    report=dict(source_pool=lock['source_pool'],schema_sha256=bundle.schema['schema_sha256'],bundles=[dict(path=str(target))])
    write_json(folder/'diagnostics.json',report);write_collection_index(folder,report)
    audit=verify_collection(folder)
    if audit['status']!='PASS':raise ValueError('New E source collection failed native verification')
    write_json(out/'selected_train_pool.json',signed(dict(product='FG',files=[lock['row']])))
    profile=ETrainAdmission.from_train_pool(lock['source_pool']['path'],folder,out/'selected_train_pool.json',contracts=[source.contract])
    return profile,audit


def _factory(row,profile):
    cache=EntryMarketCache(1)
    def factory():return FixedDirectionEntryEnv(config_path=row['env_config'],e_profile=profile,market_cache=cache)
    return factory


def _prefix(factory,start):
    env=factory();env.reset(seed=0)
    for i in range(start):
        if env.truncated:
            if env.status!='CHUNK_END':raise ValueError('Warmup reached unresolved native boundary')
            env.reset()
        before=env.current_step;env.step(0)
        if env.current_step<=before or env.terminated:raise ValueError('Native warmup cannot reach selected sample')
    if env.truncated:env.reset()
    state=env.export_state()
    if env.current_step!=start or env.entries or env._calculate_equity()!=300000:
        raise ValueError('Initial sample account is not the real native WAIT prefix')
    return env,state


def _code_lock():
    roots=[PROJECT/'src/gym_trading_env',PROJECT/'e_data_v1',PROJECT.parent/'dreamerv3/dreamerv3',PROJECT.parent/'dreamerv3/embodied']
    files={str(p):digest(p) for root in roots for p in root.rglob('*.py')
           if 'tests' not in p.relative_to(root).parts}
    for name in ('fg_s3_e_quality_corpus.py','fg_s3_v2_experiment.py'):
        p=PROJECT/'tools'/name;files[str(p)]=digest(p)
    repos={}
    for name,path in [('gym',PROJECT),('dreamer',PROJECT.parent/'dreamerv3')]:
        repos[name]=dict(sha=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
            dirty=bool(subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True)))
    return dict(files=files,repositories=repos,execution_source_hash=fingerprint(files))


def _formal_course(lock,profile,out,seed):
    value=json.loads((PROJECT/'docs/fixtures/fg_s3_e_quality_course_v2.example.json').read_text())
    bounds=dict(start=pd.Timestamp(lock['first_time']).isoformat(),end=(pd.Timestamp(lock['last_time'])+pd.Timedelta(minutes=1)).isoformat())
    value.update(spec_mode='formal',initialization=dict(kind='direct_init',seed=seed))
    data=value['data_identity'];row=lock['row']
    data.update(contracts=[row['contract']],time_bounds=bounds,
        source_manifest=lock['source_pool'],e_profile_schema_sha256=profile.source_receipt['schema_sha256'],
        market_sources=[dict(contract=row['contract'],source_sha256=row['sha256'],env_config_sha256=row['env_sha256'],metadata_sha256=row['contract_sha256'])],
        teacher_manifest=dict(path=str(out/'teacher_manifest.json'),sha256=digest(out/'teacher_manifest.json')))
    specs=behavior_specs(seed)+[dict(id='P4',kind='quality_oracle_schedule',seed_components=[seed,0,5])]
    value['curriculum']['behaviors']=[]
    for spec in specs:
        parameters=({'probability':spec['probability']} if spec['kind']=='probabilistic' else
                    {'wait_market_minutes':15} if spec['kind']=='single_delay' else
                    {'source':dict(path=str(out/'p4_plan.json'),sha256=digest(out/'p4_plan.json'))} if spec['kind']=='quality_oracle_schedule' else {})
        value['curriculum']['behaviors'].append(dict(behavior_id=spec['id'],kind=spec['kind'],seed=spec['seed_components'][0],
            parameters=parameters,boundary=dict(contracts=[row['contract']],time_bounds=bounds,initial_equity='300000')))
    validated=course.validate_course_spec(value)
    write_json(out/'course_v2.json',value)
    return fingerprint(course.course_identity_payload(validated))


def _publish_streams(out,names):
    """Publish byte-identical native chunks separately from posterior traces.

    No alternate Replay format: existing iter_replay_rows reads these chunks
    with the same E V2 schema and the original physical UUID/offset stepids.
    """
    for name in names:
        original=out/'policies'/name
        trace=read_manifest(original/'manifest.json')
        student=out/'student_replay'/name;student.mkdir(parents=True)
        for chunk in trace['chunks']:
            target=student/chunk['path'];target.parent.mkdir(exist_ok=True)
            shutil.copyfile(original/chunk['path'],target)
        manifest={k:trace[k] for k in ('schema','reward_version','obs_version','records','chunks','array_hash','e_observation_identity','data_identity')}
        manifest['files']={c['path']:digest(student/c['path']) for c in trace['chunks']}
        write_json(student/'manifest.json',signed(manifest))
        teacher=out/'teacher_sidecar'/name;teacher.mkdir(parents=True)
        shutil.copyfile(original/'steps.jsonl.gz',teacher/'steps.jsonl.gz')


def _verify_publications(path,names):
    from .fg_replay import trace_manifest,iter_replay_rows
    for name in names:
        source=path/'policies'/name;student=path/'student_replay'/name
        original=trace_manifest(source,schema=E_QUALITY_SCHEMA)
        published=trace_manifest(student,schema=E_QUALITY_SCHEMA)
        if set(published)!={'schema','reward_version','obs_version','records','chunks','array_hash','e_observation_identity','data_identity','files','content_hash'}:
            raise ValueError('Student publication includes foreign or posterior metadata')
        if any(published[k]!=original[k] for k in published if k not in ('files','content_hash')):
            raise ValueError('Student native Replay differs from its account trace')
        for _,step in iter_replay_rows(student,published):
            if any(k.startswith(('teacher','quality','oracle','outcome')) for k in step):
                raise ValueError('Posterior field entered student Replay')
        if digest(source/'steps.jsonl.gz')!=digest(path/'teacher_sidecar'/name/'steps.jsonl.gz'):
            raise ValueError('Posterior sidecar publication differs')


def _behavior_job(job):
    began=time.monotonic()
    factory=_factory(job['row'],job['profile'])
    spec=job['spec']
    from .fg_s3_quality_label_pool import NativeLabelPool
    labels=(lambda identity:NativeLabelPool(job['row'],job['profile'],job['initial'],
        job['source'],job['strict'],identity,job['label_workers'])) if job.get('label_workers') else None
    trace,receipt=generate_behavior(factory,spec,job['out'],PROJECT.parent/'dreamerv3',
        source=job['source'],initial_state=job['initial'],end_cursor=job['end'],strict=job['strict'],
        labeler_factory=labels,verify_labels=False)
    return dict(behavior=spec['id'],trace_hash=trace['replay_trace_hash'],reconciliation=receipt,
        seconds=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def _worker_limits(jobs, *, available=None, peak_rss=None, cpus=None, process_memory=None, requested_workers=None):
    """Size the CPU pool from the initialized native process, not a fixed cap.

    PSS accounts for shared library pages instead of charging their complete
    RSS once per process. Keep 25% over the initialized process footprint,
    at least 256 MiB per worker, plus 512 MiB coordinator growth headroom.
    """
    if available is None:
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                           if line.startswith('MemAvailable:')))*1024
    if peak_rss is None:peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if cpus is None:cpus=len(os.sched_getaffinity(0))
    if requested_workers is not None and (type(requested_workers) is not int or not 1<=requested_workers<=cpus):
        raise ValueError('Requested workers must be positive and fit the available CPU affinity')
    if process_memory is None:
        process_memory=int(next(line.split()[1] for line in Path('/proc/self/smaps_rollup').read_text().splitlines()
                                if line.startswith('Pss:')))*1024
    budget=max(256*1024**2,(5*process_memory+3)//4)
    reserve=512*1024**2
    capacity=(available-reserve)//budget
    if capacity<1:raise ValueError('Insufficient available memory for one native Corpus worker')
    requested=cpus if requested_workers is None else requested_workers
    slots=min(requested,capacity)
    workers=min(jobs,max(1,slots//2))
    label_slots=slots-workers if slots>workers else 0
    label_workers=[label_slots//workers+int(jobs==workers and i<label_slots%workers) for i in range(workers)]
    slots=workers+sum(label_workers)
    return dict(workers=workers,label_workers=label_workers,total_worker_processes=slots,available_memory_bytes=available,
        observed_parent_peak_rss_bytes=peak_rss,worker_budget_bytes=budget,
        observed_process_pss_bytes=process_memory,
        coordinator_reserve_bytes=reserve,available_logical_cpus=cpus,native_threads_per_worker=1,
        requested_worker_processes=requested,worker_limit_reason='memory' if capacity<requested else 'cpu_budget')


def _verification_job(job):
    """One independent complete account replay, with original source checks."""
    began=time.monotonic()
    factory=_factory(job['row'],job['profile'])
    env=factory();env.import_state(job['initial'])
    facts=NativeQualityTeacher(env,factory,fold='TRAIN',source=job['source'])
    labels=l1_labeler(facts,job['strict'])
    if job.get('label_workers'):
        from .fg_s3_quality_label_pool import NativeLabelPool
        labels=NativeLabelPool(job['row'],job['profile'],job['initial'],job['source'],
            job['strict'],facts.identity_hash,job['label_workers'])
        facts.close()
    try:
        receipt=reconcile_trace(job['path'],PROJECT.parent/'dreamerv3',env_factory=factory,
            schema=E_QUALITY_SCHEMA,label_verifier=labels)
        if job.get('label_workers'):
            receipt['label_transport']=dict(labels.report_statistics())
    finally:
        if job.get('label_workers'):labels.close()
        facts.close()
    result=dict(behavior=job['name'],receipt=receipt,seconds=time.monotonic()-began,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    print(json.dumps(result),flush=True)
    return result


def _execution_scope(lock, market, mode, max_market_steps):
    if mode not in ('dev','data'):
        raise ValueError('Corpus mode must be dev or data')
    if mode=='data':
        if max_market_steps is not None:
            raise ValueError('A step-limited diagnostic requires --mode dev; DATA uses explicit official days')
        return lock
    if type(max_market_steps) is not int or max_market_steps<1:
        raise ValueError('DEV requires an explicit positive --max-market-steps')
    value={k:v for k,v in lock.items() if k!='content_hash'}
    advances=np.flatnonzero(market.market_clock_counter.valid)
    advances=advances[(advances>lock['start_cursor']) & (advances<=lock['end_cursor'])]
    end=int(advances[max_market_steps-1]) if len(advances)>=max_market_steps else lock['end_cursor']
    value.update(execution_mode='dev',requested_max_market_steps=max_market_steps,
        source_window_end_cursor=lock['end_cursor'],end_cursor=end,
        last_time=str(market.frame.index[end]),sample_scope='bounded_dev_observations_with_full_train_suffix')
    value['official_days']=list(market.frame.iloc[lock['start_cursor']:end+1].trading_date.unique())
    return signed(value)


def generate_small(pool,env_receipt,contract,out,*,seed=202,start_day=None,end_day=None,workers=None,
                   mode='data',max_market_steps=None):
    began=time.monotonic()
    out=_local_path(out)
    independent_output(out,inputs=[pool])
    if workers is not None and (type(workers) is not int or not 1<=workers<=len(os.sched_getaffinity(0))):
        raise ValueError('Requested workers must be positive and fit the available CPU affinity')
    lock,source,market=source_inputs(pool,env_receipt,contract,start_day=start_day,end_day=end_day)
    lock=_execution_scope(lock,market,mode,max_market_steps)
    if mode=='dev' and workers is None:workers=1
    independent_output(out,inputs=[pool,env_receipt,lock['row']['path'],lock['row']['env_config'],
        lock['row']['contract_config'],lock['partition_manifest']['path'],PROJECT,PROJECT.parent/'dreamerv3'])
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'source_lock.json',lock)
    profile,audit=_collection(lock,source,market,out)
    rows,frozen,_,members=_derive_rows(dict(env_receipt=str(env_receipt),contracts=[contract],contract=contract,
        reward=asdict(__import__('gym_trading_env.rewards.single3_reward',fromlist=['Single3RewardConfig']).Single3RewardConfig('C1',.5,.4,3.,'FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1'))),out)
    row=rows[0];factory=_factory(row,profile)
    env,initial=_prefix(factory,lock['start_cursor'])
    write_json(out/'initial_account.json',initial)
    code=_code_lock()
    anchors={lock['source_pool']['path']:lock['source_pool']['sha256'],
        str(Path(env_receipt)/'manifest.json'):lock['env_receipt']['sha256'],
        str(Path(env_receipt)/'inputs/inputs.json'):lock['inputs_sha256'],
        lock['partition_manifest']['path']:lock['partition_manifest']['sha256']}
    source_binding=dict(fold='TRAIN',source_lock_hash=lock['content_hash'],files={**frozen,**code['files'],**anchors},
        state_cache_entries=0,calendar_complete=True,source_guard_interval=1024)
    strict=QualityStrictTeacher(env,source=source_binding,start_cursor=lock['start_cursor'])
    strict.compile_value_store(out/'strict_value_cache',pair_end_cursor=lock['end_cursor'])
    write_json(out/'p4_plan.json',strict.plan())
    write_json(out/'teacher_manifest.json',signed(dict(protocol=TEACHER_SCHEMA,fold='TRAIN',
        value_witness_encoding=strict.identity['value_witness_encoding'],
        reward_schema=strict.identity['reward_schema'],score_policy=strict.identity['score_policy'],
        strict_identity=strict.identity_hash,source_lock_hash=lock['content_hash'],
        e_profile_identity=profile.identity,objective=strict.identity['objective'],
        gamma_rational=[332,333],native_validation=strict.plan()['native_validation'])))
    traces=[]
    specs=behavior_specs(seed)+[dict(id='P4',kind='quality_oracle_schedule',seed_components=[seed,0,5],teacher_identity=strict.identity_hash,hindsight_behavior=True)]
    limits=_worker_limits(len(specs),requested_workers=workers);workers=limits['workers']
    print(json.dumps(dict(phase='generate',resource_budget=limits)),flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        futures=[executor.submit(_behavior_job,dict(row=row,profile=profile,spec=spec,
            out=out/'policies'/spec['id'],source=source_binding,initial=initial,
            end=lock['end_cursor'],strict=strict,label_workers=limits['label_workers'][i%workers])) for i,spec in enumerate(specs)]
        jobs=[]
        for future in as_completed(futures):
            result=future.result();jobs.append(result);print(json.dumps(result),flush=True)
    traces=[out/'policies'/spec['id'] for spec in specs]
    _publish_streams(out,[s['id'] for s in specs])
    _verify_publications(out,[s['id'] for s in specs])
    report=coverage(traces);write_json(out/'coverage.json',report)
    from .fg_s3_e_quality_corpus import write_coverage_csv
    write_coverage_csv(out/'coverage.csv',report)
    course_hash=_formal_course(lock,profile,out,seed)
    performance=dict(seconds=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_bytes=Path(lock['row']['path']).stat().st_size,replay_bytes=sum(p.stat().st_size for p in (out/'student_replay').rglob('*.npz')),
        candidate_cost=strict.query.candidate_cost,workers=workers,resource_budget=limits,behavior_jobs=jobs,
        source_rows=len(source.bars),sample_rows=lock['end_cursor']-lock['start_cursor']+1,
        scope=lock['sample_scope'])
    write_json(out/'performance.json',performance)
    check_files(source_binding['files'])
    model_contract=dict(protocol='FG_S3_E_MODEL_CONTRACT_V2',observation=observation_contract(),
        encoder_code_sha256=digest(PROJECT.parent/'dreamerv3/dreamerv3/fg_s3_e_encoder.py'),
        minute_layers=3,context_units=128,macd_units=16,fusion_units='DREAMER_PRESET',
        decoder_keys=['market_seq_e_shared','position_state'],horizon=333,clock='S3_MARKET_MINUTE',
        initialization='FRESH_PARAMETERS_ONLY',legacy_migration=False)
    write_json(out/'model_contract.json',signed(model_contract))
    manifest=signed(dict(protocol=SCHEMA,fold='TRAIN',execution_mode=mode,
        status='DEV_ONLY_NOT_CERTIFIED' if mode=='dev' else 'CORPUS_PRODUCED',
        certification='NOT_CERTIFIED' if mode=='dev' else 'INDEPENDENT_NATIVE_G3_REQUIRED',
        verification_state='COLLECTED_UNVERIFIED',
        generation_label_verification='DEFERRED_TO_INDEPENDENT_NATIVE_G3',
        training_admitted=False,source_lock_hash=lock['content_hash'],course_identity_hash=course_hash,
        observation=observation_contract(),model_contract_hash=fingerprint(model_contract),
        teacher_identity=strict.identity_hash,e_profile_identity=profile.identity,code=code,
        behaviors=[s['id'] for s in specs],source_binding=source_binding,
        evidence=dict(native_replay='ALL_P0_P4_ROWS',strict_suffix=strict.plan()['native_validation']),
        files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}))
    write_json(out/'corpus_manifest.json',manifest)
    return manifest


def _inventory(path,m):
    for name in [*m['files'],*m['behaviors']]:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe Corpus manifest path')
        _local_path(path/name)
    if m['behaviors']!=['P0','P1','P2_low','P2_mid','P3','P4']:
        raise ValueError('Corpus does not contain the complete five-policy set')
    check_files({str(path/name):sha for name,sha in m['files'].items()})


def _expected_binding_files(path,lock,code):
    row=lock['row'];source=lock['provenance']
    original=read_config_yaml(row['env_config']);target=copy.deepcopy(original)
    target['entry'].pop('state_path',None)
    target['entry']['contract_config']=row['contract_config'];target['trading']['data_path']=row['path']
    folder=path/'inputs'/row['contract'];envpath=folder/'env.yaml'
    if not original['entry'].get('strategy_config'):raise ValueError('Unreviewed Quality source derivation')
    masterpath=folder/'single3.yaml';target['entry']['strategy_config']=str(masterpath)
    master=read_config_yaml(source['master']['path']);master['reward']=course.REWARD
    if target!=read_config_yaml(envpath) or master!=read_config_yaml(masterpath):
        raise ValueError('Origin/Target changed financial rules beyond the Quality Reward snapshot')
    files={**code,str(envpath):digest(envpath),str(masterpath):digest(masterpath),
        lock['source_pool']['path']:lock['source_pool']['sha256'],
        str(Path(lock['env_receipt']['path'])/'manifest.json'):lock['env_receipt']['sha256'],
        str(Path(lock['env_receipt']['path'])/'inputs/inputs.json'):lock['inputs_sha256'],
        lock['partition_manifest']['path']:lock['partition_manifest']['sha256']}
    for role in ('market','env','contract','master'):
        files[source[role]['path']]=source[role]['sha256']
    return files


def verify_corpus(path,*,workers=None):
    """Re-establish source, market math and every native account row read-only."""
    path=_local_path(path);m=read_manifest(path/'corpus_manifest.json')
    if m.get('protocol')!=SCHEMA or m.get('training_admitted') is not False or m.get('fold')!='TRAIN':
        raise ValueError('Unknown Corpus or forged training admission')
    if m.get('execution_mode','data')!='data':
        raise ValueError('DEV_CORPUS_NOT_CERTIFIED: generate a new DATA version for native admission')
    _inventory(path,m)
    if _code_lock()['files']!=m['code']['files']:
        raise ValueError('Corpus runtime code inventory differs')
    check_files(m['code']['files'])
    saved=read_manifest(path/'source_lock.json')
    current,_,_=source_inputs(saved['source_pool']['path'],saved['env_receipt']['path'],saved['contract'],
        **saved.get('selection',{}))
    if current!=saved or saved['content_hash']!=m['source_lock_hash']:raise ValueError('Corpus source authority differs')
    expected=_expected_binding_files(path,saved,m['code']['files'])
    if m['source_binding']['files']!=expected:
        raise ValueError('Corpus source file allowlist differs')
    check_files(expected)
    if read_manifest(path/'model_contract.json')['content_hash']!=m['model_contract_hash']:
        raise ValueError('Model contract hash differs')
    profile=ETrainAdmission.from_train_pool(saved['source_pool']['path'],path/'e_source',path/'selected_train_pool.json',contracts=[saved['contract']])
    row=copy.deepcopy(saved['row']);row['env_config']=str(path/'inputs'/saved['contract']/'env.yaml')
    factory=_factory(row,profile)
    env,initial=_prefix(factory,saved['start_cursor'])
    if initial!=json.loads((path/'initial_account.json').read_text()):raise ValueError('Initial native WAIT account proof differs')
    strict=QualityStrictTeacher(env,source=m['source_binding'],start_cursor=saved['start_cursor'])
    if strict.identity_hash!=m['teacher_identity'] or strict.plan()!=read_manifest(path/'p4_plan.json'):
        raise ValueError('Strict suffix proof differs')
    limits=_worker_limits(len(m['behaviors']),requested_workers=workers)
    print(json.dumps(dict(phase='verify',resource_budget=limits)),flush=True)
    # Rebuild a separate cache from this verifier's native Oracle. Never load
    # the producer's value cache as the expected answer for G3.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='fg_g3_exact_') as temporary:
        strict.compile_value_store(Path(temporary)/'values',pair_end_cursor=saved['end_cursor'])
        try:
            with ProcessPoolExecutor(max_workers=limits['workers'],mp_context=multiprocessing.get_context('spawn')) as executor:
                futures=[executor.submit(_verification_job,dict(name=name,path=path/'policies'/name,
                    row=row,profile=profile,initial=initial,source=m['source_binding'],strict=strict,
                    label_workers=limits['label_workers'][i%limits['workers']])) for i,name in enumerate(m['behaviors'])]
                jobs=[future.result() for future in as_completed(futures)]
        finally:strict._value_store.close()
    by_name={job['behavior']:job['receipt'] for job in jobs}
    receipts=[by_name[name] for name in m['behaviors']]
    if any(r['teacher_labels_verified']!=r['records'] or not r['records'] for r in receipts):
        raise ValueError('Native G3 must independently recompute every teacher label')
    check_files(expected)
    _verify_publications(path,m['behaviors'])
    report=coverage([path/'policies'/name for name in m['behaviors']])
    from .fg_s3_e_quality_corpus import coverage_csv
    if (report!=read_manifest(path/'coverage.json') or
            coverage_csv(report)!=(path/'coverage.csv').read_bytes().decode()):
        raise ValueError('Coverage cannot be reproduced')
    spec=course.validate_course_spec(json.loads((path/'course_v2.json').read_text()))
    if fingerprint(course.course_identity_payload(spec))!=m['course_identity_hash']:raise ValueError('Course hash differs')
    return signed(dict(status='PASS',protocol=SCHEMA,training_admitted=False,
        verification_state='NATIVE_G3_VERIFIED',
        corpus_identity=m['content_hash'],native_replay=receipts,G3='GO',test_prices_opened=0,
        resource_budget=limits,behavior_jobs=jobs))


def main(argv=None):
    parser=argparse.ArgumentParser(description='WP02 E Quality CPU Corpus, never training')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['validate','generate-small','generate']:
        p=sub.add_parser(name);p.add_argument('--source-pool');p.add_argument('--env-receipt');p.add_argument('--contract',default='CZCE.FG201')
        p.add_argument('--start-day',help='First official TRAIN trading day, inclusive (YYYY-MM-DD)')
        p.add_argument('--end-day',help='Last Replay trading day, inclusive; labels may use the locked complete TRAIN suffix')
        p.add_argument('--debug',action='store_true',help='Preserve the full diagnostic traceback on stderr')
        if name=='validate':p.add_argument('--course')
        else:
            p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=202)
            p.add_argument('--workers',type=int,help='Total CPU worker budget across account and teacher processes; default: all affinity CPUs, memory bounded')
            p.add_argument('--mode',choices=['dev','data'],default='data')
            p.add_argument('--max-market-steps',type=int,help='Required only for DEV; cap actual market advances, not teacher suffix or account history')
    for name in ['verify','coverage']:
        p=sub.add_parser(name);p.add_argument('--corpus',required=True)
        p.add_argument('--debug',action='store_true')
        if name=='verify':p.add_argument('--workers',type=int)
    args=parser.parse_args(argv)
    try:
        if args.command=='validate' and args.course:
            result=preflight(args.course);print(json.dumps(result));return result['exit_code']
        if args.command in ['validate','generate-small','generate']:
            if not args.source_pool or not args.env_receipt:raise ValueError('Explicit real TRAIN source pool and Env receipt required')
            if args.command=='generate' and args.start_day is None and args.end_day is None:
                raise ValueError('Production generate requires explicit official start/end days')
            if args.command=='validate':
                lock,_,_=source_inputs(args.source_pool,args.env_receipt,args.contract,start_day=args.start_day,end_day=args.end_day)
                result=dict(status='SOURCE_VALIDATED_ONLY',source=lock,training_admitted=False)
            else:result=generate_small(args.source_pool,args.env_receipt,args.contract,args.out,seed=args.seed,
                start_day=args.start_day,end_day=args.end_day,workers=args.workers,
                mode=args.mode,max_market_steps=args.max_market_steps)
        elif args.command=='verify':result=verify_corpus(args.corpus,workers=args.workers)
        else:
            path=_local_path(args.corpus);m=read_manifest(path/'corpus_manifest.json')
            _inventory(path,m)
            result=coverage([path/'policies'/name for name in m['behaviors']])
        print(json.dumps(result,ensure_ascii=False));return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        if args.debug:traceback.print_exc(file=sys.stderr)
        print(json.dumps(dict(status='NO_GO',error=str(exc),diagnostics=getattr(exc,'events',[]),training_admitted=False)),flush=True);return 2

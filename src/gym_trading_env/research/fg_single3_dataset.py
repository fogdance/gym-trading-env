"""Fresh Single3 native datasets over the existing Driver/Replay/reconcile.

Reuse is restricted to already reconciled fresh Single3 R/O accounts. Legacy
six-lot traces are refused. Reward views rerun the real Env with frozen causal
source actions. Hard links retain immutable trace bytes and native UUIDs.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import time

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import (SINGLE3_SCHEMA,check_files,digest,read_manifest,signed,trace_manifest,write_json)
from .fg_single3 import execution_provenance,make_output,required_tests,source_witness
from .fg_single3_replay import require_e,specs,trace_job
from .fg_single3_oracle import require_r


def require_o(path):
    path=Path(path).resolve();m=read_manifest(path/'manifest.json')
    if (m.get('stage')!='O' or m.get('status')!='PASS' or m.get('oracle_optimality_status')!='exact_proven'
            or not m.get('full_training_fold')):
        raise ValueError('Dataset requires full TRAIN exact O Gate')
    check_files(m['frozen_inputs']);check_files({str(path/p):h for p,h in m['files'].items()})
    return m


def require_input_audit(config,inputs):
    """Admission proof for the same frozen TRAIN/Env observation bytes."""
    for path in sorted((Path(config['output_root'])/'input_audit').glob('*/manifest.json'),reverse=True):
        m=read_manifest(path)
        if m.get('stage')!='INPUT_AUDIT' or m.get('status')!='PASS' or not m.get('full_training_fold'):
            continue
        if any(m['frozen_inputs'].get(p)!=h for p,h in inputs['frozen_files'].items()):continue
        if {r['contract'] for r in m['contracts']}!={r['contract'] for r in inputs['views']['C2']['rows']}:continue
        check_files(m['frozen_inputs']);check_files({str(path.parent/p):h for p,h in m['files'].items()})
        check_files({p:h for p,h in m['provenance']['code_files'].items()
            if '/gym_trading_env/envs/' in p or '/gym_trading_env/utils/' in p
            or p.endswith('/single3_reward.py') or p.endswith('/fg_reward_candidates.py')})
        return path
    raise ValueError('Fresh Single3 dataset requires full TRAIN real-prefix observation causality audit')


def install_trace(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    trace=trace_manifest(source,schema=SINGLE3_SCHEMA)
    receipt=read_manifest(source/'reconciliation.json')
    if receipt['status']!='PASS' or receipt['trace_manifest_sha256']!=digest(source/'manifest.json'):
        raise ValueError('Only reconciled fresh Single3 traces may be reused')
    destination.mkdir(parents=True,exist_ok=True)
    for p in source.rglob('*'):
        if not p.is_file():continue
        target=destination/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=digest(p):raise ValueError('Immutable trace copy changed')
        else:os.link(p,target)
    return dict(directory=str(destination),manifest=trace,receipt=receipt,resumed=True,
                immutable_source=str(source))


def _attempt(root,candidate,policy,contract,resume):
    base=root/'traces'/candidate/policy/contract
    if not base.exists():return base
    if not resume:raise FileExistsError('Existing trace retained; use --resume or a new run')
    if (base/'reconciliation.json').exists():return base
    for n in range(2,10000):
        path=base.with_name(base.name+f'_attempt_{n:02d}')
        if not path.exists() or (path/'reconciliation.json').exists():return path
    raise ValueError('Trace attempt budget exceeded')


def _trace_with_resources(job):
    started=time.monotonic()
    result=trace_job(job)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if peak>job['worker_rss_budget_bytes']:
        raise RuntimeError('Per-worker RSS budget exceeded; no dataset completion')
    result['resources']=dict(seconds=time.monotonic()-started,peak_rss_bytes=peak,
                             worker_rss_budget_bytes=job['worker_rss_budget_bytes'])
    return result


def _collect(jobs,workers):
    result=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(_trace_with_resources,job) for job in jobs]):
            item=future.result();result.append(item)
            print(f"Dataset [{len(result)}/{len(jobs)}]: {item['manifest']['reward_candidate']} {item['manifest']['policy']['id']} {item['manifest']['contract']}",flush=True)
    return result


def record_trace(root,item):
    directory=Path(item['directory']);m=item['manifest']
    return dict(candidate=m['reward_candidate'],policy_id=m['policy']['id'],contract=m['contract'],
        directory=str(directory.relative_to(root)),manifest_sha256=digest(directory/'manifest.json'),
        receipt_sha256=digest(directory/'reconciliation.json'),replay_trace_hash=m['replay_trace_hash'],
        trace_identity=m['identity'],records=m['records'],sequence_starts=m['sequence_starts'],
        reward_params_hash=m['reward_params_hash'],observation_identity=m['observation_identity'],
        source_stage='R/O' if item.get('immutable_source') else 'D',resources=item.get('resources'),
        immutable_source=item.get('immutable_source'))


def pool_members(records):
    pools={'C2_P0':[i for i,r in enumerate(records) if r['policy_id']=='P0']}
    for candidate in ('C0','C1','C2','C3'):
        for policy in ('P1','P2a','P2b','P3','P4'):
            pools[candidate+'_'+policy]=[i for i,r in enumerate(records) if r['candidate']==candidate and r['policy_id']==policy]
        pools[candidate+'_natural']=[i for i,r in enumerate(records) if r['candidate']==candidate and r['policy_id'] in ('P1','P2a','P2b','P3')]
        pools[candidate+'_natural_oracle']=[i for i,r in enumerate(records) if r['candidate']==candidate and r['policy_id'] in ('P1','P2a','P2b','P3','P4')]
    return pools


def validate_dataset(root,*,manifest_path=None):
    root=Path(root).resolve();m=read_manifest(root/'manifest.json' if manifest_path is None else manifest_path)
    inputs=read_manifest(root/'inputs.json')
    if (m.get('schema')!=SINGLE3_SCHEMA or m.get('stage')!='D_GENERATED'
            or m.get('status')!='PASS' or m['identity']!=inputs['identity']
            or m['input_manifest_sha256']!=digest(root/'inputs.json')):
        raise ValueError('Single3 dataset identity mismatch')
    check_files(inputs['frozen_inputs']);check_files(inputs['source_snapshots'])
    expected={(c,p,k) for c in ('C0','C1','C2','C3') for p in ('P1','P2a','P2b','P3','P4')
              for k in m['identity']['selected_contracts']}
    expected|={('C2','P0',k) for k in m['identity']['selected_contracts']}
    actual=[(r['candidate'],r['policy_id'],r['contract']) for r in m['traces']]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Missing fresh policy/candidate/contract accounts')
    if m['pool_members']!=pool_members(m['traces']) or set(m['views'])!=set(m['pool_members']):
        raise ValueError('Replay pool membership changed')
    seen=set();expected_views={p:{} for p in m['views']}
    for i,r in enumerate(m['traces']):
        directory=root/r['directory']
        check_files({str(directory/'manifest.json'):r['manifest_sha256'],str(directory/'reconciliation.json'):r['receipt_sha256']})
        t=trace_manifest(directory,schema=SINGLE3_SCHEMA);receipt=read_manifest(directory/'reconciliation.json')
        if (receipt['status']!='PASS' or receipt['trace_manifest_sha256']!=r['manifest_sha256']
                or t['identity']!=r['trace_identity'] or t['reward_candidate']!=r['candidate']
                or t['policy']['id']!=r['policy_id'] or t['contract']!=r['contract']
                or t['replay_trace_hash']!=r['replay_trace_hash'] or t['records']!=r['records']
                or t['sequence_starts']!=r['sequence_starts'] or t['reward_params_hash']!=r['reward_params_hash']
                or t['observation_identity']!=r['observation_identity'] or t.get('protocol')!='fg_single3_v1'
                or t['open_lots'] not in (0,3)):
            raise ValueError('Dataset has an unreconciled or foreign trace')
        for chunk in t['chunks']:
            if chunk['uuid'] in seen:raise ValueError('Independent accounts share native chunks')
            seen.add(chunk['uuid'])
            for pool,members in m['pool_members'].items():
                if i in members:
                    expected_views[pool][Path(chunk['path']).name]=t['files'][chunk['path']]
    for pool,files in expected_views.items():
        view=root/m['views'][pool]
        if {p.name for p in view.glob('*.npz')}!=set(files):raise ValueError('Missing/extra pool chunks')
        check_files({str(view/p):h for p,h in files.items()})
        candidates={m['traces'][i]['candidate'] for i in m['pool_members'][pool]}
        if len(candidates)!=1:raise ValueError('Multiple Rewards in one training pool')
        if not pool.endswith('_P0') and any(m['traces'][i]['policy_id']=='P0' for i in m['pool_members'][pool]):
            raise ValueError('All-WAIT diluted a training pool')
    return m


def generate_dataset(config,output,env_receipt,reward_receipt,oracle_receipt,*,workers=12,contracts=None,resume=False):
    if not 1<=workers<=config['max_workers']:raise ValueError('Worker budget exceeded')
    tests=required_tests('D');inputs,e=require_e(env_receipt);reward=require_r(reward_receipt);oracle=require_o(oracle_receipt)
    input_audit=require_input_audit(config,inputs)
    selected=set(contracts or [r['contract'] for r in inputs['views']['C2']['rows']])
    all_contracts={r['contract'] for r in inputs['views']['C2']['rows']}
    if not selected or selected-all_contracts:raise ValueError('Dataset escaped TRAIN whitelist')
    root=Path(output).resolve()
    if not resume:root=make_output(root,config)
    code=execution_provenance(config,'D')
    for name in ('fg_replay_coverage.py','fg_single3_oracle.py'):
        path=Path(__file__).with_name(name)
        code['code_files'][str(path)]=digest(path)
    identity=dict(schema=SINGLE3_SCHEMA,protocol='fg_single3_v1',original_training_hash=inputs['original_content_hash'],
        selected_contracts=sorted(selected),code_hash=fingerprint(code['code_files']),fold='TRAIN',conditional_direction_only=True)
    if resume:
        started=read_manifest(root/'inputs.json')
        if started['identity']!=identity:raise ValueError('Dataset resume code/input identity changed')
        snapshots=started['source_snapshots']
    else:snapshots=source_witness(root,code)
    frozen=dict(inputs['frozen_files'])
    for path in (Path(env_receipt)/'manifest.json',Path(reward_receipt)/'manifest.json',Path(oracle_receipt)/'manifest.json',input_audit):
        frozen[str(path.resolve())]=digest(path)
    live={**frozen,**code['code_files']}
    write_json(root/'inputs.json',signed(dict(identity=identity,config=config,provenance=code,
        source_snapshots=snapshots,frozen_inputs=frozen,training_views=inputs['views'],tests=tests,
        run_parameters=dict(workers=workers,total_worker_rss_budget_bytes=config['rss_budget_gib']*2**30))))
    # Reuse the already real, reconciled single-position accounts, never old6.
    reused=[];known={}
    for item in reward['collection']:
        t=item['manifest'];candidate=t['reward_candidate'];p=t['policy']['id'];contract=t['contract']
        if contract not in selected:continue
        target=_attempt(root,candidate,p,contract,resume)
        new=install_trace(item['directory'],target);reused.append(new);known[(candidate,p,contract)]=new
    for contract_result in oracle['contracts']:
        if contract_result['contract'] not in selected:continue
        for item in contract_result['collection']:
            t=item['manifest'];candidate=t['reward_candidate'];contract=t['contract']
            target=_attempt(root,candidate,'P4',contract,resume)
            new=install_trace(item['directory'],target);reused.append(new);known[(candidate,'P4',contract)]=new
    jobs=[]
    worker_budget=int(config['rss_budget_gib']*2**30/workers)
    for index,row in enumerate(inputs['views']['C2']['rows']):
        if row['contract'] not in selected:continue
        for spec in specs(config,index):
            if ('C2',spec['id'],row['contract']) in known:continue
            jobs.append(dict(directory=str(_attempt(root,'C2',spec['id'],row['contract'],resume)),row=row,spec=spec,
                identity=identity,dreamer_root=config['dreamer_root'],frozen_files=live,resume=resume,
                worker_rss_budget_bytes=worker_budget))
    collected=_collect(jobs,workers)
    for item in collected:
        t=item['manifest'];known[('C2',t['policy']['id'],t['contract'])]=item
    derived=[]
    for index,base_row in enumerate(inputs['views']['C2']['rows']):
        contract=base_row['contract']
        if contract not in selected:continue
        for spec in specs(config,index):
            if spec['id']=='P0':continue
            source=known[('C2',spec['id'],contract)]
            for candidate in ('C0','C1','C3'):
                row=next(r for r in inputs['views'][candidate]['rows'] if r['contract']==contract)
                derived.append(dict(directory=str(_attempt(root,candidate,spec['id'],contract,resume)),
                    row=row,spec=dict(spec,source_directory=source['directory'],source_trace_hash=source['manifest']['replay_trace_hash']),
                    identity=identity,dreamer_root=config['dreamer_root'],frozen_files=live,resume=resume,
                    worker_rss_budget_bytes=worker_budget))
    generated=_collect(derived,workers)
    records=[record_trace(root,item) for item in [*reused,*collected,*generated]]
    records.sort(key=lambda r:(r['candidate'],r['policy_id'],r['contract']))
    members=pool_members(records);views={}
    for pool,indices in members.items():
        view=root/'replay'/pool;view.mkdir(parents=True,exist_ok=True)
        for i in indices:
            directory=root/records[i]['directory'];t=trace_manifest(directory,schema=SINGLE3_SCHEMA)
            for chunk in t['chunks']:
                source=directory/chunk['path'];target=view/source.name
                if target.exists():
                    if digest(target)!=t['files'][chunk['path']]:raise ValueError('Existing pool changed')
                else:os.link(source,target)
        views[pool]=str(view.relative_to(root))
    payload=signed(dict(schema=SINGLE3_SCHEMA,stage='D_GENERATED',status='PASS',D_Gate='REQUIRES_COVERAGE',
        identity=identity,full_training_fold=selected==all_contracts,traces=records,views=views,pool_members=members,
        input_manifest_sha256=digest(root/'inputs.json'),training_reward_selected=None,gradient_updates=0))
    write_json(root/'manifest.pending.json',payload);check_files(live)
    validate_dataset(root,manifest_path=root/'manifest.pending.json')
    (root/'manifest.pending.json').replace(root/'manifest.json')
    return payload

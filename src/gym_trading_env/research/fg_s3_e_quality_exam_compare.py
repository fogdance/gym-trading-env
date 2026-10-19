"""Causal native baselines and same-source account comparisons."""
from __future__ import annotations

import copy
from pathlib import Path

from .fg_single3_replay import Single3BehaviorPolicy
from .fg_s3_e_quality_exam_contract import identity,validate_plan
from .fg_replay import read_manifest,write_json,signed,E_QUALITY_SCHEMA,reconcile_trace
from .fg_single3_evaluation import embodied_runtime


class ExamBaseline(Single3BehaviorPolicy):
    """Only the existing causal policies; no Oracle schedule/source views."""
    def __init__(self,name,*,seed,contract,random_seed=None):
        specs={'ALWAYS_WAIT':dict(kind='always_wait'),'ALWAYS_ADD':dict(kind='always_add'),
               'RANDOM_P10':dict(kind='probabilistic',probability=.1),
               'DELAY_15':dict(kind='single_delay',wait_minutes=15)}
        if name not in specs:raise ValueError('EXAM_CAUSAL_BASELINE_REQUIRED_NO_ORACLE')
        if name=='RANDOM_P10' and (type(random_seed) is not int or random_seed<0):
            raise ValueError('EXAM_PREREGISTERED_RANDOM_SEED_REQUIRED')
        super().__init__(dict(id=name,**specs[name],seed_components=[seed,int(contract[7:]),random_seed or 0]))
        self.last_diagnostics={}

    def __call__(self,carry,obs):
        carry,action,out=super().__call__(carry,obs)
        self.last_diagnostics=dict(p_add=.1 if self.spec['kind']=='probabilistic' and obs['action_mask'][0,1]
            else float(action['action'][0]),baseline=self.spec['id'],teacher_used=False)
        return carry,action,out

    def save(self):
        return copy.deepcopy(dict(protocol='FG_S3_E_QUALITY_EXAM_BASELINE_STATE_V1',
            spec_identity=identity(self.spec),rng=self.rng.bit_generator.state,cursor=self.cursor,
            waited=self.waited,waiting=self.waiting,diagnostics=self.last_diagnostics))

    def load(self,state):
        if state.get('protocol')!='FG_S3_E_QUALITY_EXAM_BASELINE_STATE_V1' or state.get('spec_identity')!=identity(self.spec):
            raise ValueError('EXAM_BASELINE_STATE_IDENTITY_MISMATCH')
        self.rng.bit_generator.state=copy.deepcopy(state['rng'])
        for name in ('cursor','waited','waiting'):setattr(self,name,copy.deepcopy(state[name]))
        self.last_diagnostics=copy.deepcopy(state['diagnostics'])


def policy_jobs(plan):
    plan=validate_plan(plan);jobs=[]
    for contract in plan['contracts']:
        for name in plan['baselines']:
            if name=='RANDOM_P10':
                jobs.extend(dict(contract=contract,policy=name,random_seed=s,label=f'{name}_seed{s}') for s in plan['random_seeds'])
            else:jobs.append(dict(contract=contract,policy=name,random_seed=None,label=name))
    return jobs


def run_case(plan,source,directory,job,*,resume=False,stop_at=None):
    """Thin evaluation lifecycle; all execution is the original ExamSession."""
    from .fg_s3_e_quality_exam_session import ExamSession
    from .fg_s3_e_quality_exam_actor import ExamFrozenActor
    from .fg_s3_e_quality_exam_metrics import analyze_trace
    p=validate_plan(plan);root=Path(directory);name=job['policy'];label=job['label']
    if name in p['actors']:
        a=p['actors'][name]
        loader=ExamFrozenActor.from_f3_checkpoint if a['kind']=='F3' else ExamFrozenActor.from_f4_checkpoint
        policy=loader(a['run'],a['checkpoint'],root.parent/(root.name+'_model'),engineering=p['exam_mode']=='SYNTHETIC_DEV',
            evaluation_seed=p['seed'])
        model_receipt=policy.receipt
    else:
        policy=ExamBaseline(name,seed=p['seed'],contract=job['contract'],random_seed=job['random_seed'])
        model_receipt=signed(dict(protocol='FG_S3_E_QUALITY_EXAM_BASELINE_V1',policy=policy.spec,
            teacher_used=False,training_updates=0))
    session=ExamSession(p,source,policy,root,policy_id=label,existing=resume)
    try:
        if resume:
            session.restore()
            if not session.finished and (root/'trace').exists():
                recovery=root/'recovery_outputs';recovery.mkdir(exist_ok=True)
                index=0;target=recovery/'trace'
                while target.exists():index+=1;target=recovery/f'trace_{index}'
                (root/'trace').rename(target)
        else:
            write_json(root/'MODEL_READONLY_RECEIPT.json',model_receipt);session.commit()
        if session.finished:
            receipt=read_manifest(root/'SESSION_RECEIPT.json')
            if not (root/'metrics/METRICS.json').exists():
                if (root/'metrics').exists():
                    recovery=root/'recovery_outputs';recovery.mkdir(exist_ok=True)
                    index=0;target=recovery/'metrics'
                    while target.exists():index+=1;target=recovery/f'metrics_{index}'
                    (root/'metrics').rename(target)
                analyze_trace(root/'trace',source,root/'metrics')
            return receipt
        while session.raw.current_step<source.end_cursor and not session.raw.terminated:
            if stop_at is not None and session.valid_minutes>=stop_at:break
            target=session.valid_minutes+p['chunk_steps']
            if stop_at is not None:target=min(target,stop_at)
            session.advance(target);session.commit()
            if session.raw.status in ('DATA_END','DATA_GAP'):break
        if stop_at is not None and session.raw.current_step<source.end_cursor and not session.raw.terminated:
            return dict(status='EXAM_PAUSED',valid_minutes=session.valid_minutes,training_updates=0)
        receipt=session.finish();session.commit()
        analyze_trace(root/'trace',source,root/'metrics')
        return receipt
    finally:session.close()


def compare_sessions(cases,output):
    """Recompute each account from native trace; never trust imported scores.

    cases contain (session directory, freshly admitted native ExamSource).
    Account ownership differs, while source, observed interval and plan match.
    """
    from .fg_s3_e_quality_foundation import PROJECT
    from .fg_s3_e_quality_exam_metrics import analyze_trace,_table
    out=Path(output)
    if out.exists():raise ValueError('EXAM_COMPARISON_OUTPUT_MUST_BE_NEW')
    prepared=[];by_contract={}
    for directory,source in cases:
        root=Path(directory);receipt=read_manifest(root/'SESSION_RECEIPT.json')
        plan=validate_plan(__import__('json').loads((root/'PLAN.json').read_text()))
        source.require_mode(plan);source.verify()
        if receipt['source_hash']!=identity(source.receipt):raise ValueError('EXAM_COMPARISON_SOURCE_IDENTITY_MISMATCH')
        original=read_manifest(root/'SOURCE.json')
        if original!=source.receipt:raise ValueError('EXAM_COMPARISON_NATIVE_SOURCE_CHANGED')
        key=receipt['contract']
        common=(receipt['plan_hash'],receipt['source_hash'],receipt['valid_minutes'])
        if key in by_contract and by_contract[key]!=common:raise ValueError('EXAM_COMPARISON_UNEQUAL_SOURCE_OR_INTERVAL')
        by_contract[key]=common;prepared.append((root,source,receipt))
    if not prepared:raise ValueError('EXAM_COMPARISON_REQUIRES_CASES')
    out.mkdir(parents=True);summaries=[];blocks=[];evidence=[]
    seen=set()
    for root,source,receipt in prepared:
        key=(receipt['contract'],receipt['policy_id'])
        if key in seen:raise ValueError('EXAM_COMPARISON_DUPLICATE_ACCOUNT_ROLE')
        seen.add(key)
    # Independent accounts are CPU jobs. Spawn avoids forking an initialized
    # JAX runtime; only immutable native source descriptions cross processes.
    import cloudpickle
    import multiprocessing as mp
    import concurrent.futures
    import os
    payloads=[cloudpickle.dumps((root,source,receipt,out/f'case_{i:04d}')) for i,(root,source,receipt) in enumerate(prepared)]
    workers=max(1,min(len(payloads),os.cpu_count() or 1,28))
    with concurrent.futures.ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn')) as pool:
        verified=list(pool.map(_verify_case,payloads))
    for (root,source,receipt),(data,replay) in zip(prepared,verified):
        summaries.append(dict(policy=receipt['policy_id'],exam_mode=receipt['exam_mode'],**data['summary']))
        blocks.extend(dict(policy=receipt['policy_id'],**day) for day in data['daily'])
        evidence.append(dict(directory=str(root),receipt=receipt,replay=replay))
    paired=[]
    for block in blocks:
        baseline=next((x for x in blocks if x['contract']==block['contract'] and x['official_day']==block['official_day']
            and x['policy']=='ALWAYS_WAIT'),None)
        if baseline is None:raise ValueError('EXAM_FAIR_COMPARISON_REQUIRES_WAIT_REFERENCE')
        paired.append(dict(contract=block['contract'],official_day=block['official_day'],policy=block['policy'],
            reference='ALWAYS_WAIT',net_pnl_delta=block['net_pnl']-baseline['net_pnl'],
            reward_delta=block['reward']-baseline['reward'],drawdown_delta=block['max_drawdown']-baseline['max_drawdown'],
            entries=block['entries'],tp_le_5=block['tp_le_5'],mae_15_mean=block['mae_15_mean']))
    unique={(b['contract'],b['official_day']) for b in blocks}
    report=signed(dict(protocol='FG_S3_E_QUALITY_EXAM_COMPARISON_V1',status='SAME_SOURCE_NATIVE_ACCOUNTS_RECONCILED',
        summaries=summaries,blocks=blocks,paired_deltas=paired,independent_contract_days=len(unique),
        independent_contracts=len(by_contract),account_count=len(prepared),evidence=evidence,
        sample_unit='unique_contract_official_day; seeds_and_policies_are_repeated_accounts',
        strategy_quality='NOT_DECIDED_BY_COMPARISON',test_prices_accessed=False))
    _table(out/'COMPARISONS.csv',paired);_table(out/'BLOCKS.csv',blocks)
    write_json(out/'COMPARISONS.json',report)
    return report


def _verify_case(payload):
    import cloudpickle
    from .fg_s3_e_quality_foundation import PROJECT
    from .fg_s3_e_quality_exam_metrics import analyze_trace
    root,source,receipt,out=cloudpickle.loads(payload)
    source.verify()
    replay=reconcile_trace(root/'trace',PROJECT.parent/'dreamerv3',env_factory=source.factory,
        schema=E_QUALITY_SCHEMA,runtime_factory=embodied_runtime)
    if replay['replay_trace_hash']!=receipt['trace_hash']:raise ValueError('EXAM_COMPARISON_TRACE_CHANGED')
    return analyze_trace(root/'trace',source,out),replay

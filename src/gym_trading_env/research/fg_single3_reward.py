"""CPU Reward comparison on immutable fresh Single3 original-Driver traces."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.rewards.single3_reward import Single3Reward, Single3RewardConfig
from .fg_replay import (SINGLE3_SCHEMA, check_files, execute_observed, iter_replay_rows,
                       iter_sidecar, read_manifest, signed, trace_manifest, write_json)
from .fg_single3 import (execution_provenance, load_env, make_output,required_tests,source_witness)
from .fg_single3_replay import require_e, specs, trace_job
from .fg_history_study import digest


def score_trace(job):
    """Pure event relabel plus separate original-Env feedback verification."""
    directory=Path(job['directory'])
    source=Path(job['source_directory'])
    trace=trace_manifest(source,schema=SINGLE3_SCHEMA)
    directory.mkdir(parents=True,exist_ok=False)
    scorer=Single3Reward(Single3RewardConfig.for_candidate(job['candidate']))
    raw=load_env(job['row'],job['frozen_files'])
    pure_hash=hashlib.sha256()
    counts=Counter()
    reward_sum=native_sum=discounted=0.
    weight=1.
    previous=None
    closed={}
    rows=0
    with gzip.open(directory/'reward_rows.jsonl.gz','wt') as stream:
        for row,(ordinal,step) in zip(iter_sidecar(source),iter_replay_rows(source,trace)):
            incoming=None if previous is None or previous['is_last'] else previous['action']
            obs,expected=execute_observed(raw,incoming,seed=0 if ordinal==0 else None)
            for key in expected:
                if key not in ('official_reward','reward_events') and expected[key]!=row[key]:
                    raise ValueError(f'Reward candidate changed financial trace: {ordinal}/{key}')
            for key,value in obs.items():
                np.testing.assert_array_equal(step[key],value,err_msg=f'Reward view {ordinal}/{key}')
            components=scorer.step(row['bar'],raw.terms.multiplier) if row['advanced'] else dict(
                total=0.,tp=0.,early=0.,sl=0.,other=0.,events=[],early_first5=0,early_next10=0)
            value=components['total']
            if value!=expected['official_reward'] or components['events']!=expected['reward_events']:
                raise ValueError('Pure Reward view differs from actual original-Env feedback')
            if job['candidate']=='C2' and np.float32(value)!=step['reward']:
                raise ValueError('Source native Reward was shifted or changed')
            native=float(np.float32(value))
            reward_sum+=value;native_sum+=native
            if row['advanced']:
                discounted+=weight*native
                weight*=332/333
            counts['early_first5']+=components['early_first5']
            counts['early_next10']+=components['early_next10']
            counts['negative_feedback_rows']+=int(native<0)
            counts['positive_feedback_rows']+=int(native>0)
            for event in row['exits']:
                counts[event['reason']]+=1
                closed[event['entry_id']]=event
            target=dict(ordinal=ordinal,source_stepid=row['stepid'],candidate=job['candidate'],
                        original_reward=row['official_reward'],reward=value,native_reward=native,
                        components=components)
            pure_hash.update(json.dumps(target,sort_keys=True,separators=(',',':')).encode())
            stream.write(json.dumps(target,separators=(',',':'))+'\n')
            if expected['status']=='CHUNK_END':
                state=raw.export_state();raw.import_state(state)
                if raw.export_state()!=state:
                    raise RuntimeError('Candidate Env reward/financial restore differs')
                scorer=Single3Reward.restore(scorer.snapshot(),scorer.config)
            previous=row;rows+=1
    if rows!=trace['records']:
        raise ValueError('Reward view is missing source rows')
    trades=scorer.scorer.trade_rows()
    for trade in trades:
        event=closed.get(trade['entry_id'])
        trade['net_trade_pnl']=event['net_trade_pnl'] if event else None
        trade['SL_reward']=-scorer.config.L if event and event['reason']=='STOP_LOSS' else 0.
        trade['actual_exit']=event
    write_json(directory/'trades.json',trades)
    receipt=dict(status='PASS',candidate=job['candidate'],reward_params_hash=scorer.config.hash,
        contract=trace['contract'],policy_id=trace['policy']['id'],source_trace_hash=trace['replay_trace_hash'],
        source_manifest_sha256=digest(source/'manifest.json'),records=rows,
        undiscounted_raw=reward_sum,undiscounted_float32=native_sum,
        discounted_H333_EnvSteps=discounted,counts=dict(counts),
        account_net_pnl=float(raw._calculate_equity()-300000),
        feedback_hash=pure_hash.hexdigest(),
        files={str(p.relative_to(directory)):digest(p) for p in directory.rglob('*') if p.is_file()})
    raw.close()
    check_files(job['frozen_files'])
    write_json(directory/'manifest.json',signed(receipt))
    return dict(directory=str(directory),receipt=receipt)


def reward_study(config,output,env_receipt,*,workers=12,contracts=('CZCE.FG809','CZCE.FG909')):
    if not 1<=workers<=config['max_workers']:
        raise ValueError('Worker count exceeds configured budget')
    tests=required_tests('R')
    inputs,evidence=require_e(env_receipt)
    selected=set(contracts)
    rows=inputs['views']['C2']['rows']
    if not selected or selected-{r['contract'] for r in rows}:
        raise ValueError('Reward study contracts are outside TRAIN')
    output=make_output(output,config)
    code=execution_provenance(config,'R')
    snapshots=source_witness(output,code)
    frozen={**inputs['frozen_files'],**code['code_files'],
            str(Path(env_receipt).resolve()/'manifest.json'):digest(Path(env_receipt)/'manifest.json')}
    identity=dict(schema=SINGLE3_SCHEMA,stage='R',original_training_hash=inputs['original_content_hash'],
                  code_hash=fingerprint(code['code_files']),reward_candidate='C2')
    write_json(output/'inputs.json',signed(dict(stage='R',config=config,provenance=code,
        frozen_files=frozen,training_views=inputs['views'],identity=identity)))
    jobs=[]
    for index,row in enumerate(rows):
        if row['contract'] not in selected:continue
        for spec in specs(config,index):
            jobs.append(dict(directory=str(output/'traces'/spec['id']/row['contract']),
                row=row,spec=spec,identity=identity,frozen_files=frozen,dreamer_root=config['dreamer_root']))
    collected=[]
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        for future in as_completed([pool.submit(trace_job,job) for job in jobs]):
            item=future.result();collected.append(item)
            print(f"Reward source [{len(collected)}/{len(jobs)}]: {item['manifest']['policy']['id']} {item['manifest']['contract']}",flush=True)
    score_jobs=[]
    lookup={name:{r['contract']:r for r in view['rows']} for name,view in inputs['views'].items()}
    for item in collected:
        trace=item['manifest']
        for candidate in config['candidates']:
            score_jobs.append(dict(directory=str(output/'scores'/candidate/trace['policy']['id']/trace['contract']),
                source_directory=item['directory'],candidate=candidate,
                row=lookup[candidate][trace['contract']],frozen_files=frozen))
    scored=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        for future in as_completed([pool.submit(score_trace,job) for job in score_jobs]):
            item=future.result();scored.append(item)
            r=item['receipt']
            print(f"Reward views [{len(scored)}/{len(score_jobs)}]: {r['candidate']} {r['policy_id']} {r['contract']} verified",flush=True)
    collected.sort(key=lambda r:(r['manifest']['policy']['id'],r['manifest']['contract']))
    scored.sort(key=lambda r:(r['receipt']['candidate'],r['receipt']['policy_id'],r['receipt']['contract']))
    receipt=dict(schema='FG_SINGLE3_CPU_V1',stage='R',status='PASS',
        selected_contracts=sorted(selected),full_training_fold=len(selected)==len(rows),
        training_content_hash=inputs['original_content_hash'],provenance=code,
        frozen_inputs={p:h for p,h in frozen.items() if p not in code['code_files']},
        source_snapshots=snapshots,collection=collected,scores=scored,selected_training_reward=None,tests=tests,
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    write_json(output/'manifest.pending.json',signed(receipt))
    check_files(frozen)
    check_files({str(output/p):h for p,h in receipt['files'].items()})
    (output/'manifest.pending.json').replace(output/'manifest.json')
    return receipt

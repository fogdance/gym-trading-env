"""Bounded, stratified real P4 prefix replay and full-account ADD/WAIT forks."""
from fractions import Fraction
import gzip
import json
from pathlib import Path
import time

import numpy as np

from .fg_replay import digest,iter_sidecar,read_manifest,signed,write_json
from .fg_s3a_labels import load_labels
from .fg_single3 import load_env
from .fg_single3_oracle import GAMMA
from .fg_s3b_advantage import audit_values,verify_counterfactual


def verify_real_prefix(values,output,*,contracts=('CZCE.FG001','CZCE.FG201','CZCE.FG809'),
                       max_cursor=4096,random_count=8,max_local_steps=2048,seed=202):
    vm = audit_values(values)
    labels = Path(vm['identity']['s3a_labels'])
    lm = load_labels(labels,verify_sources=False)
    inputs = read_manifest(Path(lm['source']['dataset'])/'inputs.json')
    views = {r['contract']:r for r in inputs['training_views']['C1']['rows']}
    recs = {r['contract']:r for r in lm['traces']}
    rng = np.random.default_rng(seed)
    reports = []
    for contract in contracts:
        if contract not in vm['identity']['contracts']: raise ValueError('Requested non-TRAIN contract')
        began = time.monotonic()
        rec = recs[contract]
        trace = Path(lm['source']['dataset'])/rec['directory']
        tm = read_manifest(trace/'manifest.json')
        cp = Path(tm['policy']['schedule_path']).parent/'candidates.jsonl.gz'
        with gzip.open(cp,'rt') as f: candidates = {r['cursor']:r for r in map(json.loads,f)}
        a = np.load(Path(values)/'contracts'/contract/'index.npy',allow_pickle=False)
        v = np.load(Path(values)/'contracts'/contract/'values.npy',allow_pickle=False)
        eligible = []
        groups = {}
        skipped_long = 0
        for i,row in enumerate(a):
            t = int(row['cursor'])
            if t > max_cursor: break
            c = candidates[t]
            duration = c.get('observed_until_cursor',t+1)-t
            if duration > max_local_steps:
                skipped_long += 1
                continue
            eligible.append(i)
            event = c.get('exit') or {}
            key = (int(row['year']),int(row['direction']),event.get('reason','CENSORED' if c.get('filled') else 'REJECTED'),
                   bool(c.get('first_adverse_cursor') is not None),bool(event.get('ambiguous_tp_sl')),
                   bool(event.get('sl_pending_retry')),bool(row['is_first']))
            groups.setdefault(key,i)
        if not eligible: raise ValueError('No actual prefix fork samples')
        selected = set(groups.values())
        selected.update(int(i) for i in rng.choice(eligible,min(random_count,len(eligible)),replace=False))
        selected.update(i for i in eligible if int(a[i]['cursor']) % 1024 in (0,1023))
        ordinals = {int(a[i]['ordinal']):i for i in selected}
        env = load_env(views[contract],lm['source']['frozen_inputs'])
        env.reset(seed=0)
        forks = []
        last = max(ordinals)
        advanced = 0
        feedback = 0.
        for row in iter_sidecar(trace):
            if row['ordinal'] > last: break
            if row['reset'] and row['ordinal'] != 0:
                if not env.truncated or env.status != 'CHUNK_END':
                    raise ValueError('Original P4 administrative continuation differs')
                env.reset()
            if env.current_step != row['cursor'] or float(env.net_lots) != row['lots_after']:
                raise ValueError('Actual teacher prefix account/cursor differs from Sidecar')
            if bool(env._get_obs()['action_mask'][1]) != row['legal_add']:
                raise ValueError('Actual teacher prefix mask differs from Sidecar')
            expected_feedback = feedback if row['advanced'] else 0.
            if float(np.float32(row['official_reward'])) != float(np.float32(expected_feedback)):
                raise ValueError('Actual teacher prefix incoming feedback differs from Sidecar')
            if row['ordinal'] in ordinals:
                i = ordinals[row['ordinal']]
                state = env.export_state()
                from decimal import Decimal
                if Decimal(state['balances']['user_cash']) != Decimal(v[i]['cash_start'].decode()):
                    raise ValueError('Value proof used a different actual cash state')
                fork = verify_counterfactual(env,state,candidates[row['cursor']])
                local = sum((GAMMA**(s-row['cursor']-1)*Fraction.from_float(r)
                             for s,r in fork['local_rewards']),Fraction(0))
                if v[i]['has_intervals']:
                    # Q contains nonnegative suffix, so actual local is a
                    # feasible lower bound. Never pretend local alone is Q*.
                    if Fraction.from_float(float(v[i]['q_add'][1])) < local:
                        raise ValueError('Certified ADD upper bound excludes its actual local return')
                forks.append(dict(fork,ordinal=row['ordinal'],stepid=row['stepid'],
                    year=int(a[i]['year']),direction=int(a[i]['direction']),teacher_action=int(a[i]['label']),
                    value_certificate=str(v[i]['certificate']),value_proof=str(v[i]['proof']),
                    actual_cash=state['balances']['user_cash'],state_sha256=__import__('hashlib').sha256(
                        json.dumps(state,sort_keys=True).encode()).hexdigest()))
            if row['decision_executed']:
                _,reward,_,_,_ = env.step(row['action'])
                feedback = reward
                advanced += 1
                # The next Sidecar row carries this feedback; every local fork
                # independently verifies float32 C1 against frozen candidates.
                if not np.isfinite(reward): raise ValueError('Nonfinite original prefix reward')
        env.close()
        if len(forks) != len(selected): raise ValueError('Not all actual prefix samples were forked')
        reports.append(dict(contract=contract,status='PASS',advanced_steps=advanced,forks=forks,
            strata={str(k):i for k,i in groups.items()},skipped_over_local_step_budget=skipped_long,
            elapsed_seconds=time.monotonic()-began,trace_sha256=rec['manifest_sha256']))
        print(f'actual forks {contract}: {len(forks)}, {advanced} teacher steps',flush=True)
    result = signed(dict(status='PASS',protocol='FG_S3B_REAL_ACCOUNT_FORKS_V1',seed=seed,
        values_manifest_sha256=digest(Path(values)/'manifest.json'),max_cursor=max_cursor,
        max_local_steps=max_local_steps,random_count=random_count,
        verification_code_sha256=digest(__file__),contracts=reports,
        scope='Stratified actual prefix account forks; not exhaustive full TRAIN account enumeration'))
    destination = Path(output)
    if destination.exists(): raise FileExistsError('Fork receipt exists')
    write_json(destination,result)
    return result

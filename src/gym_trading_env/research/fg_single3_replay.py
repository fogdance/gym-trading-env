"""Single3 policies and orchestration over the original R0-R2 collector."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .fg_replay import (BehaviorPolicy, SINGLE3_SCHEMA, check_files, collect_trace,
                       digest, iter_replay_rows, iter_sidecar, read_manifest,
                       reconcile_trace, signed, trace_manifest, write_json)
from .fg_single3 import load_env


class Single3BehaviorPolicy(BehaviorPolicy):
    def __init__(self, spec):
        super().__init__(spec)
        self.cursor=0
        self.waiting=spec['kind']=='single_delay'
        self.source=None
        if spec.get('source_directory'):
            directory=Path(spec['source_directory'])
            manifest=trace_manifest(directory,schema=SINGLE3_SCHEMA)
            if manifest['replay_trace_hash']!=spec['source_trace_hash']:
                raise ValueError('Derived Reward view source identity changed')
            self.source=zip(iter_sidecar(directory),iter_replay_rows(directory,manifest))
        if spec['kind']=='oracle_schedule':
            plan=read_manifest(spec['schedule_path'])
            if (plan['oracle_optimality_status']!='exact_proven' or
                    plan['candidate']!=spec['candidate'] or plan['gamma']!=332/333 or
                    digest(spec['schedule_path'])!=spec['schedule_sha256']):
                raise ValueError('P4 requires a frozen exact H333 Oracle schedule')
            self.schedule=set(plan['add_cursors'])

    def observe_row(self,row,*,legal_after):
        self.cursor=row['cursor']
        if self.spec['kind']!='single_delay' or not row['advanced']:
            return
        filled=any(f['reason']=='ENTRY' for f in row['fills'])
        if filled or row['exits'] or row['lots_after']!=0 or not legal_after or not row['legal_add_before']:
            self.waited=0
            self.waiting=True
        elif row['valid_bar']:
            self.waited+=1
            self.waiting=self.waited<self.spec['wait_minutes']

    def __call__(self,carry,obs):
        if self.source is not None:
            item=next(self.source,None)
            if item is None:
                raise ValueError('Derived view outlived its complete source account')
            row,(_,step)=item
            if row['cursor']!=self.cursor:
                raise ValueError('Derived Reward view changed the account cursor')
            for key in ('market_seq','position_state','action_mask','is_first','is_last','is_terminal'):
                np.testing.assert_array_equal(obs[key][0],step[key],err_msg=f'Derived view {key}')
            self.waited,self.waiting=row['p3_waited'],row['p3_waiting']
            return carry,{'action':np.array([row['action']],np.int32)},{}
        kind=self.spec['kind']
        if kind in ('always_add','probabilistic'):
            return super().__call__(carry,obs)
        active=not bool(obs['is_last'][0])
        legal=bool(obs['action_mask'][0,1])
        if kind=='always_wait':
            action=0
        elif kind=='single_delay':
            action=int(active and legal and not self.waiting)
        elif kind=='oracle_schedule':
            planned=self.cursor in self.schedule
            if planned and active and not legal:
                raise ValueError('Exact Oracle action is not legal in the actual account')
            action=int(active and planned)
        else:
            raise ValueError('Unknown Single3 behavior policy')
        return carry,{'action':np.array([action],np.int32)},{}


def specs(config,index):
    result=[]
    for number,original in enumerate(config['policies']):
        spec=dict(original,seed_components=[config['seed'],index,number])
        if (spec['kind'] not in ('always_wait','always_add','probabilistic','single_delay') or
                Path(spec['id']).name!=spec['id'] or spec['id'] in ('.','..','all') or
                (spec['kind']=='probabilistic' and not 0<=spec['probability']<=1) or
                (spec['kind']=='single_delay' and spec['wait_minutes']!=15)):
            raise ValueError('Invalid Single3 causal policy')
        result.append(spec)
    if len({s['id'] for s in result})!=len(result):
        raise ValueError('Duplicate policy IDs')
    return result


def trace_job(job):
    """Spawn-safe wrapper; original collector and original reconcile do the work."""
    check_files(job['frozen_files'])
    directory=Path(job['directory'])
    if job.get('resume') and (directory/'reconciliation.json').exists():
        manifest=trace_manifest(directory,schema=SINGLE3_SCHEMA)
        receipt=read_manifest(directory/'reconciliation.json')
        if (manifest['identity']!=job['identity'] or manifest['policy']!=job['spec'] or
                receipt['trace_manifest_sha256']!=digest(directory/'manifest.json')):
            raise ValueError('Single3 resume identity mismatch')
        return dict(directory=str(directory),manifest=manifest,receipt=receipt,resumed=True)
    if directory.exists():
        raise FileExistsError('Incomplete trace retained; select a new attempt directory')
    factory=lambda:load_env(job['row'],job['frozen_files'])
    manifest=collect_trace(job['row']['env_config'],job['spec'],directory,
        job['identity'],job['dreamer_root'],env_factory=factory,
        protocol='fg_single3_v1',policy_factory=Single3BehaviorPolicy)
    receipt=reconcile_trace(directory,job['dreamer_root'],env_factory=factory,schema=SINGLE3_SCHEMA)
    check_files(job['frozen_files'])
    receipt=signed(dict(receipt,trace_manifest_sha256=digest(directory/'manifest.json')))
    write_json(directory/'reconciliation.json',receipt)
    return dict(directory=str(directory),manifest=manifest,receipt=receipt,resumed=False)


def require_e(path):
    path=Path(path).resolve()
    receipt=read_manifest(path/'manifest.json')
    if receipt.get('stage')!='E' or receipt.get('status')!='PASS':
        raise ValueError('Single3 collection requires the real Env E Gate')
    check_files({str(path/p):h for p,h in receipt['files'].items()})
    check_files(receipt['frozen_inputs'])
    engine={p:h for p,h in receipt['provenance']['code_files'].items()
            if '/envs/' in p and '/gym_trading_env/' in p or '/utils/' in p or
            p.endswith('/single3_reward.py') or p.endswith('/fg_reward_candidates.py')}
    check_files(engine)
    inputs=read_manifest(path/'inputs/inputs.json')
    return inputs,receipt

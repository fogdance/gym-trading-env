"""Teacher-free adapter of the original Collector, Driver and financial Env."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_s3_v2_collector import V2Collector
from .fg_replay import (E_QUALITY_SCHEMA, E_QUALITY_PROTOCOL, execute_observed, replay_fields,
    observed_trace_row, observed_trace_manifest, update_array_hash, write_json, signed)


class OnlineCollector(V2Collector):
    protocol = 'FG_S3_E_QUALITY_ONLINE_COLLECTOR_V1'

    def __init__(self, env, source, directory, dreamer_root, *, agent, seed, budget,
                 capacity=2048, initial_state=None, existing=False, replay_length=65, chunksize=1024):
        from embodied.run.fg_s3_e_quality_online import OnlinePolicy
        from .fg_s3_oracle_state_query import rules_identity
        from .fg_s3_e_observation import OBS_SCHEMA
        if source.get('fold') != 'TRAIN' or getattr(env,'_e_profile',None) is None:
            raise ValueError('ONLINE_NATIVE_TRAIN_SOURCE_REQUIRED')
        if env.reward_config.schema != 'FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1':
            raise ValueError('ONLINE_NATIVE_QUALITY_TP_REQUIRED')
        if type(replay_length) is not int or replay_length < 3 or capacity < replay_length:
            raise ValueError('ONLINE_REPLAY_GEOMETRY_INVALID')
        if initial_state is None:
            env.reset(seed=seed)
            initial_state=env.export_state()
        self.initial_state=copy.deepcopy(initial_state)
        self.replay_length=replay_length;self.chunk_size=chunksize
        self.trace_policy=OnlinePolicy(agent)
        # Identity-only handle. No teacher/query object or future reader exists
        # in this adapter; the inherited disabled mailbox only reads metadata.
        handle=SimpleNamespace(identity=dict(fold='TRAIN',source=source),
            identity_hash=fingerprint(source),rules=rules_identity(env))
        super().__init__(env,handle,directory,dreamer_root,seed=seed,beta=0.,
            actor=self.trace_policy,agent=agent,existing=existing,teacher_enabled=False,
            observation_schema=OBS_SCHEMA,e_profile=env._e_profile,
            replay_capacity=capacity,max_env_steps=max(65,budget))
        self.identity.update(execution_protocol=self.protocol,source=copy.deepcopy(source),
            replay_length=replay_length,chunksize=chunksize)
        self.identity_hash=fingerprint(self.identity)
        self.rows=[];self.snapshots={};self.index={};self.scorer=env._new_reward()
        self.archived=[]
        self.archive=self.root/'trace_archive'
        self.archive.mkdir(exist_ok=existing)

    def make_replay(self,embodied,seed):
        return embodied.Replay(length=self.replay_length,capacity=self.replay_capacity,online=True,
            directory=str(self.root/'replay'),chunksize=self.chunk_size,save_wait=True,seed=seed)

    def context_snapshot(self):
        # Inherited V2's implementation calls oracle.decision_state even with
        # beta=0. A real account snapshot has no need for that Oracle route.
        return self.decision_env.export_state()

    def make_observed(self,env):
        import gymnasium as gym
        initial=self.initial_state
        class Observed(gym.Wrapper):
            first_reset=True
            @property
            def e_snapshot(self):return self.env.e_snapshot
            def reset(self,*,seed=None,options=None):
                if options:
                    result=self.env.reset(seed=seed,options=options)
                    self.first_reset=False
                    return result
                obs,self.last=execute_observed(self.env,seed=seed,
                    resume_state=initial if self.first_reset else None)
                self.first_reset=False
                return obs,self.env._get_info()
            def step(self,action):
                obs,self.last=execute_observed(self.env,action)
                return obs,self.last['official_reward'],self.last['is_terminal'],self.env.truncated,self.env._get_info()
        return Observed(env)

    def capture(self,step,worker):
        fields=replay_fields(E_QUALITY_SCHEMA)
        native={k:np.asarray(step[k]) for k in fields if k!='market_advance'}
        row,native=observed_trace_row(self.decision_env,native,self.observed.last,self.scorer,
            self.trace_policy,len(self.rows),single3=True,e_quality=True)
        if int(native['action']) != self.policy.decision['executed_action']:
            raise ValueError('ONLINE_DRIVER_CHANGED_ACTOR_ACTION')
        with self.commit_lock:
            self.replay.add(native,worker)
            chunkid,offset=self.replay.streams[worker][-1]
            row['stepid']=(bytes(chunkid)+offset.to_bytes(4,'big')).hex()
            row.update(decision_executed=False,decision_valid_minute=False,
                       account_owner=fingerprint(dict(run=self.identity_hash,contract=self.env.contract,
                           account_sequence=self.env.account_sequence)),
                       model_update=int(self.agent.n_updates))
            if self.rows:
                self.rows[-1].update(decision_executed=row['advanced'],decision_valid_minute=row['valid_bar'])
            self.rows.append(row);self.index[row['stepid']]=row
            self.snapshots[row['stepid']]=self.env.export_state()
            self.counts['env_steps']+=int(row['advanced'])
            self.counts['native_rows']+=1;self.counts['resets']+=int(row['reset'])
            self.counts['tp_events']+=sum(x['reason']=='TAKE_PROFIT' for x in row['exits'])
            self.counts['sl_events']+=sum(x['reason']=='STOP_LOSS' for x in row['exits'])
            archived={c['uuid'] for c in self.archived}
            for chunk in self.replay.chunks.values():
                if chunk.length==chunk.size and str(chunk.uuid) not in archived:
                    # Save the ORIGINAL immutable native chunk before capacity
                    # eviction. This is a trace archive, not a second Replay.
                    chunk.save(self.archive)
                    from .fg_replay import digest
                    self.archived.append(dict(path=chunk.filename,uuid=str(chunk.uuid),
                        succ=str(chunk.succ),length=chunk.length,sha256=digest(self.archive/chunk.filename)))

    def prune(self):
        # The original Replay owns capacity eviction. Full account audit rows
        # remain separate from sampleable rows and never become observations.
        self.query_seconds=[]

    def snapshot_trace(self,directory,identity):
        out=Path(directory)
        if out.exists():raise ValueError('ONLINE_TRACE_OUTPUT_MUST_BE_NEW')
        out.mkdir(parents=True);(out/'replay').mkdir()
        rows=copy.deepcopy(self.rows)
        import shutil
        from .fg_replay import digest
        chunks=[];hasher=hashlib.sha256();events=hashlib.sha256();count=0
        for saved in self.archived:
            path=self.archive/saved['path']
            if digest(path)!=saved['sha256']:raise ValueError('ONLINE_TRACE_ARCHIVE_CHANGED')
            shutil.copyfile(path,out/'replay'/saved['path'])
            chunks.append({k:saved[k] for k in ('uuid','succ','length')}|dict(path='replay/'+saved['path']))
            with np.load(path,allow_pickle=False) as arrays:
                for i in range(saved['length']):update_array_hash(hasher,{k:arrays[k][i] for k in arrays.files})
            count+=saved['length']
        archived={c['uuid'] for c in self.archived}
        for chunk in self.replay.chunks.values():
            if not chunk.length or str(chunk.uuid) in archived:continue
            arrays={k:v[:chunk.length] for k,v in chunk.data.items()}
            np.savez_compressed(out/'replay'/chunk.filename,**arrays)
            chunks.append(dict(path='replay/'+chunk.filename,uuid=str(chunk.uuid),succ=str(chunk.succ),length=chunk.length))
            for i in range(chunk.length):
                update_array_hash(hasher,{k:v[i] for k,v in arrays.items()})
                count+=1
        if count!=len(rows):raise ValueError('ONLINE_FULL_TRACE_ARCHIVE_REQUIRED_AFTER_REPLAY_EVICTION')
        with gzip.open(out/'steps.jsonl.gz','wt',encoding='utf8') as f:
            for row in rows:
                events.update(json.dumps({k:v for k,v in row.items() if k not in ('stepid','ordinal')},
                    sort_keys=True,separators=(',',':'),allow_nan=False).encode())
                f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        write_json(out/'account_state.json',self.env.export_state())
        write_json(out/'trade_diagnostics.json',self.scorer.scorer.trade_rows())
        if self.initial_state is not None:write_json(out/'initial_state.json',self.initial_state)
        state=dict(records=len(rows),advanced=self.counts['env_steps'],
            valid_minutes=sum(r['valid_bar'] for r in rows),chunk_resets=sum(r['reset'] and r['cursor']>0 for r in rows))
        m=observed_trace_manifest(self.env,out,state,chunks,identity,dict(id='ONLINE_QUALITY_ACTOR',kind='STUDENT_ONLY'),
            None,E_QUALITY_PROTOCOL,self.initial_state,self.env.current_step,hasher.hexdigest(),events.hexdigest(),
            0.,single3=True,e_quality=True)
        write_json(out/'manifest.json',signed(m))
        return m

    def save(self):
        with self.commit_lock:
            return copy.deepcopy(dict(protocol=self.protocol,identity=self.identity_hash,rows=self.rows,
                snapshots=self.snapshots,counts=dict(self.counts),scorer=self.scorer,
                observed_last=getattr(self.observed,'last',None),
                mixing_rng=self.policy.rng.bit_generator.state,decision=self.policy.decision,
                actor_diagnostics=self.trace_policy.last_diagnostics,archived=self.archived))

    def load(self,state):
        if state['protocol']!=self.protocol or state['identity']!=self.identity_hash:
            raise ValueError('ONLINE_COLLECTOR_IDENTITY_MISMATCH')
        with self.commit_lock:
            self.rows=copy.deepcopy(state['rows']);self.snapshots=copy.deepcopy(state['snapshots'])
            self.index={r['stepid']:r for r in self.rows};self.counts.clear();self.counts.update(state['counts'])
            self.scorer=copy.deepcopy(state['scorer']);self.observed.last=copy.deepcopy(state['observed_last'])
            self.policy.rng.bit_generator.state=copy.deepcopy(state['mixing_rng'])
            self.policy.decision=copy.deepcopy(state['decision'])
            self.trace_policy.original.last_diagnostics=copy.deepcopy(state['actor_diagnostics'])
            self.archived=copy.deepcopy(state['archived'])

    def close(self):
        self.replay.workers.shutdown(wait=True)
        self.driver.close()

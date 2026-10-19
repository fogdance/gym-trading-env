"""Persistent teacher-free native Driver/Trace session. No training Replay."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .fg_replay import (execute_observed,observed_trace_row,observed_trace_manifest,
    update_array_hash,write_json,signed,digest,E_QUALITY_PROTOCOL,E_QUALITY_SCHEMA,reconcile_trace)
from .fg_s3_e_quality_exam_contract import validate_plan,plan_hash,identity
from .fg_s3_e_quality_course_preflight import _local_path


class ExamSession:
    """One contract, account, pending action and RSSM carry across all chunks."""
    protocol='FG_S3_E_QUALITY_EXAM_SESSION_V1'

    def __init__(self,plan,source,policy,directory,*,policy_id,existing=False):
        import embodied
        import gymnasium as gym
        from embodied.envs.from_gymnasium import FromGymnasium
        from embodied.run.fg_s3_e_quality_online_state import ChunkIdentity
        from .fg_single3_evaluation import embodied_runtime
        from .fg_s3_e_quality_foundation import PROJECT,execution_identity
        self.plan=validate_plan(plan);self.source=source;self.policy=policy
        self.root=_local_path(directory);self.policy_id=policy_id
        if not self.root.is_relative_to(Path(plan['output_dir'])):raise ValueError('EXAM_SESSION_OUTSIDE_PLAN_OUTPUT')
        if self.root.exists()!=existing:raise ValueError('EXAM_SESSION_NEW_OUTPUT_OR_EXPLICIT_RESUME_REQUIRED')
        source.require_mode(plan)
        if hasattr(policy,'require_mode'):policy.require_mode(plan)
        source.verify();self.raw=source.factory()
        if (self.raw._e_profile is None or self.raw.reward_config.schema!='FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1'
                or self.raw.entry_config.get('state_path') or self.raw.chunk_steps):
            self.raw.close();raise ValueError('EXAM_NATIVE_CONTINUOUS_QUALITY_E_ENV_REQUIRED')
        role=source.receipt.get('role')
        if plan['exam_mode']=='DEV_EXAM' and (role!='validation' or source.receipt.get('eligible_for_training') is not False):
            self.raw.close();raise ValueError('EXAM_DEV_SOURCE_ADMISSION_REQUIRED')
        if self.raw.contract not in plan['contracts']:raise ValueError('EXAM_CONTRACT_NOT_IN_PLAN')
        self.initial_state=copy.deepcopy(source.initial_state)
        if self.initial_state['entries']:raise ValueError('EXAM_NEW_FLAT_ACCOUNT_INITIAL_STATE_REQUIRED')
        self.raw.import_state(self.initial_state)
        self.scorer=copy.deepcopy(self.raw.reward_function)
        self.identity=dict(protocol=self.protocol,plan_hash=plan_hash(plan),source=identity(source.receipt),
            actor=identity(policy.receipt) if hasattr(policy,'receipt') else identity(policy.spec),
            contract=self.raw.contract,policy_id=policy_id,code=execution_identity())
        self.chunk_ids=ChunkIdentity(identity(self.identity));self.chunks=[];self.chunk=None
        self.chunk_rows=[];self.pending=None;self.records=0;self.advanced=0;self.valid_minutes=0
        self.finished=False;self.metrics_state={};self._new_chunk()
        session=self
        class Observed(gym.Wrapper):
            first_reset=True
            last=None
            def reset(self,*,seed=None,options=None):
                if options is not None:
                    if set(options)!={'resume'}:raise ValueError('EXAM_ACCOUNT_RESET_FORBIDDEN')
                    return self.env.reset(options=options)
                if not self.first_reset:raise ValueError('EXAM_UNEXPECTED_ACCOUNT_RESET')
                obs,self.last=execute_observed(self.env,seed=seed,resume_state=session.initial_state)
                self.first_reset=False
                return obs,self.env._get_info()
            def step(self,action):
                obs,self.last=execute_observed(self.env,action)
                session.policy.observe_row(self.last,legal_after=bool(obs['action_mask'][1]))
                return obs,self.last['official_reward'],self.last['is_terminal'],self.env.truncated,self.env._get_info()
        self.observed=Observed(self.raw)
        wrapped=embodied.wrappers.CheckSpaces(embodied.wrappers.UnifyDtypes(
            FromGymnasium(self.observed,reset_seed=0)))
        self.driver=embodied.Driver([lambda:wrapped],parallel=False)
        self.driver.on_step(self._capture)
        embodied_runtime(PROJECT.parent/'dreamerv3')
        if not existing:
            self.root.mkdir(parents=True);(self.root/'chunks').mkdir()
            write_json(self.root/'SESSION_IDENTITY.json',signed(self.identity))
            write_json(self.root/'PLAN.json',plan)
            write_json(self.root/'SOURCE.json',source.receipt)

    def _new_chunk(self):
        from embodied.core.chunk import Chunk
        self.chunk=Chunk(self.plan['chunk_steps'])
        self.chunk.uuid,self.chunk.time=self.chunk_ids.allocate()
        # The successor is deterministic without consuming its sequence yet.
        future=copy.deepcopy(self.chunk_ids);self.chunk.succ,_=future.allocate()
        self.chunk_rows=[]

    def _append(self,row,step):
        offset=self.chunk.length
        stepid=bytes(self.chunk.uuid)+offset.to_bytes(4,'big')
        row=copy.deepcopy(row);row['stepid']=stepid.hex()
        self.chunk.append({**step,'stepid':np.frombuffer(stepid,dtype=np.uint8).copy()})
        self.chunk_rows.append(row)
        if self.chunk.length==self.chunk.size:self._flush()

    def _flush(self):
        if not self.chunk.length:return
        folder=self.root/'chunks';self.chunk.save(folder)
        rows_path=folder/(self.chunk.filename+'.jsonl')
        with rows_path.open('x') as stream:
            for row in self.chunk_rows:stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        self.chunks.append(dict(path='chunks/'+self.chunk.filename,rows_path='chunks/'+rows_path.name,
            uuid=str(self.chunk.uuid),succ=str(self.chunk.succ),length=self.chunk.length,
            sha256=digest(folder/self.chunk.filename),rows_sha256=digest(rows_path)))
        self._new_chunk()

    def _capture(self,step,worker):
        row,step=observed_trace_row(self.raw,step,self.observed.last,self.scorer,self.policy,
            self.records,single3=True,e_quality=True)
        row.update(decision_executed=False,decision_valid_minute=False,
            account_owner=identity(dict(session=identity(self.identity),contract=self.raw.contract,
                account_sequence=self.raw.account_sequence)))
        if self.pending is not None:
            prior,arrays=self.pending
            prior.update(decision_executed=row['advanced'],decision_valid_minute=row['valid_bar'])
            self._append(prior,arrays)
        self.pending=(row,{k:np.array(v,copy=True) for k,v in step.items()})
        self.records+=1;self.advanced+=int(row['advanced']);self.valid_minutes+=int(row['valid_bar'])
        self.metrics_state['reward']=self.metrics_state.get('reward',0.)+row['official_reward']
        self.metrics_state['entries']=self.metrics_state.get('entries',0)+sum(f['reason']=='ENTRY' and f['committed'] for f in row['fills'])
        days=self.metrics_state.setdefault('official_day_valid_minutes',{})
        if row['valid_bar']:days[row['trading_date']]=days.get(row['trading_date'],0)+1

    def advance(self,target_minutes=None):
        if self.finished:raise ValueError('EXAM_SESSION_ALREADY_FINISHED')
        if target_minutes is not None and (type(target_minutes) is not int or target_minutes<self.valid_minutes):
            raise ValueError('EXAM_FORWARD_MINUTE_BOUNDARY_REQUIRED')
        while (target_minutes is None or self.valid_minutes<target_minutes) and self.raw.current_step<self.source.end_cursor:
            self.driver(self.policy,steps=1)
            if self.raw.status in ('DATA_END','DATA_GAP') or self.raw.terminated:break
        return dict(valid_minutes=self.valid_minutes,advanced=self.advanced,cursor=self.raw.current_step)

    def rows(self,*,include_pending=True):
        for c in self.chunks:
            if digest(self.root/c['rows_path'])!=c['rows_sha256'] or digest(self.root/c['path'])!=c['sha256']:
                raise ValueError('EXAM_IMMUTABLE_CHUNK_CHANGED')
            for line in (self.root/c['rows_path']).read_text().splitlines():yield json.loads(line)
        yield from copy.deepcopy(self.chunk_rows)
        if include_pending and self.pending is not None:yield copy.deepcopy(self.pending[0])

    def finish(self,*,verify=True):
        from .fg_s3_e_quality_foundation import PROJECT
        from .fg_single3_evaluation import embodied_runtime
        if self.finished:raise ValueError('EXAM_SESSION_ALREADY_FINISHED')
        if self.plan['exam_mode']=='DEV_EXAM' and self.raw.current_step<self.source.end_cursor and not self.raw.terminated:
            raise ValueError('EXAM_DEV_INTERVAL_INCOMPLETE')
        if self.pending is None:raise ValueError('EXAM_NO_OBSERVATIONS')
        self._append(*self.pending);self.pending=None;self._flush();self.finished=True
        out=self.root/'trace'
        if out.exists():raise ValueError('EXAM_TRACE_OUTPUT_MUST_BE_NEW')
        out.mkdir();(out/'replay').mkdir()
        array_hash=hashlib.sha256();event_hash=hashlib.sha256();chunks=[]
        for c in self.chunks:
            source=self.root/c['path'];dest=out/'replay'/source.name
            if digest(source)!=c['sha256']:raise ValueError('EXAM_IMMUTABLE_CHUNK_CHANGED')
            shutil.copyfile(source,dest)
            chunks.append({k:c[k] for k in ('uuid','succ','length')}|dict(path='replay/'+dest.name))
            with np.load(source,allow_pickle=False) as arrays:
                for i in range(c['length']):update_array_hash(array_hash,{k:arrays[k][i] for k in arrays.files})
        with gzip.open(out/'steps.jsonl.gz','wt',encoding='utf8') as stream:
            for row in self.rows():
                canonical={k:v for k,v in row.items() if k not in ('stepid','ordinal')}
                event_hash.update(json.dumps(canonical,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
                stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        write_json(out/'account_state.json',self.raw.export_state())
        write_json(out/'initial_state.json',self.initial_state)
        write_json(out/'trade_diagnostics.json',self.scorer.scorer.trade_rows())
        manifest=observed_trace_manifest(self.raw,out,dict(records=self.records,advanced=self.advanced,
            valid_minutes=self.valid_minutes,chunk_resets=0),chunks,self.identity,
            dict(id=self.policy_id,kind='STUDENT_ONLY' if hasattr(self.policy,'agent') else self.policy.spec['kind']),
            None,E_QUALITY_PROTOCOL,self.initial_state,self.raw.current_step,array_hash.hexdigest(),
            event_hash.hexdigest(),None,single3=True,e_quality=True)
        write_json(out/'manifest.json',signed(manifest))
        replay=reconcile_trace(out,PROJECT.parent/'dreamerv3',env_factory=self.source.factory,
            schema=E_QUALITY_SCHEMA,runtime_factory=embodied_runtime) if verify else None
        frozen=self.policy.verify_frozen() if hasattr(self.policy,'verify_frozen') else dict(training_updates=0)
        self.source.verify()
        result=signed(dict(protocol=self.protocol,exam_mode=self.plan['exam_mode'],contract=self.raw.contract,
            policy_id=self.policy_id,plan_hash=plan_hash(self.plan),session_identity=identity(self.identity),
            source_hash=identity(self.source.receipt),actor_identity=self.identity['actor'],
            valid_minutes=self.valid_minutes,trace_hash=manifest['replay_trace_hash'],reconciliation=replay,
            frozen=frozen,teacher_used=False,training_updates=0,test_prices_accessed=False,
            complete_interval=self.raw.current_step>=self.source.end_cursor or self.raw.terminated))
        write_json(self.root/'SESSION_RECEIPT.json',result)
        return result

    def close(self):self.driver.close()

    def commit(self):
        from .fg_s3_e_quality_exam_state import commit
        return commit(self)

    def restore(self):
        from .fg_s3_e_quality_exam_state import restore
        return restore(self)

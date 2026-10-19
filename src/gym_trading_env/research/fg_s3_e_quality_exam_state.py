"""Atomic evaluation state, distinct from a training FullCheckpoint."""
from pathlib import Path
import copy
import os
import pickle
import shutil

import numpy as np

from .fg_replay import read_manifest,write_json,signed,digest
from .fg_s3_e_quality_course_preflight import _local_path

PROTOCOL='FG_S3_E_QUALITY_EVAL_STATE_V1'


def _member(root,value):
    if not isinstance(value,str) or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('EXAM_STATE_MEMBER_PATH_INVALID')
    path=_local_path(Path(root)/value)
    if not path.is_relative_to(Path(root)):raise ValueError('EXAM_STATE_MEMBER_PATH_INVALID')
    return path


def _chunk_state(chunk):
    return dict(size=chunk.size,length=chunk.length,uuid=str(chunk.uuid),succ=str(chunk.succ),time=chunk.time,
        data=None if chunk.data is None else {k:v[:chunk.length].copy() for k,v in chunk.data.items()})


def _chunk_load(state):
    import elements
    from embodied.core.chunk import Chunk
    c=Chunk(state['size']);c.length=state['length'];c.uuid=elements.UUID(state['uuid'])
    c.succ=elements.UUID(state['succ']);c.time=state['time']
    if state['data'] is not None:
        c.data={k:np.empty((c.size,*v.shape[1:]),v.dtype) for k,v in state['data'].items()}
        for k,v in state['data'].items():c.data[k][:c.length]=v
    return c


def snapshot(session):
    policy=session.policy
    if hasattr(policy,'verify_frozen'):policy.verify_frozen()
    session.source.verify()
    return dict(protocol=PROTOCOL,identity=copy.deepcopy(session.identity),
        driver=session.driver.save(),policy=policy.save(),initial_state=session.initial_state,
        source_end=session.source.end_cursor,scorer=session.scorer.snapshot(),
        chunk_ids=session.chunk_ids.save(),chunk=_chunk_state(session.chunk),
        chunk_rows=copy.deepcopy(session.chunk_rows),pending=copy.deepcopy(session.pending),
        chunks=copy.deepcopy(session.chunks),records=session.records,advanced=session.advanced,
        valid_minutes=session.valid_minutes,finished=session.finished,
        observed_last=copy.deepcopy(session.observed.last),first_reset=session.observed.first_reset,
        metrics_state=copy.deepcopy(session.metrics_state),
        log_prefixes=[dict(path=c['rows_path'],sha256=c['rows_sha256'],records=c['length']) for c in session.chunks])


def validate_snapshot(session,state):
    from .fg_s3_e_quality_exam_contract import identity,plan_hash
    if (plan_hash(session.plan)!=session.identity['plan_hash']
            or identity(session.source.receipt)!=session.identity['source']
            or identity(session.policy.receipt if hasattr(session.policy,'receipt') else session.policy.spec)!=session.identity['actor']):
        raise ValueError('EXAM_CURRENT_PLAN_OR_SOURCE_CHANGED')
    if (state.get('protocol')!=PROTOCOL or state.get('identity')!=session.identity
            or state.get('initial_state')!=session.initial_state or state.get('source_end')!=session.source.end_cursor):
        raise ValueError('EXAM_STATE_IDENTITY_MISMATCH')
    count=sum(c['length'] for c in state['chunks'])+state['chunk']['length']+int(state['pending'] is not None)
    if (count!=state['records'] or state['chunk']['length']!=len(state['chunk_rows'])
            or state['chunk']['size']!=session.plan['chunk_steps']
            or state['log_prefixes']!=[dict(path=c['rows_path'],sha256=c['rows_sha256'],records=c['length']) for c in state['chunks']]):
        raise ValueError('EXAM_STATE_COMMIT_BOUNDARY_MISMATCH')
    seen=set()
    for i,c in enumerate(state['chunks']):
        if c['uuid'] in seen or i and state['chunks'][i-1]['succ']!=c['uuid']:
            raise ValueError('EXAM_STATE_CHUNK_CHAIN_MISMATCH')
        seen.add(c['uuid'])
        for name,key in (('path','sha256'),('rows_path','rows_sha256')):
            if digest(_member(session.root,c[name]))!=c[key]:raise ValueError('EXAM_STATE_IMMUTABLE_CHUNK_CHANGED')
    if state['chunks'] and state['chunks'][-1]['succ']!=state['chunk']['uuid']:
        raise ValueError('EXAM_STATE_CHUNK_BUFFER_CHAIN_MISMATCH')
    session.chunk_ids.validate_snapshot(state['chunk_ids'])
    session.source.verify()
    if hasattr(session.policy,'verify_frozen'):session.policy.verify_frozen()


def commit(session):
    root=session.root;states=root/'states';states.mkdir(exist_ok=True)
    state=snapshot(session);validate_snapshot(session,state)
    latest=states/'latest.json'
    sequence=read_manifest(latest)['sequence']+1 if latest.exists() else 1
    stem=f'state_{sequence:012d}'
    payload=states/(stem+'.pkl');manifest=states/(stem+'.json')
    if payload.exists() or manifest.exists():raise ValueError('EXAM_STATE_IMMUTABLE_NODE_EXISTS')
    data=pickle.dumps(state,protocol=5)
    with payload.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    write_json(manifest,signed(dict(protocol=PROTOCOL,identity=session.identity,sequence=sequence,
        payload=payload.name,payload_sha256=digest(payload),records=state['records'],valid_minutes=state['valid_minutes'])))
    with manifest.open('rb') as stream:os.fsync(stream.fileno())
    write_json(latest,signed(dict(protocol=PROTOCOL,sequence=sequence,node=manifest.name,
        node_sha256=digest(manifest))))
    with latest.open('rb') as stream:os.fsync(stream.fileno())
    fd=os.open(states,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return str(manifest)


def restore(session):
    root=session.root;states=root/'states';latest=read_manifest(_local_path(states/'latest.json'))
    if latest.get('protocol')!=PROTOCOL:raise ValueError('EXAM_STATE_LATEST_PROTOCOL_MISMATCH')
    manifest=_member(states,latest['node'])
    if digest(manifest)!=latest['node_sha256']:raise ValueError('EXAM_STATE_LATEST_NODE_CHANGED')
    meta=read_manifest(manifest)
    if (meta.get('protocol')!=PROTOCOL or meta.get('identity')!=session.identity
            or meta.get('sequence')!=latest['sequence']):raise ValueError('EXAM_STATE_LATEST_IDENTITY_MISMATCH')
    payload=_member(states,meta['payload'])
    if digest(payload)!=meta['payload_sha256']:raise ValueError('EXAM_STATE_PAYLOAD_CHANGED')
    state=pickle.loads(payload.read_bytes());validate_snapshot(session,state)
    if meta['records']!=state['records'] or meta['valid_minutes']!=state['valid_minutes']:
        raise ValueError('EXAM_STATE_MANIFEST_BOUNDARY_MISMATCH')
    session.driver.load(state['driver'])
    session.policy.load(state['policy'])
    from gym_trading_env.rewards.single3_reward import Single3Reward
    session.scorer=Single3Reward.restore(state['scorer'],session.raw.reward_config)
    session.chunk_ids.load(state['chunk_ids']);session.chunk=_chunk_load(state['chunk'])
    for name in ('chunk_rows','pending','chunks','records','advanced','valid_minutes','finished','metrics_state'):
        setattr(session,name,copy.deepcopy(state[name]))
    session.observed.last=copy.deepcopy(state['observed_last']);session.observed.first_reset=state['first_reset']
    # An interrupted writer may have produced an uncommitted future chunk.
    # Preserve it outside the committed namespace before deterministic retry.
    committed={Path(c[k]).name for c in session.chunks for k in ('path','rows_path')}
    extras=[p for p in (root/'chunks').iterdir() if p.name not in committed]
    if extras:
        recovery=root/'recovery_chunks'/f'after_{latest["sequence"]:012d}'
        n=0
        while recovery.exists():n+=1;recovery=root/'recovery_chunks'/f'after_{latest["sequence"]:012d}_{n}'
        recovery.mkdir(parents=True)
        for p in extras:shutil.move(str(p),str(recovery/p.name))
    if session.raw.current_step!=state['driver']['envs'][0]['env']['current_step']:
        raise ValueError('EXAM_RESTORED_NATIVE_CURSOR_MISMATCH')
    return str(manifest)

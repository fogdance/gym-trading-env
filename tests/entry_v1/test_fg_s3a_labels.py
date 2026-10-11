"""Native original-Env C1_P4 regression fixtures and fail-closed checks."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gym_trading_env.research.fg_replay import (SINGLE3_SCHEMA, collect_trace, digest,
    iter_sidecar, iter_replay_rows, read_manifest, reconcile_trace, signed, write_json)
from gym_trading_env.research.fg_single3_replay import Single3BehaviorPolicy
from gym_trading_env.research.fg_s3a_labels import (aligned_rows, build_labels, exclusion,
    load_labels, sources)
from test_fg_single3_env import single_env

DREAMER = '/home/v/Documents/work/dreamerv3-dev'


def native_dataset(root):
    """Explicit synthetic Oracle receipt; no production admission bypass."""
    envdir = root/'env'; envdir.mkdir(parents=True)
    prices = [(1000,1000,1000,1000)]*1100
    prices[2] = prices[1030] = (1000,1007,1000,1007)
    sessions=[];times=[]
    for day in pd.date_range('2026-01-05',periods=4):
        date=day.strftime('%Y-%m-%d')
        for begin,end in [('09:31','10:14'),('10:31','11:29'),('13:31','14:59'),('21:01','22:59')]:
            sessions.append(dict(start=str(pd.Timestamp(date+' '+begin)-pd.Timedelta(minutes=1)),end=date+' '+end))
            times.extend(pd.date_range(date+' '+begin,date+' '+end,freq='min'))
    factory = lambda: single_env(envdir, prices, candidate='C1', chunk=1024,
        times=times[:1100],calendar=sessions,liquidation='2026-01-09 15:00',
        intervals=[dict(contract='CZCE.FG2701',start='2026-01-05',end='2026-01-10',direction='LONG')])
    env = factory(); contract = env.contract; env.close()
    oracle = root/'oracle'; oracle.mkdir()
    identity = dict(schema='FG_SINGLE3_ORACLE_V1', training_hash='synthetic_TRAIN',
                    selected_contracts=[contract], objective='native_float32_feedback_rational_gamma_env_advance_v1')
    plan_path = oracle/'C1_H333.json'
    plan = dict(candidate='C1', gamma=332/333, gamma_rational=[332,333],
                oracle_optimality_status='exact_proven', training_only=True,
                contract=contract, identity=identity, add_cursors=[0,1022])
    write_json(plan_path, signed(plan))
    dataset = root/'dataset'; trace = dataset/'traces/C1/P4'/contract
    spec = dict(id='P4', kind='oracle_schedule', candidate='C1', seed_components=[0],
                schedule_path=str(plan_path), schedule_sha256=digest(plan_path))
    t = collect_trace(envdir/'env.yaml', spec, trace, identity, DREAMER,
                      env_factory=factory, protocol='fg_single3_v1', policy_factory=Single3BehaviorPolicy)
    receipt = reconcile_trace(trace, DREAMER, env_factory=factory, schema=SINGLE3_SCHEMA)
    write_json(trace/'reconciliation.json', signed(dict(receipt,trace_manifest_sha256=digest(trace/'manifest.json'))))
    write_json(oracle/'manifest.json', signed(dict(stage='O', status='PASS',
        full_training_fold=True, oracle_optimality_status='exact_proven', identity=identity,
        files={'C1_H333.json':digest(plan_path)})))
    # Frozen synthetic TRAIN prices/config are verified through the same code
    # as production. Explicit metadata does not pretend to be an independent O Gate proof.
    row = dict(contract=contract, role='train', eligible_for_training=True, complete_lifecycle=True,
               path=str(envdir/'env.yaml'), sha256=digest(envdir/'env.yaml'),
               env_config=str(envdir/'env.yaml'), env_sha256=digest(envdir/'env.yaml'),
               contract_config=str(envdir/'env.yaml'), contract_sha256=digest(envdir/'env.yaml'))
    train_path = root/'train.json'
    write_json(train_path, signed(dict(reward_candidate='C1', single3_protocol='fg_single3_v1',
        original_content_hash='synthetic_TRAIN', files=[row])))
    did = dict(schema=SINGLE3_SCHEMA, fold='TRAIN', conditional_direction_only=True,
               selected_contracts=[contract], original_training_hash='synthetic_TRAIN')
    write_json(dataset/'inputs.json', signed(dict(identity=did,
        training_views={'C1':dict(path=str(train_path),content_hash=read_manifest(train_path)['content_hash'], rows=[row])})))
    rec = dict(candidate='C1', policy_id='P4', contract=contract,
        directory=str(trace.relative_to(dataset)),manifest_sha256=digest(trace/'manifest.json'),
        receipt_sha256=digest(trace/'reconciliation.json'), trace_identity=t['identity'],
        records=t['records'], replay_trace_hash=t['replay_trace_hash'],
        reward_params_hash=t['reward_params_hash'],observation_identity=t['observation_identity'])
    write_json(dataset/'manifest.json', signed(dict(schema=SINGLE3_SCHEMA,stage='D_GENERATED',
        status='PASS',full_training_fold=True,identity=did,input_manifest_sha256=digest(dataset/'inputs.json'),
        traces=[rec],pool_members={'C1_P4':[0]})))
    return dataset, oracle, trace, t


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    return native_dataset(tmp_path_factory.mktemp('s3a_native'))


def test_native_outgoing_empty_after_exit_and_1024_labels(corpus,tmp_path):
    dataset, oracle, trace, manifest = corpus
    result = build_labels(dataset, oracle, tmp_path/'labels')
    a = np.load(tmp_path/'labels'/result['traces'][0]['labels_file'])
    assert a['cursor'][a['label']==1].tolist()==[0,1022]
    assert 2 in a['cursor']  # lots_before==3; current observation is now empty.
    assert 1024 not in a['cursor']  # still holding across the real chunk reset.
    assert result['summary']['counts']['administrative_last_excluded']==1
    assert result['summary']['counts']['lots_before_filter_would_lose']>0
    rows=list(iter_sidecar(trace))
    assert rows[0]['action']==1 and rows[1]['incoming_action']==1 and rows[1]['action']==0
    assert rows[1025]['is_first'] and rows[1025]['lots_after']==3
    assert load_labels(tmp_path/'labels')['summary']==result['summary']
    with pytest.raises(FileExistsError):build_labels(dataset,oracle,tmp_path/'labels')


def test_rejects_foreign_candidate_identity_and_hash(corpus,tmp_path):
    dataset,oracle,_,_=corpus
    with pytest.raises(ValueError,match='only accepts'):sources(dataset,oracle,'C2')
    from shutil import copytree
    cloned=tmp_path/'dataset';copytree(dataset,cloned)
    p=cloned/'manifest.json';m=read_manifest(p);m.pop('content_hash');m['identity']['fold']='DEV';write_json(p,signed(m))
    with pytest.raises(ValueError,match='TRAIN'):sources(cloned,oracle)
    p.write_text(p.read_text().replace('FG_SINGLE3_REPLAY_V1','FG_REPLAY_R0_R2_V1'))
    with pytest.raises(ValueError,match='integrity'):sources(cloned,oracle)


@pytest.mark.parametrize('field',['stepid','ordinal','action','legal_add','incoming_action'])
def test_alignment_mutations_fail_closed(corpus,monkeypatch,field):
    _,_,trace,manifest=corpus
    rows=list(iter_sidecar(trace));rows=copy.deepcopy(rows)
    i=1 if field=='incoming_action' else 0
    rows[i][field]={'stepid':'00'*20,'ordinal':9,'action':0,'legal_add':False,'incoming_action':0}[field]
    monkeypatch.setattr('gym_trading_env.research.fg_s3a_labels.iter_sidecar',lambda _:iter(rows))
    with pytest.raises(ValueError,match='mismatch|shifted'):
        list(aligned_rows(trace,manifest,{0,1022}))


def test_exclusion_and_label_cache_integrity(corpus,tmp_path):
    dataset,oracle,trace,manifest=corpus
    row=next(iter_sidecar(trace));step=next(iter_replay_rows(trace,manifest))[1]
    for changed,reason in [(dict(decision_executed=False),'unexecuted'),
        (dict(direction='NO_TRADE'),'no_trade'),(dict(lots_after=3),'held_forced_wait'),
        (dict(is_last=True,status='CHUNK_END'),'administrative_last')]:
        assert exclusion(dict(row,**changed),step)==reason
    assert exclusion(row,dict(step,action_mask=np.array([1,0],np.float32)))=='restricted_wait'
    result=build_labels(dataset,oracle,tmp_path/'labels')
    p=tmp_path/'labels'/result['traces'][0]['labels_file'];a=np.load(p);a['label'][0]=0;np.save(p,a)
    with pytest.raises(ValueError,match='changed'):load_labels(tmp_path/'labels')

"""Small real native packages, exact index joins, and calibration isolation."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from test_fg_single3_env import single_env
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.research.fg_replay import (SINGLE3_SCHEMA, collect_trace,digest,
    read_manifest,reconcile_trace,signed,write_json)
from gym_trading_env.research.fg_single3_oracle import CandidateIndex,optimize
from gym_trading_env.research.fg_single3_replay import Single3BehaviorPolicy
from gym_trading_env.research.fg_s3a_labels import build_labels
from gym_trading_env.research.fg_s3b_advantage import build_values,audit_values
from gym_trading_env.research.fg_s3b_weights import (
    audit_weights,build_weight_index,fit_weight_scale,scaled_weights,weight_statistics)

DREAMER = '/home/v/Documents/work/dreamerv3-dev'


def certified_native_dataset(root):
    """Synthetic TRAIN fixture, using ORIGINAL Env and Oracle for its receipt."""
    envdir = root/'env'; envdir.mkdir(parents=True)
    prices = [(1000,1000,1000,1000),(1000,1008,965,1007),
              (1000,1007,1000,1007),(1000,1007,1000,1007),
              (1000,1000,989,1000),(1000,1008,1000,1007),
              (1000,1000,1000,1000)]
    prototype = single_env(envdir,prices,candidate='C1',chunk=2,fees=3)
    market = envdir/'market.csv'
    prototype._raw.to_csv(market,index_label='Date')
    cfg = yaml.safe_load((envdir/'env.yaml').read_text())
    cfg['trading']['data_path'] = str(market)
    (envdir/'env.yaml').write_text(yaml.safe_dump(cfg))
    factory = lambda: FixedDirectionEntryEnv(config_path=envdir/'env.yaml')
    env = factory(); contract = env.contract
    index = CandidateIndex(env)
    rows = [index.candidate(t) for t in range(index.flat_end)]
    oracle = root/'oracle'; od = oracle/'contracts'/contract; od.mkdir(parents=True)
    with gzip.open(od/'candidates.jsonl.gz','wt') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
    identity = dict(schema='FG_SINGLE3_ORACLE_V1',training_hash='synthetic_TRAIN',
                    selected_contracts=[contract],objective='native_float32_feedback_rational_gamma_env_advance_v1')
    plan = optimize(rows,index.flat_end,'C1')
    plan.update(contract=contract,training_only=True,identity=identity,
                candidates_sha256=digest(od/'candidates.jsonl.gz'),funds_certificate=index.proof)
    pp = od/'C1_H333.json'; write_json(pp,signed(plan))
    write_json(od/'manifest.json',signed(dict(contract=contract,status='PASS',identity=identity,
        flat_end_cursor=index.flat_end,funds_certificate=index.proof)))
    write_json(oracle/'manifest.json',signed(dict(stage='O',status='PASS',full_training_fold=True,
        oracle_optimality_status='exact_proven',identity=identity,
        files={str(p.relative_to(oracle)):digest(p) for p in od.iterdir()})))
    dataset = root/'dataset'; trace = dataset/'traces/C1/P4'/contract
    spec = dict(id='P4',kind='oracle_schedule',candidate='C1',seed_components=[0],
                schedule_path=str(pp),schedule_sha256=digest(pp))
    t = collect_trace(envdir/'env.yaml',spec,trace,identity,DREAMER,
        env_factory=factory,protocol='fg_single3_v1',policy_factory=Single3BehaviorPolicy)
    receipt = reconcile_trace(trace,DREAMER,env_factory=factory,schema=SINGLE3_SCHEMA)
    write_json(trace/'reconciliation.json',signed(dict(receipt,trace_manifest_sha256=digest(trace/'manifest.json'))))
    row = dict(contract=contract,role='train',eligible_for_training=True,complete_lifecycle=True,
        path=str(market),sha256=digest(market),env_config=str(envdir/'env.yaml'),env_sha256=digest(envdir/'env.yaml'),
        contract_config=str(envdir/'contract.yaml'),contract_sha256=digest(envdir/'contract.yaml'))
    train_path = root/'train.json'
    write_json(train_path,signed(dict(reward_candidate='C1',single3_protocol='fg_single3_v1',
        original_content_hash='synthetic_TRAIN',direction_path=str(envdir/'directions.yaml'),
        direction_sha256=digest(envdir/'directions.yaml'),files=[row])))
    did = dict(schema=SINGLE3_SCHEMA,fold='TRAIN',conditional_direction_only=True,
               selected_contracts=[contract],original_training_hash='synthetic_TRAIN')
    write_json(dataset/'inputs.json',signed(dict(identity=did,training_views={'C1':dict(
        path=str(train_path),content_hash=read_manifest(train_path)['content_hash'],rows=[row])})))
    rec = dict(candidate='C1',policy_id='P4',contract=contract,directory=str(trace.relative_to(dataset)),
        manifest_sha256=digest(trace/'manifest.json'),receipt_sha256=digest(trace/'reconciliation.json'),
        trace_identity=t['identity'],records=t['records'],replay_trace_hash=t['replay_trace_hash'],
        reward_params_hash=t['reward_params_hash'],observation_identity=t['observation_identity'])
    write_json(dataset/'manifest.json',signed(dict(schema=SINGLE3_SCHEMA,stage='D_GENERATED',status='PASS',
        full_training_fold=True,identity=did,input_manifest_sha256=digest(dataset/'inputs.json'),
        traces=[rec],pool_members={'C1_P4':[0]})))
    labels = root/'labels'; build_labels(dataset,oracle,labels)
    return dataset,oracle,labels


@pytest.fixture(scope='module')
def packages(tmp_path_factory):
    root = tmp_path_factory.mktemp('s3b_native')
    dataset,oracle,labels = certified_native_dataset(root/'native')
    values = root/'values'
    vm = build_values(dataset,oracle,labels,values)
    weights = root/'weights'
    wm = build_weight_index(values,labels,weights)
    return dict(root=root,dataset=dataset,oracle=oracle,labels=labels,values=values,weights=weights,vm=vm,wm=wm)


def test_native_exact_index_weights_ess_and_three_interfaces(packages,tmp_path):
    p = packages
    wm = audit_weights(p['weights'])
    assert wm['same_supervision_index_as_s3a'] and wm['complete']
    assert wm['summary']['classes']['ADD']['positive'] > 0
    assert wm['summary']['classes']['WAIT']['positive'] > 0
    assert wm['summary']['training_admission'] == 'INSUFFICIENT_SUPERVISION_COVERAGE'
    assert wm['interfaces']['S3A_MATCHED']['mask_sha256'] == wm['interfaces']['S3B_CERTIFIED']['mask_sha256']
    again = build_weight_index(p['values'],p['labels'],tmp_path/'weights2')
    assert again['files'] == wm['files'] and again['summary'] == wm['summary']
    with pytest.raises(FileExistsError): build_weight_index(p['values'],p['labels'],p['weights'])
    assert build_values(p['dataset'],p['oracle'],p['labels'],p['values'],resume=True) == p['vm']
    with pytest.raises(ValueError,match='resume identity'):
        build_values(p['dataset'],p['oracle'],p['labels'],p['values'],resume=True,seed=9)


def test_scale_extreme_nan_zero_and_tiny_weights():
    s = fit_weight_scale([0.,1.,2.,9.],[False,True,True,True])
    assert s['fold'] == 'TRAIN' and s['scale'] == np.percentile([1.,2.,9.],90)
    assert scaled_weights(np.array([1e-300,1.,1e300]),1e-100,3.).tolist() == [1e-200,3.,3.]
    with pytest.raises(ValueError,match='no certified'): fit_weight_scale([0.,0.],[True,True])
    with pytest.raises(ValueError,match='Invalid'): fit_weight_scale([float('inf')],[True])
    with pytest.raises(ValueError,match='Invalid'): scaled_weights([float('nan')],1.,3.)
    assert weight_statistics(np.ones(8))['ess'] == 8
    assert weight_statistics(np.zeros(8))['ess'] == 0


def test_foreign_source_hash_fold_and_weight_mutations_rejected(packages,tmp_path):
    import shutil
    p = packages
    wrong = tmp_path/'weights'; shutil.copytree(p['weights'],wrong)
    m = read_manifest(wrong/'manifest.json'); m.pop('content_hash'); m['fold'] = 'TEST'
    write_json(wrong/'manifest.json',signed(m))
    with pytest.raises(ValueError,match='identity'): audit_weights(wrong)
    wrong = tmp_path/'weights2'; shutil.copytree(p['weights'],wrong)
    shard = wrong/p['wm']['shards'][0]['directory']/'weights.npy'
    w = np.load(shard); w[0] += 1; np.save(shard,w)
    with pytest.raises(ValueError,match='changed'): audit_weights(wrong)
    wrong = tmp_path/'values'; shutil.copytree(p['values'],wrong)
    m = read_manifest(wrong/'manifest.json'); m.pop('content_hash')
    m['identity']['gamma_rational'] = [1,1]; write_json(wrong/'manifest.json',signed(m))
    with pytest.raises(ValueError,match='identity'): audit_values(wrong)
    partial = tmp_path/'partial'
    build_values(p['dataset'],p['oracle'],p['labels'],partial,limit=1)
    with pytest.raises(ValueError,match='full supervision'):
        build_weight_index(partial,p['labels'],tmp_path/'partial_weights')


def test_future_price_bitrot_invalidates_old_proofs_without_relabeling(packages):
    import csv
    import io
    p = packages
    lm = read_manifest(p['labels']/'manifest.json')
    inputs = read_manifest(p['dataset']/'inputs.json')
    market = Path(inputs['training_views']['C1']['rows'][0]['path'])
    before = market.read_bytes()
    label_hash = digest(p['labels']/'manifest.json')
    try:
        # Change a future OHLC byte in this TEMPORARY synthetic TRAIN source.
        rows = list(csv.reader(before.decode().splitlines()))
        rows[-1][rows[0].index('High')] = '1010'
        stream = io.StringIO(); csv.writer(stream).writerows(rows)
        market.write_text(stream.getvalue())
        with pytest.raises(ValueError,match='changed'): audit_weights(p['weights'])
        assert digest(p['labels']/'manifest.json') == label_hash
    finally:
        market.write_bytes(before)
    assert audit_weights(p['weights'])['same_supervision_index_as_s3a']

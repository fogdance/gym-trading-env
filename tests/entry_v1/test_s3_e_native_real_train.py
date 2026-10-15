"""Local real TRAIN regression: original Oracle, rotation and cold Replay restore."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from e_data_v1.provider import ETrainAdmission
from gym_trading_env.research.fg_replay import read_manifest,signed,write_json
from gym_trading_env.research.fg_s3_e_observation import OBS_SCHEMA
from gym_trading_env.research.fg_s3_oracle_state_query import load_train_query
from gym_trading_env.research.fg_s3_v2_collector import V2Collector
from gym_trading_env.research.fg_s3_v2_train_pool import ObservedTrainingPool,PoolOracleQuery

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT.parents[1]/'fg-s3-b2-full-train/gym-trading-env/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/train_pool.json'
COLLECTION=Path(os.environ.get('S3_E_NATIVE_COLLECTION',str(ROOT.parent/'native_env_review_20261015/e_data_native_v1')))
pytestmark=pytest.mark.skipif(not SOURCE.is_file() or not (COLLECTION/'index_manifest.json').is_file(),
                            reason='Real signed TRAIN/native collection unavailable; no synthetic substitute')


def test_real_native_pool_original_oracle_and_exact_replay_cold_resume(tmp_path):
    from embodied.run.fg_checkpoint import save_replay,load_replay,file_hash
    names=['CZCE.FG201','CZCE.FG301']
    original=read_manifest(SOURCE)
    chosen=[r for r in original['files'] if r['contract'] in names]
    assert len(chosen)==2
    pool=signed(dict(product='FG',files=chosen));write_json(tmp_path/'pool.json',pool)
    selector=tmp_path/'selector.yaml'
    selector.write_text(yaml.safe_dump(dict(strategy_mode='entry_dataset',role='train',
        manifest_path=str(tmp_path/'pool.json'),scheduler=dict(mode='least_progress',chunk_bars=1024))))
    profile=ETrainAdmission.from_train_pool(SOURCE,COLLECTION,tmp_path/'pool.json',contracts=names)
    queries={}
    for name in names:
        query,env,*_=load_train_query('/data/logdir/fg_s3_market_minute/20261013_v2/env/v1',
            '/data/logdir/fg_s3_market_minute/20261013_v2/oracle/v1',name)
        queries[name]=query;env.close()
    oracle=PoolOracleQuery(queries,pool['content_hash'])
    native=V2Collector(ObservedTrainingPool(selector,e_profile=profile),oracle,tmp_path/'native',
        ROOT.parent/'dreamerv3',seed=202,beta=1.,replay_capacity=4096,max_env_steps=2300,
        observation_schema=OBS_SCHEMA,e_profile=profile)
    legacy=restored=None
    try:
        legacy=V2Collector(ObservedTrainingPool(selector),oracle,tmp_path/'legacy',
            ROOT.parent/'dreamerv3',seed=202,beta=1.,replay_capacity=4096,max_env_steps=2300)
        native.advance(2200);legacy.advance(2200)
        assert native.env.export_state()==legacy.env.export_state()
        assert {r['contract'] for r in native.records}==set(names)
        assert native.counts['resets']>=3 and native.counts['certified_labels']>0
        assert any(r['execution_result']['status']=='CHUNK_END' for r in native.records)
        fields=('contract','cursor','state_id','next_state_id','executed_action','actor_proposal',
                'teacher_action','teacher_valid','reward','execution_result','market_advance')
        assert [{k:r[k] for k in fields} for r in native.records]==[
            {k:r[k] for k in fields} for r in legacy.records]
        for k,v in native.sample(1).items():
            assert np.isfinite(v).all(),k
        state,driver=native.save(),native.driver.save()
        replay=save_replay(native.replay,tmp_path/'replay_snapshot')
        # Randomly sampled sequences can lie wholly inside the required
        # 20-bucket warmup. Check non-padding data in the persisted Replay,
        # while the cold sample below must match exactly even during warmup.
        context_valid_rows=0
        for path in (tmp_path/'replay_snapshot').glob('*.npz'):
            with np.load(path) as chunk:
                context_valid_rows+=int((chunk['market_context_15m'][...,6].sum(-1)>0).sum())
        assert context_valid_rows>0
        restored=V2Collector(ObservedTrainingPool(selector,e_profile=profile),oracle,tmp_path/'restored',
            ROOT.parent/'dreamerv3',seed=202,beta=1.,replay_capacity=4096,max_env_steps=2300,
            observation_schema=OBS_SCHEMA,e_profile=profile)
        restored.journal.write_bytes(native.journal.read_bytes())
        restored.driver.load(copy.deepcopy(driver))
        load_replay(restored.replay,replay,tmp_path/'replay_snapshot');restored.load(copy.deepcopy(state))
        assert native.records==restored.records and native.pending==restored.pending
        native.advance(2250);restored.advance(2250)
        assert native.records==restored.records
        assert native.env.export_state()==restored.env.export_state()
        a,b=native.sample(1),restored.sample(1)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        assert native.counts['rejects']==0
        receipt=signed(dict(status='PASS',scope='REAL_TRAIN_NATIVE_ENV_DRIVER_REPLAY_NO_MODEL',
            source_pool_sha256=file_hash(SOURCE),e_profile_identity=profile.identity,
            collection_identity=profile.source_receipt['collection_identity'],
            original_oracle_identities={k:q.identity_hash for k,q in queries.items()},
            env_steps_saved=2200,env_steps_after=2250,contracts=names,
            counts=dict(native.counts),resets=native.counts['resets'],
            exact_financial_state_vs_legacy=True,exact_original_teacher_labels=True,
            exact_cold_driver_env_replay_sidecar_rng=True,exact_next_batch=True,
            replay_rows_with_valid_15m_context=context_valid_rows,
            model_optimizer='NOT_PART_OF_THIS_DATA_CONTRACT',formal_1m='NOT_RUN',test='SEALED_NOT_READ'))
        write_json(tmp_path/'native_real_replay_receipt.json',receipt)
    finally:
        native.close()
        if legacy:legacy.close()
        if restored:restored.close()

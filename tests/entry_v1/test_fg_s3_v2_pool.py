"""Existing account scheduling, teacher ownership and formal-budget boundaries."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.research.fg_s3_v2_collector import V2Collector
from gym_trading_env.research.fg_s3_v2_train_pool import ObservedTrainingPool, PoolOracleQuery
from test_fg_single3_env import single_env
from test_fg_s3_dagger_d0 import query_for


DREAMER = Path(__file__).resolve().parents[2].parent/'dreamerv3'


def make_pool(tmp_path, monkeypatch):
    paths, frames, queries = [], {}, {}
    for i in range(2):
        directory=tmp_path/str(i);directory.mkdir()
        prices=[(1000,1000,1000,1000),(1000,1000,1000,1000),(1000,1007,1000,1007)]*30
        env=single_env(directory, prices, candidate='C1', fees=3, chunk=10,
            times=pd.date_range('2026-01-05 13:31',periods=90,freq='min'),
            calendar=[dict(start='2026-01-05 13:30',end='2026-01-05 15:30')],
            liquidation='2026-01-05 15:30')
        config=directory/'env.yaml'
        raw=yaml.safe_load(config.read_text())
        contract=f'CZCE.FG270{i+1}'
        raw['entry']['contract']=contract
        raw['trading'].update(future_symbol=contract,currency_pair=f'FG270{i+1}')
        config.write_text(yaml.safe_dump(raw))
        for filename in ('contract.yaml','directions.yaml'):
            target=directory/filename;data=yaml.safe_load(target.read_text())
            if filename=='contract.yaml':data['contract']=contract
            else:
                for row in data['intervals']:row['contract']=contract
            target.write_text(yaml.safe_dump(data))
        env=FixedDirectionEntryEnv(config_path=str(config),df=env._raw.copy())
        paths.append(str(config));frames[str(config)]=env._raw.copy()
        queries[contract],_=query_for(env,config)
    monkeypatch.setattr('gym_trading_env.envs.entry_dataset_env.FixedDirectionEntryEnv',
        lambda config_path:FixedDirectionEntryEnv(config_path=config_path,df=frames[config_path]))
    manifest=dict(product='FG',files=[dict(role='train',eligible_for_training=True,rows=90,env_config=p) for p in paths])
    manifest['content_hash']=fingerprint(manifest)
    path=tmp_path/'pool.json';path.write_text(json.dumps(manifest))
    selector=tmp_path/'selector.yaml'
    selector.write_text(yaml.safe_dump(dict(strategy_mode='entry_dataset',role='train',manifest_path=str(path),
        scheduler=dict(mode='least_progress',chunk_bars=10))))
    return selector, queries


def test_pool_rotation_keeps_raw_account_and_checkpoint_owner(tmp_path,monkeypatch):
    selector, queries=make_pool(tmp_path,monkeypatch)
    oracle=PoolOracleQuery(queries,json.loads((tmp_path/'pool.json').read_text())['content_hash'])
    env=ObservedTrainingPool(selector)
    a=V2Collector(env,oracle,tmp_path/'a',DREAMER,beta=1.)
    b=None
    try:
        from gym_trading_env.research.fg_single3_evaluation import embodied_runtime
        embodied=embodied_runtime(DREAMER)
        a.replay.workers.shutdown(wait=True)
        a.replay=embodied.Replay(length=65,capacity=65,online=True,chunksize=16,seed=202,
                                directory=str(a.root/'small_replay'),save_wait=True)
        a.advance(140)
        assert a.counts['env_steps']==sum(env.advanced)==140
        assert len(env.accounts)==2 and a.counts['resets']>1
        saved=a.save();driver=a.driver.save()
        assert len(saved['records']) < saved['commit_seq']
        from embodied.run.fg_checkpoint import save_replay,load_replay
        replay=save_replay(a.replay,tmp_path/'snapshot')
        b=V2Collector(ObservedTrainingPool(selector),oracle,tmp_path/'b',DREAMER,beta=1.)
        b.replay.workers.shutdown(wait=True)
        b.replay=embodied.Replay(length=65,capacity=65,online=True,chunksize=16,seed=202,
                                directory=str(b.root/'small_replay'),save_wait=True)
        b.journal.write_bytes(a.journal.read_bytes()+b'{uncheckpointed crash tail')
        b.driver.load(driver);load_replay(b.replay,replay,tmp_path/'snapshot');b.load(saved)
        assert b.journal.read_bytes()==a.journal.read_bytes()
        assert next((b.root/'recovery_logs').iterdir()).read_bytes()==b'{uncheckpointed crash tail'
        a.advance(160);b.advance(160)
        a.prune();b.prune()
        # Existing UUID owners survive restore. Chunks first created after
        # the boundary get fresh UUIDs in each independent process.
        assert [r['stepid'] for r in a.records if r['commit_seq']<=saved['commit_seq']]==[
            r['stepid'] for r in b.records if r['commit_seq']<=saved['commit_seq']]
        assert [{k:v for k,v in r.items() if k!='stepid'} for r in a.records]==[
            {k:v for k,v in r.items() if k!='stepid'} for r in b.records]
        assert a.env.export_state()==b.env.export_state()
        x,y=a.sample(1),b.sample(1)
        for key in x:
            if key!='stepid':np.testing.assert_array_equal(x[key],y[key])
        assert x['teacher_valid'].any()
        assert all(not r['teacher_valid'] for r in a.records if not r['execution_result']['advanced'])
    finally:
        a.close()
        if b:b.close()


def test_formal_budget_needs_explicit_authorization_and_isolated_root(tmp_path):
    from gym_trading_env.research.fg_s3_v2_experiment import read_settings,require_training_authorization,PROJECT
    cfg=yaml.safe_load((PROJECT/'configs/experiments/fg_s3_v2/A2_GPU_FG201_1K.yaml').read_text())
    cfg.update(scope='formal',contracts='ALL_TRAIN',env_steps=1000000,
               run_dir='/data/logdir/fg_single3/s3_v2/unit_formal')
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    settings=read_settings(path)
    with pytest.raises(PermissionError,match='authorize-training'):
        require_training_authorization(settings,False)
    require_training_authorization(settings,True)
    cfg['scope']='diagnostic';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='bounded'):
        read_settings(path)
    cfg.update(scope='formal',run_dir='/data/logdir/fg_single3/s3/old_run')
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='isolated'):
        read_settings(path)

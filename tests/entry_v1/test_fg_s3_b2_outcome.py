"""Controlled original-Env outcome proofs; not production training evidence."""
import copy
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_fg_single3_env import single_env
from test_fg_s3_dagger_d0 import query_for
from gym_trading_env.research.fg_single3_oracle import empty_at
from gym_trading_env.research.fg_s3_oracle_state_query import OracleStateQuery
from gym_trading_env.research.fg_s3_b2_outcome_catalog import OutcomeCatalog, outcome_label
from gym_trading_env.research.fg_replay import digest, write_json

DREAMER = Path(__file__).resolve().parents[2].parent/'dreamerv3'


def fixture_catalog(env, directory):
    query, rows = query_for(env, directory/'env.yaml')
    path = directory/'controlled_candidates.json'
    write_json(path, dict(rows=rows))
    query = OracleStateQuery(env, rows, query.end, fold='TRAIN', source_identity={
        'kind': 'controlled_original_env_fixture', 'candidate_files': {str(path):digest(path)}})
    return OutcomeCatalog(query, env, rows), query


@pytest.mark.parametrize('bars,expected', [(5,0),(6,1),(15,1),(16,2),(60,2),(61,5)])
def test_tp_boundaries_use_real_inclusive_env_event(tmp_path, bars, expected):
    prices = [(1000,1000,1000,1000)] * (bars+2)
    prices[bars] = (1000,1007,1000,1007)
    env = single_env(tmp_path, prices, candidate='C1',
        times=pd.date_range('2026-01-05 09:01', periods=len(prices), freq='min'),
        calendar=[dict(start='2026-01-05 09:00', end='2026-01-05 10:15')])
    catalog, _ = fixture_catalog(env, tmp_path)
    label = catalog.labels[0]
    empty_at(env,0)
    for i in range(1,bars+1): env.step(int(i == 1))
    assert env.attributions[0]['tp_market_minutes'] == bars
    assert label['valid_event'] and label['event'] == expected
    assert label['valid_early'] and label['early10'] == 0
    assert label['observed_market_bars'] == min(60, bars)


def test_risk_first_early_exit_and_distinct_censor_masks(tmp_path):
    env = single_env(tmp_path, [(1000,1000,1000,1000), (1000,1008,965,1007),
                               (1000,1000,1000,1000)], candidate='C1')
    catalog, query = fixture_catalog(env, tmp_path)
    obs = empty_at(env,0)
    label = catalog.query_state(env, obs['action_mask'], snapshot=query.decision_state(env))
    assert query.query(env,obs['action_mask'])['action'] == 0
    assert label['valid_event'] and label['event'] == 3
    assert label['valid_early'] and label['early10'] == 1 and label['ambiguous_intrabar']
    env.step(1)
    assert env.attributions[0]['reason'] == 'STOP_LOSS'
    short = dict(eligible=True, filled=True, held_market_bars=4, exit=None,
                 first_adverse_effective_bar=2)
    assert not outcome_label(short)['valid_event'] and not outcome_label(short)['valid_early']
    short['held_market_bars'] = 5
    assert not outcome_label(short)['valid_event'] and outcome_label(short)['valid_early']
    short['held_market_bars'] = 60
    assert outcome_label(short)['event'] == 5 and outcome_label(short)['valid_event']


def test_current_account_proof_stale_mask_funds_and_no_mutation(tmp_path):
    env = single_env(tmp_path, [(1000,1000,1000,1000)]*3+[(1000,1007,1000,1007)], candidate='C1')
    catalog, query = fixture_catalog(env,tmp_path)
    obs = empty_at(env,0); before = env.export_state()
    state = query.decision_state(env)
    label = catalog.query_state(env, obs['action_mask'], snapshot=state)
    assert label['valid_event'] and env.export_state() == before
    assert not catalog.query_state(env,[1,0],snapshot=state)['valid_event']
    env.step(1)
    assert catalog.query_state(env,env._published_mask,snapshot=state)['reason'] == 'STALE_ACCOUNT_SNAPSHOT'
    assert not catalog.query_state(env,env._published_mask,snapshot=query.decision_state(env))['valid_event']
    folder = tmp_path/'losing'; folder.mkdir()
    losing = single_env(folder, [(1000,1000,1000,1000),(1000,1000,965,980),
        (1000,1000,1000,1000),(1000,1000,1000,1000)], candidate='C1', fees=3)
    other, q = fixture_catalog(losing,folder)
    empty_at(losing,0); losing.step(1)
    assert losing._calculate_equity() < Decimal(300000) and losing.net_lots == 0
    assert other.query_state(losing,losing._get_obs()['action_mask'],snapshot=q.decision_state(losing))['reason'] == 'UNPROVEN_CURRENT_FUNDS'


def test_metadata_gap_and_no_fill_do_not_fabricate_outcomes(tmp_path):
    env = single_env(tmp_path, [(1000,1000,1000,1000)]*3+[(1000,1007,1000,1007)],
        candidate='C1', zero_duration_indices=(2,))
    catalog, query = fixture_catalog(env,tmp_path)
    assert catalog.labels[0]['tp_market_minutes'] == 2
    assert not catalog.labels[1]['valid_event']  # immediate ADD cannot fill metadata.
    gap_dir = tmp_path/'gap'; gap_dir.mkdir()
    gap = single_env(gap_dir, [(1000,1000,1000,1000)]*3,
        times=pd.DatetimeIndex(['2026-01-05 09:31','2026-01-05 09:32','2026-01-05 09:34']), candidate='C1')
    other, _ = fixture_catalog(gap,gap_dir)
    assert not other.labels[0]['valid_event'] and not other.labels[0]['valid_early']


def test_atomic_outcome_join_restore_and_corruption_rejection(tmp_path):
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    from gym_trading_env.research.fg_s3_v2_collector import V2Collector
    from embodied.run.fg_checkpoint import save_replay, load_replay
    folder = tmp_path/'fixture'; folder.mkdir()
    env = single_env(folder, [(1000,1000,1000,1000),(1000,1000,1000,1000),
        (1000,1007,1000,1007)]*30, candidate='C1', fees=3, chunk=90,
        times=pd.date_range('2026-01-05 13:31',periods=90,freq='min'),
        calendar=[dict(start='2026-01-05 13:30',end='2026-01-05 15:30')],
        liquidation='2026-01-05 15:30')
    catalog, query = fixture_catalog(env,folder)
    a = V2Collector(env,query,tmp_path/'a',DREAMER,beta=1.,outcome_catalog=catalog)
    b = None
    try:
        a.advance(70)
        state, driver = a.save(), a.driver.save()
        replay = save_replay(a.replay,tmp_path/'snapshot')
        assert a.pending['stepid'] is None and a.commit_seq == 70
        raw_keys = next(iter(a.replay.chunks.values())).data
        assert not any(k.startswith(('outcome', 'teacher', 'b2/')) for k in raw_keys)
        b_env = FixedDirectionEntryEnv(df=env._raw.copy(),config_path=str(folder/'env.yaml'))
        b = V2Collector(b_env,query,tmp_path/'b',DREAMER,beta=1.,outcome_catalog=catalog)
        b.journal.write_bytes(a.journal.read_bytes())
        b.driver.load(driver); load_replay(b.replay,replay,tmp_path/'snapshot'); b.load(state)
        a.advance(75); b.advance(75)
        assert a.records == b.records and a.pending == b.pending
        x,y = a.sample(1),b.sample(1)
        for k in x: np.testing.assert_array_equal(x[k],y[k])
        assert x['outcome_event_valid'].any()
        assert not x['outcome_event_valid'][x['action_mask'][...,1] == 0].any()
        keys = ('outcome_event','outcome_event_valid','outcome_early10','outcome_early_valid',
                'teacher_action','teacher_valid')
        unjoined = {k:v for k,v in x.items() if k not in keys}
        key = x['stepid'][0,0].tobytes().hex()
        old = copy.deepcopy(b.index[key]['outcome'])
        b.index[key]['outcome']['event'] = 3
        with pytest.raises(ValueError,match='proof'): b.join(unjoined)
        b.index[key]['outcome'] = old
        bad = copy.deepcopy(unjoined); bad['action'][0,1] = 1-bad['action'][0,1]
        with pytest.raises(ValueError,match='executed action'): b.join(bad)
    finally:
        a.close()
        if b: b.close()

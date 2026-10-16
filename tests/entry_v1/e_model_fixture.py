"""Deterministic artificial E arrays over the original controlled trading Env.

These arrays do not implement E-Data normalization/aggregation or prove it.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from test_fg_single3_env import single_env
from test_fg_s3_b2_outcome import fixture_catalog
from gym_trading_env.research.fg_replay import signed, write_json
from gym_trading_env.research.fg_s3_v2_collector import V2Collector
from gym_trading_env.research.fg_s3_e_observation import OBS_SCHEMA


class SyntheticEProvider:
    def __init__(self, directory):
        self.manifest = Path(directory)/'synthetic_e_manifest.json'
        value = signed(dict(schema='FG_S3_E_MODEL_SYNTHETIC_V2', seed=71,
            scope='CPU_MODEL_CONTRACT_ONLY', provider='deterministic_fixture_v1',
            real_aggregation=False, real_price_coordinate=False))
        write_json(self.manifest, value)
        self.identity = value['content_hash']

    def snapshot(self, env, obs):
        state = env.export_state()
        cursor = state['current_step']
        shared = obs['market_seq'].copy()
        shared[:, 0] += np.float32(np.sin(cursor*.1))
        context = np.zeros((24,7),np.float32)
        count = min(24, cursor//15)
        if count:
            rng=np.random.default_rng(71+cursor)
            context[-count:,:6]=rng.normal(size=(count,6)).astype(np.float32)*.1
            context[-count:,6]=1.
        time = state['cursor_time']
        return dict(market_seq_e_shared=shared,market_context_15m=context,
            market_macd_5m=np.array([np.sin(cursor*.1),.1,.2,1.],np.float32)), dict(
            source_contract=state['contract'],visible_cutoff=time,decision_time=time,
            data_identity=self.identity,context_available_at=[time]*count,
            fixture_cursor=cursor)


def controlled_env(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    prices=[(1000,1000,1000,1000),(1000,1007,1000,1007),
            (1000,1000,1000,1000),(1000,1000,965,980),(1000,1000,1000,1000)]*18
    return single_env(folder,prices,candidate='C1',fees=3,chunk=90,
        times=pd.date_range('2026-01-05 13:31',periods=90,freq='min'),
        calendar=[dict(start='2026-01-05 13:30',end='2026-01-05 15:30')],
        liquidation='2026-01-05 15:30')


def collector_fixture(root, *, agent=None, env=None, provider=None, query=None, outcomes=None):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    env=env or controlled_env(root/'market')
    if query is None:
        outcomes,query=fixture_catalog(env,root/'market')
    provider=provider or SyntheticEProvider(root)
    dreamer=Path(__file__).resolve().parents[3]/'dreamerv3'
    collector=V2Collector(env,query,root/'collector',dreamer,beta=1.,agent=agent,
        actor=None if agent is None else lambda c,o:agent.policy(c,o,mode='train'),
        outcome_catalog=outcomes,observation_schema=OBS_SCHEMA,e_provider=provider)
    return collector,provider

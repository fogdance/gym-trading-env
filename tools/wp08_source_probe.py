"""Bounded, teacher-free source readiness check; never a strategy examination."""
import json
from pathlib import Path
import sys

from gym_trading_env.research.fg_s3_e_quality_exam_contract import PROTOCOL
from gym_trading_env.research.fg_s3_e_quality_exam_source import prepare_validation
from gym_trading_env.research.fg_replay import signed,write_json

out=Path(sys.argv[1]).absolute()
if out.exists():raise ValueError('New probe output required')
plan=dict(protocol=PROTOCOL,exam_mode='DEV_EXAM',contracts=['CZCE.FG409'],
    source=dict(kind='VALIDATION',manifest='/home/v/Documents/work/gym-trading-env/artifacts/fg_abcd_v1/v3_shared_prefix/FG409/all/manifest.json',
        env_receipt='/data/logdir/fg_s3_market_minute/20261013_v2/env/v1'),
    actors={},baselines=['ALWAYS_WAIT'],start_day='2024-04-11',end_day='2024-04-11',
    chunk_steps=256,seed=202,random_seeds=[101,202,303],output_dir=str(out),evidence_policy=None)
source=prepare_validation(plan,'CZCE.FG409',out/'source')
env=source.factory()
try:
    env.import_state(source.initial_state)
    for _ in range(32):env.step(0)
    r=signed(dict(status='DEV_SOURCE_ENGINEERING_ONLY_NOT_STRATEGY_EXAM',
        source=source.receipt,market_advances=32,teacher_used=False,training_updates=0,
        real_dev_strategy_exam='NOT_RUN',reason='FORMAL_TRAINED_F3_F4_MODEL_NOT_PROVIDED',
        cursor=env.current_step,e_snapshot=env.e_snapshot))
    write_json(out/'readiness.json',r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('source','e_snapshot')}))
finally:env.close()

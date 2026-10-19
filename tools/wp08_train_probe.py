"""One authenticated finite TRAIN account, no model or teacher or updates."""
from pathlib import Path
import json
import sys

from gym_trading_env.research.fg_s3_e_quality_exam_contract import PROTOCOL
from gym_trading_env.research.fg_s3_e_quality_exam_source import prepare_finite_train
from gym_trading_env.research.fg_s3_e_quality_exam_compare import ExamBaseline
from gym_trading_env.research.fg_s3_e_quality_exam_session import ExamSession
from gym_trading_env.research.fg_replay import write_json,signed

root=Path(sys.argv[1]).absolute()
if root.exists():raise ValueError('New finite engineering probe output required')
plan=dict(protocol=PROTOCOL,exam_mode='SYNTHETIC_DEV',contracts=['CZCE.FG809'],
    source=dict(kind='FINITE_TRAIN',manifest='/home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/train_pool.json',
        env_receipt='/data/logdir/fg_s3_market_minute/20261013_v2/env/v1'),
    actors={},baselines=['ALWAYS_WAIT'],start_day='2018-04-13',end_day='2018-04-13',chunk_steps=256,
    seed=202,random_seeds=[101,202,303],output_dir=str(root),evidence_policy=None)
source=prepare_finite_train(plan,'CZCE.FG809',root/'source')
session=ExamSession(plan,source,ExamBaseline('ALWAYS_WAIT',seed=202,contract='CZCE.FG809'),
    root/'session',policy_id='ALWAYS_WAIT')
try:
    session.advance(32);result=session.finish()
    record=signed(dict(status='FINITE_TRAIN_ENGINEERING_ONLY_NOT_DEV_EXAM',result=result,
        source=source.receipt,teacher_used=False,training_updates=0,test_prices_accessed=False))
    write_json(root/'readiness.json',record)
    print(json.dumps(dict(status=record['status'],valid_minutes=result['valid_minutes'],replay=result['reconciliation'])))
finally:session.close()

from pathlib import Path

import pandas as pd
import pytest
import yaml

from test_wp08_contract import plan


def native_trade(tmp_path,*,tp_at=None,cutoff=None,adverse_at=None):
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_replay import digest,signed
    from gym_trading_env.research.fg_s3_e_quality_exam_source import ExamSource
    from gym_trading_env.research.fg_s3_e_quality_exam_session import ExamSession
    from gym_trading_env.research.fg_s3_e_quality_exam_compare import ExamBaseline
    from gym_trading_env.research.fg_s3_e_quality_exam_metrics import analyze_trace
    factory,path,profile=native_factory(tmp_path/'source',minutes=max(90,(tp_at or cutoff or 0)+2),chunk=0)
    directions=tmp_path/'source/directions.yaml';d=yaml.safe_load(directions.read_text())
    for row in d['intervals']:row['direction']='SHORT'
    directions.write_text(yaml.safe_dump(d))
    market=tmp_path/'source/market.csv';frame=pd.read_csv(market)
    frame['Open']=1000;frame['High']=1002;frame['Low']=999;frame['Close']=1001
    if adverse_at is not None:frame.loc[adverse_at,'High']=1011
    if tp_at is not None:frame.loc[tp_at,['Low','Close']]=[994,994]
    frame.to_csv(market,index=False)
    profile.source_receipt['sources']['CZCE.FG2701']['market']['sha256']=digest(market)
    env=factory()
    try:initial=env.reset(seed=202);initial=env.export_state()
    finally:env.close()
    source=ExamSource(factory,initial,cutoff or tp_at,signed(dict(role='synthetic',scope='EXPLICIT_NATIVE_TRADE_FIXTURE',
        frozen_inputs={str(p):digest(p) for p in (market,path,directions,tmp_path/'source/contract.yaml')})))
    p=plan(tmp_path/'exam');p['start_day']=p['end_day']='2026-01-06'
    session=ExamSession(p,source,ExamBaseline('ALWAYS_ADD',seed=202,contract='CZCE.FG2701'),
        tmp_path/'exam/session',policy_id='ALWAYS_ADD')
    try:
        session.advance();session.finish()
        return analyze_trace(session.root/'trace',source,tmp_path/'metrics')
    finally:session.close()


@pytest.mark.parametrize('age,expected',[(5,[True,True,True,True]),(15,[False,True,True,True]),
    (30,[False,False,True,True]),(60,[False,False,False,True]),(61,[False,False,False,False])])
def test_native_tp_market_minute_window_boundaries(tmp_path,age,expected):
    report=native_trade(tmp_path,tp_at=age);assert report['summary']['entries']==1
    trade=report['trades'][0]
    assert trade['exit_reason']=='TAKE_PROFIT' and trade['held_market_minutes']==age
    assert [trade['tp_le_'+str(h)] for h in (5,15,30,60)]==expected
    assert all(trade['tp_le_'+str(h)+'_valid'] for h in (5,15,30,60))


def test_adverse_tp_is_not_sl_and_exit_bar_not_underwater(tmp_path):
    report=native_trade(tmp_path,tp_at=8,adverse_at=3);trade=report['trades'][0]
    assert trade['mae_full']==trade['mae_5']==trade['mae_15']==11
    assert trade['first_adverse10']==3 and trade['max_adverse_cash_observed']==660
    assert trade['first_adverse10_observed_bar_time']==trade['native_reward_adverse10_time']
    assert not trade['adverse10_le_5_then_sl30'] and trade['adverse10_le_5_then_sl30_valid']
    assert trade['underwater']==trade['longest_underwater']==7
    assert trade['underwater_full_valid'] and trade['net_pnl']==282.


def test_censored_risk_does_not_invent_negative_sl_or_tp_label(tmp_path):
    report=native_trade(tmp_path,cutoff=3,adverse_at=2);trade=report['trades'][0]
    assert trade['exit_reason']=='RIGHT_CENSORED' and trade['first_adverse10']==2
    assert not trade['mae_5_valid'] and not trade['mae_15_valid'] and not trade['mae_full_valid']
    assert not trade['adverse10_le_5_then_sl30_valid'] and not trade['tp_le_5_valid']
    assert not trade['underwater_full_valid'] and report['summary']['entries']==1
    assert report['summary']['tp_le_5_rate']==0. and report['summary']['tp_le_5_valid_count']==0
    assert report['daily'][0]['mae_15_mean'] is None
    assert report['daily'][0]['mae_15_valid_count']==0


def test_tp_crosses_official_day_without_reset_or_forced_exit(tmp_path):
    report=native_trade(tmp_path,tp_at=350);trade=report['trades'][0]
    assert trade['entry_official_day']=='2026-01-06' and trade['exit_official_day']=='2026-01-07'
    assert trade['crossed_official_day_tp'] and not trade['same_official_day_tp']
    assert trade['held_market_minutes']==350 and trade['net_pnl']==282.
    assert [d['entries'] for d in report['daily']]==[1,0]
    assert all(d['ledger_reconciled'] for d in report['daily'])
    assert sum(d['net_pnl'] for d in report['daily'])==282.

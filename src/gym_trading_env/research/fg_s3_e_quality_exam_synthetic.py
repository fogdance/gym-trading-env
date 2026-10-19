"""Fixed synthetic input files for DEV examination, not a financial simulator."""
from pathlib import Path
import json

import pandas as pd
import yaml

from .fg_s3_e_quality_exam_source import ExamSource,ExamEProfile
from .fg_s3_e_quality_exam_contract import identity
from .fg_replay import digest,signed


def native_source(folder,*,contract='CZCE.FG2701',start_day='2026-01-08',end_day='2026-01-13'):
    from dataclasses import asdict
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    from gym_trading_env.rewards.single3_reward import Single3RewardConfig,QUALITY_TP_SCHEMA
    folder=Path(folder).absolute();descriptor=folder/'source.json';path=folder/'env.yaml'
    if descriptor.exists():
        saved=json.loads(descriptor.read_text());receipt=saved['receipt']
        if (receipt['contract'],receipt['start_day'],receipt['end_day'])!=(contract,start_day,end_day):
            raise ValueError('EXAM_SYNTHETIC_EXISTING_SOURCE_SELECTION_CHANGED')
        profile=ExamEProfile(saved['profile'])
        def factory():return FixedDirectionEntryEnv(config_path=path,e_profile=profile)
        return ExamSource(factory,saved['initial_state'],saved['end_cursor'],receipt)
    if folder.exists():raise ValueError('EXAM_SYNTHETIC_SOURCE_OUTPUT_MUST_BE_NEW')
    days=['2026-01-06','2026-01-07','2026-01-08','2026-01-09','2026-01-12','2026-01-13','2026-01-14']
    if contract not in ('CZCE.FG2701','CZCE.FG2705') or start_day not in days[2:] or end_day not in days[2:] or start_day>end_day:
        raise ValueError('EXAM_BUILTIN_FIXTURE_CONTRACT_OR_DAYS_UNSUPPORTED')
    segments=[['21:00','23:00'],['09:00','10:15'],['10:30','11:30'],['13:30','15:00']]
    times=[];trading_days=[];sessions=[]
    for day in days:
        td=pd.Timestamp(day);night=td-pd.Timedelta(days=3 if td.weekday()==0 else 1)
        for a,b in segments:
            clock_day=night if a=='21:00' else td
            start=pd.Timestamp(str(clock_day.date())+' '+a,tz='Asia/Shanghai')
            end=pd.Timestamp(str(clock_day.date())+' '+b,tz='Asia/Shanghai')
            sessions.append(dict(start=str(start),end=str(end)))
            bars=list(pd.date_range(start+pd.Timedelta(minutes=1),end,freq='min'))
            times.extend(bars);trading_days.extend([day]*len(bars))
    prices=[(1000,1035 if i%17==3 else 1002,990 if i%13==4 and i%17!=3 else 999,1000) for i in range(len(times))]
    last=max(i for i,d in enumerate(trading_days) if d==end_day)
    for i in range(last-8,last+1):prices[i]=(1000,1001,999,1000)
    frame=pd.DataFrame(prices,index=pd.DatetimeIndex(times),columns=['Open','High','Low','Close'])
    frame['Volume']=100;frame['OpenInterest']=1000;frame['bob']=frame.index-pd.Timedelta(minutes=1)
    frame['trading_date']=trading_days;frame['symbol']=contract
    folder.mkdir(parents=True);market=folder/'market.csv';frame.to_csv(market,index_label='Date')
    directions=folder/'directions.yaml';terms_path=folder/'contract.yaml'
    directions.write_text(yaml.safe_dump(dict(timezone='Asia/Shanghai',date_basis='trading_date',intervals=[
        dict(contract=contract,start='2026-01-01',end='2026-02-01',direction='SHORT')])))
    terms_path.write_text(yaml.safe_dump(dict(contract=contract,tick_size=1,lot_size=20,
        last_open_at=str(times[-1]),liquidate_at=str(times[-1]),sessions=sessions,source_sha256=digest(market),
        terms=[dict(start='2026-01-01',end='2026-02-01',margin_rate=.1,open_fee=3,close_fee=3,spread=0,slippage=0)])))
    path.write_text(yaml.safe_dump(dict(strategy_mode='entry_only',entry=dict(contract=contract,protocol='fg_single3_v1',
        direction_config=str(directions),contract_config=str(terms_path),chunk_steps=0,main_exit_at=str(times[-1]),
        reward=asdict(Single3RewardConfig(candidate='C1',beta=.5,rho=.4,L=3.,schema=QUALITY_TP_SCHEMA)),
        market_clock=dict(timezone='Asia/Shanghai',minutes_per_day=345,segments=segments)),
        trading=dict(is_future=True,future_symbol=contract,currency_pair=contract.split('.')[1],
            initial_balance=300000,trade_lot=3,max_long_position=3,max_short_position=3,
            obs_feature_mode='obs',live_mode=False,data_tz='Asia/Shanghai',data_path=str(market),stop_loss_enabled=True),
        training=dict(window_size=60),debug=dict(log_level='CRITICAL'))))
    profile=ExamEProfile(dict(protocol='FG_S3_E_EXAM_PROFILE_V1',role='synthetic',
        sources={contract:{'market':dict(path=str(market),sha256=digest(market))}}))
    def factory():return FixedDirectionEntryEnv(config_path=path,e_profile=profile)
    env=factory()
    try:
        env.reset(seed=202);first=next(i for i,d in enumerate(trading_days) if d==start_day)
        while env.current_step<first-1:env.step(0)
        initial=env.export_state()
    finally:env.close()
    frozen={str(p):digest(p) for p in (market,path,terms_path,directions)}
    receipt=signed(dict(protocol='FG_S3_E_QUALITY_EXAM_SOURCE_V1',role='synthetic',
        exam_mode='SYNTHETIC_DEV',contract=contract,scope='BUILTIN_SYNTHETIC_ONLY',
        eligible_for_training=False,training_admitted=False,profile_identity=profile.identity,
        start_day=start_day,end_day=end_day,start_cursor=first-1,end_cursor=last,
        initial_account_hash=identity(initial),frozen_inputs=frozen,test_prices_accessed=False))
    descriptor.write_text(json.dumps(dict(profile=profile.source_receipt,initial_state=initial,end_cursor=last,receipt=receipt),indent=2)+'\n')
    return ExamSource(factory,initial,last,receipt)

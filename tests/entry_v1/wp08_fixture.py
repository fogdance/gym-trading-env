"""Explicitly synthetic, native accounts and frozen models; no DEV claims."""
from pathlib import Path
import json
import os

RELEASE=Path('/home/v/Documents/work/research/fg-s3-e-quality-wp07-release-20261019/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp07')
ARTIFACTS=Path(__file__).resolve().parents[2]/'artifacts/fg_s3_v2/e_quality_wp08'


def f3_reference():
    path=Path(os.environ.get('FG_WP08_F3_SETTINGS',RELEASE/'synthetic_source_01/f3_settings.json'))
    return json.loads(path.read_text())


def controlled_actor(output,*,add=True):
    from gym_trading_env.research.fg_s3_e_quality_actor import create_agent
    from gym_trading_env.research.fg_s3_e_quality_exam_actor import ExamFrozenActor
    import jax.numpy as jnp
    agent=create_agent(f3_reference(),output,controlled_small=True)
    for k,v in agent.params.items():
        if k.startswith('quality_actor/') and k.endswith('/kernel') and v.shape[-1]==2:
            agent.params[k]=jnp.zeros_like(v)
        if k.startswith('quality_actor/') and k.endswith('/bias') and v.shape==(2,):
            agent.params[k]=jnp.array([0.,2.] if add else [2.,0.],dtype=v.dtype)
    return ExamFrozenActor(agent,dict(scope='BUILTIN_SYNTHETIC_FIXED_ACTOR_NOT_TRAINED',kind='SYNTHETIC'))


def native_source(folder,*,contract='CZCE.FG2701',start_day='2026-01-08',end_day='2026-01-13'):
    """Dated synthetic prices; native matching/reward/features are unmodified."""
    import hashlib
    import numpy as np
    import pandas as pd
    import yaml
    from test_fg_single3_env import single_env
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    from gym_trading_env.research.fg_s3_e_quality_exam_source import ExamEProfile,ExamSource
    from gym_trading_env.research.fg_replay import digest,signed
    from gym_trading_env.research.fg_s3_e_quality_exam_contract import identity
    folder=Path(folder).absolute();folder.mkdir(parents=True,exist_ok=True)
    path=folder/'env.yaml';descriptor=folder/'source.json'
    if descriptor.exists():
        saved=json.loads(descriptor.read_text());profile=ExamEProfile(saved['profile'])
        def factory():return FixedDirectionEntryEnv(config_path=path,e_profile=profile)
        return ExamSource(factory,saved['initial_state'],saved['end_cursor'],saved['receipt'])
    days=['2026-01-06','2026-01-07','2026-01-08','2026-01-09','2026-01-12','2026-01-13','2026-01-14']
    segments=[['21:00','23:00'],['09:00','10:15'],['10:30','11:30'],['13:30','15:00']]
    times=[];trading_days=[];sessions=[]
    for day in days:
        td=pd.Timestamp(day)
        night=td-pd.Timedelta(days=3 if td.weekday()==0 else 1)
        for a,b in segments:
            clock_day=night if a=='21:00' else td
            start=pd.Timestamp(str(clock_day.date())+' '+a,tz='Asia/Shanghai')
            end=pd.Timestamp(str(clock_day.date())+' '+b,tz='Asia/Shanghai')
            sessions.append(dict(start=str(start),end=str(end)))
            bars=list(pd.date_range(start+pd.Timedelta(minutes=1),end,freq='min'))
            times.extend(bars);trading_days.extend([day]*len(bars))
    prices=[(1000,1035 if i%17==3 else 1002,990 if i%13==4 and i%17!=3 else 999,1000) for i in range(len(times))]
    # A deliberately unfinished position near the requested interval end.
    last=max(i for i,d in enumerate(trading_days) if d==end_day)
    for i in range(last-8,last+1):prices[i]=(1000,1001,999,1000)
    old=single_env(folder,prices[:690],side='SHORT',candidate='C1',fees=3,chunk=0,
        times=pd.DatetimeIndex(times[:690]),calendar=sessions,liquidation=str(times[-1]),main_exit=str(times[689]),
        intervals=[dict(contract='CZCE.FG2701',start='2026-01-01',end='2026-02-01',direction='SHORT')])
    # single_env mirrors SHORT input quotes; set declared OHLC exactly once as
    # synthetic input data, never by altering execution or reward algorithms.
    frame=pd.DataFrame(prices,index=pd.DatetimeIndex(times),columns=['Open','High','Low','Close'])
    frame['Volume']=100;frame['OpenInterest']=1000;frame['bob']=frame.index-pd.Timedelta(minutes=1)
    frame['trading_date']=trading_days;frame['symbol']=contract
    market=folder/'market.csv';frame.to_csv(market,index_label='Date');old.close()
    raw=yaml.safe_load(path.read_text());raw['entry']['contract']=contract
    raw['entry']['main_exit_at']=str(times[-1])
    raw['entry']['direction_config']=str(folder/'directions.yaml')
    raw['entry']['contract_config']=str(folder/'contract.yaml')
    raw['entry']['reward']['schema']='FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1'
    raw['entry']['market_clock']=dict(timezone='Asia/Shanghai',minutes_per_day=345,segments=segments)
    raw['trading'].update(data_path=str(market),data_tz='Asia/Shanghai',future_symbol=contract)
    terms_path=folder/'contract.yaml';terms=yaml.safe_load(terms_path.read_text());terms['contract']=contract
    terms['terms'][0]['end']='2026-02-01';terms['source_sha256']=digest(market)
    terms_path.write_text(yaml.safe_dump(terms))
    directions=folder/'directions.yaml';d=yaml.safe_load(directions.read_text());d['date_basis']='trading_date'
    for row in d['intervals']:row['contract']=contract
    directions.write_text(yaml.safe_dump(d));path.write_text(yaml.safe_dump(raw))
    profile=ExamEProfile(dict(protocol='FG_S3_E_EXAM_PROFILE_V1',role='synthetic',
        sources={contract:{'market':dict(path=str(market),sha256=digest(market))}}))
    def factory():return FixedDirectionEntryEnv(config_path=path,e_profile=profile)
    env=factory()
    try:
        env.reset(seed=202)
        first=next(i for i,d in enumerate(trading_days) if d==start_day)
        while env.current_step<first-1:env.step(0)
        initial=env.export_state()
    finally:env.close()
    frozen={str(p):digest(p) for p in (market,path,terms_path,directions)}
    receipt=signed(dict(protocol='FG_S3_E_QUALITY_EXAM_SOURCE_V1',role='synthetic',
        exam_mode='SYNTHETIC_DEV',contract=contract,scope='BUILTIN_SYNTHETIC_ONLY',
        eligible_for_training=False,training_admitted=False,profile_identity=profile.identity,
        start_day=start_day,end_day=end_day,start_cursor=first-1,end_cursor=last,
        initial_account_hash=identity(initial),frozen_inputs=frozen,test_prices_accessed=False))
    descriptor.write_text(json.dumps(dict(profile=profile.source_receipt,initial_state=initial,
        end_cursor=last,receipt=receipt),indent=2)+'\n')
    return ExamSource(factory,initial,last,receipt)

"""Synthetic native CSV fixture: actual financial Env, never REAL_TRAIN."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
from test_fg_single3_env import single_env
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv

# Graded teacher thresholds of the synthetic corpus (the formal values, given
# explicitly): a TP has room 8, so a TP with mae_full 1 is GOOD (ADD); one
# with mae_full 15 and every SL30 are POOR (WAIT).
GRADES=dict(time_minutes=[15,120],mae_points=[3,5,10],room_points=[10,7.5,6],room_after_tp_minutes=60)


def native_factory(folder, minutes=210, chunk=64, shift=0):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    times=[];sessions=[];days=[]
    segments=[['21:00','23:00'],['09:00','10:15'],['10:30','11:30'],['13:30','15:00']]
    for day in ['2026-01-06','2026-01-07']:
        td=pd.Timestamp(day)
        for a,b in segments:
            date=td-pd.Timedelta(days=1) if a=='21:00' else td
            start=pd.Timestamp(str(date.date())+' '+a,tz='Asia/Shanghai')
            end=pd.Timestamp(str(date.date())+' '+b,tz='Asia/Shanghai')
            sessions.append(dict(start=str(start),end=str(end)))
            new=list(pd.date_range(start+pd.Timedelta(minutes=1),end,freq='min'))
            times+=new;days += [day]*len(new)
    times=times[:minutes];days=days[:minutes]
    # `shift` moves the price pattern: another synthetic market of the same rules.
    prices=[(1000,1008 if (i+shift)%11==1 else 1002,
             965 if (i+shift)%17==3 else 985 if (i+shift)%13==4 else 999,1000)
            for i in range(minutes)]
    old=single_env(folder,prices,candidate='C1',fees=3,chunk=chunk,times=pd.DatetimeIndex(times),
        calendar=sessions,liquidation=str(times[-1]),main_exit=str(times[-1]))
    frame=old._raw.copy();frame['trading_date']=days;frame['symbol']='CZCE.FG2701'
    market=folder/'market.csv';frame.to_csv(market,index_label='Date')
    path=folder/'env.yaml';config=yaml.safe_load(path.read_text())
    config['entry'].update(market_clock=dict(timezone='Asia/Shanghai',minutes_per_day=345,segments=segments))
    config['entry']['reward']['schema']='FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1'
    config['trading'].update(data_path=str(market),data_tz='Asia/Shanghai')
    path.write_text(yaml.safe_dump(config));old.close()
    profile=SimpleNamespace(identity='c'*64,contracts={'CZCE.FG2701'})
    def factory():return FixedDirectionEntryEnv(config_path=path,e_profile=profile)
    return factory,path,profile


def ts_corpus(folder, minutes=210, workers=2, shift=0, grades=GRADES, rules=None):
    """Thinking Student teaching material of the synthetic market: (folder, source);
    `rules` are other Native TP / SL distances."""
    from gym_trading_env.research import ts_v1_corpus as C
    folder=Path(folder);_,config,_=native_factory(folder/'market',minutes=minutes,shift=shift)
    row=dict(contract='CZCE.FG2701',path=str(config.parent/'market.csv'),env_config=str(config),
             role='train',eligible_for_training=True)
    corpus=folder/'corpus'/row['contract']
    return corpus,C.produce(row,corpus,grades,workers=workers,log=lambda *_:None,rules=rules)

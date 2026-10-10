from contextlib import nullcontext
import pandas as pd
import pytest
from gym_trading_env.envs.entry_contract import read_direction_text, DirectionSchedule
from gym_trading_env.research.fg_entry_v1 import export_fg


def test_fg_split_keeps_main_lifecycles_and_rejects_mid_contract_boundaries():
    from gym_trading_env.research.fg_entry_v1 import split_main_contract_days
    frames={name:pd.DataFrame({'_day':days}) for name,days in (
        ('FG301',[20221208,20221209]),
        ('FG409',[20221212,20240819]),
        ('FG501',[20240820,20240821]))}
    config=dict(mode='main_contract_boundaries',train_end_inclusive='2022-12-09',
                validation_end_inclusive='2024-08-19')
    assert split_main_contract_days(frames,config)==dict(
        train=[20221208,20221209],validation=[20221212,20240819],test=[20240820,20240821])
    with pytest.raises(ValueError,match='cuts main-contract lifecycle'):
        split_main_contract_days(frames,dict(config,train_end_inclusive='2022-12-08'))
    with pytest.raises(ValueError,match='ascending exported trading days'):
        split_main_contract_days(frames,dict(config,validation_end_inclusive='2024-08-18'))


def test_manual_direction_uses_inclusive_trading_dates(tmp_path):
    text=tmp_path/'fg.txt'
    text.write_text('fg501,2024-01-16,2025-01-15,SHORT\n')
    schedule=DirectionSchedule(read_direction_text(text))
    # Night session wall date precedes the supplied trading_date.
    assert schedule.at('CZCE.FG501','2025-01-14 21:01', '2025-01-15')=='SHORT'
    assert schedule.at('CZCE.FG501','2025-01-15 21:01', '2025-01-16')=='NO_TRADE'
    assert schedule.at('CZCE.FG601','2025-01-15','2025-01-15')=='NO_TRADE'
    with pytest.raises(ValueError,match='trading_date'):
        schedule.at('CZCE.FG501','2025-01-15')


class Database:
    def __init__(self, rows):
        self.rows=rows
    def _connect(self):
        return nullcontext(self)
    def cursor(self):
        return nullcontext(self)
    def execute(self,sql,params):
        self.sql,self.params=sql,params
    def fetchall(self):
        if 'SELECT trading_date,symbol' in self.sql:
            return pd.DataFrame(self.rows)[['trading_date','symbol']].drop_duplicates().to_dict('records')
        return self.rows


def row(time,symbol,day):
    return dict(Date=pd.Timestamp(time),Open=100,High=101,Low=99,Close=100,
                Volume=10,OpenInterest=100,symbol=symbol,bob=pd.Timestamp(time)-pd.Timedelta(minutes=1),
                trading_date=day,source='fixture',provider='JUEJIN')


def test_export_joins_main_and_preserves_contract_runs(tmp_path):
    db=Database([row('2025-01-01 09:01','CZCE.FG501','2025-01-01'),
                 row('2025-01-02 09:01','CZCE.FG505','2025-01-02'),
                 row('2025-01-03 09:01','CZCE.FG501','2025-01-03')])
    result=export_fg(db,'2025-01-01','2026-01-01',tmp_path/'export')
    assert 'b.trading_date=m.trading_date AND b.symbol=m.symbol' in db.sql
    assert db.params[:2]==('CZCE.FG','JUEJIN')
    assert len(result['files'])==3 and result['main_only']
    assert sum(f['rows'] for f in result['files'])==3
    assert all('symbol' in pd.read_csv(f['path']) for f in result['files'])
    with pytest.raises(FileExistsError):
        export_fg(db,'2025-01-01','2026-01-01',tmp_path/'export')


def test_export_rejects_ambiguous_minute(tmp_path):
    db=Database([row('2025-01-01 09:01','CZCE.FG501','2025-01-01')]*2)
    with pytest.raises(ValueError,match='ambiguous'):
        export_fg(db,'2025-01-01','2026-01-01',tmp_path/'export')
    assert not (tmp_path/'export').exists()


def test_fg_export_truncates_late_night_before_writing_and_preserves_old_days(tmp_path, capsys):
    db=Database([row(t,'CZCE.FG901','2018-09-03') for t in (
        '2018-08-31 22:59','2018-08-31 23:00','2018-08-31 23:01','2018-08-31 23:30')])
    manifest=export_fg(db,'2018-08-31','2018-09-04',tmp_path/'export')
    assert manifest['removed_late_night_rows']==2 and manifest['minute_clock']==345
    log=capsys.readouterr().out
    assert 'contract=CZCE.FG901 trading_date=2018-09-03' in log
    assert '23:01:00' in log and '23:30:00' in log and 'rows=2' in log
    omitted=pd.read_csv(manifest['truncation_audit'])
    assert len(omitted)==2 and omitted.Date.iloc[0].endswith('23:01:00')
    exported=pd.read_csv(manifest['files'][0]['path'])
    assert len(exported)==2 and exported.Date.iloc[-1].endswith('23:00:00')
    from gym_trading_env.research.fg_entry_v1 import truncate_existing_fg_export
    replay=truncate_existing_fg_export(manifest,tmp_path/'reprocessed')
    assert replay['removed_late_night_rows']==0
    pd.testing.assert_frame_equal(exported,pd.read_csv(replay['files'][0]['path']))


def test_existing_fg_preset_retains_per_side_cost_semantics(tmp_path):
    from gym_trading_env.research.fg_entry_v1 import configured_fg_costs
    source=tmp_path/'1.yaml'
    source.write_text('CZCE_FG:\n  leverage: 10\n  lot_size: 20\n  trading_fee_per_lot: 3\n  is_round_turn: true\n  spread: 1\n')
    values=configured_fg_costs(source)
    assert values['lot_size']==20 and values['margin_rate']==.1
    assert values['open_fee']==values['close_fee']==values['close_today_fee']==3
    assert values['spread']==1 and values['slippage']==0
    assert values['cost_basis']=='user_configured_constant_not_historical_exchange_schedule'


def test_real_futures_clock_rejects_untrimmed_late_night_and_is_causal():
    from gym_trading_env.utils.market_features import build_market_features,FEATURES_MARKET_OBS_OUTPUT
    idx=pd.date_range('2017-12-21 22:59',periods=4,freq='min',tz='Asia/Shanghai')
    frame=pd.DataFrame(dict(Open=[1000]*4,High=[1000]*4,Low=[1000]*4,Close=[1000]*4,
        Volume=[10]*4,OpenInterest=[100]*4,trading_date=['2017-12-22']*4),index=idx)
    from gym_trading_env.envs.entry_contract import read_yaml
    schedule=read_yaml('configs/fg_entry_v1/fg.yaml')['market_clock']
    with pytest.raises(ValueError,match='truncate in FG export'):
        build_market_features(frame,is_future=True,real_futures_clock=True,futures_schedule=schedule)
    frame=frame.iloc[:2]
    left=build_market_features(frame,is_future=True,real_futures_clock=True,futures_schedule=schedule)
    assert left.index.equals(idx[:2]) and (left.trading_day==20171222).all()
    changed=frame.copy();changed.loc[frame.index[-1],['Open','High','Low','Close']]=2000
    right=build_market_features(changed,is_future=True,real_futures_clock=True,futures_schedule=schedule)
    pd.testing.assert_frame_equal(left[FEATURES_MARKET_OBS_OUTPUT].iloc[:-1],right[FEATURES_MARKET_OBS_OUTPUT].iloc[:-1])

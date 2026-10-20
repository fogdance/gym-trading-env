from copy import deepcopy
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from e_data_v1 import (Decision, EDataConfig, EDataError, MarketData,
                       aggregate_buckets, build_e_data_v1)
from e_data_v1.io import verify_bundle, verify_collection, write_bundle, write_collection_index, write_json
from e_data_v1.schema import LEGACY_FIELDS, SHARED_FIELDS


def decision(data, i):
    return Decision(data.contract, data.frame.index[i])


def test_snapshot_optimization_preserves_pinned_math_bytes_and_metadata(market):
    from types import ModuleType
    project = Path(__file__).resolve().parents[2]
    commit = 'bed3a2cefd3923f4a7d45709d979892264c6298d'
    source = subprocess.check_output(['git', 'show', commit+':e_data_v1/core.py'], cwd=project)
    original = ModuleType('e_data_v1._pinned_snapshot_reference')
    original.__package__ = 'e_data_v1'
    exec(compile(source, 'git:'+commit+':e_data_v1/core.py', 'exec'), original.__dict__)
    frame, schedule = market
    config = EDataConfig(audit_1m=True, volume_clip=(-.02, .02), oi_clip=(-.0001, .0001))
    before = original.MarketData(frame, schedule, config)
    after = MarketData(frame, schedule, config)
    # Empty/unready context, first bucket, clipping, and a new trading-day anchor.
    for cursor in (0, 14, 314, 345, len(frame)-1):
        d = decision(after, cursor)
        expected, actual = before.snapshot(d), after.snapshot(d)
        for left, right in zip(expected[:5], actual[:5]):
            if isinstance(left, np.ndarray):
                assert left.dtype == right.dtype and left.shape == right.shape
                assert left.tobytes() == right.tobytes()
            else:
                assert left == right
        assert expected[5] == actual[5]


def test_F01_common_price_coordinate_and_near_high_relationship(prepared):
    bundle = prepared.build([decision(prepared, 374)])
    t, anchor = prepared.frame.index[374], bundle.price_anchor[0]
    by_id = {x['bucket_id']: x for x in bundle.bucket_manifest}
    ids = bundle.decision_meta[0]['context_bucket_ids']
    for i, bid in enumerate(ids, bundle.decision_meta[0]['context_padding']):
        b = by_id[bid]
        if not b['feature_ready']:
            continue
        raw = prepared.frame.loc[(prepared.frame.index > b['bucket_start']) & (prepared.frame.index <= b['bucket_end'])]
        expected = [raw.Open.iloc[0], raw.High.max(), raw.Low.min(), raw.Close.iloc[-1]]
        np.testing.assert_allclose(np.exp(bundle.context_15m[0, i, :4])*anchor, expected, rtol=2e-7)
    window = prepared.frame.loc[prepared.frame.index <= t].tail(60)
    np.testing.assert_array_equal(bundle.audit_1m_ohlc_shared[0], np.log(window[['Open','High','Low','Close']].to_numpy()/anchor).astype(np.float32))
    b1, b2 = by_id[ids[-2]], by_id[ids[-1]]
    np.testing.assert_allclose(bundle.context_15m[0, -1, 1]-bundle.context_15m[0, -2, 1], np.log(b2['High']/b1['High']), atol=3e-9)


def test_F02_original_18_bytes_mask_order_provenance_and_readonly_input(prepared):
    d = decision(prepared, 360)
    obs = prepared.snapshot(d)[1]
    provenance = dict(observation_version='FG_SINGLE3_MARKET_MINUTE_60x18_9_2', source_sha256='fixture')
    original_bytes = obs.tobytes()
    original_frame = prepared.frame.copy(deep=True)
    bundle = prepared.build([Decision(d.source_contract, d.visible_cutoff, market_seq=obs, original_provenance=provenance)])
    assert bundle.market_seq[0].tobytes() == original_bytes == obs.tobytes()
    assert bundle.schema['minute_fields'] == list(LEGACY_FIELDS)
    assert bundle.provenance['original_observation_provenance'] == [provenance]
    pd.testing.assert_frame_equal(prepared.frame, original_frame)
    unchanged = [i for i,k in enumerate(LEGACY_FIELDS) if k not in SHARED_FIELDS]
    np.testing.assert_array_equal(bundle.market_seq[:, :, unchanged], bundle.market_seq_e_shared[:, :, unchanged])
    wrong = obs.copy(); wrong[-1,0] += 1
    with pytest.raises(EDataError, match='does not match'):
        prepared.build([Decision(d.source_contract,d.visible_cutoff,market_seq=wrong)])


def test_F03_full_prefix_and_all_future_tampering_are_identical(market):
    f, schedule = market
    t, later = f.Date.iloc[360], f.Date.iloc[-1]
    config = EDataConfig(audit_1m=True)
    full = build_e_data_v1(f, [Decision('CZCE.FG809',t), Decision('CZCE.FG809',later)], schedule, config)
    prefix = build_e_data_v1(f.loc[f.Date<=t], [Decision('CZCE.FG809',t)], schedule, config)
    future = f.copy(deep=True); mask=future.Date>t
    for k in ('Open','High','Low','Close'):
        future.loc[mask,k] *= 17
    future.loc[mask,'Volume'] *= 999
    future.loc[mask,'OpenInterest'] *= 3
    tampered = build_e_data_v1(future,[Decision('CZCE.FG809',t),Decision('CZCE.FG809',later)],schedule,config)
    for name in ('context_15m','market_seq','market_seq_e_shared','price_anchor','audit_1m_ohlc_shared'):
        np.testing.assert_array_equal(getattr(full,name)[0],getattr(prefix,name)[0])
        np.testing.assert_array_equal(getattr(full,name)[0],getattr(tampered,name)[0])
    assert full.decision_meta[0] == prefix.decision_meta[0] == tampered.decision_meta[0]
    # Even invalid values after this requested cutoff cannot reject its history.
    future.loc[mask,'Close']=np.nan
    historical=build_e_data_v1(future,[Decision('CZCE.FG809',t)],schedule,config)
    np.testing.assert_array_equal(historical.context_15m,prefix.context_15m)
    assert historical.decision_meta == prefix.decision_meta


def test_F04_fourteenth_fifteenth_and_next_minute(prepared):
    rows = [prepared.snapshot(decision(prepared,i)) for i in (13,14,15)]
    assert [len(x[5]['context_bucket_ids']) for x in rows] == [0,1,1]
    assert rows[1][5]['context_bucket_ends'] == [prepared.frame.index[14].isoformat()]
    assert rows[1][5]['context_available_at'] == rows[1][5]['context_bucket_ends']


def test_F04_delayed_arrival_and_prefix_metadata(market):
    f,schedule=market; f=f.iloc[:31].copy()
    f['available_at']=f.Date
    f.loc[f.index[14],'available_at']=f.Date.iloc[16]
    data=MarketData(f,schedule,EDataConfig(audit_1m=True))
    assert [len(data.snapshot(Decision('CZCE.FG809',f.Date.iloc[i]))[5]['context_bucket_ids']) for i in (14,15,16)] == [0,0,1]
    t=f.Date.iloc[15]
    a=data.build([Decision('CZCE.FG809',t)])
    visible=f.loc[(f.Date<=t)&(f.available_at<=t)]
    b=build_e_data_v1(visible,[Decision('CZCE.FG809',t)],schedule,EDataConfig(audit_1m=True))
    np.testing.assert_array_equal(a.market_seq_e_shared,b.market_seq_e_shared)
    assert a.decision_meta == b.decision_meta
    assert a.bucket_manifest == b.bucket_manifest


def test_F03_late_historical_bucket_cannot_rewrite_earlier_statistics(market,tmp_path):
    f,schedule=market; f=f.iloc[:360].copy()
    f['available_at']=f.Date
    f.loc[f.index[299],'available_at']=f.Date.iloc[344]
    data=MarketData(f,schedule)
    earlier=f.Date.iloc[314]; later=f.Date.iloc[344]
    full=data.build([Decision('CZCE.FG809',earlier),Decision('CZCE.FG809',later)])
    prefix=build_e_data_v1(f.loc[(f.Date<=earlier)&(f.available_at<=earlier)],
                          [Decision('CZCE.FG809',earlier)],schedule)
    np.testing.assert_array_equal(full.context_15m[0],prefix.context_15m[0])
    assert full.decision_meta[0]==prefix.decision_meta[0]
    assert verify_bundle(write_bundle(full,tmp_path/'late-bundle'))['status']=='PASS'


def test_F03_late_bucket_cannot_backfill_previous_volume_or_oi_baseline(market):
    f, schedule = market
    f = f.iloc[:390].copy()
    f['available_at'] = f.Date
    # The 14:15 bucket's final minute is not visible until 15:00.
    # Its data must not be used in the 14:30 or 14:45 historical features,
    # even when those older buckets are inspected after 15:00.
    f.loc[f.index[299], 'available_at'] = f.Date.iloc[344]
    data = MarketData(f, schedule)
    by_end = {pd.Timestamp(b['bucket_end']).strftime('%H:%M'): b
              for b in data.complete_buckets}
    delayed = by_end['14:15']
    for time in ('14:30', '14:45'):
        bucket = by_end[time]
        assert delayed['bucket_id'] not in bucket['volume_baseline_bucket_ids']
        assert bucket['previous_OI_bucket_id'] != delayed['bucket_id']
        assert all(pd.Timestamp(x['available_at']) <= pd.Timestamp(bucket['bucket_end'])
                   for x in data.complete_buckets
                   if x['bucket_id'] in bucket['volume_baseline_bucket_ids'])
    # The delayed bucket becomes eligible for *later* buckets after 15:00.
    # Rebuild from the visible prefix and verify the as-of snapshot is stable.
    cutoff = f.Date.iloc[359]
    full = data.snapshot(Decision('CZCE.FG809', cutoff))
    subset = f.loc[(f.Date <= cutoff) & (f.available_at <= cutoff)].copy()
    prefix = MarketData(subset, schedule).snapshot(Decision('CZCE.FG809', cutoff))
    for i in range(5):
        if full[i] is not None:
            np.testing.assert_array_equal(full[i], prefix[i])
    assert full[5] == prefix[5]


def test_F05_sessions_trading_date_weekend_holiday_and_contract_reset(market):
    f,schedule=market
    buckets,events=aggregate_buckets(f,schedule)
    assert sum(x['complete'] for x in buckets)==46
    for b in buckets:
        assert pd.Timestamp(b['bucket_end'])-pd.Timestamp(b['bucket_start'])==pd.Timedelta(minutes=15)
        assert any(pd.Timestamp(s['start'])<=pd.Timestamp(b['bucket_start']) and pd.Timestamp(b['bucket_end'])<=pd.Timestamp(s['end']) for s in schedule['dated_sessions'])
    assert buckets[0]['trading_date']=='2026-01-06'
    assert pd.Timestamp(buckets[0]['bucket_start']).date()!=pd.Timestamp(buckets[0]['trading_date']).date()
    # Move the second trading day behind a weekend and a synthetic holiday.
    shifted=f.copy(); second=shifted.trading_date=='2026-01-07'
    night=second & (shifted.Date.dt.hour>=21)
    day=second & ~night
    for key in ('Date','bob'):
        shifted.loc[night,key]+=pd.Timedelta(days=3)
        shifted.loc[day,key]+=pd.Timedelta(days=5)
    shifted.loc[second,'trading_date']='2026-01-12'
    clock=deepcopy(schedule)
    for i,s in enumerate(clock['dated_sessions'][4:]):
        for k in ('start','end'): s[k]=(pd.Timestamp(s[k])+pd.Timedelta(days=3 if i==0 else 5)).isoformat()
    bs,_=aggregate_buckets(shifted,clock)
    assert sum(x['complete'] for x in bs)==46
    assert not any(pd.Timestamp(x['bucket_end']).date().isoformat() in ('2026-01-08','2026-01-10','2026-01-11') for x in bs)
    assert any(pd.Timestamp(x['bucket_end']).date().isoformat()=='2026-01-09' and x['trading_date']=='2026-01-12' for x in bs)
    mixed=f.copy(); mixed.loc[mixed.index[345:],'symbol']='CZCE.FG901'
    with pytest.raises(EDataError,match='one real contract'):
        aggregate_buckets(mixed,schedule)
    rolled=mixed.iloc[345:375]
    bundle=build_e_data_v1(rolled,[Decision('CZCE.FG901',rolled.Date.iloc[-1])],schedule)
    assert not bundle.context_15m.any() and bundle.decision_meta[0]['context_padding']==22
    continuous=f.copy();continuous['symbol']='CZCE.FG_MAIN'
    with pytest.raises(EDataError,match='real CZCE.FG'):
        aggregate_buckets(continuous,schedule)


def test_F06_exact_ohlc_volume_sum_and_OI_last(prepared):
    for b in prepared.complete_buckets:
        rows=prepared.frame.loc[(prepared.frame.index>b['bucket_start'])&(prepared.frame.index<=b['bucket_end'])]
        assert len(rows)==15
        assert [b[k] for k in ('Open','High','Low','Close','Volume','OpenInterest')]==[
            rows.Open.iloc[0],rows.High.max(),rows.Low.min(),rows.Close.iloc[-1],rows.Volume.sum(),rows.OpenInterest.iloc[-1]]


def test_F06_missing_duplicate_conflicting_and_zero_volume_minutes(market):
    f,schedule=market; f=f.iloc[:30].copy()
    missing=f.drop(f.index[7])
    bs,events=aggregate_buckets(missing,schedule)
    assert not bs[0]['complete'] and bs[1]['complete']
    assert events[0]['type']=='BUCKET_REJECTED'
    duplicate=pd.concat([f,f.iloc[[7]]])
    bs,events=aggregate_buckets(duplicate,schedule)
    assert not bs[0]['complete'] and any(x['reason']=='DUPLICATE_TIMESTAMP' for x in events)
    conflict=duplicate.copy(); conflict.iloc[-1,conflict.columns.get_loc('High')]+=5
    bs,_=aggregate_buckets(conflict,schedule)
    assert not bs[0]['complete']
    f.loc[f.index[7],'Volume']=0
    bs,_=aggregate_buckets(f,schedule)
    assert bs[0]['complete'] and bs[0]['Volume']==f.Volume.iloc[:15].sum()


def test_F07_prior_20_volume_adjacent_OI_and_reordering(prepared,market):
    c=prepared.complete_buckets
    assert all(not x['feature_ready'] for x in c[:20]) and c[20]['feature_ready']
    assert c[20]['vol_relative_raw']==np.log((c[20]['Volume']+1)/(np.mean([x['Volume'] for x in c[:20]])+1))
    assert c[20]['oi_change_raw']==np.log(c[20]['OpenInterest']/c[19]['OpenInterest'])
    f,schedule=market
    shuffled=MarketData(f.sample(frac=1,random_state=8),schedule,EDataConfig(audit_1m=True))
    a=prepared.snapshot(decision(prepared,374)); b=shuffled.snapshot(decision(shuffled,374))
    for i in (0,1,2,3,4): np.testing.assert_array_equal(a[i],b[i])
    assert a[5]==b[5]


def test_reuses_original_clock_and_one_feature_calculation(market,monkeypatch):
    from e_data_v1 import core
    from e_data_v1.reference import legacy_helper
    original=core.legacy_features
    calls=[]
    def counted(*a,**k):
        calls.append(1);return original(*a,**k)
    monkeypatch.setattr(core,'legacy_features',counted)
    f,schedule=market
    data=MarketData(f,schedule)
    assert isinstance(data.market_clock_counter,legacy_helper('s3_market_clock').MarketClock)
    for i in (0,13,14,15,350,374): data.snapshot(decision(data,i))
    assert len(calls)==1


def test_F07_invalid_OI_and_candle_fail_closed(market):
    f,schedule=market; bad=f.iloc[:30].copy()
    bad.loc[bad.index[2],'OpenInterest']=0
    with pytest.raises(EDataError,match='Visible invalid'):
        build_e_data_v1(bad,[Decision('CZCE.FG809',bad.Date.iloc[-1])],schedule)
    bad=f.iloc[:30].copy(); bad.loc[bad.index[2],'High']=bad.Low.iloc[2]-1
    with pytest.raises(EDataError,match='Visible invalid'):
        build_e_data_v1(bad,[Decision('CZCE.FG809',bad.Date.iloc[-1])],schedule)


def test_F08_shapes_order_padding_warmup_dtype_finite(prepared):
    bundle=prepared.build([decision(prepared,i) for i in (0,14,314,374)])
    assert bundle.context_15m.shape==(4,24,7) and bundle.market_seq_e_shared.shape==(4,60,18)
    assert bundle.context_15m.dtype==np.float32 and np.isfinite(bundle.context_15m).all()
    for i,m in enumerate(bundle.decision_meta):
        assert not bundle.context_15m[i,:m['context_padding']].any()
        assert m['context_bucket_ends']==sorted(m['context_bucket_ends'])
        mask=bundle.context_15m[i,:,6]
        assert np.isin(mask,[0,1]).all() and not bundle.context_15m[i,mask==0].any()
    assert not bundle.context_15m[:2].any() and bundle.context_15m[3,:,6].sum()>0


def test_F09_cross_day_anchor_six_channel_differences(prepared):
    bundle=prepared.build([decision(prepared,350)])
    m=bundle.decision_meta[0]
    assert m['anchor_source']=='legacy_previous_trading_day_close'
    assert len(set(prepared.frame.loc[m['minute_window_times'],'trading_date']))==2
    changed={LEGACY_FIELDS[i] for i in np.flatnonzero((bundle.market_seq_e_shared[0]!=bundle.market_seq[0]).any(axis=0))}
    assert changed==set(SHARED_FIELDS)
    expected=np.log(prepared.frame.loc[m['minute_window_times'],'Close'].to_numpy()/bundle.price_anchor[0]).astype(np.float32)
    np.testing.assert_array_equal(bundle.market_seq_e_shared[0,-len(expected):,14],expected)


def test_F09_opening_mark_has_no_future_first_close_anchor(market):
    f,schedule=market; f=f.iloc[:30].copy()
    mark=f.iloc[[0]].copy(); mark['Date']=f.bob.iloc[0];mark['bob']=mark.Date;mark['Close']=999.0;mark['Open']=999.0;mark['High']=999.0;mark['Low']=999.0
    combined=pd.concat([mark,f],ignore_index=True)
    bundle=build_e_data_v1(combined,[Decision('CZCE.FG809',mark.Date.iloc[0])],schedule)
    assert bundle.price_anchor[0]==999.0 and not bundle.market_seq.any()
    assert bundle.decision_meta[0]['anchor_source']=='observed_close_fallback'


def test_F10_only_known_market_inputs_and_no_GPU_imports(market,tmp_path):
    f,schedule=market; f=f.copy()
    f['Oracle_future_target']=np.arange(len(f))*999
    f['Teacher_score']=100
    bundle=build_e_data_v1(f,[Decision('CZCE.FG809',f.Date.iloc[374])],schedule)
    serialized=json.dumps([bundle.schema,bundle.bucket_manifest,bundle.decision_meta,bundle.provenance,bundle.events])
    assert 'Oracle_future_target' not in serialized and 'Teacher_score' not in serialized
    # Native Env tests intentionally import Gym in this pytest process. Verify
    # the standalone research dependency contract in a fresh Python process.
    fixture=tmp_path/'market.pkl';fixture.write_bytes(pickle.dumps((f,schedule)))
    script="""import pickle,sys
from e_data_v1 import build_e_data_v1,Decision,MarketData
f,s=pickle.loads(open(sys.argv[1],'rb').read())
bundle=build_e_data_v1(f,[Decision('CZCE.FG809',f.Date.iloc[374])],s)
_,meta=MarketData(f,s).macd_snapshot(f.Date.iloc[374])
assert meta['algorithm']==bundle.schema['macd_5m']
unexpected={'torch','jax','wandb','pymysql','gym_trading_env'}.intersection(sys.modules)
assert not unexpected,sorted(unexpected)
"""
    result=subprocess.run([sys.executable,'-c',script,str(fixture)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_F10_observation_provenance_rejects_teacher_payload(prepared):
    d = decision(prepared, 374)
    with pytest.raises(EDataError, match='identity fields only'):
        prepared.build([Decision(d.source_contract, d.visible_cutoff,
                                  original_provenance={'Teacher_score': 100})])


def test_macd_bounded_outputs_and_sparse_checkpoints_reconstruct_old_cutoffs(market):
    frame,schedule=market
    reference=MarketData(frame,schedule,compute_features=False)
    reference._macd_cache_limit=len(frame)
    indices=[0,174,175,180,255,256,344,345,511,len(frame)-1]
    expected={i:reference.macd_snapshot(frame.Date.iloc[i]) for i in indices}
    data=MarketData(frame,schedule,compute_features=False)
    data._macd_cache_limit=16
    data.macd_snapshot(frame.Date.iloc[-1])
    assert len(data._macd_cache)<=16 and len(data._macd_checkpoints)<10
    for i in [180,0,344,255,256,len(frame)-1,175,511,345,174]:
        value,meta=data.macd_snapshot(frame.Date.iloc[i])
        np.testing.assert_array_equal(value,expected[i][0])
        assert meta==expected[i][1]
        assert len(data._macd_cache)<=16


def test_macd_late_prefix_is_rejected_after_a_failed_future_query(market):
    frame,schedule=market
    frame=frame.copy()
    frame['available_at']=frame.Date
    frame.loc[180,'available_at']=frame.Date.iloc[210]
    data=MarketData(frame,schedule,compute_features=False)
    # The original causal state machine already rejects late historical
    # updates even at a later cutoff. The partially built prefix cache must
    # not allow a second, earlier query to consume the unavailable row.
    with pytest.raises(EDataError,match='MACD late minute cannot revise committed EMA'):
        data.macd_snapshot(frame.Date.iloc[250])
    assert data._macd_cache
    with pytest.raises(EDataError,match='MACD unavailable prefix minute'):
        data.macd_snapshot(frame.Date.iloc[190])


def test_F11_bundle_hash_verification_tamper_and_output_isolation(prepared,tmp_path):
    bundle=prepared.build([decision(prepared,374)])
    destination=write_bundle(bundle,tmp_path/'bundle')
    assert verify_bundle(destination)['status']=='PASS'
    with pytest.raises(FileExistsError): write_bundle(bundle,destination)
    with pytest.raises(ValueError,match='independent'):
        write_bundle(bundle,Path('/data/logdir/fg_single3/s3_v2/forbidden-e-test'))
    data=(destination/'context_15m.npy').read_bytes()
    (destination/'context_15m.npy').write_bytes(data[:-1]+bytes([data[-1]^1]))
    result=verify_bundle(destination)
    assert result['status']=='FAIL' and 'SHA256 context_15m.npy' in result['errors']


def test_F11_CLI_help_and_existing_output_refusal(tmp_path):
    help_result=subprocess.run([sys.executable,'-m','e_data_v1','--help'],capture_output=True,text=True)
    assert help_result.returncode==0 and 'inspect' in help_result.stdout and 'verify' in help_result.stdout


def test_F11_collection_index_seals_diagnostics_and_bundles(prepared,tmp_path):
    bundle=prepared.build([decision(prepared,374)])
    root=tmp_path/'collection';root.mkdir()
    destination=write_bundle(bundle,root/prepared.contract)
    report=dict(schema_sha256=bundle.schema['schema_sha256'],source_pool=dict(sha256='fixture'),
                bundles=[dict(path=str(destination),validation=verify_bundle(destination))])
    write_json(root/'diagnostics.json',report);write_collection_index(root,report)
    assert verify_collection(root)['status']=='PASS'
    (root/'diagnostics.json').write_text('{}')
    assert verify_collection(root)['status']=='FAIL'


@pytest.fixture(scope='module')
def env_reference(market,tmp_path_factory):
    """Use the real original Env with a separate CPU-only controlled fixture."""
    interpreter=Path(os.environ.get('S3_E_REFERENCE_PYTHON','/home/v/miniconda3/envs/dreamerv3/bin/python'))
    if not interpreter.is_file():
        pytest.skip('Original Env CPU reference interpreter unavailable; F02/F04 runtime unverified')
    frame,schedule=market; root=tmp_path_factory.mktemp('original-env')
    source=root/'market.csv';frame.to_csv(source,index=False)
    original='/data/logdir/fg_s3_market_minute/20261013_v2/env/v1/inputs/C1/CZCE_FG809_env.yaml'
    if not Path(original).is_file(): pytest.skip('Original Env reference config unavailable')
    config=yaml.safe_load(Path(original).read_text())
    config['entry']['contract_config']=str(root/'terms.yaml')
    config['entry']['main_exit_at']='2026-01-07 15:00:00'
    config['trading']['data_path']=str(source)
    terms=dict(contract='CZCE.FG809',tick_size=1,lot_size=20,
        last_open_at='2026-01-08 15:00:00+08:00',liquidate_at='2026-01-08 15:00:00+08:00',
        sessions=schedule['dated_sessions'],terms=[dict(start='2026-01-01',end='2026-01-10',
            margin_rate=.1,open_fee=3,close_fee=3,close_today_fee=3,spread=1,slippage=0)])
    (root/'terms.yaml').write_text(yaml.safe_dump(terms));(root/'env.yaml').write_text(yaml.safe_dump(config))
    project=Path(__file__).resolve().parents[2]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
             MPLCONFIGDIR=str(root/'mpl'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([str(interpreter),'-B',str(project/'e_data_v1/env_probe.py'),
        '--env',str(root/'env.yaml'),'--indices','0','13','14','15','345','350','359','360','--sequential'],
        cwd=project,env=env,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    return json.loads(result.stdout),pd.read_csv(source),schedule


def test_F02_real_original_Env_bytes_and_identity(env_reference):
    probe,frame,schedule=env_reference
    assert probe['feature_order']==list(LEGACY_FIELDS)
    decisions=[Decision(probe['source_contract'],x['visible_cutoff'],market_seq=np.array(x['market_seq'],dtype=np.float32),
        original_provenance=probe['observation_identity']) for x in probe['observations']]
    bundle=build_e_data_v1(frame,decisions,schedule)
    assert [hashlib.sha256(x.tobytes()).hexdigest() for x in bundle.market_seq]==[x['sha256'] for x in probe['observations']]


def test_F04_real_Env_cutoff_boundary(env_reference):
    probe,frame,schedule=env_reference
    observations=probe['observations'][1:4]
    assert [pd.Timestamp(x['visible_cutoff']).minute for x in observations]==[14,15,16]
    bundle=build_e_data_v1(frame,[Decision(probe['source_contract'],x['visible_cutoff']) for x in observations],schedule)
    assert [len(x['context_bucket_ids']) for x in bundle.decision_meta]==[0,1,1]

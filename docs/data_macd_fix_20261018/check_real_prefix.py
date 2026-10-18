"""Read-only FG809 TRAIN diagnostics; no Corpus, model or optimizer is created."""
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from e_data_v1.core import MarketData
from gym_trading_env.research.fg_s3_e_quality_corpus_cli import source_inputs


def main():
    began = time.monotonic()
    root = Path('artifacts/fg_s3_v2/data_macd_blocker_20261018')
    output = root/'real_prefix_final.json'
    if output.exists():
        raise SystemExit('Diagnostic receipt already exists; preserve it.')
    pool = Path('/home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env/'
                'artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/train_pool.json')
    lock, source, market = source_inputs(pool,
        Path('/data/logdir/fg_s3_market_minute/20261013_v2/env/v1'), 'CZCE.FG809',
        start_day='2018-08-08', end_day='2018-08-08')
    # Scan the real main-qualification history through its authentic final row.
    last, meta = market.macd_snapshot(market.frame.index[-1])
    centers = [7816, 8009]
    selected = sorted({i for center in centers for i in range(center-20, center+21)})
    cutoff = selected[-1]
    changed = market.frame.copy()
    changed.iloc[cutoff+1:, changed.columns.get_indexer(['Open','High','Low','Close'])] *= 3
    changed['available_at'] = market.frame.available_at
    changed.iloc[cutoff+1:, changed.columns.get_loc('available_at')] += pd.Timedelta(minutes=5)
    future = MarketData(changed, source.schedule, compute_features=False)
    prefix = MarketData(market.frame.iloc[:cutoff+1], source.schedule, compute_features=False)
    rows = []
    for i in selected:
        t = market.frame.index[i]
        value, details = market.macd_snapshot(t)
        expected = value.tobytes()
        encoded = json.dumps(details, sort_keys=True).encode()
        for view in (market, future, prefix):
            actual, actual_meta = view.macd_snapshot(t)
            assert actual.tobytes() == expected
            assert json.dumps(actual_meta, sort_keys=True).encode() == encoded
        state = details['state']['state']
        rows.append(dict(cursor=i,time=t.isoformat(),value=value.tolist(),
            anchor_close=details['anchor_close'],completed=details['completed_buckets'],
            slot=state['slot'],discarded_partial_buckets=state['discarded_partial_buckets'],
            observed_interval_end_ns=state['observed_interval_end_ns'],
            metadata_sha256=hashlib.sha256(encoded).hexdigest()))
    report = dict(status='PASS_REAL_FG809_MAIN_PERIOD',contract=source.contract,
        source_sha256=lock['row']['sha256'],source_lock_hash=lock['content_hash'],
        schema=meta['algorithm']['schema'],scanned_rows=len(market.frame),
        valid_market_minutes=int(market.market_clock_counter.valid.sum()),
        full_prefix_start=market.frame.index[0].isoformat(),
        full_prefix_end=market.frame.index[-1].isoformat(),
        completed_buckets=meta['completed_buckets'],
        discarded_partial_buckets=meta['state']['state']['discarded_partial_buckets'],
        first_failure_centers=centers,snapshots=rows,
        same_prefix_and_future_price_arrival_invariance='BYTE_IDENTICAL',
        repeated_queries='BYTE_IDENTICAL',seconds=time.monotonic()-began,
        formal_corpus='NOT_RUN',optimizer_updates=0)
    with output.open('x') as stream:
        json.dump(report,stream,indent=2); stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='snapshots'}))


if __name__ == '__main__':
    main()

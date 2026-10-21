"""Read raw FG snapshots before normalization; never modify the trading environment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def digest(path):
    # Corpus sidecars can be several GiB. Incremental SHA256 hashes exactly
    # the same bytes without allocating a whole file beside the live model.
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def inspect_frame(frame, contract, schedule):
    """Internal gaps are missing records, not proof of a market interruption.

    Only count holes between observed closes within each scheduled segment.
    Whole absent sessions and edge minutes require an independent calendar.
    """
    required = {'Date', 'bob', 'trading_date', 'symbol', 'provider', 'source',
                'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest'}
    if not required.issubset(frame):
        raise ValueError(f'Missing raw columns: {sorted(required - set(frame))}')
    times = pd.to_datetime(frame.Date)
    starts = pd.to_datetime(frame.bob)
    seconds = (times - starts).dt.total_seconds()
    numeric = frame[['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']].to_numpy(float)
    o, h, l, c = (numeric[:, i] for i in range(4))
    bad_ohlc = (l > np.minimum(o, c)) | (h < np.maximum(o, c)) | (h < l)
    minute = times.dt.hour * 60 + times.dt.minute
    segment = np.full(len(frame), -1)
    for i, (begin, end) in enumerate(schedule['segments']):
        a = int(begin[:2]) * 60 + int(begin[3:])
        b = int(end[:2]) * 60 + int(end[3:])
        segment[(minute > a) & (minute <= b)] = i
    groups = pd.DataFrame({'day': frame.trading_date.astype(str), 'segment': segment,
                           'minute': minute, 'valid': seconds.eq(60)})
    gaps = []
    for (day, seg), part in groups[groups.valid & (groups.segment >= 0)].groupby(['day', 'segment']):
        values = np.sort(part.minute.unique())
        missing = sorted(set(range(int(values[0]), int(values[-1]) + 1)) - set(values))
        if missing:
            gaps.append({'contract': contract, 'trading_date': day, 'segment': int(seg),
                         'missing_minutes': missing, 'count': len(missing)})
    counts = frame.loc[seconds.eq(60)].groupby('trading_date').size()
    return {
        'contract': contract, 'delivery_month': contract[-2:], 'rows': len(frame),
        'first_timestamp': str(times.iloc[0]), 'last_timestamp': str(times.iloc[-1]),
        'first_trading_date': str(frame.trading_date.iloc[0]),
        'last_trading_date': str(frame.trading_date.iloc[-1]),
        'trading_days': int(frame.trading_date.nunique()),
        'standard_minutes': int(seconds.eq(60).sum()), 'zero_duration': int(seconds.eq(0).sum()),
        'invalid_duration': int((~seconds.isin([0, 60])).sum()),
        'duplicate_timestamps': int(times.duplicated().sum()),
        'ascending': bool(times.is_monotonic_increasing),
        'bad_ohlc': int(bad_ohlc.sum()), 'nonfinite_values': int((~np.isfinite(numeric)).sum()),
        'nonpositive_prices': int((numeric[:, :4] <= 0).sum()),
        'negative_volume_oi': int((numeric[:, 4:] < 0).sum()),
        'zero_volume': int((numeric[:, 4] == 0).sum()),
        'zero_oi': int((numeric[:, 5] == 0).sum()),
        'identity_mismatch': int(frame.symbol.ne(contract).sum()),
        'provider_mismatch': int(frame.provider.ne('JUEJIN').sum()),
        'outside_345_clock': int((segment < 0).sum()),
        'day_bar_histogram': {str(k): int(v) for k, v in counts.value_counts().sort_index().items()},
        'internal_missing_minutes': sum(g['count'] for g in gaps),
    }, gaps


def audit_snapshot(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    master = yaml.safe_load(Path(config['master_config']).read_text())
    path = Path(config['export_manifest'])
    manifest = json.loads(path.read_text())
    if not manifest.get('main_only') or manifest.get('minute_clock') != 345:
        raise ValueError('Study requires the current main-only 345-minute export')
    mapping = pd.read_csv(manifest['mapping_path'], dtype=str)
    if digest(manifest['mapping_path']) != manifest['mapping_sha256']:
        raise ValueError('Mapping hash mismatch')
    if mapping.trading_date.duplicated().any():
        raise ValueError('Ambiguous daily main mapping')
    main = mapping.set_index('trading_date').symbol
    records, gaps, seen_days = [], [], set()
    for item in manifest['files']:
        if digest(item['path']) != item['sha256']:
            raise ValueError(f"Snapshot hash mismatch: {item['path']}")
        frame = pd.read_csv(item['path'])
        record, holes = inspect_frame(frame, item['contract'], master['market_clock'])
        record['mapping_mismatch'] = int(frame.trading_date.astype(str).map(main).ne(frame.symbol).sum())
        record['manifest_rows_match'] = len(frame) == item['rows']
        record['next_main_contract'] = item.get('next_main_contract')
        record['main_exit_at'] = item.get('main_exit_at')
        records.append(record)
        gaps.extend(holes)
        seen_days.update(frame.trading_date.astype(str))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 1, 'manifest_sha256': digest(path), 'config_sha256': digest(config_path),
              'master_config_sha256': digest(config['master_config']),
              'rows': sum(r['rows'] for r in records), 'contracts': len(records),
              'trading_days': len(seen_days), 'mapping_days_without_bars': sorted(set(main.index) - seen_days),
              'calendar_independently_verified': False,
              'gap_interpretation': 'Absent records; cause not established. Edges/absent sessions not counted.',
              'blind_start': config['blind_start'], 'blind_end_inclusive': config['blind_end_inclusive'],
              'contracts_detail': records, 'internal_gaps': gaps}
    (output / 'coverage.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    pd.DataFrame(records).to_csv(output / 'coverage.csv', index=False)
    pd.DataFrame(gaps).to_csv(output / 'internal_gaps.csv', index=False)
    return report


def audit_features(config_path):
    """Check real prefixes and altered future suffixes without event labels."""
    from gym_trading_env.utils.market_features import build_market_features, FEATURES_MARKET_OBS_OUTPUT

    config = yaml.safe_load(Path(config_path).read_text())
    master = yaml.safe_load(Path(config['master_config']).read_text())
    manifest = json.loads(Path(config['export_manifest']).read_text())
    checks = []
    for index in [0, 5, 6, 12, 17, 18]:
        item = manifest['files'][index]
        if digest(item['path']) != item['sha256']:
            raise ValueError('Snapshot hash mismatch')
        frame = pd.read_csv(item['path'], nrows=720)
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.Date)).tz_localize(master['market_clock']['timezone'])

        def features(data):
            return build_market_features(data, is_future=True, real_futures_clock=True,
                                         futures_schedule=master['market_clock'])[FEATURES_MARKET_OBS_OUTPUT]

        full = features(frame)
        expected_volume = np.tanh(np.log1p(np.clip(frame.Volume.to_numpy(float), 0, None)) / 5)
        np.testing.assert_allclose(full.obs_V_t, expected_volume, rtol=0, atol=1e-12)
        for stop in [1, 2, 60, 120, 225, 345, 720]:
            prefix = features(frame.iloc[:stop])
            np.testing.assert_allclose(prefix, full.iloc[:stop], rtol=0, atol=1e-12)
            checks.append({'contract': item['contract'], 'kind': 'prefix', 'rows': stop,
                           'max_abs_diff': float(np.max(np.abs(prefix.to_numpy() - full.iloc[:stop].to_numpy())))})
        changed = frame.copy()
        changed.iloc[60:, changed.columns.get_indexer(['Open', 'High', 'Low', 'Close'])] *= 2
        changed.iloc[60:, changed.columns.get_indexer(['Volume', 'OpenInterest'])] += 1000
        np.testing.assert_allclose(features(changed).iloc[:60], full.iloc[:60], rtol=0, atol=1e-12)
        checks.append({'contract': item['contract'], 'kind': 'future_suffix_mutation', 'rows': 60,
                       'max_abs_diff': 0.0})
    result = {'schema': 1, 'fields': FEATURES_MARKET_OBS_OUTPUT, 'checks': checks,
              'scope': 'Six real contract samples, first 720 rows; not proof for malformed inputs.',
              'config_sha256': digest(config_path),
              'master_config_sha256': digest(config['master_config']),
              'manifest_sha256': digest(config['export_manifest'])}
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'feature_causality.json').write_text(json.dumps(result, indent=2))
    return result

"""Paper inputs of a new product from its direction file, FG-like for everything else (user 2026-11-06: "支持新品种的
工具，比如导入导出的，建议写个通用的脚本，方便后续使用"; RB: "rb你主要看方向那块，其他的和fg应该差不多，数据缺少，可以从掘金归档导入").

The direction file is the user's (schema 2, `product`, `directions.intervals` of contract / direction /
[start, end) trading dates); only its intervals are read. Everything else comes from the FG paper:
the master strategy `strategy_quality_tp_v1.yaml` (market clock, costs, reward, environment), the env
`envs/CZCE_FG501_env.yaml`, and the contract files as fg_entry_v1.prepare_main_contracts (the sessions
of the bars, the last bar the main exit, the FG master costs). The product must be an Entry env product
(gym envs.entry_contract.CONTRACT_NAMES).

  coverage  per interval: DB trading days of the contract / the product's calendar days, and the archive months
  ingest    archive -> DB (fut_bar_1m_v2) for the contracts with missing days, with the Juejin
            ingest_archive_to_db_v2 job (skips months already DONE); --dry-run prints the plan only
  build     the exports (OUT/export), and under CONFIGS/markets/<name>/ the contract files, the envs, the master
            strategy_quality_tp_v1_<name>.yaml and paper_pool_<name>.json; contracts without bars are reported
            and left out; OUT/build_report.json. --costs PRESET (a cost preset as costs_fg_real_20261105.yaml,
            under the product's key; user 2026-11-06: "paper 里rb是1点10元") sets the master's costs and the
            contracts' lot size in place of FG's; --strategy S (a paper strategy under markets/, repeatable)
            writes markets/<name>/<S stem>_<name>.yaml and copies of its rule_gate files whose costs are the preset

Usage (gym-trading-env/, PYTHONPATH=.:src):
  python -m gym_trading_env.research.product_paper coverage --directions D.yaml
  python -m gym_trading_env.research.product_paper ingest --directions D.yaml [--dry-run]
  python -m gym_trading_env.research.product_paper build --directions D.yaml --configs ../dreamerv3/configs/thinking_student --out DIR \
      [--costs markets/<name>/costs.yaml] [--strategy markets/strategy_v3_trend_trail.yaml]
"""
import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ARCHIVE = Path('/data/juejin')
MANIFEST = ARCHIVE / '_manifest' / '3a1a389f-56e9-46ce-8a04-047cbd456a44.jsonl'
JUEJIN = Path('/home/v/Documents/work/juejin/ed27929a-30a5-11f0-a3a6-366f24a5ee2d')
JUEJIN_PY = '/home/v/miniconda3/envs/juejin/bin/python'
COLUMNS = 'Date,Open,High,Low,Close,Volume,OpenInterest,symbol,bob,trading_date,source,provider'.split(',')


def load_directions(path):
    """The user's direction intervals, grouped by contract in file order (a contract may have several)."""
    from gym_trading_env.envs.entry_contract import is_entry_contract
    raw = yaml.safe_load(Path(path).read_text())
    intervals = raw['directions']['intervals']
    groups = {}
    for x in intervals:
        if not is_entry_contract(x['contract']) or x['direction'] not in ('LONG', 'SHORT'):
            raise ValueError(f'{path}: {x} is not an Entry env contract with LONG / SHORT '
                             f'(add the product to entry_contract.CONTRACT_NAMES)')
        if pd.Timestamp(x['start']) >= pd.Timestamp(x['end']):
            raise ValueError(f'{path}: {x} has start >= end')
        groups.setdefault(x['contract'], []).append(x)
    return raw, groups


def _days(x):
    return x['start'][:10], x['end'][:10]


def _conn():
    from gym_trading_env.utils.bar_source import connect_juejin_db
    return connect_juejin_db(connect_timeout=5, read_timeout=600, write_timeout=5)


def coverage(groups, archive=ARCHIVE):
    """Per interval: the contract's DB trading days, the product calendar's days (fut_continuous_map_v2),
    the missing days and the archive months of the contract."""
    out = {}
    with _conn().cursor() as c:
        for contract, ivs in groups.items():
            csymbol = re.sub(r'\d+$', '', contract)
            months = sorted(f'{p.parent.name[5:]}-{int(p.name[6:]):02d}'
                            for p in (archive / 'provider=JUEJIN' / 'freq=1m' / f'symbol={contract}').glob('year=*/month=*'))
            rows = []
            for x in ivs:
                a, b = _days(x)
                c.execute('SELECT DISTINCT trading_date d FROM fut_bar_1m_v2 WHERE symbol=%s AND trading_date>=%s '
                          'AND trading_date<%s', (contract, a, b))
                have = {str(r['d']) for r in c.fetchall()}
                c.execute('SELECT DISTINCT trading_date d FROM fut_continuous_map_v2 WHERE csymbol=%s '
                          'AND trading_date>=%s AND trading_date<%s', (csymbol, a, b))
                cal = {str(r['d']) for r in c.fetchall()}
                missing = sorted(cal - have)
                rows.append(dict(direction=x['direction'], start=a, end=b, db_days=len(have), calendar_days=len(cal),
                                 missing_days=len(missing), missing_first=missing[0] if missing else None,
                                 missing_last=missing[-1] if missing else None))
            out[contract] = dict(intervals=rows, archive_months=[months[0], months[-1]] if months else None)
    return out


def ingest(contracts, dry_run=False, archive=ARCHIVE, manifest=MANIFEST):
    """Archive -> DB for `contracts` with the Juejin job (its own env and repo); returns its exit code."""
    cmd = [JUEJIN_PY, str(JUEJIN / 'jobs' / 'v2' / 'ingest_archive_to_db_v2.py'), '--archive-dir', str(archive),
           '--manifest', str(manifest), '--symbols', ','.join(contracts)] + (['--dry-run'] if dry_run else [])
    print(' '.join(cmd), flush=True)
    return subprocess.call(cmd, cwd=JUEJIN, env={'PYTHONPATH': str(JUEJIN), 'PATH': '/usr/bin:/bin'})


def product_strategy(markets, target, name, strategy, preset):
    """A copy of the paper strategy `strategy` (relative to markets/) for the product: its costs the
    product preset, its rule_gate files copied with the same costs; all paths relative to markets/<name>/."""
    from gym_trading_env.envs.entry_contract import resolve_path
    src = (markets / strategy).resolve()
    text = yaml.safe_load(src.read_text())
    rel = lambda p: str(Path('..') / Path(p).resolve().relative_to(markets.resolve()))  # noqa: E731
    cost = Path(preset).resolve().relative_to(target.resolve())
    text['costs'] = str(cost)
    for side, path in (text.get('entry', {}).get('rule_gate') or {}).items():
        rule_src = resolve_path(src, path)
        rule = yaml.safe_load(Path(rule_src).read_text())
        rule['costs'] = str(cost)
        rule_name = f'{Path(rule_src).stem}_{name}.yaml'
        (target / rule_name).write_text(f'# {Path(rule_src).name} with the {name.upper()} cost preset {cost} '
                                        '(gym research.product_paper).\n' + yaml.safe_dump(rule, sort_keys=False, allow_unicode=True))
        text['entry']['rule_gate'][side] = rule_name
    for key in ('strategy_config',):
        if key in text:
            text[key] = rel(resolve_path(src, text[key]))
    out = target / f'{src.stem}_{name}.yaml'
    out.write_text(f'# {src.name} for {name.upper()}: costs {cost} (gym research.product_paper).\n'
                   + yaml.safe_dump(text, sort_keys=False, allow_unicode=True))
    return out


def build(directions, configs, out, name=None, costs_preset=None, strategies=()):
    """The paper inputs of the direction file's product (module docstring); returns the build report."""
    from gym_trading_env.research.entry_pipeline import file_hash
    from gym_trading_env.research.fg_entry_v1 import configured_fg_costs, trim_fg_export_minutes
    raw, groups = load_directions(directions)
    product = str(raw['product']).upper()
    name = name or product.lower()
    markets, out = Path(configs).resolve() / 'markets', Path(out).resolve()
    target = markets / name
    for d in (target / 'envs', target / 'contracts', out / 'export'):
        d.mkdir(parents=True, exist_ok=True)
    master = yaml.safe_load((markets / 'strategy_quality_tp_v1.yaml').read_text())
    fg_env = yaml.safe_load((markets / 'envs' / 'CZCE_FG501_env.yaml').read_text())
    costs = configured_fg_costs(costs_preset) if costs_preset else master['costs']
    rows, report, used = [], {}, []
    with _conn().cursor() as c:
        for k, (contract, ivs) in enumerate(groups.items()):
            parts = []
            for x in ivs:
                c.execute("""SELECT eob AS Date, open AS Open, high AS High, low AS Low, close AS Close,
                    volume AS Volume, position AS OpenInterest, symbol, bob, trading_date, source, provider
                    FROM fut_bar_1m_v2 WHERE symbol=%s AND trading_date >= %s AND trading_date < %s ORDER BY eob""",
                          (contract, *_days(x)))
                parts.append(pd.DataFrame(c.fetchall()))
            parts = [p for p in parts if not p.empty]
            if not parts:
                report[contract] = dict(skipped='no bars in fut_bar_1m_v2', intervals=[_days(x) for x in ivs])
                continue
            s, removed = trim_fg_export_minutes(pd.concat(parts))
            s = s[COLUMNS].sort_values('Date').reset_index(drop=True)
            for col in ('Date', 'bob'):
                s[col] = pd.to_datetime(s[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
            s['trading_date'] = pd.to_datetime(s.trading_date).dt.strftime('%Y-%m-%d')
            if s.Date.duplicated().any() or (s[['Open', 'High', 'Low', 'Close']].to_numpy() % 1).any():
                raise ValueError(f'{contract}: duplicated minutes or prices off the one-point tick')
            stem = contract.replace('.', '_')
            path = out / 'export' / f'{k:04d}_{stem}_main_1m.csv'
            s.to_csv(path, index=False)
            sha = file_hash(path)
            times = pd.DatetimeIndex(pd.to_datetime(s.Date)).tz_localize('Asia/Shanghai')
            runs = np.r_[True, np.diff(times.asi8) != 60 * 10 ** 9].cumsum()
            sessions = [dict(start=str(times[runs == r][0] - pd.Timedelta(minutes=1)), end=str(times[runs == r][-1]))
                        for r in np.unique(runs)]
            terms = {key: costs[key] for key in ('margin_rate', 'open_fee', 'close_fee', 'close_today_fee', 'spread',
                                                 'slippage')}
            terms.update(start=str(times[0] - pd.Timedelta(minutes=1)), end=str(times[-1] + pd.Timedelta(minutes=1)))
            meta = dict(contract=contract, tick_size=1, lot_size=costs['lot_size'], last_open_at=str(times[-1]),
                        liquidate_at=str(times[-1]), terms=[terms], sessions=sessions,
                        calendar_basis='user_confirmed_complete_market', expiry_basis='main_qualification',
                        costs_basis=costs['cost_basis'], source_sha256=sha)
            (target / 'contracts' / f'{stem}_contract.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
            env = copy.deepcopy(fg_env)
            env['entry'].update(contract=contract, contract_config=f'../contracts/{stem}_contract.yaml',
                                main_exit_at=str(times[-1].tz_localize(None)),
                                strategy_config=f'../strategy_quality_tp_v1_{name}.yaml')
            env['trading'].update(currency_pair=contract.split('.')[1], future_symbol=contract, data_path=str(path))
            (target / 'envs' / f'{stem}_env.yaml').write_text(yaml.safe_dump(env, sort_keys=False))
            rows.append(dict(contract=contract, role='paper', eligible_for_training=False, path=str(path), sha256=sha,
                             rows=len(s), first_timestamp=str(s.Date.iloc[0]), last_timestamp=str(s.Date.iloc[-1]),
                             trading_days=int(s.trading_date.nunique()), calendar_basis='user_confirmed_observed_clock',
                             env_config=f'envs/{stem}_env.yaml'))
            used += ivs
            per_day = s.groupby('trading_date').size()
            report[contract] = dict(intervals=[(x['direction'], *_days(x)) for x in ivs], rows=len(s),
                                    days=int(len(per_day)), days_below_300_bars=int((per_day < 300).sum()),
                                    removed_late_night=removed, first=rows[-1]['first_timestamp'],
                                    last=rows[-1]['last_timestamp'])
    strategy = copy.deepcopy(master)
    strategy['product'] = product
    strategy['costs'] = costs
    strategy['directions'] = dict(master['directions'], intervals=used)  # the FG date basis and timezone
    strategy['data'].update(start=rows[0]['first_timestamp'][:10], end_inclusive=rows[-1]['last_timestamp'][:10],
                            excluded_contracts=[])
    head = (f'# {product} master strategy for the paper: strategy_quality_tp_v1.yaml with product {product} and the\n'
            f'# direction intervals of {Path(directions).resolve()} (sha256 {file_hash(directions)}), the contracts\n'
            f'# with DB bars only. Built by gym research.product_paper.\n')
    (target / f'strategy_quality_tp_v1_{name}.yaml').write_text(head + yaml.safe_dump(strategy, sort_keys=False,
                                                                                         allow_unicode=True))
    pool = dict(product=product, note=(f'{product} paper on the direction file\'s intervals, each contract on its own; '
                                       'FG master costs and env; built by gym research.product_paper. '
                                       'Never training material.'),
                directions_source=dict(path=str(Path(directions).resolve()), sha256=file_hash(directions)), files=rows)
    (target / f'paper_pool_{name}.json').write_text(json.dumps(pool, indent=1))
    for s in strategies:
        report.setdefault('_strategies', []).append(str(product_strategy(markets, target, name, s, costs_preset)))
    report['_costs'] = costs
    (out / 'build_report.json').write_text(json.dumps(report, indent=1, default=str))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('coverage', 'ingest', 'build'))
    p.add_argument('--directions', required=True)
    p.add_argument('--configs', help='dreamerv3/configs/thinking_student (build)')
    p.add_argument('--out', help='export and report directory (build)')
    p.add_argument('--name', help='markets/<name> (build; default the product, lower case)')
    p.add_argument('--contracts', default='', help='ingest: these contracts (default: those with missing days)')
    p.add_argument('--costs', help='build: the product cost preset (default: the FG master costs)')
    p.add_argument('--strategy', action='append', default=[], help='build: a paper strategy under markets/ to copy with --costs')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    _, groups = load_directions(a.directions)
    if a.command == 'coverage':
        print(json.dumps(coverage(groups), indent=1))
    elif a.command == 'ingest':
        todo = [x for x in a.contracts.split(',') if x] or [
            k for k, v in coverage(groups).items() if any(r['missing_days'] for r in v['intervals'])]
        if not todo:
            print('nothing missing')
            return 0
        return ingest(todo, a.dry_run)
    else:
        if not (a.configs and a.out):
            p.error('build needs --configs and --out')
        if a.strategy and not a.costs:
            p.error('--strategy needs --costs')
        print(json.dumps(build(a.directions, a.configs, a.out, a.name, a.costs, a.strategy), indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())

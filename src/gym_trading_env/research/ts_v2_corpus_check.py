"""Teacher check of the TRAIN corpora of a V2 material (V2.2-V3: the checks/corpus_teacher_check.py copied into each
DEV folder since V2_2_DEV_20261104; a tool since 2026-11-08, user: "这个可以做成一个工具，方便后续比对").

Every corpus folder under --corpus is checked against the material (--material, a combined SHORT + LONG one), the
corpora's strategy (--strategy) and the product profile (--profile); what it expects is read from them, not written
here:
- labels: the ADDs are exactly the material's positives (TP facts, exit = the material's), the WAITs exactly the
  WAIT_IN_RULE rows of kept touches, nothing else labelled or dropped but WAIT_OFF_RULE / ACTIVATION;
- rule gate (user 2026-11-05: "规则决定学生能在哪里交易"): every label, legal fact and behaviour ADD at a rule
  position; every ADD's stop of one --stop-source (V2.4: FIXED, fill +/- 5);
- inputs: the strategy's version, the profile's product and costs, rule gate files = the material's rules (both in
  points at the env's tick), max risk = the strategy's in points, the material's teacher version, net PnL reward,
  48 x 7 15m context, direction = the material's.

  cd gym-trading-env && PYTHONPATH=.:src JAX_PLATFORMS=cpu python -m gym_trading_env.research.ts_v2_corpus_check \\
      --corpus ROOT --material DIR --strategy markets/S.yaml --profile products/fg.yaml --out OUT.json \\
      [--stop-source FIXED] [--contracts 18] [--workers 9]

Prints the totals and exits 1 when a check failed or the corpus count is not --contracts."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from functools import partial
from pathlib import Path

import numpy as np

from gym_trading_env.envs import entry_product as P
from gym_trading_env.research import ts_v1_corpus as C
from gym_trading_env.research import ts_v2_positive as V
from gym_trading_env.rewards.single3_reward import NET_PNL_SCHEMA

COST_KEYS = ('lot_size',) + P.TERM_COSTS


def check(folder, material, strategy, profile, stop_source):
    c = folder.name
    tables = V.load_positive(material)
    msource = json.loads((material / 'source.json').read_text())
    side = {k: x['direction'] for x in msource['combined'] for k in x['contracts']}[c]
    graded, facts = C.load_arrays(folder / 'graded'), C.load_arrays(folder / 'facts')
    E = tables['positive_entries'][tables['positive_entries'].contract == c].set_index('cursor')
    T = tables['positive_timeline'][tables['positive_timeline'].contract == c]
    summary = json.loads((folder / 'graded' / 'summary.json').read_text())
    source = json.loads((folder / 'source.json').read_text())
    env = C.corpus_env(folder, market=False)
    tick = env.tick_points
    n = len(graded['valid'])
    gate = env._gate(side)['open'][:n]
    adds = np.flatnonzero(graded['valid'] & (graded['action'] == 1))
    waits_g = np.flatnonzero(graded['valid'] & (graded['action'] == 0))
    labelled = np.flatnonzero(graded['valid'])
    ok = {}
    ok['ADDs = all positives'] = adds.tolist() == sorted(E.index)
    ok['nothing dropped but WAIT_OFF_RULE / ACTIVATION'] = (
        summary['dropped']['RISK_ABOVE_MAX'] == summary['dropped']['STOP_LINE_DIFFERS'] == 0
        and summary['dropped']['WAIT_OF_DROPPED_TOUCH'] == 0)
    ok['ADD facts TP'] = bool((facts['kind'][adds] == C.KINDS.index('TP')).all())
    ok['ADD exit = material'] = bool((facts['exit_cursor'][adds] == E.loc[adds, 'label_exit_cursor'].to_numpy()).all())
    ok['ready_at = exit'] = bool((graded['ready_at'][adds] == E.loc[adds, 'label_exit_cursor'].to_numpy()).all())
    events = set(E.loc[adds, 'event_id'])
    waits = T[T.teach.isin(V.RULE_WAIT_ROWS) & T.event_id.isin(events)]
    ok['WAITs = WAIT_IN_RULE rows of kept touches'] = waits_g.tolist() == sorted(waits.cursor)
    ok['WAIT_OFF_RULE = pre-touch rows'] = summary['dropped']['WAIT_OFF_RULE'] == int(T.teach.isin(V.WAIT_ROWS).sum())
    ok['no demo elsewhere'] = int(graded['valid'].sum()) == len(adds) + len(waits)
    ok['labels at rule positions'] = bool(gate[labelled].all())
    ok['labels legal'] = bool(facts['eligible'][labelled].all())
    ok['facts eligible only at rule positions'] = not bool((facts['eligible'] & ~gate).any())
    behaviors = {}
    for k in sorted(p.name for p in (folder / 'behaviors').iterdir()):
        cursor = np.load(folder / 'behaviors' / k / 'cursor.npy')
        action = np.load(folder / 'behaviors' / k / 'action.npy')
        at = cursor[action == 1]
        behaviors[k] = dict(adds=int(len(at)), at_rule=int(gate[at].sum()))
    ok['behavior ADDs at rule positions'] = all(v['adds'] == v['at_rule'] for v in behaviors.values())
    sources = {}
    for t in adds:
        plan = env.planned_exit(int(t), Decimal(str(E.loc[t, 'label_fill_expected'])), side)
        sources[plan['stop_source']] = sources.get(plan['stop_source'], 0) + 1
    ok[f'ADD stops {stop_source}'] = set(sources) <= {stop_source}
    rule = summary.get('rule_gate') or {}
    ok['rule_gate summary'] = (rule.get('rule_positions') == int(gate.sum())
                               and rule.get('labels') == len(adds) + len(waits))
    ok['label / max risk'] = (summary['label']['source'] == 'V2_POSITIVE_TEACHER'
                              and summary['grades']['max_risk_points'] == strategy['entry']['max_risk_points'] * tick)
    ok['teacher = material'] = (summary['grades']['teacher'] == msource['version']
                                and summary['grades']['material'] == str(material))
    ok['net PnL reward'] = source['inputs']['reward_schema'] == NET_PNL_SCHEMA
    ok['strategy version'] = source['inputs']['strategy']['version'] == strategy['version']
    ok['product and tick'] = (source.get('product'), source.get('tick_points')) == (profile['product'], profile['tick_points'])
    ok['profile costs'] = {k: env.product['costs'][k] for k in COST_KEYS} == {k: profile['costs'][k] for k in COST_KEYS}
    material_rules = V.material_rules_points(material, tick)
    ok['rule gate files = material rules'] = all(
        V._json(P.in_points({k: v for k, v in V.load_config(folder / 'inputs' / f'rule_gate_{s.lower()}.yaml').items()
                             if k != 'direction'}, P.RULE_TICKS, tick)) == material_rules for s in ('SHORT', 'LONG'))
    ok['context 48x7'] = np.load(folder / 'market' / 'market_context_15m.npy', mmap_mode='r').shape[1:] == (48, 7)
    ok['direction = material'] = V.contract_direction(source['pool_row']) == side
    return c, dict(side=side, add=len(adds), wait=len(waits), dropped=summary['dropped'], rule_gate=rule,
                   behaviors=behaviors, stop_sources=sources, grades=json.dumps(summary['grades'], sort_keys=True),
                   failed=[k for k, v in ok.items() if not v])


def run(corpus, material, strategy, profile, stop_source='FIXED', workers=9):
    folders = sorted(p for p in Path(corpus).iterdir() if p.is_dir() and (p / 'source.json').exists())
    job = partial(check, material=Path(material).resolve(), strategy=C.load_strategy(strategy),
                  profile=P.load_profile(profile), stop_source=stop_source)
    with ProcessPoolExecutor(workers) as pool:
        out = dict(pool.map(job, folders))
    totals = dict(add=sum(v['add'] for v in out.values()), wait=sum(v['wait'] for v in out.values()), dropped={},
                  rule_positions=sum(v['rule_gate'].get('rule_positions', 0) for v in out.values()))
    for v in out.values():
        for k, x in v['dropped'].items():
            totals['dropped'][k] = totals['dropped'].get(k, 0) + x
    totals['label_coverage'] = round((totals['add'] + totals['wait']) / max(totals['rule_positions'], 1), 4)
    return dict(corpora=len(out), one_teacher=len({v.pop('grades') for v in out.values()}) == 1, totals=totals,
                by_side={s: dict(add=sum(v['add'] for v in out.values() if v['side'] == s),
                                 wait=sum(v['wait'] for v in out.values() if v['side'] == s)) for s in ('SHORT', 'LONG')},
                failed={c: v['failed'] for c, v in out.items() if v['failed']}, contracts=out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus', required=True, help='the corpus root (one folder per contract)')
    p.add_argument('--material', required=True, help='the combined SHORT + LONG material (--teacher of the corpora)')
    p.add_argument('--strategy', required=True, help='the corpora\'s strategy (--strategy of ts_v1_corpus)')
    p.add_argument('--profile', required=True, help='the product profile (configs products/<code>.yaml)')
    p.add_argument('--out', required=True, help='the result JSON')
    p.add_argument('--stop-source', default='FIXED', help='the stop source of every ADD (V2.4: FIXED)')
    p.add_argument('--contracts', type=int, default=18, help='the expected number of corpora')
    p.add_argument('--workers', type=int, default=9)
    a = p.parse_args(argv)
    result = run(a.corpus, a.material, a.strategy, a.profile, a.stop_source, a.workers)
    Path(a.out).write_text(json.dumps(result, indent=1))
    ok = result['corpora'] == a.contracts and result['one_teacher'] and not result['failed']
    print('teacher check', 'ok' if ok else 'FAILED',
          json.dumps({k: v for k, v in result.items() if k != 'contracts'}))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

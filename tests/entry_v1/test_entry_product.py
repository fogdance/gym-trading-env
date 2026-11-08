"""Product profiles (user 2026-11-08: "后续品种可能增加，这些你要在这次看下怎么设计方案 ... 不然到时候不好维护";
"止损固定5跳，10点 ... 其余参数按 2 倍换算"; "教材生成这次没有改"; "尽量不做兼容，一次性做好"): one profile per
product, required, holds its contract names, tick, costs (only there) and rule differences; strategy, rules,
material and grade distances are in ticks, prices and labels in points; one paper strategy
(markets/strategy_v3.yaml) serves every product."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from gym_trading_env.envs import entry_product as P
from gym_trading_env.envs.entry_contract import is_entry_contract
from gym_trading_env.research import ts_v1_corpus as C
from gym_trading_env.research import ts_v2_positive as V
from gym_trading_env.envs.entry_product import read_preset

CONFIGS = Path(__file__).resolve().parents[3] / 'dreamerv3' / 'configs/thinking_student'
M = CONFIGS / 'markets'
# FG501's rule gate under the per-product rules V7 zone H + 2 (markets/strategy_v3_trend_trail_zone_h2.yaml,
# removed 2026-11-08 for FG's profile rule_overrides), saved before the removal: open, event, H per side.
ZONE_H2_GATE = Path(__file__).parent / 'fixtures/fg501_zone_h2_gate.npz'
real = pytest.mark.skipif(not (CONFIGS / 'products/fg.yaml').exists(), reason='Thinking Student configs missing')


def paper_row(pool, contract):
    row = next(r for r in json.loads(Path(pool).read_text())['files'] if r['contract'] == contract)
    return dict(row, env_config=str((Path(pool).parent / row['env_config']).resolve()))


def env_of(tmp, pool, contract, strategy, edit=None):
    folder = Path(tmp) / contract
    config = C.write_inputs(paper_row(pool, contract), folder, strategy=C.load_strategy(strategy))
    if edit:
        edit(folder)
    return C.native_env(config, market=False)


def test_contract_names_and_ticks():
    for name in ('CZCE.FG501', 'CZCE.SA605', 'SHFE.rb2505', 'SHFE.sp2509', 'DCE.m2509'):
        assert is_entry_contract(name)
    for name in ('FG501', 'CZCE.FG', 'XXX.fg501'):
        assert not is_entry_contract(name)
    cfg = {'entry': {'zone_points': 5, 'stop_above_points': 5, 'target_r': 2.0}, 'prior_high': {'departure_points': 10}}
    assert P.in_points(cfg, P.RULE_TICKS, 2) == {'entry': {'zone_points': 10, 'stop_above_points': 10, 'target_r': 2.0},
                                                'prior_high': {'departure_points': 20}}
    assert P.in_points(cfg, P.RULE_TICKS, 1) == cfg and P.in_points(cfg, P.RULE_TICKS, 2) is not cfg
    assert P.with_overrides(cfg, {'entry': {'zone_above_points': 2}}, 'x')['entry']['zone_above_points'] == 2
    with pytest.raises(ValueError, match='not a key of the rules'):
        P.with_overrides(cfg, {'entry': {'zone_typo': 2}}, 'x')


@real
def test_profiles():
    fg, sa, rb = (P.load_profile(CONFIGS / f'products/{x}.yaml') for x in ('fg', 'sa', 'rb'))
    real_costs = read_preset(M / 'costs_fg_real_20261105.yaml')
    assert fg['costs'] == real_costs and sa['costs'] == real_costs
    assert rb['costs'] == read_preset(M / 'rb/costs_rb_20261106.yaml') and rb['costs']['lot_size'] == 10
    assert [x['tick_points'] for x in (fg, sa, rb)] == [1, 1, 1]
    assert fg['rule_overrides'] == {'entry': {'zone_above_points': 2}} and sa['rule_overrides'] == {}
    assert Path(fg['paper_pool']) == (M / 'paper_pool.json').resolve()
    inline = yaml.safe_load((CONFIGS / 'products/fg.yaml').read_text())
    inline['costs'] = yaml.safe_load((M / 'costs_fg_real_20261105.yaml').read_text())['CZCE_FG']
    assert {k: v for k, v in P.preset_costs({'FG': inline['costs']}, M / 'costs_fg_real_20261105.yaml').items()
            if k not in ('source_path', 'source_sha256')} == {
                k: v for k, v in real_costs.items() if k not in ('source_path', 'source_sha256')}


@real
def test_one_strategy_for_every_product(tmp_path):
    # FG trades the zone up to H + 2 (its profile's rule_overrides), SA and RB the rules' [H-5, H]; each its costs.
    fg = env_of(tmp_path, M / 'paper_pool.json', 'CZCE.FG501', M / 'strategy_v3.yaml')
    sa = env_of(tmp_path, M / 'sa/paper_pool_sa.json', 'CZCE.SA501', M / 'strategy_v3.yaml')
    rb = env_of(tmp_path, M / 'rb/paper_pool_rb.json', 'SHFE.rb1805', M / 'strategy_v3.yaml')
    assert [e.product['product'] for e in (fg, sa, rb)] == ['FG', 'SA', 'RB']
    assert [e.rule_gate['SHORT']['entry'].get('zone_above_points', 0) for e in (fg, sa, rb)] == [2, 0, 0]
    assert [float(e.terms.multiplier) for e in (fg, sa, rb)] == [20, 20, 10]
    assert all(float(e.terms.periods[0][2]['open_fee']) == 2.1 and e.terms.periods[0][2]['spread'] == 0 for e in (fg, sa, rb))
    assert (tmp_path / 'CZCE.FG501' / 'product.yaml').exists()
    # FG's rule positions are those of the earlier per-product file (rules V7 zone H + 2).
    old = np.load(ZONE_H2_GATE)
    for side in ('SHORT', 'LONG'):
        a = fg._gate(side)
        assert np.array_equal(a['open'], old[f'{side}_open']) and np.array_equal(a['event'], old[f'{side}_event'])
        assert np.array_equal(a['H'], old[f'{side}_H']) and a['open'].any()
    # Training strategies keep their rules: no product_rules, no override.
    train = env_of(tmp_path / 'train', M / 'paper_pool.json', 'CZCE.FG501', M / 'strategy_v2_4_fixed_stop.yaml')
    assert train.rule_gate['SHORT']['entry'].get('zone_above_points', 0) == 0


@real
def test_distances_in_ticks(tmp_path):
    # A product of 2 points a tick (SP, "止损固定5跳，10点"): every distance doubles, the fixed rules and exit too.
    def two(folder):
        for name, key in (('product.yaml', 'tick_points'), ('contract.yaml', 'tick_size')):
            raw = yaml.safe_load((folder / name).read_text())
            raw[key] = 2
            (folder / name).write_text(yaml.safe_dump(raw))
    env = env_of(tmp_path, M / 'paper_pool.json', 'CZCE.FG501', M / 'strategy_v3.yaml', two)
    assert env.tick_points == 2
    assert (env.stop_rule['stop_above_points'], env.stop_rule['max_risk_points'],
            env.stop_rule['prior_high']['departure_points']) == (10, 20, 20)
    e = env.rule_gate['SHORT']
    assert (e['entry']['zone_points'], e['entry']['zone_above_points'], e['entry']['stop_above_points'],
            e['entry']['max_risk_points'], e['event']['near_points'], e['prior_high']['departure_points']) == (
                10, 4, 10, 20, 10, 20)
    assert (env.exit_rule['breakeven_points'], env.exit_rule['lock_points']) == (16, 10)
    assert (env.take_profit_points, env.stop_loss_points) == (10, 60)

    def mismatch(folder):
        raw = yaml.safe_load((folder / 'product.yaml').read_text())
        raw['tick_points'] = 2
        (folder / 'product.yaml').write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='tick_size 1'):
        env_of(tmp_path / 'mismatch', M / 'paper_pool.json', 'CZCE.FG501', M / 'strategy_v3.yaml', mismatch)


@real
def test_the_profile_is_required(tmp_path):
    # A master strategy without its product profile, or with costs of its own, is refused; so are a
    # strategy's and a rules file's costs (test_rule_gate).
    from gym_trading_env.envs.entry_contract import resolve_path
    row = paper_row(M / 'paper_pool.json', 'CZCE.FG501')
    env = yaml.safe_load(Path(row['env_config']).read_text())
    for key in ('contract_config', 'direction_config', 'strategy_config'):
        if env['entry'].get(key):
            env['entry'][key] = str(resolve_path(row['env_config'], env['entry'][key]))
    master = yaml.safe_load(Path(env['entry']['strategy_config']).read_text())
    env['entry']['strategy_config'] = str(tmp_path / 'master.yaml')
    (tmp_path / 'env.yaml').write_text(yaml.safe_dump(env))
    for edit, error in ((dict(product_profile=None), 'names its product profile'),
                        (dict(costs=P.read_preset(M / 'costs_fg_real_20261105.yaml')), "not the master strategy's")):
        (tmp_path / 'master.yaml').write_text(yaml.safe_dump(dict(master, **edit)))
        with pytest.raises(ValueError, match=error):
            C.write_inputs(dict(row, env_config=str(tmp_path / 'env.yaml')), tmp_path / 'inputs',
                           strategy=C.load_strategy(M / 'strategy_v3.yaml'))


def test_material_in_ticks(tmp_path, monkeypatch):
    # The material's rules are converted to points once (produce); its Native trades get their TP / SL in
    # ticks, as every env input, and the env multiplies them once (test_distances_in_ticks).
    cfg = {'entry': {'zone_points': 5, 'stop_above_points': 5, 'max_risk_points': 10, 'fixed_stop_points': 5,
                     'target_r': 2.0, 'parallel_r': 1.5}}
    pts = P.in_points(cfg, P.RULE_TICKS, 2)
    assert pts['entry']['fixed_stop_points'] == 10 and pts['entry']['max_risk_points'] == 20
    assert (V.tp_ticks(5, 2.0), V.tp_points(10, 2.0, 2), V.tp_points(10, 1.5, 2), V.tp_points(7, 1.5, 1)) == (
        10, 20, 16, 11)
    written = []
    monkeypatch.setattr(V, 'write_inputs', lambda row, folder, rules: written.append((Path(folder).name, rules))
                        or Path(folder) / 'env.yaml')
    rows = [dict(contract='SHFE.sp2509')]
    candidates = [dict(contract='SHFE.sp2509', cursor=1, label_R=10.0, label_fill_in_zone=True)]
    jobs = V._trade_inputs(rows, pts, tmp_path, candidates, 2)
    assert written == [('R5_TP10', dict(take_profit_points=10, stop_loss_points=5)),
                       ('R5_TP8', dict(take_profit_points=8, stop_loss_points=5))]
    assert [(j['risk'], j['tp']) for j in jobs] == [(5, 10), (5, 8)]
    with pytest.raises(ValueError, match='tick grid'):
        V.risk_ticks(dict(contract='SHFE.sp2509', cursor=1, label_R=9.0), 2)
    # A material's rules in points for an env of its tick; another tick is refused.
    folder = tmp_path / 'material'
    folder.mkdir()
    (folder / 'source.json').write_text(json.dumps(dict(tick_points=2)))
    (folder / 'params.yaml').write_text(yaml.safe_dump(dict(cfg, direction='SHORT')))
    assert V.material_rules_points(folder, 2) == json.loads(json.dumps(pts))
    with pytest.raises(ValueError, match='2-point ticks'):
        V.material_rules_points(folder, 1)


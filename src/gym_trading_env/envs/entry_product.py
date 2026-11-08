"""Product profiles (user 2026-11-08: "后续品种可能增加，这些你要在这次看下怎么设计方案 ... 不然到时候不好维护").

One yaml per product in the Thinking Student configs (`products/<code>.yaml`) holds what is the
product's own: its contract names, how many points a tick is, its trading costs, its paper pool
and its rule differences. A master strategy (the product's market file) points to it with
`product_profile` (required; user 2026-11-08: "尽量不做兼容，一次性做好"); the env reads it from there.
Costs are only here: a master strategy, a strategy or a rules file with `costs` is refused.

    product: SP                       # the master strategy's product
    contract_pattern: 'SHFE\\.sp\\d{4}' # its contracts
    tick_points: 2                    # points of one tick (FG 1; SP 2, user 2026-11-08: "sp一跳2点")
    costs: costs_sp.yaml              # a cost preset path (relative to this file) or its block
    paper_pool: ../markets/sp/paper_pool_sp.json   # optional
    rule_overrides: {entry: {zone_above_points: 2}} # optional, in ticks

Distances of the strategies, the rules, the grades and the reward are in ticks (user 2026-11-08:
"止损固定5跳，10点 ... 其余参数按 2 倍换算"; "教材生成这次没有改"): the env multiplies the keys below by
tick_points and the material converts its rules once (in_points); prices, MAE / MFE, R, trades and
reports are in points; observations are scaled in ticks. A 1-point-tick product (FG, SA, RB) keeps
every number. rule_overrides apply only for a strategy with entry.product_rules (paper /
live, ts-per-product-rules: FG zone [H-5, H+2], SA [H-5, H]); training strategies keep theirs."""
from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from .entry_contract import read_yaml, resolve_path

PROFILE_KEYS = {'product', 'contract_pattern', 'tick_points', 'costs'}
PROFILE_OPTIONAL = {'paper_pool', 'rule_overrides'}
PRESET_KEYS = {'lot_size', 'leverage', 'trading_fee_per_lot', 'is_round_turn', 'spread'}
COST_BASIS = 'user_configured_constant_not_historical_exchange_schedule'
# Distances in ticks, as paths of keys: a rules file (entry.rule_gate, research.ts_v2_positive) ...
RULE_TICKS = (('prior_high', 'departure_points'), ('event', 'near_points'), ('entry', 'zone_points'),
              ('entry', 'zone_above_points'), ('entry', 'touch_break_points'), ('entry', 'stop_above_points'),
              ('entry', 'max_risk_points'), ('entry', 'fixed_stop_points'))
# ... the env's entry.stop_rule ...
STOP_RULE_TICKS = (('stop_above_points',), ('max_risk_points',), ('prior_high', 'departure_points'))
# ... and the Native fixed TP / SL distances (the entry block; ts_v1_corpus / exam rules).
NATIVE_TICKS = ('take_profit_points', 'stop_loss_points')
# The costs of a profile's costs block that the env sets on every period of its contract terms.
TERM_COSTS = ('margin_rate', 'open_fee', 'close_fee', 'close_today_fee', 'spread', 'slippage')
# rule_overrides may set the keys a rules file has, and these optional ones.
OVERRIDE_OPTIONAL = {('entry', 'zone_above_points'), ('entry', 'touch_break_points'), ('entry', 'fixed_stop_points')}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preset_costs(raw, path):
    """The costs block of a cost preset `raw` (one product key, or CZCE_FG; the user's FG preset format,
    legacy per-side fee semantics) read from `path`."""
    from decimal import Decimal
    if 'CZCE_FG' not in raw and len(raw) != 1:
        raise ValueError(f'{path}: a cost preset has CZCE_FG or one product key')
    raw = raw['CZCE_FG'] if 'CZCE_FG' in raw else next(iter(raw.values()))
    leverage = Decimal(str(raw['leverage']))
    if leverage <= 0 or Decimal(str(raw['lot_size'])) <= 0:
        raise ValueError(f'{path}: invalid leverage or multiplier')
    fee = Decimal(str(raw['trading_fee_per_lot']))
    close = fee if raw['is_round_turn'] else Decimal(0)
    return dict(source_path=str(Path(path).resolve()), source_sha256=_sha256(path), cost_basis=COST_BASIS,
                lot_size=float(raw['lot_size']), margin_rate=float(1 / leverage), open_fee=float(fee),
                close_fee=float(close), close_today_fee=float(close), spread=float(raw['spread']), slippage=0.0)


def read_preset(path):
    """The costs block of the cost preset file `path` (preset_costs)."""
    return preset_costs(read_yaml(path), path)


def load_profile(path):
    """A product profile (module docstring), checked; `costs` comes back as a costs block, `path` added."""
    path = Path(path).resolve()
    raw = read_yaml(path)
    if not PROFILE_KEYS <= set(raw) <= PROFILE_KEYS | PROFILE_OPTIONAL:
        raise ValueError(f'{path}: a product profile has {sorted(PROFILE_KEYS)} (and optionally '
                         f'{sorted(PROFILE_OPTIONAL)})')
    if not re.fullmatch(r'[A-Z]{1,2}', str(raw['product'])) or type(raw['tick_points']) not in (int, float) \
            or raw['tick_points'] <= 0:
        raise ValueError(f'{path}: product is an upper-case code, tick_points a positive number')
    re.compile(raw['contract_pattern'])
    costs = raw['costs']
    if isinstance(costs, str):
        raw['costs'] = read_preset(resolve_path(path, costs))
    elif isinstance(costs, dict) and set(costs) == PRESET_KEYS:
        raw['costs'] = preset_costs({raw['product']: costs}, path)
    else:
        raise ValueError(f'{path}: costs is a cost preset path or a preset block {sorted(PRESET_KEYS)}')
    overrides = raw.get('rule_overrides') or {}
    if not isinstance(overrides, dict) or any(not isinstance(v, dict) for v in overrides.values()):
        raise ValueError(f'{path}: rule_overrides is {{section: {{key: value}}}} of a rules file')
    raw['rule_overrides'] = overrides
    if 'paper_pool' in raw:
        raw['paper_pool'] = str(resolve_path(path, raw['paper_pool']))
    raw['path'] = str(path)
    return raw


def profile_copy(profile):
    """A loaded profile as a self-contained profile file (write_inputs: product.yaml next to env.yaml):
    its cost preset inlined, no paper pool."""
    raw = read_yaml(profile['path'])
    if isinstance(raw['costs'], str):
        preset = read_yaml(resolve_path(profile['path'], raw['costs']))
        raw['costs'] = preset['CZCE_FG'] if 'CZCE_FG' in preset else next(iter(preset.values()))
    raw.pop('paper_pool', None)
    return raw


def master_profile(master, master_path):
    """The profile of a master strategy (its `product_profile`, relative to it, required); its product
    must be the master's, and the master has no costs of its own."""
    if not master.get('product_profile'):
        raise ValueError(f'{master_path}: a master strategy names its product profile (product_profile)')
    if 'costs' in master:
        raise ValueError(f'{master_path}: costs are the product profile\'s, not the master strategy\'s')
    profile = load_profile(resolve_path(master_path, master['product_profile']))
    if profile['product'] != str(master['product']).upper():
        raise ValueError(f'{master_path}: product {master["product"]}, its profile {profile["product"]}')
    return profile


def check_contract(profile, contract, tick):
    """The contract is the profile's, and its terms' tick is the profile's tick_points."""
    if not re.fullmatch(profile['contract_pattern'], contract):
        raise ValueError(f'{contract} is not a {profile["product"]} contract ({profile["contract_pattern"]})')
    if tick_number(tick) != profile['tick_points']:
        raise ValueError(f'{contract}: tick_size {tick}, the {profile["product"]} profile {profile["tick_points"]}')


def tick_number(tick):
    """A contract's tick (Decimal) as the profile number: an int when whole, else a float."""
    return int(tick) if tick == int(tick) else float(tick)


def in_points(cfg, keys, tick):
    """A copy of cfg with the distances at `keys` (paths; absent ones skipped) from ticks to points."""
    out = copy.deepcopy(cfg)
    if tick == 1:
        return out
    for path in keys:
        node = out
        for k in path[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict) and node.get(path[-1]) is not None:
            node[path[-1]] = node[path[-1]] * tick
    return out


def with_overrides(cfg, overrides, name):
    """A copy of the rules cfg with the profile's rule_overrides (in ticks) applied; a key the rules don't
    have is refused unless it is an optional one (OVERRIDE_OPTIONAL)."""
    out = copy.deepcopy(cfg)
    for section, values in overrides.items():
        for key, value in values.items():
            if key not in out.get(section, {}) and (section, key) not in OVERRIDE_OPTIONAL:
                raise ValueError(f'{name}: rule_overrides {section}.{key} is not a key of the rules')
            out.setdefault(section, {})[key] = value
    return out

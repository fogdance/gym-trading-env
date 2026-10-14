"""Pure, shared evaluation schema; only canonical effective bars measure time.

This module deliberately uses the standard library so the isolated read-only
tracker can load it without importing Gym, JAX, or the training environment.
"""
from decimal import Decimal
import math

SCHEMA = 'S3_EVAL_METRIC_V2'
WINDOWS = ('adverse10_first5', 'adverse10_6_15', 'adverse10_after15',
           'adverse10_any_hold', 'adverse10_unclassified')


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return float(values[lo] + (values[hi] - values[lo]) * (index - lo))


def unique_events(events):
    """One attribution per entry/reason; identical repeats are harmless."""
    result = {}
    for event in events:
        key = (event.get('entry_id'), event.get('reason'))
        if key[0] is None:
            # Missing IDs cannot prove two records describe the same entry.
            key = ('unknown-entry', len(result), key[1])
        if key in result and result[key] != event:
            raise ValueError('Conflicting evaluation attribution for one entry')
        result[key] = event
    return list(result.values())


def evaluation_metrics(state, receipt=None):
    """Derive fresh metrics without changing an immutable historical receipt.

    Missing logs are unknown. Missing effective_bar is explicitly unclassified;
    timestamp differences are never a substitute for market minutes.
    """
    receipt = receipt or {}
    metrics = dict(metric_schema=SCHEMA, **{k: None for k in WINDOWS},
                   adverse10_windows_complete=False,
                   closed_trade_net_pnl=None, closed_trade_gross_pnl=None,
                   tp_market_minutes_p50=None, tp_market_minutes_p90=None)
    legacy = receipt.get('early_mae10')
    if legacy is not None:
        metrics['legacy_early_mae10_any_hold'] = legacy
    if state is None:
        metrics['adverse10_any_hold'] = receipt.get('adverse10_any_hold', legacy)
        return metrics
    if receipt.get('contract') and state.get('contract') != receipt['contract']:
        raise ValueError('Evaluation account contract differs from receipt')
    if receipt.get('reward') is not None and state.get('total_reward') != receipt['reward']:
        raise ValueError('Evaluation account reward differs from receipt')
    if state.get('attributions') is not None:
        attrs = unique_events(state['attributions'])
        tp = [a for a in attrs if a.get('reason') == 'TAKE_PROFIT']
        sl = [a for a in attrs if a.get('reason') == 'STOP_LOSS']
        for name, count in (('tp_count', len(tp)), ('sl30', len(sl))):
            if receipt.get(name) is not None and receipt[name] != count:
                raise ValueError('Evaluation account '+name+' differs from receipt')
        for field, target in (('net_trade_pnl', 'closed_trade_net_pnl'),
                              ('gross_trade_pnl', 'closed_trade_gross_pnl')):
            if all(a.get(field) is not None for a in attrs):
                metrics[target] = float(sum((Decimal(str(a[field])) for a in attrs), Decimal(0)))
        minutes = [a.get('tp_market_minutes') for a in tp]
        if all(type(b) is int and b >= 1 for b in minutes):
            metrics['tp_market_minutes_p50'] = quantile(minutes, .5)
            metrics['tp_market_minutes_p90'] = quantile(minutes, .9)
        metrics.update(tp_count=len(tp), sl30=len(sl), closed_trade_count=len(attrs))
    if state.get('reward_events') is None:
        metrics['adverse10_any_hold'] = receipt.get('adverse10_any_hold', legacy)
        return metrics
    events = unique_events([e for e in state['reward_events']
                            if e.get('reason') == 'FIRST_ADVERSE10'])
    count = len(events)
    if receipt.get('adverse10_any_hold', legacy) is not None:
        if count != receipt.get('adverse10_any_hold', legacy):
            raise ValueError('Evaluation adverse events differ from receipt count')
    clock = receipt.get('clock_protocol', state.get('clock_protocol'))
    bars = [e.get('effective_bar') for e in events]
    if any(type(b) is int and b < 1 for b in bars):
        raise ValueError('Market-minute attribution must start at effective_bar 1')
    valid = [b for b in bars if clock == 'S3_MARKET_MINUTE' and type(b) is int]
    metrics.update(adverse10_first5=sum(b <= 5 for b in valid),
        adverse10_6_15=sum(6 <= b <= 15 for b in valid),
        adverse10_after15=sum(b >= 16 for b in valid),
        adverse10_any_hold=count, adverse10_unclassified=count-len(valid),
        adverse10_windows_complete=len(valid) == count)
    return metrics

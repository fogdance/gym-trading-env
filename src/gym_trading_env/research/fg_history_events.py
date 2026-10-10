"""Independent single-entry research probes; this module is not a trading Env.

Queries use only the already sliced contract/role frame. No maximum holding age.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class RangeMax:
    """O(log n) range maximum and earliest threshold crossing, O(n) storage."""
    def __init__(self, values):
        self.length = len(values)
        self.size = 1 << max(0, (self.length - 1).bit_length())
        self.tree = np.full(2 * self.size, -np.inf)
        self.tree[self.size:self.size + self.length] = values
        for start in range(self.size - 1, 0, -1):
            self.tree[start] = max(self.tree[2 * start], self.tree[2 * start + 1])

    def maximum(self, start, end):
        if end < start:
            return -np.inf
        left, right = start + self.size, end + self.size + 1
        result = -np.inf
        while left < right:
            if left & 1:
                result = max(result, self.tree[left]); left += 1
            if right & 1:
                right -= 1; result = max(result, self.tree[right])
            left //= 2; right //= 2
        return result

    def first(self, start, end, threshold):
        def visit(node, left, right):
            if right <= start or left > end or self.tree[node] < threshold:
                return -1
            if right - left == 1:
                return left
            middle = (left + right) // 2
            found = visit(node * 2, left, middle)
            return found if found >= 0 else visit(node * 2 + 1, middle, right)
        return visit(1, 0, self.size) if end >= start else -1


def minute_segments(times, schedule):
    minute = times.hour * 60 + times.minute
    segment = np.full(len(times), -1)
    bounds = []
    for i, (start, end) in enumerate(schedule['segments']):
        a, b = int(start[:2]) * 60 + int(start[3:]), int(end[:2]) * 60 + int(end[3:])
        segment[(minute > a) & (minute <= b)] = i
        bounds.append((a, b))
    return segment, minute, bounds


def observation_barriers(frame, schedule, omitted_bobs=()):
    """Return starts of unobservable intervals, never read omitted OHLC prices.

    Whole absent sessions remain an independent-calendar limitation. Detected
    partial-session edges are barriers only where both observed sides exist.
    """
    times = pd.DatetimeIndex(pd.to_datetime(frame.Date))
    bobs = pd.DatetimeIndex(pd.to_datetime(frame.bob))
    segment, minute, bounds = minute_segments(times, schedule)
    barriers = {pd.Timestamp(x).value: 'policy_truncated_night' for x in omitted_bobs}
    for i in range(len(times)):
        if times[i] - bobs[i] != pd.Timedelta(minutes=1):
            barriers[bobs[i].value] = 'unverified_bar_time'
        if i == 0:
            continue
        if segment[i] == segment[i - 1] and times[i].date() == times[i - 1].date():
            if times[i] - times[i - 1] > pd.Timedelta(minutes=1):
                barriers[times[i - 1].value] = 'missing_minute_record'
        else:
            # Do not invent an exchange calendar for an entirely absent session.
            if minute[i - 1] != bounds[segment[i - 1]][1]:
                barriers[times[i - 1].value] = 'missing_session_edge'
            if minute[i] != bounds[segment[i]][0] + 1:
                start = times[i].normalize() + pd.Timedelta(minutes=bounds[segment[i]][0])
                barriers[start.value] = 'missing_session_edge'
    return sorted(barriers.items())


def generate_events(frame, *, contract, role, mode, costs, schedule,
                    terminal=False, omitted_bobs=(), take_profit=5, quantity=3,
                    initial_balance=300000):
    """Yield independent LONG/SHORT outcomes, including rejected decisions.

    conservative_spread matches the constant-cost, fresh-account single probe
    execution convention, not a multi-entry account or actual broker fills.
    """
    if mode not in {'touch_spread', 'conservative_spread', 'touch_zero_spread'}:
        raise ValueError('Unknown execution mode')
    if frame.empty:
        return
    frame = frame.reset_index(drop=True)
    times = pd.DatetimeIndex(pd.to_datetime(frame.Date))
    bobs = pd.DatetimeIndex(pd.to_datetime(frame.bob))
    if times.has_duplicates or not times.is_monotonic_increasing or times.hasnans:
        raise ValueError('Probe frame must be unique and chronological')
    if set(frame.symbol) != {contract}:
        raise ValueError('Probe cannot mix real contracts')
    values = frame[['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any():
        raise ValueError('Invalid raw probe values')
    o, h, l, c, volume, _ = values.T
    if ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)).any():
        raise ValueError('Invalid OHLC probe')
    if (values[:, :4] % 1).any():
        raise ValueError('FG probes require the confirmed one-point tick')
    segment, _, _ = minute_segments(times, schedule)
    if (segment < 0).any():
        raise ValueError('Probe price outside the frozen 345 clock')
    valid = (times - bobs == pd.Timedelta(minutes=1))
    barriers = observation_barriers(frame, schedule, omitted_bobs)
    barrier_times = np.array([x[0] for x in barriers], dtype=np.int64)
    spread = 0 if mode == 'touch_zero_spread' else costs['spread'] + costs.get('slippage', 0)
    conservative = mode == 'conservative_spread'
    n = len(frame)
    days = frame.trading_date.astype(str).to_numpy()
    unique_days = list(dict.fromkeys(days))
    day_index = {day: i for i, day in enumerate(unique_days)}
    # A partial final session must not redefine a trading day's close.
    day_close = {day: pd.Timestamp(day, tz=times.tz) + pd.Timedelta(hours=15) for day in unique_days}
    observed_bars = np.cumsum(valid)
    liquidity = valid & (volume >= quantity)
    liquidity_tree = RangeMax(liquidity.astype(float))
    for direction, sign in [('LONG', 1), ('SHORT', -1)]:
        # Price rounding follows entry_env._price: buy ceil, sell floor.
        entry_prices = np.ceil(o + spread) if sign == 1 else np.floor(o - spread)
        favorable = h if sign == 1 else -l
        exit_extreme = np.floor(h - spread) if sign == 1 else -np.ceil(l + spread)
        exit_close = np.floor(c - spread) if sign == 1 else -np.ceil(c + spread)
        search = exit_extreme if conservative else favorable
        target_tree = RangeMax(np.where(liquidity if conservative else valid, search, -np.inf))
        adverse_raw = l if sign == 1 else h
        adverse_exit = np.floor(l - spread) if sign == 1 else np.ceil(h + spread)
        adverse_tree = RangeMax(-l if sign == 1 else h)
        # Equity after one independent opening, before any exit fees.
        cash = initial_balance - costs['open_fee'] * quantity
        units = quantity * costs['lot_size']
        margin_metric = costs['margin_rate'] * adverse_raw - sign * adverse_exit
        margin_tree = RangeMax(margin_metric)
        for decision in range(n):
            j = decision + 1
            record = dict(contract=contract, trading_date=None, decision_trading_date=days[decision],
                          decision_timestamp=times[decision], direction=direction, role=role,
                          execution_assumption=mode, entry_timestamp=None, entry_price=None,
                          tp_price=None, first_touch_timestamp=None, touch_time_lower=None,
                          touch_time_upper=None, duration_minutes=None, duration_minutes_lower=None,
                          duration_trading_bars=None, MAE=None, MAE_lower=None, MAE_upper=None,
                          completion_status='no_entry', censor_reason=None, terminal_reason=None,
                          reward=None, observation_end=None, exit_price=None, account_net_pnl=None,
                          followup_minutes=None,
                          entry_session=None, horizon_1day_end=None, horizon_5day_end=None,
                          rejection_reason=None)
            if not valid[decision]:
                record['rejection_reason'] = 'unverified_decision_bar'
                yield record; continue
            barrier_index = int(np.searchsorted(barrier_times, times[decision].value, side='left'))
            barrier = barriers[barrier_index] if barrier_index < len(barriers) else None
            if j >= n:
                record['rejection_reason'] = 'no_next_bar_in_role'
            elif not valid[j] or bobs[j] < times[decision]:
                record['rejection_reason'] = 'unverified_entry_bar'
            elif barrier and barrier[0] <= bobs[j].value:
                record['rejection_reason'] = 'unobservable_before_entry'
            elif volume[j] < quantity:
                record['rejection_reason'] = 'insufficient_entry_volume'
            elif conservative and units * entry_prices[j] * costs['margin_rate'] > cash:
                record['rejection_reason'] = 'insufficient_entry_margin'
            if record['rejection_reason']:
                yield record; continue
            entry = float(entry_prices[j])
            target = entry + sign * take_profit
            day = days[j]
            fifth = day_index[day] + 4
            record.update(trading_date=day, entry_timestamp=bobs[j], entry_price=entry, tp_price=target,
                          entry_session='night' if times[j].hour >= 18 else 'day',
                          horizon_1day_end=day_close[day],
                          horizon_5day_end=day_close[unique_days[fifth]] if fifth < len(unique_days) else None)
            end = n - 1
            reason = 'research_window_end'
            if barrier and barrier[0] <= times[-1].value:
                end = int(times.searchsorted(pd.Timestamp(barrier[0], tz=times.tz), side='right')) - 1
                # Zero-duration records are not usable evidence of fills/touches.
                if end >= j and not valid[end]:
                    end -= 1
                reason = barrier[1]
            can_terminal = terminal and end == n - 1
            stop = end - 1 if conservative and can_terminal else end
            hit = -1
            threshold = sign * target + (1 if conservative else 0)
            if j <= stop:
                if conservative:
                    if volume[j] >= 2 * quantity and exit_close[j] >= threshold:
                        hit = j
                    else:
                        hit = target_tree.first(j + 1, stop, threshold)
                else:
                    hit = target_tree.first(j, stop, threshold)
            risk = -1
            if conservative and j <= end:
                margin_threshold = cash / units - sign * entry
                risk = margin_tree.first(j, end, np.nextafter(margin_threshold, np.inf))
            if risk >= 0 and (hit < 0 or risk <= hit):
                # Pending margin exits persist when that minute lacks liquidity.
                risk_fill = risk
                if not liquidity[risk] or (risk == j and volume[j] < 2 * quantity):
                    risk_fill = liquidity_tree.first(risk + 1, end, 1)
                endpoint = risk_fill if risk_fill >= 0 else end
                record.update(completion_status='terminal_failure', terminal_reason='margin_call', reward=0.0)
                if risk_fill >= 0:
                    record['exit_price'] = float(adverse_exit[risk_fill])
                else:
                    record['terminal_reason'] = 'margin_call_unfilled'
            elif hit >= 0:
                endpoint = hit
                # Only primary touch mode uses a favorable gap-open timestamp.
                gap_open = not conservative and sign * o[hit] >= sign * target
                touch = bobs[hit] if gap_open else times[hit]
                lower = touch if gap_open or conservative else bobs[hit]
                elapsed = (touch - bobs[j]).total_seconds() / 60
                record.update(completion_status='success', first_touch_timestamp=touch,
                              touch_time_lower=lower, touch_time_upper=touch,
                              duration_minutes=elapsed,
                              duration_minutes_lower=(lower - bobs[j]).total_seconds() / 60,
                              reward=60 / (60 + elapsed))
                if conservative:
                    record['exit_price'] = target
            elif can_terminal:
                endpoint = end
                record.update(completion_status='terminal_failure', terminal_reason='main_qualification_end', reward=0.0)
                if liquidity[end] and (end != j or volume[end] >= 2 * quantity):
                    record['exit_price'] = float(np.floor(c[end] - spread) if sign == 1 else np.ceil(c[end] + spread))
                else:
                    record['terminal_reason'] = 'main_qualification_end_unfilled'
            else:
                endpoint = end
                record.update(completion_status='right_censored', censor_reason=reason)
            mae = max(0.0, sign * entry + adverse_tree.maximum(j, endpoint))
            lower_mae = max(0.0, sign * entry + adverse_tree.maximum(j, endpoint - 1)) if hit == endpoint and hit >= 0 else mae
            # A known opening gap reaches target before that bar's later extreme.
            if record['completion_status'] == 'success' and record['first_touch_timestamp'] == bobs[endpoint]:
                mae = lower_mae
            gap_touch = record['completion_status'] == 'success' and record['first_touch_timestamp'] == bobs[endpoint]
            record.update(MAE=mae, MAE_lower=lower_mae, MAE_upper=mae,
                          duration_trading_bars=int(observed_bars[endpoint] - (observed_bars[j - 1] if j else 0) - int(gap_touch)),
                          observation_end=record['first_touch_timestamp'] if record['completion_status'] == 'success' else times[endpoint])
            record['followup_minutes'] = (record['observation_end'] - bobs[j]).total_seconds() / 60
            if record['exit_price'] is not None:
                closing_fee = costs['close_today_fee'] if days[endpoint] == day else costs['close_fee']
                record['account_net_pnl'] = sign * (record['exit_price'] - entry) * units - quantity * (costs['open_fee'] + closing_fee)
            yield record

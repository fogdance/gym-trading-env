"""Opt-in, post-hoc Single3 quality labels using a REAL account snapshot.

Strict Oracle and its 300k funds domain are independent.  A valid trade
outcome is not proof that ADD dominates WAIT.  All future reads occur ONLY in
an isolated copy of the native Env, after the caller's decision state is frozen.
No Agent/Actor/Replay/legacy Reward code is modified by this module.
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_s3_quality_score import PROTOCOL as SCORE_PROTOCOL, quality_for_exit, tp_event_reward
from .fg_s3_oracle_state_query import decision_state, rules_identity
from .fg_single3_horizon import fork_account

PROTOCOL = 'FG_S3_QUALITY_TEACHER_POSTHOC_V1'


def later_official_days(env, entry_cursor: int, exit_cursor: int) -> int:
    """Count real official trading-day transitions, including night sessions."""
    if not 0 <= entry_cursor <= exit_cursor < len(env._raw):
        raise ValueError('Invalid entry/exit cursor')
    column = ('trading_date' if 'trading_date' in env._raw else
              'trading_day' if 'trading_day' in env._raw else None)
    if column is None:
        raise ValueError('No official trading dates: wall-date fallback forbidden')
    days = env._raw.iloc[entry_cursor:exit_cursor + 1][column].astype(str).tolist()
    # Monotonic official-day order. Identical day in multiple sessions counts once.
    if len(set(days)) != sum(a != b for a, b in zip(days, days[1:])) + 1:
        raise ValueError('Non-contiguous official trading dates')
    return sum(a != b for a, b in zip(days, days[1:]))


def observed_mae(env, entry_cursor: int, exit_cursor: int,
                 side: str, entry_price: str) -> float:
    """OHLC execution-quote envelope, not tick-order precision."""
    sign = 1 if side == 'LONG' else -1 if side == 'SHORT' else None
    if sign is None:
        raise ValueError('Unknown entry side')
    entry = Decimal(str(entry_price))
    adverse = 'Low' if sign == 1 else 'High'
    worst = Decimal(0)
    for i in range(entry_cursor, exit_cursor + 1):
        if not env.valid_fill_clock[i]:
            continue
        time = env._raw.index[i]
        quote = env._price(env._raw.iloc[i][adverse], side, time)
        worst = max(worst, (entry - quote) * sign)
    return float(worst)


def _intact(env, before):
    if fingerprint(env.export_state()) != before:
        raise RuntimeError('Quality Teacher mutated the live input account')


def quality_record(env, *, fold: str, strict_query=None, max_market_steps=None):
    """One legal current-state counterfactual ADD->WAIT until exit/boundary.

    Returned ``quality_valid`` only establishes realized *single-entry* result;
    ``strict_decision_valid`` belongs to the optional, separate original Oracle.
    The V1 record explicitly forbids direct use as a TRAIN Actor action label.
    """
    if fold not in ('TRAIN', 'DEV'):
        raise ValueError('V1 quality teacher refuses TEST and unknown folds')
    if not env.single3 or env.reward_config.candidate != 'C1':
        raise ValueError('Expected frozen Single3 C1 environment')
    # E's mutable observation cache is shared by copy.copy(env). Do not claim
    # nonmutation until its independent fork semantics have been verified.
    if getattr(env, '_e_profile', None) is not None:
        raise ValueError('E-profile counterfactual fork is NOT_VERIFIED in V1')
    state = env.export_state()
    before = fingerprint(state)
    decision = decision_state(env)
    mask = np.asarray(env._published_mask)
    if (env.status != 'ACTIVE' or env.terminated or env.truncated or
            env.net_lots or env.entries or env.pending_reason or
            env.reward_function.scorer.active or not np.array_equal(mask, [1, 1])):
        raise ValueError('Only actual ACTIVE flat states with both legal actions are eligible')
    if max_market_steps is not None and (type(max_market_steps) is not int or max_market_steps < 1):
        raise ValueError('max_market_steps must be positive; censored paths are not scored')
    strict = dict(valid=False, certified=False, action=None, reason='NOT_REQUESTED',
                  oracle_identity=None, proof_hash=None)
    if strict_query is not None:
        answer = strict_query.query(env, mask, snapshot=decision)
        strict = {k: answer.get(k) for k in strict}
        strict['action'] = answer.get('action') if answer.get('certified') else None
    record = dict(protocol=PROTOCOL, score_protocol=SCORE_PROTOCOL, fold=fold,
        training_action_label_allowed=False,
        rule_identity=rules_identity(env), decision_snapshot_hash=before,
        decision_state_hash=fingerprint(decision),
        decision_cursor=int(env.current_step),
        starting_equity=str(env._calculate_equity()),
        actual_action_mask=[int(x) for x in mask],
        strict_decision_valid=bool(strict['valid'] and strict['certified']),
        strict_oracle=strict, immediate_add_outcome_valid=False,
        quality_valid=False, quality_score=None, tp_event_reward_new=None,
        new_reward_undiscounted=None, old_reward_undiscounted=None,
        outcome='UNKNOWN', reason='NOT_SIMULATED')
    fork = fork_account(env, state)
    try:
        original_fills = len(fork.fills)
        original_events = len(fork.attributions)
        first_cursor = fork.current_step
        _, reward, *_ = fork.step(1)
        rewards = [float(np.float32(reward))]
        fills = [f for f in fork.fills[original_fills:] if f.get('reason') == 'ENTRY']
        if not fills:
            record.update(outcome='REJECTED', reason='NO_REAL_ENTRY_FILL')
            return record
        if len(fills) != 1:
            raise ValueError('Single3 counterfactual executed multiple entries')
        entry = fills[0]
        advanced = fork.market_clock_counter.advance(first_cursor, fork.current_step)
        while (fork.entries and not fork.terminated and
               fork.status not in ('DATA_END', 'DATA_GAP')):
            if max_market_steps is not None and advanced >= max_market_steps:
                break
            if fork.truncated:
                fork.reset()  # original chunk semantics; does not create a new bar
            cursor = fork.current_step
            _, reward, *_ = fork.step(0)
            if fork.current_step == cursor:
                break
            rewards.append(float(np.float32(reward)))
            advanced += fork.market_clock_counter.advance(cursor, fork.current_step)
        events = [e for e in fork.attributions[original_events:]
                  if e['entry_id'] == entry['entry_id']]
        if not events:
            record.update(outcome='RIGHT_CENSORED', reason='NO_COMMITTED_EXIT',
                          last_observed_cursor=int(fork.current_step))
            return record
        if len(events) != 1:
            raise ValueError('Single3 requires exactly one atomic exit event')
        event = events[0]
        entry_cursor = int(event['entry_cursor'])
        exit_cursor = int(event['exit_cursor'])
        t = int(event['held_market_bars'])
        if t != fork.market_clock_counter.held(entry_cursor, exit_cursor):
            raise ValueError('Reported TP/SL market clock differs from original clock')
        d = later_official_days(fork, entry_cursor, exit_cursor)
        grade, grade_reason = quality_for_exit(event['reason'], t, d)
        old_tp = 0.0
        new_tp = None
        if event['reason'] == 'TAKE_PROFIT':
            old_tp = float(fork.reward_function.scorer.closed[event['entry_id']].timeboxed_tp)
            new_tp = tp_event_reward(t, d)
        baseline = float(sum(rewards))
        revised = baseline - old_tp + (new_tp if new_tp is not None else 0.0)
        record.update(outcome=event['reason'], reason=grade_reason,
            immediate_add_outcome_valid=True, quality_valid=(grade is not None),
            quality_score=grade, tp_event_reward_new=new_tp,
            old_tp_event_reward=old_tp if new_tp is not None else None,
            old_reward_undiscounted=baseline, new_reward_undiscounted=revised,
            entry_cursor=entry_cursor, exit_cursor=exit_cursor,
            entry_time=event['entry_time'], exit_time=event['exit_time'],
            market_minutes=t, later_trading_days=d,
            mae_quote_envelope_points=observed_mae(fork,entry_cursor,exit_cursor,
                                                    event['side'],event['entry_price']),
            net_trade_pnl=str(event['net_trade_pnl']), exit_event_hash=fingerprint(event),
            official_entry_date=fork._trading_day(event['entry_time']),
            official_exit_date=fork._trading_day(event['exit_time']),
            # Strict optimality is intentionally NOT inferred from quality.
            add_better_than_wait=None)
        return record
    finally:
        _intact(env, before)
        # Do NOT call fork.close(): the production Env.close() may write the
        # configured state_path. This post-hoc analysis must be read-only.

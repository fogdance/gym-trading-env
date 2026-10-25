"""Thinking Student V1 counterfactual teacher: same-state ADD/WAIT branches of
a frozen complete student pi_k (C3 5.5; A4 and A6 2.4).

The reference is pi_k's own formal Native trajectory on one TRAIN contract,
one decision per market cursor from the first minute to the true terminal.
At a reference decision where the account is flat and may ADD, a branch
restores the same Native account, market memory and policy seeds, executes
the other first action and then lets the same frozen pi_k decide. The
policy runs at every step, also for the forced first action, so its memory
and seeds advance as on the reference path. Gym never imports the student:
a branch drives a stepper (see `run_branch`).

Returns are discounted per valid market minute (A4): the step of a decision
at cursor t pays r_t with feedback at t+1, and

  G(t) = r_t + gamma ** advance(t, t+1) * G(t+1),  G(last) = r_last at the true terminal.

Coupling. Exact equality of the future-relevant Native and frozen-policy
state at a common flat cursor is sufficient, even below the funds floor.
`native_continuation` defines the deliberately limited Single3 projection;
the caller supplies the policy continuation (including market carry/seeds).
Alternatively, once a branch is flat at a cursor c where the reference is flat
too, with equal position_state, action_mask and account_native, the two
accounts differ only in cash and history. pi_k reads the same market memory
(the market is exogenous) and the same account inputs, so it takes the same
action, and a Native trade and its reward depend on the market and the
position, not on cash, while funds cannot block an ADD or unsaturate a cover
input (`coupling_floor`). With the equity gap d = equity_branch(c) -
equity_ref(c) that holds for the rest of the trajectory when
min(reference equity from c) + min(d, 0) >= floor, and then

  G_branch(t) = sum_{t<=i<c} gamma ** advance(t, i) * r_i + gamma ** advance(t, c) * G_ref(c)

exactly: VALID_COMPLETE without a tail estimate. A branch that reaches the
true terminal is complete too. A data end or the step cap before either is
NOT_COMPARABLE and gets no numeric label. VALID_ESTIMATED (a declared tail
estimator) belongs to the label contract but is not produced here.

Budget. The funds floor cannot predict exact equality or a branch terminal.
Every legal candidate may be probed. `advance_branch` pauses without making
a label; its progress and the stepper state must be saved together. A caller
may sample unfinished probes for continuation and mark the rest
UNAVAILABLE_BUDGET, with no numeric Q. Probe and continuation share one cap.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from decimal import Decimal

import numpy as np

from .fg_single3_oracle import GAMMA

STATUSES = ('VALID_COMPLETE', 'VALID_ESTIMATED', 'NOT_COMPARABLE', 'UNAVAILABLE_BUDGET')
WAIT, ADD = 0, 1
INPUTS = ('position_state', 'action_mask', 'account_native')


def same_state(a, b):
    """Exact structured equality, including array dtype and shape (no tolerance)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and np.array_equal(a, b)
    return type(a) is type(b) and a == b


def native_continuation(env):
    """Small, read-only sufficient state at a flat Single3 decision, or None.

    This is NOT a replacement for the full restore snapshot. It is restricted
    to FixedDirectionEntryEnv + Single3Reward, on the same immutable market
    and configuration. Stepper/Reference must also compare policy state.

    Exclusions follow the actual transition code:
    * EntryEnv.step slices fills from fills_before, and _open_entry/settle_exit
      read only the trade record just appended by the atomic Native commit.
      Old journals, trades, attributions, equity and reward logs are reports.
    * Single3Reward.step reads active positions and newly closed positions;
      scorer totals/discounted/ambiguities and reward.total only accumulate.
      Closed payloads never enter the next reward or account observation.
    * seen and processed_fills guard duplicate IDs. At flat we verify that
      every ID is from a completed entry in this account's monotone namespace,
      with exactly one exit per entry. Future _open_entry IDs have a strictly
      larger sequence, so their timestamps/reasons cannot collide. Unknown or
      future IDs fail closed; historical payloads are not compared.
    * legal_add_decisions, voluntary_waits, max_floating_loss, completed_accounts,
      closed_trade_profits and last_close_position are reporting/rollback only.
      No account reset or external settle_exit is performed by a CF stepper.

    All ledger balances remain Decimal, including broker accounts. No
    observation equality or float equity gap substitutes for financial state.
    """
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    from gym_trading_env.rewards.single3_reward import Single3Reward
    from gym_trading_env.research.fg_reward_candidates import FGRewardScorer
    from .ts_v1_account import StudentEnv
    if type(env) is StudentEnv:
        env = env.env
    if type(env) is not FixedDirectionEntryEnv or not env.single3:
        return None
    reward = env.reward_function
    if type(reward) is not Single3Reward or type(reward.scorer) is not FGRewardScorer:
        return None
    scorer = reward.scorer
    positions = (env.position_manager.long_positions, env.position_manager.short_positions)
    if (env.net_lots or env.entries or scorer.active or
            any(p is not None for side in positions for p in side)):
        return None
    prefix = f'{env.contract}:{env.account_sequence}:'
    def past(entry_id):
        suffix = entry_id[len(prefix):] if isinstance(entry_id, str) and entry_id.startswith(prefix) else ''
        return suffix.isdecimal() and str(int(suffix)) == suffix and 1 <= int(suffix) <= env.entry_sequence
    if (len(scorer.seen) != env.entry_sequence or set(scorer.closed) != scorer.seen
            or not all(past(k) for k in scorer.seen)):
        return None
    fills = [k.split('|') for k in env.processed_fills]
    if (len(fills) != env.entry_sequence or
            any(len(k) != 4 or k[-1] != '3' for k in fills) or
            {k[0] for k in fills} != scorer.seen):
        return None
    # Quality-day dictionaries must be empty after all positions close.
    if (getattr(scorer, '_quality_last_day', {}) or
            getattr(scorer, '_quality_crossed_days', {})):
        return None
    tr = env.config.trading
    return deepcopy(dict(
        contract=env.contract, config=env.config_hash,
        cursor_time=str(env._raw.index[env.current_step]),
        balances=env.ledger.snapshot(), ledger_total=env._ledger_total0,
        ledger_rules={k: v.strict_nonnegative for k, v in env.ledger._accounts.items()},
        realized_pnl=env.user_accounts.realized_pnl,
        unrealized_pnl=env.user_accounts.unrealized_pnl,
        position_slots=tuple(len(side) for side in positions),
        counters={k: getattr(env, k) for k in ('current_step', 'episode_step_count',
                  'chunk_count', 'entry_sequence', 'account_sequence', '_entries_used_today')},
        current_price=env.current_price, published_mask=np.array(env._published_mask),
        exit={k: getattr(env, k) for k in ('status', 'terminated', 'truncated', 'pending_reason')},
        rules={k: getattr(tr, k) for k in ('intraday_single_position', 'max_entries_per_day',
               'trade_lot', 'max_long_position', 'max_short_position', 'lot_size',
               'leverage', 'trading_fee_per_lot', 'spread', 'is_round_turn')},
        chunk_steps=env.chunk_steps, main_exit_at=env.main_exit_at,
        capacity=(env.max_lots, env.max_entries), live=env._live_mode,
        rng=env.np_random.bit_generator.state,
        reward=dict(config=asdict(reward.config), v2=asdict(reward.v2_config),
                    profile=scorer.tp_profile, schema=scorer.SCHEMA,
                    last_mark=scorer.last_mark, last_end=scorer.last_end,
                    step_number=scorer.step_number)))


def coupling_floor(env):
    """Equity above which cash cannot change a decision or a trade: a flat
    account may always ADD and the funds_cover / margin_cover account inputs
    stay saturated (ts_v1_account). Held margin plus four 3-lot ADD costs
    at the highest execution quote of the contract and its costliest terms."""
    periods = [terms for _, _, terms in env.terms.periods]
    rate = max(Decimal(str(t['margin_rate'])) for t in periods)
    fee = max(Decimal(str(t['open_fee'])) for t in periods)
    cost = max(Decimal(str(t['spread'])) + Decimal(str(t['slippage'])) for t in periods)
    high = Decimal(str(env._raw.High.max())) + cost + env.terms.tick
    margin = 3 * env.terms.multiplier * high * rate
    return float(margin + 4 * (margin + 3 * fee))


def discounted_returns(clock, cursors, rewards, gamma=GAMMA):
    """G of every decision of one trajectory that ends at its true terminal."""
    cursors = np.asarray(cursors)
    g, out = float(gamma), np.zeros(len(rewards))
    total = 0.
    for i in range(len(rewards) - 1, -1, -1):
        if i + 1 < len(rewards):
            total *= g ** clock.advance(int(cursors[i]), int(cursors[i + 1]))
        total += float(rewards[i])
        out[i] = total
    return out


class Reference:
    """pi_k's formal trajectory: decision inputs, equity and complete returns
    per cursor, the fork points and the coupling test."""

    def __init__(self, clock, cursor, reward, equity, inputs, action, terminal,
                 floor, gamma=GAMMA, continuations=None):
        cursor = np.asarray(cursor)
        if len(cursor) == 0 or not np.array_equal(cursor, np.arange(cursor[0], cursor[0] + len(cursor))):
            raise ValueError('A reference has one decision per consecutive market cursor')
        self.clock, self.gamma, self.floor = clock, gamma, float(floor)
        self.start, self.n = int(cursor[0]), len(cursor)
        self.reward = np.asarray(reward, np.float64)
        self.equity = np.asarray(equity, np.float64)
        self.action = np.asarray(action, np.int64)
        self.inputs = {k: np.asarray(inputs[k]) for k in INPUTS}
        self.continuations = continuations if continuations is not None else {}
        # Only a trajectory that reached the true terminal has complete returns.
        self.complete = bool(terminal)
        self.g = discounted_returns(clock, cursor, self.reward, gamma) if self.complete else None
        self.future_min_equity = np.minimum.accumulate(self.equity[::-1])[::-1]
        self.flat = self.inputs['position_state'][:, 1] == 0
        # Diagnostic for the funds-safe route ONLY. Never a candidate filter.
        reach = np.flatnonzero(self.future_min_equity >= self.floor)
        self.couple_from = self.start + (int(reach[0]) if len(reach) else self.n)

    def row(self, cursor):
        i = int(cursor) - self.start
        if not 0 <= i < self.n:
            raise ValueError(f'Cursor {cursor} is outside the reference')
        return i

    def forks(self):
        """Cursors where the reference account is flat and may ADD (its last
        decision excepted: no step follows the true terminal)."""
        legal = self.flat & (self.inputs['action_mask'][:, ADD] > 0.5)
        legal[-1] = False
        return self.start + np.flatnonzero(legal)

    def same_inputs(self, cursor, state):
        i = self.row(cursor)
        return all(np.array_equal(np.asarray(state[k]), self.inputs[k][i]) for k in INPUTS)

    def couples(self, state):
        """True when a branch state at state['cursor'] provably continues as
        the reference from there on (module docstring)."""
        return self.coupling_kind(state) is not None

    def coupling_kind(self, state):
        c = int(state['cursor'])
        if not self.start <= c < self.start + self.n:
            return None
        i = c - self.start
        if not (self.flat[i] and np.asarray(state['position_state'])[1] == 0
                and self.same_inputs(c, state)):
            return None
        saved = self.continuations.get(c)
        if self.continuations:
            # The actual production caller records the market-only carry and
            # seed counter. An equal current z/action alone is not a proof of
            # equal future policy decisions.
            if saved is None or not same_state(state.get('policy'), saved['policy']):
                return None
            native = state.get('native')
            if native is not None and saved['native'] is not None and same_state(native, saved['native']):
                return 'EXACT_STATE'
            # Keep the funds-safe proof, but only for otherwise compatible
            # Native continuations. Different prior entry counts are harmless
            # while neither account can hit its daily entry cap in this tail.
            if native is None or saved['native'] is None:
                return None
            controls = []
            for item in (native, saved['native']):
                item = {k: v for k, v in item.items()
                        if k not in ('balances', 'realized_pnl', 'unrealized_pnl')}
                counts = dict(item['counters'])
                counts.pop('entry_sequence')
                used = counts.pop('_entries_used_today')
                limit = item['rules']['max_entries_per_day']
                if limit > 0 and used + self.n - i >= limit:
                    return None
                controls.append(dict(item, counters=counts))
            if not same_state(*controls):
                return None
        gap = float(state['equity']) - self.equity[i]
        if self.future_min_equity[i] + min(gap, 0.) >= self.floor:
            return 'FUNDS_SAFE'
        return None


def unavailable(reference, t, first_action, reason, progress=None):
    """No complete Q within the allocation. An unfinished probe may report
    observed step/reward totals, but never a numeric return or Q label."""
    reference.row(t)
    if progress is not None:
        if (progress.cursor, progress.first_action) != (t, first_action):
            raise ValueError('Budget record and branch progress differ')
        return _result(reference, t, first_action, 'UNAVAILABLE_BUDGET', reason, progress.rewards, None)
    return dict(cursor=int(t), first_action=int(first_action), status='UNAVAILABLE_BUDGET',
                reason=reason, steps=0, coupled_at=None, equity_gap=None,
                branch_rewards=None, g=None)


@dataclass
class BranchProgress:
    """Save together with the Native/policy state, never restart a probe."""
    cursor: int
    first_action: int
    rewards: list = field(default_factory=list)
    done: bool = False
    terminal: bool = False


def run_branch(stepper, reference, first_action, max_steps=8192, witness=None):
    """One counterfactual branch from the stepper's current state.

    stepper.state() -> dict(cursor, position_state, action_mask,
      account_native, equity): the decision inputs before deciding.
    stepper.step(action) -> (reward, done, terminal): execute `action`, or
      the frozen policy's own action when it is None; done ends the episode
      (terminal: at the true terminal).
    witness(cursor) is called once the branch couples and must confirm that
    the policy decides exactly as the reference there (e.g. equal answers).
    """
    progress = BranchProgress(int(stepper.state()['cursor']), int(first_action))
    return advance_branch(stepper, reference, progress, max_steps, max_steps, witness)


def advance_branch(stepper, reference, progress, stop_after, max_steps=8192, witness=None):
    """Advance to an absolute branch step count. None means a paused probe,
    not a label. Resume with the SAME progress and saved stepper state.
    A terminal or coupling on the last allowed step takes precedence over CAP.
    """
    if max_steps < 1 or stop_after < 1:
        raise ValueError('Branch cap and pause step must be positive')
    t, first_action, rewards = progress.cursor, progress.first_action, progress.rewards
    state = stepper.state()
    if int(state['cursor']) != t + len(rewards):
        raise ValueError('Branch progress and Native cursor differ')
    if not reference.complete:
        return _result(reference, t, first_action, 'NOT_COMPARABLE', 'REFERENCE_INCOMPLETE', [], None)
    if not rewards and (t not in set(reference.forks().tolist()) or not reference.same_inputs(t, state)):
        raise ValueError(f'Cursor {t} is not a legal flat decision of the reference state')
    while True:
        if progress.done:
            if progress.terminal:
                return _result(reference, t, first_action, 'VALID_COMPLETE', 'TERMINAL', rewards, None)
            return _result(reference, t, first_action, 'NOT_COMPARABLE', 'DATA_END', rewards, None)
        state = stepper.state()
        if int(state['cursor']) != t + len(rewards):
            raise ValueError('A branch step must advance exactly one market cursor')
        kind = reference.coupling_kind(state) if rewards else None
        if kind is not None:
            c = int(state['cursor'])
            if witness is not None and not witness(c):
                raise RuntimeError(f'The frozen policy decides differently at the coupling cursor {c}')
            gap = float(state['equity']) - reference.equity[reference.row(c)]
            return dict(_result(reference, t, first_action, 'VALID_COMPLETE', 'COUPLED', rewards, c, gap),
                        coupling_kind=kind)
        if len(rewards) >= max_steps:
            return _result(reference, t, first_action, 'NOT_COMPARABLE', 'CAP', rewards, None)
        if len(rewards) >= stop_after:
            return None
        reward, progress.done, progress.terminal = stepper.step(int(first_action) if not rewards else None)
        rewards.append(float(reward))


def _result(reference, t, first_action, status, reason, rewards, coupled, gap=None):
    g = float(reference.gamma)
    clock = reference.clock
    head = sum(g ** clock.advance(t, t + i) * r for i, r in enumerate(rewards))
    value = None
    if status == 'VALID_COMPLETE':
        value = head
        if coupled is not None:
            value += g ** clock.advance(t, coupled) * reference.g[reference.row(coupled)]
    return dict(cursor=t, first_action=int(first_action), status=status, reason=reason,
                steps=len(rewards), coupled_at=coupled, equity_gap=gap,
                branch_rewards=float(sum(rewards)), g=value)


def pair(reference, branch):
    """ADD/WAIT returns of one fork: the reference's own action takes its
    complete return, the other action the branch's (None unless valid)."""
    i = reference.row(branch['cursor'])
    own = int(reference.action[i])
    if branch['first_action'] == own:
        raise ValueError('A branch executes the action the reference did not take')
    g_own = float(reference.g[i]) if reference.complete else None
    g = {own: g_own, branch['first_action']: branch['g']}
    valid = branch['status'] == 'VALID_COMPLETE' and g_own is not None
    return dict(cursor=branch['cursor'], reference_action=own, status=branch['status'],
                reason=branch['reason'], g_add=g[ADD], g_wait=g[WAIT],
                delta_g=g[ADD] - g[WAIT] if valid else None)

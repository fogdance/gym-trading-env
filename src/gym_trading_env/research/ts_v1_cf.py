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

Coupling. Once a branch is flat at a cursor c where the reference is flat
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

Budget. Where the reference stays below the floor for the whole cap window
no cursor of it may couple, and a fork whose true terminal also lies beyond
the cap can end validly only if the branch's own account terminates first
(a Native MARGIN_CALL); only the full branch tells (Reference.decidable).
A caller with a finite compute budget simulates an exact sample of these
forks and gives the others UNAVAILABLE_BUDGET (`unavailable`): not
computed, without reward, return, terminal or label, and without any claim
that their branch could not have ended.
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np

from .fg_single3_oracle import GAMMA

STATUSES = ('VALID_COMPLETE', 'VALID_ESTIMATED', 'NOT_COMPARABLE', 'UNAVAILABLE_BUDGET')
WAIT, ADD = 0, 1
INPUTS = ('position_state', 'action_mask', 'account_native')


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
                 floor, gamma=GAMMA):
        cursor = np.asarray(cursor)
        if len(cursor) == 0 or not np.array_equal(cursor, np.arange(cursor[0], cursor[0] + len(cursor))):
            raise ValueError('A reference has one decision per consecutive market cursor')
        self.clock, self.gamma, self.floor = clock, gamma, float(floor)
        self.start, self.n = int(cursor[0]), len(cursor)
        self.reward = np.asarray(reward, np.float64)
        self.equity = np.asarray(equity, np.float64)
        self.action = np.asarray(action, np.int64)
        self.inputs = {k: np.asarray(inputs[k]) for k in INPUTS}
        # Only a trajectory that reached the true terminal has complete returns.
        self.complete = bool(terminal)
        self.g = discounted_returns(clock, cursor, self.reward, gamma) if self.complete else None
        self.future_min_equity = np.minimum.accumulate(self.equity[::-1])[::-1]
        self.flat = self.inputs['position_state'][:, 1] == 0
        # The first cursor where a branch may couple (start + n when none may):
        # couples() needs future_min_equity >= floor, which never decreases.
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
        c = int(state['cursor'])
        if not self.start <= c < self.start + self.n:
            return False
        i = c - self.start
        if not (self.flat[i] and np.asarray(state['position_state'])[1] == 0
                and self.same_inputs(c, state)):
            return False
        gap = float(state['equity']) - self.equity[i]
        return self.future_min_equity[i] + min(gap, 0.) >= self.floor

    def decidable(self, t, max_steps):
        """False when a branch from fork t can neither couple nor reach the
        true terminal within max_steps (module docstring, Budget)."""
        end, last = self.start + self.n, int(t) + max_steps
        return (self.couple_from < end and self.couple_from <= last) or end <= last


def unavailable(reference, t, first_action, reason):
    """The record of a fork not computed within the budget: no reward,
    return, terminal or label."""
    reference.row(t)
    return dict(cursor=int(t), first_action=int(first_action), status='UNAVAILABLE_BUDGET',
                reason=reason, steps=0, coupled_at=None, equity_gap=None,
                branch_rewards=None, g=None)


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
    state = stepper.state()
    t = int(state['cursor'])
    if not reference.complete:
        return _result(reference, t, first_action, 'NOT_COMPARABLE', 'REFERENCE_INCOMPLETE', [], None)
    if t not in set(reference.forks().tolist()) or not reference.same_inputs(t, state):
        raise ValueError(f'Cursor {t} is not a legal flat decision of the reference state')
    rewards = []
    reward, done, terminal = stepper.step(int(first_action))
    rewards.append(float(reward))
    while True:
        if done:
            if terminal:
                return _result(reference, t, first_action, 'VALID_COMPLETE', 'TERMINAL', rewards, None)
            return _result(reference, t, first_action, 'NOT_COMPARABLE', 'DATA_END', rewards, None)
        state = stepper.state()
        if int(state['cursor']) != t + len(rewards):
            raise ValueError('A branch step must advance exactly one market cursor')
        if reference.couples(state):
            c = int(state['cursor'])
            if witness is not None and not witness(c):
                raise RuntimeError(f'The frozen policy decides differently at the coupling cursor {c}')
            gap = float(state['equity']) - reference.equity[reference.row(c)]
            return _result(reference, t, first_action, 'VALID_COMPLETE', 'COUPLED', rewards, c, gap)
        if len(rewards) >= max_steps:
            return _result(reference, t, first_action, 'NOT_COMPARABLE', 'CAP', rewards, None)
        reward, done, terminal = stepper.step(None)
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

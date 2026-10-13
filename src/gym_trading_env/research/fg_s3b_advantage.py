"""TRAIN-only conditional account proofs over the frozen C1 Oracle candidates.

No teacher search, new matching engine, or change to the original Oracle.
EXACT_Q means the relaxed optimum is attained by a funded witness; numerical
values remain outward rounded intervals, never midpoint pseudo-exact values.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
import resource
import time
import uuid

import numpy as np

from .fg_replay import check_files, digest, iter_sidecar, read_manifest, signed, write_json
from .fg_s3a_labels import INDEX_DTYPE, load_labels, sources
from .fg_single3 import load_env
from .fg_single3_oracle import GAMMA, OBJECTIVE, _add, _down, _mul, _up, funds_certificate
from gym_trading_env.rewards.single3_reward import Single3RewardConfig

PROTOCOL = 'FG_S3B_CERTIFIED_ADVANTAGE_V1'
DEFAULT_PROOF = dict(exact_comparisons=4096, exact_events=8192)
VALUE_DTYPE = np.dtype([
    ('q_add', '<f8', (2,)), ('q_wait', '<f8', (2,)), ('delta', '<f8', (2,)),
    ('has_intervals', '?'), ('delta_refined', '?'), ('margin_lower', '<f8'),
    ('certificate', 'U16'), ('proof', 'U24'), ('branch', 'U20'), ('reason', 'U48'),
    ('cash_start', 'S40'), ('cash_after', 'S40')])


def code_identity():
    """Bind the real execution/reward/identity dependencies, not just this file."""
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), Path(__file__).with_name('fg_s3a_labels.py'),
             Path(__file__).with_name('fg_replay.py'),
             Path(__file__).with_name('fg_single3.py'),
             Path(__file__).with_name('fg_single3_oracle.py'),
             Path(__file__).with_name('fg_reward_candidates.py')]
    for name in ('envs', 'utils', 'rewards'):
        paths.extend((root/'src/gym_trading_env'/name).rglob('*.py'))
    return {str(p.relative_to(root)): digest(p) for p in sorted(set(paths))}


class ExactBudget:
    def __init__(self, comparisons=4096, events=8192):
        if comparisons < 0 or events < 1:
            raise ValueError('Invalid exact arithmetic budget')
        self.limit, self.events = comparisons, events
        self.used = self.exhausted = 0

    def difference(self, left, right, origin, clock):
        if self.used >= self.limit:
            self.exhausted += 1
            return None
        self.used += 1
        coefficients = Counter()
        count = 0
        for route, sign in ((left, 1), (right, -1)):
            for step, reward in route:
                count += 1
                if count > self.events:
                    self.exhausted += 1
                    return None
                coefficients[step] += sign*Fraction.from_float(float(reward))
        return sum((GAMMA**clock.reward_exponent(origin,s)*v for s, v in coefficients.items() if v), Fraction(0))


def rational_interval(value):
    f = float(value)
    if not math.isfinite(f):
        raise ValueError('Nonfinite rational value')
    q = Fraction.from_float(f)
    return (f if q <= value else _down(f), f if q >= value else _up(f))


def validate_candidate(row, end):
    t = row.get('cursor')
    if not isinstance(t, int) or not 0 <= t < end:
        raise ValueError('Candidate cursor outside TRAIN horizon')
    if not row.get('filled'):
        return
    if not row.get('eligible') or row.get('entry_cursor') != t+1:
        raise ValueError('Candidate entry timing/eligibility mismatch')
    event = row.get('exit')
    u = row.get('release_cursor')
    if event:
        if (not isinstance(u, int) or u < t+1 or row.get('exit_cursor') != u
                or row.get('right_censored') or not event.get('committed')
                or event.get('closed_quantity') != 3):
            raise ValueError('Candidate full atomic exit/release mismatch')
        cash = Decimal(event['net_trade_pnl'])
        if not cash.is_finite():
            raise ValueError('Nonfinite candidate cash')
        if event['reason'] == 'TAKE_PROFIT' and cash <= 0:
            raise ValueError('TP-only dominance requires strictly positive net cash')
    elif not row.get('right_censored') or u is not None:
        raise ValueError('Unclosed candidate must remain censored')
    previous = t
    for s, reward in row['rewards']['C1']:
        if (not isinstance(s, int) or s <= previous or s > row['observed_until_cursor']
                or not math.isfinite(reward) or float(np.float32(reward)) != reward):
            raise ValueError('Sparse reward timing/float32 contract mismatch')
        if (not event or event['reason'] != 'TAKE_PROFIT') and reward > 0:
            raise ValueError('Non-TP candidate contradicts nonpositive dominance proof')
        previous = s


class RelaxedValues:
    """O(N) suffix bounds with feasible TP-only witnesses and capped exact work."""
    def __init__(self, candidates, end, *, clock, budget=None):
        if not isinstance(end, int) or end < 1:
            raise ValueError('Invalid TRAIN horizon')
        self.end = end
        self.clock = clock
        clock._cursor(end)
        candidates = list(candidates)
        self.budget = budget or ExactBudget()
        self.rows = {}
        for row in candidates:
            validate_candidate(row, end)
            if row['cursor'] in self.rows:
                raise ValueError('Duplicate candidate cursor')
            self.rows[row['cursor']] = row
        if set(self.rows) != set(range(end)):
            raise ValueError('Frozen candidate stream is incomplete')
        self.gamma_interval = rational_interval(GAMMA)
        # Include late global liquidation after the flat decision horizon.
        last = max([end]+[r.get('observed_until_cursor', end) for r in candidates])
        self.powers = [(1., 1.)]
        for _ in range(int(clock.prefix[-1])):
            self.powers.append(_mul(self.powers[-1], self.gamma_interval))
        self.bounds = np.zeros((end+1, 2), np.float64)
        self.solved = np.ones(end+1, bool)
        self.head = np.full(end+1, -1, np.int64)
        self.next_trade = {}
        self.local = {}
        for t in range(end-1, -1, -1):
            wait = _mul(self.powers[clock.advance(t,t+1)], self.bounds[t+1])
            best, head, solved = wait, int(self.head[t+1]), bool(self.solved[t+1])
            row = self.rows[t]
            if (row.get('filled') and row.get('exit')
                    and row['exit']['reason'] == 'TAKE_PROFIT'
                    and row['release_cursor'] <= end):
                local = self.local_value(row)
                # Keeping a nonpositive TP option is safe; prune only with a
                # proven upper bound. Negative TP intervals are dominated by WAIT.
                if local[1] > 0:
                    u = row['release_cursor']
                    option = _add(local, _mul(self.powers[clock.advance(t,u)], self.bounds[u]))
                    take = option[0] > wait[1]
                    reject = option[1] < wait[0]
                    diff = None
                    if not take and not reject and self.solved[u] and self.solved[t+1]:
                        diff = self.route_difference(t, int(self.head[u]), head, t)
                        if diff is not None:
                            take, reject = diff > 0, diff <= 0  # original WAIT tie rule
                    if take:
                        best, head, solved = option, t, bool(self.solved[u])
                        self.next_trade[t] = int(self.head[u])
                    elif not reject:
                        # Budget exhaustion or uncertain continuations: retain
                        # max upper bound and a genuinely feasible lower witness.
                        if option[0] > wait[0]:
                            head = t
                            self.next_trade[t] = int(self.head[u])
                        best = (max(option[0], wait[0]), max(option[1], wait[1]))
                        solved = False
            self.bounds[t] = (max(0., best[0]), max(0., best[1]))
            self.head[t], self.solved[t] = head, solved

    def local_value(self, row):
        t = row['cursor']
        if t not in self.local:
            total = (0., 0.)
            for s, reward in row['rewards']['C1']:
                total = _add(total, _mul(self.powers[self.clock.reward_exponent(t,s)], (reward, reward)))
            self.local[t] = total
        return self.local[t]

    def chain(self, head):
        while head >= 0:
            yield head
            head = self.next_trade[head]

    def route_difference(self, add, add_head, wait_head, origin):
        # Cancel the identical suffix without materializing full schedules.
        # Route traversal itself is bounded, including zero-reward trades.
        if self.budget.used >= self.budget.limit:
            self.budget.exhausted += 1
            return None
        if add_head == wait_head:
            return self.budget.difference(self.rows[add]['rewards']['C1'], [], origin, self.clock)
        cap = self.budget.events
        left, right = [], []
        for route, head in ((left, add_head), (right, wait_head)):
            for t in self.chain(head):
                route.append(t)
                if len(route) > cap:
                    self.budget.exhausted += 1
                    return None
        left = [add]+left
        right_set = set(right)
        common = next((t for t in left if t in right_set), None)
        def events(route):
            for t in route:
                if t == common:
                    break
                yield from self.rows[t]['rewards']['C1']
        return self.budget.difference(events(left), events(right), origin, self.clock)


def funding_terms(env):
    """A new one-forced-interval lemma, conditional on the ACTUAL P4 cash."""
    if not env.single3 or env.reward_config != Single3RewardConfig.for_candidate('C1'):
        raise ValueError('C1 Single3 execution/reward contract required')
    original = funds_certificate(env)
    p = env.terms.at(env._raw.index[0])
    lo = Decimal(str(env._raw.Low.min()))-p['spread']-p['slippage']-env.terms.tick
    hi = Decimal(str(env._raw.High.max()))+p['spread']+p['slippage']+env.terms.tick
    if lo <= 0 or hi < lo or p['margin_rate'] <= 0:
        raise ValueError('Unproved quote/margin envelope')
    loss = 3*env.terms.multiplier*(hi-lo)+3*(p['open_fee']+max(p['close_fee'], p['close_today_fee']))
    required = loss+3*env.terms.multiplier*hi*p['margin_rate']
    if str(required) != original['required_equity_upper_bound']:
        raise ValueError('Original/new funding envelope differs')
    return dict(original_certificate=original, quote_low=str(lo), quote_high=str(hi),
                one_interval_loss_upper=str(loss), required_cash=str(required),
                proof='actual_cash_one_forced_interval_then_TP_only_v1',
                arbitrary_losing_paths_funds_irrelevant=False)


def certify_advantage(q_add, q_wait, teacher_action=None, *, exact_difference=None):
    """Only signed OUTWARD endpoints determine a positive certified margin."""
    for q in (q_add, q_wait):
        if len(q) != 2 or not all(math.isfinite(x) for x in q) or q[0] > q[1]:
            raise ValueError('Invalid value enclosure')
    delta = (_down(q_add[0]-q_wait[1]), _up(q_add[1]-q_wait[0]))
    if exact_difference is not None:
        exact = rational_interval(exact_difference)
        if exact[1] < delta[0] or exact[0] > delta[1]:
            raise ValueError('Exact difference contradicts interval')
        delta = (max(delta[0], exact[0]), min(delta[1], exact[1]))
    if delta[0] > 0:
        status, margin, label = 'ADD_BETTER', delta[0], 1
    elif delta[1] < 0:
        status, margin, label = 'WAIT_BETTER', -delta[1], 0
    elif exact_difference == 0:
        status, margin, label = 'TIE_EXACT', 0., 0
        delta = (0., 0.)
    else:
        status, margin, label = 'UNCERTIFIED', 0., None
    if teacher_action is not None and label is not None and teacher_action != label:
        raise ValueError(f'Teacher conflict: {status} but P4 action={teacher_action}')
    return dict(certificate=status, margin_lower=margin, delta=list(delta))


def build_value_evidence(row, relaxed, actual_cash, required_cash, *, teacher_action=None,
                         branch_verified=False, rejection_equivalent=False):
    """No assumptions about an arbitrary loss path; UNKNOWN has null QADD."""
    t = row['cursor']
    cash, bound = Decimal(str(actual_cash)), Decimal(str(required_cash))
    if not cash.is_finite() or not bound.is_finite() or bound <= 0:
        raise ValueError('Invalid account evidence')
    result = dict(q_add=None, q_wait=None, delta=None, margin_lower=0.,
                  certificate='UNCERTIFIED', proof='UNCERTIFIED', branch='UNKNOWN',
                  reason='START_FUNDS_UNPROVEN', cash_start=str(cash), cash_after=None)
    wait = _mul(relaxed.powers[relaxed.clock.advance(t,t+1)], relaxed.bounds[t+1])
    wait_funded = cash > bound
    # At low cash, WAIT-forever is feasible but the relaxed optimum is only an upper bound.
    result['q_wait'] = list(wait if wait_funded else (0., wait[1]))
    if not row.get('eligible'):
        result['reason'] = 'CANDIDATE_NOT_ELIGIBLE'
        return result
    if not row.get('filled'):
        result['branch'] = 'REJECTED'
        independent_rejects = {'DIRECTION_OR_DEADLINE_CHANGED', 'UNVERIFIED_BAR_TIME',
                               'INSUFFICIENT_BAR_VOLUME'}
        equivalent = rejection_equivalent or (wait_funded and row.get('reject_reason') in independent_rejects)
        if not equivalent:
            result['reason'] = 'REJECTION_STATE_UNPROVEN'
            return result
        result.update(q_add=result['q_wait'], proof='TIE_EXACT', reason='NO_FILL_STATE_EQUIVALENT')
        result.update(certify_advantage(result['q_add'], result['q_wait'], teacher_action,
                                       exact_difference=Fraction(0)))
        return result
    result['branch'] = row['exit']['reason'] if row.get('exit') else 'CENSORED'
    if not row.get('exit'):
        result.update(proof='CENSORED', reason='UNRELEASED_POSITION')
        return result
    if not wait_funded and not branch_verified:
        return result
    u = row['release_cursor']
    terminal = row['exit']['reason'] in ('MAIN_ROLL', 'CONTRACT_EXPIRY', 'MARGIN_CALL')
    if u > relaxed.end and not terminal:
        result.update(proof='CENSORED', reason='RELEASE_OUTSIDE_PROVEN_HORIZON')
        return result
    after = cash+Decimal(row['exit']['net_trade_pnl'])
    result['cash_after'] = str(after)
    local = relaxed.local_value(row)
    suffix = (0., 0.) if terminal or u >= relaxed.end else tuple(relaxed.bounds[u])
    funded = terminal or u >= relaxed.end or after > bound
    lower_suffix = suffix[0] if funded else 0.
    add = _add(local, _mul(relaxed.powers[relaxed.clock.advance(t,u)], (lower_suffix, suffix[1])))
    exact_q = (wait_funded and relaxed.solved[t+1]
               and funded and (terminal or u >= relaxed.end or relaxed.solved[u]))
    result.update(q_add=list(add), proof='EXACT_Q' if exact_q else 'CERTIFIED_SIGN_ONLY',
                  reason='FUNDED_RELAXED_WITNESS' if exact_q else 'WAIT_FOREVER_LOWER_RELAXED_UPPER')
    diff = None
    delta = (_down(add[0]-result['q_wait'][1]), _up(add[1]-result['q_wait'][0]))
    if exact_q and delta[0] <= 0 <= delta[1]:
        add_head = -1 if terminal or u >= relaxed.end else int(relaxed.head[u])
        diff = relaxed.route_difference(t, add_head, int(relaxed.head[t+1]), t)
        if diff is None:
            result['reason'] = 'EXACT_COMPARISON_BUDGET_EXHAUSTED'
    result.update(certify_advantage(add, result['q_wait'], teacher_action, exact_difference=diff))
    if diff is not None:
        result['delta_exact_rational'] = dict(numerator_hex=hex(diff.numerator),
                                              denominator_hex=hex(diff.denominator))
    if result['certificate'] == 'TIE_EXACT':
        result['proof'] = 'TIE_EXACT'
    elif not exact_q and result['certificate'] == 'UNCERTIFIED':
        result['proof'] = 'UNCERTIFIED'
    return result


def verify_counterfactual(env, state, row, *, continuation=()):
    """Fork a FULL actual account snapshot, not Oracle's fresh reset helper."""
    env.import_state(state)
    if env.entries or env.pending_reason or env.terminated or env.truncated or not env._get_obs()['action_mask'].all():
        raise ValueError('Counterfactual requires the actual active legal empty state')
    t = env.current_step
    if t != row['cursor']:
        raise ValueError('Counterfactual cursor differs from candidate')
    sparse, new_exit = [], None
    old_exits, old_fills = len(env.attributions), len(env.fills)
    _, reward, _, _, _ = env.step(1)
    if reward:
        sparse.append([env.current_step, float(np.float32(reward))])
    opened = [f for f in env.fills[old_fills:] if f['reason'] == 'ENTRY']
    reject = env.reject_reason
    while env.entries and not env.terminated and env.status not in ('DATA_END', 'DATA_GAP'):
        if env.truncated:
            env.reset()
        previous = env.current_step
        _, reward, _, _, _ = env.step(0)
        if reward:
            sparse.append([env.current_step, float(np.float32(reward))])
        if env.current_step == previous:
            break
    if len(env.attributions) > old_exits:
        new_exit = env.attributions[-1]
    if bool(opened) != bool(row.get('filled')) or sparse != row.get('rewards', {}).get('C1', []):
        raise ValueError('Actual account/candidate fill or C1 reward mismatch')
    if opened:
        if (opened[0]['price'] != row['entry_price'] or opened[0]['time'] != row['entry_time']
                or bool(new_exit) != bool(row.get('exit'))):
            raise ValueError('Actual account/candidate price or exit mismatch')
        if new_exit:
            if env.current_step != row['release_cursor']:
                raise ValueError('Actual account release cursor mismatch')
            for key, value in row['exit'].items():
                if key != 'entry_id' and new_exit[key] != value:
                    raise ValueError('Actual counterfactual exit mismatch: '+key)
    elif reject != row.get('reject_reason'):
        raise ValueError('Actual counterfactual rejection differs')
    add_state = env.export_state()
    equivalent = False
    if not opened:
        env.import_state(state)
        _, reward, _, _, _ = env.step(0)
        wait_state = env.export_state()
        # These diagnostic counters never enter execution/reward eligibility.
        ignore = {'legal_add_decisions', 'voluntary_waits'}
        equivalent = reward == 0 and {k:v for k,v in add_state.items() if k not in ignore} == {
            k:v for k,v in wait_state.items() if k not in ignore}
        if not equivalent:
            raise ValueError('Unfilled ADD and WAIT do not have equivalent value states')
        env.import_state(add_state)
    local_cash = str(env.user_accounts.cash_balance.get_balance()) if not env.entries else None
    suffix = set(continuation)
    actual_add = []
    suffix_rewards = []
    while suffix and not env.terminated and env.status not in ('DATA_END', 'DATA_GAP'):
        if env.truncated:
            env.reset()
        previous = env.current_step
        action = int(previous in suffix)
        if action and not env._get_obs()['action_mask'][1]:
            raise ValueError('Funded relaxed suffix witness encountered a restricted real mask')
        before_fills = len(env.fills)
        before_exits = len(env.attributions)
        _, reward, _, _, _ = env.step(action)
        new_entries = [f for f in env.fills[before_fills:] if f['reason'] == 'ENTRY']
        if action:
            if len(new_entries) != 1:
                raise ValueError('Funded relaxed suffix witness ADD was not actually filled')
            actual_add.append(previous)
        if any(e['reason'] != 'TAKE_PROFIT' or Decimal(e['net_trade_pnl']) <= 0
               for e in env.attributions[before_exits:]):
            raise ValueError('Funded relaxed suffix witness is not cash-positive TP-only')
        if reward:
            suffix_rewards.append([env.current_step, float(np.float32(reward))])
        if env.current_step == previous:
            break
    if suffix and env.entries:
        raise ValueError('Funded relaxed suffix witness has an unreleased position')
    if suffix and actual_add != sorted(suffix):
        raise ValueError('Funded relaxed suffix witness was not executed')
    result = dict(status='PASS', cursor=t, filled=bool(opened), exit_cursor=row.get('release_cursor'),
                  local_rewards=sparse, cash_after=local_cash, rejection_equivalent=equivalent,
                  suffix_rewards=suffix_rewards, suffix_add=actual_add,
                  full_actual_account_snapshot=True)
    env.import_state(state)
    return result


def _pack(evidence):
    out = np.zeros((), VALUE_DTYPE)
    for name in ('certificate', 'proof', 'branch', 'reason', 'margin_lower', 'cash_start'):
        value = evidence[name]
        if VALUE_DTYPE[name].kind in 'SU' and len(str(value)) > VALUE_DTYPE[name].itemsize//(4 if VALUE_DTYPE[name].kind == 'U' else 1):
            raise ValueError('Evidence field would be truncated: '+name)
        out[name] = value
    if evidence['cash_after'] is not None:
        out['cash_after'] = evidence['cash_after']
    out['has_intervals'] = evidence['q_add'] is not None
    out['delta_refined'] = 'delta_exact_rational' in evidence
    if out['has_intervals']:
        out['q_add'], out['q_wait'], out['delta'] = evidence['q_add'], evidence['q_wait'], evidence['delta']
    return out


def _account_cash(trace, labels, candidates, plan):
    """Replay-derived empty cash; verify every selected fill and final ledger."""
    cash = Decimal(300000)
    active, closed, opened = {}, set(), []
    by_cursor = {r['cursor']: r for r in candidates}
    expected_rewards = {s:v for t in plan['add_cursors'] for s,v in by_cursor[t]['rewards']['C1']}
    result = []
    pos = 0
    for row in iter_sidecar(trace):
        for fill in row['fills']:
            if fill['reason'] == 'ENTRY':
                entry = fill['entry_id']
                if not row['advanced'] or entry in active or entry in closed:
                    raise ValueError('Repeated or unexecuted actual account fill')
                active[entry] = (fill, row['cursor']-1)
                opened.append(row['cursor']-1)
        for event in row['exits']:
            if event['entry_id'] not in active or event['entry_id'] in closed:
                raise ValueError('Actual account close identity mismatch')
            start, cursor = active.pop(event['entry_id'])
            candidate = by_cursor[cursor]
            if event['reason'] != 'TAKE_PROFIT' or Decimal(event['net_trade_pnl']) <= 0:
                raise ValueError('P4 actual account is not cash-positive TP-only')
            if start['price'] != candidate['entry_price'] or row['cursor'] != candidate['release_cursor']:
                raise ValueError('Actual P4 fill/release differs from frozen candidate')
            for key, value in candidate['exit'].items():
                if key != 'entry_id' and event[key] != value:
                    raise ValueError('Actual P4 attribution mismatch: '+key)
            closed.add(event['entry_id'])
            cash += Decimal(event['net_trade_pnl'])
        if row['advanced'] and float(np.float32(row['official_reward'])) != expected_rewards.get(row['cursor'], 0.):
            raise ValueError('Actual P4 incoming C1 feedback differs from frozen plan')
        if row['lots_after'] != 3*len(active):
            raise ValueError('Actual account position conservation mismatch')
        if pos < len(labels) and row['ordinal'] == int(labels[pos]['ordinal']):
            label = labels[pos]
            if (active or row['lots_after'] or row['action'] != int(label['label'])
                    or row['cursor'] != int(label['cursor']) or row['stepid'] != label['stepid'].tobytes().hex()
                    or not row['legal_add'] or not row['decision_executed'] or row['is_last']):
                raise ValueError('S3-A index/actual account identity mismatch')
            result.append(cash)
            pos += 1
    state = json.loads((Path(trace)/'account_state.json').read_text())
    if (pos != len(labels) or active or state['entries'] or Decimal(state['balances']['user_margin']) != 0
            or Decimal(state['balances']['user_cash']) != cash or Decimal(state['unrealized_pnl']) != 0
            or opened != plan['add_cursors']):
        raise ValueError('Actual P4 Decimal cash/final account reconciliation failed')
    return result, dict(status='PASS', initial_cash='300000', final_cash=str(cash),
                        minimum_supervised_cash=str(min(result)), closed_trades=len(closed),
                        all_selected_fills_feedback_and_attributions_matched=True,
                        account_state_sha256=digest(Path(trace)/'account_state.json'))


def _input_identity(dataset, oracle_receipt, labels):
    lm = load_labels(labels, verify_sources=False)
    src = sources(dataset, oracle_receipt)
    traces = src.pop('traces')
    if len(traces) != len(lm['traces']) or lm['source'] != src or lm['traces'] != [dict(t, **{k:rec[k] for k in
            ('labels_file', 'samples', 'counts')}) for t, rec in zip(traces, lm['traces'])]:
        raise ValueError('S3-A label source differs from requested TRAIN/Oracle identity')
    return lm


def build_values(dataset, oracle_receipt, labels, output, *, seed=202, resume=False,
                 contracts=None, limit=None, proof_config=None, config_sha256=None):
    lm = _input_identity(dataset, oracle_receipt, labels)
    cfg = dict(DEFAULT_PROOF, **(proof_config or {}))
    if set(cfg) != set(DEFAULT_PROOF) or (limit is not None and limit < 1):
        raise ValueError('Unknown proof settings or invalid prefix')
    selected = [r for r in lm['traces'] if contracts is None or r['contract'] in contracts]
    if not selected or (contracts is not None and set(contracts) != {r['contract'] for r in selected}):
        raise ValueError('Non-TRAIN contract selection')
    root = Path(output).resolve()
    for source in (dataset, oracle_receipt, labels):
        p = Path(source).resolve()
        if root.is_relative_to(p) or p.is_relative_to(root):
            raise ValueError('Output overlaps immutable input')
    identity = dict(schema=PROTOCOL, dataset=str(Path(dataset).resolve()), oracle_receipt=str(Path(oracle_receipt).resolve()),
        s3a_labels=str(Path(labels).resolve()), s3a_labels_sha256=digest(Path(labels)/'manifest.json'),
        dataset_sha256=lm['source']['dataset_sha256'], oracle_sha256=lm['source']['oracle_sha256'],
        training_hash=lm['source']['training_hash'], reward_params_hash=lm['source']['reward_params_hash'],
        gamma_rational=[332,333], objective=OBJECTIVE, code_sha256=code_identity(), seed=seed,
        contracts=[r['contract'] for r in selected], limit=limit, proof_config=cfg, config_sha256=config_sha256)
    if resume:
        if read_manifest(root/'run_identity.json')['identity'] != identity:
            raise ValueError('Values resume identity mismatch')
        if (root/'manifest.json').exists():
            return audit_values(root)
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root/'run_identity.json', signed(dict(identity=identity)))
    inputs = read_manifest(Path(dataset)/'inputs.json')
    view = {r['contract']:r for r in inputs['training_views']['C1']['rows']}
    om = read_manifest(Path(oracle_receipt)/'manifest.json')
    shards = []
    for rec in selected:
        target = root/'contracts'/rec['contract']
        if (target/'manifest.json').exists():
            sm = read_manifest(target/'manifest.json')
            if sm['identity'] != identity:
                raise ValueError('Completed contract identity mismatch')
            check_files({str(target/p):h for p,h in sm['files'].items()})
            check_files(sm['frozen_inputs'])
            shards.append(sm)
            continue
        began = time.monotonic()
        pending = root/'contracts'/('.pending_'+rec['contract']+'_'+uuid.uuid4().hex)
        pending.mkdir(parents=True)
        trace = Path(dataset)/rec['directory']
        tm = read_manifest(trace/'manifest.json')
        pp = Path(tm['policy']['schedule_path']).resolve()
        plan = read_manifest(pp)
        cp = pp.parent/'candidates.jsonl.gz'
        contract_receipt = pp.parent/'manifest.json'
        frozen = {str(cp):plan['candidates_sha256'], str(pp):tm['policy']['schedule_sha256']}
        for path in (cp, contract_receipt):
            if not path.is_relative_to(Path(oracle_receipt).resolve()) or om['files'].get(
                    str(path.relative_to(Path(oracle_receipt).resolve()))) != digest(path):
                raise ValueError('Candidate/contract receipt not bound to frozen Oracle')
            frozen[str(path)] = digest(path)
        check_files(frozen)
        cm = read_manifest(contract_receipt)
        if cm['contract'] != rec['contract'] or cm['status'] != 'PASS' or cm['identity'] != plan['identity']:
            raise ValueError('Foreign Oracle contract receipt')
        with gzip.open(cp, 'rt') as stream:
            candidates = [json.loads(line) for line in stream]
        env = load_env(view[rec['contract']], lm['source']['frozen_inputs'])
        terms = funding_terms(env)
        env.close()
        if terms['original_certificate'] != plan['funds_certificate'] or terms['original_certificate'] != cm['funds_certificate']:
            raise ValueError('Current/static original funding proof differs from frozen Oracle')
        dp_began = time.monotonic()
        relaxed = RelaxedValues(candidates, cm['flat_end_cursor'],
                                clock=env.market_clock_counter,
                                budget=ExactBudget(cfg['exact_comparisons'], cfg['exact_events']))
        dp_seconds = time.monotonic()-dp_began
        if relaxed.solved[0] and list(relaxed.chain(int(relaxed.head[0]))) != plan['add_cursors']:
            raise ValueError('Suffix proof disagrees with frozen teacher schedule')
        if relaxed.bounds[0, 1] < plan['value_interval'][0] or relaxed.bounds[0, 0] > plan['value_interval'][1]:
            raise ValueError('Relaxed value contradicts original initial certificate')
        index = np.load(Path(labels)/rec['labels_file'], allow_pickle=False)
        cash, account = _account_cash(trace, index, candidates, plan)
        take = len(index) if limit is None else min(limit, len(index))
        values = np.empty(take, VALUE_DTYPE)
        refinements = {}
        for i in range(take):
            a = index[i]
            evidence = build_value_evidence(relaxed.rows[int(a['cursor'])], relaxed, cash[i], terms['required_cash'],
                                            teacher_action=int(a['label']))
            values[i] = _pack(evidence)
            if 'delta_exact_rational' in evidence:
                refinements[str(i)] = evidence['delta_exact_rational']
        np.save(pending/'index.npy', index[:take], allow_pickle=False)
        np.save(pending/'values.npy', values, allow_pickle=False)
        np.save(pending/'suffix_bounds.npy', relaxed.bounds, allow_pickle=False)
        np.save(pending/'suffix_solved.npy', relaxed.solved, allow_pickle=False)
        np.save(pending/'suffix_head.npy', relaxed.head, allow_pickle=False)
        write_json(pending/'suffix_next.json', {str(k):v for k,v in relaxed.next_trade.items()})
        write_json(pending/'delta_refinements.json', refinements)
        summary = value_summary(index[:take], values)
        sm = signed(dict(identity=identity, contract=rec['contract'], trace_sha256=rec['manifest_sha256'],
            trace_id=rec['replay_trace_hash'], oracle_plan_sha256=digest(pp), candidates_sha256=digest(cp),
            label_file_sha256=digest(Path(labels)/rec['labels_file']), rows=take,
            full_contract=take == len(index), funding=terms, actual_account=account, summary=summary,
            dp_seconds=dp_seconds, elapsed_seconds=time.monotonic()-began,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            exact_comparisons=relaxed.budget.used, budget_exhausted=relaxed.budget.exhausted,
            frozen_inputs=frozen, files={p.name:digest(p) for p in pending.iterdir()}))
        write_json(pending/'manifest.json', sm)
        (pending/'done').write_text('')
        pending.rename(target)
        shards.append(sm)
        print(f"values {rec['contract']}: {take}, {summary['certificates']}, {sm['elapsed_seconds']:.2f}s", flush=True)
    check_files(lm['source']['frozen_inputs'])
    files = {str(p.relative_to(root)):digest(p) for p in (root/'contracts').rglob('*') if p.is_file() and '.pending_' not in str(p)}
    files['run_identity.json'] = digest(root/'run_identity.json')
    result = signed(dict(protocol=PROTOCOL, status='PASS', identity=identity,
        complete=len(selected) == len(lm['traces']) and limit is None,
        same_supervision_index_as_s3a=len(selected) == len(lm['traces']) and limit is None,
        shards=[dict(contract=s['contract'], directory='contracts/'+s['contract'], rows=s['rows'], summary=s['summary']) for s in shards],
        files=files, source_frozen_inputs=lm['source']['frozen_inputs']))
    write_json(root/'manifest.json', result)
    (root/'done').write_text('')
    return audit_values(root)


def value_summary(index, values):
    valid = values['has_intervals']
    widths = values['delta'][valid,1]-values['delta'][valid,0]
    return dict(rows=len(index), certificates=dict(Counter(values['certificate'].tolist())),
        proofs=dict(Counter(values['proof'].tolist())), branches=dict(Counter(values['branch'].tolist())),
        reasons=dict(Counter(values['reason'].tolist())),
        certified_positive=int((values['margin_lower'] > 0).sum()),
        certified_ADD=int(((values['margin_lower'] > 0) & (index['label'] == 1)).sum()),
        certified_WAIT=int(((values['margin_lower'] > 0) & (index['label'] == 0)).sum()),
        maximum_delta_interval_width=float(widths.max()) if len(widths) else None,
        maximum_absolute_value=float(np.max(np.abs(values['q_add'][valid]))) if valid.any() else None)


def audit_values(directory, *, verify_sources=True):
    root = Path(directory).resolve()
    m = read_manifest(root/'manifest.json')
    if (m['protocol'] != PROTOCOL or m['status'] != 'PASS' or not (root/'done').is_file()
            or m['identity']['code_sha256'] != code_identity()
            or m['identity']['gamma_rational'] != [332,333]):
        raise ValueError('Foreign/incomplete value proof implementation identity')
    check_files({str(root/p):h for p,h in m['files'].items()})
    labels = Path(m['identity']['s3a_labels'])
    if digest(labels/'manifest.json') != m['identity']['s3a_labels_sha256']:
        raise ValueError('S3-A labels identity changed')
    lm = load_labels(labels, verify_sources=verify_sources)
    if verify_sources:
        check_files(m['source_frozen_inputs'])
    by_contract = {r['contract']:r for r in lm['traces']}
    total = 0
    for shard in m['shards']:
        p = root/shard['directory']
        if not p.resolve().is_relative_to(root):
            raise ValueError('Proof shard path escape')
        cm = read_manifest(p/'manifest.json')
        if cm['identity'] != m['identity'] or not (p/'done').is_file():
            raise ValueError('Proof contract identity/incomplete publication')
        if verify_sources:
            check_files(cm['frozen_inputs'])
        a = np.load(p/'index.npy', mmap_mode='r', allow_pickle=False)
        v = np.load(p/'values.npy', mmap_mode='r', allow_pickle=False)
        refinements = json.loads((p/'delta_refinements.json').read_text())
        rec = by_contract[shard['contract']]
        expected = np.load(labels/rec['labels_file'], mmap_mode='r', allow_pickle=False)
        if (a.dtype != INDEX_DTYPE or v.dtype != VALUE_DTYPE or a.shape != (shard['rows'],)
                or v.shape != a.shape or not np.array_equal(a, expected[:len(a)])
                or cm['trace_sha256'] != rec['manifest_sha256'] or cm['trace_id'] != rec['replay_trace_hash']
                or (m['complete'] and len(a) != len(expected))):
            raise ValueError('Same S3-A supervision index proof failed')
        if set(refinements) != {str(i) for i in np.flatnonzero(v['delta_refined'])}:
            raise ValueError('Exact difference evidence set differs')
        for i, (row, value) in enumerate(zip(a, v)):
            if value['has_intervals']:
                tie = value['certificate'] == 'TIE_EXACT'
                exact = Fraction(0) if tie else None
                if value['delta_refined']:
                    r = refinements[str(i)]
                    exact = Fraction(int(r['numerator_hex'],16), int(r['denominator_hex'],16))
                # Audit exact tie without pretending unknown values equal zero.
                result = certify_advantage(value['q_add'], value['q_wait'], int(row['label']),
                                           exact_difference=exact)
                if (value['certificate'] != result['certificate'] or value['margin_lower'] < 0
                        or value['margin_lower'] > result['margin_lower']
                        or not np.array_equal(value['delta'], result['delta'])
                        or value['delta'][0] < _down(value['q_add'][0]-value['q_wait'][1])
                        or value['delta'][1] > _up(value['q_add'][1]-value['q_wait'][0])):
                    raise ValueError('Invalid signed interval/margin evidence')
            elif value['margin_lower'] != 0 or value['certificate'] != 'UNCERTIFIED':
                raise ValueError('Unknown evidence carries a pseudo-exact margin')
        if value_summary(a,v) != cm['summary'] or cm['summary'] != shard['summary']:
            raise ValueError('Value audit summary differs')
        total += len(a)
    if m['complete'] and (total != lm['summary']['counts']['eligible'] or set(by_contract) != {s['contract'] for s in m['shards']}):
        raise ValueError('Incomplete full S3-A index')
    return m

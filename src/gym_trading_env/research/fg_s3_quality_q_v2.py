"""Bounded financial suffix DP, with explicit unknown terminal/Q states."""
from __future__ import annotations

import copy
from decimal import Decimal
from fractions import Fraction
import time

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.rewards.single3_reward import QUALITY_TP_SCHEMA, SCHEMA as LEGACY_REWARD_SCHEMA
from .fg_replay import digest, signed
from .fg_s3_oracle_state_query import decision_state, rules_identity
from .fg_s3_quality_teacher_v2 import trade_facts

PROTOCOL = 'FG_S3_QUALITY_FIRST_FINANCIAL_Q_V1'
PROOF_STATUSES=('Q_VALID','Q_INTERVAL','NO_PROOF','TIMEOUT','CENSORED')
GAMMA = Fraction(332, 333)
UTILITY = 'net_pnl/nominal_net_tp + .25*TP*(B_TP_MARKET_DAY_V1-60)/40'


def trade_utility(env, event):
    """Dimensionless V2 utility; actual contract terms, never fixed 282 CNY."""
    if event['reason'] not in ('TAKE_PROFIT', 'STOP_LOSS'):
        return None, dict(status='NO_PROOF', reason='OTHER_EXIT_QUALITY_UNKNOWN')
    entry = next(f for f in reversed(env.fills)
                 if f['entry_id'] == event['entry_id'] and f['reason'] == 'ENTRY')
    fact = trade_facts(env, entry, event, env.current_step)
    terms = env.terms.at(event['exit_time'])
    same_day = env._trading_day(event['entry_time']) == env._trading_day(event['exit_time'])
    fee = terms['close_today_fee' if same_day else 'close_fee']
    nominal = Decimal(3) * env.terms.multiplier * 5 - Decimal(event['open_fee']) - Decimal(3) * fee
    if nominal <= 0 or not fact['score_valid']:
        return None, dict(status='NO_PROOF', reason='NONPOSITIVE_NOMINAL_TP_OR_UNKNOWN_SCORE')
    value = float(Decimal(event['net_trade_pnl']) / nominal)
    if event['reason'] == 'TAKE_PROFIT': value += .25 * (fact['score'] - 60) / 40
    return value, dict(status='VERIFIED', nominal_net_tp_pnl=str(nominal), score=fact['score'])


class CertifiedQEngine:
    """Exact rational discounted native reward; budget exhaustion yields null.

    The V2 local utility is represented by its computed binary64 rational;
    this preserves the exponential score policy and deterministic comparisons.
    A flat, authentic data end has zero continuation in the declared finite
    observed experiment. An open data end is unknown, not a zero-loss trade.
    """
    def __init__(self, factory, *, max_states=4096, max_seconds=2., max_suffix_bars=12):
        if type(max_states) is not int or max_states < 1 or max_seconds <= 0:
            raise ValueError('Invalid financial Q proof budget')
        self.env = factory()
        self.max_states, self.max_seconds, self.max_suffix_bars = max_states, max_seconds, max_suffix_bars
        self.identity = dict(protocol=PROTOCOL, reward_schema=self.env.reward_config.schema,
            rules=rules_identity(self.env), gamma=[332,333], utility=UTILITY,
            max_states=max_states, max_seconds=max_seconds, max_suffix_bars=max_suffix_bars,
            terminal_policy='finite_observed_flat_boundary_only; open_is_CENSORED',
            feedback_discount='native_MarketClock_reward_exponent; first_feedback_exponent_zero',
            code_sha256=digest(__file__))
        self.identity_hash = fingerprint(self.identity)

    def solve(self, env):
        if (rules_identity(env) != self.identity['rules'] or not self.env._raw.equals(env._raw)):
            raise ValueError('Q engine source identity mismatch')
        root = env.export_state()
        result = dict(protocol=PROTOCOL, engine_identity=self.identity_hash,
            state_id=fingerprint(decision_state(env)), status='NO_PROOF',
            q_c1=[None,None], q_c1_pair_valid=False, q_v2=[None,None], q_v2_pair_valid=False,
            states=0, target_reward_schema=env.reward_config.schema)
        if env.net_lots or not np.array_equal(env._published_mask, [1,1]) or env.status != 'ACTIVE':
            return signed({**result, 'reason':'NO_TWO_LEGAL_FLAT_ACTIONS'})
        remaining = len(env._raw) - 1 - env.current_step
        if remaining > self.max_suffix_bars:
            return signed({**result, 'reason':'OUTSIDE_BOUNDED_PROOF_DOMAIN'})
        started = time.monotonic()
        memo = {}
        scratch = self.env
        unknown = set()
        transitions=[]

        def value(state):
            key = fingerprint(state)
            if key in memo: return memo[key]
            if len(memo) >= self.max_states or time.monotonic() - started > self.max_seconds:
                raise TimeoutError('Financial suffix proof budget exhausted')
            scratch.import_state(state)
            if scratch.truncated and scratch.status == 'CHUNK_END': scratch.reset()
            if scratch.terminated:
                return (Fraction(0), Fraction(0)) if not scratch.entries else (None,None)
            if scratch.current_step == len(scratch._raw)-1:
                if scratch.entries:
                    unknown.add('CENSORED'); return None,None
                return Fraction(0),Fraction(0)
            actions = [a for a in (0,1) if scratch._published_mask[a]]
            current = scratch.export_state()
            options = [branch(current, a) for a in actions]
            out = tuple(max(v[k] for v in options) if all(v[k] is not None for v in options) else None
                        for k in (0,1))
            memo[key] = out
            return out

        def branch(state, action):
            scratch.import_state(state)
            if scratch.truncated and scratch.status == 'CHUNK_END': scratch.reset()
            before, count = scratch.current_step, len(scratch.attributions)
            _, reward, *_ = scratch.step(action)
            after = scratch.current_step
            if after == before:
                unknown.add('CENSORED'); return None,None
            c1 = Fraction.from_float(float(np.float32(reward)))
            utility = Fraction(0)
            for event in scratch.attributions[count:]:
                u, _ = trade_utility(scratch, event)
                utility = None if u is None else (utility + Fraction.from_float(u) if utility is not None else None)
            next_state = scratch.export_state()
            exponent = scratch.market_clock_counter.advance(before, after)
            transitions.append(dict(state_hash=fingerprint(state),action=action,
                next_state_hash=fingerprint(next_state),before_cursor=before,after_cursor=after,
                discount_advance=exponent,configured_reward=str(c1),utility=str(utility) if utility is not None else None,
                equity_after=str(scratch._calculate_equity())))
            continuation = value(next_state)
            return tuple(local + GAMMA**exponent * future if local is not None and future is not None else None
                         for local, future in zip((c1,utility), continuation))

        try:
            options = [branch(root,a) for a in (0,1)]
            for k, name in enumerate(('q_c1','q_v2')):
                pair = [o[k] for o in options]
                result[name] = [float(v) if v is not None else None for v in pair]
                result[name+'_pair_valid'] = all(v is not None for v in pair)
                result[name+'_rational_witness']=[str(v) if v is not None else None for v in pair]
            result['status'] = ('Q_VALID' if result['q_c1_pair_valid'] and result['q_v2_pair_valid']
                                else 'CENSORED' if unknown else 'NO_PROOF')
            if result['q_c1_pair_valid']:
                result['strict_action'] = int(options[1][0] > options[0][0])
                result['strict_tie'] = options[1][0] == options[0][0]
        except TimeoutError:
            result.update(status='TIMEOUT', reason='STATE_OR_TIME_BUDGET')
        result.update(states=len(memo), elapsed_seconds=time.monotonic()-started)
        result['transition_witness']=transitions
        result['transition_witness_hash']=fingerprint(transitions)
        if env.export_state() != root: raise RuntimeError('Q proof mutated actual Env')
        return signed(result)

    def close(self): self.env=None


def build_quality_suffix(env, *, source):
    """Native parity and a separately bound strict auditor for this Reward.

    The original dominance proof requires non-TP intervals to be nonpositive,
    retained TP net cash to be positive and the full quote envelope to fit the
    original floor. The new TP profile preserves those premises; its labels
    are nevertheless rebuilt and bound to a new reward/source identity.
    """
    from .fg_single3_oracle import CandidateIndex,reference_candidate
    from .fg_s3_oracle_state_query import OracleStateQuery
    from .fg_reward_candidates import quality_tp_reward
    from .fg_s3_quality_teacher_v2 import official_days
    if env.reward_config.schema not in (LEGACY_REWARD_SCHEMA,QUALITY_TP_SCHEMA):
        raise ValueError('Quality suffix target mismatch')
    index = CandidateIndex(env)
    days = official_days(env)
    ordinal = {day:i for i,day in enumerate(dict.fromkeys(days))}
    rows = []
    for t in range(index.flat_end):
        row = index.candidate(t)
        if (env.reward_config.schema==QUALITY_TP_SCHEMA and row.get('filled') and
                row.get('exit') and row['exit']['reason'] == 'TAKE_PROFIT'):
            feedback = {}
            early = row.get('first_adverse_effective_bar')
            if early is not None and early <= 15:
                feedback[row['first_adverse_cursor']] = -env.reward_config.beta * (1 if early <= 5 else env.reward_config.rho)
            u, i = row['exit_cursor'], row['entry_cursor']
            feedback[u] = feedback.get(u,0.) + quality_tp_reward(row['held_market_bars'], ordinal[days[u]]-ordinal[days[i]])
            row['rewards']['C1'] = [[s,float(np.float32(v))] for s,v in sorted(feedback.items()) if v]
        rows.append(row)
    # Native execution/feedback parity is a gate for each rebuilt target, not
    # an assumption inherited from the legacy scorer's acceptance.
    chosen=set(np.linspace(0,max(0,len(rows)-1),min(8,len(rows)),dtype=int).tolist())
    for reason in ('TAKE_PROFIT','STOP_LOSS'):
        found=next((r['cursor'] for r in rows if (r.get('exit') or {}).get('reason')==reason),None)
        if found is not None: chosen.add(found)
    for t in sorted(chosen):
        actual=reference_candidate(env,t)
        if (actual['rewards']!=rows[t].get('rewards',{}).get('C1',[]) or actual['filled']!=rows[t].get('filled',False)
                or actual.get('exit_cursor')!=rows[t].get('exit_cursor') and actual['filled']):
            raise ValueError('New QualityReward candidate/native feedback parity failed at '+str(t))
    # CandidateIndex mutates its private builder Env; query identity excludes
    # its cursor/account but binds complete market and native rules.
    query = OracleStateQuery(env,rows,index.flat_end,fold='TRAIN',source_identity={
        **source, 'target_reward_schema':env.reward_config.schema,
        'candidate_hash':fingerprint(rows), 'builder_code_sha256':digest(__file__)})
    query.identity.update(protocol='FG_S3_QUALITY_NATIVE_STRICT_AUDITOR_V1',
        target_reward_schema=env.reward_config.schema,
        native_candidate_parity_cursors=sorted(chosen),
        funding_proof_scope='new_target_proof_at_original_floor; no_legacy_label_reuse')
    query.identity_hash = fingerprint(query.identity)
    return query


class QualityOracle:
    """DAgger-compatible strict teacher with independently versioned facts/Q."""
    def __init__(self, teacher, q_engine, *, suffix=None):
        if q_engine is not None and teacher.rules['reward_hash'] != q_engine.identity['rules']['reward_hash']:
            raise ValueError('Quality strict and Q reward targets differ')
        target_schema=teacher.clone.reward_config.schema
        if target_schema not in (LEGACY_REWARD_SCHEMA,QUALITY_TP_SCHEMA):
            raise ValueError('Strict auditor target reward schema is unknown')
        self.teacher, self.engine, self.suffix = teacher, q_engine, suffix
        self.rules = teacher.rules
        self.identity = dict(protocol=PROTOCOL, fold='TRAIN', rules=self.rules,
            teacher_identity=teacher.identity_hash, q_identity=q_engine.identity_hash if q_engine else None,
            suffix_identity=suffix.identity_hash if suffix else None,
            clock_protocol='S3_MARKET_MINUTE', target_reward_schema=target_schema)
        self.identity_hash = fingerprint(self.identity)
        self.last_label = None

    decision_state = staticmethod(decision_state)

    def query(self, env, mask, *, snapshot=None):
        label = self.teacher.query_state(env,mask,snapshot=snapshot)
        q = self.engine.solve(env) if self.engine is not None else signed(dict(
            status='NO_PROOF', reason='QUALITY_FIRST_Q_RESEARCH_DISABLED',
            q_c1=[None,None],q_v2=[None,None],q_c1_pair_valid=False,q_v2_pair_valid=False))
        result = dict(action=0, valid=False, certified=False, reason='NO_PROOF',
            q_add=None, q_wait=None, q_margin=None,
            state_id=fingerprint(decision_state(env)), oracle_identity=self.identity_hash)
        if q['q_c1_pair_valid']:
            result.update(action=q['strict_action'], valid=True, certified=True, reason='CERTIFIED_QUALITY_REWARD_SUFFIX')
        elif self.suffix is not None:
            answer = self.suffix.query(env,mask,snapshot=snapshot)
            result.update({k:answer[k] for k in ('action','valid','certified','reason')})
        label = {k:v for k,v in label.items() if k != 'content_hash'}
        from .fg_s3_quality_teacher_v2 import wait_reasons
        label.update(q=q, decision_oracle_certified=result['valid'], strict_action=result['action'],
                     strict_identity=self.identity_hash, **wait_reasons(label,result))
        label['strict_action']=result['action'] if result['valid'] else None
        exact=q.get('q_c1_pair_valid',False)
        label.update(strict_auditor_identity=self.suffix.identity_hash if self.suffix else None,
            strict_certificate_floor=str(self.suffix.floor) if self.suffix and not exact else None)
        label['layers']['L3']=dict(valid=result['valid'],reason=result['reason'],
            protocol='EXACT_NATIVE_FINANCIAL_Q' if exact else 'STRICT_SUFFIX_AUDITOR',source=self.identity_hash,
            action=label['strict_action'],quality_is_action_label=False)
        if q.get('q_c1_pair_valid'):
            label['strict_preference']='TIE' if q['strict_tie'] else 'ADD_STRICT' if result['action'] else 'WAIT_STRICT'
            if (not result['action'] and q['q_c1'][0]>q['q_c1'][1] and q['q_c1'][0]>0 and
                    self.suffix is not None and self.suffix.first[env.current_step] is not None):
                label['reasons'].append('WAIT_BETTER_OPPORTUNITY_VERIFIED')
                label['causal_explanation']='Q_VERIFIED_AND_CERTIFIED_SUFFIX_FUTURE_ADD'
        from .fg_s3_quality_catalog_v2 import soft_teacher
        label.update(soft_teacher(label,result))
        self.last_label = signed(label)
        result['proof_hash'] = fingerprint(result)
        return result

    def accepts_identity(self, contract, identity_hash):
        return contract == self.rules['contract'] and identity_hash == self.identity_hash

    def close(self):
        self.teacher.close()
        if self.engine is not None: self.engine.close()

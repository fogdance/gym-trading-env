"""Native-account counterfactual facts. Future fields are TRAIN sidecar only."""
from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import asdict,dataclass
from decimal import Decimal

import numpy as np
import time

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import check_files, digest, signed
from .fg_s3_oracle_state_query import decision_state, rules_identity
from gym_trading_env.utils.account_fingerprint import account_fingerprint,prime_account_fingerprint,capture_account_fingerprint
from .fg_s3_quality_score import PROTOCOL as SCORE_POLICY, quality_for_exit

PROTOCOL = 'FG_S3_QUALITY_TEACHER_V1'
HORIZONS = (5, 15, 60, 120, 240, 'entry_day_end', 'next_day_end')
TOUCH_HORIZONS = (5, 60, 240, 'entry_day_end')
EVENTS = ('TP5', 'SL30', 'OTHER', 'OPEN')


@dataclass(frozen=True)
class StateProof:
    state_id: str
    source_hash: str
    rules_hash: str
    funds_basis: str = 'actual_equity_and_native_ledger'


def official_days(env):
    from .fg_reward_candidates import _official_day_key
    col = 'trading_date' if 'trading_date' in env._raw else 'trading_day'
    if col not in env._raw:
        raise ValueError('Quality facts require supplied official trading dates')
    days = tuple(_official_day_key(v) for v in env._raw[col])
    if any(a > b for a, b in zip(days, days[1:])):
        raise ValueError('Official trading days are not ordered')
    return days


def official_day_index(env):
    """Immutable teacher-private calendar metadata, built once per market."""
    days=official_days(env)
    unique=tuple(dict.fromkeys(days))
    return dict(days=days,unique=unique,ordinal={d:i for i,d in enumerate(unique)},
                last={day:i for i,day in enumerate(days)})


def _window_label(age, exit_age, exit_type, observed, firsts):
    closed = exit_type in ('TP5', 'SL30', 'OTHER_EXIT')
    complete = age is not None and (closed or observed >= age)
    event_valid = age is not None and ((closed and exit_age <= age) or observed >= age)
    event = (('TP5', 'SL30', 'OTHER_EXIT').index(exit_type)
             if event_valid and closed and exit_age <= age else 3)
    # A known touch remains known even on a censored path; a negative needs
    # complete exposure. Each binary event owns its own validity bit.
    labels, valid = [], []
    for first in firsts:
        reached = age is not None and first is not None and first <= age
        labels.append(int(reached))
        valid.append(bool(reached or complete))
    return event, bool(event_valid), labels, valid


def trade_facts(env, entry, event, observed_until, *, day_complete=False,calendar_index=None):
    """Extract only the executed trade interval, clipped at its native exit."""
    calendar_index=official_day_index(env) if calendar_index is None else calendar_index
    days=calendar_index['days']
    if len(days)!=len(env._raw): raise ValueError('Teacher calendar does not match its native market')
    i = int(env._raw.index.searchsorted(entry['time'], side='right'))
    # Fill clocks are bob, not eob; native entries retain their physical cursor.
    if 'entry_cursor' in entry:
        i = int(entry['entry_cursor'])
    elif event is not None and 'entry_cursor' in event:
        i = int(event['entry_cursor'])
    u = int(observed_until)
    rows = np.flatnonzero(env.valid_fill_clock[i:u + 1]) + i
    if not len(rows):
        raise ValueError('A real entry requires an observed fill bar')
    price = float(entry.get('price', entry.get('entry_price')))
    sign = 1 if entry['side'] == 'LONG' else -1
    raw = env._raw.iloc[rows]
    # Risk uses native execution quotes, including spread/slippage and tick
    # rounding. The quality score itself is B(t,d), without an MAE deduction.
    highs = np.asarray([float(env._price(value,entry['side'],time))
                        for time,value in zip(raw.index,raw.High)])
    lows = np.asarray([float(env._price(value,entry['side'],time))
                       for time,value in zip(raw.index,raw.Low)])
    favorable = highs - price if sign == 1 else price - lows
    adverse = price - lows if sign == 1 else highs - price
    favorable, adverse = np.maximum(0., favorable), np.maximum(0., adverse)
    def first(values, threshold):
        hit = np.flatnonzero(values >= threshold - 1e-10)
        return int(hit[0] + 1) if len(hit) else None
    first_f = {str(k): first(favorable, k) for k in (1, 3, 5)}
    first_a = {str(k): first(adverse, k) for k in (5, 10, 20, 30)}
    reason = event['reason'] if event else None
    kind = {'TAKE_PROFIT': 'TP5', 'STOP_LOSS': 'SL30'}.get(reason,
                'OTHER_EXIT' if event else 'RIGHT_CENSORED')
    unique,ordinal=calendar_index['unique'],calendar_index['ordinal']
    d = ordinal[days[u]] - ordinal[days[i]]
    observed = len(rows)
    maes = [float(adverse[:min(h, observed)].max()) for h in (5, 60, 240, observed)]
    end_ages = []
    for delta in (0, 1):
        day_index = ordinal[days[i]] + delta
        target = unique[day_index] if day_index < len(unique) else None
        end = calendar_index['last'].get(target)
        # Source completeness is independently certified. Last observed bar
        # is not itself proof that an official day was fully observed.
        end_ages.append(env.market_clock_counter.held(i, end)
                        if end is not None and day_complete else None)
    # A committed exit within an official target day is already a known
    # by-day event, even when the dataset does not certify the full day end.
    # Use the observed exit age only for labeling; never invent a day-end clock.
    day_label_ages=[observed if event is not None and d<=delta else age
                    for delta,age in enumerate(end_ages)]
    ages = [*HORIZONS[:5], *day_label_ages]
    event_y, event_v = [], []
    for age in ages:
        y, v, _, _ = _window_label(age, observed if event else None, kind, observed, ())
        event_y.append(y); event_v.append(v)
    touch_y, touch_v, adverse_y, adverse_v = [], [], [], []
    for age in (5, 60, 240, day_label_ages[0]):
        _, _, y, v = _window_label(age, observed if event else None, kind, observed,
                                   (first_f['1'], first_f['3'], first_a['10']))
        touch_y.append(y[:2]); touch_v.append(v[:2])
        adverse_y.append(y[2]); adverse_v.append(v[2])
    close_quotes=np.asarray([float(env._price(value,entry['side'],time))
                            for time,value in zip(raw.index,raw.Close)])
    underwater=(sign*(close_quotes-price)<0)
    if event is not None: underwater=underwater[:-1]  # no post-exit close
    longest=run=0
    for value in underwater:
        run=run+1 if value else 0
        longest=max(longest,run)
    score, score_reason = quality_for_exit(reason or 'RIGHT_CENSORED',observed,d)
    return dict(exit_type=kind, entry_cursor=i, observed_until_cursor=u,
        exit_cursor=u if event else None, held_market_minutes=observed,
        exit_natural_minutes=((env._raw.index[u] - env.bobs[i]).total_seconds() / 60
                              if event else None),
        entry_official_trading_date=days[i], exit_official_trading_date=days[u] if event else None,
        crossed_trading_days=d if event else None, market_minutes_to_entry_day_end=end_ages[0],
        first_favorable=first_f, first_adverse=first_a,
        first_favorable_valid={k:bool(event is not None or v is not None) for k,v in first_f.items()},
        first_adverse_valid={k:bool(event is not None or v is not None) for k,v in first_a.items()},
        favorable_event_not_reached={k:bool(event is not None and v is None) for k,v in first_f.items()},
        adverse_event_not_reached={k:bool(event is not None and v is None) for k,v in first_a.items()},
        mae_15=float(adverse[:min(15,observed)].max()),
        mae_15_complete=bool(event is not None or observed>=15),
        underwater_minutes=int(underwater.sum()), underwater_longest_minutes=longest,
        underwater_observed_minutes=len(underwater), underwater_observed_valid=True,
        underwater_full_valid=event is not None,
        underwater_definition='native_executable_close_while_held_at_eob; settled_exit_bar_excluded',
        early10_5=first_a['10'] is not None and first_a['10']<=5,
        early10_15=first_a['10'] is not None and first_a['10']<=15,
        early10_5_valid=bool(event is not None or observed>=5 or first_a['10'] is not None),
        early10_15_valid=bool(event is not None or observed>=15 or first_a['10'] is not None),
        mae_5=maes[0], mae_60=maes[1], mae_240=maes[2], mae_full=maes[3],
        mae_window_complete=[bool(event or observed >= h) for h in (5, 60, 240)],
        mfe_full=float(favorable.max()), path_complete=event is not None,
        net_pnl=event['net_trade_pnl'] if event else None,
        intrabar_ambiguous=bool(event and event.get('ambiguous_tp_sl')),
        intrabar_rule='native_risk_wins_sl_before_tp',
        score=score, score_valid=score is not None, score_policy=SCORE_POLICY,
        score_reason=score_reason,
        score_band='UNKNOWN' if score is None else 'HIGH' if score>=80 else 'QUALIFIED' if score>=60 else 'LOW',
        event=event_y, event_valid=event_v, touch=touch_y, touch_valid=touch_v,
        adverse=adverse_y, adverse_valid=adverse_v,
        entry_day_observed=bool(end_ages[0] is not None and observed >= end_ages[0]),
        next_day_observed=bool(end_ages[1] is not None and observed >= end_ages[1]))


def wait_reasons(label, strict):
    reason = label.get('invalid_reason')
    if reason in ('FORCED_WAIT_POSITION', 'FORCED_WAIT_ACTION_MASK'):
        return dict(primary=reason, reasons=[reason], evidence='FACT',wait_class='FORCED',
                    strict_preference='UNVERIFIED',causal_explanation='UNKNOWN_REASON')
    if not strict.get('valid'):
        tag = ('ORACLE_UNVERIFIED_FUNDS' if 'FUNDS' in strict.get('reason', '')
               else 'ORACLE_UNVERIFIED_OTHER')
        tags = [tag]
    elif strict['action'] == 1:
        tags = ['STRICT_ADD']
    else:
        tags = []
        if label.get('exit_type') == 'SL30': tags.append('STRICT_WAIT_SL30')
        adverse = label.get('first_adverse', {}).get('10')
        if adverse is not None and adverse <= 5: tags.append('STRICT_WAIT_EARLY_ADVERSE10')
        if label.get('exit_type') == 'TP5' and label.get('crossed_trading_days', 0) >= 1:
            tags.append('STRICT_WAIT_TP_AFTER_DAY')
        if label.get('entry_day_observed') and (label.get('first_favorable', {}).get('5') is None
                or label.get('held_market_minutes', 0) > label['market_minutes_to_entry_day_end']):
            tags.append('STRICT_WAIT_NO_TP_BY_DAY')
        if label.get('exit_type') == 'TP5' and label.get('score_valid') and label['score'] >= 80:
            tags.append('STRICT_WAIT_HIGH_QUALITY_ADD')
        if not tags: tags.append('STRICT_WAIT_OTHER')
    if label.get('exit_type') == 'RIGHT_CENSORED': tags.append('OUTCOME_RIGHT_CENSORED')
    risk=any(t in tags for t in ('STRICT_WAIT_SL30','STRICT_WAIT_EARLY_ADVERSE10','STRICT_WAIT_TP_AFTER_DAY','STRICT_WAIT_NO_TP_BY_DAY'))
    group=('UNVERIFIED' if not strict.get('valid') else 'STRICT_ADD' if strict['action']==1 else
           'VOLUNTARY_RISK' if risk else 'VOLUNTARY_HIGH_QUALITY' if 'STRICT_WAIT_HIGH_QUALITY_ADD' in tags else 'VOLUNTARY_OTHER')
    return dict(primary=tags[0], reasons=tags,wait_class=group,
                strict_preference='UNVERIFIED' if not strict.get('valid') else 'ADD_STRICT' if strict['action']==1 else 'WAIT_WEAK',
                evidence='STRICT_ORACLE' if strict.get('valid') else 'FACT',
                causal_explanation='UNKNOWN_REASON')


def verify_native_e_fork(env, clone):
    """Prove isolation with the actual native E executor, not a bypass flag."""
    from .fg_s3_e_observation import OBS_SCHEMA, observation_contract
    if (getattr(env, '_e_profile', None) is None or getattr(clone, '_e_profile', None) is None
            or env.reward_config.schema != 'FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1'
            or env.e_observation_identity != clone.e_observation_identity
            or not hasattr(env._e_profile, 'source_receipt')
            or not hasattr(clone._e_profile, 'source_receipt')):
        raise ValueError('E-profile native Quality fork identity is NOT_VERIFIED')
    if (env._e_market is clone._e_market or
            np.shares_memory(env._features, clone._features) or
            np.shares_memory(env._e_market.obs_array, clone._e_market.obs_array)):
        raise ValueError('E-profile native Quality fork aliases mutable market state')
    state = env.export_state()
    expected = env._get_obs()
    clone.import_state(state)
    restored = clone._get_obs()
    for k,v in expected.items():
        if not np.array_equal(v, restored[k]):
            raise ValueError('E-profile native Quality fork observation differs')
    if clone.status == 'ACTIVE' and not clone.terminated:
        clone.step(0)
    if env.export_state() != state:
        raise ValueError('E-profile native Quality fork mutated the source account')
    clone.import_state(state)
    return dict(protocol='FG_S3_E_NATIVE_FORK_V2',observation_schema=OBS_SCHEMA,
        observation_contract_hash=observation_contract()['contract_hash'],
        source_identity=env._e_profile.identity,
        profile_receipt_hash=fingerprint(env._e_profile.source_receipt),
        independent_account_and_market=True, exact_restore=True,
        snapshot_hash=fingerprint(state))


class NativeQualityTeacher:
    """Restore a full financial snapshot into a private original Env executor."""
    def __init__(self, env, factory, *, fold, source):
        if fold != 'TRAIN' or not env.single3:
            raise ValueError('Quality labels require Single3 TRAIN')
        from gym_trading_env.envs.entry_market_cache import _file_identity
        from pathlib import Path
        self._source_stats={}
        for name,expected in source.get('files',{}).items():
            path=Path(name)
            actual,stat=_file_identity(path)
            if actual!=expected: raise ValueError('Quality source file changed: '+name)
            self._source_stats[name]=(str(path.resolve(strict=True)),stat)
        self.source = copy.deepcopy(source)
        self.rules = rules_identity(env)
        self.market_hash = fingerprint(env._raw.to_dict(orient='split'))
        self.clone = factory()
        self.e_fork = None
        if getattr(env,'_e_profile',None) is not None or getattr(self.clone,'_e_profile',None) is not None:
            self.e_fork = verify_native_e_fork(env,self.clone)
            from .fg_s3_e_observation import NativeEObservationCache
            self.clone._e_observation_cache = NativeEObservationCache()
            from gym_trading_env.envs.entry_execution import NativeMarketExecutionCache
            self.clone._native_execution_cache = NativeMarketExecutionCache()
        if rules_identity(self.clone) != self.rules or not self.clone._raw.equals(env._raw):
            raise ValueError('Counterfactual clone differs from native rules or market')
        self.calendar_index=official_day_index(self.clone)
        self.max_market_steps=source.get('max_market_steps')
        if self.max_market_steps is not None and (type(self.max_market_steps) is not int or self.max_market_steps<1):
            raise ValueError('Quality native exposure budget must be a positive market-minute integer')
        self.cache_entries=source.get('state_cache_entries',128)
        if type(self.cache_entries) is not int or not 0<=self.cache_entries<=4096:
            raise ValueError('Quality exact-state cache must contain 0..4096 entries')
        self._cache=OrderedDict()
        self.source_guard_interval=source.get('source_guard_interval',1)
        if type(self.source_guard_interval) is not int or not 1<=self.source_guard_interval<=1024:
            raise ValueError('Quality source guard interval must be 1..1024 queries')
        self.statistics=dict(queries=0,native_traces=0,cache_hits=0,trace_seconds=0.,
                             independently_observed_market_minutes=0,cache_peak_entries=0)
        if self.e_fork is not None:
            self.statistics['e_observation_cache'] = self.clone._e_observation_cache.statistics
            self.statistics['native_execution_cache'] = self.clone._native_execution_cache.statistics
        self.identity = dict(protocol=PROTOCOL, fold=fold, source=source, rules=self.rules,
            market_hash=self.market_hash, code_sha256=digest(__file__),
            clock_protocol='S3_MARKET_MINUTE', score_policy=SCORE_POLICY,
            source_guard=('SHA256_on_open_then_device_inode_size_mtime_ns_ctime_ns_each_query'
                if self.source_guard_interval==1 else dict(open='SHA256',
                    block_queries=self.source_guard_interval,block_and_close='device_inode_size_mtime_ns_ctime_ns')),
            state_cache=dict(protocol='COMPLETE_NATIVE_STATE_LRU_V1',entries=self.cache_entries,
                             key='teacher_identity + full export_state SHA256; source/mask checked before lookup'),
            max_market_steps=self.max_market_steps,calendar_metadata='immutable_teacher_private_index',
            horizons=list(HORIZONS), touch_horizons=list(TOUCH_HORIZONS), funds_proof='actual_snapshot_clone')
        import inspect
        self.identity['native_env_code_sha256']=digest(inspect.getfile(type(env)))
        if self.e_fork is not None:
            self.identity['e_fork'] = self.e_fork
            self.identity['native_execution_reuse'] = dict(
                protocol='TEACHER_PRIVATE_NATIVE_QUOTES_V1',
                code_sha256=digest(inspect.getfile(NativeMarketExecutionCache)),
                quote_key='exact_raw_price_side_opening_effective_costs_tick_decimal_context',
                row_capacity=self.clone._native_execution_cache.row_capacity,
                quote_capacity=self.clone._native_execution_cache.quote_capacity,
                term_capacity=self.clone._native_execution_cache.term_capacity,
                financial_transitions='original_Env_each_market_minute_each_actual_account')
        self.identity_hash = fingerprint(self.identity)
        initial_snapshot=env.export_state()
        prime_account_fingerprint(env,initial_snapshot)
        prime_account_fingerprint(self.clone,initial_snapshot)

    def _check_source_stats(self):
        from pathlib import Path
        from gym_trading_env.envs.entry_market_cache import _stat_identity
        for name,(resolved,stat) in self._source_stats.items():
            path=Path(name)
            if str(path.resolve(strict=True))!=resolved or _stat_identity(path)!=stat:
                raise ValueError('Frozen quality source changed during collection: '+name)

    def query_state(self, env, mask, *, snapshot=None):
        if self.statistics['queries']%self.source_guard_interval==0:self._check_source_stats()
        if self.e_fork is not None and (
                env.e_observation_identity != self.clone.e_observation_identity or
                env._e_profile.identity != self.e_fork['source_identity'] or
                fingerprint(env._e_profile.source_receipt) != self.e_fork['profile_receipt_hash']):
            raise ValueError('E-profile source identity changed during Quality collection')
        actual = decision_state(env)
        state_id = fingerprint(actual)
        if (rules_identity(env) != self.rules or not env._raw.equals(self.clone._raw)
                or (snapshot is not None and fingerprint(snapshot) != state_id)):
            raise ValueError('Quality source/state identity mismatch')
        if not np.array_equal(mask, env._published_mask):
            raise ValueError('Quality requires the real published action mask')
        self.statistics['queries']+=1
        state=env.export_state()
        private_snapshot=None
        if (self.e_fork is not None and not env.net_lots and not env.entries and all(mask)
                and env.status=='ACTIVE' and not env.terminated and not env.truncated):
            snapshot_hash,private_snapshot=capture_account_fingerprint(env,state)
        else:
            snapshot_hash=account_fingerprint(env,state)
        if snapshot_hash in self._cache:
            self.statistics['cache_hits']+=1
            self._cache.move_to_end(snapshot_hash)
            return copy.deepcopy(self._cache[snapshot_hash])
        result = dict(protocol=PROTOCOL, state_id=state_id, teacher_identity=self.identity_hash,
            state_proof=asdict(StateProof(state_id,fingerprint(self.source),fingerprint(self.rules))),
            immediate_outcome_verified=False, score=None, score_valid=False,
            score_policy=SCORE_POLICY, fold='TRAIN',
            actual_legal_wait=bool(mask[0]), actual_legal_add=bool(mask[1]),
            training_action_label_allowed=False,
            invalid_reason=None, equity=str(env._calculate_equity()),
            event=[0]*7, event_valid=[False]*7, touch=[[0,0] for _ in range(4)],
            touch_valid=[[False,False] for _ in range(4)], adverse=[0]*4, adverse_valid=[False]*4)
        result['account_snapshot_hash']=snapshot_hash
        if env.net_lots or env.entries:
            result['invalid_reason'] = 'FORCED_WAIT_POSITION'
        elif not all(mask):
            result['invalid_reason'] = 'FORCED_WAIT_ACTION_MASK'
        elif env.status != 'ACTIVE' or env.terminated or env.truncated:
            result['invalid_reason'] = 'INACTIVE_ACCOUNT'
        if not result['invalid_reason']:
            started=time.perf_counter()
            result.update(trace_native_trade(env,self.clone,state=state,
                calendar_index=self.calendar_index,max_market_steps=self.max_market_steps,
                day_complete=bool(self.source.get('calendar_complete',False)),
                _private_snapshot=private_snapshot))
            self.statistics['native_traces']+=1
            self.statistics['trace_seconds']+=time.perf_counter()-started
            self.statistics['independently_observed_market_minutes']+=result.get('held_market_minutes',0)
        reason=result.get('invalid_reason') or result.get('exit_type','UNKNOWN')
        result['layers']={
            'L0':dict(valid=True,reason='NATIVE_CURRENT_MASK',protocol=env.protocol,source=self.rules),
            'L1':dict(valid=bool(result['immediate_outcome_verified']),reason=reason,
                      protocol='NATIVE_ACCOUNT_ADD_THEN_WAIT_V1',source=self.identity_hash),
            'L2':dict(valid=bool(result['score_valid']),reason=result.get('score_reason',reason),
                      protocol=SCORE_POLICY,source=self.identity_hash),
        }
        result=signed(result)
        if self.cache_entries and result['immediate_outcome_verified']:
            self._cache[snapshot_hash]=copy.deepcopy(result)
            while len(self._cache)>self.cache_entries:self._cache.popitem(last=False)
            self.statistics['cache_peak_entries']=max(self.statistics['cache_peak_entries'],len(self._cache))
        return result

    def close(self):
        # Native Env.close() may persist state_path. A research executor owns
        # no external resources requiring that side effect.
        try:self._check_source_stats()
        finally:
            self.clone=None
            self._cache.clear()


def trace_native_trade(env,clone, *, day_complete=False,state=None,calendar_index=None,max_market_steps=None,
                       _private_snapshot=None):
    """Read-only native trace, including rejected and exceptional paths."""
    state=env.export_state() if state is None else state
    if max_market_steps is not None and (type(max_market_steps) is not int or max_market_steps<1):
        raise ValueError('Quality native exposure budget must be a positive market-minute integer')
    try:
        return _trace_native_trade(env,clone,day_complete=day_complete,state=state,
            calendar_index=calendar_index,max_market_steps=max_market_steps,
            private_snapshot=_private_snapshot)
    finally:
        if env.export_state()!=state:
            raise RuntimeError('Counterfactual mutated the real account')


def _trace_native_trade(env,clone, *, day_complete=False,state,calendar_index,max_market_steps,
                        private_snapshot=None):
        """Shared native trace primitive, also usable AFTER sealed DEV inference.

        This does not publish a TRAIN label or an Oracle decision. Callers
        separately enforce their role, source identity and legality contracts.
        """
        from .fg_replay import execute_observed
        from gym_trading_env.rewards.single3_reward import Single3Reward,Single3RewardConfig
        if private_snapshot is None:clone.import_state(state)
        else:clone._restore_private_state(private_snapshot,source_env=env)
        shadow=(Single3Reward(Single3RewardConfig.for_candidate('C1'))
                if getattr(env,'_e_profile',None) is None else None)
        before = clone._calculate_equity()
        fills, exits = len(clone.fills), len(clone.attributions)
        decision = clone.current_step
        rewards = []
        original_reward=0.
        def advance(action):
            nonlocal original_reward
            if shadow is None:
                # The native E Env already computes Quality TP feedback and
                # the complete causal observation. No Replay row is consumed
                # by this posterior trace; retain its float32 feedback exactly.
                _,reward,*_=clone.step(action)
                rewards.append([clone.current_step,float(np.float32(reward))])
                return
            _,row=execute_observed(clone,action)
            rewards.append([clone.current_step,float(np.float32(row['official_reward']))])
            if row['advanced'] and shadow is not None:
                original_reward+=float(np.float32(shadow.step(row['bar'],clone.terms.multiplier)['total']))
        advance(1)
        opened = next((copy.deepcopy(f) for f in clone.fills[fills:] if f['reason'] == 'ENTRY'), None)
        if opened is None:
            return dict(exit_type='REJECTED', invalid_reason=clone._get_info()['reject_reason'] or clone.status)
        opened['entry_cursor'] = clone.current_step
        budget_censored=False
        while len(clone.attributions) == exits and not clone.terminated:
            if max_market_steps is not None and clone.market_clock_counter.advance(decision,clone.current_step)>=max_market_steps:
                budget_censored=True;break
            if clone.truncated:
                if clone.status != 'CHUNK_END': break
                clone.reset()
            old = clone.current_step
            advance(0)
            if clone.current_step == old: break
        event = clone.attributions[-1] if len(clone.attributions) > exits else None
        facts = trade_facts(clone, opened, event, clone.current_step,
                            day_complete=day_complete,calendar_index=calendar_index)
        if event is not None and clone._calculate_equity() - before != Decimal(event['net_trade_pnl']):
            raise ValueError('Native counterfactual account/PnL reconciliation failed')
        result=dict(facts, immediate_outcome_verified=True, decision_cursor=decision,
            account_snapshot_hash=account_fingerprint(env,state),
            native_entry=copy.deepcopy(opened),
            native_event=copy.deepcopy(event), original_c1_reward=original_reward if shadow is not None else None,
            censor_reason=('EXPOSURE_BUDGET' if budget_censored else 'NATIVE_DATA_BOUNDARY') if event is None else None,
            configured_reward=sum(r[1] for r in rewards), reward_feedback=rewards,
            reward_schema=env.reward_config.schema, funds_after=str(clone._calculate_equity()),
            final_state_hash=account_fingerprint(clone))
        is_tp=event is not None and event['reason']=='TAKE_PROFIT'
        old_tp=float(shadow.scorer.closed[event['entry_id']].timeboxed_tp) if is_tp and shadow is not None else None
        new_tp=facts['score']/100 if is_tp else None
        result.update(tp_event_reward_new=new_tp,old_tp_event_reward=old_tp,
            old_reward_undiscounted=original_reward if shadow is not None else None,
            new_reward_undiscounted=(sum(r[1] for r in rewards) if shadow is None and event is not None else
                                    original_reward-old_tp+new_tp if is_tp else
                                    original_reward if event is not None else None),
            native_event_hash=fingerprint(event) if event is not None else None,
            risk=dict(mae_5=facts['mae_5'],mae_60=facts['mae_60'],mae_240=facts['mae_240'],
                mae_full=facts['mae_full'],first_adverse=facts['first_adverse'],
                source='native_OHLC_execution_quote_envelope; no_tick_order_claim'))
        return result

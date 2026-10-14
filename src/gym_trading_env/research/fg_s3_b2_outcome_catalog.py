"""Certified ADD outcomes from TRAIN candidates; never an Actor input."""
from collections import Counter
from decimal import Decimal

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_s3_oracle_state_query import rules_identity, outcome_account_reason
from .fg_replay import check_files, signed, write_json

PROTOCOL = 'FG_S3_B2_OUTCOME_MARKET_MINUTE_V1'
EVENTS = ('TP_FAST', 'TP_MID', 'TP_LATE', 'SL', 'OTHER', 'OPEN_AT_H')
FIELDS = ('outcome_event', 'outcome_event_valid', 'outcome_early10', 'outcome_early_valid')


def outcome_label(candidate, horizon=60):
    """Separate event and early-risk observation/censor boundaries."""
    if type(horizon) is not int or horizon != 60:
        raise ValueError('B2 V0 requires a frozen 60-market-bar outcome window')
    label = dict(event=0, valid_event=False, early10=0, valid_early=False,
        reason='NO_CERTIFIED_FILL', observed_market_bars=0,
        first_adverse_market_bar=None, tp_market_minutes=None,
        ambiguous_intrabar=False)
    if not candidate.get('eligible') or not candidate.get('filled'):
        return label
    bars = int(candidate['held_market_bars'])
    event = candidate.get('exit')
    if bars < 1:
        raise ValueError('A filled candidate requires an inclusive market bar')
    # CandidateIndex observes until a real exit or its authentic data boundary.
    # An exit after H is OPEN within H, never a late-window TP/SL target.
    closed = event is not None
    first = candidate.get('first_adverse_effective_bar')
    label.update(observed_market_bars=min(bars, horizon),
        first_adverse_market_bar=first,
        ambiguous_intrabar=bool(event and event.get('ambiguous_tp_sl')))
    if closed and bars <= horizon:
        reason = event['reason']
        if reason == 'TAKE_PROFIT':
            label['event'] = 0 if bars <= 5 else 1 if bars <= 15 else 2
            label['tp_market_minutes'] = bars
        else:
            label['event'] = 3 if reason == 'STOP_LOSS' else 4
        label.update(valid_event=True, reason='CERTIFIED_EXIT_WITHIN_H')
    elif bars >= horizon:
        label.update(event=5, valid_event=True, reason='CERTIFIED_OPEN_AT_H')
    else:
        label['reason'] = 'RIGHT_CENSORED_BEFORE_H'
    # Early exits have a complete risk history. An open short tail has not.
    if closed or bars >= 5:
        label.update(valid_early=True, early10=int(first is not None and first <= min(5, bars)))
    return label


class OutcomeCatalog:
    """O(1) cursor lookup plus independent current-account certification.

    The conservative cash floor is intentionally retained. The bound in the
    original constant-terms certificate covers the full observed quote/margin
    envelope of one 3-lot position, including a losing outcome. We certify this
    ADD's funding independently; no Oracle action or valid bit is consulted.
    """
    def __init__(self, query, env, candidates):
        if query.identity['fold'] != 'TRAIN' or rules_identity(env) != query.rules:
            raise ValueError('Outcome source must be the authenticated TRAIN account')
        if (query.identity['clock_protocol'] != 'S3_MARKET_MINUTE'
                or query.proof['status'] != 'exact_proven' or not query.proof['constant_terms']):
            raise ValueError('Outcome requires exact market-minute constant-terms proof')
        check_files(query.identity['source']['candidate_files'])
        self.query, self.rules = query, query.rules
        self.floor = Decimal(query.proof['retained_paths_cash_lower_bound'])
        self.required_equity = Decimal(query.proof['required_equity_upper_bound'])
        if self.floor <= self.required_equity:
            raise ValueError('Outcome funding envelope is not certified')
        self.rows = {int(c['cursor']): c for c in candidates}
        if len(self.rows) != len(candidates) or sorted(self.rows) != list(range(query.end)):
            raise ValueError('Outcome candidate coverage differs from certified contract')
        self.labels = {t: outcome_label(c) for t, c in self.rows.items()}
        counts = Counter()
        for label in self.labels.values():
            counts[label['reason']] += 1
            counts['event_'+EVENTS[label['event']]] += int(label['valid_event'])
            counts['valid_event'] += int(label['valid_event'])
            counts['valid_early'] += int(label['valid_early'])
        self.identity = dict(protocol=PROTOCOL, fold='TRAIN', horizon_market_bars=60,
            clock_protocol='S3_MARKET_MINUTE', events=EVENTS,
            oracle_source=query.identity, rule_identity=self.rules,
            funds_domain='active_flat_equity_at_least_certified_floor',
            funding_bound=str(self.required_equity), funding_floor=str(self.floor),
            outcome_label_hash=fingerprint(self.labels), counts=dict(counts))
        self.identity_hash = fingerprint(self.identity)

    def query_state(self, env, mask, *, snapshot):
        state_id = fingerprint(snapshot)
        result = dict(protocol=PROTOCOL, state_id=state_id,
            catalog_identity=self.identity_hash, event=0, valid_event=False,
            early10=0, valid_early=False, reason='UNPROVEN_CURRENT_ACCOUNT',
            observed_market_bars=0, first_adverse_market_bar=None,
            tp_market_minutes=None, ambiguous_intrabar=False)
        reason = outcome_account_reason(env, mask, snapshot, rules=self.rules,
            market=self.query.market, floor=self.floor, covered=env.current_step in self.rows)
        if reason is None:
            result.update(self.labels[env.current_step])
            c = self.rows[env.current_step]
            result['candidate_identity'] = fingerprint(c)
            result['funding_proof'] = dict(equity=str(env._calculate_equity()),
                required_equity=str(self.required_equity), floor=str(self.floor))
            reason = result['reason']
        result['reason'] = reason
        return signed(result)

    def accepts_identity(self, contract, identity):
        return contract == self.rules['contract'] and identity == self.identity_hash

    def write_receipt(self, output):
        write_json(output, signed(dict(identity=self.identity, identity_hash=self.identity_hash)))


class PoolOutcomeCatalog:
    """Dispatch existing certified outcomes by the active TRAIN contract."""
    def __init__(self, catalogs, oracle_pool_identity):
        self.catalogs = dict(catalogs)
        if not self.catalogs or any(c.identity['fold'] != 'TRAIN' or c.rules['contract'] != contract
                for contract, c in self.catalogs.items()):
            raise ValueError('Outcome pool requires matching certified TRAIN members')
        self.identity = dict(protocol='FG_S3_B2_OUTCOME_POOL_MARKET_MINUTE_V1', fold='TRAIN',
            clock_protocol='S3_MARKET_MINUTE', horizon_market_bars=60, events=EVENTS,
            oracle_pool_identity=oracle_pool_identity,
            catalogs={contract: c.identity_hash for contract, c in sorted(self.catalogs.items())})
        self.identity_hash = fingerprint(self.identity)

    def query_state(self, env, mask, *, snapshot):
        if env.contract not in self.catalogs:
            raise ValueError('Active account escaped the admitted TRAIN Outcome pool')
        return self.catalogs[env.contract].query_state(env, mask, snapshot=snapshot)

    def accepts_identity(self, contract, identity):
        return contract in self.catalogs and self.catalogs[contract].accepts_identity(contract, identity)

    def write_receipt(self, output):
        write_json(output, signed(dict(identity=self.identity, identity_hash=self.identity_hash)))


def attach_outcome(batch, records, catalog):
    """Called under the existing teacher publication lock, after teacher join."""
    if any(k in batch for k in FIELDS):
        raise ValueError('Outcome batch was already joined')
    ids = np.asarray(batch['stepid'])
    shape = ids.shape[:2]
    index = {r['stepid']: r for r in records}
    fields = {k: np.zeros(shape, bool if k.endswith('valid') else np.int32) for k in FIELDS}
    for pos in np.ndindex(shape):
        row = index[ids[pos].tobytes().hex()]
        result = row['outcome']
        if (not catalog.accepts_identity(row['contract'], result['catalog_identity'])
                or result['state_id'] != row['state_id']
                or fingerprint({k:v for k,v in result.items() if k != 'content_hash'}) != result['content_hash']):
            raise ValueError('Outcome Sidecar proof or account identity differs')
        advanced = bool(row['execution_result']['advanced'])
        valid_event, valid_early = (advanced and result[k] for k in ('valid_event', 'valid_early'))
        if (valid_event or valid_early) and not all(row['action_mask']):
            raise ValueError('Outcome supervision for a forced action')
        fields['outcome_event'][pos] = result['event']
        fields['outcome_event_valid'][pos] = valid_event
        fields['outcome_early10'][pos] = result['early10']
        fields['outcome_early_valid'][pos] = valid_early
    return {**batch, **fields}

"""TRAIN-only D0 suffix queries in the original funding proof's domain.

This is deliberately not an Oracle for arbitrary losing accounts. Unknown
states return an invalid sentinel, never a copied P4 label or an estimated Q.
"""
from decimal import Decimal
import gzip
import json
from pathlib import Path
import time

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import check_files, digest, read_manifest
from .fg_single3 import load_env
from .fg_single3_oracle import GAMMA, funds_certificate, optimize
from .fg_s3_reader_compat import PRODUCERS, reviewed_reader_transition


PROTOCOL = 'FG_S3_ORACLE_STATE_QUERY_MARKET_MINUTE_V2'
DECISION_PROTOCOL = 'FG_S3_V2_CURRENT_DECISION_STATE_V1'


def rules_identity(env):
    return dict(contract=env.contract, config_hash=env.config_hash,
                observation_identity=env.observation_identity,
                direction_hash=env.directions.hash, reward_hash=env.reward_config.hash)


def decision_state(env):
    """Private causal account descriptor shared by TRAIN and offline DEV proof."""
    state = dict(protocol=DECISION_PROTOCOL, contract=env.contract,
        config_hash=env.config_hash, current_step=env.current_step,
        cursor_time=str(env._raw.index[env.current_step]),
        account_sequence=env.account_sequence, entry_sequence=env.entry_sequence,
        status=env.status, terminated=env.terminated, truncated=env.truncated,
        pending_reason=env.pending_reason, net_lots=str(env.net_lots),
        equity=str(env._calculate_equity()), balances=env.ledger.snapshot(),
        current_price=str(env.current_price), action_mask=env._published_mask.tolist(),
        entries=env.entries, reward_active=sorted(env.reward_function.scorer.active),
        long_positions=[p.to_dict() if p else None for p in env.position_manager.long_positions],
        short_positions=[p.to_dict() if p else None for p in env.position_manager.short_positions],
        realized_pnl=str(env.user_accounts.realized_pnl), unrealized_pnl=str(env.user_accounts.unrealized_pnl),
        legal_add_decisions=env.legal_add_decisions, voluntary_waits=env.voluntary_waits,
        total_reward=env.total_reward)
    return json.loads(json.dumps(state, default=str))


def outcome_account_reason(env, mask, snapshot, *, rules, market, floor, covered):
    """Shared account proof only; this function does not publish teacher labels."""
    actual = decision_state(env) if snapshot.get('protocol') == DECISION_PROTOCOL else env.export_state()
    mask = np.asarray(mask)
    if rules_identity(env) != rules or not env._raw.equals(market):
        return 'RULE_OR_MARKET_IDENTITY_MISMATCH'
    if fingerprint(actual) != fingerprint(snapshot):
        return 'STALE_ACCOUNT_SNAPSHOT'
    if (mask.shape != (2,) or not np.array_equal(mask, env._published_mask)
            or not np.array_equal(mask, [1, 1])):
        return 'TWO_REAL_ACTIONS_REQUIRED'
    if env.status != 'ACTIVE' or env.terminated or env.truncated:
        return 'INACTIVE_ACCOUNT'
    if env.net_lots or env.entries or env.pending_reason or env.reward_function.scorer.active:
        return 'ACTUAL_ACCOUNT_NOT_FLAT'
    if env._calculate_equity() < floor:
        return 'UNPROVEN_CURRENT_FUNDS'
    if not covered:
        return 'CONTRACT_OR_GAP_BOUNDARY'
    return None


class OracleStateQuery:
    def __init__(self, env, candidates, end, *, fold, source_identity):
        if fold != 'TRAIN' or not env.single3:
            raise ValueError('OracleStateQuery requires the actual Single3 TRAIN Env')
        self.rules = rules_identity(env)
        self.market = env._raw.copy(deep=True)
        self.proof = funds_certificate(env)
        self.floor = Decimal(self.proof['retained_paths_cash_lower_bound'])
        self.end = int(end)
        if not 0 <= self.end < len(env._raw):
            raise ValueError('Oracle horizon must be the original contract boundary')
        started = time.perf_counter()
        plan = optimize(candidates, self.end, env.reward_config.candidate,
                        clock=env.market_clock_counter, return_suffix=True)
        self.dp_seconds = time.perf_counter() - started
        self.first = tuple(plan.pop('suffix')['first'])
        self.plan = plan
        self.identity = dict(protocol=PROTOCOL, fold=fold, rules=self.rules,
            source=source_identity, funds_certificate=self.proof,
            market_hash=fingerprint(self.market.to_dict(orient='split')),
            gamma_rational=[GAMMA.numerator, GAMMA.denominator], clock_protocol='S3_MARKET_MINUTE',
            boundary=self.end, tie_break='WAIT',
            certified_domain='flat_active_cash_at_least_original_certificate_floor',
            query_code_sha256=digest(__file__),
            optimizer_code_sha256=digest(Path(__file__).with_name('fg_single3_oracle.py')),
            suffix_first_hash=fingerprint(self.first))
        self.identity_hash = fingerprint(self.identity)

    def decision_state(self, env):
        """Current financial state for labels; FullCheckpoint keeps full history.

        The suffix certificate reads cursor, real availability, status, open
        positions/reward episodes and current equity. Include these exact
        inputs and financial balances, rather than serializing all historical
        prices, fills and rewards for every new decision. The complete frozen
        market is still checked separately by query(). This private descriptor
        is never forwarded to Actor observations or carry.
        """
        return decision_state(env)

    def query(self, env, action_mask, *, snapshot=None):
        """Read an actual Env decision state; never step, reset, or mutate it."""
        state = env.export_state() if snapshot is None else snapshot
        current = (self.decision_state if state.get('protocol') == DECISION_PROTOCOL
                   else lambda e: e.export_state())
        state_id = fingerprint(state)
        mask = np.asarray(action_mask)
        result = dict(action=0, valid=False, certified=False, reason='',
                      q_add=None, q_wait=None, q_margin=None,
                      state_id=state_id, oracle_identity=self.identity_hash)
        if (rules_identity(env) != self.rules or state['config_hash'] != env.config_hash
                or not env._raw.equals(self.market)):
            reason = 'RULE_OR_MARKET_IDENTITY_MISMATCH'
        elif snapshot is not None and fingerprint(current(env)) != state_id:
            reason = 'STALE_ACCOUNT_SNAPSHOT'
        elif (mask.shape != (2,) or not np.isin(mask, [0, 1]).all()
              or mask[0] != 1 or not np.array_equal(mask, env._published_mask)):
            reason = 'REAL_ACTION_MASK_MISMATCH'
        elif env.status != 'ACTIVE' or env.terminated or env.truncated:
            reason = 'NO_ACTIVE_DECISION'
        elif env.current_step >= self.end:
            reason = 'CONTRACT_OR_GAP_BOUNDARY'
        elif (env.net_lots or env.entries or env.pending_reason
              or env.reward_function.scorer.active):
            reason = 'NO_LEARNABLE_DECISION'
        elif env._calculate_equity() < self.floor:
            reason = 'FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE'
        elif not mask[1]:
            reason = 'NO_LEARNABLE_DECISION'
        else:
            result.update(action=int(self.first[env.current_step] == env.current_step),
                          valid=True, certified=True)
            reason = 'CERTIFIED_SUFFIX_DOMINANCE'
        result['reason'] = reason
        result['proof_hash'] = fingerprint(result)
        return result

    def accepts_identity(self, contract, identity_hash):
        return contract == self.rules['contract'] and identity_hash == self.identity_hash


def load_train_query(env_receipt, oracle_receipt, contract, *, receipt_scope='aggregate'):
    """Reuse frozen data, while checking producer Env bytes in this worktree.

    Legacy receipt paths can name the authorized data symlink's target. No
    external Python file is opened or imported as a code baseline.
    """
    import json
    began = time.perf_counter()
    eroot, oroot = Path(env_receipt).resolve(), Path(oracle_receipt).resolve()
    if receipt_scope not in ('aggregate', 'diagnostic_contract'):
        raise ValueError('Unknown Oracle receipt scope')
    em = read_manifest(eroot/'manifest.json')
    croot = oroot/'contracts'/contract
    cm = read_manifest(croot/'manifest.json')
    om_path = oroot/('manifest.json' if receipt_scope == 'aggregate' else 'inputs.json')
    om = read_manifest(om_path)
    if em['status'] != 'PASS' or em['stage'] != 'E' or em.get('clock_protocol') != 'S3_MARKET_MINUTE':
        raise ValueError('Oracle requires signed market-minute E evidence')
    if receipt_scope == 'aggregate':
        if om['status'] != 'PASS' or om['stage'] != 'O':
            raise ValueError('Oracle requires completed aggregate O evidence')
        parent = next(c for c in om['contracts'] if c['contract'] == contract)
        if {k: v for k, v in cm.items() if k != 'content_hash'} != parent:
            raise ValueError('Oracle contract receipt differs from parent')
        frozen_o = om['frozen_inputs']
    else:
        # A completed original per-contract job is a bounded diagnostic input,
        # never a fabricated PASS for the unfinished/full TRAIN fold.
        frozen_o = om['frozen_files']
        if cm['identity'] != om['identity']:
            raise ValueError('Oracle contract differs from its frozen job inputs')
    if (om['identity'].get('schema') != 'FG_SINGLE3_ORACLE_MARKET_MINUTE_V2'
            or om['identity'].get('clock_protocol') != 'S3_MARKET_MINUTE'
            or cm['stage'] != 'O_CONTRACT' or cm['status'] != 'PASS'
            or cm['oracle_optimality_status'] != 'exact_proven'):
        raise ValueError('Oracle requires new exact market-minute contract evidence')
    check_files({str(eroot/'manifest.json'): frozen_o[str(eroot/'manifest.json')]})
    check_files({str(croot/p): h for p, h in cm['files'].items()})
    check_files({str(eroot/'inputs/inputs.json'): em['files']['inputs/inputs.json']})
    inputs = read_manifest(eroot/'inputs/inputs.json')
    rows = [r for r in inputs['views']['C1']['rows'] if r['contract'] == contract]
    if len(rows) != 1 or rows[0]['role'] != 'train' or not rows[0]['eligible_for_training']:
        raise ValueError('D0 contract escaped the original TRAIN whitelist')
    row = rows[0]
    project = Path(__file__).resolve().parents[3]
    checked_code = {}
    reader_transitions = {}
    for old, expected in em['provenance']['code_files'].items():
        marker = '/src/gym_trading_env/'
        if marker not in old:
            continue
        rel = old.split(marker, 1)[1]
        if rel.startswith(('envs/', 'utils/')) or rel == 'rewards/single3_reward.py':
            current = project/'src/gym_trading_env'/rel
            actual = digest(current)
            name = 'gym_trading_env/'+rel
            if actual != expected and name in PRODUCERS:
                reader_transitions[name] = reviewed_reader_transition(project, name, expected, digest)
            else:
                check_files({str(current): expected})
            checked_code[str(current)] = actual
    if not checked_code:
        raise ValueError('Original Env producer evidence is missing')
    files = {str(croot/f): cm['files'][f] for f in ('candidates.jsonl.gz', 'C1_H333.json')}
    check_files(files)
    with gzip.open(croot/'candidates.jsonl.gz', 'rt') as stream:
        candidates = [json.loads(line) for line in stream]
    if len(candidates) != cm['candidates'] or [r['cursor'] for r in candidates] != list(range(cm['flat_end_cursor'])):
        raise ValueError('Cached candidate coverage changed')
    frozen = inputs['frozen_files']
    env = load_env(row, frozen)
    dependencies = [env.strategy_path] if env.strategy_path is not None else []
    dependency_hashes = {}
    for path in dependencies:
        path = Path(path).resolve()
        expected = (em['files'][str(path.relative_to(eroot))] if path.is_relative_to(eroot)
                    else frozen[str(path)])
        dependency_hashes[str(path)] = expected
    check_files(dependency_hashes)
    source = dict(contract=contract, market_sha256=row['sha256'],
        env_receipt_sha256=digest(eroot/'manifest.json'),
        oracle_receipt_sha256=digest(om_path), receipt_scope=receipt_scope,
        receipt_files={str(eroot/'manifest.json'): digest(eroot/'manifest.json'),
                       str(om_path): digest(om_path), str(croot/'manifest.json'): digest(croot/'manifest.json')},
        candidate_files=files, checked_env_code=checked_code)
    if reader_transitions:
        source['reviewed_reader_transitions'] = reader_transitions
    source.update(training_hash=inputs['original_content_hash'], strategy_files=dependency_hashes)
    query = OracleStateQuery(env, candidates, cm['flat_end_cursor'], fold='TRAIN', source_identity=source)
    saved = read_manifest(croot/'C1_H333.json')
    if query.proof != cm['funds_certificate'] or any(query.plan[k] != saved[k] for k in query.plan):
        raise ValueError('Current suffix DP or funding proof differs from audited C1 result')
    query.precompute_seconds = time.perf_counter() - began
    return query, env, row, frozen, candidates

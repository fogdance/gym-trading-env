"""Observe the existing FG account scheduler and dispatch original Oracle queries."""
from pathlib import Path

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.envs.entry_dataset_env import FGEntryDatasetEnv
from .fg_replay import execute_observed, read_manifest, signed, write_json
from .fg_s3_oracle_state_query import load_train_query


class ObservedAccount:
    """Delegate all financial behavior to the original single-contract Env."""
    def __init__(self, raw, capture):
        object.__setattr__(self, 'raw', raw)
        object.__setattr__(self, 'capture', capture)

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __setattr__(self, name, value):
        setattr(self.raw, name, value)

    def reset(self, *, seed=None, options=None):
        if options:
            return self.raw.reset(seed=seed, options=options)
        obs, row = execute_observed(self.raw, seed=seed)
        self.capture(row)
        return obs, self.raw._get_info()

    def step(self, action):
        obs, row = execute_observed(self.raw, action)
        self.capture(row)
        return obs, row['official_reward'], row['is_terminal'], self.raw.truncated, self.raw._get_info()


class ObservedTrainingPool(FGEntryDatasetEnv):
    """Use the original rotation/reset/save mechanics, observing each raw call."""
    def __init__(self, selector):
        self.last = None
        super().__init__(str(selector))

    @property
    def env(self):
        return self._observed_account

    @env.setter
    def env(self, raw):
        self._observed_account = ObservedAccount(raw, self.capture)

    @property
    def decision_env(self):
        return self.env.raw

    def capture(self, row):
        self.last = row

    def step(self, action):
        result = super().step(action)
        # The original scheduler may turn an ACTIVE raw return into CHUNK_END.
        # Reflect that scheduling status without changing reward or execution.
        self.last.update(status=self.env.status, is_last=bool(result[2] or result[3]))
        if self.last['bar'] is not None:
            self.last['bar']['chunk_end'] = self.env.status == 'CHUNK_END'
        return result


class PoolOracleQuery:
    """Select a certified original query by the active contract, never by cursor."""
    def __init__(self, queries, manifest_hash):
        self.queries = dict(queries)
        if not self.queries or any(q.identity['fold'] != 'TRAIN' for q in self.queries.values()):
            raise ValueError('Pool Oracle requires certified TRAIN members')
        self.rules = {c: q.rules for c, q in self.queries.items()}
        self.identity = dict(protocol='FG_S3_V2_ORACLE_POOL_V1', fold='TRAIN',
            manifest_hash=manifest_hash,
            queries={c:q.identity_hash for c,q in sorted(self.queries.items())})
        self.identity_hash = fingerprint(self.identity)

    def query(self, env, mask, *, snapshot=None):
        if env.contract not in self.queries:
            raise ValueError('Active account escaped the admitted TRAIN Oracle pool')
        return self.queries[env.contract].query(env, mask, snapshot=snapshot)

    def accepts_identity(self, contract, identity_hash):
        return contract in self.queries and self.queries[contract].identity_hash == identity_hash

    def decision_state(self, env):
        return self.queries[env.contract].decision_state(env)


def _load_members(settings, contracts, manifest_hash):
    queries, outcomes = {}, {}
    for contract in contracts:
        q, env, _, _, candidates = load_train_query(settings['env_receipt'], settings['oracle_receipt'], contract)
        try:
            queries[contract] = q
            if settings['mode'] == 'b2':
                from .fg_s3_b2_outcome_catalog import OutcomeCatalog
                outcomes[contract] = OutcomeCatalog(q, env, candidates)
        finally:
            env.close()
    query = PoolOracleQuery(queries, manifest_hash)
    if outcomes:
        from .fg_s3_b2_outcome_catalog import PoolOutcomeCatalog
        query.outcome_catalog = PoolOutcomeCatalog(outcomes, query.identity_hash)
    return query


def prepare_pool(settings, directory):
    """Read existing E/O caches; do not regenerate prices, labels or candidates."""
    inputs = read_manifest(Path(settings['env_receipt'])/'inputs/inputs.json')
    rows = inputs['views']['C1']['rows']
    selected = settings['contracts']
    if selected == 'ALL_TRAIN':
        selected = [r['contract'] for r in rows]
    if (not selected or len(selected) != len(set(selected))
            or set(selected) - {r['contract'] for r in rows}):
        raise ValueError('TRAIN pool members are not unique eligible original contracts')
    by_contract = {r['contract']:r for r in rows}
    chosen = [by_contract[c] for c in selected]
    if any(r['role'] != 'train' or not r['eligible_for_training'] for r in chosen):
        raise ValueError('Training pool includes an ineligible fold')
    manifest = signed(dict(product='FG', files=chosen, source_training_hash=inputs['original_content_hash']))
    write_json(directory/'train_pool.json', manifest)
    from dreamerv3.configuration import read_config_yaml
    import yaml
    selector = dict(strategy_mode='entry_dataset', role='train',
        manifest_path=str(directory/'train_pool.json'),
        scheduler=dict(mode='least_progress', chunk_bars=settings['env_slice_bars']))
    (directory/'selector.yaml').write_text(yaml.safe_dump(selector, sort_keys=False))
    # Use the common config reader to validate the actual saved selector.
    if read_config_yaml(directory/'selector.yaml') != selector:
        raise ValueError('Resolved TRAIN selector changed type')
    query = _load_members(settings, selected, manifest['content_hash'])
    write_json(directory/'oracle_pool.json', signed(dict(identity=query.identity)))
    if settings['mode'] == 'b2':
        query.outcome_catalog.write_receipt(directory/'outcome_catalog.json')
    return query


def load_pool(settings, directory):
    manifest = read_manifest(directory/'train_pool.json')
    expected = read_manifest(directory/'oracle_pool.json')['identity']
    query = _load_members(settings, [row['contract'] for row in manifest['files']], manifest['content_hash'])
    if query.identity != expected:
        raise ValueError('Prepared Oracle pool identity changed')
    if settings['mode'] == 'b2' and read_manifest(directory/'outcome_catalog.json')['identity_hash'] != query.outcome_catalog.identity_hash:
        raise ValueError('Prepared Outcome pool identity changed')
    pool = ObservedTrainingPool(directory/'selector.yaml')
    return query, pool

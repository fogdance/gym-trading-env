"""F3 role composition over the shared MultiCorpus stream and WP06 adapters."""
from __future__ import annotations

import copy

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_s3_e_quality_multi import MultiCorpusCourseStream


PROTOCOL = 'FG_S3_E_MULTICORPUS_F3_STREAM_V1'
STAT_FIELDS = ('windows', 'target_positions', 'teacher_add', 'teacher_wait', 'teacher_invalid')


class MultiCorpusF3Stream:
    @classmethod
    def open(cls, sources, *, settings, per_source_sampling_spec, source_mix_weights,
             seed, aux_per_source_sampling_spec=None, cache_bytes=64 << 20):
        return cls(sources, settings, per_source_sampling_spec, source_mix_weights,
                   seed, aux_per_source_sampling_spec, cache_bytes)

    def __init__(self, sources, settings, specs, weights, seed, aux_specs, cache_bytes):
        from .fg_s3_e_quality_actor import F3WindowAdapter
        from embodied.run.fg_s3_e_quality_actor import ActorStreams
        if (type(seed) is not int or seed < 0 or type(cache_bytes) is not int or cache_bytes < 0):
            raise ValueError('MULTI_F3_SEED_OR_CACHE_INVALID')
        if (not isinstance(specs, dict) or not specs or any(not isinstance(s, dict)
                or s.get('mode') != 'NATURAL' or set(s) - {'mode', 'policies'} for s in specs.values())):
            raise ValueError('MULTI_F3_NATURAL_SPECS_REQUIRED')
        role_specs = [('NATURAL', specs)]
        if aux_specs is not None:
            role_specs.append(('AUXILIARY', aux_specs))
        self.role_streams = {}
        records = []
        try:
            for number, (role, sampling) in enumerate(role_specs):
                role_seed = int(fingerprint(dict(protocol=PROTOCOL, seed=seed, role=role)), 16)
                core = MultiCorpusCourseStream.open(sources, stage='ACTOR_STRICT',
                    per_source_sampling_spec=sampling, source_mix_weights=weights, seed=role_seed,
                    cache_bytes=cache_bytes // len(role_specs) + int(number < cache_bytes % len(role_specs)))
                try:
                    bound = core.bind(settings, F3WindowAdapter)
                except BaseException:
                    core.close()
                    raise
                self.role_streams[role] = bound
                records.append(dict(role=role, seed=role_seed, stream=core.identity,
                                    stream_identity=core.identity_hash, adapter_identity=bound.identity))
            self._identity = dict(protocol=PROTOCOL, settings=copy.deepcopy(settings),
                roles=records, master_seed=seed, seed_rule='FINGERPRINT_MASTER_SEED_AND_F3_ROLE_V1',
                training_admitted=False)
            self.identity_hash = fingerprint(self._identity)
            self.group = ActorStreams.from_streams(list(self.role_streams.items()), identity=self._identity)
            self.statistics = {role: {name: dict.fromkeys(STAT_FIELDS, 0) for name in bound.stream.ids}
                               for role, bound in self.role_streams.items()}
            from .fg_s3_reader_transactions import TransactionGroup
            self._transactions=TransactionGroup(self,[self.group],lambda:self.statistics,
                lambda state:setattr(self,'statistics',state))
        except BaseException:
            for stream in self.role_streams.values():
                stream.close()
            raise

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    @property
    def last_batch_audit(self):
        return copy.deepcopy(self.group.last_batch_audit)

    @property
    def readers(self):
        return self.group.readers

    def sample(self):
        from .fg_s3_reader_transactions import batch_transaction
        with batch_transaction(self):
            data = next(self.group)
            audit = self.last_batch_audit
            start = 1 + self._identity['settings']['burnin_steps']
            for role, sources in self.statistics.items():
                for name, counts in sources.items():
                    positions = [i for i, row in enumerate(audit) if row['role'] == role and row['corpus_id'] == name]
                    actions = data['teacher_action'][positions, start:]
                    valid = data['teacher_valid'][positions, start:]
                    add = int(np.count_nonzero(valid & (actions == 1)))
                    wait = int(np.count_nonzero(valid & (actions == 0)))
                    values = dict(windows=len(positions), target_positions=int(valid.size),
                                  teacher_add=add, teacher_wait=wait, teacher_invalid=int(valid.size)-add-wait)
                    for key, value in values.items():
                        counts[key] += value
            return data, audit

    def begin_batch(self):return self._transactions.begin()
    def commit_batch(self,mark):self._transactions.commit(mark)
    def rollback_batch(self,mark):self._transactions.rollback(mark)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()[0]

    def save(self):
        return dict(protocol=PROTOCOL, identity=self.identity_hash, group=self.group.save(),
                    statistics=copy.deepcopy(self.statistics), last_batch_audit=self.last_batch_audit)

    save_full = state_dict = save

    def load(self, state):
        self.validate_snapshot(state)
        self.group.load(state['group'])
        self.statistics = copy.deepcopy(state['statistics'])

    load_full = load_state_dict = load

    def validate_snapshot(self, state=None):
        if state is None:
            return self.group.validate_snapshot()
        if (not isinstance(state, dict) or state.get('protocol') != PROTOCOL
                or state.get('identity') != self.identity_hash
                or not isinstance(state.get('group'), dict)
                or state.get('last_batch_audit') != state['group'].get('last_batch_audit')
                or set(state.get('statistics', {})) != set(self.role_streams)):
            raise ValueError('MULTI_F3_RESTORE_IDENTITY_MISMATCH')
        self.group.validate_snapshot(state['group'])
        for role, saved in zip(self.role_streams, state['group']['readers']):
            counts = state['statistics'][role]
            if set(counts) != set(self.statistics[role]):
                raise ValueError('MULTI_F3_STATISTICS_SOURCES_MISMATCH')
            for name, values in counts.items():
                windows = saved['stream']['source_counts'][name]['windows']
                if (set(values) != set(STAT_FIELDS) or any(type(v) is not int or v < 0 for v in values.values())
                        or values['windows'] != windows
                        or values['target_positions'] != windows*self._identity['settings']['target_steps']
                        or sum(values[k] for k in ('teacher_add', 'teacher_wait', 'teacher_invalid')) != values['target_positions']):
                    raise ValueError('MULTI_F3_STATISTICS_COUNTER_MISMATCH')

    def coverage(self):
        roles = self.group.coverage()
        natural = self.statistics['NATURAL']
        global_natural = {key: sum(s[key] for s in natural.values()) for key in STAT_FIELDS}
        reference = {name: {tag: sum(p.get(tag, 0) for p in reader.report['policies'].values())
                           for tag in ('STRICT_ADD', 'STRICT_WAIT')}
                     for name, reader in self.role_streams['NATURAL'].stream.readers.items()}
        return dict(protocol=PROTOCOL, identity=self.identity, identity_hash=self.identity_hash,
            roles=roles, cache_budget_bytes=sum(s.stream.cache_bytes for s in self.role_streams.values()),
            peak_cached_bytes_upper_bound=sum(r['peak_cached_bytes_upper_bound'] for r in roles.values()),
            sampling_statistics=copy.deepcopy(self.statistics),
            natural_distribution=dict(scope='DELIVERED_POST_BURNIN_TEACHER_LABEL_EXPOSURES',
                per_source=copy.deepcopy(natural), global_counts=global_natural,
                source_reference_account_labels=reference,
                source_selection=roles['NATURAL']['source_selection'],
                represents_true_market_natural_distribution=False,
                loss_or_optimizer_exposure='NOT_MEASURED_BY_DATA_LAYER'),
            denominators='Per-role, per-source counts; totals are delivered exposures, not unique market data.',
            actual_optimizer_exposures=0, training_admitted=False)

    def close(self):
        self.group.close()

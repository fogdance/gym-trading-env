"""Read-only multiplexing of WP03 Readers; no new Replay or training authority."""
from __future__ import annotations

from collections import Counter
import copy
import json

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import digest, replay_fields, E_QUALITY_SCHEMA
from .fg_s3_e_quality_reader import EQualityCourseReader, _stat
from .fg_s3_e_quality_supervision import Admission, inspect_source


PROTOCOL = 'FG_S3_E_MULTICORPUS_STREAM_V1'
ADAPTER_PROTOCOL = 'FG_S3_E_MULTICORPUS_WINDOW_ADAPTER_V1'
STAGES = dict(WM_NATURAL='FOUNDATION_WM', HEADS_VALID='QUALITY_HEADS',
              ACTOR_STRICT='ACTOR_SUPERVISED', ACTOR_QUALITY_PAIR='ACTOR_SUPERVISED')


def _clone(value):
    return copy.deepcopy(value)


def _seed(seed, domain):
    return int(fingerprint(dict(protocol=PROTOCOL, seed=seed, domain=domain)), 16)


def _merge(parts, size):
    """Only the batch axis may be assembled; never concatenate time axes."""
    keys = set(parts[0][1])
    first = parts[0][1]
    result = {k: np.empty((size, *v.shape[1:]), dtype=v.dtype) for k, v in first.items()}
    for positions, batch in parts:
        if set(batch) != keys:
            raise ValueError('MULTI_BATCH_FIELDS_DIFFER')
        for key, value in batch.items():
            if (not isinstance(value, np.ndarray) or value.dtype != first[key].dtype
                    or value.shape != (len(positions), *first[key].shape[1:])):
                raise ValueError('MULTI_BATCH_SHAPE_OR_DTYPE_DIFFERS: ' + key)
            result[key][positions] = value
    return result


class MultiCorpusCourseStream:
    @classmethod
    def open(cls, sources, *, stage, per_source_sampling_spec, source_mix_weights,
             seed, cache_bytes=64 << 20):
        return cls(sources, stage, per_source_sampling_spec, source_mix_weights, seed, cache_bytes)

    def __init__(self, sources, stage, specs, weights, seed, cache_bytes):
        if stage not in STAGES or type(seed) is not int or seed < 0:
            raise ValueError('MULTI_STAGE_OR_SEED_INVALID')
        if not isinstance(sources, list) or not sources:
            raise ValueError('MULTI_EXPLICIT_SOURCES_REQUIRED')
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError('MULTI_CACHE_BUDGET_INVALID')
        if any(not isinstance(s, dict) or set(s) != {'corpus_id', 'admission', 'index_path'} for s in sources):
            raise ValueError('MULTI_SOURCE_DESCRIPTOR_INVALID')
        ids = [s['corpus_id'] for s in sources]
        if any(not isinstance(i, str) or not i.strip() or i != i.strip() for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('MULTI_SOURCE_IDS_MUST_BE_UNIQUE')
        if (not isinstance(specs, dict) or set(specs) != set(ids)
                or not isinstance(weights, dict) or set(weights) != set(ids)):
            raise ValueError('MULTI_EXPLICIT_PER_SOURCE_SPECS_AND_WEIGHTS_REQUIRED')
        if any(type(weights[i]) not in (int, float) for i in ids):
            raise ValueError('MULTI_SOURCE_WEIGHTS_MUST_BE_POSITIVE_FINITE')
        try:
            values = np.asarray([weights[i] for i in ids], np.float64)
        except (OverflowError, ValueError) as exc:
            raise ValueError('MULTI_SOURCE_WEIGHTS_MUST_BE_POSITIVE_FINITE') from exc
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('MULTI_SOURCE_WEIGHTS_MUST_BE_POSITIVE_FINITE')
        values /= values.max()  # Avoid overflow in the sum of explicit weights.
        probabilities = values / values.sum()
        if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
            raise ValueError('MULTI_SOURCE_WEIGHTS_NOT_REPRESENTABLE')
        if any(not isinstance(s['admission'], Admission)
               or s['admission'].source.stage != STAGES[stage] for s in sources):
            raise ValueError('MULTI_ADMISSION_STAGE_MISMATCH')
        contents = [s['admission'].source.corpus_content_hash for s in sources]
        if len(set(contents)) != len(contents):
            raise ValueError('MULTI_DUPLICATE_CORPUS_CONTENT')
        self.ids = tuple(ids)
        self.stage = stage
        self.cache_bytes = cache_bytes
        self.readers = {}
        self._closed = self._failed = False
        self._receipt_stats = {}
        records = []
        compatibility = None
        try:
            for number, source in enumerate(sources):
                name = source['corpus_id']
                admission = source['admission']
                child_seed = _seed(seed, dict(kind='reader', corpus_id=name,
                                             content=admission.receipt['content_identity']))
                reader = EQualityCourseReader.open(admission, stage=stage, sampling_spec=specs[name],
                    seed=child_seed, index_path=source['index_path'],
                    cache_bytes=cache_bytes // len(ids) + int(number < cache_bytes % len(ids)))
                self.readers[name] = reader
                course = json.loads((reader.root/'course_v2.json').read_text())
                contract = dict(observation=reader.corpus['observation'],
                    model_contract=reader.corpus['model_contract_hash'], reward=course['reward'],
                    teacher=course['teacher'])
                if compatibility is not None and compatibility != contract:
                    raise ValueError('MULTI_INCOMPATIBLE_NATIVE_CONTRACTS')
                compatibility = contract
                receipt_path = reader.index_path.with_suffix('.json')
                receipt_sha = digest(receipt_path)
                self._receipt_stats[name] = (receipt_sha, _stat(receipt_path))
                records.append(dict(corpus_id=name, source_identity=reader.report['source_identity'],
                    corpus_identity=admission.source.corpus_content_hash,
                    course_identity=admission.source.course_identity_hash,
                    teacher_identity=admission.source.teacher_identity_hash,
                    source_lock=admission.source.source_lock_hash,
                    index_sha256=reader.index_hash, index_receipt_sha256=receipt_sha,
                    sampling_spec=_clone(reader.spec), weight=float(weights[name]), reader_seed=child_seed))
        except BaseException:
            self.close()
            raise
        self._probabilities = probabilities
        self._identity = dict(protocol=PROTOCOL, stage=stage, sources=records,
            seed=seed, seed_rule='FINGERPRINT_SEED_AND_STABLE_SOURCE_ID_V1',
            scheduler='IID_PER_BATCH_ROW_SOURCE_ORDER_V1', compatibility=compatibility,
            fold='TRAIN', training_admitted=False)
        self.identity_hash = fingerprint(self._identity)
        self.rng = np.random.default_rng(_seed(seed, dict(kind='scheduler')))
        self.counts = Counter(batches=0, windows=0, effective_positions=0)
        self.source_counts = {name: Counter(windows=0, effective_positions=0, calls=0) for name in ids}
        self.last_batch_audit = []

    @property
    def identity(self):
        return _clone(self._identity)

    def _ensure_open(self):
        if self._closed or self._failed:
            raise ValueError('MULTI_STREAM_INVALID_REOPEN_REQUIRED')

    def _guard(self):
        self._ensure_open()
        try:
            for name, reader in self.readers.items():
                reader._guard()
                expected, previous = self._receipt_stats[name]
                path = reader.index_path.with_suffix('.json')
                current = _stat(path)
                if current != previous:
                    if digest(path) != expected:
                        raise ValueError('MULTI_INDEX_RECEIPT_CHANGED: ' + name)
                    self._receipt_stats[name] = (expected, current)
        except BaseException:
            self._failed = True
            raise

    def support(self, effective_length):
        if type(effective_length) is not int or effective_length < 1:
            raise ValueError('MULTI_POSITIVE_LENGTH_REQUIRED')
        self._guard()
        result = {}
        for name, reader in self.readers.items():
            tags = [None, *sorted(reader.weights)]
            result[name] = {str(tag or self.stage): sum(g[-1] for g in reader._groups(effective_length, tag))
                            for tag in tags}
        return result

    def sample(self, batch_size, effective_length):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('MULTI_POSITIVE_BATCH_REQUIRED')
        support = self.support(effective_length)
        for name, groups in support.items():
            if not all(groups.values()):
                raise ValueError('INSUFFICIENT_CURRICULUM_SUPPORT: ' + name + ' ' + str(groups))
        before = self.state_dict()
        try:
            choices = self.rng.choice(len(self.ids), size=batch_size, p=self._probabilities)
            parts, audit = [], [None] * batch_size
            for number, name in enumerate(self.ids):
                positions = np.flatnonzero(choices == number)
                if not len(positions):
                    continue
                reader = self.readers[name]
                raw = reader.sample(len(positions), effective_length)
                metadata = reader.audit_windows(raw)
                parts.append((positions, raw))
                for position, native in zip(positions, metadata):
                    audit[int(position)] = dict(corpus_id=name,
                        source_identity=reader.report['source_identity'], index_sha256=reader.index_hash,
                        **native)
                self.source_counts[name].update(windows=len(positions),
                    effective_positions=len(positions)*effective_length, calls=1)
            batch = _merge(parts, batch_size)
            self.counts.update(batches=1, windows=batch_size,
                               effective_positions=batch_size*effective_length)
            self.last_batch_audit = _clone(audit)
            return batch, audit
        except BaseException:
            self._rollback(before)
            raise

    def audit_windows(self, batch, audit_rows):
        """Verify explicit provenance; stepids alone need not be unique across sources."""
        self._guard()
        if not isinstance(audit_rows, list) or len(audit_rows) != len(batch['stepid']):
            raise ValueError('MULTI_BATCH_PROVENANCE_REQUIRED')
        result = [None] * len(audit_rows)
        for name, reader in self.readers.items():
            positions = [i for i, row in enumerate(audit_rows) if row.get('corpus_id') == name]
            if not positions:
                continue
            native = reader.audit_windows({k: v[positions] for k, v in batch.items()})
            for position, row in zip(positions, native):
                expected = dict(corpus_id=name, source_identity=reader.report['source_identity'],
                                index_sha256=reader.index_hash, **row)
                if audit_rows[position] != expected:
                    raise ValueError('MULTI_WINDOW_PROVENANCE_MISMATCH')
                result[position] = expected
        if any(row is None for row in result):
            raise ValueError('MULTI_UNKNOWN_CORPUS_ID')
        return result

    def state_dict(self):
        return _clone(dict(protocol=PROTOCOL, identity=self.identity_hash,
            scheduler_rng=self.rng.bit_generator.state,
            readers={name: reader.state_dict() for name, reader in self.readers.items()},
            counts=dict(self.counts), source_counts={k: dict(v) for k, v in self.source_counts.items()},
            last_batch_audit=self.last_batch_audit))

    save = save_full = state_dict

    def _check_state(self, state):
        if (not isinstance(state, dict) or state.get('protocol') != PROTOCOL
                or state.get('identity') != self.identity_hash
                or set(state.get('readers', {})) != set(self.ids)
                or set(state.get('source_counts', {})) != set(self.ids)):
            raise ValueError('MULTI_RESTORE_IDENTITY_MISMATCH')
        rng = np.random.default_rng()
        rng.bit_generator.state = _clone(state['scheduler_rng'])
        for name, reader in self.readers.items():
            child = state['readers'][name]
            current = reader.state_dict()
            if any(child.get(k) != current[k] for k in ('protocol', 'source', 'index', 'stage', 'sampling_spec')):
                raise ValueError('MULTI_CHILD_RESTORE_IDENTITY_MISMATCH: ' + name)
            rng.bit_generator.state = _clone(child['rng'])
        for actual, keys in [(state['counts'], self.counts),
                             *((state['source_counts'][i], self.source_counts[i]) for i in self.ids)]:
            if set(actual) != set(keys) or any(type(v) is not int or v < 0 for v in actual.values()):
                raise ValueError('MULTI_RESTORE_COUNTER_INVALID')
        if not isinstance(state['last_batch_audit'], list):
            raise ValueError('MULTI_RESTORE_AUDIT_INVALID')
        for field in ('windows', 'effective_positions'):
            if state['counts'][field] != sum(c[field] for c in state['source_counts'].values()):
                raise ValueError('MULTI_RESTORE_COUNTER_TOTAL_MISMATCH')
        for name in self.ids:
            source = state['source_counts'][name]
            child = state['readers'][name]['counts']
            if (source['calls'] != child.get('batches', 0)
                    or source['effective_positions'] != child.get('delivered_candidate_exposures', 0)):
                raise ValueError('MULTI_RESTORE_CHILD_COUNTER_MISMATCH')

    def _apply_state(self, state):
        for name, reader in self.readers.items():
            reader.load_state_dict(state['readers'][name])
        self.rng.bit_generator.state = _clone(state['scheduler_rng'])
        self.counts = Counter(state['counts'])
        self.source_counts = {k: Counter(v) for k, v in state['source_counts'].items()}
        self.last_batch_audit = _clone(state['last_batch_audit'])

    def _rollback(self, state):
        try:
            self._apply_state(state)
        except BaseException:
            # Changed external bytes cannot be made valid by rewinding RNGs.
            self._failed = True

    def load_state_dict(self, state):
        self._guard()
        self._check_state(state)
        before = self.state_dict()
        try:
            self._apply_state(state)
        except BaseException:
            self._rollback(before)
            raise

    load = load_full = load_state_dict

    def validate_snapshot(self, state=None):
        self._guard()
        for name, reader in self.readers.items():
            inspect_source(reader.admission.source)
            if (digest(reader.index_path) != reader.index_hash
                    or digest(reader.index_path.with_suffix('.json')) != self._receipt_stats[name][0]):
                raise ValueError('MULTI_EXTERNAL_INDEX_CHANGED: ' + name)
        if state is not None:
            self._check_state(state)

    def coverage(self):
        self._guard()
        return dict(protocol=PROTOCOL, identity=self.identity, identity_hash=self.identity_hash,
            delivered_counts=dict(self.counts), per_source_counts={k: dict(v) for k, v in self.source_counts.items()},
            sources={name: reader.coverage() for name, reader in self.readers.items()},
            cache_budget_bytes=self.cache_bytes,
            peak_cached_bytes_upper_bound=sum(r.peak_cached_bytes for r in self.readers.values()),
            source_selection=dict(basis='EXPLICIT_CONFIGURED_SOURCE_WEIGHTS',
                probabilities={name: float(p) for name, p in zip(self.ids, self._probabilities)},
                delivered_window_shares={name: self.source_counts[name]['windows']/self.counts['windows']
                                         if self.counts['windows'] else None for name in self.ids},
                represents_true_market_natural_distribution=False),
            denominators='Per-source unique counts may overlap; totals here are delivered exposures.',
            actual_optimizer_exposures=0, training_admitted=False)

    def bind(self, settings, adapter_cls):
        return MultiCorpusWindowAdapter(self, settings, adapter_cls)

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()
        self._closed = True


class MultiCorpusWindowAdapter:
    """Generic lifecycle adapter; stage adapters continue to own field selection."""
    def __init__(self, stream, settings, adapter_cls):
        self.stream = stream
        self.settings = _clone(settings)
        self.adapter_cls = adapter_cls
        self.batch_size = settings['batch_size']
        target, burnin = settings['target_steps'], settings['burnin_steps']
        if any(type(v) is not int or v < 1 for v in (self.batch_size, target, burnin)):
            raise ValueError('MULTI_ADAPTER_POSITIVE_DIMENSIONS_REQUIRED')
        self.length = target + burnin
        self._adapters = {}
        for name in stream.ids:
            adapter = self._adapter(name, 1)
            if hasattr(adapter, 'validate_support'):
                adapter.validate_support()
        self.identity = fingerprint(dict(protocol=ADAPTER_PROTOCOL, stream=stream.identity_hash,
            settings=self.settings, adapter=adapter_cls.__module__+'.'+adapter_cls.__qualname__))

    def _adapter(self, name, size):
        key = (name, size)
        if key not in self._adapters:
            self._adapters[key] = self.adapter_cls(self.stream.readers[name],
                                                 {**self.settings, 'batch_size': size})
        return self._adapters[key]

    @property
    def readers(self):
        return tuple(self.stream.readers.values())

    @property
    def last_batch_audit(self):
        return _clone(self.stream.last_batch_audit)

    def coverage(self):
        return self.stream.coverage()

    def sample(self):
        before = self.stream.state_dict()
        try:
            raw, audit = self.stream.sample(self.batch_size, self.length)
            parts = []
            for name in self.stream.ids:
                positions = [i for i, row in enumerate(audit) if row['corpus_id'] == name]
                if positions:
                    subset = {k: v[positions] for k, v in raw.items()}
                    data, _ = self._adapter(name, len(positions)).adapt(subset)
                    for key in (*replay_fields(E_QUALITY_SCHEMA), 'stepid', 'consec'):
                        expected = raw[key][positions]
                        if (key not in data or data[key].dtype != expected.dtype
                                or data[key].shape != expected.shape
                                or data[key].tobytes() != expected.tobytes()):
                            raise ValueError('MULTI_ADAPTER_NATIVE_BYTES_CHANGED: ' + key)
                    parts.append((positions, data))
            return _merge(parts, self.batch_size), audit
        except BaseException:
            self.stream._rollback(before)
            raise

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()[0]

    def save(self):
        return dict(protocol=ADAPTER_PROTOCOL, identity=self.identity, stream=self.stream.state_dict(),
                    last_batch_audit=_clone(self.stream.last_batch_audit))

    save_full = state_dict = save

    def _check_state(self, state):
        if (not isinstance(state, dict) or state.get('protocol') != ADAPTER_PROTOCOL
                or state.get('identity') != self.identity or not isinstance(state.get('stream'), dict)
                or state.get('last_batch_audit') != state['stream'].get('last_batch_audit')):
            raise ValueError('MULTI_ADAPTER_RESTORE_IDENTITY_MISMATCH')

    def load(self, state):
        self._check_state(state)
        self.stream.load_state_dict(state['stream'])

    load_full = load_state_dict = load

    def validate_snapshot(self, state=None):
        if state is not None:
            self._check_state(state)
        self.stream.validate_snapshot(None if state is None else state['stream'])

    def close(self):
        self.stream.close()

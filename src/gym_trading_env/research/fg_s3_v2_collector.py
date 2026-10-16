"""Live DAgger publication over the already verified D0 Driver adapter."""
import copy
import json
from pathlib import Path
import threading
import uuid

import numpy as np

from .fg_s3_dagger_d0 import D0Collector, join_d0_teacher
from .fg_s3_v2_beta import BetaSchedule
from .fg_single3_evaluation import embodied_runtime


class V2Collector(D0Collector):
    output_directory = 'fg_s3_v2'
    max_env_steps = 10000  # Formal budgets additionally require entrypoint authorization.
    protocol = 'FG_S3_V2_LIVE_DAGGER_MARKET_MINUTE_V4'
    runtime = staticmethod(embodied_runtime)

    def __init__(self, *args, replay_capacity=2048, max_env_steps=10000, beta_schedule=None,
                 outcome_catalog=None, quality_catalog=None, **kwargs):
        if type(replay_capacity) is not int or replay_capacity < 65:
            raise ValueError('Live Replay capacity must retain a complete sequence')
        if type(max_env_steps) is not int or not 65 <= max_env_steps <= 1000000:
            raise ValueError('V2 requires an explicit diagnostic or 1M budget')
        self.replay_capacity = replay_capacity
        self.max_env_steps = max_env_steps
        self.commit_lock = threading.RLock()
        self.pending_native = None
        self.index = {}
        self._pruned_chunks = None
        self.commit_seq = 0
        self.model_versions = set()
        self.outcome_catalog = outcome_catalog
        self.quality_catalog = quality_catalog
        if quality_catalog is not None:
            self.protocol='FG_S3_QUALITY_LIVE_DAGGER_V2_1'
        if outcome_catalog is not None:
            self.protocol = 'FG_S3_B2_LIVE_DAGGER_MARKET_MINUTE_V1'
        initial_beta = kwargs.get('beta', .5) if kwargs.get('mode', 'mixed') == 'mixed' else 0.
        if not kwargs.get('teacher_enabled', True) and (
                initial_beta != 0 or beta_schedule is not None
                or outcome_catalog is not None or quality_catalog is not None):
            raise ValueError('A disabled teacher forbids mixing schedules and auxiliary catalogs')
        self.beta_schedule = BetaSchedule(initial_beta, beta_schedule)
        super().__init__(*args, **kwargs)
        self.identity['beta_schedule'] = self.beta_schedule.points
        if quality_catalog is not None:
            self.identity['quality_catalog']=quality_catalog.identity_hash
        if outcome_catalog is not None:
            self.identity['outcome_catalog'] = outcome_catalog.identity_hash
        from gym_trading_env.envs.entry_contract import fingerprint
        self.identity_hash = fingerprint(self.identity)
        self.journal = self.root/'teacher_journal.jsonl'
        if not kwargs.get('existing', False):
            self.journal.touch(exist_ok=False)

    @property
    def decision_env(self):
        return getattr(self.env, 'decision_env', self.env)

    def make_observed(self, env):
        from .fg_s3_v2_train_pool import ObservedTrainingPool
        return env if isinstance(env, ObservedTrainingPool) else super().make_observed(env)

    def make_replay(self, embodied, seed):
        return embodied.Replay(length=65, capacity=self.replay_capacity, online=True,
            directory=str(self.root/'replay'), chunksize=1024, save_wait=True, seed=seed)

    def context_snapshot(self):
        # Driver steps the previous action before asking for the next one;
        # capture() commits that feedback only after this policy call. Count
        # that real advance now, without counting reset/native rows as steps.
        completed = int(self.counts['env_steps']) + int(self.pending is not None
            and self.observed.last.get('advanced', False))
        self.policy.beta = self.beta_schedule.advance(completed)
        return self.oracle.decision_state(self.decision_env)

    def output_allowed(self, root):
        return super().output_allowed(root) or root.is_relative_to(Path('/data/logdir/fg_single3/s3_v2'))

    def advance(self, target):
        if not 1 <= target <= self.max_env_steps or target < self.counts['env_steps']:
            raise ValueError('Cumulative V2 environment budget is outside the frozen run')
        while self.counts['env_steps'] < target:
            self.driver(self.policy, steps=1)
            if self.decision_env.status in ('DATA_END', 'DATA_GAP'):
                raise RuntimeError('TRAIN lifecycle has an unresolved data boundary; refusing to discard its account')

    def _add_native(self, native, worker):
        if self.pending_native is not None:
            raise RuntimeError('Previous native row was not committed')
        self.pending_native = (copy.deepcopy(native), worker)
        return None

    def capture(self, step, worker):
        super().capture(step, worker)
        if self.quality_catalog is not None:
            self.pending['quality']=self.quality_catalog.label_for(self.decision_env,self.pending['state_id'])
            self.pending['quality_policy']={k:np.asarray(step[k]).tolist() for k in step if k.startswith('quality/')}
        if self.outcome_catalog is not None:
            self.pending['outcome'] = self.outcome_catalog.query_state(
                self.decision_env, self.pending['action_mask'],
                snapshot=self.policy.decision['context']['snapshot'])
            self.pending['b2_policy'] = {k: np.asarray(step[k]).tolist() for k in step if k.startswith('b2/')}
        self.pending['policy_model_update'] = int(self.agent.n_updates) if self.agent else None
        self.pending['beta_env_steps'] = self.beta_schedule.env_steps

    def _complete(self, result, next_context):
        if self.pending is None:
            return
        with self.commit_lock:
            super()._complete(result, next_context)
            row = self.records[-1]
            if self.pending_native is None:
                raise RuntimeError('Completed teacher has no native row')
            native, worker = self.pending_native
            stepid = super()._add_native(native, worker)
            self.commit_seq += 1
            row.update(stepid=stepid, commit_seq=self.commit_seq)
            if stepid in self.index:
                raise ValueError('Duplicate live Replay owner')
            self.index[stepid] = row
            version = row.get('policy_model_update')
            if version is not None:
                self.model_versions.add(version)
            if row['execution_result']['advanced']:
                self.counts['reward_sum'] += row['reward']
                if row['teacher_valid'] and row['teacher_action'] == 1:
                    self.counts['actor_oracle_ADD_tp'] += int(row['actor_proposal'] == 1)
                    self.counts['actor_oracle_ADD_fn'] += int(row['actor_proposal'] == 0)
                if self.outcome_catalog is not None:
                    from .fg_s3_b2_outcome_catalog import EVENTS
                    label = row['outcome']
                    self.counts['outcome_event_valid'] += int(label['valid_event'])
                    self.counts['outcome_early_valid'] += int(label['valid_early'])
                    self.counts['outcome_'+label['reason']] += 1
                    self.counts['ambiguous_intrabar'] += int(label['ambiguous_intrabar'])
                    if label['valid_event']:
                        self.counts['event_'+EVENTS[label['event']]] += 1
                    legal = all(row['action_mask']) and not row['is_last']
                    self.counts['b2_legal_decisions'] += int(legal)
                    if legal and row.get('b2_policy'):
                        p = row['b2_policy']
                        self.counts['b2_mix_active_decisions'] += int(p['b2/mix_value'] > 0)
                        self.counts['b2_nonzero_delta_decisions'] += int(abs(p['b2/delta_logit']) > 1e-8)
                        self.counts['b2_argmax_changed'] += int(p['b2/argmax_changed'])
            with self.journal.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            self.pending_native = None
            self.counts['published_native_rows'] += 1

    def prune(self):
        """Retain every physical Replay owner, including sequence prefixes.

        Historical decisions remain in the append-only reporting journal.
        The full checkpoint owns all teachers needed by its saved Replay;
        uncommitted rows remain separate and cannot enter the sampler.
        """
        with self.commit_lock:
            live = {bytes(k).hex() for k in self.replay.chunks}
            # _complete publishes every new row to records and index together.
            # Only a physical chunk change can retire one of those owners.
            if live != self._pruned_chunks:
                self.records = [r for r in self.records if r['stepid'][:32] in live]
                self.index = {r['stepid']: r for r in self.records}
                self._pruned_chunks = live
            self.query_seconds = self.query_seconds[-8192:]

    def sample(self, batch_size):
        # Only called after at least 65 completed rows. This lock also covers
        # teacher validation, so Prefetch cannot see half a publication.
        with self.commit_lock:
            batch = self.replay.sample(batch_size)
            return self.join(batch)

    def join(self, batch):
        with self.commit_lock:
            keys = dict.fromkeys(x.tobytes().hex() for x in np.asarray(batch['stepid']).reshape(-1, 20))
            if any(k not in self.index for k in keys):
                raise ValueError('Uncommitted or foreign live Replay owner')
            rows = [self.index[k] for k in keys]
            if any(not self.oracle.accepts_identity(r['contract'], r['oracle_identity']) for r in rows):
                raise ValueError('Foreign Oracle in committed sidecar')
            batch = join_d0_teacher(batch, rows, observation_schema=self.observation_schema)
            if self.outcome_catalog is not None:
                from .fg_s3_b2_outcome_catalog import attach_outcome
                batch = attach_outcome(batch, rows, self.outcome_catalog)
            if self.quality_catalog is not None:
                from .fg_s3_quality_catalog_v2 import attach_quality
                batch=attach_quality(batch,rows,teacher_identities=self.quality_catalog.teacher_identities)
            return batch

    def close(self):
        super().close()
        if self.quality_catalog is not None: self.quality_catalog.close()

    def save(self):
        with self.commit_lock:
            self.prune()
            if (self.beta_schedule.env_steps != self.counts['env_steps']
                    or self.policy.beta != self.beta_schedule.beta):
                raise ValueError('Beta schedule has not reached the committed Env boundary')
            return dict(protocol=self.protocol, identity=self.identity,
                records=copy.deepcopy(self.records), pending=copy.deepcopy(self.pending),
                pending_native=copy.deepcopy(self.pending_native), commit_seq=self.commit_seq,
                journal_bytes=self.journal.stat().st_size, model_versions=sorted(self.model_versions),
                counts=copy.deepcopy(self.counts), query_seconds=list(self.query_seconds),
                mixing=dict(beta=self.policy.beta, schedule=self.beta_schedule.save(),
                            rng=copy.deepcopy(self.policy.rng.bit_generator.state),
                            actor=self.policy.actor.save() if self.agent is None and
                                  hasattr(self.policy.actor, 'save') else None,
                            decision=copy.deepcopy(self.policy.decision)))

    def load(self, state):
        if state['protocol'] != self.protocol or state['identity'] != self.identity:
            raise ValueError('Live collector checkpoint identity differs')
        with self.commit_lock:
            self._pruned_chunks = None
            self.records, self.pending = state['records'], state['pending']
            self.pending_native, self.commit_seq = state['pending_native'], state['commit_seq']
            self.counts, self.query_seconds = state['counts'], state['query_seconds']
            self.index = {r['stepid']: r for r in self.records}
            seqs = [r['commit_seq'] for r in self.records]
            owners = {x.tobytes().hex() for chunk in self.replay.chunks.values()
                      if chunk.length for x in chunk.data['stepid'][:chunk.length]}
            if (len(self.index) != len(self.records) or set(self.index) != owners
                    or seqs != sorted(set(seqs)) or (seqs and seqs[-1] != self.commit_seq)):
                raise ValueError('Sidecar committed boundary differs')
            self.model_versions = set(state['model_versions'])
            # A crash may have written diagnostics after the last complete
            # checkpoint. Archive that tail before restoring its log boundary.
            boundary = state['journal_bytes']
            if not self.journal.exists() or self.journal.stat().st_size < boundary:
                raise ValueError('Committed teacher journal is missing')
            with self.journal.open('r+b') as stream:
                stream.seek(boundary)
                tail = stream.read()
                if tail:
                    backup = self.root/'recovery_logs'; backup.mkdir(exist_ok=True)
                    (backup/('teacher_tail_'+uuid.uuid4().hex+'.jsonl')).write_bytes(tail)
                    stream.truncate(boundary)
            mix = state['mixing']
            self.beta_schedule.load(mix['schedule'], int(self.counts['env_steps']))
            self.policy.beta = self.beta_schedule.beta
            if mix['beta'] != self.policy.beta:
                raise ValueError('Live mixing beta differs from its restored schedule')
            self.policy.rng.bit_generator.state = copy.deepcopy(mix['rng'])
            if mix['actor'] is not None:
                if self.agent is not None or not hasattr(self.policy.actor, 'load'):
                    raise ValueError('Unexpected standalone Actor state in live checkpoint')
                self.policy.actor.load(mix['actor'])
            self.policy.decision = mix['decision']

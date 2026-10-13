"""Live DAgger publication over the already verified D0 Driver adapter."""
import copy
import threading

import numpy as np

from .fg_s3_dagger_d0 import D0Collector, join_d0_teacher
from .fg_single3_evaluation import embodied_runtime


class V2Collector(D0Collector):
    output_directory = 'fg_s3_v2'
    max_env_steps = 10000  # Diagnostic runs; no automatic official 1M path.
    protocol = 'FG_S3_V2_LIVE_DAGGER_V1'
    runtime = staticmethod(embodied_runtime)

    def __init__(self, *args, **kwargs):
        self.commit_lock = threading.RLock()
        self.pending_native = None
        self.index = {}
        self.commit_seq = 0
        super().__init__(*args, **kwargs)

    def _add_native(self, native, worker):
        if self.pending_native is not None:
            raise RuntimeError('Previous native row was not committed')
        self.pending_native = (copy.deepcopy(native), worker)
        return None

    def capture(self, step, worker):
        super().capture(step, worker)
        self.pending['policy_model_update'] = int(self.agent.n_updates) if self.agent else None

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
            self.pending_native = None
            self.counts['published_native_rows'] += 1

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
            if any(r['oracle_identity'] != self.oracle.identity_hash for r in rows):
                raise ValueError('Foreign Oracle in committed sidecar')
            return join_d0_teacher(batch, rows)

    def save(self):
        with self.commit_lock:
            return dict(protocol=self.protocol, identity=self.identity,
                records=copy.deepcopy(self.records), pending=copy.deepcopy(self.pending),
                pending_native=copy.deepcopy(self.pending_native), commit_seq=self.commit_seq,
                counts=copy.deepcopy(self.counts), query_seconds=list(self.query_seconds),
                mixing=dict(beta=self.policy.beta, rng=copy.deepcopy(self.policy.rng.bit_generator.state),
                            decision=copy.deepcopy(self.policy.decision)))

    def load(self, state):
        if state['protocol'] != self.protocol or state['identity'] != self.identity:
            raise ValueError('Live collector checkpoint identity differs')
        with self.commit_lock:
            self.records, self.pending = state['records'], state['pending']
            self.pending_native, self.commit_seq = state['pending_native'], state['commit_seq']
            self.counts, self.query_seconds = state['counts'], state['query_seconds']
            self.index = {r['stepid']: r for r in self.records}
            if len(self.index) != self.commit_seq or self.commit_seq != len(self.records):
                raise ValueError('Sidecar committed boundary differs')
            mix = state['mixing']
            if mix['beta'] != self.policy.beta:
                raise ValueError('Live mixing schedule differs')
            self.policy.rng.bit_generator.state = copy.deepcopy(mix['rng'])
            self.policy.decision = mix['decision']

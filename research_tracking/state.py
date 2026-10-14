"""Single-writer state and immutable event batches, owned by the tracker."""
import fcntl
import json
import os
from pathlib import Path
import tempfile
import uuid

from . import SCHEMA
from .io import TrackingError, atomic_json, digest, fingerprint, read_json
from .redaction import redact


class Store:
    def __init__(self, source, base):
        self.source = source
        base = Path(base).resolve()
        protected = (source.root, source.prepared, source.requested_root)
        if any(base.is_relative_to(p) for p in protected):
            raise TrackingError('Tracker output must be outside source Run and preparation')
        self.root = (base/source.ids['train']).resolve()
        if not self.root.is_relative_to(base) or any(self.root.is_relative_to(p) for p in protected):
            raise TrackingError('Tracker output symlink escaped its independent root')
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = None
        self.state = None

    def __enter__(self):
        self.lock = (self.root/'tracker.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise TrackingError('Another importer/follower owns this source Run') from exc
        path = self.root/'tracker_state.json'
        trace = self.source.trace()
        if path.exists():
            self.state = read_json(path)
            if (self.state['local_run_id'] != self.source.run_id
                    or self.state['source_identity_sha256'] != trace['source_identity_sha256']):
                self.__exit__(None, None, None)
                raise TrackingError('Stored tracker identity conflicts with source Run')
        else:
            self.state = dict(schema=SCHEMA, local_run_id=self.source.run_id,
                source_identity_sha256=trace['source_identity_sha256'], ids=self.source.ids,
                canonical=dict(local_update=0, cycle_env_steps=0, updates={}, cycles={}),
                monitor=dict(segment=0, initialized=False), next_sequences={},
                batches=[], evaluations={}, checkpoints={}, cloud_entity=None)
        atomic_json(self.root/'trace_identity.json', trace)
        atomic_json(self.root/'config_preview.json', self.source.configuration())
        return self

    def __exit__(self, *args):
        if self.lock and not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()

    def save(self):
        atomic_json(self.root/'tracker_state.json', self.state)

    def artifact(self, name, value):
        target = self.root/'artifacts'/name
        if not target.resolve().is_relative_to(self.root) or not target.resolve().is_relative_to((self.root/'artifacts').resolve()):
            raise TrackingError('Tracker artifact path escaped allowlist directory')
        atomic_json(target, redact(value))
        return str(target.relative_to(self.root))

    def queue(self, job_type, run_id, events, *, config=None, artifacts=()):
        directory = self.root/'outbox'
        if not directory.resolve().is_relative_to(self.root):
            raise TrackingError('Outbox symlink escaped tracker root')
        directory.mkdir(exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.batch.', dir=directory)
        first = self.state['next_sequences'].get(run_id, 0)
        sequence, count = first, 0
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                for event in events:
                    payload = redact(event['metrics'])
                    event_hash = fingerprint(dict(kind=event['kind'], payload=payload))
                    record = dict(seq=sequence, kind=event['kind'], sha256=event_hash, metrics=payload)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                    sequence += 1
                    count += 1
                stream.flush()
                os.fsync(stream.fileno())
            if not count:
                return None
            token = uuid.uuid4().hex[:12]
            target = directory/f'{job_type}_{first}_{sequence-1}_{token}.jsonl'
            os.replace(temporary, target)
            batch = dict(job_type=job_type, run_id=run_id, first_seq=first, last_seq=sequence-1,
                count=count, file=str(target.relative_to(self.root)), sha256=digest(target),
                config=redact(config or {}), artifacts=list(artifacts), exported=False, synced=False)
            self.state['batches'].append(batch)
            self.state['next_sequences'][run_id] = sequence
            return batch
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def summary(self):
        return dict(local_run_id=self.source.run_id, tracker_dir=str(self.root),
            wandb_ids=self.state['ids'], committed_local_update=self.state['canonical']['local_update'],
            committed_env_steps=self.state['canonical']['cycle_env_steps'],
            monitor_segment=self.state['monitor']['segment'], evaluation_count=len(self.state['evaluations']),
            queued_events=sum(b['count'] for b in self.state['batches']),
            exported_events=sum(b['count'] for b in self.state['batches'] if b['exported']),
            synced_events=sum(b['count'] for b in self.state['batches'] if b['synced']))

"""A failed network transport cannot mutate or stop a source trainer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from . import PROJECT
from .auth import readiness
from .io import TrackingError, atomic_json, read_json
from .redaction import safe_error


class Writer:
    def __init__(self, mode, entity=None):
        if mode not in ('offline', 'online', 'disabled'):
            raise TrackingError('Unknown W&B tracking mode')
        self.mode = mode
        self.entity = entity

    def _process(self, store, command, *, phase, **kwargs):
        try:
            return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=600, **kwargs)
        except (subprocess.TimeoutExpired, TimeoutError, OSError) as exc:
            # SDK sync errors can include a whole history payload; retain only
            # bounded, redacted diagnostics in the independent tracker root.
            detail = getattr(exc, 'stderr', '') or str(exc)
            if isinstance(detail, bytes):
                detail = detail.decode('utf-8', errors='replace')
            result = dict(status='TRANSPORT_FAILED', phase=phase,
                          error=safe_error(detail[-1000:]))
            atomic_json(store.root/'last_transport_error.json', result)
            raise TrackingError('W&B transport failed; durable batches remain queued: '+result['error']) from exc

    def _transport(self, store, batch, mode):
        configuration = store.source.wandb_configuration()
        configuration.update(batch['config'])
        tracker_meta = store.root/'tracker_provenance.json'
        if tracker_meta.exists():
            configuration['tracker'] = read_json(tracker_meta)
        token = uuid.uuid4().hex[:12]
        spec_path = store.root/'transport'/f'spec_{token}.json'
        result_path = spec_path.with_name(f'result_{token}.json')
        scope = store.source.identity.get('scope') or ('diagnostic' if store.source.identity.get('diagnostic') else 'unknown')
        spec = dict(mode=mode, entity=self.entity, tracker_root=str(store.root), batch=batch,
            family=store.source.family,
            display_name=store.source.settings.get('experiment', store.source.family.removeprefix('s3-'))+' '+batch['job_type'],
            tags=[scope, 'clock:'+(store.source.clock or 'unknown'), store.source.identity['mode'],
                'dev' if batch['job_type'] == 'eval' else 'train',
                'beta:schedule' if store.source.identity.get('beta_schedule') else 'beta:constant',
                'source:'+store.source.identity['source_model_identity'][:12],
                'imported' if batch['job_type'] != 'live_monitor' else 'provisional'],
            configuration=configuration, result_path=str(result_path))
        atomic_json(spec_path, spec)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        if mode == 'offline':
            env.pop('WANDB_API_KEY', None)
        process = self._process(store, [sys.executable, '-m', 'research_tracking.sdk_worker', '--spec', str(spec_path)],
            phase='export' if mode == self.mode else 'offline_sync',
            cwd=Path(__file__).resolve().parents[1], env=env)
        result = read_json(result_path) if result_path.exists() else dict(
            status='TRANSPORT_FAILED', error=safe_error(process.stderr[-1000:]))
        expected_status = 'ONLINE_VERIFIED' if mode == 'online' else 'OFFLINE_EXPORTED'
        if process.returncode or result['status'] != expected_status:
            atomic_json(store.root/'last_transport_error.json', result)
            raise TrackingError('W&B transport failed; durable batches remain queued: '+
                safe_error(result.get('traceback') or result.get('error', 'unknown SDK error')))
        return result

    def export(self, store):
        if self.mode == 'disabled':
            return []
        if self.mode == 'online':
            pending_offline = [b for b in store.state['batches'] if b.get('export_mode') == 'offline' and not b['synced']]
            if pending_offline:
                raise TrackingError('Sync closed offline segments before continuing this Run online')
            status = readiness(self.entity)
            atomic_json(store.root/'cloud_readiness.json', status)
            if store.state['cloud_entity'] not in (None, self.entity):
                raise TrackingError('Stored Run is bound to a different Entity')
            store.state['cloud_entity'] = self.entity
        results = []
        (store.root/'sdk').mkdir(exist_ok=True)
        for batch in store.state['batches']:
            if batch['exported']:
                continue
            result = self._transport(store, batch, self.mode)
            batch.update(exported=True, export_mode=self.mode, transport=result,
                         synced=self.mode == 'online')
            store.save()
            results.append(result)
        return results

    def sync(self, store):
        status = readiness(self.entity)
        atomic_json(store.root/'cloud_readiness.json', status)
        if store.state['cloud_entity'] not in (None, self.entity):
            raise TrackingError('Cannot sync this local mapping into a different Entity')
        results = []
        for batch in store.state['batches']:
            if batch.get('export_mode') != 'offline' or batch['synced']:
                continue
            # Replay the immutable, redacted offline spool through the same SDK
            # transport as online mode. Legacy `wandb sync` sends this real
            # Tutor history in a single request that returned HTTP 500. The SDK
            # worker caps request size and confirms every event sequence/hash.
            result = self._transport(store, batch, 'online')
            batch['synced'] = True
            batch['sync_transport'] = result
            batch['cloud_url'] = f'https://wandb.ai/{self.entity}/{PROJECT}/runs/{batch["run_id"]}'
            store.state['cloud_entity'] = self.entity
            store.save()
            results.append(dict(status='OFFLINE_SYNC_VERIFIED', run_id=batch['run_id'],
                last_seq=batch['last_seq'], url=batch['cloud_url'], strategy='SDK_EVENT_REPLAY',
                sent_events=result['sent_events'], skipped_remote_events=result['skipped_remote_events']))
        return results


class MockWriter:
    """The test transport sees exactly the same redacted batches as the SDK."""
    def __init__(self):
        self.events = {}

    def export(self, store):
        for batch in store.state['batches']:
            if batch['exported']:
                continue
            with (store.root/batch['file']).open() as stream:
                rows = self.events.setdefault(batch['run_id'], [])
                rows.extend(map(json.loads, stream))
            batch.update(exported=True, export_mode='mock')
        store.save()
        return []

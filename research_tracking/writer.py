"""A failed network transport cannot mutate or stop a source trainer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from . import PROJECT
from .auth import credential, readiness
from .io import TrackingError, atomic_json, read_json
from .redaction import safe_error


class Writer:
    def __init__(self, mode, entity=None):
        if mode not in ('offline', 'online', 'disabled'):
            raise TrackingError('Unknown W&B tracking mode')
        self.mode = mode
        self.entity = entity

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
            configuration = store.source.wandb_configuration()
            configuration.update(batch['config'])
            tracker_meta = store.root/'tracker_provenance.json'
            if tracker_meta.exists():
                configuration['tracker'] = read_json(tracker_meta)
            token = uuid.uuid4().hex[:12]
            spec_path = store.root/'transport'/f'spec_{token}.json'
            result_path = spec_path.with_name(f'result_{token}.json')
            scope = store.source.identity.get('scope') or ('diagnostic' if store.source.identity.get('diagnostic') else 'unknown')
            spec = dict(mode=self.mode, entity=self.entity, tracker_root=str(store.root), batch=batch,
                family=store.source.family, display_name=store.source.family.removeprefix('s3-')+' '+batch['job_type'],
                tags=[scope, 'clock:'+(store.source.clock or 'unknown'), store.source.identity['mode'],
                    'dev' if batch['job_type'] == 'eval' else 'train',
                    'beta:schedule' if store.source.identity.get('beta_schedule') else 'beta:constant',
                    'source:'+store.source.identity['source_model_identity'][:12],
                    'imported' if batch['job_type'] != 'live_monitor' else 'provisional'],
                configuration=configuration, result_path=str(result_path))
            atomic_json(spec_path, spec)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            if self.mode == 'offline':
                env.pop('WANDB_API_KEY', None)
            process = subprocess.run([sys.executable, '-m', 'research_tracking.sdk_worker', '--spec', str(spec_path)],
                cwd=Path(__file__).resolve().parents[1], env=env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, timeout=180)
            if result_path.exists():
                result = read_json(result_path)
            else:
                result = dict(status='TRANSPORT_FAILED', error=safe_error(process.stderr[-1000:]))
            if process.returncode or result['status'] == 'TRANSPORT_FAILED':
                atomic_json(store.root/'last_transport_error.json', result)
                raise TrackingError('W&B transport failed; durable batches remain queued: ' +
                                    safe_error(result.get('traceback') or result.get('error', 'unknown SDK error')))
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
        import wandb
        from .sdk_worker import remote_history
        api = wandb.Api(api_key=credential(), timeout=15)
        results = []
        for batch in store.state['batches']:
            if batch.get('export_mode') != 'offline' or batch['synced']:
                continue
            directory = Path(batch['transport']['sdk_directory']).resolve()
            if not directory.is_relative_to(store.root/'sdk'):
                raise TrackingError('SDK sync directory escaped tracker root')
            # The source directory is never given to wandb sync.
            process = subprocess.run([str(Path(sys.executable).with_name('wandb')), 'sync', '--append',
                '--id', batch['run_id'], '--project', PROJECT, '--entity', self.entity,
                '--skip-console', str(directory)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=180)
            if process.returncode:
                raise TrackingError('Offline sync failed; keep the closed SDK segment for retry')
            _, remote = remote_history(api, self.entity, batch['run_id'], batch['first_seq'], batch['last_seq'])
            with (store.root/batch['file']).open() as stream:
                expected = {e['seq']: e['sha256'] for e in map(json.loads, stream)}
            if remote != expected:
                raise TrackingError('Synced event sequence/hash acknowledgement differs')
            batch['synced'] = True
            batch['cloud_url'] = f'https://wandb.ai/{self.entity}/{PROJECT}/runs/{batch["run_id"]}'
            store.state['cloud_entity'] = self.entity
            store.save()
            results.append(dict(status='OFFLINE_SYNC_VERIFIED', run_id=batch['run_id'],
                last_seq=batch['last_seq'], url=batch['cloud_url']))
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

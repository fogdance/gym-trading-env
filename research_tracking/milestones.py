"""Existing-Run archive milestones, using the shared tracker upload boundary."""
import fcntl
from pathlib import Path

from . import PROJECT
from .auth import credential, readiness
from .io import TrackingError, fingerprint, read_json
from .redaction import redact


def mapping(source, base, publication=None):
    matches = []
    for p in Path(base).glob('*/trace_identity.json'):
        trace = read_json(p)
        if (trace.get('local_run_id') != source['local_run_id']
                or trace.get('source_identity_sha256') != source['source_identity_sha256']
                or publication is not None and trace.get('publication') != publication):
            continue
        state = read_json(p.parent/'tracker_state.json')
        evaluation = state['evaluations'].get(source['eval_receipt_sha256'])
        if evaluation and evaluation['checkpoint_manifest_sha256'] == source['checkpoint_manifest_sha256']:
            matches.append(dict(tracker_dir=str(p.parent), entity=state['cloud_entity'],
                train_id=state['ids']['train'], eval_id=evaluation['run_id']))
    unique = {(m['entity'],m['train_id'],m['eval_id']) for m in matches}
    if len(unique) != 1:
        raise TrackingError('No unique authenticated tracker mapping; select --publication or import-evals')
    return matches[0]


def merge_tags(existing, additions):
    return sorted(set(existing or ())|set(additions))


def summary_fields(value, prefix):
    """W&B Cloud returns nested summary objects as dotted leaf fields."""
    rows = {}
    for name, item in value.items():
        key = prefix+'.'+name
        if isinstance(item,dict):
            rows.update(summary_fields(item,key))
        else:
            rows[key] = item
    return rows


def publish(record, mapped):
    import wandb
    entity = mapped['entity']
    readiness(entity)
    api = wandb.Api(api_key=credential(), timeout=15)
    path = f'https://wandb.ai/{entity}/{PROJECT}/runs/'
    evaluation = api.run(f'{entity}/{PROJECT}/{mapped["eval_id"]}')
    required = dict(receipt_sha256=record['receipt_sha256'],
        checkpoint_manifest_sha256=record['checkpoint_manifest_sha256'],
        parent_training_run_id=mapped['train_id'])
    if any(evaluation.config.get(k) != v for k,v in required.items()):
        raise TrackingError('Cloud Eval identity differs from authenticated tracker mapping')
    payload = redact(record)
    key = 's3_archive/'+record['idempotency_key']
    existing = evaluation.summary.get(key+'.receipt_sha256')
    if existing and (existing != record['receipt_sha256']
            or evaluation.summary.get(key+'.checkpoint_manifest_sha256') != record['checkpoint_manifest_sha256']):
        raise TrackingError('Existing cloud milestone identity conflicts')
    old_tags = tuple(evaluation.tags or ())
    evaluation.tags = merge_tags(old_tags, record['tags'])
    evaluation.update()
    fields = summary_fields(payload,key)
    evaluation.summary.update(fields)
    train_pending = False
    with (Path(mapped['tracker_dir'])/'tracker.lock').open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            train_pending = True
        else:
            try:
                train = api.run(f'{entity}/{PROJECT}/{mapped["train_id"]}')
                if train.config.get('local_run_id') != record['local_run_id']:
                    raise TrackingError('Cloud training Run identity differs')
                train.summary.update(fields)
            finally:
                fcntl.flock(stream,fcntl.LOCK_UN)
    fresh = wandb.Api(api_key=credential(), timeout=15).run(f'{entity}/{PROJECT}/{mapped["eval_id"]}')
    if (not set(record['tags']).issubset(fresh.tags) or not set(old_tags).issubset(fresh.tags)
            or any(fresh.summary.get(k) != v for k,v in fields.items())):
        raise TrackingError('Cloud milestone readback differs; retain pending task')
    return dict(status='PUBLISHED', eval_run_url=path+mapped['eval_id'], train_run_url=path+mapped['train_id'],
                cloud_readback_verified=True, parent_link_pending=train_pending,
                new_run_created=False, raw_files_uploaded=False)

"""Small verified summaries from posthoc DEV scoring, not a second tracker."""
from pathlib import Path

from .evals import parse_eval
from .io import TrackingError, digest, fingerprint, read_json
from .metrics import number


def parse_prediction_eval(source, path):
    receipt = read_json(path,signed=True,limit=source.limit)
    if (receipt.get('schema') != 'S3_DEV_PREDICTION_EVAL_V1'
            or receipt.get('status') != 'DEV_PREDICTION_EVALUATION_COMPLETE'
            or receipt.get('role') != 'DEV_EVALUATION_ONLY' or receipt.get('fold') != 'DEV'
            or receipt.get('eligible_for_training') is not False
            or receipt.get('oracle_files_read_in_policy') is not False
            or receipt.get('policy_observations_modified') is not False
            or receipt.get('future_data_read_posthoc') is not True
            or receipt.get('account_replay_exact') is not True
            or receipt.get('source_run_id') != source.run_id
            or receipt.get('clock_protocol') != source.clock):
        raise TrackingError('Invalid independent posthoc DEV prediction identity/role')
    kind = {'b2':'outcome_prediction','tutor':'tutor_scoring'}.get(source.identity['mode'])
    if kind is None or receipt.get('evaluation_kind') != kind:
        raise TrackingError('Prediction evaluation kind differs from trained model mode')
    actor_path = Path(receipt['source_eval_dir']).resolve()/'receipt.json'
    if not actor_path.is_relative_to(source.root/'evaluations') or digest(actor_path) != receipt['source_eval_receipt_sha256']:
        raise TrackingError('Prediction evaluation refers to a different Actor-only receipt')
    actor = parse_eval(source,actor_path)
    if (actor['checkpoint_manifest_sha256'] != receipt['checkpoint_manifest_sha256']
            or actor['receipt_content_hash'] != receipt['source_eval_receipt_content_hash']):
        raise TrackingError('Prediction and Actor-only Checkpoint identities differ')
    if receipt.get('source_files'):
        expected = {'receipt.json','actions.jsonl','account_state.json','dev_manifest.json','dev_env.yaml'}
        targets = {Path(name).resolve() for name in receipt['source_files']}
        if targets != {actor_path.parent/name for name in expected}:
            raise TrackingError('Prediction source file allowlist differs')
        if any(digest(name) != sha for name,sha in receipt['source_files'].items()):
            raise TrackingError('Sealed prediction source changed')
    run_id = source.publishing_id('prediction_eval',receipt['source_eval_receipt_sha256'],
        receipt['checkpoint_manifest_sha256'],receipt['schema'],
        fingerprint(receipt.get('posthoc_code_files',{})))
    metrics = {'eval/checkpoint_env_steps':actor['metrics']['eval/checkpoint_env_steps']}
    for k in ('certified_accounts','executed_decisions','effective_supervision_coverage'):
        metrics['eval/prediction/'+k] = receipt.get(k)
    for k,v in receipt.get('outcome',{}).items():
        if v is None or number(v):metrics['eval/outcome/'+k] = v
    for name,row in receipt.get('outcome',{}).get('classes',{}).items():
        for k in ('support','predictions','recall','precision'):
            metrics['eval/outcome/'+name+'/'+k] = row.get(k)
    for key in ('event_calibration','early_calibration'):
        metrics['eval/outcome/'+key+'_ece'] = receipt.get('outcome',{}).get(key,{}).get('ece')
    decision = receipt.get('decision',{})
    for k in ('legal_samples','argmax_changed','clip_saturation_fraction'):
        metrics['eval/decision/'+k] = decision.get(k)
    for k,v in decision.items():
        if isinstance(v,dict) and 'mean' in v:metrics['eval/decision/'+k+'_mean'] = v['mean']
    return dict(schema='FG_S3_TRACKER_FORECAST_DERIVED_V1',eval_run_id=run_id,
        receipt_sha256=digest(path),receipt_content_hash=receipt['content_hash'],
        checkpoint_manifest_sha256=receipt['checkpoint_manifest_sha256'],
        parent_training_run_id=source.ids['train'],parent_actor_eval_run_id=actor['eval_run_id'],
        clock_protocol=source.clock,contract=actor['contract'],fold='DEV',
        evaluation_kind=receipt['evaluation_kind'],metric_schema=receipt['schema'],
        oracle_files_read_in_policy=False,future_data_read_posthoc=True,eligible_for_training=False,
        posthoc_versions=receipt.get('posthoc_versions'),posthoc_code_files=receipt.get('posthoc_code_files'),
        calibration=receipt.get('outcome',{}),decision=decision,
        tutor=receipt.get('tutor'),metrics=metrics)

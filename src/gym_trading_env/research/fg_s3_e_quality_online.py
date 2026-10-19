"""Thin Gym execution boundary for E+Quality online practice."""
from __future__ import annotations

from dreamerv3.fg_s3_e_quality_online import PROTOCOL, validate_execution, execution_contract

CONFIG_SCHEMA='FG_S3_E_QUALITY_ONLINE_CONFIG_V1'


def preflight(spec, *, allow_engineering=False):
    from embodied.run.fg_s3_e_quality_online import inspect_f3
    from .fg_replay import signed
    from gym_trading_env.envs.entry_contract import fingerprint
    s = validate_execution(spec)
    proof, _ = inspect_f3(s, allow_engineering=allow_engineering)
    from .fg_s3_e_quality_supervision import SupervisionSource, QualitySupervisionAdmission
    original = proof['identity']['preparation']
    source = QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(
        original['settings']['corpus_root'], stage='ACTOR_SUPERVISED'))
    if (source.source.course_identity_hash != original['course']
            or source.receipt['content_identity'] != original['source']):
        raise ValueError('ONLINE_F3_CORPUS_OR_COURSE_CHANGED')
    return signed(dict(protocol=PROTOCOL, spec=s, execution_contract=execution_contract(s['execution_mode']),
        source=proof, execution_identity=fingerprint(dict(spec=s, source_manifest=proof['manifest_sha256'])),
        status='DEV_ONLY' if proof['engineering_only'] else 'SOURCE_VERIFIED_NOT_AUTHORIZED',
        training_admitted=False, research_admitted=False))


def read_settings(path):
    from dreamerv3.configuration import read_config_yaml
    from .fg_s3_e_quality_course_preflight import _local_path
    return validate_settings(read_config_yaml(_local_path(path)))


def validate_settings(raw):
    import copy
    import math
    from pathlib import Path
    from .fg_s3_e_quality_course_preflight import _local_path
    required={'schema','spec_mode','execution','prepared_dir','history_config','wm_policy'}
    if type(raw) is not dict or set(raw)!=required or raw['schema']!=CONFIG_SCHEMA:
        raise ValueError('ONLINE_CONFIG_EXPLICIT_FIELDS_REQUIRED')
    if raw['spec_mode'] not in ('draft','formal'):raise ValueError('ONLINE_EXPLICIT_DRAFT_OR_FORMAL_REQUIRED')
    result=copy.deepcopy(raw);spec=copy.deepcopy(raw['execution'])
    if not isinstance(spec,dict):raise ValueError('ONLINE_EXPLICIT_EXECUTION_FIELDS_REQUIRED')
    missing=[]
    for key in ('source_run','source_checkpoint'):
        if key not in spec:raise ValueError('ONLINE_EXPLICIT_EXECUTION_FIELDS_REQUIRED')
        if spec[key] is None and raw['spec_mode']=='draft':
            missing.append(key)
            # Pure structural validation only. These sentinels are never opened,
            # persisted as source identities or passed to an execution constructor.
            spec[key]='/tmp/wp07_contract_only_source'+('/ckpt/step_000000000001' if key=='source_checkpoint' else '')
    checked=validate_execution(spec)
    for key in missing:checked[key]=None
    result['execution']=checked;result['prepared_dir']=str(_local_path(raw['prepared_dir']))
    if raw['history_config'] is not None:result['history_config']=str(_local_path(raw['history_config']))
    if raw['spec_mode']=='formal' and checked['execution_mode']!='SHADOW' and raw['history_config'] is None:
        raise ValueError('ONLINE_FORMAL_HISTORY_CONFIG_REQUIRED')
    wm=raw['wm_policy']
    if wm is not None and (type(wm) is not dict or set(wm)!={'max_reward_mae','max_continue_mae'}
            or any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in wm.values())):
        raise ValueError('ONLINE_WM_CALIBRATION_POLICY_INVALID')
    output=Path(result['prepared_dir'])
    for candidate in (checked['run_dir'],checked['source_run']):
        if candidate is not None:
            other=Path(candidate)
            if other.is_relative_to(output) or output.is_relative_to(other):raise ValueError('ONLINE_PREPARED_PATH_OVERLAP')
    return result


def _live_inputs(proof,budget):
    """Reuse the F3 Corpus's authenticated TRAIN view and original E admission.

    This opens no arbitrary caller-supplied market file, constructs no new Corpus,
    and retains the original complete TRAIN suffix for delayed teachers.
    """
    import copy
    import json
    from pathlib import Path
    from .fg_replay import read_manifest
    from .fg_s3_e_quality_corpus_cli import source_inputs,_factory
    from e_data_v1.provider import ETrainAdmission
    corpus=Path(proof['identity']['preparation']['settings']['corpus_root'])
    saved=read_manifest(corpus/'source_lock.json')
    current,_,_=source_inputs(saved['source_pool']['path'],saved['env_receipt']['path'],saved['contract'],**saved.get('selection',{}))
    if current!=saved:raise ValueError('ONLINE_ORIGINAL_TRAIN_SOURCE_MISMATCH')
    if budget>saved['end_cursor']-saved['start_cursor']:raise ValueError('ONLINE_BUDGET_ESCAPES_CERTIFIED_VISIBLE_E_WINDOW')
    profile=ETrainAdmission.from_train_pool(saved['source_pool']['path'],corpus/'e_source',corpus/'selected_train_pool.json',contracts=[saved['contract']])
    row=copy.deepcopy(saved['row']);row['env_config']=str(corpus/'inputs'/saved['contract']/'env.yaml')
    initial=json.loads((corpus/'initial_account.json').read_text())
    if initial['current_step']!=saved['start_cursor']:raise ValueError('ONLINE_NATIVE_PREFIX_CURSOR_MISMATCH')
    return _factory(row,profile),saved,initial


def _history(settings,proof,*,native=False,producer_gym=None):
    from .fg_s3_e_quality_multi_checks import read_config,_sources,published_sources
    from .fg_s3_e_quality_multi_actor import MultiCorpusF3Stream
    from .fg_s3_e_quality_supervision import QualitySupervisionAdmission,SupervisionSource
    if settings['history_config'] is None:return None,[]
    config=read_config(settings['history_config'])
    if config['stage']!='ACTOR_STRICT':raise ValueError('ONLINE_HISTORY_F3_STAGE_REQUIRED')
    publications=published_sources(config)
    sources=_sources(config)
    if native:
        for descriptor in sources:
            original=descriptor['admission']
            descriptor['admission']=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(
                original.source.corpus_root,stage='ACTOR_SUPERVISED'),purpose='native-verification',producer_gym=producer_gym)
    geometry={k:proof['identity']['preparation']['settings'][k] for k in ('batch_size','burnin_steps','target_steps')}
    stream=MultiCorpusF3Stream.open(sources,settings=geometry,
        per_source_sampling_spec=config['per_source_sampling_spec'],source_mix_weights=config['source_mix_weights'],
        seed=config['seed'],aux_per_source_sampling_spec=config.get('f3_auxiliary_sampling_spec'),
        cache_bytes=config.get('cache_bytes',64<<20))
    try:validate_history_contract(stream,proof)
    except BaseException:stream.close();raise
    return stream,publications


def validate_history_contract(history,proof):
    import json
    from pathlib import Path
    from .fg_replay import read_manifest
    from .fg_s3_e_quality_multi import native_compatibility_contract
    origin=Path(proof['identity']['preparation']['settings']['corpus_root'])
    anchor=native_compatibility_contract(read_manifest(origin/'corpus_manifest.json'),
        json.loads((origin/'course_v2.json').read_text()))
    if any(role['stream']['compatibility']!=anchor for role in history.identity['roles']):
        raise ValueError('ONLINE_HISTORY_NATIVE_CONTRACT_MISMATCH')


def prepare(config,*,diagnostic_output=None):
    from pathlib import Path
    from .fg_replay import signed,write_json,digest
    from .fg_s3_e_quality_foundation import execution_identity
    from gym_trading_env.envs.entry_contract import fingerprint
    s=read_settings(config)
    if s['spec_mode']=='draft':
        blockers=['DRAFT_IS_DIAGNOSTIC_ONLY','SOURCE_F3_NOT_VERIFIED','LIVE_TRAIN_SOURCE_NOT_VERIFIED']
        blockers.extend('MISSING_'+key.upper() for key in ('source_run','source_checkpoint') if s['execution'][key] is None)
        if s['execution']['execution_mode']!='SHADOW':
            blockers.extend(['HISTORY_TRAIN_SOURCES_NOT_VERIFIED','NOT_AUTHORIZED_FOR_TRAINING'])
            if s['history_config'] is None:blockers.append('MISSING_HISTORY_CONFIG')
        if s['execution']['execution_mode']=='ONLINE_RL_CONTROLLED':
            blockers.append('INDEPENDENT_RL_RESEARCH_REVIEW_NOT_RUN')
            if s['wm_policy'] is None:blockers.append('WM_CALIBRATION_POLICY_NOT_PROVIDED')
        result=signed(dict(protocol=PROTOCOL,status='CONTRACT_ONLY',admission='NOT_ADMITTED',
            spec_mode='draft',settings=s,blockers=blockers,authenticated=False,training_admitted=False,
            prepared_created=False,checkpoint_source_authenticated=False,research_admitted=False))
        if diagnostic_output is not None:
            from .fg_s3_e_quality_course_preflight import _local_path
            out=_local_path(diagnostic_output)
            if out.exists():raise ValueError('ONLINE_DIAGNOSTIC_NEW_OUTPUT_REQUIRED')
            out.mkdir(parents=True);write_json(out/'contract_diagnostic.json',result)
        return result
    if diagnostic_output is not None:raise ValueError('ONLINE_FORMAL_OUTPUT_FROZEN_IN_CONFIG')
    # All source validation precedes Prepared/Run creation and model construction.
    receipt=preflight(s['execution']);proof=receipt['source']
    factory,source,initial=_live_inputs(proof,s['execution']['env_steps'])
    history,publications=_history(s,proof)
    try:
        identity=dict(settings=s,code=execution_identity(),source_manifest=proof['manifest_sha256'],
            live_source=source,initial_account=fingerprint(initial),
            history_identity=None if history is None else history.identity_hash,publications=publications,
            history_config_sha256=None if s['history_config'] is None else digest(Path(s['history_config'])))
    finally:
        if history is not None:history.close()
    out=Path(s['prepared_dir'])
    if out.exists():raise ValueError('ONLINE_NEW_PREPARED_OUTPUT_REQUIRED')
    result=signed(dict(protocol=PROTOCOL,status='SOURCE_VERIFIED_NOT_AUTHORIZED',identity=identity,
        blockers=['EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED','FRESH_NATIVE_G3_REQUIRED_FOR_UPDATES',
                  'INDEPENDENT_RL_RESEARCH_REVIEW_REQUIRED'],training_admitted=False,research_admitted=False))
    out.mkdir(parents=True);write_json(out/'settings.json',signed(dict(settings=s)));write_json(out/'prepared.json',result)
    return result


def start_prepared(path,*,resume=False,authorized=False,producer_gym=None,stop_at=None,
                   rl_review_path=None,rl_review_sha256=None,authorize_rl=False):
    from pathlib import Path
    from .fg_replay import read_manifest,write_json,signed,digest
    from .fg_s3_e_quality_course_preflight import _local_path
    from .fg_s3_e_quality_foundation import execution_identity
    from gym_trading_env.envs.entry_contract import fingerprint
    from embodied.run.fg_s3_e_quality_online_run import OnlineRun
    root=_local_path(path);p=read_manifest(root/'prepared.json');identity=p['identity'];s=validate_settings(identity['settings'])
    if (s['spec_mode']!='formal' or p.get('protocol')!=PROTOCOL or identity['code']!=execution_identity()
            or str(root)!=s['prepared_dir'] or read_manifest(root/'settings.json')['settings']!=s):
        raise ValueError('ONLINE_PREPARED_IDENTITY_MISMATCH')
    update=s['execution']['execution_mode']!='SHADOW'
    if update and authorized is not True:raise ValueError('ONLINE_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED')
    receipt=preflight(s['execution']);proof=receipt['source']
    if proof['manifest_sha256']!=identity['source_manifest']:raise ValueError('ONLINE_F3_SOURCE_CHANGED')
    factory,source,initial=_live_inputs(proof,s['execution']['env_steps'])
    if source!=identity['live_source'] or fingerprint(initial)!=identity['initial_account']:
        raise ValueError('ONLINE_PREPARED_LIVE_SOURCE_CHANGED')
    history,publications=_history(s,proof,native=update,producer_gym=producer_gym)
    run=None
    try:
        if (publications!=identity['publications'] or (None if history is None else history.identity_hash)!=identity['history_identity']
                or (None if s['history_config'] is None else digest(Path(s['history_config'])))!=identity['history_config_sha256']):
            raise ValueError('ONLINE_PREPARED_HISTORY_CHANGED')
        native=None
        if update:
            from .fg_s3_e_quality_actor import preflight as f3_preflight,require_training_authority
            native=f3_preflight(proof['identity']['preparation']['settings'],purpose='native-verification',producer_gym=producer_gym)
            require_training_authority(native,authorized=authorized)
        run=OnlineRun(s['execution'],factory,source,history,resume=resume,authorized=authorized,
            native_preparation=native,initial_state=initial,wm_policy=s['wm_policy'])
        if rl_review_path is not None or rl_review_sha256 is not None or authorize_rl:
            run.rl_review=run.gates.review_rl(native,rl_review_path,rl_review_sha256,authorized=authorize_rl)
        result=run.run_to(stop_at);write_json(run.root/'online_result.json',signed(result));return result
    finally:
        if run is not None:run.close()
        elif history is not None:history.close()


def report(run_dir,output=None):
    from pathlib import Path
    import pickle
    import elements
    from .fg_replay import read_manifest,signed,write_json,digest
    from .fg_s3_e_quality_course_preflight import _local_path
    from embodied.run.fg_checkpoint import FullCheckpoint
    root=_local_path(run_dir);identity=read_manifest(root/'run_identity.json')['identity']
    if identity.get('protocol')!='FG_S3_E_QUALITY_ONLINE_RUN_V1':raise ValueError('ONLINE_RUN_PROTOCOL_REQUIRED')
    node=FullCheckpoint(root/'ckpt',elements.Counter(),identity,(),strict_latest=True).latest()
    if node is None:raise ValueError('ONLINE_COMPLETED_NODE_REQUIRED')
    load=lambda name:pickle.loads((node/(name+'.pkl')).read_bytes())
    result=signed(dict(protocol=PROTOCOL,status='READ_ONLY_CHECKPOINT_REPORT',checkpoint=str(node),
        checkpoint_sha256=digest(node/'manifest.json'),execution_mode=identity['spec']['execution_mode'],
        agent_state_sha256=digest(node/'agent.pkl'),
        counters=load('counters'),gates=load('gates'),source_exposures=load('source_exposures'),
        market_steps=int(load('market_step')),source_manifest=identity['source_manifest'],
        data_identity=identity['live_source'],model_identity=identity['source_nonopt'],
        controlled_small=identity['controlled_small'],formal_training_authority='NOT_EVALUATED_BY_REPORT',
        research_admitted=False))
    if output is not None:
        out=_local_path(output)
        if out.exists():raise ValueError('ONLINE_REPORT_NEW_OUTPUT_REQUIRED')
        out.mkdir(parents=True);write_json(out/'online_report.json',result)
    return result


def evaluate_checkpoint(run_dir,checkpoint,output,*,steps=32):
    import pickle
    from pathlib import Path
    from .fg_replay import read_manifest,digest
    from .fg_s3_e_quality_course_preflight import _local_path
    from .fg_s3_e_quality_actor import create_agent
    from .fg_s3_e_quality_actor_checks import evaluate_native
    from .fg_s3_e_quality_foundation import execution_identity
    from embodied.run.fg_checkpoint import validate_bundle
    root=_local_path(run_dir);node=_local_path(checkpoint);out=_local_path(output)
    identity=read_manifest(root/'run_identity.json')['identity']
    if (identity.get('protocol')!='FG_S3_E_QUALITY_ONLINE_RUN_V1' or identity['code']!=execution_identity()
            or not node.is_relative_to(root/'ckpt')):raise ValueError('ONLINE_EVALUATION_CODE_OR_RUN_MISMATCH')
    validate_bundle(node,identity)
    from embodied.run.fg_s3_e_quality_online import inspect_f3
    proof,_=inspect_f3(identity['spec'],allow_engineering=identity['controlled_small'])
    if proof['manifest_sha256']!=identity['source_manifest']:raise ValueError('ONLINE_EVALUATION_SOURCE_CHANGED')
    # A public evaluator only accepts authenticated TRAIN. Synthetic probes call
    # the same original evaluate_native primitive with their fixed factory.
    if identity['controlled_small']:raise ValueError('ONLINE_PUBLIC_EVALUATION_REQUIRES_AUTHENTICATED_TRAIN')
    if type(steps) is not int or not 1<=steps<=512:raise ValueError('ONLINE_EVALUATION_FINITE_BUDGET_REQUIRED')
    factory,source,initial=_live_inputs(proof,steps)
    settings=proof['identity']['preparation']['settings'].copy();settings.update(device='cpu')
    settings['model_init']={**settings['model_init'],'seed':identity['spec']['model_seed']}
    # Full parameter schema is unchanged by Reader role batch geometry.
    agent=create_agent(settings,out/'model',execution_mode=identity['spec']['execution_mode'])
    agent.load_exact(pickle.loads((node/'agent.pkl').read_bytes()))
    return evaluate_native(agent,factory,out/'trace',steps=steps,initial_state=initial,
        source=dict(checkpoint_manifest_sha256=digest(node/'manifest.json'),live_source=source),
        scope='FINITE_TRAIN_READ_ONLY_NOT_RESEARCH')

"""F3 settings and thin adapters for original E+Quality components."""
from __future__ import annotations
import copy
from pathlib import Path
from dreamerv3.fg_s3_e_quality_actor import PROTOCOL,STAGE,SAMPLING_STAGE,model_extensions,validate_loss
from .fg_s3_e_quality_foundation import execution_identity
from .fg_s3_e_quality_course_preflight import _local_path
from .fg_s3_e_quality_supervision import require_stage_producer_pin,verifier_binding,require_prepared_verifier

CONFIG_SCHEMA='FG_S3_E_QUALITY_F3_CONFIG_V1'
_NATIVE_PROOFS={}


from dataclasses import dataclass
@dataclass(frozen=True)
class F3Preparation:
    settings: dict
    source: object
    receipt: dict


def preflight(settings,*,purpose='development',producer_gym=None,log_path=None,producer_lock=None,producer_lock_sha256=None,_resuming=False):
    require_stage_producer_pin(purpose,producer_lock,producer_lock_sha256)
    from .fg_s3_e_quality_supervision import SupervisionSource,QualitySupervisionAdmission
    from .fg_s3_e_observation import observation_contract
    from .fg_replay import signed
    from embodied.run.fg_s3_e_quality_transfer import inspect_stage
    from embodied.run.fg_checkpoint import plain
    from gym_trading_env.envs.entry_contract import fingerprint
    from dreamerv3.fg_s3_e_quality_actor import execution_contract
    s=validate_settings(settings)
    admission=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(s['corpus_root'],stage=STAGE),
        purpose=purpose,producer_gym=producer_gym,log_path=log_path,producer_lock=producer_lock,producer_lock_sha256=producer_lock_sha256)
    proof,_=inspect_stage(s,allow_engineering=True,_check_target=not _resuming);p=proof['identity']['preparation']
    if p['course']!=admission.source.course_identity_hash or p['source']!=admission.receipt['content_identity']:
        raise ValueError('F3_SOURCE_CORPUS_COURSE_MISMATCH')
    identity=plain(dict(protocol=PROTOCOL,settings=s,source=admission.receipt['content_identity'],
        course=admission.source.course_identity_hash,observation=observation_contract(),source_model=proof,
        effective_model=model_config(s,controlled_small=proof['engineering_only']),
        execution_contract=execution_contract(),execution_contract_hash=fingerprint(execution_contract()),
        code=execution_identity(),backend='external_immutable_course_reader'))
    blockers=['F3_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED']
    if proof['engineering_only']:blockers.append('FOUNDATION_NOT_TRAINED')
    if admission.native_result is None:blockers.append('F3_NATIVE_G3_NOT_RUN')
    receipt=signed(dict(protocol=PROTOCOL,status='F3_ENGINEERING_READY' if proof['engineering_only'] or admission.native_result is None else 'F3_DATA_READY',
        identity=identity,identity_hash=fingerprint(identity),blockers=blockers,training_admitted=False,research_admitted=False,source_receipt=admission.receipt,compatibility=admission.receipt['compatibility'],
        verifier_binding=verifier_binding(admission.native_result)))
    result=F3Preparation(s,admission,receipt)
    if admission.native_result is not None:_NATIVE_PROOFS[id(result)]=(result,fingerprint(admission.native_result),fingerprint(receipt))
    return result


def require_training_authority(preparation,*,authorized=False):
    from gym_trading_env.envs.entry_contract import fingerprint
    if authorized is not True:raise ValueError('F3_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED')
    if preparation.receipt['identity']['source_model']['engineering_only']:raise ValueError('F3_FOUNDATION_NOT_TRAINED')
    proof=_NATIVE_PROOFS.get(id(preparation))
    if (not proof or proof[0] is not preparation or proof[1]!=fingerprint(preparation.source.native_result)
            or proof[2]!=fingerprint(preparation.receipt) or preparation.receipt['status']!='F3_DATA_READY'):
        raise ValueError('F3_ACTUAL_NATIVE_EXECUTION_PROOF_REQUIRED')


def validate_settings(raw):
    from .fg_s3_e_quality_foundation import validate_settings as original
    required={'schema','stage','sampling_stage','sampling_spec','model_init','batch_size','target_steps','burnin_steps',
        'optimizer_updates','max_consumed_batches','report_every_updates','save_every_updates','device',
        'corpus_root','prepared_dir','run_dir','source_stage','source_run','source_checkpoint','actor_loss'}
    if type(raw) is not dict or not required<=raw.keys() or set(raw)-required-{'aux_sampling_spec'}:raise ValueError('F3_CONFIG_EXPLICIT_FIELDS_REQUIRED')
    if raw['schema']!=CONFIG_SCHEMA or raw['stage']!=STAGE or raw['sampling_stage']!=SAMPLING_STAGE:raise ValueError('F3_STAGE_OR_SCHEMA_MISMATCH')
    base={k:copy.deepcopy(v) for k,v in raw.items() if k not in ('max_consumed_batches','source_stage','source_run','source_checkpoint','actor_loss','aux_sampling_spec')}
    base.update(schema='FG_S3_E_QUALITY_F1_CONFIG_V1',stage='FOUNDATION_WM',sampling_stage='WM_NATURAL')
    result=original(base)
    if type(raw['max_consumed_batches']) is not int or raw['max_consumed_batches']<raw['optimizer_updates']:raise ValueError('F3_EXPLICIT_CONSUMPTION_BUDGET_REQUIRED')
    if raw['source_stage'] not in ('F1','F2'):raise ValueError('F3_SOURCE_STAGE_REQUIRED')
    aux=raw.get('aux_sampling_spec')
    if aux is not None:
        from .fg_s3_e_quality_heads import validate_settings as heads
        proxy={**base,'schema':'FG_S3_E_QUALITY_F2_CONFIG_V1','stage':'QUALITY_HEADS','sampling_stage':'HEADS_VALID',
            'sampling_spec':aux,'source_f1_run':raw['source_run'],'source_f1_checkpoint':raw['source_checkpoint']}
        heads(proxy)
    result.update(schema=CONFIG_SCHEMA,stage=STAGE,sampling_stage=SAMPLING_STAGE,
        max_consumed_batches=raw['max_consumed_batches'],source_stage=raw['source_stage'],
        actor_loss=validate_loss(raw['actor_loss']),aux_sampling_spec=copy.deepcopy(aux))
    for k in ('source_run','source_checkpoint'):result[k]=str(_local_path(raw[k]))
    if not Path(result['source_checkpoint']).is_relative_to(Path(result['source_run'])/'ckpt'):raise ValueError('F3_SOURCE_NODE_OUTSIDE_RUN')
    for out in ('prepared_dir','run_dir'):
        for src in ('corpus_root','source_run'):
            a,b=Path(result[out]),Path(result[src])
            if a.is_relative_to(b) or b.is_relative_to(a):raise ValueError('F3_OUTPUT_OVERLAPS_IMMUTABLE_SOURCE')
    return result


def read_settings(path):
    from dreamerv3.configuration import read_config_yaml
    return validate_settings(read_config_yaml(_local_path(path)))


def model_config(settings,*,controlled_small=False):
    from .fg_s3_e_quality_foundation import model_config as original
    cfg=original(settings,controlled_small=controlled_small)
    cfg['batch_size']=settings['batch_size']*(2 if settings.get('aux_sampling_spec') is not None else 1)
    return cfg


def create_agent(settings,output,*,controlled_small=False,execution_mode=None):
    from embodied.run.fg_s3_source_transfer import _make_agent
    from dreamerv3.fg_s3_v2_agent import _S3V2Model
    from dreamerv3.fg_s3_composition import resolve_composition
    from dreamerv3.fg_s3_e_quality_actor import register_state_schema
    extensions=model_extensions(settings['target_steps'],settings['burnin_steps'],settings['actor_loss'])
    if execution_mode is not None:
        from dreamerv3.fg_s3_e_quality_online import model_extensions as online_extensions
        extensions=online_extensions(settings['target_steps'],settings['burnin_steps'],execution_mode,settings['actor_loss'])
    agent=_make_agent(model_config(settings,controlled_small=controlled_small),output,agent_cls=_S3V2Model,
        device=settings['device'],bc_lambda=0.,extra_agent_config=extensions,
        composition=resolve_composition(dict(composition=dict(encoder='e_multiscale_1m15m',teaching='quality_v2'))))
    register_state_schema(agent.model,agent.params)
    return agent


def prepare(config,*,purpose='development',producer_gym=None,producer_lock=None,producer_lock_sha256=None):
    from .fg_replay import signed,write_json,digest
    from .fg_s3_e_quality_reader import build_join_index
    from embodied.run.fg_s3_e_quality_actor import ActorStreams
    s=read_settings(config);out=Path(s['prepared_dir'])
    if out.exists():raise ValueError('F3_PREPARATION_OUTPUT_MUST_BE_NEW')
    p=preflight(s,purpose=purpose,producer_gym=producer_gym,producer_lock=producer_lock,producer_lock_sha256=producer_lock_sha256,log_path=out/'native.log' if purpose=='native-verification' else None)
    joined=build_join_index(p.source,out/'join.sqlite');streams=ActorStreams(p,out/'join.sqlite')
    try:support={role:stream.adapter.validate_support() for role,stream in zip(streams.roles,streams.streams)}
    finally:streams.close()
    write_json(out/'settings.json',signed(dict(protocol=PROTOCOL,settings=p.settings)))
    write_json(out/'f3_preparation.json',p.receipt)
    write_json(out/'reader_binding.json',signed(dict(index_sha256=digest(out/'join.sqlite'),source_identity=p.source.receipt['content_identity'],join_report=joined,target_support=support)))
    if p.source.native_result is not None:write_json(out/'native_verification.json',p.source.native_result)
    return p.receipt


def start_prepared(path,*,resume=False,authorized=False,producer_gym=None,stop_at=None,producer_lock=None,producer_lock_sha256=None):
    if authorized is not True:raise ValueError('F3_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED')
    import uuid
    from .fg_replay import read_manifest,digest
    from .fg_s3_e_quality_foundation import PROJECT
    from embodied.run.fg_s3_e_quality_actor import ActorRun
    path=_local_path(path);s=validate_settings(read_manifest(path/'settings.json')['settings'])
    saved=read_manifest(path/'f3_preparation.json');binding=read_manifest(path/'reader_binding.json')
    if saved['identity']['source_model']['engineering_only']:raise ValueError('F3_FOUNDATION_NOT_TRAINED')
    if saved['status']!='F3_DATA_READY':raise ValueError('F3_NATIVE_PREPARATION_REQUIRED')
    require_prepared_verifier(saved,producer_gym=producer_gym,producer_lock=producer_lock,producer_lock_sha256=producer_lock_sha256)
    index=path/'join.sqlite'
    if digest(index)!=binding['index_sha256']:raise ValueError('F3_PREPARED_INDEX_CHANGED')
    root=Path(s['run_dir'])
    if not any(root.is_relative_to(p) for p in (PROJECT/'artifacts/fg_s3_v2',Path('/data/logdir/fg_single3/s3_v2'))):raise ValueError('F3_RUN_DIRECTORY_INVALID')
    audit=PROJECT/'artifacts/fg_s3_v2/e_quality_wp06/preflights'/uuid.uuid4().hex
    current=preflight(s,purpose='native-verification',producer_gym=producer_gym,producer_lock=producer_lock,producer_lock_sha256=producer_lock_sha256,log_path=audit/'native.log',_resuming=resume)
    if (current.receipt['verifier_binding']!=saved['verifier_binding'] or
            current.receipt['compatibility']!=saved['compatibility']):
        raise ValueError('SUPERVISION_PREPARED_SOURCE_OR_ABI_CHANGED')
    if (saved['identity']['source']!=current.source.receipt['content_identity'] or
            current.source.receipt['content_identity']!=binding['source_identity']):
        raise ValueError('F3_PREPARED_SOURCE_CHANGED')
    require_training_authority(current,authorized=True)
    return ActorRun(current,index,root,resume=resume,authorize_training=True).run(stop_at)


class F3WindowAdapter:
    """Loss-only labels from the existing V3 Join, with no second projection."""
    def __init__(self,reader,settings):
        from dreamerv3.fg_s3_e_quality_actor import dimensions
        if reader.stage!=SAMPLING_STAGE or reader.admission.source.stage!=STAGE:raise ValueError('F3_READER_STAGE_MISMATCH')
        self.reader,self.settings=reader,settings
        self.length=dimensions(settings['target_steps'],settings['burnin_steps'])
        self.target_support=None

    def validate_support(self):
        if self.target_support is not None:return self.target_support
        self.reader._guard();support={}
        tags=([None] if self.reader.mode in ('NATURAL','MIXED') else [])+sorted(self.reader.weights)
        for tag in tags:
            condition,extra=self.reader._condition(self.length,tag)
            query=('SELECT 1 FROM rows r WHERE '+condition+' AND EXISTS (SELECT 1 FROM eligibility a '
                'WHERE a.policy=r.policy AND a.segment=r.segment AND a.ordinal BETWEEN r.ordinal+? AND r.ordinal+? '
                'AND a.strict=1) LIMIT 1')
            supported=any(self.reader.db.execute(query,[policy,segment,lo,hi,*extra,
                1+self.settings['burnin_steps'],self.length]).fetchone() is not None
                for policy,segment,lo,hi,_ in self.reader._groups(self.length,tag))
            if not supported:raise ValueError('INSUFFICIENT_CURRICULUM_SUPPORT: F3_TARGET_L2 '+str(tag or self.reader.stage))
            support[str(tag or 'NATURAL')]=True
        self.target_support=dict(burnin_steps=self.settings['burnin_steps'],target_steps=self.settings['target_steps'],components=support)
        return self.target_support

    def adapt(self,raw):
        import numpy as np
        from dreamerv3.fg_s3_e_quality_actor import LABELS
        from .fg_replay import replay_fields,E_QUALITY_SCHEMA
        from .fg_s3_quality_catalog_v2 import FIELDS
        shape=(self.settings['batch_size'],self.length+1)
        if any(raw[k].any() for k in ('quality_q_c1_valid','quality_q_v2_valid','quality_soft_valid','quality_soft_source')):
            raise ValueError('F3_Q_SOFT_TARGETS_FORBIDDEN')
        if raw['stepid'].shape!=(*shape,20) or raw['consec'].dtype!=np.int32 or raw['consec'].shape!=shape or raw['consec'].any():raise ValueError('F3_NATIVE_WINDOW_GEOMETRY')
        labels=tuple(k for k in LABELS if k!='f3_natural')
        for key in labels:
            dtype,dims=FIELDS[key] if key in FIELDS else (np.dtype('int32') if key=='teacher_action' else np.dtype('bool'),())
            if raw[key].shape!=(*shape,*dims) or raw[key].dtype!=dtype:raise ValueError('F3_LABEL_SCHEMA: '+key)
        metadata=self.reader.audit_windows(raw,supervision_fields=labels)
        data={k:raw[k].copy() for k in (*replay_fields(E_QUALITY_SCHEMA),'stepid','consec',*labels)}
        valid=data['teacher_valid'];score_valid=data['quality_score_valid']
        if not np.isin(data['teacher_action'][valid],[0,1]).all():raise ValueError('F3_VALID_L2_ACTION_INVALID')
        score=data['quality_score'];known=score[score_valid]
        if not (np.isfinite(known)&(known>=0)&(known<=100)).all():raise ValueError('F3_VALID_SCORE_INVALID')
        data['teacher_action']=np.where(valid,data['teacher_action'],0).astype(np.int32)
        data['quality_score']=np.where(score_valid,score,0).astype(np.float32)
        data['f3_natural']=np.ones(shape,bool)
        for k,v in data.items():
            if np.issubdtype(v.dtype,np.floating) and not np.isfinite(v).all():raise ValueError('F3_NONFINITE_INPUT: '+k)
        return data,metadata

    def sample(self):
        self.validate_support()
        from .fg_s3_reader_transactions import batch_transaction
        with batch_transaction(self.reader):
            return self.adapt(self.reader.sample(self.settings['batch_size'],self.length))

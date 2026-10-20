"""F2 configuration and the loss-only boundary over the WP03 native Reader."""
from __future__ import annotations
import copy
from pathlib import Path
from dreamerv3.fg_s3_e_quality_heads import PROTOCOL,STAGE,SAMPLING_STAGE,LABELS,dimensions,model_extensions
from .fg_s3_e_quality_foundation import model_config,execution_identity
from .fg_s3_e_quality_course_preflight import _local_path
from .fg_s3_e_quality_supervision import require_stage_producer_pin

CONFIG_SCHEMA='FG_S3_E_QUALITY_F2_CONFIG_V1'

from dataclasses import dataclass
@dataclass(frozen=True)
class F2Preparation:
    settings: dict
    source: object
    receipt: dict

_NATIVE_PROOFS={}

def preflight(settings,*,purpose='development',producer_gym=None,log_path=None,producer_lock=None,producer_lock_sha256=None):
    require_stage_producer_pin(purpose,producer_lock,producer_lock_sha256)
    from .fg_s3_e_quality_supervision import SupervisionSource,QualitySupervisionAdmission
    from .fg_replay import signed
    from .fg_s3_e_observation import observation_contract
    from gym_trading_env.envs.entry_contract import fingerprint
    from embodied.run.fg_checkpoint import plain
    from embodied.run.fg_s3_e_quality_transfer import inspect_f1
    from dreamerv3.fg_s3_e_quality_heads import execution_contract
    settings=validate_settings(settings)
    admission=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(settings['corpus_root'],stage=STAGE),
        purpose=purpose,producer_gym=producer_gym,log_path=log_path,producer_lock=producer_lock,producer_lock_sha256=producer_lock_sha256)
    proof,_=inspect_f1(settings,allow_engineering=True)
    p=proof['identity']['preparation']
    if p['course']!=admission.source.course_identity_hash or p['source']!=admission.receipt['content_identity']:
        raise ValueError('F2_F1_CORPUS_COURSE_MISMATCH')
    identity=plain(dict(protocol=PROTOCOL,settings=settings,source=admission.receipt['content_identity'],
        course=admission.source.course_identity_hash,observation=observation_contract(),source_f1=proof,
        effective_model=model_config(settings,controlled_small=proof['engineering_only']),
        execution_contract=execution_contract(),execution_contract_hash=fingerprint(execution_contract()),
        code=execution_identity(),backend='external_immutable_course_reader'))
    blockers=['F2_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED']
    if proof['engineering_only']:blockers.append('FOUNDATION_NOT_TRAINED')
    if admission.native_result is None:blockers.append('F2_NATIVE_G3_NOT_RUN')
    receipt=signed(dict(protocol=PROTOCOL,status='F2_ENGINEERING_READY' if proof['engineering_only'] or admission.native_result is None else 'F2_DATA_READY',
        identity=identity,identity_hash=fingerprint(identity),blockers=blockers,training_admitted=False,research_admitted=False,
        source_receipt=admission.receipt))
    prep=F2Preparation(settings,admission,receipt)
    if admission.native_result is not None:_NATIVE_PROOFS[id(prep)]=(prep,fingerprint(admission.native_result),fingerprint(receipt))
    return prep

def require_training_authority(preparation,*,authorized=False):
    from gym_trading_env.envs.entry_contract import fingerprint
    if authorized is not True:raise ValueError('F2_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED')
    if preparation.receipt['identity']['source_f1']['engineering_only']:raise ValueError('F2_FOUNDATION_NOT_TRAINED')
    proof=_NATIVE_PROOFS.get(id(preparation))
    if (not proof or proof[0] is not preparation or proof[1]!=fingerprint(preparation.source.native_result)
            or proof[2]!=fingerprint(preparation.receipt) or preparation.receipt['status']!='F2_DATA_READY'):
        raise ValueError('F2_ACTUAL_NATIVE_EXECUTION_PROOF_REQUIRED')

def validate_settings(raw):
    from .fg_s3_e_quality_foundation import validate_settings as f1_validate
    required={'schema','stage','sampling_stage','sampling_spec','model_init','batch_size',
        'target_steps','burnin_steps','optimizer_updates','report_every_updates','save_every_updates',
        'device','corpus_root','prepared_dir','run_dir','source_f1_run','source_f1_checkpoint'}
    if type(raw) is not dict or not required<=raw.keys() or set(raw)-required-{'prediction_weights'}:
        raise ValueError('F2_CONFIG_EXPLICIT_FIELDS_REQUIRED')
    if (raw['schema']!=CONFIG_SCHEMA or raw['stage']!=STAGE or raw['sampling_stage']!=SAMPLING_STAGE):
        raise ValueError('F2_STAGE_OR_SCHEMA_MISMATCH')
    # Reuse native geometry/model/path checks, with explicit F2 sampling checks.
    base={k:copy.deepcopy(v) for k,v in raw.items() if k in required-{'source_f1_run','source_f1_checkpoint'}}
    base.update(schema='FG_S3_E_QUALITY_F1_CONFIG_V1',stage='FOUNDATION_WM',sampling_stage='WM_NATURAL',sampling_spec={'mode':'NATURAL'})
    result=f1_validate(base)
    spec=raw['sampling_spec'];mode=spec.get('mode') if type(spec) is dict else None
    keys={'mode','policies'}|({'strata_weights'} if mode=='STRATIFIED' else {'strata_weights','natural_weight'} if mode=='MIXED' else set())
    if mode not in ('NATURAL','STRATIFIED','MIXED') or set(spec)-keys:raise ValueError('F2_SAMPLING_INVALID')
    from .fg_s3_e_quality_supervision import POLICIES
    if 'policies' in spec and (type(spec['policies']) is not list or not spec['policies'] or len(set(spec['policies']))!=len(spec['policies']) or not set(spec['policies'])<=set(POLICIES)):
        raise ValueError('F2_POLICY_SELECTION_INVALID')
    import math
    if mode!='NATURAL':
        w=spec.get('strata_weights')
        if type(w) is not dict or not w or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in w.values()):raise ValueError('F2_EXPLICIT_STRATA_WEIGHTS_REQUIRED')
    if mode=='MIXED' and (type(spec.get('natural_weight')) not in (int,float) or not 0<spec['natural_weight']<1):raise ValueError('F2_MIX_WEIGHT_REQUIRED')
    weights=raw.get('prediction_weights',{})
    if type(weights) is not dict or set(weights)-{'quality_outcome_lambda','quality_score_lambda','quality_mono_lambda'}:raise ValueError('F2_PREDICTION_WEIGHTS_INVALID')
    model_extensions(raw['target_steps'],raw['burnin_steps'],weights)
    result.update(schema=CONFIG_SCHEMA,stage=STAGE,sampling_stage=SAMPLING_STAGE,sampling_spec=copy.deepcopy(spec),prediction_weights=copy.deepcopy(weights))
    for k in ('source_f1_run','source_f1_checkpoint'):result[k]=str(_local_path(raw[k]))
    for out in ('run_dir','prepared_dir'):
        for src in ('corpus_root','source_f1_run'):
            a,b=Path(result[out]),Path(result[src])
            if a.is_relative_to(b) or b.is_relative_to(a):raise ValueError('F2_OUTPUT_OVERLAPS_IMMUTABLE_SOURCE')
    return result

def read_settings(path):
    from dreamerv3.configuration import read_config_yaml
    return validate_settings(read_config_yaml(_local_path(path)))

def prepare(config,*,purpose='development',producer_gym=None,producer_lock=None,producer_lock_sha256=None):
    from .fg_replay import signed,write_json,digest
    from .fg_s3_e_quality_reader import build_join_index
    settings=read_settings(config);out=Path(settings['prepared_dir'])
    if out.exists():raise ValueError('F2_PREPARATION_OUTPUT_MUST_BE_NEW')
    p=preflight(settings,purpose=purpose,producer_gym=producer_gym,producer_lock=producer_lock,producer_lock_sha256=producer_lock_sha256,log_path=out/'native.log' if purpose=='native-verification' else None)
    joined=build_join_index(p.source,out/'join.sqlite')
    from .fg_s3_e_quality_reader import EQualityCourseReader
    reader=EQualityCourseReader.open(p.source,stage=SAMPLING_STAGE,sampling_spec=settings['sampling_spec'],seed=settings['model_init']['seed'],index_path=out/'join.sqlite')
    try:support=F2WindowAdapter(reader,settings).validate_support()
    finally:reader.close()
    write_json(out/'settings.json',signed(dict(protocol=PROTOCOL,settings=p.settings)))
    write_json(out/'f2_preparation.json',p.receipt)
    write_json(out/'reader_binding.json',signed(dict(index_sha256=digest(out/'join.sqlite'),source_identity=p.source.receipt['content_identity'],join_report=joined,target_support=support)))
    if p.source.native_result is not None:write_json(out/'native_verification.json',p.source.native_result)
    return p.receipt

def start_prepared(path,*,resume=False,authorized=False,producer_gym=None,stop_at=None,producer_lock=None,producer_lock_sha256=None):
    if authorized is not True:raise ValueError('F2_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED')
    import uuid
    from .fg_replay import read_manifest,digest
    from .fg_s3_e_quality_foundation import PROJECT
    path=_local_path(path);settings=validate_settings(read_manifest(path/'settings.json')['settings'])
    saved=read_manifest(path/'f2_preparation.json');binding=read_manifest(path/'reader_binding.json')
    if saved['identity']['source_f1']['engineering_only']:raise ValueError('F2_FOUNDATION_NOT_TRAINED')
    if saved['status']!='F2_DATA_READY':raise ValueError('F2_NATIVE_PREPARATION_REQUIRED')
    if saved['identity']['code']!=execution_identity():raise ValueError('F2_PREPARED_CODE_CHANGED')
    index=path/'join.sqlite'
    if digest(index)!=binding['index_sha256']:raise ValueError('F2_PREPARED_INDEX_CHANGED')
    root=Path(settings['run_dir'])
    if not any(root.is_relative_to(p) for p in (PROJECT/'artifacts/fg_s3_v2',Path('/data/logdir/fg_single3/s3_v2'))):raise ValueError('F2_RUN_DIRECTORY_INVALID')
    audit=PROJECT/'artifacts/fg_s3_v2/e_quality_wp05/preflights'/uuid.uuid4().hex
    current=preflight(settings,purpose='native-verification',producer_gym=producer_gym,producer_lock=producer_lock,producer_lock_sha256=producer_lock_sha256,log_path=audit/'native.log')
    if current.receipt['identity']!=saved['identity'] or current.source.receipt['content_identity']!=binding['source_identity']:raise ValueError('F2_PREPARED_SOURCE_OR_CONFIGURATION_CHANGED')
    require_training_authority(current,authorized=True)
    from embodied.run.fg_s3_e_quality_heads import HeadsRun
    return HeadsRun(current,index,root,resume=resume,authorize_training=True).run(stop_at)

def create_agent(settings,output,*,controlled_small=False):
    from embodied.run.fg_s3_source_transfer import _make_agent
    from dreamerv3.fg_s3_v2_agent import _S3V2Model
    from dreamerv3.fg_s3_composition import resolve_composition
    from dreamerv3.fg_s3_e_quality_foundation import parameter_layout
    agent=_make_agent(model_config(settings,controlled_small=controlled_small),output,agent_cls=_S3V2Model,
        device=settings['device'],bc_lambda=0.,extra_agent_config=model_extensions(settings['target_steps'],settings['burnin_steps'],settings.get('prediction_weights')),
        composition=resolve_composition(dict(composition=dict(encoder='e_multiscale_1m15m',teaching='quality_v2'))))
    parameter_layout(agent.params)
    from dreamerv3.fg_s3_e_quality_heads import register_state_schema
    register_state_schema(agent.model,agent.params)
    return agent

class F2WindowAdapter:
    def __init__(self,reader,settings):
        if reader.stage!=SAMPLING_STAGE or reader.admission.source.stage!=STAGE:raise ValueError('F2_READER_STAGE_MISMATCH')
        self.reader,self.settings=reader,settings
        self.length=dimensions(settings['target_steps'],settings['burnin_steps'])
        self.target_support=None

    def validate_support(self):
        """Prove target support in the existing index, without changing sampling.

        HEADS_VALID alone covers K+T. A view whose only labels are in K would
        otherwise consume/skip forever. Each requested sampling component must
        have at least one enabled target loss; this check consumes no RNG/rows.
        """
        if self.target_support is not None:return self.target_support
        self.reader._guard()
        cfg=model_extensions(self.settings['target_steps'],self.settings['burnin_steps'],self.settings.get('prediction_weights'))
        extract=lambda path:"COALESCE(json_extract(a.labels,'$."+path+"'),0)"
        terms=[]
        if cfg['quality_outcome_lambda']>0:
            paths=([f'quality_event_valid[{h}]' for h in range(7)]+
                [f'quality_touch_valid[{h}][{side}]' for h in range(4) for side in range(2)]+
                [f'quality_adverse_valid[{h}]' for h in range(4)])
            terms.extend(extract(path)+'=1' for path in paths)
        if cfg['quality_score_lambda']>0:terms.append(extract('quality_score_valid')+'=1')
        if cfg['quality_mono_lambda']>0:
            terms.append('('+'+'.join(extract(f'quality_event_valid[{h}]') for h in range(5))+')>=2')
        tags=([None] if self.reader.mode in ('NATURAL','MIXED') else [])+sorted(self.reader.weights)
        support={}
        for tag in tags:
            condition,extra=self.reader._condition(self.length,tag)
            query=('SELECT 1 FROM rows r WHERE '+condition+' AND EXISTS (SELECT 1 FROM rows a '
                'WHERE a.policy=r.policy AND a.segment=r.segment AND a.ordinal BETWEEN r.ordinal+? AND r.ordinal+? AND ('+
                ' OR '.join(terms)+')) LIMIT 1')
            supported=any(self.reader.db.execute(query,[policy,segment,lo,hi,*extra,
                1+self.settings['burnin_steps'],self.length]).fetchone() is not None
                for policy,segment,lo,hi,_ in self.reader._groups(self.length,tag))
            if not supported:raise ValueError('INSUFFICIENT_CURRICULUM_SUPPORT: F2_TARGET_LOSS '+str(tag or self.reader.stage))
            support[str(tag or 'NATURAL')]=True
        self.target_support=dict(burnin_steps=self.settings['burnin_steps'],target_steps=self.settings['target_steps'],components=support)
        return self.target_support

    def adapt(self,raw):
        import numpy as np
        from .fg_replay import replay_fields,E_QUALITY_SCHEMA
        from .fg_s3_quality_catalog_v2 import FIELDS
        shape=(self.settings['batch_size'],self.length+1)
        if any(raw[k].any() for k in ('quality_q_c1_valid','quality_q_v2_valid','quality_soft_valid','quality_soft_source')):
            raise ValueError('F2_Q_SOFT_TARGETS_FORBIDDEN')
        if raw['stepid'].shape!=(*shape,20) or raw['consec'].dtype!=np.int32 or raw['consec'].shape!=shape or raw['consec'].any():raise ValueError('F2_NATIVE_WINDOW_GEOMETRY')
        metadata=self.reader.audit_windows(raw)
        data={k:raw[k].copy() for k in (*replay_fields(E_QUALITY_SCHEMA),'stepid','consec')}
        for key in LABELS:
            dtype,dims=FIELDS[key]
            if raw[key].shape!=(*shape,*dims) or raw[key].dtype!=dtype:raise ValueError('F2_LABEL_SCHEMA: '+key)
            data[key]=raw[key].copy()
        for name in ('event','touch','adverse','score'):
            key='quality_'+name;valid=data[key+'_valid'];value=data[key]
            safe=np.where(valid,value,0)
            ok=np.isfinite(safe)
            if name=='score':ok&=(safe>=0)&(safe<=100)
            else:ok&=np.isin(safe,range(4 if name=='event' else 2))
            if not ok.all():raise ValueError('F2_VALID_TARGET_INVALID: '+name)
            data[key]=safe
        for k,v in data.items():
            if np.issubdtype(v.dtype,np.floating) and not np.isfinite(v).all():raise ValueError('F2_NONFINITE_INPUT: '+k)
        return data,metadata

    def sample(self):
        self.validate_support()
        state=self.reader.state_dict()
        try:return self.adapt(self.reader.sample(self.settings['batch_size'],self.length))
        except BaseException:
            self.reader.load_state_dict(state);raise

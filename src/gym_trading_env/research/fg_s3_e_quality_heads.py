"""F2 configuration and the loss-only boundary over the WP03 native Reader."""
from __future__ import annotations
import copy
from pathlib import Path
from dreamerv3.fg_s3_e_quality_heads import PROTOCOL,STAGE,SAMPLING_STAGE,LABELS,dimensions,model_extensions
from .fg_s3_e_quality_foundation import model_config,execution_identity
from .fg_s3_e_quality_course_preflight import _local_path

CONFIG_SCHEMA='FG_S3_E_QUALITY_F2_CONFIG_V1'

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

def create_agent(settings,output,*,controlled_small=False):
    from embodied.run.fg_s3_source_transfer import _make_agent
    from dreamerv3.fg_s3_v2_agent import _S3V2Model
    from dreamerv3.fg_s3_composition import resolve_composition
    from dreamerv3.fg_s3_e_quality_foundation import parameter_layout
    agent=_make_agent(model_config(settings,controlled_small=controlled_small),output,agent_cls=_S3V2Model,
        device=settings['device'],bc_lambda=0.,extra_agent_config=model_extensions(settings['target_steps'],settings['burnin_steps'],settings.get('prediction_weights')),
        composition=resolve_composition(dict(composition=dict(encoder='e_multiscale_1m15m',teaching='quality_v2'))))
    parameter_layout(agent.params)
    return agent

class F2WindowAdapter:
    def __init__(self,reader,settings):
        if reader.stage!=SAMPLING_STAGE or reader.admission.source.stage!=STAGE:raise ValueError('F2_READER_STAGE_MISMATCH')
        self.reader,self.settings=reader,settings
        self.length=dimensions(settings['target_steps'],settings['burnin_steps'])

    def adapt(self,raw):
        import numpy as np
        from .fg_replay import replay_fields,E_QUALITY_SCHEMA
        from .fg_s3_quality_catalog_v2 import FIELDS
        shape=(self.settings['batch_size'],self.length+1)
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
        state=self.reader.state_dict()
        try:return self.adapt(self.reader.sample(self.settings['batch_size'],self.length))
        except BaseException:
            self.reader.load_state_dict(state);raise

"""F3 settings and thin adapters for original E+Quality components."""
from __future__ import annotations
import copy
from pathlib import Path
from dreamerv3.fg_s3_e_quality_actor import PROTOCOL,STAGE,SAMPLING_STAGE,model_extensions,validate_loss
from .fg_s3_e_quality_foundation import execution_identity
from .fg_s3_e_quality_course_preflight import _local_path

CONFIG_SCHEMA='FG_S3_E_QUALITY_F3_CONFIG_V1'


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


def create_agent(settings,output,*,controlled_small=False):
    from embodied.run.fg_s3_source_transfer import _make_agent
    from dreamerv3.fg_s3_v2_agent import _S3V2Model
    from dreamerv3.fg_s3_composition import resolve_composition
    from dreamerv3.fg_s3_e_quality_actor import register_state_schema
    agent=_make_agent(model_config(settings,controlled_small=controlled_small),output,agent_cls=_S3V2Model,
        device=settings['device'],bc_lambda=0.,extra_agent_config=model_extensions(settings['target_steps'],settings['burnin_steps'],settings['actor_loss']),
        composition=resolve_composition(dict(composition=dict(encoder='e_multiscale_1m15m',teaching='quality_v2'))))
    register_state_schema(agent.model,agent.params)
    return agent

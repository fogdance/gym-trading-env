"""F1 preparation and source boundaries; no live Env or training permission.

Network execution stays in Dreamer. Dataset execution stays in the WP03 Reader.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import subprocess

from dreamerv3.fg_s3_e_quality_foundation import PROTOCOL, STAGE, SAMPLING_STAGE, dimensions
from dreamerv3.fg_s3_e_quality_course_contract import expected_stage_contract, REWARD_SCHEMA
from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import signed, write_json, read_manifest, digest
from .fg_s3_e_quality_course_preflight import _local_path
from .fg_s3_e_quality_supervision import SupervisionSource, QualitySupervisionAdmission, Admission
from .fg_s3_e_observation import observation_contract

PROJECT = Path(__file__).resolve().parents[3]
CONFIG_SCHEMA = 'FG_S3_E_QUALITY_F1_CONFIG_V1'


def validate_settings(raw):
    required = {'schema','stage','sampling_stage','sampling_spec','model_init','batch_size',
        'target_steps','burnin_steps','optimizer_updates','report_every_updates','save_every_updates',
        'device','corpus_root','prepared_dir','run_dir'}
    if type(raw) is not dict or set(raw) != required:
        raise ValueError('F1_CONFIG_EXPLICIT_FIELDS_REQUIRED')
    if (raw['schema'] != CONFIG_SCHEMA or raw['stage'] != STAGE
            or raw['sampling_stage'] != SAMPLING_STAGE):
        raise ValueError('F1_STAGE_OR_SCHEMA_MISMATCH')
    for key in ('batch_size','optimizer_updates','report_every_updates','save_every_updates'):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError('F1_POSITIVE_INTEGER_REQUIRED: '+key)
    dimensions(raw['target_steps'],raw['burnin_steps'])
    if raw['device'] not in ('cpu','gpu'):
        raise ValueError('F1_DEVICE_INVALID')
    spec = raw['sampling_spec']
    if not isinstance(spec,dict) or set(spec)-{'mode','policies'} or spec.get('mode') != 'NATURAL':
        raise ValueError('F1_WM_NATURAL_REQUIRED')
    from .fg_s3_e_quality_supervision import POLICIES
    if 'policies' in spec and (not isinstance(spec['policies'],list) or not spec['policies']
            or len(set(spec['policies'])) != len(spec['policies']) or not set(spec['policies']) <= set(POLICIES)):
        raise ValueError('F1_POLICY_SELECTION_INVALID')
    from dreamerv3.fg_s3_e_config import resolve_model_init
    resolve_model_init(raw['model_init'])
    result = copy.deepcopy(raw)
    for key in ('corpus_root','prepared_dir','run_dir'):
        result[key] = str(_local_path(raw[key]))
    if (Path(result['prepared_dir']).is_relative_to(Path(result['corpus_root'])) or
            Path(result['run_dir']).is_relative_to(Path(result['corpus_root'])) or
            result['prepared_dir'] == result['run_dir']):
        raise ValueError('F1_OUTPUT_MUST_NOT_MODIFY_CORPUS_OR_SHARE_RUN')
    return result


def read_settings(path):
    from dreamerv3.configuration import read_config_yaml
    return validate_settings(read_config_yaml(_local_path(path)))


def model_config(settings, *, controlled_small=False):
    from dreamerv3.fg_s3_e_config import resolve_model_init, e_target_config
    cfg = resolve_model_init(settings['model_init'])
    if controlled_small:
        # Only the built-in CPU engineering probe uses this existing preset.
        from .fg_s3_v2_gpu_check import small_config
        cfg = e_target_config(small_config())
        cfg['seed'] = settings['model_init']['seed']
    cfg['batch_size'] = settings['batch_size']
    cfg['batch_length'] = cfg['report_length'] = dimensions(settings['target_steps'],settings['burnin_steps'])
    cfg['jax']['compute_dtype'] = 'float32' if settings['device'] == 'cpu' else cfg['jax']['compute_dtype']
    return cfg


def create_agent(settings, output, *, controlled_small=False):
    from embodied.run.fg_s3_source_transfer import _make_agent
    from dreamerv3.fg_s3_v2_agent import _S3V2Model
    from dreamerv3.fg_s3_composition import resolve_composition
    from dreamerv3.fg_s3_e_quality_foundation import model_extensions, parameter_layout, register_state_schema
    cfg = model_config(settings,controlled_small=controlled_small)
    agent = _make_agent(cfg,output,agent_cls=_S3V2Model,device=settings['device'],bc_lambda=0.,
        extra_agent_config=model_extensions(settings['target_steps'],settings['burnin_steps']),
        composition=resolve_composition(dict(composition=dict(encoder='e_multiscale_1m15m',teaching='quality_v2'))))
    parameter_layout(agent.params)
    register_state_schema(agent.model,agent.params)
    return agent


def execution_identity():
    """Current run provenance, never compared to the historical Corpus producer."""
    result = {}
    for role,root in (('gym',PROJECT),('dreamer',PROJECT.parent/'dreamerv3')):
        git = lambda *a: subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
        # Bound to run/resume, not to ordinary Reader development. Relative
        # module names make moving an otherwise identical checkout harmless.
        folders = ('src/gym_trading_env/research',) if role == 'gym' else ('dreamerv3','embodied')
        files = {str(p.relative_to(root)):digest(p) for folder in folders
            for p in sorted((root/folder).rglob('*.py')) if 'tests' not in p.parts}
        result[role] = dict(head=git('rev-parse','HEAD'),runtime_hash=fingerprint(files))
    return result


@dataclass(frozen=True)
class F1Preparation:
    settings: dict
    source: Admission
    receipt: dict


def preflight(settings, *, purpose='development', producer_gym=None, log_path=None):
    settings = validate_settings(settings)
    source = SupervisionSource.from_corpus(settings['corpus_root'],stage=STAGE)
    admission = QualitySupervisionAdmission.preflight(source,purpose=purpose,
        producer_gym=producer_gym,log_path=log_path)
    cfg = model_config(settings)
    identity = dict(protocol=PROTOCOL,course_stage=STAGE,sampling_stage=SAMPLING_STAGE,
        sampling_spec=settings['sampling_spec'],source=admission.receipt['content_identity'],
        course=source.course_identity_hash,observation=observation_contract(),reward_schema=REWARD_SCHEMA,
        model=cfg,model_init=settings['model_init'],batch_size=settings['batch_size'],
        target_steps=settings['target_steps'],burnin_steps=settings['burnin_steps'],
        optimizer_updates=settings['optimizer_updates'],
        report_every_updates=settings['report_every_updates'],save_every_updates=settings['save_every_updates'],
        device=settings['device'],code=execution_identity(),
        checkpoint_backend='external_immutable_course_reader',
        parameter_policy=expected_stage_contract(STAGE)['parameter_group_policy'])
    native = admission.native_result is not None
    receipt = signed(dict(protocol=PROTOCOL,status='F1_DATA_READY' if native else 'F1_ENGINEERING_READY',
        stage=STAGE,training_admitted=False,research_admitted=False,
        prepared_for_training=False,identity=identity,identity_hash=fingerprint(identity),
        native_receipt_hash=admission.receipt['native_receipt_hash'],
        blockers=['F1_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED']+
            ([] if native else ['F1_NATIVE_G3_NOT_RUN']),source_receipt=admission.receipt))
    return F1Preparation(settings,admission,receipt)


def prepare(config, *, purpose='development', producer_gym=None):
    settings = read_settings(config)
    out = _local_path(settings['prepared_dir'])
    if out.exists():
        raise ValueError('F1_PREPARATION_OUTPUT_MUST_BE_NEW')
    result = preflight(settings,purpose=purpose,producer_gym=producer_gym,
        log_path=out/'native_verify.log' if purpose == 'native-verification' else None)
    write_json(out/'settings.json',signed(dict(protocol=PROTOCOL,settings=settings)))
    write_json(out/'f1_preparation.json',result.receipt)
    if result.source.native_result is not None:
        write_json(out/'native_verification.json',result.source.native_result)
    return result.receipt


def require_training_authority(preparation, *, authorized=False):
    if authorized is not True:
        raise ValueError('F1_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED')
    if (not isinstance(preparation,F1Preparation) or preparation.source.native_result is None
            or preparation.source.native_result['result'].get('G3') != 'GO'
            or preparation.receipt.get('status') != 'F1_DATA_READY'):
        raise ValueError('F1_ACTUAL_NATIVE_G3_REQUIRED; loaded JSON is not authority')
    return preparation


class F1WindowAdapter:
    """Validated native windows; posterior fields never enter Agent.spaces."""
    def __init__(self, reader, settings):
        if reader.stage != SAMPLING_STAGE or reader.mode != 'NATURAL' or reader.admission.source.stage != STAGE:
            raise ValueError('F1_READER_STAGE_MISMATCH')
        self.reader,self.settings = reader,settings
        self.length = dimensions(settings['target_steps'],settings['burnin_steps'])

    def adapt(self, raw):
        import numpy as np
        from .fg_replay import replay_fields,E_QUALITY_SCHEMA
        if raw['stepid'].shape != (self.settings['batch_size'],self.length+1,20):
            raise ValueError('F1_WINDOW_LENGTH_OR_BATCH_MISMATCH')
        if ('consec' not in raw or raw['consec'].dtype!=np.int32 or
                raw['consec'].shape!=raw['is_first'].shape or raw['consec'].any()):
            raise ValueError('F1_RANDOM_WINDOWS_CANNOT_INHERIT_CARRY')
        metadata = self.reader.audit_windows(raw)
        keys=(*replay_fields(E_QUALITY_SCHEMA),'stepid','consec')
        data={k:raw[k].copy() for k in keys}
        for key,value in data.items():
            if np.issubdtype(value.dtype,np.floating) and not np.isfinite(value).all():
                raise ValueError('F1_NONFINITE_NATIVE_INPUT: '+key)
        return data,metadata

    def sample(self):
        state=self.reader.state_dict()
        try:
            return self.adapt(self.reader.sample(self.settings['batch_size'],self.length))
        except BaseException:
            self.reader.load_state_dict(state)
            raise

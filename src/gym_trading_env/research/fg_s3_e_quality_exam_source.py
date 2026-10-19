"""Independent read-only validation admission for native E/Quality evaluation.

The reviewed original partition is the authority, not a self-signed receipt or
a TRAIN Corpus. No Teacher is constructed; only selected validation files open.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json

from .fg_replay import digest, read_manifest, signed, write_json, check_files
from .fg_s3_e_quality_course_preflight import _local_path
from .fg_s3_e_quality_exam_contract import identity, validate_plan

TRUSTED_VALIDATION_ROOTS={
    'ebe0477a4ff9bad5b0d720b3ddb9f37288591d8df63db73a280224b698068bac':
    'cf8fbf93953142b743f718effe2f377a80eba8a0cddf196a901d5344cf3d19ad'}
_VALIDATION_ADMISSIONS={}


@dataclass
class ExamSource:
    factory: object
    initial_state: dict
    end_cursor: int
    receipt: dict

    def verify(self):
        for path in self.receipt['frozen_inputs']:_local_path(path)
        check_files(self.receipt['frozen_inputs'])

    def require_mode(self,plan):
        if plan['exam_mode']=='DEV_EXAM':
            proof=_VALIDATION_ADMISSIONS.get(id(self))
            if (proof is None or proof[0] is not self or proof[1] is not self.factory
                    or proof[2:]!=(identity(self.receipt),identity(self.initial_state),self.end_cursor)
                    or self.receipt['start_day']!=plan['start_day'] or self.receipt['end_day']!=plan['end_day']):
                raise ValueError('EXAM_NATIVE_VALIDATION_ADMISSION_REQUIRED_NOT_JSON')


class ExamEProfile:
    """Same native Env profile interface; deliberately no TRAIN authority."""
    def __init__(self,receipt):
        self.source_receipt=receipt
        self.identity=identity(receipt)
        self.contracts=frozenset(receipt['sources'])


def validation_members(plan):
    """Role and reviewed metadata before ANY selected price-file access."""
    plan=validate_plan(plan)
    if plan['exam_mode']!='DEV_EXAM':raise ValueError('EXAM_VALIDATION_MODE_REQUIRED')
    source=plan['source'];path=_local_path(source['manifest']);root=_local_path(source['env_receipt'])
    partition=json.loads(path.read_text())
    if partition.get('product')!='FG' or not isinstance(partition.get('files'),list):
        raise ValueError('DEV_SOURCE_NOT_AVAILABLE: validation partition required')
    selected=[]
    for contract in plan['contracts']:
        rows=[r for r in partition['files'] if r.get('contract')==contract]
        if len(rows)!=1:raise ValueError('EXAM_VALIDATION_WHITELIST_MEMBER_REQUIRED')
        row=rows[0]
        if (row.get('role')!='validation' or row.get('eligible_for_training') is not False
                or row.get('complete_lifecycle') is not True or row.get('calendar_complete') is not True):
            raise ValueError('EXAM_VALIDATION_ROLE_REQUIRED_PRICE_NOT_OPENED')
        selected.append(row)
    roles=partition.get('roles',{});train=roles.get('train',[]);dev=roles.get('validation',[])
    if not train or not dev or set(train)&set(dev) or max(train)>=min(dev):
        raise ValueError('EXAM_TRAIN_DEV_OFFICIAL_DAYS_OVERLAP')
    em=read_manifest(root/'manifest.json');partition_hash=digest(path)
    if TRUSTED_VALIDATION_ROOTS.get(partition_hash)!=digest(root/'manifest.json'):
        raise ValueError('EXAM_UNREVIEWED_VALIDATION_TRUST_ROOT')
    original=em.get('config',{}).get('training_manifest')
    if (em.get('stage')!='E' or em.get('status')!='PASS' or em.get('clock_protocol')!='S3_MARKET_MINUTE'
            or em['frozen_inputs'].get(original)!=partition_hash):
        raise ValueError('EXAM_ORIGINAL_NATIVE_SOURCE_IDENTITY_REQUIRED')
    input_path=root/'inputs/inputs.json';check_files({str(input_path):em['files']['inputs/inputs.json']})
    inputs=read_manifest(input_path)
    if inputs['original_content_hash']!=partition['content_hash']:
        raise ValueError('EXAM_PARTITION_ORIGINAL_IDENTITY_CHANGED')
    train_rows=[r for r in partition['files'] if r.get('role')=='train']
    import pandas as pd
    if not train_rows or any(pd.Timestamp(r['first_timestamp'])<=max(pd.Timestamp(t['last_timestamp']) for t in train_rows) for r in selected):
        raise ValueError('EXAM_TRAIN_DEV_TIMESTAMPS_OVERLAP')
    for row in selected:
        if not row['first_day']<=plan['start_day']<=plan['end_day']<=row['last_day']:
            raise ValueError('EXAM_DATES_OUTSIDE_VALIDATION')
        for k in ('path','env_config','contract_config'):_local_path(row[k])
    return selected,partition,em,inputs


def prepare_validation(plan,contract,directory):
    """Derive immutable evaluation configs, preserving native market/cost/clock."""
    plan=validate_plan(plan);rows,partition,em,inputs=validation_members(plan)
    row=next((r for r in rows if r['contract']==contract),None)
    if row is None:raise ValueError('EXAM_CONTRACT_NOT_IN_PLAN')
    frozen={plan['source']['manifest']:digest(plan['source']['manifest']),
            str(_local_path(plan['source']['env_receipt'])/'manifest.json'):digest(_local_path(plan['source']['env_receipt'])/'manifest.json')}
    for p,h in (('path','sha256'),('env_config','env_sha256'),('contract_config','contract_sha256')):
        frozen[row[p]]=row[h]
    check_files(frozen)
    import yaml
    from dataclasses import asdict
    from e_data_v1.schema import EDataConfig
    from e_data_v1.reference import load_legacy
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    from gym_trading_env.rewards.single3_reward import Single3RewardConfig,QUALITY_TP_SCHEMA
    from .fg_s3_e_observation import observation_contract
    # The original native C1 *configuration* supplies financial/clock rules;
    # no old C1 Oracle, label, Replay or Checkpoint is accessed.
    _local_path(inputs['views']['C1']['path'])
    c1=read_manifest(inputs['views']['C1']['path']);check_files({inputs['views']['C1']['path']:em['frozen_inputs'][inputs['views']['C1']['path']]})
    master_path=c1['direction_path']
    _local_path(master_path)
    frozen[master_path]=inputs['frozen_files'][master_path];check_files({master_path:frozen[master_path]})
    master=yaml.safe_load(Path(master_path).read_text());master=copy.deepcopy(master)
    master['reward']=asdict(Single3RewardConfig(candidate='C1',beta=.5,rho=.4,L=3.,schema=QUALITY_TP_SCHEMA))
    master['environment']['entry']['chunk_steps']=0
    raw=yaml.safe_load(Path(row['env_config']).read_text());raw=copy.deepcopy(raw)
    terms=yaml.safe_load(Path(row['contract_config']).read_text())
    from gym_trading_env.envs.entry_contract import resolve_path
    if (resolve_path(row['env_config'],raw['trading']['data_path'])!=Path(row['path'])
            or terms.get('source_sha256')!=row['sha256'] or terms.get('contract')!=contract
            or raw['entry']['contract']!=contract):
        raise ValueError('EXAM_MARKET_ENV_CONTRACT_SOURCE_MISMATCH')
    out=_local_path(directory)
    if out.exists():raise ValueError('EXAM_SOURCE_OUTPUT_MUST_BE_NEW')
    # Resolve once, never modify the original configs or source files.
    master_file=out/'quality_tp.yaml';env_file=out/'env.yaml'
    raw['entry'].pop('state_path',None)
    raw['entry'].update(protocol='fg_single3_v1',strategy_config=str(master_file),
        contract_config=row['contract_config'],chunk_steps=0)
    raw['trading'].update(master['environment']['trading'])
    raw['trading']['data_path']=row['path']
    profile=ExamEProfile(dict(protocol='FG_S3_E_EXAM_PROFILE_V1',role='validation',
        observation=observation_contract(),schema_sha256=EDataConfig().schema(load_legacy()[1])['schema_sha256'],
        sources={contract:{'market':dict(path=row['path'],sha256=row['sha256'])}}))
    out.mkdir(parents=True)
    master_file.write_text(yaml.safe_dump(master,sort_keys=False));env_file.write_text(yaml.safe_dump(raw,sort_keys=False))
    frozen.update({str(p):digest(p) for p in (master_file,env_file)})
    def factory():
        check_files(frozen)
        return FixedDirectionEntryEnv(config_path=env_file,e_profile=profile)
    env=factory()
    try:
        import numpy as np
        import pandas as pd
        if (len(env._raw)!=row['rows'] or pd.Timestamp(env._raw.index[0])!=pd.Timestamp(row['first_timestamp'])
                or pd.Timestamp(env._raw.index[-1])!=pd.Timestamp(row['last_timestamp'])):
            raise ValueError('EXAM_VALIDATION_ACTUAL_MARKET_BOUNDS_CHANGED')
        days=[env._trading_day(t) for t in env._raw.index]
        if not set(int(d.replace('-','')) for d in days)<=set(partition['roles']['validation']):
            raise ValueError('EXAM_MARKET_OUTSIDE_VALIDATION_OFFICIAL_DAYS')
        selected=np.flatnonzero([(plan['start_day']<=d<=plan['end_day']) for d in days])
        if not len(selected) or days[selected[0]]!=plan['start_day'] or days[selected[-1]]!=plan['end_day']:
            raise ValueError('EXAM_OFFICIAL_DAY_NOT_AVAILABLE')
        start=int(selected[0])-1;end=int(selected[-1])
        if start<60:raise ValueError('EXAM_DEV_VISIBLE_WARMUP_INSUFFICIENT')
        env.reset(seed=plan['seed'])
        while env.current_step<start:
            env.step(0)
            if env.status!='ACTIVE':raise ValueError('EXAM_NATIVE_PREFIX_NOT_CONTINUOUS')
        obs=env._get_obs()
        if not obs['market_context_15m'][:,-1].any():raise ValueError('EXAM_E_CONTEXT_WARMUP_UNAVAILABLE')
        initial=env.export_state()
        receipt=signed(dict(protocol='FG_S3_E_QUALITY_EXAM_SOURCE_V1',role='validation',
            exam_mode='DEV_EXAM',contract=contract,training_admitted=False,eligible_for_training=False,
            train_days=partition['roles']['train'],dev_days=partition['roles']['validation'],
            start_day=plan['start_day'],end_day=plan['end_day'],start_cursor=start,end_cursor=end,
            profile_identity=profile.identity,profile=profile.source_receipt,
            observation=observation_contract(),reward_schema=QUALITY_TP_SCHEMA,
            native_config=env.config_hash,initial_account_hash=identity(initial),
            costs_basis=terms.get('costs_basis','native_contract_terms'),cost_terms=terms['terms'],
            frozen_inputs=frozen,test_prices_accessed=False,teacher_used=False))
    finally:env.close()
    write_json(out/'source_receipt.json',receipt);write_json(out/'initial_state.json',initial)
    result=ExamSource(factory,initial,end,receipt)
    _VALIDATION_ADMISSIONS[id(result)]=(result,factory,identity(receipt),identity(initial),end)
    return result

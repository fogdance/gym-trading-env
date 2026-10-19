"""Pure examination contract. No filesystem, market or training authority."""
from __future__ import annotations

import copy
from datetime import date
import hashlib
import json
from pathlib import PurePosixPath

PROTOCOL = 'FG_S3_E_QUALITY_EXAM_PLAN_V1'
BASELINES = ('ALWAYS_WAIT', 'ALWAYS_ADD', 'RANDOM_P10', 'DELAY_15')


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def local_name(value):
    return isinstance(value,str) and bool(value) and value.replace('_','').replace('-','').isalnum()


def local_path(value):
    if (not isinstance(value,str) or not value or value.strip()!=value or '\\' in value
            or '\x00' in value or '://' in value or not value.startswith('/')
            or '..' in PurePosixPath(value).parts):
        raise ValueError('EXAM_EXPLICIT_ABSOLUTE_LOCAL_PATH_REQUIRED')
    return value


def validate_plan(raw):
    required={'protocol','exam_mode','contracts','source','actors','baselines','start_day','end_day',
              'chunk_steps','seed','random_seeds','output_dir','evidence_policy'}
    if not isinstance(raw,dict) or set(raw)!=required or raw['protocol']!=PROTOCOL:
        raise ValueError('EXAM_EXPLICIT_PLAN_FIELDS_REQUIRED')
    if raw['exam_mode']=='SEALED_TEST_FINAL':
        raise ValueError('EXAM_TEST_SEALED_UNTIL_WP10')
    if raw['exam_mode'] not in ('SYNTHETIC_DEV','DEV_EXAM'):
        raise ValueError('EXAM_EXPLICIT_MODE_REQUIRED')
    contracts=raw['contracts']
    if (not isinstance(contracts,list) or not contracts or len(set(contracts))!=len(contracts)
            or any(not isinstance(c,str) or not c.startswith('CZCE.FG') or not c[7:].isdigit() for c in contracts)):
        raise ValueError('EXAM_UNIQUE_FG_CONTRACTS_REQUIRED')
    for key in ('start_day','end_day'):
        try:
            if date.fromisoformat(raw[key]).isoformat()!=raw[key]:raise ValueError()
        except (TypeError,ValueError):raise ValueError('EXAM_OFFICIAL_ISO_DAYS_REQUIRED') from None
    if raw['start_day']>raw['end_day']:raise ValueError('EXAM_REVERSED_DAYS')
    for key,minimum in (('chunk_steps',1),('seed',0)):
        if type(raw[key]) is not int or raw[key]<minimum:raise ValueError('EXAM_INTEGER_REQUIRED: '+key)
    seeds=raw['random_seeds']
    if (not isinstance(seeds,list) or len(seeds)<2 or len(set(seeds))!=len(seeds)
            or any(type(s) is not int or s<0 for s in seeds)):
        raise ValueError('EXAM_PREREGISTERED_RANDOM_SEEDS_REQUIRED')
    source=raw['source']
    if not isinstance(source,dict) or set(source)!={'kind','manifest','env_receipt'}:
        raise ValueError('EXAM_EXPLICIT_SOURCE_REQUIRED')
    allowed=('VALIDATION',) if raw['exam_mode']=='DEV_EXAM' else ('BUILTIN_SYNTHETIC','FINITE_TRAIN')
    if source['kind'] not in allowed:raise ValueError('EXAM_SOURCE_ROLE_MISMATCH')
    for key in ('manifest','env_receipt'):
        if source['kind']=='BUILTIN_SYNTHETIC':
            if source[key] is not None:raise ValueError('EXAM_SYNTHETIC_CANNOT_OPEN_USER_MARKET')
        else:local_path(source[key])
    actors=raw['actors']
    if not isinstance(actors,dict):raise ValueError('EXAM_ACTORS_REQUIRED')
    for name,actor in actors.items():
        if (not local_name(name) or name in BASELINES or not isinstance(actor,dict)
                or set(actor)!={'kind','run','checkpoint'} or actor['kind'] not in ('F3','F4')):
            raise ValueError('EXAM_EXPLICIT_F3_F4_ACTOR_REQUIRED')
        local_path(actor['run']);local_path(actor['checkpoint'])
        if not PurePosixPath(actor['checkpoint']).is_relative_to(PurePosixPath(actor['run'])/'ckpt'):
            raise ValueError('EXAM_CHECKPOINT_OUTSIDE_SOURCE_RUN')
    baselines=raw['baselines']
    if (not isinstance(baselines,list) or not baselines or len(set(baselines))!=len(baselines)
            or any(b not in (*BASELINES,*actors) for b in baselines)):
        raise ValueError('EXAM_SUPPORTED_UNIQUE_BASELINES_REQUIRED')
    output=PurePosixPath(local_path(raw['output_dir']))
    for actor in actors.values():
        run=PurePosixPath(actor['run'])
        if output.is_relative_to(run) or run.is_relative_to(output):raise ValueError('EXAM_SOURCE_OUTPUT_OVERLAP')
    policy=raw['evidence_policy']
    if policy is not None:
        fields={'min_trades','min_contracts','min_days','min_tp5_rate','max_sl_rate',
                'max_mae15','min_net_pnl','max_drawdown'}
        if not isinstance(policy,dict) or set(policy)!=fields:raise ValueError('EXAM_EXPLICIT_EVIDENCE_POLICY_REQUIRED')
        import math
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in policy.values()):
            raise ValueError('EXAM_FINITE_EVIDENCE_POLICY_REQUIRED')
        if any(type(policy[k]) is not int or policy[k]<1 for k in ('min_trades','min_contracts','min_days')):
            raise ValueError('EXAM_POSITIVE_SUPPORT_REQUIRED')
        if any(not 0<=policy[k]<=1 for k in ('min_tp5_rate','max_sl_rate')) or min(policy['max_mae15'],policy['max_drawdown'])<0:
            raise ValueError('EXAM_INVALID_EVIDENCE_BOUND')
    return copy.deepcopy(raw)


def plan_hash(raw):
    return identity(validate_plan(raw))

"""Controlled metadata/Env fixtures; never presented as real TRAIN authority."""
import gzip
import json
from pathlib import Path

from dreamerv3 import fg_s3_e_quality_course_contract as course
from e_data_v1.reference import load_legacy
from e_data_v1.schema import EDataConfig
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import signed, write_json, digest, E_QUALITY_SCHEMA
from gym_trading_env.research.fg_s3_e_observation import observation_contract
from gym_trading_env.research.fg_s3_e_quality_supervision import POLICIES

ROOT = Path(__file__).resolve().parents[2]


def inventory(root, m):
    m = {k:v for k,v in m.items() if k != 'content_hash'}
    m['files'] = {str(p.relative_to(root)):digest(p) for p in root.rglob('*')
                  if p.is_file() and p.name != 'corpus_manifest.json'}
    write_json(root/'corpus_manifest.json', signed(m))
    return signed(m)


def metadata_corpus(root):
    root.mkdir()
    spec = json.loads((ROOT/'docs/fixtures/fg_s3_e_quality_course_v2.example.json').read_text())
    spec['spec_mode'] = 'formal'
    data = spec['data_identity']
    data['source_manifest'] = dict(path='controlled_pool.json', sha256='a'*64)
    data['teacher_manifest'] = dict(path='teacher_manifest.json', sha256='b'*64)
    data['e_profile_schema_sha256'] = EDataConfig().schema(load_legacy()[1])['schema_sha256']
    data['market_sources'] = [dict(contract=c, source_sha256='c'*64,
        env_config_sha256='d'*64, metadata_sha256='e'*64) for c in data['contracts']]
    for behavior in spec['curriculum']['behaviors']:
        if behavior['kind'] == 'quality_oracle_schedule':
            behavior['parameters']['source'] = dict(path='p4_plan.json', sha256='f'*64)
    normalized = course.validate_course_spec(spec)
    write_json(root/'course_v2.json', spec)
    lock = signed(dict(fold='TRAIN', row=dict(role='train', eligible_for_training=True)))
    write_json(root/'source_lock.json', lock)
    teacher = signed(dict(protocol='FG_S3_E_QUALITY_TEACHER_V2', fold='TRAIN',
        strict_identity='1'*64, source_lock_hash=lock['content_hash'],
        objective='discounted_native_float32_Quality_TP_feedback', gamma_rational=[332,333],
        reward_schema=spec['reward']['schema']))
    write_json(root/'teacher_manifest.json', teacher)
    model = signed(dict(protocol='FG_S3_E_MODEL_CONTRACT_V2', observation=observation_contract()))
    write_json(root/'model_contract.json', model)
    for policy in POLICIES:
        trace = signed(dict(schema=E_QUALITY_SCHEMA, reward_version=spec['reward']['schema'],
                            files={}, chunks=[], records=0))
        for kind in ('policies','student_replay'):
            write_json(root/kind/policy/'manifest.json', trace)
        p = root/'teacher_sidecar'/policy/'steps.jsonl.gz'
        p.parent.mkdir(parents=True)
        with gzip.open(p,'wt') as f: f.write('')
    m = dict(protocol='FG_S3_E_QUALITY_CORPUS_V2',fold='TRAIN',training_admitted=False,
        behaviors=list(POLICIES),observation=observation_contract(),teacher_identity='1'*64,
        source_lock_hash=lock['content_hash'],model_contract_hash=model['content_hash'],
        course_identity_hash=fingerprint(course.course_identity_payload(normalized)))
    inventory(root,m)
    return root


def native_corpus(root, *, minutes=30, chunk=9):
    """Six original Env accounts and Replay on controlled, non-real prices."""
    import shutil
    import pandas as pd
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_s3_e_quality_strict import QualityStrictTeacher
    from gym_trading_env.research.fg_s3_e_quality_corpus import generate_behavior, behavior_specs
    from gym_trading_env.research.fg_s3_e_quality_corpus_cli import _publish_streams
    root.mkdir()
    factory,_,profile=native_factory(root.parent/(root.name+'_market'),minutes=minutes,chunk=chunk)
    env=factory();source=dict(fold='TRAIN',state_cache_entries=0,calendar_complete=True)
    strict=QualityStrictTeacher(env,source=source)
    specs=behavior_specs(202)+[dict(id='P4',kind='quality_oracle_schedule',seed_components=[202,0,5],
                                  teacher_identity=strict.identity_hash)]
    for spec in specs:
        generate_behavior(factory,spec,root/'policies'/spec['id'],ROOT.parent/'dreamerv3',
                          source=source,strict=strict)
    _publish_streams(root,list(POLICIES))
    metadata=metadata_corpus(root.parent/(root.name+'_metadata'))
    for name in ('source_lock.json','teacher_manifest.json','course_v2.json','model_contract.json'):
        shutil.copyfile(metadata/name,root/name)
    lock=signed(dict(fold='TRAIN',contract=env.contract,row=dict(role='train',eligible_for_training=True)))
    write_json(root/'source_lock.json',lock)
    teacher=signed(dict(protocol='FG_S3_E_QUALITY_TEACHER_V2',fold='TRAIN',
        strict_identity=strict.identity_hash,source_lock_hash=lock['content_hash'],
        objective=strict.identity['objective'],gamma_rational=[332,333],reward_schema=env.reward_config.schema))
    write_json(root/'teacher_manifest.json',teacher)
    spec=json.loads((root/'course_v2.json').read_text());data=spec['data_identity']
    bounds=dict(start=env._raw.index[0].isoformat(),end=(env._raw.index[-1]+pd.Timedelta(minutes=1)).isoformat())
    data['contracts']=[env.contract];data['time_bounds']=bounds
    data['market_sources']=[dict(contract=env.contract,source_sha256='c'*64,env_config_sha256='d'*64,metadata_sha256='e'*64)]
    data['teacher_manifest']=dict(path='teacher_manifest.json',sha256=digest(root/'teacher_manifest.json'))
    for behavior in spec['curriculum']['behaviors']:
        behavior['boundary'].update(contracts=[env.contract],time_bounds=bounds)
    write_json(root/'course_v2.json',spec)
    model=json.loads((root/'model_contract.json').read_text())
    m=dict(protocol='FG_S3_E_QUALITY_CORPUS_V2',fold='TRAIN',training_admitted=False,
        behaviors=list(POLICIES),observation=observation_contract(),teacher_identity=strict.identity_hash,
        source_lock_hash=lock['content_hash'],model_contract_hash=model['content_hash'],
        e_profile_identity=profile.identity,source_binding=source,
        course_identity_hash=fingerprint(course.course_identity_payload(course.validate_course_spec(spec))))
    inventory(root,m)
    env.close()
    return root


def rewrite_rows(root, policy, rows):
    """Re-sign an attack fixture; behavioral mismatch must still be rejected."""
    import hashlib,shutil
    p=root/'policies'/policy
    with gzip.open(p/'steps.jsonl.gz','wt') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    m=json.loads((p/'manifest.json').read_text());m.pop('content_hash')
    m['files']['steps.jsonl.gz']=digest(p/'steps.jsonl.gz')
    h=hashlib.sha256()
    for row in rows:h.update(json.dumps({k:v for k,v in row.items() if k not in ('stepid','ordinal')},
                                       sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    m['event_hash']=h.hexdigest()
    m['replay_trace_hash']=fingerprint({k:m[k] for k in ('identity','policy','env_config_hash','reward_version','array_hash','event_hash')})
    write_json(p/'manifest.json',signed(m))
    shutil.copyfile(p/'steps.jsonl.gz',root/'teacher_sidecar'/policy/'steps.jsonl.gz')
    inventory(root,json.loads((root/'corpus_manifest.json').read_text()))

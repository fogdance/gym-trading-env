"""Historical provenance, live source guards and exact native reconstruction."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import digest
from gym_trading_env.research.fg_s3_corpus_provenance import SourceVerification, authenticate_code


def git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def inventory(project):
    records, files = {}, {}
    for role, root in [('gym',project),('dreamer',project.parent/'dreamerv3')]:
        records[role] = dict(sha=git(root,'rev-parse','HEAD'),dirty=False)
        files.update({str(path):digest(path) for path in root.rglob('*.py')})
    return dict(repositories=records,files=files,execution_source_hash=fingerprint(files))


def old_pair(root):
    project=root/'gym-trading-env'
    for folder,filename in [('gym-trading-env','src/gym_trading_env/producer.py'),
                             ('dreamerv3','dreamerv3/producer.py')]:
        repo=root/folder;repo.mkdir(parents=True)
        subprocess.run(['git','init','-q',str(repo)],check=True)
        path=repo/filename;path.parent.mkdir(parents=True);path.write_text('VERSION = 1\n')
        git(repo,'add','.')
        git(repo,'-c','user.name=Corpus fixture','-c','user.email=fixture@example.invalid',
            'commit','-qm','original immutable code')
    return project,inventory(project)


def move_and_update(old, target):
    old.parent.rename(target)
    project=target/'gym-trading-env'
    (project/'src/gym_trading_env/producer.py').write_text('VERSION = 2\n')
    git(project,'add','.')
    git(project,'-c','user.name=Corpus fixture','-c','user.email=fixture@example.invalid',
        'commit','-qm','nonfinancial execution fix')
    return project,inventory(project)


def test_git_provenance_rejects_resigned_hash_or_missing_inventory(tmp_path):
    project,code=old_pair(tmp_path/'old')
    assert authenticate_code(code,project=project)['files_verified']==dict(gym=1,dreamer=1)
    wrong=copy.deepcopy(code);wrong['files'][next(iter(wrong['files']))]='0'*64
    wrong['execution_source_hash']=fingerprint(wrong['files'])
    with pytest.raises(ValueError,match='SHA256 differs'):
        authenticate_code(wrong,project=project)
    wrong=copy.deepcopy(code);wrong['files'].pop(next(iter(wrong['files'])))
    wrong['execution_source_hash']=fingerprint(wrong['files'])
    with pytest.raises(ValueError,match='inventory differs'):
        authenticate_code(wrong,project=project)


def test_six_native_behaviors_exact_after_checkout_removed_with_parallel_labels(tmp_path):
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_s3_e_quality_strict import QualityStrictTeacher
    from gym_trading_env.research.fg_s3_e_quality_corpus import generate_behavior,behavior_specs
    from gym_trading_env.research import fg_s3_e_quality_corpus_cli as cli
    from gym_trading_env.research.fg_s3_e_quality_supervision import POLICIES

    project,code=old_pair(tmp_path/'old')
    authority=tmp_path/'source.lock';authority.write_text('controlled synthetic source\n')
    source=dict(fold='TRAIN',state_cache_entries=0,calendar_complete=True,
                files={**code['files'],str(authority):digest(authority)})
    factory,_,profile=native_factory(tmp_path/'market',minutes=20,chunk=7)
    env,initial=cli._prefix(factory,0)
    strict=QualityStrictTeacher(env,source=source)
    original_identity=strict.identity_hash
    specs=behavior_specs()+[dict(id='P4',kind='quality_oracle_schedule',seed_components=[202,0,5],
                               teacher_identity=strict.identity_hash)]
    out=tmp_path/'corpus'
    for spec in specs:
        generate_behavior(factory,spec,out/spec['id'],Path(cli.PROJECT).parent/'dreamerv3',
                          source=source,strict=strict,initial_state=initial)
    before={str(p):digest(p) for p in out.rglob('*') if p.is_file()}
    current,current_code=move_and_update(project,tmp_path/'new')
    assert not project.exists()
    verification=SourceVerification.create(dict(code=code,source_binding=source),current_code,project=current)
    rebuilt=verification.strict(QualityStrictTeacher(env,source=source))
    assert rebuilt.identity_hash==original_identity and rebuilt.plan()==strict.plan()
    for index,name in enumerate(POLICIES):
        job=dict(name=name,path=out/name,row=dict(env_config=str(tmp_path/'market/env.yaml')),
            profile=profile,initial=initial,source=source,strict=rebuilt,
            label_workers=0 if index==0 else 2,source_verification=verification)
        receipt=cli._verification_job(job)['receipt']
        assert receipt['status']=='PASS' and receipt['records']>0
        assert receipt['teacher_labels_verified']==receipt['records']
    assert {str(p):digest(p) for p in out.rglob('*') if p.is_file()}==before
    authority.write_text('changed source\n')
    with pytest.raises(ValueError,match='source file changed'):
        verification.teacher(env,factory)
    env.close()


def test_descriptor_changes_only_authenticated_code_provenance():
    verification=SourceVerification({}, {}, {'a'*64:{'b'*64}}, {})
    value=dict(code_sha256='a'*64,action_value_add=7.,teacher_valid=True,
               source_lock_hash='a'*64,nested=dict(query_code_sha256='a'*64))
    result=verification.descriptor(value)
    assert result==dict(code_sha256='b'*64,action_value_add=7.,teacher_valid=True,
                        source_lock_hash='a'*64,nested=dict(query_code_sha256='b'*64))
    assert value['code_sha256']=='a'*64

import json
import shutil

import pytest

from gym_trading_env.research.fg_replay import read_manifest, signed, write_json
from gym_trading_env.research.fg_s3_e_quality_supervision import (
    SupervisionSource, QualitySupervisionAdmission, member)
from wp03_fixture import metadata_corpus, inventory


def test_A01_development_checks_content_without_claiming_native_or_training(tmp_path):
    root = metadata_corpus(tmp_path/'corpus')
    before = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    a = QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root))
    assert a.receipt['status'] == 'DEVELOPMENT_VALIDATED'
    assert a.receipt['admission_status'] == 'NOT_ADMITTED'
    assert a.native_result is None and not a.receipt['training_admitted']
    assert not a.receipt['prepared_certified'] and not a.receipt['checkpoint_certified']
    assert 'NATIVE_G3_NOT_RUN_IN_THIS_ADMISSION' in a.receipt['blockers']
    assert before == {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_A02_relocated_corpus_and_unrelated_dirty_code_keep_content_identity(tmp_path):
    a = metadata_corpus(tmp_path/'original')
    b = tmp_path/'moved'; shutil.copytree(a,b)
    original = QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(a))
    (tmp_path/'new_reader.py').write_text('development work in progress\n')
    moved = QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(b))
    assert original.receipt['content_identity'] == moved.receipt['content_identity']
    assert original.receipt['corpus_root'] != moved.receipt['corpus_root']


def test_A03_changed_source_or_course_bytes_rejected(tmp_path):
    root = metadata_corpus(tmp_path/'corpus')
    source = SupervisionSource.from_corpus(root)
    (root/'course_v2.json').write_text('{}')
    with pytest.raises(ValueError, match='CONTENT_CHANGED'):
        QualitySupervisionAdmission.preflight(source)


@pytest.mark.parametrize('name',['../TEST/prices.csv','/tmp/TEST/prices.csv','TEST/prices.csv','DEV/data.csv'])
def test_A04_unsafe_nontrain_paths_rejected_before_open(tmp_path,name):
    with pytest.raises(ValueError):member(tmp_path,name)


def test_A05_symlink_rejected_before_hashing(tmp_path):
    (tmp_path/'link').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError,match='symlink'):member(tmp_path,'link/market.npz')
    with pytest.raises(ValueError,match='NON_TRAIN_PATH'):
        SupervisionSource.from_corpus(tmp_path/'TEST'/'corpus')


def test_A06_nontrain_and_forged_training_permission_rejected(tmp_path):
    root=metadata_corpus(tmp_path/'corpus'); m=read_manifest(root/'corpus_manifest.json')
    m['fold']='TEST';write_json(root/'corpus_manifest.json',signed({k:v for k,v in m.items() if k!='content_hash'}))
    with pytest.raises(ValueError,match='CORPUS_CONTRACT'):
        QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root))
    m['fold']='TRAIN';m['training_admitted']=True
    write_json(root/'corpus_manifest.json',signed({k:v for k,v in m.items() if k!='content_hash'}))
    with pytest.raises(ValueError,match='CORPUS_CONTRACT'):
        QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root))


def test_A07_self_signed_pass_cannot_replace_native_execution(tmp_path):
    root=metadata_corpus(tmp_path/'corpus')
    fake=dict(status='PASS',G3='GO',training_admitted=False)
    with pytest.raises(TypeError):
        QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root),native_receipt=fake)
    with pytest.raises(ValueError,match='EXECUTION_ARGUMENTS_REQUIRED'):
        QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root),purpose='native-verification')


def test_A08_missing_or_empty_purpose_never_means_draft(tmp_path):
    source=SupervisionSource.from_corpus(metadata_corpus(tmp_path/'corpus'))
    for purpose in ('',None,'formal','draft'):
        with pytest.raises(ValueError,match='PURPOSE_REQUIRED'):
            QualitySupervisionAdmission.preflight(source,purpose=purpose)


def test_A09_changed_declared_identity_and_old_teacher_objective_rejected(tmp_path):
    from dataclasses import replace
    root=metadata_corpus(tmp_path/'corpus'); source=SupervisionSource.from_corpus(root)
    with pytest.raises(ValueError,match='SOURCE_IDENTITY_MISMATCH'):
        QualitySupervisionAdmission.preflight(replace(source,teacher_identity_hash='2'*64))
    teacher=read_manifest(root/'teacher_manifest.json');teacher.pop('content_hash')
    teacher['objective']='LEGACY_C1_UTILITY';write_json(root/'teacher_manifest.json',signed(teacher))
    inventory(root,read_manifest(root/'corpus_manifest.json'))
    with pytest.raises(ValueError,match='CONTRACT_IDENTITY_MISMATCH'):
        QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root))


def test_A10_missing_published_owner_or_inventory_rejected(tmp_path):
    root=metadata_corpus(tmp_path/'corpus');m=read_manifest(root/'corpus_manifest.json')
    m.pop('content_hash');del m['files']['teacher_sidecar/P0/steps.jsonl.gz']
    write_json(root/'corpus_manifest.json',signed(m))
    with pytest.raises(ValueError,match='INCOMPLETE_INVENTORY'):
        QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root))


def test_A11_draft_course_cannot_be_a_supervision_source(tmp_path):
    root=metadata_corpus(tmp_path/'corpus');spec=json.loads((root/'course_v2.json').read_text())
    spec['spec_mode']='draft';write_json(root/'course_v2.json',spec)
    m=read_manifest(root/'corpus_manifest.json');inventory(root,m)
    with pytest.raises(ValueError,match='FORMAL_CORPUS_CONTRACT'):
        QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root))

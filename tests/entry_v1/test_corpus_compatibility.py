"""Compatibility is independent of code provenance and never grants authority."""
import copy
import json
from pathlib import Path

import pytest

from gym_trading_env.research.fg_replay import read_manifest, signed, write_json
from gym_trading_env.research.fg_s3_corpus_compatibility import CorpusCompatibility
from gym_trading_env.research.fg_s3_e_quality_supervision import SupervisionSource, QualitySupervisionAdmission
from wp03_fixture import metadata_corpus, inventory


def admit(root):
    return QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root))


def test_existing_v3_mapping_is_read_only_and_consumer_head_independent(tmp_path, monkeypatch):
    root = metadata_corpus(tmp_path/'corpus')
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    original = admit(root)
    monkeypatch.setattr('gym_trading_env.research.fg_s3_e_quality_foundation.execution_identity',
                        lambda: {'new_consumer': 'different HEAD and runtime implementation'})
    current = admit(root)
    assert current.receipt['content_identity'] == original.receipt['content_identity']
    assert current.receipt['compatibility']['course_abi_version'] == '3.1.0'
    assert current.receipt['compatibility']['resolution'] == 'REVIEWED_V3_MAPPING'
    assert current.native_result is None and not current.receipt['training_admitted']
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('version', ['3.1.0', '3.1.1', '3.9.0', None])
def test_explicit_version_is_exact_not_major_based(tmp_path, version):
    root = metadata_corpus(tmp_path/'corpus')
    manifest = read_manifest(root/'corpus_manifest.json')
    manifest['course_abi_version'] = version
    inventory(root, manifest)
    if version == '3.1.0':
        assert admit(root).receipt['compatibility']['resolution'] == 'EXPLICIT'
    else:
        with pytest.raises(ValueError, match='UNSUPPORTED_VERSION'): admit(root)


def test_reviewed_implementation_profiles_share_semantics_but_unknown_profile_rejects(tmp_path):
    root = metadata_corpus(tmp_path/'corpus')
    spec = json.loads((root/'course_v2.json').read_text())
    from dreamerv3.fg_s3_e_quality_course_contract import validate_course_spec, course_identity_payload
    from gym_trading_env.envs.entry_contract import fingerprint
    current = admit(root)
    spec['data_identity']['e_profile_schema_sha256'] = '7d82865eb7981d9aeaab1cd4991c5c38521e027c62ff18a6ecc0efeb44636ee6'
    write_json(root/'course_v2.json', spec)
    manifest = read_manifest(root/'corpus_manifest.json')
    manifest['course_identity_hash'] = fingerprint(course_identity_payload(validate_course_spec(spec)))
    inventory(root, manifest)
    historical = admit(root)
    assert historical.receipt['compatibility']['semantic_contract_hash'] == current.receipt['compatibility']['semantic_contract_hash']
    schema = __import__('e_data_v1.schema', fromlist=['EDataConfig']).EDataConfig().schema(
        __import__('e_data_v1.reference', fromlist=['load_legacy']).load_legacy()[1])
    with pytest.raises(ValueError, match='UNSUPPORTED_E_PROFILE'):
        CorpusCompatibility.validate_e_profile('f'*64, schema)
    schema['availability'] = 'future values allowed'
    with pytest.raises(ValueError, match='UNSUPPORTED_E_PROFILE'):
        CorpusCompatibility.validate_e_profile(spec['data_identity']['e_profile_schema_sha256'], schema)


@pytest.mark.parametrize('attack', ['v2', 'behaviors', 'teacher', 'source', 'macd', 'bytes'])
def test_abi_does_not_relax_existing_integrity_guards(tmp_path, attack):
    root = metadata_corpus(tmp_path/'corpus')
    source = SupervisionSource.from_corpus(root)
    manifest = read_manifest(root/'corpus_manifest.json')
    if attack == 'v2': manifest['protocol'] = 'FG_S3_E_QUALITY_CORPUS_V2'
    elif attack == 'behaviors': manifest['behaviors'] = ['P4']
    elif attack == 'teacher': manifest['teacher_identity'] = 'a'*64
    elif attack == 'source': manifest['source_lock_hash'] = 'a'*64
    elif attack == 'macd': manifest['observation']['macd_5m']['partial'] = 'future'
    else:
        (root/'course_v2.json').write_text('{}')
        with pytest.raises(ValueError): QualitySupervisionAdmission.preflight(source)
        return
    manifest.pop('content_hash')
    write_json(root/'corpus_manifest.json', signed(manifest))
    with pytest.raises(ValueError): admit(root)

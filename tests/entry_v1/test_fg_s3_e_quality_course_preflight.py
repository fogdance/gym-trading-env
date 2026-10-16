"""WP01 native consistency, real CLI exit codes, output and side-effect guards."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from dreamerv3.fg_s3_e_quality_course_contract import course_identity_payload, validate_course_spec
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research import fg_s3_e_quality_course_preflight as p
from gym_trading_env.research.fg_replay import read_manifest, signed, write_json


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'docs/fixtures/fg_s3_e_quality_course_v1.example.json'
CLI = ROOT / 'tools/fg_s3_e_quality_course_preflight.py'


def draft():
    return json.loads(EXAMPLE.read_text())


def spec_file(tmp_path, value):
    path = tmp_path / 'course.json'
    path.write_text(json.dumps(value, indent=2) + '\n')
    return path


def cli(path, report=None):
    args = [sys.executable, '-B', str(CLI), '--spec', str(path)]
    if report is not None: args += ['--report', str(report)]
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True,
        env={**os.environ, 'JAX_PLATFORMS': 'cpu', 'CUDA_VISIBLE_DEVICES': '',
             'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
    assert result.stdout, result.stderr
    return result, json.loads(result.stdout)


def tree(path):
    return {str(x.relative_to(path)) for x in path.rglob('*')}


def complete_metadata(tmp_path, *, mode='formal'):
    """Synthetic metadata only: no market file, teacher corpus or real approval."""
    raw = draft(); raw['spec_mode'] = mode
    metadata = tmp_path / 'metadata'; metadata.mkdir()
    source = raw['data_identity']
    source['market_sources'] = [dict(contract='CZCE.FG201', source_sha256='a' * 64,
        env_config_sha256='b' * 64, metadata_sha256='c' * 64)]
    member = dict(contract='CZCE.FG201', role='train', eligible_for_training=True,
        path=str(metadata / 'NONEXISTENT_MARKET.csv'), sha256='a' * 64,
        env_config_sha256='b' * 64, metadata_sha256='c' * 64)
    pool = metadata / 'pool.json'; pool.write_text(json.dumps(dict(product='FG', files=[member])))
    source['source_manifest'] = dict(path='metadata/pool.json', sha256=p.sha256(pool))
    source['e_profile_schema_sha256'] = p.native_contracts(validate_course_spec(draft()))['e_data_schema']['schema_sha256']
    binding = dict(fold='TRAIN', reward_schema=raw['reward']['schema'], score_policy=raw['teacher']['score_policy'],
        kind='SYNTHETIC_METADATA_ONLY')
    for name in ('teacher', 'oracle'):
        path = metadata / (name + '.json'); write_json(path, signed(binding))
        ref = dict(path='metadata/' + name + '.json', sha256=p.sha256(path))
        if name == 'teacher': source['teacher_manifest'] = ref
        else: raw['curriculum']['behaviors'][-1]['parameters']['source'] = ref
    return raw


def assert_not_admitted(report):
    assert report['admission_status'] == 'NOT_ADMITTED'
    assert report['authenticated'] is report['execution_admitted'] is False
    assert report['blocking_reasons']


def test_WP01_01_12_native_draft_calls_reward_validator_and_never_loads_markets(tmp_path, monkeypatch):
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    from e_data_v1 import sources
    def denied(*a, **kw): pytest.fail('Preflight tried to load market/model data or construct an Env')
    monkeypatch.setattr(pd, 'read_csv', denied)
    monkeypatch.setattr(np, 'load', denied)
    monkeypatch.setattr(FixedDirectionEntryEnv, '__init__', denied)
    monkeypatch.setattr(sources, 'read_fg_source', denied)
    calls = []
    native_reward = p.Single3RewardConfig
    def checked_reward(**kw):
        calls.append(kw)
        return native_reward(**kw)
    monkeypatch.setattr(p, 'Single3RewardConfig', checked_reward)
    path = spec_file(tmp_path, draft()); before = tree(tmp_path)
    report = p.preflight(path)
    assert report['exit_code'] == 0 and report['status'] == 'CONTRACT_ONLY'
    assert_not_admitted(report)
    assert calls == [draft()['reward']]
    assert report['native_contracts']['reward']['config_hash'] == native_reward(**draft()['reward']).hash
    assert report['native_contracts']['public_composition']['guard'] == 'S3_QUALITY_E_NOT_IMPLEMENTED'
    assert report['source_readiness']['metadata_checks'] == {}
    assert tree(tmp_path) == before


def test_WP01_13_cli_repeated_drafts_have_identical_non_authenticated_receipts(tmp_path):
    path = spec_file(tmp_path, draft())
    first, report = cli(path, tmp_path / 'report1')
    second, again = cli(path, tmp_path / 'report2')
    assert first.returncode == second.returncode == 0
    assert report == again and report['report_kind'] == 'NON_AUTHENTICATED_DIAGNOSTIC'
    assert_not_admitted(report)
    assert report['status'] == 'CONTRACT_ONLY'
    one, two = [tmp_path / name / p.RECEIPT_NAME for name in ('report1', 'report2')]
    assert one.read_bytes() == two.read_bytes()
    assert read_manifest(one) == report
    assert tree(tmp_path) == {'course.json', 'report1', 'report2', 'report1/' + p.RECEIPT_NAME, 'report2/' + p.RECEIPT_NAME}
    assert not any(s['training_enabled'] for s in report['stage_contracts'].values())


def test_WP01_16_cli_disables_bytecode_without_external_python_flags():
    env = dict(os.environ)
    env.pop('PYTHONDONTWRITEBYTECODE', None)
    script = ('import runpy, sys; '
              'runpy.run_path(sys.argv[1], run_name="wp01_readonly_probe"); '
              'assert sys.dont_write_bytecode, "CLI must disable implicit bytecode writes"')
    result = subprocess.run([sys.executable, '-c', script, str(CLI)], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_WP01_13_native_hash_changes_for_order_seed_or_source_hash():
    base = draft()
    def digest(raw): return fingerprint(course_identity_payload(validate_course_spec(raw)))
    h = digest(base)
    assert digest(json.loads(json.dumps(base, sort_keys=True))) == h
    altered = copy.deepcopy(base); altered['curriculum']['behaviors'].reverse()
    assert digest(altered) != h
    altered = copy.deepcopy(base); altered['initialization']['seed'] += 1
    assert digest(altered) != h
    altered = copy.deepcopy(base)
    altered['data_identity']['source_manifest'] = dict(path='source.json', sha256='a' * 64)
    a = digest(altered)
    altered['data_identity']['source_manifest']['sha256'] = 'b' * 64
    assert digest(altered) != a != h


def test_WP01_12_formal_complete_matching_metadata_still_fails_without_certification(tmp_path, monkeypatch):
    raw = complete_metadata(tmp_path)
    def denied(*a, **kw): pytest.fail('Metadata inspection opened market bytes')
    monkeypatch.setattr(pd, 'read_csv', denied)
    path = spec_file(tmp_path, raw); before = tree(tmp_path)
    report = p.preflight(path, report_dir=tmp_path / 'formal_report')
    assert report['exit_code'] == 1 and report['status'] == 'CONTRACT_ONLY'
    assert_not_admitted(report)
    assert report['source_readiness']['metadata_checks']
    assert report['source_readiness']['source_bytes_verified'] is False
    assert tree(tmp_path) == before
    result, actual = cli(path, tmp_path / 'formal_report')
    assert result.returncode == 1 and actual == report
    assert tree(tmp_path) == before


def test_WP01_08_declared_oracle_evidence_never_certifies_a_draft(tmp_path):
    path = spec_file(tmp_path, complete_metadata(tmp_path, mode='draft'))
    result, report = cli(path, tmp_path / 'diagnostic')
    assert result.returncode == 0
    assert_not_admitted(report)
    assert 'QUALITY_ORACLE_SCHEDULE_NOT_IMPLEMENTED:P4' in report['blocking_reasons']
    assert report['source_readiness']['authenticated'] is False
    assert set((tmp_path / 'diagnostic').iterdir()) == {tmp_path / 'diagnostic' / p.RECEIPT_NAME}


INVALID_CASES = [
    ('composition', ('composition', 'encoder'), 'legacy_1m'),
    ('old_reward', ('reward', 'schema'), 'FG_SINGLE3_REWARD_MARKET_MINUTE_V2'),
    ('candidate', ('reward', 'candidate'), 'C2'),
    ('minute_shape', ('observation', 'fields', 'market_seq_e_shared', 'shape'), [59, 18]),
    ('context_shape', ('observation', 'fields', 'market_context_15m', 'shape'), [24, 6]),
    ('clock', ('reward', 'clock_protocol'), 'WALL_CLOCK'),
    ('lots', ('trading', 'lots'), 1),
    ('test_fold', ('teacher', 'fold'), 'TEST'),
    ('leak', ('observation', 'policy_observation_fields'), ['quality_score']),
    ('fake_oracle', ('curriculum', 'behaviors', -1, 'parameters', 'source'), {'certified': True}),
    ('unknown_behavior', ('curriculum', 'behaviors', 0, 'kind'), 'risk_selected'),
    ('blank_mode', ('spec_mode',), ''),
    ('blank_source', ('data_identity', 'source_manifest'), ''),
    ('empty_sources', ('data_identity', 'market_sources'), []),
    ('implicit_formal', ('spec_mode',), 'formal'),
    ('forged_ready', ('ready',), True),
    ('forged_matrix', ('stage_contracts',), {'FOUNDATION_WM': {'quality_actor/': 'TRAIN'}}),
]


@pytest.mark.parametrize('case,path,value', INVALID_CASES, ids=[x[0] for x in INVALID_CASES])
def test_WP01_02_to_12_cli_negative_status_exit_files_and_no_side_effects(tmp_path, case, path, value):
    raw = draft(); target = raw
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    source = spec_file(tmp_path, raw); before = tree(tmp_path)
    result, report = cli(source, tmp_path / 'diagnostic')
    assert result.returncode == report['exit_code'] == 2, (case, result.stderr)
    assert_not_admitted(report)
    assert report['status'] == 'INVALID_SPEC_OR_SOURCE'
    assert tree(tmp_path) == before


@pytest.mark.parametrize('kind', ['missing_mode', 'missing_source', 'duplicate_json', 'malformed_json'])
def test_WP01_12_16_missing_or_invalid_json_never_becomes_draft(tmp_path, kind):
    raw = draft()
    if kind == 'missing_mode': del raw['spec_mode']
    if kind == 'missing_source': del raw['data_identity']['source_manifest']
    path = spec_file(tmp_path, raw)
    if kind == 'duplicate_json': path.write_text('{"spec_mode":"formal",' + path.read_text()[1:])
    if kind == 'malformed_json': path.write_text('{')
    before = tree(tmp_path)
    result, report = cli(path, tmp_path / 'diagnostic')
    assert result.returncode == report['exit_code'] == 2
    assert_not_admitted(report)
    assert tree(tmp_path) == before


@pytest.mark.parametrize('kind', ['missing', 'hash', 'profile', 'test_source', 'teacher_reward', 'forged', 'malformed_pool', 'false_eligibility'])
def test_WP01_06_08_formal_source_errors_are_nonzero_and_write_nothing(tmp_path, kind):
    raw = complete_metadata(tmp_path)
    ref = raw['data_identity']['source_manifest']
    pool_path = tmp_path / ref['path']
    if kind == 'missing': ref['path'] = 'absent.json'
    elif kind == 'hash': ref['sha256'] = '0' * 64
    elif kind == 'profile': raw['data_identity']['e_profile_schema_sha256'] = '0' * 64
    elif kind in ('test_source', 'forged', 'malformed_pool', 'false_eligibility'):
        pool = json.loads(pool_path.read_text())
        if kind == 'test_source': pool['files'][0]['role'] = 'test'
        elif kind == 'forged': pool['certified'] = True
        elif kind == 'false_eligibility': pool['files'][0]['eligible_for_training'] = 'false'
        else: pool['files'] = [dict(role='train', eligible_for_training=True)]
        pool_path.write_text(json.dumps(pool)); ref['sha256'] = p.sha256(pool_path)
    else:
        teacher = tmp_path / raw['data_identity']['teacher_manifest']['path']
        body = {k: v for k, v in read_manifest(teacher).items() if k != 'content_hash'}
        body['reward_schema'] = 'FG_SINGLE3_REWARD_MARKET_MINUTE_V2'
        write_json(teacher, signed(body)); raw['data_identity']['teacher_manifest']['sha256'] = p.sha256(teacher)
    path = spec_file(tmp_path, raw); before = tree(tmp_path)
    result, report = cli(path, tmp_path / 'formal_report')
    assert result.returncode == report['exit_code'] == 2, result.stderr
    assert_not_admitted(report)
    assert tree(tmp_path) == before


@pytest.mark.parametrize('kind', ['existing', 'reserved', 'formal_root', 'symlink_parent', 'source_directory'])
def test_WP01_16_report_paths_cannot_overwrite_or_touch_protected_outputs(tmp_path, kind):
    raw = complete_metadata(tmp_path, mode='draft') if kind == 'source_directory' else draft()
    path = spec_file(tmp_path, raw)
    out = tmp_path / 'diagnostic'
    if kind == 'existing':
        out.mkdir(); (out / 'keep.txt').write_text('keep')
    elif kind == 'reserved': out = tmp_path / 'prepared' / 'new'
    elif kind == 'formal_root': out = Path('/data/logdir/fg_single3/s3_v2/wp01_forbidden_output')
    elif kind == 'source_directory': out = tmp_path / 'metadata' / 'report'
    else:
        (tmp_path / 'link').symlink_to(tmp_path, target_is_directory=True)
        out = tmp_path / 'link' / 'report'
    before = tree(tmp_path)
    result, report = cli(path, out)
    assert result.returncode == report['exit_code'] == 2
    assert_not_admitted(report)
    assert tree(tmp_path) == before
    if kind == 'existing': assert (out / 'keep.txt').read_text() == 'keep'
    else: assert not out.exists()


@pytest.mark.parametrize('boundary', ['reward', 'observation', 'score', 'public_guard'])
def test_WP01_native_validator_failures_are_not_suppressed(tmp_path, monkeypatch, boundary):
    if boundary == 'reward':
        def reject(**kw): raise ValueError('native reward rejected')
        monkeypatch.setattr(p, 'Single3RewardConfig', reject)
    elif boundary == 'observation':
        native = p.observation_contract()
        native['fields']['market_context_15m']['shape'] = [1, 7]
        monkeypatch.setattr(p, 'observation_contract', lambda: native)
    elif boundary == 'score': monkeypatch.setattr(p, 'SCORE_POLICY', 'different_score_policy')
    else: monkeypatch.setattr(p, 'require_ready', lambda x: x)
    path = spec_file(tmp_path, draft()); before = tree(tmp_path)
    with pytest.raises(ValueError): p.preflight(path, report_dir=tmp_path / 'diagnostic')
    assert tree(tmp_path) == before


def test_WP01_14_real_legacy_quality_and_e_configurations_keep_their_contracts():
    from gym_trading_env.research.fg_s3_v2_experiment import read_settings
    from dreamerv3.configuration import read_config_yaml
    from dreamerv3.fg_s3_composition import normalize_config
    base = ROOT / 'configs/experiments/fg_s3_v2'
    legacy = read_settings(base / 'QUALITY_V2_CPU_SMOKE.yaml')
    assert legacy['quality_stage'] == 'HEADS_ONLY'
    assert legacy['reward']['schema'] == 'FG_SINGLE3_REWARD_MARKET_MINUTE_V2'
    for filename, teaching in [('E_ADMISSION_V2_A2_CPU_20261015_v1', 'a2'),
            ('E_ADMISSION_V2_TUTOR_CPU_20261015_v1', 'tutor_cd'),
            ('E_ADMISSION_V2_B2_CPU_20261015_v1', 'b2')]:
        raw = read_config_yaml(base / (filename + '.yaml'))
        resolved = normalize_config(raw)['resolved_composition']
        assert resolved['encoder'] == 'e_multiscale_1m15m'
        assert resolved['teaching'] == teaching and resolved['support_state'] == 'ready'

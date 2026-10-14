"""F0 contracts; controlled metadata fixtures are not real E checkpoints."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from dreamerv3.fg_s3_composition import (CONFIG_SCHEMA, LEGACY_SCHEMA, KEY, ENCODERS,
    TEACHING, resolve_composition, normalize_config, require_ready, validate_records)
from dreamerv3.configuration import read_config_yaml
from gym_trading_env.research.fg_replay import signed, write_json
from embodied.run.fg_checkpoint import file_hash
from gym_trading_env.research import fg_s3_v2_experiment as experiment

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('encoder', ENCODERS)
@pytest.mark.parametrize('teaching', TEACHING)
def test_all_eight_compositions_have_unique_state_and_one_base_bc(encoder, teaching):
    raw = dict(schema=CONFIG_SCHEMA, composition=dict(encoder=encoder, teaching=teaching))
    resolved = resolve_composition(raw)
    expected = 'waiting_tutor_b2' if teaching == 'tutor_cd_b2' else (
        'waiting_e' if encoder != 'legacy_1m' else 'ready')
    assert resolved['support_state'] == expected
    assert resolved['base_oracle_bc'] is True and resolved['schema_version'] == 1
    assert resolved == normalize_config(raw, executable=False)[KEY]
    assert resolved['legacy_mode'] == {'a2':'a2','tutor_cd':'tutor','b2':'b2','tutor_cd_b2':None}[teaching]


def test_pure_parser_runs_without_site_packages_or_jax():
    script = ('import sys; from dreamerv3.fg_s3_composition import resolve_composition; '
              'assert resolve_composition({"mode":"a2"})["support_state"]=="ready"; '
              'assert "jax" not in sys.modules and "embodied" not in sys.modules')
    env = dict(os.environ, PYTHONPATH=str(ROOT.parent/'dreamerv3'))
    subprocess.run([sys.executable, '-S', '-B', '-c', script], env=env, check=True)


@pytest.mark.parametrize('raw', [
    {'mode':'a2','composition':{'encoder':'legacy_1m','teaching':'a2'}},
    {'composition':{'encoder':'legacy_1m'}}, {'composition':{'teaching':'a2'}},
    {'composition':None}, {'composition':{'encoder':'unknown','teaching':'a2'}},
    {'composition':{'encoder':'legacy_1m','teaching':True}},
    {'composition':{'encoder':'legacy_1m','teaching':'a2','ready':True}},
    {'resolved_composition':{}}, {'mode':'e'},
])
def test_invalid_or_duplicate_raw_configuration_fails_closed(raw):
    with pytest.raises(ValueError): resolve_composition(raw)


def test_schema_and_legacy_mode_contracts():
    for mode in ('a2','tutor','b2','c2'):
        raw = dict(schema=LEGACY_SCHEMA,mode=mode,beta=1)
        assert normalize_config(raw) == raw and KEY not in normalize_config(raw)
    assert resolve_composition({'mode':'c2'}) is None
    for raw in ({'schema':CONFIG_SCHEMA,'mode':'a2'},
                {'schema':LEGACY_SCHEMA,'composition':{'encoder':'legacy_1m','teaching':'a2'}}):
        with pytest.raises(ValueError): normalize_config(raw)
    resolved = resolve_composition({'mode':'a2'})
    with pytest.raises(ValueError): require_ready({**resolved, 'base_oracle_bc':1})
    future = resolve_composition({'composition':{'encoder':ENCODERS[1],'teaching':'a2'}})
    with pytest.raises(ValueError): require_ready({**future,'support_state':'ready','use_e':False})


@pytest.mark.parametrize('teaching,preset,mode', [
    ('a2','A2_MARKET_MINUTE_GPU_1K','a2'),
    ('tutor_cd','TUTOR_CD_FULL_E2_MM_10K','tutor'),
    ('b2','B2_FULL_MARKET_MINUTE_FORMAL_1M','b2'),
])
def test_ready_settings_keep_all_legacy_hyperparameters(tmp_path, teaching, preset, mode):
    path = ROOT/'configs/experiments/fg_s3_v2'/f'{preset}.yaml'
    raw = read_config_yaml(path)
    newer = {k:v for k,v in raw.items() if k != 'mode'}
    newer.update(schema=CONFIG_SCHEMA,composition=dict(encoder='legacy_1m',teaching=teaching))
    config = tmp_path/'composition.yaml'
    from ruamel.yaml import YAML
    YAML().dump(newer,config)
    old, new = experiment.read_settings(path), experiment.read_settings(config)
    for key in ('schema','config_path','config_sha256'):
        old.pop(key);new.pop(key)
    assert new.pop(KEY) == resolve_composition({'mode':mode})
    assert new == old


WAITING = [(ENCODERS[1], t) for t in TEACHING[:3]] + [(e, TEACHING[3]) for e in ENCODERS]


@pytest.mark.parametrize('encoder,teaching', WAITING)
def test_unimplemented_config_cli_and_direct_prepare_create_no_artifacts(tmp_path, encoder, teaching):
    path=tmp_path/'future.yaml'
    path.write_text(f'schema: {CONFIG_SCHEMA}\ncomposition:\n  encoder: {encoder}\n  teaching: {teaching}\n')
    error='S3_TUTOR_B2_NOT_IMPLEMENTED' if teaching==TEACHING[3] else 'S3_E_ENCODER_NOT_IMPLEMENTED'
    with pytest.raises(ValueError,match=error): experiment.prepare(path)
    for command in ('prepare','start','resume','evaluate'):
        result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/fg_s3_v2_experiment.py'),
            command,'--config',str(path)],capture_output=True,text=True)
        assert result.returncode != 0 and error in result.stderr
    described=subprocess.run([sys.executable,'-B',str(ROOT/'tools/fg_s3_v2_experiment.py'),
        'describe-composition','--config',str(path)],capture_output=True,text=True,check=True)
    assert json.loads(described.stdout)[KEY]['support_state'] != 'ready'
    assert list(tmp_path.iterdir()) == [path]


def bundle(root, resolved):
    """Signed CPU metadata only; no model/data or legitimate training receipt."""
    root.mkdir()
    settings=dict(schema=CONFIG_SCHEMA,mode=resolved['legacy_mode'],resolved_composition=resolved)
    records={'settings.json':{'settings':settings},'model_config.json':{KEY:resolved,'config':{}},
        'source_admission.json':{KEY:resolved,'target_model_identity':{KEY:resolved}}}
    for name,value in records.items(): write_json(root/name,signed(value))
    manifest=signed({'code':{},KEY:resolved,'files':{name:file_hash(root/name) for name in records}})
    write_json(root/'manifest.json',manifest)
    return settings


@pytest.mark.parametrize('encoder,teaching', WAITING)
def test_waiting_prepared_live_and_eval_are_rejected_before_model_or_output(tmp_path,monkeypatch,encoder,teaching):
    from embodied.run import fg_s3_v2_train as train
    from gym_trading_env.research import fg_s3_v2_evaluation as evaluation
    resolved=resolve_composition({'composition':dict(encoder=encoder,teaching=teaching)})
    prepared=tmp_path/'prepared';bundle(prepared,resolved)
    error='S3_TUTOR_B2_NOT_IMPLEMENTED' if teaching==TEACHING[3] else 'S3_E_ENCODER_NOT_IMPLEMENTED'
    with pytest.raises(ValueError,match=error): experiment.settings_from_prepared(prepared)
    monkeypatch.setattr(train,'execution_code',lambda:{})
    monkeypatch.setattr(train,'make_agent',lambda *a,**k:pytest.fail('Model factory must not execute'))
    for resume in (False,True):
        with pytest.raises(ValueError,match=error): train.LiveRun(prepared,tmp_path/'run',resume=resume)
    run=tmp_path/'eval_run';run.mkdir()
    write_json(run/'run_identity.json',signed({'identity':{KEY:resolved}}))
    with pytest.raises(ValueError,match=error): evaluation.evaluate(prepared,run)
    assert not (tmp_path/'run').exists() and not (run/'evaluations').exists()


def test_all_new_records_require_same_composition_and_legacy_records_stay_absent(tmp_path):
    resolved=resolve_composition({'mode':'tutor'})
    prepared=tmp_path/'prepared';settings=bundle(prepared,resolved)
    assert experiment.settings_from_prepared(prepared)==settings
    docs=[{KEY:copy.deepcopy(resolved)} for _ in range(6)]
    assert validate_records(settings,*docs)==resolved
    docs[-1][KEY]=resolve_composition({'mode':'b2'})
    with pytest.raises(ValueError): validate_records(settings,*docs)
    with pytest.raises(ValueError): validate_records(settings,{})
    with pytest.raises(ValueError): validate_records({**settings,'mode':'a2'},*docs)
    assert validate_records({'schema':LEGACY_SCHEMA,'mode':'c2'},{}) is None
    with pytest.raises(ValueError): validate_records({'schema':LEGACY_SCHEMA,'mode':'tutor'},{KEY:resolved})


def test_source_and_model_factories_reject_waiting_before_any_data_or_model(tmp_path):
    from embodied.run.fg_s3_source_admission import inspect_source
    from embodied.run.fg_s3_source_transfer import make_agent
    for encoder,teaching in WAITING:
        value=resolve_composition({'composition':dict(encoder=encoder,teaching=teaching)})
        error='S3_TUTOR_B2_NOT_IMPLEMENTED' if teaching==TEACHING[3] else 'S3_E_ENCODER_NOT_IMPLEMENTED'
        with pytest.raises(ValueError,match=error):inspect_source('direct_init',composition=value)
        with pytest.raises(ValueError,match=error):make_agent({},tmp_path/'model',composition=value)
    assert not (tmp_path/'model').exists()


def test_reviewed_reader_warm_start_still_rejects_unknown_changes(monkeypatch):
    from embodied.run import fg_s3_source_admission as admission
    from gym_trading_env.research.fg_s3_reader_compat import PRODUCERS
    mapping={n:h for n,h in PRODUCERS.items() if not n.endswith('/entry_env.py')}
    result=admission.verify_producer_code(mapping,warm_start=True)
    assert all(r['disposition']=='AUTHENTICATED_REVIEWED_READER_PARAMETER_WARM_START' for r in result.values())
    with pytest.raises(ValueError):admission.verify_producer_code(mapping,warm_start=False)
    name=next(iter(mapping))
    with pytest.raises(ValueError,match='Unreviewed'):admission.verify_producer_code({name:'0'*64},warm_start=True)
    original=admission.file_hash
    monkeypatch.setattr(admission,'file_hash',lambda p:'0'*64 if str(p).endswith('entry_market_cache.py') else original(p))
    with pytest.raises(ValueError,match='Unreviewed'):admission.verify_producer_code(mapping,warm_start=True)


def test_model_factory_rejects_encoder_identity_mismatch_before_building(tmp_path):
    from embodied.run.fg_s3_source_transfer import make_agent
    value=resolve_composition({'mode':'a2'})
    with pytest.raises(ValueError,match='S3_COMPOSITION_ENCODER_MISMATCH'):
        make_agent({'agent':{'enc':{'typ':'unknown'}}},tmp_path/'model',composition=value,
                   extra_agent_config={'s3_v2_mode':'a2'})
    assert not (tmp_path/'model').exists()

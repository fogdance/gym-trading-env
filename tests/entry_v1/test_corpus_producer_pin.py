"""Real CLI dispatch and stage pin forwarding, without production G3/training."""
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from gym_trading_env.research.fg_replay import signed, write_json
from gym_trading_env.research.fg_s3_e_quality_supervision import QualitySupervisionAdmission, SupervisionSource
from wp03_fixture import ROOT


STAGES = dict(f1='foundation', f2='heads', f3='actor')


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.delenv('LD_LIBRARY_PATH', raising=False)
    spec = importlib.util.spec_from_file_location('corpus_pin_cli', ROOT/'tools/fg_s3_v2_experiment.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('stage', STAGES)
@pytest.mark.parametrize('action', ['prepare', 'read-check', 'start', 'resume'])
def test_cli_forwards_exact_pin_to_each_public_route(cli, tmp_path, monkeypatch, stage, action):
    name = STAGES[stage]
    owner = importlib.import_module('gym_trading_env.research.fg_s3_e_quality_'+name+('_checks' if action=='read-check' else ''))
    function = 'start_prepared' if action in ('start','resume') else 'read_check' if action=='read-check' else 'prepare'
    calls = []
    def receive(*args, **kwargs):
        calls.append((args,kwargs)); return dict(status='STUB_ROUTING_ONLY_NOT_CERTIFIED')
    monkeypatch.setattr(owner, function, receive)
    pin = str(tmp_path/'reviewed.json'); producer = str(tmp_path/'frozen/gym-trading-env')
    argv = ['cli', stage+'-'+action, '--producer-gym', producer,
            '--producer-lock',pin,'--producer-lock-sha256','a'*64]
    if action in ('start','resume'):
        prepared = tmp_path/'prepared'; prepared.mkdir()
        write_json(prepared/'settings.json', signed(dict(settings=dict(device='cpu'))))
        argv += ['--prepared',str(prepared),'--authorize-training']
    else:
        argv += ['--config',str(tmp_path/'config.yaml'),'--source-purpose','native-verification']
        if action=='read-check': argv += ['--output-dir',str(tmp_path/'out')]
    monkeypatch.setattr(sys,'argv',argv)
    cli.main()
    assert len(calls)==1
    kwargs=calls[0][1]
    assert kwargs['producer_gym']==producer and kwargs['producer_lock']==pin
    assert kwargs['producer_lock_sha256']=='a'*64
    if action in ('start','resume'): assert kwargs['resume']==(action=='resume')


@pytest.mark.parametrize('stage', STAGES)
def test_stage_preflight_reaches_original_admission_with_pin(monkeypatch, stage):
    owner=importlib.import_module('gym_trading_env.research.fg_s3_e_quality_'+STAGES[stage])
    monkeypatch.setattr(owner,'validate_settings',lambda settings:settings)
    source=object();monkeypatch.setattr(SupervisionSource,'from_corpus',lambda *a,**k:source)
    class Reached(Exception):pass
    def stop(actual,**kwargs):
        assert actual is source
        assert kwargs['producer_lock']=='reviewed.json' and kwargs['producer_lock_sha256']=='a'*64
        assert kwargs['producer_gym']=='frozen/gym-trading-env'
        raise Reached
    monkeypatch.setattr(QualitySupervisionAdmission,'preflight',stop)
    with pytest.raises(Reached):
        owner.preflight(dict(corpus_root='corpus'),purpose='native-verification',
            producer_gym='frozen/gym-trading-env',producer_lock='reviewed.json',producer_lock_sha256='a'*64)


@pytest.mark.parametrize('stage', STAGES)
@pytest.mark.parametrize('pin', [{},dict(producer_lock='pin.json'),dict(producer_lock_sha256='a'*64)])
def test_formal_stage_rejects_missing_or_half_pin_before_source_io(stage,pin):
    owner=importlib.import_module('gym_trading_env.research.fg_s3_e_quality_'+STAGES[stage])
    with pytest.raises(ValueError,match='EXPLICIT_PRODUCER_PIN_SHA256_REQUIRED'):
        owner.preflight({},purpose='native-verification',**pin)


@pytest.mark.parametrize('flags', [ ['--producer-lock','pin.json'],
    ['--producer-lock-sha256','a'*64],
    ['--producer-lock','pin.json','--producer-lock-sha256','a'*64] ])
def test_unrelated_cli_or_half_pair_rejects_without_output(cli,monkeypatch,flags):
    monkeypatch.setattr(sys,'argv',['cli','f1-probe','--output-dir','unused',*flags])
    with pytest.raises(SystemExit) as error:cli.main()
    assert error.value.code==2

import json
from pathlib import Path
import subprocess
import sys

import pytest

from gym_trading_env.research.fg_s3_e_quality_supervision import SupervisionSource,QualitySupervisionAdmission
from wp03_fixture import ROOT,metadata_corpus


def test_D09_source_cli_development_is_non_authenticating_and_creates_no_training_files(tmp_path):
    import uuid
    root=metadata_corpus(tmp_path/'corpus')
    output=ROOT/'artifacts/fg_s3_v2/e_quality_wp03'/('cli_'+uuid.uuid4().hex)
    run=subprocess.run([sys.executable,'-B',str(ROOT/'tools/fg_s3_e_quality_reader.py'),'source',
        '--corpus',str(root),'--out',str(output)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr+run.stdout
    receipt=json.loads(run.stdout.splitlines()[-1])
    assert receipt['status']=='DEVELOPMENT_VALIDATED' and receipt['admission_status']=='NOT_ADMITTED'
    assert receipt['training_admitted'] is False and not receipt['prepared_certified'] and not receipt['checkpoint_certified']
    assert sorted(p.name for p in output.iterdir())==['WP03_SOURCE_ADMISSION_RECEIPT.json']


@pytest.mark.parametrize('kind',['test','empty_purpose','bad_output'])
def test_D10_cli_negative_exit_and_no_output_side_effects(tmp_path,kind):
    import uuid
    root=tmp_path/'TEST' if kind=='test' else metadata_corpus(tmp_path/'corpus')
    output=ROOT/'artifacts/fg_s3_v2/e_quality_wp03'/('rejected_'+uuid.uuid4().hex)
    if kind=='bad_output':output=tmp_path/'prepared'
    command=[sys.executable,'-B',str(ROOT/'tools/fg_s3_e_quality_reader.py'),'source','--corpus',str(root),'--out',str(output)]
    if kind=='empty_purpose':command.extend(['--purpose',''])
    run=subprocess.run(command,capture_output=True,text=True)
    assert run.returncode==2
    assert not output.exists()


def test_D11_native_verification_rejects_wrong_producer_before_launch_or_write(tmp_path):
    root=metadata_corpus(tmp_path/'corpus');log=tmp_path/'must_not_exist.log'
    with pytest.raises(ValueError,match='PRODUCER_MISMATCH'):
        QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(root),
            purpose='native-verification',producer_gym=ROOT,log_path=log)
    assert not log.exists()


@pytest.mark.parametrize('command',['prepare','start','resume','evaluate'])
def test_D12_public_E_quality_lifecycle_stays_closed_without_outputs(tmp_path,command):
    config=tmp_path/'quality_e.yaml'
    config.write_text('schema: FG_S3_V2_COMPOSITION_CONFIG_V1\ncomposition:\n  encoder: e_multiscale_1m15m\n  teaching: quality_v2\n')
    before=sorted(p.name for p in tmp_path.iterdir())
    run=subprocess.run([sys.executable,'-B',str(ROOT/'tools/fg_s3_v2_experiment.py'),command,
                        '--config',str(config)],capture_output=True,text=True)
    assert run.returncode==1 and 'S3_QUALITY_E_NOT_IMPLEMENTED' in run.stderr
    assert sorted(p.name for p in tmp_path.iterdir())==before


def test_D13_matching_head_but_modified_frozen_verifier_cannot_certify(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from gym_trading_env.research import fg_s3_e_quality_supervision as module
    root=metadata_corpus(tmp_path/'corpus');log=tmp_path/'forbidden.log'
    def head(argv,**kwargs):
        if 'rev-parse' in argv:
            role='dreamer' if str(argv[2]).endswith('/dreamerv3') else 'gym'
            return module.PRODUCER_HEADS[role]+'\n'
        return ''
    def run(argv,**kwargs):
        if argv[0]=='git':return SimpleNamespace(returncode=1)
        raise AssertionError('Launched a modified native verifier')
    monkeypatch.setattr(module.subprocess,'check_output',head)
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(ValueError,match='FROZEN_PRODUCER_CODE_CHANGED'):
        module.native_verification(SupervisionSource.from_corpus(root),producer_gym=ROOT,log_path=log)
    assert not log.exists()

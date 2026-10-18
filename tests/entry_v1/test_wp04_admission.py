import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gym_trading_env.research.fg_s3_e_quality_foundation import (
    preflight,prepare,validate_settings,create_agent,require_training_authority)
from dreamerv3.fg_s3_e_quality_foundation import model_extensions,parameter_layout
from wp03_fixture import metadata_corpus,ROOT
from wp04_fixture import settings


def test_A01_engineering_receipt_is_not_native_or_training_authority(tmp_path):
    p=preflight(settings(metadata_corpus(tmp_path/'corpus')))
    assert p.receipt['status']=='F1_ENGINEERING_READY'
    assert p.receipt['training_admitted'] is False and p.receipt['research_admitted'] is False
    assert 'F1_NATIVE_G3_NOT_RUN' in p.receipt['blockers']
    with pytest.raises(ValueError,match='AUTHORIZATION_REQUIRED'):require_training_authority(p)
    with pytest.raises(ValueError,match='ACTUAL_NATIVE_G3_REQUIRED'):
        require_training_authority(p,authorized=True)
    forged=copy.deepcopy(p.receipt);forged.update(status='F1_DATA_READY',training_admitted=True)
    with pytest.raises(ValueError,match='ACTUAL_NATIVE_G3_REQUIRED'):
        require_training_authority(forged,authorized=True)


@pytest.mark.parametrize('key,value',[('stage','QUALITY_HEADS'),('sampling_stage','FOUNDATION_WM'),
    ('sampling_spec',{'mode':'MIXED'}),('burnin_steps',0),('target_steps',True),('model_init',{'seed':202,'presets':[]})])
def test_A02_stage_shape_or_old_initialization_never_falls_back(tmp_path,key,value):
    s=settings(tmp_path/'unused');s[key]=value
    with pytest.raises(ValueError):validate_settings(s)


def test_A03_wrong_reward_observation_or_nontrain_rejected_without_prepare(tmp_path):
    from gym_trading_env.research.fg_replay import signed,write_json
    for attack in ('reward','observation','fold'):
        root=metadata_corpus(tmp_path/attack)
        name='course_v2.json' if attack=='reward' else 'corpus_manifest.json'
        p=root/name;d=json.loads(p.read_text())
        if attack=='reward':d['reward']['schema']='OLD_C1'
        elif attack=='observation':d['observation']['fields']['market_macd_5m']['shape']=[3]
        else:d['fold']='TEST'
        p.write_text(json.dumps(d))
        with pytest.raises(ValueError):preflight(settings(root))
    with pytest.raises(ValueError):preflight(settings(tmp_path/'TEST'))
    assert not (tmp_path/'prepared').exists()


@pytest.mark.parametrize('command',['f1-start','f1-resume'])
def test_A04_unauthorized_cli_fails_before_reading_or_creating_run(tmp_path,command):
    out=tmp_path/'run'
    run=subprocess.run([sys.executable,'-B',str(ROOT/'tools/fg_s3_v2_experiment.py'),command,
        '--prepared',str(tmp_path/'absent'),'--run-dir',str(out)],capture_output=True,text=True)
    assert run.returncode==1 and 'F1_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED' in run.stderr
    assert not out.exists()


def test_A05_fresh_complete_E_quality_model_and_no_teacher_training_inputs(tmp_path):
    s=validate_settings(settings(tmp_path/'not_read'))
    a=create_agent(s,tmp_path/'model',controlled_small=True)
    layout=parameter_layout(a.params)
    assert a.config.quality_stage=='FOUNDATION_WM'
    assert not any(k.startswith(('teacher_','quality_')) for k in a.spaces)
    assert int(a.n_updates)==0
    assert {'market_macd_5m','market_context_15m','market_seq_e_shared'} <= a.obs_space.keys()
    assert any(k.startswith('enc/macd5/') for k in layout)
    assert all(a.config[k]==0. for k in a.config if k.startswith('quality_') and k.endswith('_lambda'))
    from dreamerv3.fg_s3a_actor import parameter_digests
    before=parameter_digests(a.params)
    b=create_agent(s,tmp_path/'fresh_again',controlled_small=True)
    assert before==parameter_digests(b.params)
    assert all(np.isfinite(np.asarray(v)).all() for v in a.params.values())


def test_A06_F1_must_not_enable_teacher_or_decision_losses():
    from dreamerv3.fg_s3_quality_v2 import quality_extensions
    cfg=model_extensions(8,4)
    for key,value in [('quality_q_c1_lambda',.1),('quality_prediction_admitted',True),
                      ('quality_stage','HEADS_ONLY'),('quality_f1_protocol','FAKE')]:
        with pytest.raises(ValueError):quality_extensions({**cfg,key:value})


def test_A07_prepare_only_writes_engineering_files_and_refuses_overwrite(tmp_path):
    import yaml
    from wp03_fixture import native_corpus
    root=native_corpus(tmp_path/'corpus',minutes=16,chunk=20);config=tmp_path/'config.yaml'
    s=settings(root);config.write_text(yaml.safe_dump(s))
    receipt=prepare(config)
    assert receipt['status']=='F1_ENGINEERING_READY'
    assert sorted(p.name for p in Path(s['prepared_dir']).iterdir())==['f1_preparation.json','join.json','join.sqlite','reader_binding.json','settings.json']
    assert not Path(s['run_dir']).exists()
    with pytest.raises(ValueError,match='OUTPUT_MUST_BE_NEW'):prepare(config)

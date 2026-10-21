"""Exercise the real evaluation entry and Actor/Driver with a CPU fixture.

The synthetic Corpus lacks production source-pool/account metadata. Mock only
that external metadata plumbing; load the real saved Agent and run the native
financial Env, Actor policy, Replay and reconciler. No real TRAIN data is used.
"""
import json
import uuid
from pathlib import Path
from unittest.mock import patch


def evaluate(settings, preparation, root):
    from wp02_fixture import native_factory
    from e_data_v1.provider import ETrainAdmission
    from gym_trading_env.research import fg_replay
    from gym_trading_env.research import fg_s3_e_quality_actor as actor
    from gym_trading_env.research import fg_s3_e_quality_actor_checks as checks
    from gym_trading_env.research import fg_s3_e_quality_corpus_cli as corpus_cli
    from gym_trading_env.research import fg_s3_e_quality_strict as strict
    from gym_trading_env.research import fg_s3_quality_teacher_v2 as teacher
    from gym_trading_env.research.fg_s3_e_quality_foundation import PROJECT
    outputs=PROJECT/'artifacts/fg_s3_v2/e_quality_wp06/default_checkpoint_resume'/uuid.uuid4().hex
    outputs.mkdir(parents=True)
    factory,_,profile=native_factory(outputs/'market',minutes=40,chunk=9)
    env=factory()
    try:initial=env.export_state()
    finally:env.close()
    corpus=Path(settings['corpus_root'])
    snapshot=dict(contract=initial['contract'] if 'contract' in initial else 'CZCE.FG2701',
        source_pool={'path':'synthetic-fixture-pool'},env_receipt={'path':'synthetic-fixture-receipt'},row={})
    original_manifest=fg_replay.read_manifest
    original_text=Path.read_text

    def manifest(path):
        return snapshot if Path(path)==corpus/'source_lock.json' else original_manifest(path)

    def text(path,*args,**kwargs):
        return json.dumps(initial) if path==corpus/'initial_account.json' else original_text(path,*args,**kwargs)

    def forbidden(*args,**kwargs):raise AssertionError('Evaluation constructed a Teacher')

    with patch.object(fg_replay,'read_manifest',manifest),patch.object(Path,'read_text',text),\
            patch.object(actor,'preflight',return_value=preparation),\
            patch.object(corpus_cli,'source_inputs',return_value=(snapshot,None,None)),\
            patch.object(corpus_cli,'_factory',return_value=factory),\
            patch.object(ETrainAdmission,'from_train_pool',return_value=profile),\
            patch.object(strict,'QualityStrictTeacher',forbidden),\
            patch.object(teacher,'NativeQualityTeacher',forbidden):
        result=checks.evaluate_checkpoint(root,root/'ckpt/step_000000000002',outputs/'evaluation',steps=8)
    assert result['complete_account_replay']['status']=='PASS'
    assert result['scope']=='FINITE_TRAIN_READ_ONLY_NOT_RESEARCH'
    assert not result['teacher_used'] and result['all_parameter_bytes_equal']
    assert result['real_TRAIN_updates']==0
    result['fixture_evaluation_output']=str(outputs/'evaluation')
    (root/'evaluation_fixture_result.json').write_text(json.dumps(result,indent=2)+'\n')

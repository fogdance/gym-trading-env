"""F4 lifecycle and authority boundaries; no production/source admission inferred."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from embodied.run.fg_s3_e_quality_online_rl import RLAdmissionState,validate_rl_policy
from embodied.run.fg_s3_e_quality_online_probe import NumericProbe,require_numeric_probe,state_digest

ROOT=Path(__file__).resolve().parents[2]
POLICY=dict(warmup_env_steps=32,calibration_batches=20,calibration_horizon=3,calibration_seed=202)


def test_explicit_finite_prefix_and_no_policy_for_bc_or_shadow():
    spec=dict(execution_mode='ONLINE_RL_CONTROLLED',env_steps=64,collect_every=16)
    assert validate_rl_policy(POLICY,spec)==POLICY
    for bad in (None,{}, {**POLICY,'warmup_env_steps':64},{**POLICY,'warmup_env_steps':17},
        {**POLICY,'calibration_batches':0},{**POLICY,'calibration_seed':True}):
        with pytest.raises(ValueError,match='EXPLICIT_BOUNDED'):validate_rl_policy(bad,spec)
    for mode in ('SHADOW','ONLINE_SUPERVISED'):
        assert validate_rl_policy(None,{**spec,'execution_mode':mode}) is None
        with pytest.raises(ValueError,match='ONLY_FOR_CONTROLLED'):validate_rl_policy(POLICY,{**spec,'execution_mode':mode})


def test_state_machine_rejects_skipping_checks_and_keeps_precise_snapshot():
    state=RLAdmissionState('RUN',POLICY)
    with pytest.raises(ValueError,match='ILLEGAL'):state.transition('RL_ACTIVE')
    state.transition('WM_CALIBRATING');state.state['attempt']=1
    state.transition('RL_PROBING');state.state['technical']=dict(evidence_hash='BOUND_SYNTHETIC_STATE_CONTRACT')
    state.transition('AWAIT_OPERATOR_REVIEW')
    snapshot=state.save();restored=RLAdmissionState('RUN',POLICY);restored.load(snapshot)
    assert restored.save()==snapshot
    bad=copy.deepcopy(snapshot);bad['technical']=None
    with pytest.raises(ValueError,match='TECHNICAL_EVIDENCE'):restored.load(bad)
    restored.transition('RL_ACTIVE');restored.transition('FINAL_VERIFY');restored.transition('COMPLETE_VERIFIED')
    with pytest.raises(ValueError,match='ILLEGAL'):restored.transition('WARMUP')


def test_fabricated_probe_dict_or_object_cannot_open_numeric_gate():
    for fake in (dict(ready=True),NumericProbe(dict(ready=True))):
        with pytest.raises(ValueError,match='ACTUAL_ISOLATED_RL_PROBE'):require_numeric_probe(fake)
    import numpy as np
    assert state_digest(dict(a=np.array([-0.],np.float32)))!=state_digest(dict(a=np.array([0.],np.float32)))
    assert state_digest(dict(a=np.array([1],np.int32)))!=state_digest(dict(a=np.array([1],np.int64)))
    import datetime
    value=datetime.datetime(2026,10,21,1,2,3,tzinfo=datetime.timezone.utc)
    assert state_digest(value)==state_digest(copy.deepcopy(value))


def test_cold_gate_restore_never_restores_live_operator_permission():
    from embodied.run.fg_s3_e_quality_online_gate import OnlineGates
    from gym_trading_env.research.fg_s3_e_quality_online_checks import formal_mask_policy
    from wp07_fixture import f3_source
    policy=formal_mask_policy(f3_source());gate=OnlineGates('RUN',policy)
    assert gate.mask.min_train_samples>=25000 and gate.mask.required_passes==20
    gate.formal_admitted=True
    snapshot=gate.save();assert snapshot['formal_admitted'] is False
    snapshot['formal_admitted']=True
    restored=OnlineGates('RUN',policy);restored.load(snapshot)
    assert not restored.formal_admitted and not restored.report()['formal_rl_admitted']
    with pytest.raises(ValueError,match='GATE_CLOSED'):restored.require_rl(None,authorized=True)


def test_early_review_rejected_and_packet_cli_is_read_only(tmp_path,monkeypatch):
    monkeypatch.delenv('LD_LIBRARY_PATH',raising=False)
    spec=importlib.util.spec_from_file_location('batch3_rl_cli',ROOT/'tools/fg_s3_v2_experiment.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    monkeypatch.setattr(sys,'argv',['cli','online-start','--prepared',str(tmp_path/'missing'),'--authorize-rl'])
    with pytest.raises(SystemExit) as error:cli.main()
    assert error.value.code==2 and not (tmp_path/'missing').exists()
    from gym_trading_env.research import fg_s3_e_quality_online as online
    calls=[]
    monkeypatch.setattr(online,'rl_review_packet',lambda *args:calls.append(args) or dict(status='AWAIT_OPERATOR_REVIEW'))
    monkeypatch.setattr(sys,'argv',['cli','online-rl-review-packet','--run-dir',str(tmp_path/'run'),
        '--output-dir',str(tmp_path/'packet')])
    cli.main();assert calls==[(str(tmp_path/'run'),str(tmp_path/'packet'))]
    assert not (tmp_path/'run').exists() and not (tmp_path/'packet').exists()


def test_live_rl_scope_rejects_wm_and_source_replacement_before_any_update():
    from types import SimpleNamespace
    from embodied.run.fg_s3_e_quality_online_run import OnlineRun
    from gym_trading_env.envs.entry_contract import fingerprint
    run=OnlineRun.__new__(OnlineRun)
    digests={key:'a'*64 for key in ('enc/x','dyn/x','dec/x','rew/x','con/x','avail/x')}
    wm=fingerprint(digests);bindings=[dict(corpus_identity='FIXED_UNIT_SCOPE_ONLY')]
    run.identity=dict(frozen_wm_identity=wm,source_bindings=copy.deepcopy(bindings),history='FIXED_HISTORY')
    run.source_bindings=copy.deepcopy(bindings)
    run.source=SimpleNamespace(history=SimpleNamespace(identity_hash='FIXED_HISTORY'))
    run.rl_lifecycle=SimpleNamespace(state=dict(stage='RL_ACTIVE'))
    run.spec=dict(env_steps=64,max_consumed_batches=32);run.settings=dict(optimizer_updates=16)
    run.step=48;run.agent=SimpleNamespace(n_updates=4)
    run.counters=SimpleNamespace(state=dict(consumed_batches=4))
    run.native_preparation=object();run.authorized=True;run.rl_review=object()
    calls=[]
    run.gates=SimpleNamespace(wm_identity=wm,formal_admitted=False,
        require_rl=lambda *a,**kw:calls.append('UNIT_PERMISSION_CALLBACK_ONLY'))
    run._require_live_rl(digests);assert len(calls)==1
    for change in ('wm','sources','history'):
        bad=dict(digests)
        if change=='wm':bad['enc/x']='b'*64
        elif change=='sources':run.source_bindings=[dict(corpus_identity='REPLACED')]
        else:run.source.history.identity_hash='REPLACED'
        with pytest.raises(ValueError,match='IDENTITY_OR_REVIEW_SCOPE_CHANGED'):run._require_live_rl(bad)
        run.source_bindings=copy.deepcopy(bindings);run.source.history.identity_hash='FIXED_HISTORY'
    assert len(calls)==1  # No fake authority callback or optimizer on rejected paths.


def test_initial_run_freeze_uses_exact_existing_optimizer_slot_owners():
    from types import SimpleNamespace
    import numpy as np
    from dreamerv3.fg_s3_e_quality_course_contract import ALL_GROUPS
    from embodied.run.fg_s3_e_quality_online_rl import validate_frozen_online_state
    initial={prefix+'value':np.array(0.,np.float32) for prefix in ALL_GROUPS}
    initial.update({k:np.array(0,np.int32) for k in (
        'opt/step/value','opt/state/1/0','opt/state/2/0','opt/state/3/count')})
    for prefix in ('enc','qualityactor','val'):
        initial['opt/state/1/1/'+prefix+'/value']=np.array(0.,np.float32)
    config=SimpleNamespace(opt=SimpleNamespace(wd=False))
    final=copy.deepcopy(initial);final['quality_actor/value'][...]=1
    final['opt/state/1/1/qualityactor/value'][...]=2
    validate_frozen_online_state(initial,final,config,('quality_actor/',),updated=True)
    for key in ('enc/value','opt/state/1/1/enc/value','val/value'):
        bad=copy.deepcopy(final);bad[key][...]=3
        with pytest.raises(ValueError,match='FROZEN_INITIAL_PARAMETER_OR_SLOT'):
            validate_frozen_online_state(initial,bad,config,('quality_actor/',),updated=True)
    bad=copy.deepcopy(final);bad['opt/state/alien/count']=np.array(1,np.int32)
    with pytest.raises(ValueError,match='FULL_STATE_SCHEMA'):
        validate_frozen_online_state(initial,bad,config,('quality_actor/',),updated=True)


def test_finalization_requires_published_non_tail_sidecars():
    from types import SimpleNamespace
    from embodied.run.fg_s3_e_quality_online_run import OnlineRun
    run=OnlineRun.__new__(OnlineRun)
    run.collector=SimpleNamespace(rows=[dict(stepid='first'),dict(stepid='second'),dict(stepid='pending_tail')])
    run.publisher=SimpleNamespace(strict=object(),blockers=[],ledger={
        'first':dict(status='CERTIFIED',publication_seq=1),
        'second':dict(status='PENDING',publication_seq=None),
        'pending_tail':dict(status='PENDING',publication_seq=None)},validate_snapshot=lambda:None)
    with pytest.raises(ValueError,match='SIDECAR_NOT_PUBLISHED'):run._validate_final_publication()
    run.publisher.ledger['second']=dict(status='INVALID',publication_seq=2)
    run._validate_final_publication()
    run.publisher.strict=None
    with pytest.raises(ValueError,match='SUPERVISION_UNAVAILABLE'):run._validate_final_publication()

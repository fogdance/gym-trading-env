"""Observation math and append boundaries; no neural computation."""
import copy
import json

import pytest

from gym_trading_env.research.fg_replay import signed, write_json
from gym_trading_env.research.fg_s3_v2_observation import JsonTail, TrainMetrics, actor_only_metrics


def row(y,p,*,valid=True,guided=False):
    return dict(execution_result=dict(advanced=True,fills=[dict(reason='ENTRY')]),
        action_mask=[1,1],oracle_reason='CERTIFIED_SUFFIX_DOMINANCE' if valid else 'FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE',
        actor_proposal=p,oracle_selected=guided,executed_action=1,teacher_valid=valid,
        oracle_valid=valid,oracle_certified=valid,teacher_action=y,oracle_action=y)


def test_train_proposals_are_separate_from_takeover_and_partial_rows(tmp_path):
    metrics=TrainMetrics()
    for r in (row(1,1),row(1,0,guided=True),row(0,1),row(1,1,valid=False)):
        metrics.observe(r)
    result=metrics.result()
    assert result['actor_proposal_oracle_add_recall']==.5
    assert result['actor_proposal_oracle_add_precision']==.5
    assert result['label_ratio']==.75 and result['unproven_funds']==1
    assert result['actor_entries']==3 and result['oracle_guided_entries']==1
    bad=row(1,1);bad['oracle_certified']=False
    with pytest.raises(ValueError,match='uncertified'):
        metrics.observe(bad)
    path=tmp_path/'live.jsonl';tail=JsonTail(path)
    path.write_bytes(b'{"step":1}\n{"step":')
    assert list(tail.rows())==[dict(step=1)]
    with path.open('ab') as stream:stream.write(b'2}\n')
    assert list(tail.rows())==[dict(step=2)]
    path.write_bytes(b'')
    assert tail.rewound()


def test_actor_only_rates_include_censored_entries_and_reject_guidance(tmp_path):
    state=dict(attributions=[dict(reason='TAKE_PROFIT'),dict(reason='STOP_LOSS')])
    (tmp_path/'account_state.json').write_text(json.dumps(state))
    write_json(tmp_path/'dev_manifest.json',signed(dict(frozen_inputs={},rows=[dict(
        contract='CZCE.FG409',sha256='market',env_sha256='rules',contract_sha256='terms')])))
    actions=[dict(cursor=i,action=int(i==0),mask=[1,1],is_last=False,p_add=p,advanced=i>0)
             for i,p in enumerate((.9,.2,.99))]
    (tmp_path/'actions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in actions))
    receipt=dict(teacher_guidance=False,beta=0.,fold='DEV',output=str(tmp_path),
        checkpoint='step',checkpoint_manifest_sha256='checkpoint',env_steps=2,policy_mode='eval_argmax',
        add_executed=1,entries=3,right_censored=1,sl30=1,early_mae10=1,
        clock_protocol='S3_MARKET_MINUTE',tp_market_minutes_p50=2.,tp_market_minutes_p90=2.,reward=0.,all_wait=False)
    result=actor_only_metrics(receipt)
    assert result['tp5_points_rate']==pytest.approx(1/3)
    assert result['sl30_rate']==pytest.approx(1/3)
    assert result['highest_add_scores'][0]['p_add']==.9  # final .99 never executed
    bad=copy.deepcopy(receipt);bad['teacher_guidance']=True
    with pytest.raises(ValueError,match='Actor-only'):
        actor_only_metrics(bad)


def test_observer_selects_frozen_tools_entry_instead_of_same_named_module(tmp_path):
    from gym_trading_env.research.fg_replay import digest
    from gym_trading_env.research.fg_s3_v2_observation import Observer
    tools=tmp_path/'tools';tools.mkdir()
    src=tmp_path/'src';src.mkdir()
    paths=[d/'fg_s3_v2_experiment.py' for d in (tools,src)]
    for p in paths:p.write_text('# entry fixture\n')
    prepared=tmp_path/'prepared';prepared.mkdir()
    write_json(prepared/'manifest.json',signed(dict(files={})))
    write_json(prepared/'settings.json',signed(dict(settings={})))
    run=tmp_path/'run';run.mkdir()
    write_json(run/'run_identity.json',signed(dict(identity=dict(
        preparation_sha256=digest(prepared/'manifest.json'),code={str(p):digest(p) for p in paths}))))
    assert Observer(prepared,run).cli==paths[0]

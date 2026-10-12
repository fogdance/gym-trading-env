import copy

import pytest

from gym_trading_env.research.fg_single3_gate_calibration import check_config_delta, resume


def test_gate_migration_allows_only_threshold_and_new_run_location():
    source = dict(logdir='/tmp/old', agent=dict(avail_threshold=.5,horizon=333),
        reward='C1',replay={'online':True})
    target=copy.deepcopy(source)
    target['logdir']='/tmp/new'
    target['agent']['avail_threshold']=.3
    check_config_delta(source,target)
    for key in ('reward','replay'):
        bad=copy.deepcopy(target);bad[key]='changed'
        with pytest.raises(ValueError,match='other configuration'):
            check_config_delta(source,bad)
    bad=copy.deepcopy(target);bad['agent']['horizon']=200
    with pytest.raises(ValueError,match='other configuration'):
        check_config_delta(source,bad)


def test_gate_resume_without_authorization_fails_before_any_read_or_launch(tmp_path):
    with pytest.raises(PermissionError,match='not authorized'):
        resume(tmp_path/'nonexistent')
    assert not list(tmp_path.iterdir())


def test_gate_authorization_does_not_survive_launch_failure(monkeypatch):
    import os
    from gym_trading_env.research.fg_single3_gate_calibration import authorized_identity
    monkeypatch.delenv('FG_SINGLE3_GATE_AUTHORIZED_IDENTITY',raising=False)
    with pytest.raises(RuntimeError,match='failed'):
        with authorized_identity({'branch':'new'}):
            assert os.environ['FG_SINGLE3_GATE_AUTHORIZED_IDENTITY']
            raise RuntimeError('failed')
    assert 'FG_SINGLE3_GATE_AUTHORIZED_IDENTITY' not in os.environ

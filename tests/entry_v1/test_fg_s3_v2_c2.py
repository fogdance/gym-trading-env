"""Minimal real CPU C2 learning, Gate transition and FullCheckpoint contracts."""
import copy

import pytest

from gym_trading_env.research.fg_s3_v2_experiment import read_settings, require_device


def test_cpu_backend_is_only_for_isolated_c2_diagnostics():
    cfg = read_settings('C2_CPU_FG201_300')
    assert require_device(cfg) == 'cpu'
    for key, value in (('mode', 'a2'), ('scope', 'formal'), ('device', 'unknown')):
        wrong = copy.deepcopy(cfg); wrong[key] = value
        with pytest.raises(ValueError, match='CPU is limited'):
            require_device(wrong)
    assert require_device(dict(mode='c2', scope='formal', device='gpu')) == 'gpu'


def test_real_cpu_c2_gradients_gate_switch_and_full_restore(tmp_path):
    from gym_trading_env.research.fg_s3_v2_c2_check import check_c2_model
    result = check_c2_model(tmp_path/'c2', device='cpu')
    assert result['status'] == 'PASS_REAL_CPU_C2_CONTRACTS'
    assert result['actor_gradients']['bc_actor_grad'] > 0
    assert result['actor_gradients']['rl_actor_grad'] > 0
    assert result['actor_gradients']['bc_non_actor_grad'] == 0
    assert result['counters']['rl_actor_updates'] == result['counters']['rl_critic_updates'] == 3
    assert result['counters']['all_updates'] == 8
    assert result['counters']['sampled_transitions'] == 512
    assert result['production_market_gate_ready'] == result['formal_1m'] == 'NOT_RUN'

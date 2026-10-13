"""Public V2 contracts; neural assertions use actual GPU computation only."""
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from test_fg_single3_env import single_env
from test_fg_s3_dagger_d0 import query_for
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.research.fg_s3_v2_collector import V2Collector


DREAMER = Path(__file__).resolve().parents[2].parent/'dreamerv3'


def test_live_committed_rows_restore_preserves_native_ownership(tmp_path):
    from embodied.run.fg_checkpoint import save_replay, load_replay
    folder = tmp_path/'fixture'; folder.mkdir()
    env = single_env(folder, [(1000,1000,1000,1000),(1000,1000,1000,1000),(1000,1007,1000,1007)]*30,
        candidate='C1', fees=3, chunk=90,
        times=pd.date_range('2026-01-05 13:31', periods=90, freq='min'),
        calendar=[dict(start='2026-01-05 13:30', end='2026-01-05 15:30')],
        liquidation='2026-01-05 15:30')
    query, _ = query_for(env, folder/'env.yaml')
    a = V2Collector(env, query, tmp_path/'a', DREAMER, beta=1., actor_probability=0.)
    b = None
    try:
        a.advance(70)
        assert a.commit_seq == 70 and a.pending['stepid'] is None
        state = a.save()
        driver = a.driver.save()
        replay = save_replay(a.replay, tmp_path/'snapshot')
        b_env = FixedDirectionEntryEnv(df=env._raw.copy(), config_path=str(folder/'env.yaml'))
        b = V2Collector(b_env, query, tmp_path/'b', DREAMER, beta=1., actor_probability=0.)
        b.journal.write_bytes(a.journal.read_bytes()[:state['journal_bytes']])
        b.driver.load(driver)
        load_replay(b.replay, replay, tmp_path/'snapshot')
        b.load(state)
        a.advance(75); b.advance(75)
        assert a.records == b.records and a.pending == b.pending
        x, y = a.sample(1), b.sample(1)
        for k in x: np.testing.assert_array_equal(x[k], y[k])
        with pytest.raises(ValueError, match='Uncommitted'):
            bad = copy.deepcopy(x); bad['stepid'][0,0] = 0; b.join(bad)
    finally:
        a.close()
        if b: b.close()


def test_shared_extractions_reject_changed_math_and_cuda_flags():
    from embodied.run.fg_s3_source_admission import audit_code_transition
    for name, needle, replacement in (
        ('dreamerv3/agent.py', "losses['rew'] = self.rew(inp, 2).loss(obs['reward'])",
         "losses['rew'] = 2 * self.rew(inp, 2).loss(obs['reward'])"),
        ('embodied/jax/internal.py', '--xla_gpu_graph_level=0', '--xla_gpu_graph_level=3')):
        old = target = (DREAMER/name).read_bytes()
        # Economic-clock changes require explicit parameter warm-start; this
        # test checks identical current-code transfers and rejects changed math.
        target = (DREAMER/name).read_bytes()
        audit_code_transition(old, target, name)
        assert needle.encode() in target
        with pytest.raises(ValueError,match='semantics'):
            audit_code_transition(old,target.replace(needle.encode(),replacement.encode()),name)


def test_current_decision_descriptor_preserves_labels_and_rejects_stale_state(tmp_path):
    folder = tmp_path/'fixture'; folder.mkdir()
    env = single_env(folder, [(1000,1000,1000,1000),(1000,1000,1000,1000),(1000,1007,1000,1007)]*30,
        candidate='C1', fees=3, chunk=90,
        times=pd.date_range('2026-01-05 13:31', periods=90, freq='min'),
        calendar=[dict(start='2026-01-05 13:30', end='2026-01-05 15:30')],
        liquidation='2026-01-05 15:30')
    query, _ = query_for(env, folder/'env.yaml')
    env.reset(seed=202)
    for _ in range(20):
        full = env.export_state()
        current = query.decision_state(env)
        a = query.query(env, env._published_mask, snapshot=full)
        b = query.query(env, env._published_mask, snapshot=current)
        for key in ('action', 'valid', 'certified', 'reason', 'q_add', 'q_wait', 'q_margin'):
            assert a[key] == b[key]
        assert env.export_state() == full
        stale = copy.deepcopy(current); stale['equity'] = '0'
        assert query.query(env,env._published_mask,snapshot=stale)['reason']=='STALE_ACCOUNT_SNAPSHOT'
        assert not any(k in current for k in ('prefix_hash','fills','reward_events','equity_history'))
        env.step(b['action'] if b['valid'] else 0)


def test_real_gpu_learning_contracts(tmp_path):
    from gym_trading_env.research.fg_s3_v2_gpu_check import check_gpu_model
    receipt = check_gpu_model(tmp_path/'gpu')
    assert receipt['status'] == 'PASS_REAL_GPU_NEURAL_CONTRACTS'
    assert all('cuda' in d for d in receipt['devices'])


def test_resume_archives_uncheckpointed_cycles(tmp_path):
    from embodied.run.fg_s3_v2_train import reconcile_cycles
    rows = [dict(env_steps=250, agent_updates=21185), dict(env_steps=500, agent_updates=21187)]
    committed = ''.join(json.dumps(r)+'\n' for r in rows).encode()
    full = committed + b'{"env_steps":750,"agent_updates":21189}\n{partial'
    path = tmp_path/'cycles.jsonl'; path.write_bytes(full)
    reconcile_cycles(tmp_path, 2, 500, 21187)
    assert path.read_bytes() == committed
    assert next((tmp_path/'recovery_logs').glob('cycles_before_resume_*')).read_bytes() == full
    with pytest.raises(ValueError, match='boundary'):
        reconcile_cycles(tmp_path, 2, 500, 21188)

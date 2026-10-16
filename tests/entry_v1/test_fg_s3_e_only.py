"""E encoder with original gated Dreamer RL and no teaching objectives."""
import copy
import pickle
from pathlib import Path

import numpy as np
import pytest

from dreamerv3.configuration import read_config_yaml
from dreamerv3.fg_s3_composition import normalize_config, resolve_composition
from gym_trading_env.research.fg_s3_v2_experiment import _validate_settings

ROOT = Path(__file__).resolve().parents[2]
DREAMER = ROOT.parent/'dreamerv3'


def test_e_only_settings_disable_teaching_and_preserve_formal_authorization(tmp_path):
    from gym_trading_env.research.fg_s3_v2_experiment import require_training_authorization
    raw = read_config_yaml(ROOT/'configs/experiments/fg_s3_v2/E_ONLY_MARKET_MINUTE_FORMAL_1M.yaml')
    path = tmp_path/'e_only.yaml'; path.write_text('configuration contract fixture')
    settings = _validate_settings(normalize_config(raw), path)
    comp = settings['resolved_composition']
    assert settings['mode'] == 'e' and comp['use_e']
    assert not any(comp[k] for k in ('base_oracle_bc', 'use_tutor', 'use_b2'))
    assert settings['env_steps'] == 1000000 and settings['contracts'] == 'ALL_TRAIN'
    assert settings['source_kind'] == 'direct_init'
    assert settings['model_init'] == dict(seed=0, presets=['size50m'])
    with pytest.raises(PermissionError, match='authorize-training'):
        require_training_authorization(settings, False)
    require_training_authorization(settings, True)
    for key, value in (('beta', .1), ('bc_lambda', 1.),
                       ('beta_schedule', [[0, 0.], [1000, 1.]]),
                       ('b2_event_lambda', 0.), ('tutor_pair_lambda', 0.),
                       ('quality_stage', 'HEADS_ONLY')):
        with pytest.raises(ValueError, match='E-only'):
            _validate_settings(normalize_config({**copy.deepcopy(raw), key: value}), path)


def test_teacher_disabled_collector_has_no_queries_takeover_or_labels(tmp_path, monkeypatch):
    from e_model_fixture import controlled_env
    from test_fg_s3_dagger_d0 import query_for
    from gym_trading_env.research.fg_s3_v2_collector import V2Collector
    env = controlled_env(tmp_path/'market')
    query, _ = query_for(env, tmp_path/'market/env.yaml')
    monkeypatch.setattr(query, 'query', lambda *a, **k: pytest.fail('Disabled teacher was queried'))
    collector = V2Collector(env, query, tmp_path/'collector', DREAMER,
                            beta=0., actor_probability=1., teacher_enabled=False)
    try:
        collector.advance(75)
        batch = collector.sample(1)
        assert not batch['teacher_valid'].any() and not batch['teacher_action'].any()
        assert collector.counts['executed_ADD'] > 0
        assert collector.counts['certified_labels'] == 0 and collector.query_seconds == []
        assert all(not r['oracle_selected'] and not r['teacher_valid']
                   and r['oracle_reason'] == 'TEACHING_DISABLED' for r in collector.records)
        assert collector.identity['teacher_enabled'] is False
        with pytest.raises(ValueError, match='disabled teacher'):
            V2Collector(env, query, tmp_path/'invalid', DREAMER, beta=1., teacher_enabled=False)
    finally:
        collector.close()


def test_e_only_formal_source_still_requires_authenticated_real_data(tmp_path):
    from e_model_fixture import SyntheticEProvider
    from dreamerv3.fg_s3_e_config import e_target_config
    from embodied.run.fg_s3_e_source import make_binding
    from gym_trading_env.research.fg_s3_v2_gpu_check import small_config
    composition = resolve_composition({'composition': dict(encoder='e_multiscale_1m15m', teaching='none')})
    provider = SyntheticEProvider(tmp_path)
    with pytest.raises(ValueError, match='authenticated real TRAIN data'):
        make_binding(e_target_config(small_config()), composition, provider.manifest, diagnostic=False)


def test_e_only_closed_and_ready_updates_teacher_independence_and_exact_restore(tmp_path):
    import embodied
    from e_model_fixture import controlled_env, SyntheticEProvider
    from test_fg_s3_dagger_d0 import query_for
    from dreamerv3.fg_s3_e_config import e_target_config
    from dreamerv3.fg_s3_e_diagnostics import EAcceptanceModel
    from dreamerv3.fg_s3a_actor import parameter_digests
    from embodied.run.fg_s3_source_transfer import make_agent
    from embodied.run.fg_checkpoint import host
    from gym_trading_env.research.fg_s3_v2_gpu_check import small_config, gpu_batch
    from gym_trading_env.research.fg_s3_v2_collector import V2Collector
    from gym_trading_env.research.fg_s3_e_observation import OBS_SCHEMA
    from gym_trading_env.research.fg_replay import signed, write_json

    env = controlled_env(tmp_path/'market')
    query, _ = query_for(env, tmp_path/'market/env.yaml')
    provider = SyntheticEProvider(tmp_path)
    collector = V2Collector(env, query, tmp_path/'collector', DREAMER,
        beta=0., teacher_enabled=False, observation_schema=OBS_SCHEMA, e_provider=provider)
    try:
        collector.advance(75)
        raw = next(iter(embodied.streams.Consec(embodied.streams.Stateless(lambda: collector.sample(1)),
            length=64, consec=1, prefix=1, strict=True)))
        cfg = e_target_config(small_config())
        composition = resolve_composition({'composition': dict(encoder='e_multiscale_1m15m', teaching='none')})
        def model(name):
            return make_agent(cfg, tmp_path/name, device='cpu', agent_cls=EAcceptanceModel,
                bc_lambda=0., extra_agent_config={'s3_v2_mode': 'e'}, composition=composition)
        agent = model('agent')
        assert not any(k.startswith(('b2_', 'quality_')) for k in agent.params)
        before = parameter_digests(agent.params)
        carry = agent.init_train(1)
        for seed in (1300, 1301):
            carry, _, closed = agent.train(carry, gpu_batch(agent, raw, seed))
        after_closed = parameter_digests(agent.params)
        assert not bool(closed['v2/rl_active']) and not bool(closed['bc/active'])
        assert any(before[k] != after_closed[k] for k in before if k.startswith('enc/'))
        assert all(before[k] == after_closed[k] for k in before if k.startswith(('pol/', 'val/')))
        # A controlled READY fixture proves RL wiring, not production Gate admission.
        agent.set_avail_actor_enabled(True)
        state = pickle.loads(pickle.dumps(dict(agent=agent.save_exact(), carry=host(carry))))
        carry, _, ready = agent.train(carry, gpu_batch(agent, raw, 1302))
        after_ready = parameter_digests(agent.params)
        assert bool(ready['v2/rl_active']) and not bool(ready['bc/active'])
        gradients = {name: float(ready['opt/'+name+'_grad_norm']) for name in
                     ('actor', 'critic', 'e_minute', 'e_context15', 'e_fusion')}
        assert all(np.isfinite(x) and x > 0 for x in gradients.values())
        changes = {prefix: sum(after_closed[k] != after_ready[k] for k in before if k.startswith(prefix))
                   for prefix in ('enc/minute/', 'enc/context15/', 'enc/fusion/', 'pol/', 'val/', 'slowval/')}
        assert all(changes.values())
        restored = model('restored')
        restored.load_exact(state['agent'])
        _, _, restored_metrics = restored.train(restored.restore_carry(state['carry']),
                                                gpu_batch(restored, raw, 1302))
        assert parameter_digests(restored.params) == after_ready
        for key in ready:
            np.testing.assert_array_equal(ready[key], restored_metrics[key])
        restored.load_exact(state['agent'])
        poisoned = copy.deepcopy(raw)
        poisoned['teacher_valid'][:] = True
        poisoned['teacher_action'][:] = 1-poisoned['teacher_action']
        restored.train(restored.restore_carry(state['carry']), gpu_batch(restored, poisoned, 1302))
        assert parameter_digests(restored.params) == after_ready
        write_json(tmp_path/'e_only_model_receipt.json', signed(dict(
            status='PASS_CPU_CONTROLLED_MODEL', composition=composition,
            closed_actor_critic_frozen=True, ready_rl_updates=True, bc_active=False,
            gradients=gradients, changed_parameters=changes, teacher_independence=True,
            exact_model_optimizer_rng_restore=True, production_gate='NOT_ESTABLISHED', gpu='NOT_RUN')))
    finally:
        collector.close()

"""Controlled C2 neural/lifecycle checks; never a production Gate certificate."""
import copy
from functools import partial
from pathlib import Path
import time
import uuid

import elements
import embodied
import jax
import ninjax as nj
import numpy as np
import pandas as pd

from embodied.run.action_mask_warmup import ActionMaskWarmup
from embodied.run.fg_checkpoint import FullCheckpoint, RuntimeState, StepSchedule, file_hash, validate_bundle
from embodied.run.fg_s3_gate import SINGLE3, gate_policy, make_gate_controller
from embodied.run.fg_s3_source_transfer import make_agent
from embodied.run.fg_s3_v2_train import Counters
from dreamerv3.fg_s3_v2_agent import S3V2Agent
from dreamerv3.fg_s3a_actor import parameter_digests
from .fg_replay import signed, write_json
from .fg_s3_v2_collector import V2Collector
from .fg_s3_v2_gpu_check import PROJECT, DREAMER, small_config, model_batch


def rl_state(key):
    return key.startswith(('val/', 'slowval/', 'slowval_count/', 'retnorm/', 'valnorm/', 'advnorm/')) or (
        key.startswith('opt/state/') and '/val/' in key)


def actor_state(key):
    return key.startswith('pol/') or (key.startswith('opt/state/') and '/pol/' in key)


def norm(tree):
    return float(np.sqrt(sum(np.square(np.asarray(v, dtype=np.float64)).sum() for v in tree.values())))


def component_gradients(agent, carry, raw):
    """Same parameters, replay context and RNG for all three loss derivatives."""
    data = model_batch(agent, raw, 123)
    data.pop('seed')
    gradients, losses = {}, {}
    for kind in ('bc', 'rl_actor', 'joint'):
        def loss(c, d):
            c, obs, prevact, ids = agent.model._apply_replay_context(c, d)
            total, (_, _, _, mets) = agent.model.training_loss_fn(d, ids)(
                c, obs, prevact, training=True)
            return dict(bc=mets['bc/nll'] * agent.config.bc_lambda,
                        rl_actor=mets['v2/rl_actor_loss'], joint=total)[kind]
        gradient = jax.jit(nj.pure(lambda c, d: nj.grad(loss, agent.model.modules)(c, d)),
                           static_argnames=('create', 'modify'))
        _, (value, _, grads) = gradient(agent.params, carry, data, seed=123, create=False, modify=True)
        losses[kind] = float(value)
        gradients[kind] = {k: np.asarray(v) for k, v in grads.items()}
    bc, rl, joint = (gradients[k] for k in ('bc', 'rl_actor', 'joint'))
    assert all(not np.count_nonzero(v) for k, v in bc.items() if not k.startswith('pol/'))
    pol = [k for k in joint if k.startswith('pol/')]
    assert norm({k: bc[k] for k in pol}) > 0
    assert norm({k: rl[k] for k in pol}) > 0
    error = max(float(np.max(np.abs(joint[k] - bc[k] - rl[k]))) for k in pol)
    for k in pol:
        np.testing.assert_allclose(joint[k], bc[k] + rl[k], rtol=2e-5, atol=2e-6)
    return dict(losses=losses, bc_actor_grad=norm({k: bc[k] for k in pol}),
                rl_actor_grad=norm({k: rl[k] for k in pol}),
                joint_actor_grad=norm({k: joint[k] for k in pol}), max_additivity_error=error,
                bc_non_actor_grad=norm({k: v for k, v in bc.items() if k not in pol}))


class FixtureRun:
    """Existing public components around a small original-network fixture."""
    def __init__(self, output, cfg, env_factory, query, device, *, resume=False):
        self.root, self.cfg, self.device = output, cfg, device
        self.agent = a = make_agent(cfg, output/'model', agent_cls=S3V2Agent,
                                   device=device, bc_lambda=1., extra_agent_config={'s3_v2_mode': 'c2'})
        assert all(d.platform == device for d in a.train_devices + a.policy_devices)
        if not resume:
            for k, v in list(a.params.items()):
                if k.startswith('opt/state/'):
                    value = 1000 if np.issubdtype(v.dtype, np.integer) else .125
                    a.params[k] = jax.device_put(np.full(v.shape, value, v.dtype), v.sharding)
            a.sync_policy_from_train()
        self.collector = V2Collector(env_factory(), query, output/'rollout', DREAMER,
            beta=1., actor=lambda c, o: a.policy(c, o, mode='train'), agent=a,
            existing=resume, source_checkpoint_identity='controlled_small_original_network')
        source = embodied.streams.Consec(embodied.streams.Stateless(partial(self.collector.sample, 1)),
                                        length=64, consec=1, prefix=1, strict=True)
        self.stream = source
        self.carry = {'train': a.init_train(1)}
        self.step = elements.Counter()
        self.counters = Counters()
        policy = gate_policy(cfg, diagnostic=True, protocol=SINGLE3)
        self.gate = make_gate_controller(cfg, policy, dict(gate=None), False)
        self.runtime = RuntimeState(StepSchedule(100, 100), elements.when.Ratio(1), [source],
            lambda: self.carry, self.restore_carry, {'gate': elements.Agg()}, [0])
        self.identity = dict(protocol='FG_S3_V2_C2_CONTROLLED_CHECK_V1', diagnostic=True,
                             controlled_gate=True, device=device, source='small_original_network_fixture')
        self.cp = FullCheckpoint(output/'ckpt', self.step, self.identity, [source],
                                 reserve_gib=.1, keep_all=True)
        for key, value in dict(agent=a, replay=self.collector.replay, driver=self.collector.driver,
            collector=self.collector, step=self.step, runtime=self.runtime,
            action_mask_warmup=self.gate, v2_counters=self.counters).items():
            setattr(self.cp, key, value)
        if resume:
            self.cp.load()
            assert a.get_avail_actor_enabled() == self.gate.actor_enabled
            assert int(a.n_updates) == self.counters.state['all_updates']
            assert int(self.step) == self.collector.counts['env_steps']

    def restore_carry(self, data):
        self.carry = {k: self.agent.restore_carry(v) for k, v in data.items()}

    def collect(self, steps):
        self.agent.sync_policy_from_train()
        self.collector.advance(steps)
        self.step.increment(steps - int(self.step))

    def update(self, phase):
        self.runtime.start()
        raw = next(self.stream)
        assert raw['teacher_valid'][:, 1:].any()
        before = parameter_digests(self.agent.params)
        old_step = int(self.agent.params['opt/step/value'])
        old_slow = int(self.agent.params['slowval_count/value'])
        self.carry['train'], _, mets = self.agent.train(self.carry['train'],
            model_batch(self.agent, raw, self.counters.state['all_updates'] + 7))
        after = parameter_digests(self.agent.params)
        ready = self.gate.actor_enabled
        changed = [k for k in before if before[k] != after[k]]
        assert int(self.agent.params['opt/step/value']) == old_step + 1
        assert np.isfinite(float(mets['opt/loss'])) and bool(mets['opt/finite'])
        assert bool(mets['v2/rl_active']) == ready
        assert float(mets['opt/wm_grad_norm']) > 0 and float(mets['opt/actor_grad_norm']) > 0
        assert any(k.startswith(('enc/', 'dyn/')) for k in changed)
        assert any(k.startswith('pol/') for k in changed)
        if ready:
            assert float(mets['opt/critic_grad_norm']) > 0
            for prefix in ('val/', 'slowval/', 'retnorm/', 'valnorm/', 'advnorm/'):
                assert any(k.startswith(prefix) for k in changed), prefix
            assert int(self.agent.params['slowval_count/value']) == old_slow + 1
            assert any(k.startswith('opt/state/') and '/val/' in k for k in changed)
        else:
            assert float(mets['opt/critic_grad_norm']) == 0
            assert float(mets['v2/rl_actor_loss']) == float(mets['v2/rl_critic_loss']) == 0
            assert all(not rl_state(k) for k in changed)
        s = self.counters.state
        s['all_updates'] += 1
        s['wm_updates'] += 1
        s['bc_valid_updates'] += 1
        s['bc_add_updates'] += int((raw['teacher_valid'][:, 1:] & (raw['teacher_action'][:, 1:] == 1)).any())
        s['sampled_transitions'] += int(raw['teacher_valid'][:, 1:].size)
        s['rl_actor_updates'] += int(ready)
        s['rl_critic_updates'] += int(ready)
        s['gate_closed_updates'] += int(not ready)
        row = dict(phase=phase, update=s['all_updates'], optimizer_step=int(self.agent.params['opt/step/value']),
            slowcritic_count=int(self.agent.params['slowval_count/value']), changed_keys=changed,
            metrics={k: float(np.asarray(v)) for k, v in mets.items()
                     if np.asarray(v).size == 1 and np.isfinite(v).all()})
        with (self.root/'updates.jsonl').open('a') as f:
            import json
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(f"C2 check {phase} update={s['all_updates']} actor_grad={mets['opt/actor_grad_norm']:.6f} "
              f"critic_grad={mets['opt/critic_grad_norm']:.6f}", flush=True)
        self.agent.sync_policy_from_train()
        return row, raw

    def close(self):
        self.collector.close()


def check_c2_model(output=None, *, device='cpu'):
    if device not in ('cpu', 'gpu'):
        raise ValueError('Unknown diagnostic device')
    output = Path(output) if output else PROJECT/'artifacts/fg_s3_v2/checks'/('c2_'+device+'_'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    import sys
    sys.path.insert(0, str(PROJECT/'tests/entry_v1'))
    from test_fg_single3_env import single_env
    from test_fg_s3_dagger_d0 import query_for
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    folder = output/'fixture'; folder.mkdir()
    prices = [(1000,1000,1000,1000),(1000,1000,1000,1000),(1000,1007,1000,1007)] * 30
    original = single_env(folder, prices, candidate='C1', fees=3, chunk=90,
        times=pd.date_range('2026-01-05 13:31', periods=90, freq='min'),
        calendar=[dict(start='2026-01-05 13:30', end='2026-01-05 15:30')],
        liquidation='2026-01-05 15:30')
    query, _ = query_for(original, folder/'env.yaml')
    factory = lambda: FixedDirectionEntryEnv(df=original._raw.copy(), config_path=str(folder/'env.yaml'))
    cfg = small_config()
    cfg['jax']['platform'] = 'cpu' if device == 'cpu' else 'cuda'
    cfg['action_mask_warmup'].update(enabled=True, add_recall_enabled=True)
    cfg['agent']['valnorm']['impl'] = 'meanstd'
    cfg['agent']['advnorm']['impl'] = 'meanstd'
    # Stateful normalizers exercise update boundaries in this fixture only;
    # production valnorm/advnorm remain `none`, without mutable parameters.
    run = None
    try:
        run = FixtureRun(output, cfg, factory, query, device)
        run.collect(75)
        rows = []
        for _ in range(2):
            row, raw = run.update('CLOSED'); rows.append(row)
        assert not run.gate.ready(24999) and run.gate.ready(25000)
        passing = {k: 1. for k in ('avail/post_accuracy', 'avail/post_exact_accuracy',
            'avail/prior_accuracy', 'avail/prior_exact_accuracy', 'avail/post_add_recall', 'avail/prior_add_recall')}
        passing.update({'avail/prior_fpr': 0., 'avail/img_fallback_rate': 0.,
                        'avail/post_add_positives': 16, 'avail/prior_add_positives': 16})
        for i in range(19):
            assert not run.gate.update(passing, i)
            assert not run.gate.actor_enabled
        assert run.gate.update(passing, 19)
        run.agent.set_avail_actor_enabled(run.gate.actor_enabled)
        gradients = component_gradients(run.agent, run.carry['train'], raw)
        write_json(output/'actor_gradient_components.json', signed(gradients))
        for _ in range(2):
            row, _ = run.update('READY'); rows.append(row)
        run.collect(80)
        run.cp.save()
        checkpoint = run.cp.latest()
        validate_bundle(checkpoint, run.identity)
        saved = parameter_digests(run.agent.params)
        saved_teachers = copy.deepcopy(run.collector.records)
        saved_gate = run.gate.save()
        run.close()
        run = FixtureRun(output, cfg, factory, query, device, resume=True)
        assert saved == parameter_digests(run.agent.params)
        assert saved_teachers == run.collector.records and saved_gate == run.gate.save()
        assert run.agent.get_avail_actor_enabled()
        row, _ = run.update('READY_RESUMED'); rows.append(row)
        regression = dict(passing, **{'avail/prior_accuracy': .98})
        for i in range(4):
            assert not run.gate.update(regression, i + 20)
        try:
            run.gate.update(regression, 24)
            raise AssertionError('Production regression stop was bypassed')
        except RuntimeError as error:
            assert '5 consecutive reports' in str(error)
        # Explicit fixture-only reclosure. Production regression still stops;
        # no start/resume setting can enable this diagnostic override.
        closed = ActionMaskWarmup.for_single3_1m(elements.Config(cfg['action_mask_warmup']), False)
        run.gate.load(closed.save())
        run.agent.set_avail_actor_enabled(False)
        for _ in range(2):
            row, _ = run.update('CLOSED_AGAIN'); rows.append(row)
        stateless = small_config()
        stateless['jax']['platform'] = cfg['jax']['platform']
        none = make_agent(stateless, output/'none_normalizers', agent_cls=S3V2Agent,
                          device=device, bc_lambda=1., extra_agent_config={'s3_v2_mode': 'c2'})
        assert not any(k.startswith(('valnorm/', 'advnorm/')) for k in none.params)
        # No BC labels must also freeze Actor momentum when the Gate is CLOSED.
        before = parameter_digests(run.agent.params)
        empty = copy.deepcopy(raw); empty['teacher_valid'][:] = False
        run.carry['train'], _, mets = run.agent.train(run.carry['train'], model_batch(run.agent, empty, 33))
        after = parameter_digests(run.agent.params)
        assert float(mets['opt/actor_grad_norm']) == 0 and float(mets['bc/nll']) == 0
        assert all(before[k] == after[k] for k in before if actor_state(k) or rl_state(k))
        run.counters.state['all_updates'] += 1
        run.counters.state['wm_updates'] += 1
        run.counters.state['gate_closed_updates'] += 1
        run.counters.state['sampled_transitions'] += int(empty['teacher_valid'][:, 1:].size)
        run.collect(85)
        run.cp.save(final=True)
        assert int(run.agent.n_updates) == run.counters.state['all_updates'] == 8
        result = signed(dict(status='PASS_REAL_'+device.upper()+'_C2_CONTRACTS', output=str(output),
            scope='small original-network, original Env with controlled prices and Gate metrics',
            controlled_gate=True, stateful_normalizers='fixture only; production none is unchanged',
            devices=[str(d) for d in run.agent.train_devices],
            phases=[r['phase'] for r in rows], updates=8, gate_transition_updates=7,
            empty_bc_closed_updates=1, resumed_ready_updates=1,
            actor_gradients=gradients, counters=run.counters.save(),
            closed_frozen_states=True, reclosed_frozen_states=True,
            nonzero_inherited_moments=True, single_joint_optimizer=True,
            checkpoint=str(checkpoint), checkpoint_manifest_sha256=file_hash(checkpoint/'manifest.json'),
            final_checkpoint=str(run.cp.latest()),
            checkpoint_model_replay_sidecar_gate_equality=True, stateless_normalizers_have_no_parameters=True,
            production_gate_twenty_window_controller='PASS_CONTROLLED_METRICS_ONLY',
            production_market_gate_ready='NOT_RUN', formal_1m='NOT_RUN',
            elapsed_seconds=time.perf_counter()-started))
        write_json(output/'receipt.json', result)
        return result
    except BaseException as error:
        write_json(output/'failure.json', signed(dict(error_type=type(error).__name__, error=str(error))))
        raise
    finally:
        if run is not None:
            run.close()
        original.close()

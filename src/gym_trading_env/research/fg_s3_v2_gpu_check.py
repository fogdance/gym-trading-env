"""Small real GPU neural contract tests, distinct from production data runs."""
import copy
from pathlib import Path
import subprocess
import sys
import time
import uuid

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pandas as pd

from embodied.run.fg_checkpoint import plain
from embodied.run.fg_s3_source_transfer import make_agent
from embodied.run.fg_s3_source_admission import audit_code_transition
from dreamerv3.configuration import read_config_yaml
from dreamerv3.fg_s3_v2_agent import S3V2Agent
from dreamerv3.fg_s3a_actor import parameter_digests
from .fg_s3_v2_collector import V2Collector
from .fg_replay import signed, write_json


PROJECT = Path(__file__).resolve().parents[3]
DREAMER = PROJECT.parent/'dreamerv3'


def small_config():
    defaults = read_config_yaml(DREAMER/'dreamerv3/configs.yaml')['defaults']
    defaults['agent']['s3_clock_protocol'] = 'S3_MARKET_MINUTE'
    return plain(elements.Config(defaults).update({
        'jax.platform': 'cuda', 'jax.compute_dtype': 'float32',
        'batch_size': 1, 'batch_length': 64, 'report_length': 64, 'replay_context': 1,
        'agent.replay_context_mode': 'recompute_prefix', 'agent.update_diagnostics': True,
        'agent.enc.simple.layers': 1, 'agent.enc.simple.units': 16,
        'agent.dec.simple.layers': 1, 'agent.dec.simple.units': 16,
        'agent.dyn.rssm.deter': 32, 'agent.dyn.rssm.hidden': 16,
        'agent.dyn.rssm.stoch': 2, 'agent.dyn.rssm.classes': 4, 'agent.dyn.rssm.blocks': 4,
        'agent.policy.layers': 1, 'agent.policy.units': 16,
        'agent.value.layers': 1, 'agent.value.units': 16,
        'agent.rewhead.units': 16, 'agent.conhead.units': 16, 'agent.availhead.units': 16,
        'agent.imag_length': 3, 'agent.horizon': 333, 'agent.contdisc': True,
        'agent.avail_threshold': .30, 'agent.avail_actor_enabled': False,
        'agent.avail_always_available_index': 0}))


def gpu_batch(agent, raw, seed):
    from embodied.jax import internal
    data = internal.device_put(copy.deepcopy(raw), agent.train_sharded)
    return {**data, 'seed': agent._seeds(seed, agent.train_mirrored)}


def check_gpu_model(output=None):
    output = Path(output) if output else PROJECT/'artifacts/fg_s3_v2/checks'/('gpu_'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # Original Env controlled market fixture; not a production FG price test.
    sys.path.insert(0, str(PROJECT/'tests/entry_v1'))
    from test_fg_single3_env import single_env
    from test_fg_s3_dagger_d0 import query_for
    folder = output/'fixture'; folder.mkdir()
    prices = [(1000,1000,1000,1000),(1000,1000,1000,1000),(1000,1007,1000,1007)]*30
    env = single_env(folder, prices, candidate='C1', fees=3, chunk=90,
        times=pd.date_range('2026-01-05 13:31', periods=90, freq='min'),
        calendar=[dict(start='2026-01-05 13:30', end='2026-01-05 15:30')],
        liquidation='2026-01-05 15:30')
    query, _ = query_for(env, folder/'env.yaml')
    collector = V2Collector(env, query, output/'collector', DREAMER, beta=1., actor_probability=0.)
    try:
        collector.advance(75)
        assert collector.pending['stepid'] is None and collector.commit_seq == 75
        assert all(r['execution_result'] is not None for r in collector.records)
        import embodied
        stream = embodied.streams.Consec(embodied.streams.Stateless(lambda: collector.sample(1)),
            length=64, consec=1, prefix=1, strict=True)
        raw = next(iter(stream))
        assert raw['teacher_valid'][:, 1:].any() and (raw['teacher_action'][:, 1:] == 1).any()
        wrong = {**raw, 'action': 1-raw['action']}
        try:
            collector.join(wrong)
            raise AssertionError('Corrupt action was accepted')
        except ValueError:
            pass
        cfg = small_config()
        a = make_agent(cfg, output/'a2', agent_cls=S3V2Agent, device='gpu', bc_lambda=1.,
                       extra_agent_config={'s3_v2_mode': 'a2'})
        if not all(d.platform == 'gpu' for d in jax.devices()):
            raise RuntimeError('GPU tests refuse a CPU backend')
        # Real nonzero inherited moments protect against a superficial zero-grad
        # freeze. All small-fixture parameters remain on CUDA.
        for k, v in list(a.params.items()):
            if k.startswith('opt/state/'):
                value = 1000 if np.issubdtype(v.dtype, np.integer) else .125
                a.params[k] = jax.device_put(np.full(v.shape, value, v.dtype), v.sharding)
        a.sync_policy_from_train()
        initial = a.save_exact()
        frozen = lambda k: k.startswith(('val/', 'slowval/', 'slowval_count/', 'retnorm/', 'valnorm/', 'advnorm/')) or (
            k.startswith('opt/state/') and '/val/' in k)
        actor = lambda k: k.startswith('pol/') or (k.startswith('opt/state/') and '/pol/' in k)
        before = parameter_digests(a.params)
        carry, _, m = a.train(a.init_train(1), gpu_batch(a, raw, 7))
        write_json(output/'a2_first_update.json', {k: float(np.asarray(v)) for k, v in m.items()
            if np.asarray(v).size == 1 and np.isfinite(v).all()})
        after = parameter_digests(a.params)
        assert float(m['opt/actor_grad_norm']) > 0 and float(m['opt/wm_grad_norm']) > 0
        assert any(before[k] != after[k] for k in before if k.startswith('pol/'))
        assert any(before[k] != after[k] for k in before if k.startswith(('enc/', 'dyn/')))
        assert all(before[k] == after[k] for k in before if frozen(k))
        assert not bool(m['v2/rl_active']) and bool(m['bc/active'])
        empty = copy.deepcopy(raw); empty['teacher_valid'][:] = False; empty['teacher_action'][:] = 1
        b = parameter_digests(a.params)
        _, _, zero = a.train(carry, gpu_batch(a, empty, 8))
        z = parameter_digests(a.params)
        assert float(zero['bc/nll']) == 0 and float(zero['opt/actor_grad_norm']) == 0
        assert all(b[k] == z[k] for k in b if actor(k) or frozen(k))

        def only_bc(c, d):
            c, obs, prev, _ = a.model._apply_replay_context(c, d)
            _, (_, _, outs, _) = a.model.wm_only(c, obs, prev, False)
            return a.model.bc_loss(outs['repfeat'], obs['action_mask'],
                d['teacher_action'][:,1:], d['teacher_valid'][:,1:])[0]
        data = gpu_batch(a, raw, 9); data.pop('seed')
        bc_modules = [m for m in a.model.modules if m is not a.model.val]
        _, (_, _, grads) = jax.jit(nj.pure(lambda c, d: nj.grad(only_bc, bc_modules)(c, d)),
            static_argnames=('create', 'modify'))(a.params, a.init_train(1), data, seed=9, create=False, modify=True)
        assert all(np.count_nonzero(np.asarray(v)) == 0 for k, v in grads.items() if not k.startswith('pol/'))
        assert any(np.count_nonzero(np.asarray(v)) for k, v in grads.items() if k.startswith('pol/'))

        c = make_agent(cfg, output/'c2', agent_cls=S3V2Agent, device='gpu', bc_lambda=1.,
                       extra_agent_config={'s3_v2_mode': 'c2'})
        c.load_exact(initial)
        cb = parameter_digests(c.params)
        cb_values = {k: np.asarray(jax.device_get(v)).copy() for k, v in c.params.items() if frozen(k)}
        cc, _, closed = c.train(c.init_train(1), gpu_batch(c, raw, 7))
        ca = parameter_digests(c.params)
        write_json(output/'c2_closed_update.json', dict(
            changed_frozen_keys=[k for k in cb if frozen(k) and cb[k] != ca[k]],
            frozen_differences={k: dict(numerical_differences=int(np.count_nonzero(
                cb_values[k] != np.asarray(jax.device_get(c.params[k])))),
                sign_differences=int(np.count_nonzero(np.signbit(cb_values[k]) !=
                    np.signbit(np.asarray(jax.device_get(c.params[k]))))))
                for k in cb_values if cb[k] != ca[k]},
            metrics={k: float(np.asarray(v)) for k, v in closed.items()
                     if np.asarray(v).size == 1 and np.isfinite(v).all()}))
        assert float(closed['opt/actor_grad_norm']) > 0 and not bool(closed['v2/rl_active'])
        assert all(cb[k] == ca[k] for k in cb if frozen(k))
        c.set_avail_actor_enabled(True)  # Controlled READY fixture, never a production Gate pass.
        cb = parameter_digests(c.params)
        _, _, ready = c.train(cc, gpu_batch(c, raw, 10))
        ca = parameter_digests(c.params)
        assert bool(ready['v2/rl_active']) and float(ready['opt/critic_grad_norm']) > 0
        assert any(cb[k] != ca[k] for k in cb if k.startswith('val/'))
        assert any(cb[k] != ca[k] for k in cb if k.startswith('slowval/'))

        zero_agent = make_agent(cfg, output/'lambda0', agent_cls=S3V2Agent, device='gpu', bc_lambda=0.,
                                extra_agent_config={'s3_v2_mode': 'c2'})
        zero_agent.load_exact(initial); zero_agent.set_avail_actor_enabled(True)
        checkpoint = zero_agent.save_exact()
        _, _, zm = zero_agent.train(zero_agent.init_train(1), gpu_batch(zero_agent, raw, 12))
        original = parameter_digests(zero_agent.params)
        zero_agent.load_exact(checkpoint)
        swapped = copy.deepcopy(raw); swapped['teacher_action'][:] = 1-swapped['teacher_action']
        swapped['teacher_valid'][:] = False
        zero_agent.train(zero_agent.init_train(1), gpu_batch(zero_agent, swapped, 12))
        assert original == parameter_digests(zero_agent.params) and float(zm['bc/nll']) == 0

        original_code = subprocess.check_output(['git','show','e872602:dreamerv3/agent.py'], cwd=DREAMER)
        now = (DREAMER/'dreamerv3/agent.py').read_bytes()
        audit_code_transition(original_code, now, 'dreamerv3/agent.py')
        poisoned = now.replace(b"losses['rew'] = self.rew(inp, 2).loss(obs['reward'])",
                               b"losses['rew'] = 2 * self.rew(inp, 2).loss(obs['reward'])")
        try:
            audit_code_transition(original_code, poisoned, 'dreamerv3/agent.py')
            raise AssertionError('Changed World Model math passed transfer admission')
        except ValueError:
            pass
        result = signed(dict(status='PASS_REAL_GPU_NEURAL_CONTRACTS', output=str(output),
            scope='small original-network and original-Env controlled fixture, not production-market acceptance',
            devices=[str(d) for d in jax.devices()], elapsed_seconds=time.perf_counter()-started,
            checks=['live_commit_and_owner', 'a2_wm_bc_gradients', 'bc_stop_gradient',
                    'critic_and_moments_frozen', 'empty_bc_actor_moments_frozen',
                    'c2_closed', 'c2_controlled_ready', 'lambda0_teacher_independence',
                    'wm_extraction_ast_and_poison_rejection'],
            a2_actor_grad=float(m['opt/actor_grad_norm']), wm_grad=float(m['opt/wm_grad_norm']),
            c2_ready_critic_grad=float(ready['opt/critic_grad_norm']),
            production_gate_20_windows='NOT_RUN', formal_1m='NOT_RUN'))
        write_json(output/'receipt.json', result)
        return result
    except BaseException as error:
        write_json(output/'failure.json', signed(dict(error_type=type(error).__name__, error=str(error))))
        raise
    finally:
        collector.close()

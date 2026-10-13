"""Tutor numerical/CPU contracts and read-only, real TRAIN replay coverage.

Small CPU neural tests are not GPU or production training acceptance.
"""
import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import types

import embodied
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from embodied.jax.outs import MaskedCategorical
from embodied.run.fg_checkpoint import file_hash, load_replay, validate_bundle
from embodied.run.fg_s3_source_transfer import make_agent
from embodied.run.fg_s3_v2_train import Counters
from dreamerv3.fg_s3_tutor_loss import agent_extensions, eligible_teacher, pair_mask, tutor_objective
from dreamerv3.fg_s3_v2_agent import S3V2Agent
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_replay import read_manifest, signed, write_json
from gym_trading_env.research.fg_s3_dagger_d0 import join_d0_teacher
from gym_trading_env.research.fg_s3_v2_gpu_check import small_config, gpu_batch


PROJECT = Path(__file__).resolve().parents[2]
CHECKS = PROJECT/'artifacts/fg_s3_v2/tutor_checks'
E2_CHECKS = PROJECT/'artifacts/fg_s3_v2/tutor_e2_checks'
RECORDED = PROJECT.parent.parent/'fg-s3-dev/gym-trading-env/artifacts/fg_s3_v2/runs/A2_P0_GPU_SMOKE_1K_FAST_v1/ckpt/step_000000001000'


def objective(logits, y, valid, first, alpha=.8):
    safe_mask = jnp.where(valid[..., None], jnp.ones_like(logits, bool), jnp.array([True, False]))
    dist = MaskedCategorical(logits, safe_mask, unimix=.01)
    return tutor_objective(-dist.logp(jnp.where(valid, y, 0)), dist.logits, y, valid, first,
                           alpha=alpha)


def test_natural_balanced_pair_match_numpy_and_ranking_direction():
    logits = jnp.array([[[.4, -.2], [-.3, .8], [.1, .2], [0., -.5]]])
    y = jnp.array([[1, 0, 0, 0]])
    v = jnp.ones_like(y, bool); first = jnp.zeros_like(v)
    mix, m = jax.jit(objective)(logits, y, v, first)
    x = np.asarray(logits)
    p = np.exp(x-x.max(-1, keepdims=True)); p = .99*p/p.sum(-1, keepdims=True)+.005
    ce = -np.log(np.take_along_axis(p, np.asarray(y)[..., None], -1)[..., 0])
    z = np.log(p[..., 1])-np.log(p[..., 0])
    pair = np.logaddexp(0, -(z[0, 0]-z[0, 1:])).mean()
    balanced = .5*(ce[0, 0]+ce[0, 1:].mean())
    np.testing.assert_allclose([m['tutor/natural_ce'], m['tutor/balanced_ce'], m['tutor/pair_loss'], mix],
        [ce.mean(), balanced, pair, .8*ce.mean()+.2*balanced], rtol=2e-6)
    assert int(m['tutor/pair_count']) == 3
    improved = logits.at[0, 0, 1].add(1.)
    assert objective(improved, y, v, first)[1]['tutor/pair_loss'] < m['tutor/pair_loss']
    grad = jax.grad(lambda x: objective(x, y, v, first)[1]['tutor/pair_loss'])(logits)
    assert grad[0, 0, 1] < 0 and np.all(np.asarray(grad)[0, 1:, 1] > 0)
    with pytest.raises(ValueError, match='effective binary-action'):
        tutor_objective(jnp.zeros((1, 3)), logits, y, v, first)


def test_empty_single_class_and_invalid_positions_have_finite_zero_pair_gradients():
    logits = jnp.array([[[.1, -.4], [.2, .7], [-.2, .3]]])
    first = jnp.zeros((1, 3), bool)
    for y, v in [(jnp.ones((1, 3), jnp.int32), jnp.ones((1, 3), bool)),
                 (jnp.zeros((1, 3), jnp.int32), jnp.ones((1, 3), bool)),
                 (jnp.ones((1, 3), jnp.int32), jnp.zeros((1, 3), bool))]:
        mix, m = objective(logits, y, v, first)
        assert m['tutor/balanced_ce'] == m['tutor/natural_ce']
        assert m['tutor/pair_loss'] == m['tutor/pair_count'] == 0
        if not v.any(): assert mix == 0
        grad = jax.grad(lambda x: objective(x, y, v, first)[1]['tutor/pair_loss'])(logits)
        assert np.isfinite(np.asarray(grad)).all() and not np.asarray(grad).any()
    y = jnp.array([[1, 0, 1]]); v = jnp.array([[True, True, False]])
    _, m = objective(logits, y, v, first)
    assert m['tutor/pair_count'] == 1 and m['bc/valid_count'] == 2
    grad = jax.grad(lambda x: sum(objective(x, y, v, first)[0:1]))(logits)
    assert not np.asarray(grad)[0, 2].any()
    # The pure reduction also excludes a masked -inf sentinel before subtraction.
    safe = logits.at[0, 2].set(-jnp.inf)
    _, m = tutor_objective(jnp.ones((1, 3)), safe, y, v, first)
    assert all(np.isfinite(np.asarray(x)).all() for x in m.values())


def test_pair_episode_mask_does_not_pair_across_resets_or_batch_members():
    y = jnp.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    v = jnp.ones_like(y, bool)
    first = jnp.array([[True, False, True, False], [True, False, True, False]])
    assert pair_mask(y, v, first).sum() == 0
    first = first.at[:, 2].set(False)
    assert pair_mask(y, v, first).sum() == 8
    mask = jnp.array([[[1, 1], [1, 0], [0, 0], [1, 1]]])
    valid = jnp.array([[True, False, False, False]])
    np.testing.assert_array_equal(eligible_teacher(mask, y[:1], valid), valid)
    with pytest.raises(Exception, match='TUTOR_CERTIFIED_LABEL'):
        jax.jit(eligible_teacher)(mask, y[:1], valid.at[0, 1].set(True)).block_until_ready()


def test_frozen_tutor_settings_and_checkpointed_counters(tmp_path):
    from dreamerv3.configuration import read_config_yaml
    from gym_trading_env.research.fg_s3_v2_experiment import read_settings, require_training_authorization
    cfg = PROJECT/'configs/experiments/fg_s3_v2/TUTOR_MARKET_MINUTE_GPU_1K.yaml'
    settings = read_settings(cfg)
    ext = agent_extensions(settings)
    assert ext == agent_extensions(copy.deepcopy(settings))
    assert ext['tutor_alpha'] == .8 and ext['tutor_hard_kappa'] == 0
    for changed in ({'tutor_alpha': float('nan')}, {'tutor_pair_tau': 0},
                    {'tutor_pair_lambda': -1}, {'tutor_hard_kappa': -1}):
        with pytest.raises(ValueError): agent_extensions({**settings, **changed})
    with pytest.raises(PermissionError):
        require_training_authorization({**settings, 'scope': 'formal'}, False)
    assert agent_extensions({'mode': 'a2'}) == {'s3_v2_mode': 'a2'}
    assert agent_extensions({'mode': 'c2'}) == {'s3_v2_mode': 'c2'}
    counters = Counters('tutor'); counters.state['pair_update_count'] = 3
    resumed = Counters('tutor'); resumed.load(pickle.loads(pickle.dumps(counters.save())))
    assert resumed.save() == counters.save()
    assert 'pair_update_count' not in Counters().save()
    assert read_config_yaml(cfg)['env_steps'] == 1000


@pytest.fixture(scope='module')
def real_batches():
    node = Path(os.environ.get('FG_S3_TUTOR_REPLAY_CHECKPOINT', RECORDED))
    if not node.is_dir():
        pytest.skip('Needs the sealed real FG201 TRAIN checkpoint; no synthetic labels substituted')
    meta = validate_bundle(node)
    source_hash = file_hash(node/'manifest.json')
    state = pickle.loads((node/'replay.pkl').read_bytes())
    teachers = pickle.loads((node/'collector.pkl').read_bytes())
    records = teachers['records']
    replay = embodied.Replay(**{k: state[k] for k in ('length', 'capacity', 'chunksize', 'online')})
    load_replay(replay, state, node/'replay')
    stream = embodied.streams.Consec(embodied.streams.Stateless(
        lambda: join_d0_teacher(replay.sample(1), records)),
        length=64, consec=1, prefix=1, strict=True)
    stream = iter(stream)
    batches = [next(stream) for _ in range(100)]
    # Restored Replay is only an in-memory copy; no source files are written.
    assert source_hash == file_hash(node/'manifest.json')
    return node, meta, records, batches


def test_real_native_replay_sidecar_pair_coverage(real_batches):
    node, meta, records, batches = real_batches
    owners = {r['stepid']: r for r in records}
    counts, positives, negatives, resets, prevented, unique = [], [], [], 0, 0, set()
    for raw in batches:
        y, v, first = raw['teacher_action'][:, 1:], raw['teacher_valid'][:, 1:], raw['is_first'][:, 1:]
        pm = np.asarray(pair_mask(jnp.asarray(y), jnp.asarray(v), jnp.asarray(first)))
        counts.append(int(pm.sum())); positives.append(int((v & (y == 1)).sum()))
        negatives.append(int((v & (y == 0)).sum()))
        resets += int(first.any())
        prevented += positives[-1]*negatives[-1]-counts[-1]
        for pos in np.ndindex(y.shape):
            stepid = raw['stepid'][pos[0], pos[1]+1].tobytes().hex()
            unique.add(stepid); row = owners[stepid]
            assert int(raw['action'][pos[0], pos[1]+1]) == row['executed_action']
            if v[pos]:
                assert row['oracle_certified'] and row['oracle_valid'] and all(row['action_mask'])
                assert row['execution_result']['advanced'] and y[pos] == row['oracle_action']
        assert not any(k.startswith(('oracle', 'difficulty', 'pair')) for k in raw)
    assert sum(counts) > 0
    receipt = signed(dict(status='PASS_REAL_RECORDED_TRAIN_PAIR_CONTRACT', source=str(node),
        checkpoint_manifest_sha256=file_hash(node/'manifest.json'), contracts=sorted({r['contract'] for r in records}),
        committed_source_rows=len(records), sampled_batches=len(batches), batch_size=1, effective_length=64,
        prefix=1, eligible_pos=sum(positives), eligible_neg=sum(negatives), pair_count=sum(counts),
        pairable_batches=sum(n > 0 for n in counts), pairable_rate=sum(n > 0 for n in counts)/len(counts),
        pair_count_min=min(counts), pair_count_max=max(counts), reset_containing_batches=resets,
        cross_reset_pairs_excluded=prevented, unique_effective_owners=len(unique),
        fresh_env_collection='NOT_RUN; reads previously completed real TRAIN native Replay',
        neural_training='NOT_RUN in this test', gpu_acceptance='NOT_RUN'))
    write_json(CHECKS/'real_pair_coverage.json', receipt)
    assert len(unique) > 64
    # The shared join continues rejecting altered executed actions and missing owners.
    bad = {k: v.copy() for k, v in batches[0].items() if not k.startswith('teacher_')}
    bad['action'][0, 1] = 1-bad['action'][0, 1]
    with pytest.raises(ValueError, match='executed action'): join_d0_teacher(bad, records)


def frozen(k):
    return k.startswith(('val/', 'slowval/', 'slowval_count/', 'retnorm/', 'valnorm/', 'advnorm/')) or (
        k.startswith('opt/state/') and '/val/' in k)


def actor_state(k):
    return k.startswith('pol/') or (k.startswith('opt/state/') and '/pol/' in k)


@pytest.mark.parametrize('hard_kappa', [0., 1.])
def test_small_cpu_tutor_joint_gradients_freeze_and_restore(real_batches, tmp_path, hard_kappa):
    assert all(d.platform == 'cpu' for d in jax.devices()), 'CPU test must not use the A2 GPU'
    _, _, _, batches = real_batches
    raw = next(b for b in batches if np.asarray(pair_mask(jnp.asarray(b['teacher_action'][:,1:]),
        jnp.asarray(b['teacher_valid'][:,1:]), jnp.asarray(b['is_first'][:,1:]))).any())
    a = make_agent(small_config(), tmp_path/'tutor', device='cpu', agent_cls=S3V2Agent,
        bc_lambda=1., extra_agent_config=agent_extensions({'mode': 'tutor', 'tutor_hard_kappa': hard_kappa}))
    for k, v in list(a.params.items()):
        if k.startswith('opt/state/'):
            value = 1000 if np.issubdtype(v.dtype, np.integer) else .125
            a.params[k] = jax.device_put(np.full(v.shape, value, v.dtype), v.sharding)
    a.sync_policy_from_train()
    carry = a.init_train(1)
    results = []
    for seed, ready in [(10, False), (11, True), (12, False)]:
        a.set_avail_actor_enabled(ready)  # Fixture only; Tutor never enables RL even at READY.
        before = parameter_digests(a.params)
        step = int(a.params['opt/step/value'])
        carry, _, m = a.train(carry, gpu_batch(a, raw, seed))
        after = parameter_digests(a.params)
        assert int(a.params['opt/step/value']) == step+1
        assert m['tutor/pair_loss'] > 0 and m['tutor/pair_count'] > 0
        assert m['opt/actor_grad_norm'] > 0 and m['opt/wm_grad_norm'] > 0
        assert m['opt/critic_grad_norm'] == m['v2/rl_active'] == 0
        if hard_kappa:
            assert m['tutor/hard_balanced_active']
            assert 1 <= m['tutor/difficulty_weight_mean'] <= m['tutor/difficulty_weight_max'] <= 2
        assert all(before[k] == after[k] for k in before if frozen(k))
        assert any(before[k] != after[k] for k in before if k.startswith('pol/'))
        assert any(before[k] != after[k] for k in before if k.startswith(('enc/', 'dyn/')))
        results.append(dict(gate_ready=ready, actor_grad=float(m['opt/actor_grad_norm']),
            wm_grad=float(m['opt/wm_grad_norm']), pair_loss=float(m['tutor/pair_loss']),
            pair_count=int(m['tutor/pair_count']), critic_grad=float(m['opt/critic_grad_norm'])))
    saved = a.save_exact(); digest = parameter_digests(a.params)
    a.load_exact(pickle.loads(pickle.dumps(saved)))
    assert digest == parameter_digests(a.params)
    carry, _, m = a.train(carry, gpu_batch(a, raw, 13))
    assert m['opt/actor_changed'] and m['opt/wm_changed']
    before = parameter_digests(a.params)
    empty = copy.deepcopy(raw); empty['teacher_valid'][:] = False
    carry, _, m = a.train(carry, gpu_batch(a, empty, 14))
    after = parameter_digests(a.params)
    assert m['tutor/pair_loss'] == m['bc/nll'] == m['opt/actor_grad_norm'] == 0
    assert all(before[k] == after[k] for k in before if frozen(k) or actor_state(k))

    def teaching_only(c, d, pair_only):
        c, obs, prev, _ = a.model._apply_replay_context(c, d)
        _, (_, _, outs, _) = a.model.wm_only(c, obs, prev, False)
        bc, metrics = a.model.tutor_loss(outs['repfeat'], obs, d['teacher_action'][:,1:], d['teacher_valid'][:,1:])
        return metrics['tutor/pair_loss'] if pair_only else bc+.1*metrics['tutor/pair_loss']
    modules = [x for x in a.model.modules if x is not a.model.val]
    data = gpu_batch(a, raw, 15); data.pop('seed')
    for pair_only in (False, True):
        _, (_, _, grads) = jax.jit(nj.pure(lambda c, d: nj.grad(
            lambda c, d: teaching_only(c, d, pair_only), modules)(c, d)),
            static_argnames=('create', 'modify'))(a.params, a.init_train(1), data,
                seed=15, create=False, modify=True)
        assert all(not np.asarray(v).any() for k, v in grads.items() if not k.startswith('pol/'))
        assert any(np.asarray(v).any() for k, v in grads.items() if k.startswith('pol/'))
    write_json((E2_CHECKS if hard_kappa else CHECKS)/'cpu_neural.json', signed(dict(status='PASS_SMALL_CPU_TUTOR_CONTRACTS',
        hard_kappa=hard_kappa,
        devices=[str(d) for d in jax.devices()], scope='small model over previously recorded real FG201 batch',
        updates=5, gate_sequence=['CLOSED', 'READY fixture', 'CLOSED'], results=results,
        critic_and_optimizer_frozen=True, empty_teacher_actor_momentum_frozen=True,
        pair_only_gradient_actor_nonzero=True, teaching_gradient_outside_actor_zero=True,
        joint_optimizer_calls_per_update=1, model_save_load_continue='PASS',
        full_live_replay_gpu_resume='NOT_RUN in this test')))


def test_pair_only_activation_and_original_a2_c2_routes(real_batches, tmp_path):
    raw = real_batches[3][0]
    cfg = small_config()
    def model(name, mode, bc_lambda=1., **kwargs):
        return make_agent(cfg, tmp_path/name, device='cpu', agent_cls=S3V2Agent,
            bc_lambda=bc_lambda, extra_agent_config=agent_extensions({'mode': mode, **kwargs}))
    a = model('a2', 'a2'); initial = a.save_exact()
    _, _, am = a.train(a.init_train(1), gpu_batch(a, raw, 30))
    # Compare each original mode with its own pre-Tutor implementation. C2's
    # Ninjax conditional has a different RNG route from A2; they are not an
    # assertion of cross-algorithm bitwise equality.
    baseline = types.ModuleType('dreamerv3._tutor_baseline')
    baseline.__package__ = 'dreamerv3'
    source = subprocess.check_output(['git', 'show',
        '6356747d64963fb5ef55a036d837591f778c9f0a:dreamerv3/fg_s3_v2_agent.py'],
        cwd=PROJECT.parent/'dreamerv3')
    exec(compile(source, '<pre-Tutor S3V2Agent>', 'exec'), baseline.__dict__)
    for mode in ('a2', 'c2'):
        old = make_agent(cfg, tmp_path/('baseline_'+mode), device='cpu',
            agent_cls=baseline.S3V2Agent, bc_lambda=1., extra_agent_config=agent_extensions({'mode': mode}))
        old.load_exact(initial)
        old.train(old.init_train(1), gpu_batch(old, raw, 30))
        current = a if mode == 'a2' else model('c2_closed', 'c2')
        if mode == 'c2':
            current.load_exact(initial)
            current.train(current.init_train(1), gpu_batch(current, raw, 30))
        assert parameter_digests(old.params) == parameter_digests(current.params)
    b = model('tutor_off', 'tutor', tutor_alpha=1., tutor_pair_lambda=0.)
    b.load_exact(initial)
    prefix_changed = copy.deepcopy(raw)
    prefix_changed['teacher_action'][:, 0] = 1-prefix_changed['teacher_action'][:, 0]
    prefix_changed['teacher_valid'][:, 0] = ~prefix_changed['teacher_valid'][:, 0]
    _, _, m = b.train(b.init_train(1), gpu_batch(b, prefix_changed, 30))
    max_difference = 0.
    for k in a.params:
        left, right = np.asarray(a.params[k]), np.asarray(b.params[k])
        np.testing.assert_allclose(left, right, rtol=2e-5, atol=1e-6, err_msg=k)
        max_difference = max(max_difference, float(np.max(np.abs(left.astype(np.float64)-right))))
    np.testing.assert_allclose(float(m['bc/nll']), float(am['bc/nll']), rtol=2e-6)
    pair = model('pair_only', 'tutor', bc_lambda=0.)
    pair.load_exact(initial)
    before = parameter_digests(pair.params)
    carry, _, m = pair.train(pair.init_train(1), gpu_batch(pair, raw, 30))
    # The unchanged source optimizer starts its fresh 1000-update warmup at
    # LR=0. A second update verifies parameter movement as well as gradients.
    carry, _, m = pair.train(carry, gpu_batch(pair, raw, 31))
    after = parameter_digests(pair.params)
    assert m['tutor/pair_count'] > 0 and m['tutor/actor_active'] and not m['bc/active']
    assert m['opt/actor_grad_norm'] > 0 and any(before[k] != after[k] for k in before if k.startswith('pol/'))
    no_pair = copy.deepcopy(raw)
    no_pair['teacher_valid'] &= no_pair['teacher_action'] == 0
    before = parameter_digests(pair.params)
    carry, _, m = pair.train(carry, gpu_batch(pair, no_pair, 32))
    after = parameter_digests(pair.params)
    assert m['tutor/pair_count'] == m['tutor/actor_active'] == m['opt/actor_grad_norm'] == 0
    assert all(before[k] == after[k] for k in before if actor_state(k) or frozen(k))
    zero = model('all_zero', 'tutor', bc_lambda=0., tutor_pair_lambda=0.)
    zero.load_exact(pair.save_exact())  # Already nonzero Actor optimizer moments.
    before = parameter_digests(zero.params)
    _, _, m = zero.train(zero.init_train(1), gpu_batch(zero, raw, 31))
    after = parameter_digests(zero.params)
    assert m['opt/actor_grad_norm'] == 0 and m['tutor/actor_active'] == 0
    assert all(before[k] == after[k] for k in before if actor_state(k) or frozen(k))
    write_json(CHECKS/'cpu_compatibility.json', signed(dict(status='PASS_CPU_MODE_COMPATIBILITY',
        original_a2_unchanged='byte-equal to pre-Tutor A2 update',
        original_c2_closed_unchanged='byte-equal to pre-Tutor C2 CLOSED update',
        natural_only_tutor_vs_a2_max_parameter_difference=max_difference,
        prefix_teacher_excluded=True, pair_only_actor_update=True, no_pair_actor_momentum_frozen=True,
        all_zero_actor_and_optimizer_frozen=True, scope='small CPU model, recorded real TRAIN batch')))


def test_actual_gpu_tutor_1k_checkpoint_resume_and_pairs():
    """Read the actual GPU run; never substitutes an artificial checkpoint."""
    root = PROJECT/'artifacts/fg_s3_v2/runs/TUTOR_CD_V0_GPU_SMOKE_1K_v1'
    if not (root/'training_receipt.json').is_file():
        pytest.skip('Requires the actual Tutor GPU 1K start/exit/resume run')
    receipt = read_manifest(root/'training_receipt.json')
    if receipt['env_steps'] != 1000:
        pytest.skip('The real GPU run has not completed its 1K/resume budget')
    identity = read_manifest(root/'run_identity.json')['identity']
    assert identity['mode'] == 'tutor' and identity['agent_extension'] == agent_extensions({'mode': 'tutor'})
    nodes = [root/'ckpt'/f'step_{n:012d}' for n in (0, 500, 1000)]
    for node in nodes: validate_bundle(node, identity)
    def load(node, name):
        return pickle.loads((node/(name+'.pkl')).read_bytes())
    agents = [load(node, 'agent') for node in nodes]
    hashes = [parameter_digests(a['params']) for a in agents]
    assert [a['counters']['updates'] for a in agents] == [21183, 21203, 21223]
    for before, after in zip(hashes, hashes[1:]):
        assert all(before[k] == after[k] for k in before if frozen(k))
        assert any(before[k] != after[k] for k in before if k.startswith('pol/'))
        assert any(before[k] != after[k] for k in before if k.startswith(('enc/', 'dyn/')))
    for expected, node in zip((0, 20, 40), nodes):
        c = load(node, 'v2_counters')
        assert c['all_updates'] == c['wm_updates'] == expected
        assert c['rl_actor_updates'] == c['rl_critic_updates'] == 0
    boundary, final = [load(n, 'collector') for n in nodes[1:]]
    assert boundary['records'] == final['records'][:len(boundary['records'])]
    pending = boundary['pending']
    resumed = final['records'][boundary['commit_seq']]
    for k in ('state_id', 'actual_observation_id', 'executed_action', 'oracle_proof_hash', 'policy_model_update'):
        assert pending[k] == resumed[k]
    assert final['pending']['stepid'] is None and final['pending_native'] is not None
    native = {}
    for shard in load(nodes[-1], 'replay')['chunks']:
        with np.load(nodes[-1]/'replay'/shard['name'], allow_pickle=False) as data:
            assert not any(k.startswith(('teacher', 'oracle', 'tutor', 'difficulty', 'pair')) for k in data.files)
            native.update((s.tobytes().hex(), int(a)) for s, a in zip(data['stepid'], data['action']))
    assert set(native) == {r['stepid'] for r in final['records']}
    assert all(native[r['stepid']] == r['executed_action'] for r in final['records'])
    updates = [json.loads(x) for x in (root/'updates.jsonl').read_text().splitlines()]
    assert len(updates) == 40 and [r['local_update'] for r in updates] == list(range(1, 41))
    pairable = [r for r in updates if r['metrics']['tutor/pair_count'] > 0]
    assert pairable and all(r['metrics']['tutor/pair_loss'] > 0 for r in pairable)
    assert all(r['metrics']['opt/wm_grad_norm'] > 0 and r['metrics']['opt/critic_grad_norm'] == 0
               and r['metrics']['v2/rl_active'] == 0 for r in updates)
    assert all(r['metrics']['opt/actor_grad_norm'] > 0 for r in updates if r['metrics']['tutor/actor_active'])
    counters = load(nodes[-1], 'v2_counters')
    assert counters['pairable_batches'] == counters['pair_update_count'] == len(pairable)
    assert counters == receipt['counters']
    spec = read_manifest(root/'source_admission.json')
    source = Path(spec['origin']['path'])
    assert file_hash(source/'manifest.json') == spec['origin']['manifest_sha256']
    assert file_hash(source/'agent.pkl') == json.loads((source/'manifest.json').read_text())['files']['agent.pkl']
    transfer = read_manifest(root/'source_transfer.json')
    assert hashes[0] == transfer['params_after'] and not transfer['independent_bc_optimizer_imported']
    assert all('cuda' in d for d in receipt['devices'])
    write_json(CHECKS/'gpu_1k_verification.json', signed(dict(status='PASS_ACTUAL_GPU_TUTOR_1K_AND_RESUME',
        run=str(root), frozen_versions=identity['versions'], checkpoint=str(nodes[-1]),
        checkpoint_manifest_sha256=file_hash(nodes[-1]/'manifest.json'), env_steps=1000, updates=40,
        resumed_updates=20, pairable_updates=len(pairable), pairable_rate=len(pairable)/40,
        pair_count=sum(r['metrics']['tutor/pair_count'] for r in updates),
        actor_and_wm_continue_learning=True, critic_rl_optimizer_frozen=True,
        replay_executed_action_sidecar_and_pending_consistent=True, source_unchanged=True,
        scope='actual GPU FG201 1K diagnostic, not production or trading-success acceptance',
        formal_1m='NOT_RUN', ten_k='NOT_RUN')))

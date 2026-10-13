"""Batch-local bounded curriculum: numerical, gradient and static compatibility."""
from pathlib import Path
import json
import pickle
import subprocess
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embodied.jax.outs import MaskedCategorical
from dreamerv3.fg_s3_tutor_loss import agent_extensions, tutor_objective
from gym_trading_env.research.fg_replay import signed, write_json
from gym_trading_env.research.fg_replay import read_manifest
from embodied.run.fg_checkpoint import file_hash, validate_bundle
from dreamerv3.fg_s3a_actor import parameter_digests


PROJECT = Path(__file__).resolve().parents[2]
CHECKS = PROJECT/'artifacts/fg_s3_v2/tutor_e2_checks'


def distribution(x, y, valid):
    mask = jnp.where(valid[..., None], jnp.ones_like(x, bool), jnp.array([True, False]))
    dist = MaskedCategorical(x, mask, unimix=.01)
    return -dist.logp(jnp.where(valid, y, 0)), dist.logits


def test_dynamic_class_normalization_stop_gradient_and_weight_limit():
    x = jnp.array([[[.9, -.4], [-.3, .6], [.2, -.6], [-.5, 1.1], [1., -1.]]])
    y = jnp.array([[1, 1, 0, 0, 1]])
    v = jnp.array([[True, True, True, True, False]])
    first = jnp.zeros_like(v)
    def dynamic(x, kappa=1., wmax=2.):
        nll, logits = distribution(x, y, v)
        return tutor_objective(nll, logits, y, v, first, hard_kappa=kappa, hard_wmax=wmax)
    mix, m = jax.jit(lambda x: dynamic(x))(x)
    nll, logits = distribution(x, y, v)
    p = np.asarray(jax.nn.softmax(logits, -1)[..., 1])
    weights = np.clip(1+np.where(np.asarray(y) == 1, 1-p, p), 1, 2)
    pos, neg = np.asarray(v & (y == 1)), np.asarray(v & (y == 0))
    ce = np.asarray(nll)
    expected = .5*(np.sum(weights[pos]*ce[pos])/weights[pos].sum()
                   + np.sum(weights[neg]*ce[neg])/weights[neg].sum())
    np.testing.assert_allclose(m['tutor/hard_balanced_ce'], expected, rtol=2e-6)
    np.testing.assert_allclose(mix, .8*ce[np.asarray(v)].mean()+.2*expected, rtol=2e-6)
    assert m['tutor/hard_balanced_active'] and 1 <= m['tutor/difficulty_weight_mean'] <= 2
    assert dynamic(x, 10., 1.5)[1]['tutor/difficulty_weight_max'] <= 1.5
    # Frozen weights give the exact surrogate gradient; differentiating weights
    # as well would violate the reviewed difficulty contract.
    frozen = jnp.asarray(weights)
    def surrogate(x):
        ce, _ = distribution(x, y, v)
        def mean(mask):
            w = jnp.where(mask, frozen, 0.)
            return (jnp.where(mask, ce, 0.)*w).sum()/w.sum()
        natural = jnp.where(v, ce, 0.).sum()/v.sum()
        return .8*natural+.1*(mean(v & (y == 1))+mean(v & (y == 0)))
    grad = jax.grad(lambda x: dynamic(x)[0])(x)
    reference = jax.grad(surrogate)(x)
    np.testing.assert_allclose(grad, reference, rtol=2e-6, atol=1e-7)
    assert np.isfinite(np.asarray(grad)).all() and not np.asarray(grad)[0, -1].any()
    _, static = dynamic(x, 0.)
    assert m['tutor/pair_loss'] == static['tutor/pair_loss']
    assert m['tutor/natural_ce'] == static['tutor/natural_ce']
    write_json(CHECKS/'cpu_dynamic_math.json', signed(dict(status='PASS_DYNAMIC_TUTOR_NUMERIC_AND_STOPGRAD',
        hard_balanced_ce=float(m['tutor/hard_balanced_ce']), bc_mix=float(mix),
        weight_mean=float(m['tutor/difficulty_weight_mean']), weight_max=float(m['tutor/difficulty_weight_max']),
        gradient_vs_frozen_weights_max_error=float(np.max(np.abs(np.asarray(grad-reference)))),
        natural_bc_and_uniform_pair_unchanged=True, invalid_position_gradient_zero=True)))


def test_missing_class_empty_and_static_v0_equivalence():
    old = types.ModuleType('static_tutor')
    source = subprocess.check_output(['git', 'show',
        '33636c0593392b4f6ff203109ce41bfc831225a9:dreamerv3/fg_s3_tutor_loss.py'], cwd=PROJECT.parent/'dreamerv3')
    exec(compile(source, '<static V0>', 'exec'), old.__dict__)
    x = jnp.array([[[0., -.3], [.4, .2], [-.5, .8]]]); first = jnp.zeros((1, 3), bool)
    for y, v in [(jnp.array([[1, 0, 0]]), jnp.ones((1, 3), bool)),
                 (jnp.ones((1, 3), jnp.int32), jnp.ones((1, 3), bool)),
                 (jnp.zeros((1, 3), jnp.int32), jnp.zeros((1, 3), bool))]:
        nll, logits = distribution(x, y, v)
        current, m = tutor_objective(nll, logits, y, v, first)
        previous, pm = old.tutor_objective(nll, logits, y, v, first)
        np.testing.assert_array_equal(current, previous)
        for key in pm: np.testing.assert_array_equal(m[key], pm[key], err_msg=key)
        if not (np.any(np.asarray(v & (y == 1))) and np.any(np.asarray(v & (y == 0)))):
            dynamic, dm = tutor_objective(nll, logits, y, v, first, hard_kappa=1.)
            assert dynamic == m['tutor/natural_ce'] and not dm['tutor/hard_balanced_active']
            assert dm['tutor/pair_loss'] == 0
    assert agent_extensions({'mode': 'tutor', 'tutor_hard_kappa': 1})['tutor_hard_kappa'] == 1
    with pytest.raises(ValueError): agent_extensions({'mode': 'tutor', 'tutor_hard_kappa': -1})


def test_actual_e2_gpu_10k_restore_and_bounded_teaching():
    """Inspect real GPU results only, including bounded Replay retention."""
    root = PROJECT/'artifacts/fg_s3_v2/runs/TUTOR_CD_E2_GPU_TRAIN_10K_v1'
    path = root/'training_receipt.json'
    if not path.is_file() or read_manifest(path)['env_steps'] != 10000:
        pytest.skip('Needs the completed actual GPU E2 10K start/exit/resume run')
    receipt = read_manifest(path)
    identity = read_manifest(root/'run_identity.json')['identity']
    assert identity['agent_extension'] == agent_extensions({'mode': 'tutor', 'tutor_hard_kappa': 1.})
    def load(node, name):
        return pickle.loads((node/(name+'.pkl')).read_bytes())
    def frozen(k):
        return k.startswith(('val/', 'slowval/', 'slowval_count/', 'retnorm/', 'valnorm/', 'advnorm/')) or (
            k.startswith('opt/state/') and '/val/' in k)
    nodes = [root/'ckpt'/f'step_{n:012d}' for n in (0, 5000, 10000)]
    hashes = []
    for n, updates in zip(nodes, (0, 200, 400)):
        validate_bundle(n, identity)
        a = load(n, 'agent')
        assert a['counters']['updates'] == 21183+updates
        c = load(n, 'v2_counters')
        assert c['all_updates'] == c['wm_updates'] == updates
        assert c['rl_actor_updates'] == c['rl_critic_updates'] == 0
        hashes.append(parameter_digests(a['params']))
    for before, after in zip(hashes, hashes[1:]):
        assert all(before[k] == after[k] for k in before if frozen(k))
        assert any(before[k] != after[k] for k in before if k.startswith('pol/'))
        assert any(before[k] != after[k] for k in before if k.startswith(('enc/', 'dyn/')))
    rows = [json.loads(x) for x in (root/'updates.jsonl').read_text().splitlines()]
    assert [r['local_update'] for r in rows] == list(range(1, 401))
    assert all(r['metrics']['opt/wm_grad_norm'] > 0 and r['metrics']['opt/critic_grad_norm'] == 0
               and r['metrics']['v2/rl_active'] == 0 for r in rows)
    active = [r for r in rows if r['metrics']['tutor/hard_balanced_active']]
    assert active and all(1 <= r['metrics']['tutor/difficulty_weight_mean']
                         <= r['metrics']['tutor/difficulty_weight_max'] <= 2 for r in active)
    assert all(r['metrics']['opt/actor_grad_norm'] > 0 for r in active)
    pairable = sum(r['metrics']['tutor/pair_count'] > 0 for r in rows)
    assert pairable and receipt['counters']['pairable_batches'] == pairable
    boundary, final = [load(n, 'collector') for n in nodes[1:]]
    journal = [json.loads(x) for x in (root/'rollout/teacher_journal.jsonl').read_text().splitlines()]
    assert len(journal) == final['commit_seq']
    resumed = journal[boundary['commit_seq']]
    for k in ('state_id', 'actual_observation_id', 'executed_action', 'oracle_proof_hash', 'policy_model_update'):
        assert resumed[k] == boundary['pending'][k]
    assert final['pending']['stepid'] is None and final['pending_native'] is not None
    native = {}
    for shard in load(nodes[-1], 'replay')['chunks']:
        with np.load(nodes[-1]/'replay'/shard['name'], allow_pickle=False) as data:
            assert not any(k.startswith(('teacher', 'oracle', 'tutor', 'difficulty', 'pair')) for k in data.files)
            native.update((s.tobytes().hex(), int(a)) for s, a in zip(data['stepid'], data['action']))
    assert set(native) == {r['stepid'] for r in final['records']}
    assert all(native[r['stepid']] == r['executed_action'] for r in final['records'])
    assert len(final['records']) < len(journal)  # Actual bounded retention, not a duplicated teacher bank.
    spec = read_manifest(root/'source_admission.json')
    source = Path(spec['origin']['path'])
    assert file_hash(source/'manifest.json') == spec['origin']['manifest_sha256']
    assert file_hash(source/'agent.pkl') == json.loads((source/'manifest.json').read_text())['files']['agent.pkl']
    assert hashes[0] == read_manifest(root/'source_transfer.json')['params_after']
    assert all('cuda' in d for d in receipt['devices'])
    write_json(CHECKS/'gpu_10k_verification.json', signed(dict(status='PASS_ACTUAL_E2_GPU_10K_AND_RESUME',
        run=str(root), frozen_versions=identity['versions'], env_steps=10000, updates=400, resumed_updates=200,
        checkpoint=str(nodes[-1]), checkpoint_manifest_sha256=file_hash(nodes[-1]/'manifest.json'),
        dynamic_weighted_updates=len(active), weights_bounded=True, pairable_updates=pairable,
        pairable_rate=pairable/400, pair_count=receipt['counters']['pair_count'],
        actor_and_wm_continue_learning=True, critic_rl_optimizer_frozen=True,
        native_replay_sidecar_pending_consistent=True, retained_teacher_rows=len(final['records']),
        committed_teacher_rows=len(journal), source_unchanged=True, formal_1m='NOT_RUN',
        scope='actual GPU FG201 10K research diagnostic; not production/trading-success acceptance')))

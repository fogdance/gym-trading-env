"""Batch-local bounded curriculum: numerical, gradient and static compatibility."""
from pathlib import Path
import subprocess
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embodied.jax.outs import MaskedCategorical
from dreamerv3.fg_s3_tutor_loss import agent_extensions, tutor_objective
from gym_trading_env.research.fg_replay import signed, write_json


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

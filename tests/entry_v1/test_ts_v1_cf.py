"""S3 counterfactual teacher (gym ts_v1_cf) on the synthetic native market.

The frozen policy here is a deterministic causal rule of the market minute
and the account inputs, so a coupled return can be compared with the same
branch run to the true terminal.
"""
import numpy as np
import pytest

from gym_trading_env.research import ts_v1_cf as CF
from gym_trading_env.research import ts_v1_corpus as C
from gym_trading_env.research.ts_v1_account import StudentEnv
from wp02_fixture import ts_corpus


def rule(cursor, obs):
    """ADD at every 7th market minute when the Native mask allows it."""
    return int(cursor % 7 == 0 and obs['action_mask'][1] > 0.5)


class Stepper:

    def __init__(self, env, obs):
        self.env, self.obs = env, obs

    def state(self):
        return dict(cursor=int(self.env.current_step), equity=float(self.env._calculate_equity()),
                    **{k: np.array(self.obs[k]) for k in CF.INPUTS})

    def step(self, action=None):
        if action is None:
            action = rule(int(self.env.current_step), self.obs)
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        return float(reward), bool(terminated or truncated), bool(terminated)


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    folder, _ = ts_corpus(tmp_path_factory.mktemp('ts_cf'))
    env = StudentEnv(C.corpus_env(folder))
    run = Stepper(env, env.reset(seed=0)[0])
    rows, snapshots, done = [], {}, False
    while not done:
        state = run.state()
        snapshots[state['cursor']] = env.export_state()
        action = rule(state['cursor'], run.obs)
        reward, done, terminal = run.step(action)
        rows.append(dict(state, action=action, reward=reward))
    branch_env = StudentEnv(C.corpus_env(folder))
    return dict(env=env, rows=rows, snapshots=snapshots, terminal=terminal,
                floor=CF.coupling_floor(env), branch_env=branch_env)


def reference(world, floor=None, terminal=None):
    rows = world['rows']
    return CF.Reference(
        world['env'].market_clock_counter, [r['cursor'] for r in rows],
        [r['reward'] for r in rows], [r['equity'] for r in rows],
        {k: np.stack([r[k] for r in rows]) for k in CF.INPUTS}, [r['action'] for r in rows],
        world['terminal'] if terminal is None else terminal,
        world['floor'] if floor is None else floor)


def branch(world, ref, t, **kw):
    env = world['branch_env']
    obs, _ = env.reset(seed=0, options={'resume': world['snapshots'][t]})
    other = 1 - int(ref.action[ref.row(t)])
    return CF.run_branch(Stepper(env, obs), ref, other, **kw)


def test_reference_returns_are_the_discounted_native_rewards(world):
    ref = reference(world)
    assert world['terminal'] and ref.complete and len(ref.forks()) > 10
    assert 0 < world['floor'] < ref.equity.min()
    clock, g = world['env'].market_clock_counter, float(CF.GAMMA)
    rows = world['rows']
    for t in (0, 37, len(rows) - 2):
        direct = sum(g ** clock.reward_exponent(t, r['cursor'] + 1) * r['reward'] for r in rows[t:])
        assert ref.g[t] == pytest.approx(direct, rel=1e-12, abs=1e-12)


def test_a_coupled_return_equals_the_branch_run_to_the_true_terminal(world):
    ref, never = reference(world), reference(world, floor=np.inf)
    reasons, actions = set(), set()
    for t in ref.forks():
        coupled = branch(world, ref, t)
        full = branch(world, never, t)
        assert coupled['status'] == full['status'] == 'VALID_COMPLETE'
        assert full['reason'] == 'TERMINAL' and full['steps'] == ref.n - ref.row(t)
        assert coupled['g'] == pytest.approx(full['g'], rel=1e-9, abs=1e-12)
        reasons.add(coupled['reason'])
        actions.add(coupled['first_action'])
        p = CF.pair(ref, coupled)
        assert p['delta_g'] == pytest.approx(p['g_add'] - p['g_wait'])
        assert p['g_add' if p['reference_action'] else 'g_wait'] == ref.g[ref.row(t)]
    # Both first actions are taught, and coupling shortens real branches.
    assert actions == {CF.WAIT, CF.ADD} and 'COUPLED' in reasons
    long = [b for b in (branch(world, ref, t) for t in ref.forks())
            if b['reason'] == 'COUPLED' and b['steps'] > 1 and b['branch_rewards'] != 0]
    assert long


def test_a_branch_without_coupling_or_terminal_is_not_comparable(world):
    ref = reference(world)
    capped = [b for b in (branch(world, ref, t, max_steps=1) for t in ref.forks())
              if b['status'] == 'NOT_COMPARABLE']
    assert capped and all(b['reason'] == 'CAP' and b['g'] is None for b in capped)
    p = CF.pair(ref, capped[0])
    assert p['status'] == 'NOT_COMPARABLE' and p['delta_g'] is None
    # A reference without the true terminal has no complete return to couple to.
    open_ref = reference(world, terminal=False)
    t = int(ref.forks()[0])
    out = branch(world, open_ref, t)
    assert out['status'] == 'NOT_COMPARABLE' and out['reason'] == 'REFERENCE_INCOMPLETE'


def test_coupling_needs_the_policy_witness_and_a_fork_state(world):
    ref = reference(world)
    t = next(int(t) for t in ref.forks() if branch(world, ref, t)['reason'] == 'COUPLED')
    seen = []
    out = branch(world, ref, t, witness=lambda c: seen.append(c) or True)
    assert seen == [out['coupled_at']]
    with pytest.raises(RuntimeError, match='decides differently'):
        branch(world, ref, t, witness=lambda c: False)
    # A branch must start from the reference's own pre-decision state: one
    # forced step later the account no longer is the reference's.
    env = world['branch_env']
    run = Stepper(env, env.reset(seed=0, options={'resume': world['snapshots'][t]})[0])
    run.step(1 - int(ref.action[ref.row(t)]))
    with pytest.raises(ValueError, match='legal flat decision'):
        CF.run_branch(run, ref, CF.ADD)


def test_only_a_full_branch_decides_a_fork_that_cannot_couple_or_end_in_the_cap(world):
    ref, never = reference(world), reference(world, floor=np.inf)
    end, forks = ref.start + ref.n, [int(t) for t in ref.forks()]
    # Above the floor a branch may couple from the first cursor on; with an
    # unreachable floor only the true terminal within the cap settles it.
    assert ref.couple_from == ref.start and never.couple_from == end
    for cap in (1, 50):
        assert all(ref.decidable(t, cap) for t in forks)
        assert [t for t in forks if not never.decidable(t, cap)] == [t for t in forks if end - t > cap]
    # A floor first reached part way (a rising equity path): coupling is
    # possible from there on.
    rows = world['rows']
    mid = CF.Reference(world['env'].market_clock_counter, [r['cursor'] for r in rows],
                       [r['reward'] for r in rows], np.linspace(1., 2., len(rows)),
                       {k: np.stack([r[k] for r in rows]) for k in CF.INPUTS},
                       [r['action'] for r in rows], world['terminal'], 1.5)
    assert ref.start < mid.couple_from < end
    assert mid.future_min_equity[mid.couple_from - mid.start - 1] < 1.5 <= mid.future_min_equity[
        mid.couple_from - mid.start]
    for t in forks:
        assert mid.decidable(t, 50) == (mid.couple_from <= t + 50 or end - t <= 50)
    # Here the full branches of undecided forks run into the cap (the
    # synthetic account is far from a margin call). Not computed within a
    # budget, a fork records no reward, return or label.
    undecided = [t for t in forks if not never.decidable(t, 50)]
    assert len(undecided) > 5
    for t in undecided[:5]:
        full = branch(world, never, t, max_steps=50)
        assert (full['status'], full['reason'], full['steps']) == ('NOT_COMPARABLE', 'CAP', 50)
    t = undecided[0]
    out = CF.unavailable(never, t, 1 - int(never.action[never.row(t)]), 'BELOW_FLOOR_BEYOND_CAP')
    assert out['status'] == 'UNAVAILABLE_BUDGET' and out['status'] in CF.STATUSES
    assert out['steps'] == 0 and out['g'] is None and out['branch_rewards'] is None
    p = CF.pair(never, out)
    assert p['delta_g'] is None and p[('g_add', 'g_wait')[p['reference_action']]] is None
    assert p[('g_wait', 'g_add')[p['reference_action']]] == never.g[never.row(t)]

"""Public JSON ownership and native private fork equivalence."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.utils.account_fingerprint import capture_account_fingerprint, account_fingerprint
from gym_trading_env.utils.native_account import NativeHistory, native_owner, native_source
from wp02_fixture import native_factory


def test_private_restore_matches_public_ledger_reward_rng_observation_and_future_actions(tmp_path):
    factory, _, _ = native_factory(tmp_path/'market', minutes=45, chunk=7)
    source, public, private, foreign = (factory() for _ in range(4))
    try:
        # A long initial prefix plus a native tail, sealed on the private path.
        row = copy.deepcopy(source.equity_history[0])
        source.equity_history[:0] = [copy.deepcopy(row) for _ in range(600)]
        for _ in range(5): source.step(0)
        state = source.export_state()
        digest, view = capture_account_fingerprint(source, state)
        assert digest == fingerprint(state)
        public.import_state(state)
        private._restore_private_state(view, source_env=source)
        assert isinstance(private.equity_history, NativeHistory)
        assert private.equity_history._blocks[0] is view.history.blocks[0]
        assert private.export_state() == public.export_state() == state
        # Private past writes reject; public caller-owned JSON stays mutable.
        with pytest.raises(TypeError): private.equity_history[0]['equity'] = 0
        with pytest.raises(TypeError): private.equity_history[0] = row
        with pytest.raises(TypeError): del private.equity_history[0]
        public_state = private.export_state()
        public_state['equity_history'][0]['equity'] = 0
        public_state['rng'].clear()
        assert private.export_state() == state
        with pytest.raises(ValueError, match='source/owner'):
            private._restore_private_state(view, source_env=foreign)
        with pytest.raises(ValueError, match='source/owner'):
            private._restore_private_state(replace(view, source='f'*64), source_env=source)
        with pytest.raises(ValueError, match='cursor'):
            private._restore_private_state(replace(view, cursor=view.cursor+1), source_env=source)
        for _ in range(25):
            if public.truncated: public.reset(); private.reset()
            action = int(bool(public._published_mask[1]) and public.current_step % 7 == 0)
            a, b = public.step(action), private.step(action)
            for key in a[0]: np.testing.assert_array_equal(a[0][key], b[0][key])
            assert a[1:4] == b[1:4]
            assert public.export_state() == private.export_state()
            assert account_fingerprint(private) == fingerprint(public.export_state())
            assert source.export_state() == state and account_fingerprint(source) == digest
        assert public.fills and public.reward_events
        # Version stays immutable after the clone appends or restores backwards.
        private._restore_private_state(view, source_env=source)
        assert private.export_state() == state
    finally:
        for env in (source, public, private, foreign): env.close()


def test_capture_recertifies_mutated_signed_zero_boolean_replaced_deleted_and_rewound_public_prefix(tmp_path):
    factory, _, _ = native_factory(tmp_path/'market', minutes=20)
    env, clone = factory(), factory()
    try:
        for _ in range(5): env.step(0)
        original = env.export_state()
        original['equity_history'][0]['unrealized_pnl'] = -0.
        env.import_state(original)
        first, view = capture_account_fingerprint(env, env.export_state())
        old_commit = view.history.commitment
        for mutation in ('zero', 'bool', 'replace', 'delete', 'rewind'):
            state = copy.deepcopy(original)
            if mutation == 'zero': state['equity_history'][0]['unrealized_pnl'] = 0.
            if mutation == 'bool': state['equity_history'][0]['intrabar'] = 0
            if mutation == 'replace': state['equity_history'][0] = {'nested': [True, 1, -0., {'x': [3]}]}
            if mutation == 'delete': del state['equity_history'][0]
            if mutation == 'rewind': state['equity_history'] = []
            env.import_state(state)
            digest, actual = capture_account_fingerprint(env, env.export_state())
            assert digest == fingerprint(state) and digest != first
            clone._restore_private_state(actual, source_env=env)
            assert clone.export_state() == state
        assert view.history.commitment == old_commit
        clone._restore_private_state(view, source_env=env)
        assert clone.export_state() == original
    finally: env.close(); clone.close()


def test_sealed_history_has_exact_original_json_bytes_after_many_blocks_forks_and_nested_mutations():
    rows = [dict(time='中文', equity=i+.5, unrealized_pnl=-0., net_lots=0., intrabar=False)
            for i in range(530)]
    rows.append(dict(extra=[True, 1, -0., {'nested': [2]}]))
    history = NativeHistory.from_public(rows, owner='account-A', source='source-A')
    old = history.version(); fork = NativeHistory.from_version(old)
    import json
    encoded = json.dumps(rows, sort_keys=True, default=str).encode()
    assert b''.join(old.json_parts()) == encoded
    rows[0]['equity'] = 0
    rows[-1]['extra'][-1]['nested'].clear()
    assert b''.join(old.json_parts()) == encoded
    with pytest.raises(TypeError): history[-1]['extra'][-1]['nested'][0] = 0
    fork.append({'new': 'clone-only'})
    assert len(fork) == len(history)+1 and history.version().commitment == old.commitment
    assert fork.version().commitment != old.commitment
    public = history.to_public(); public[-1]['extra'][-1]['nested'].clear()
    assert history.to_public()[-1]['extra'][-1]['nested'] == [2]

"""Public JSON ownership and native private fork equivalence."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.utils.account_fingerprint import capture_account_fingerprint, account_fingerprint
from gym_trading_env.utils.native_account import NativeHistory
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
        import json
        foreign_hot=json.loads(view.hot_json);foreign_hot['account_sequence']+=1
        with pytest.raises(ValueError,match='source/owner'):
            private._restore_private_state(replace(view,hot_json=json.dumps(foreign_hot).encode()),source_env=source)
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


def test_private_hash_is_full_public_json_sha_without_pickle_or_history_materialization(tmp_path,monkeypatch):
    import pickle
    factory, _, _ = native_factory(tmp_path/'market', minutes=20)
    env, clone = factory(), factory()
    try:
        state=env.export_state()
        state['equity_history']=[dict(time='中文',equity=i+.5,unrealized_pnl=-0.,
            net_lots=0.,intrabar=False) for i in range(530)]
        state['equity_history'].append(dict(extra=[True,1,-0.,{'nested':[2]}]))
        env.import_state(state)
        expected=fingerprint(state)
        _,view=capture_account_fingerprint(env,env.export_state())
        clone._restore_private_state(view,source_env=env)
        def forbidden(*a,**kw):raise AssertionError('Immutable hash re-materialized full history')
        with monkeypatch.context() as patch:
            patch.setattr(pickle,'dumps',forbidden)
            patch.setattr(NativeHistory,'to_public',forbidden)
            assert view.digest()==expected
            assert account_fingerprint(clone)==expected
            assert view.digest()==expected
        assert clone.export_state()==state
        # A changed hot field or appended row must produce a new complete SHA.
        clone.step(0)
        assert account_fingerprint(clone)==fingerprint(clone.export_state())!=expected
    finally:env.close();clone.close()


def test_worker_private_replica_rejects_reorder_replays_stream_reset_and_full_sha(tmp_path):
    from gym_trading_env.research import fg_s3_quality_label_pool as pool
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    factory, config, profile = native_factory(tmp_path/'market', minutes=25, chunk=7)
    env = factory()
    source = dict(fold='TRAIN', state_cache_entries=0)
    teacher = NativeQualityTeacher(env, factory, fold='TRAIN', source=source)
    initial = env.export_state()
    pool._initialize(dict(env_config=str(config)), profile, initial, source, None, teacher.identity_hash)
    try:
        state = copy.deepcopy(initial); count = len(state.pop('equity_history'))
        pool._begin_stream()
        with pytest.raises(ValueError, match='missing or out of order'):
            pool._query(copy.deepcopy(state), [], env._published_mask, fingerprint(initial), count+1)
        first = pool._query(copy.deepcopy(state), [], env._published_mask, fingerprint(initial), count)
        assert pool._env.export_state() == initial
        env.step(0); updated = env.export_state()
        delta = updated['equity_history'][count:]
        hot = {k:v for k,v in updated.items() if k!='equity_history'}
        pool._query(copy.deepcopy(hot), delta, env._published_mask, fingerprint(updated), count)
        assert pool._env.export_state() == updated
        pool._begin_stream()
        assert pool._query(copy.deepcopy(state), [], pool._env._published_mask,
                           fingerprint(initial), count) == first
        with pytest.raises(ValueError, match='complete native account snapshot'):
            pool._query(copy.deepcopy(hot), delta, env._published_mask, 'f'*64, count)
    finally:
        pool._teacher.close(); pool._env.close(); teacher.close(); env.close()


def test_private_driver_opt_in_survives_public_restore_chunk_and_new_account_epoch(tmp_path):
    factory, _, _ = native_factory(tmp_path/'market', minutes=20, chunk=3)
    public, private, clone = factory(), factory(), factory()
    try:
        public.reset(seed=0);private.reset(seed=0)
        assert type(public.equity_history) is list
        private._enable_private_history()
        old=private._export_private_state();owner=old.owner
        while not public.terminated:
            if public.truncated:
                saved=public.export_state()
                public.reset(options={'resume':saved});private.reset(options={'resume':saved})
                assert isinstance(private.equity_history,NativeHistory)
                assert private.export_state()==public.export_state()==saved
                public.reset();private.reset()  # continue the restored CHUNK_END
            action=int(bool(public._published_mask[1]))
            a,b=public.step(action),private.step(action)
            assert a[1:4]==b[1:4] and public.export_state()==private.export_state()
            assert account_fingerprint(private)==fingerprint(public.export_state())
        public.reset();private.reset()
        assert private._export_private_state().owner!=owner
        assert private.export_state()==public.export_state()
        with pytest.raises(ValueError,match='source/owner'):
            clone._restore_private_state(old,source_env=private)
        assert type(public.equity_history) is list
        rows=private.equity_history.public_slice(0);rows[0]['equity']=0
        assert private.export_state()['equity_history'][0]['equity']!=0
    finally:public.close();private.close();clone.close()

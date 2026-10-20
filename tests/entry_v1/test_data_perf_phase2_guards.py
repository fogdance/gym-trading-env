"""Explicit DATA guard boundaries; controlled synthetic receipts, never admission."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gym_trading_env.research.fg_replay import read_manifest, signed, write_json, digest
from gym_trading_env.research import fg_s3_e_quality_supervision as supervision
from wp02_fixture import native_factory
from wp03_fixture import ROOT, metadata_corpus, inventory


def test_native_admission_rejects_block_source_monitoring_before_verifier_launch(tmp_path):
    root=metadata_corpus(tmp_path/'corpus')
    manifest=read_manifest(root/'corpus_manifest.json')
    manifest['source_binding']=dict(source_guard_interval=1024)
    inventory(root,manifest)
    source=supervision.SupervisionSource.from_corpus(root)
    # Content may be inspected for diagnosis; it cannot become DATA ready.
    assert supervision.QualitySupervisionAdmission.preflight(source).receipt['training_admitted'] is False
    log=tmp_path/'not_launched.log'
    with pytest.raises(ValueError,match='PER_QUERY_SOURCE_GUARD_REQUIRED'):
        supervision.QualitySupervisionAdmission.preflight(source,purpose='native-verification',
            producer_gym=ROOT,log_path=log)
    assert not log.exists()


@pytest.mark.parametrize('attack', ['none', 'missing', 'partial', 'failed', 'foreign_trace'])
def test_native_receipt_requires_every_label_of_every_authenticated_trace(tmp_path, monkeypatch, attack):
    root = metadata_corpus(tmp_path/'corpus')
    receipts = []
    for number, policy in enumerate(supervision.POLICIES):
        trace_path = root/'policies'/policy/'manifest.json'
        trace = read_manifest(trace_path)
        trace.pop('content_hash')
        trace.update(records=number+1, replay_trace_hash=str(number+1)*64)
        write_json(trace_path, signed(trace))
        receipts.append(dict(status='PASS', records=number+1,
            teacher_labels_verified=number+1, replay_trace_hash=trace['replay_trace_hash']))
    inventory(root, read_manifest(root/'corpus_manifest.json'))
    source = supervision.SupervisionSource.from_corpus(root)
    result = dict(status='PASS', G3='GO', training_admitted=False,
        verification_state='NATIVE_G3_VERIFIED', corpus_identity=source.corpus_content_hash,
        native_replay=receipts)
    if attack == 'missing': result.pop('native_replay')
    if attack == 'partial': receipts[3]['teacher_labels_verified'] -= 1
    if attack == 'failed': receipts[3]['status'] = 'FAIL'
    if attack == 'foreign_trace': receipts[3]['replay_trace_hash'] = 'f'*64
    monkeypatch.setattr(supervision, 'frozen_producer_audit', lambda *a, **kw: {'synthetic': True})
    def run(argv, **kwargs):
        kwargs['stdout'].write(json.dumps(result)+'\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(supervision.subprocess, 'run', run)
    if attack == 'none':
        assert supervision.native_verification(source, producer_gym=ROOT,
            log_path=tmp_path/'receipt.log')['result'] == result
    else:
        with pytest.raises(ValueError, match='INCOMPLETE_LABEL_VERIFICATION'):
            supervision.native_verification(source, producer_gym=ROOT, log_path=tmp_path/'receipt.log')


def test_per_query_source_guard_and_warm_cache_in_place_market_mutation(tmp_path):
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher
    marker = tmp_path/'source_identity'; marker.write_text('original')
    factory, _, _ = native_factory(tmp_path/'market', minutes=40, chunk=7)
    env = factory()
    teacher = NativeQualityTeacher(env, factory, fold='TRAIN', source=dict(
        fold='TRAIN', files={str(marker): digest(marker)}, state_cache_entries=0))
    try:
        teacher.query_state(env, env._published_mask)
        assert teacher.clone._native_execution_cache.statistics['row_misses'] > 0
        # Mutate the SAME source object after warming quotes and rows.
        raw = env._raw
        env._raw.iloc[0, env._raw.columns.get_loc('Volume')] += 1
        assert env._raw is raw
        with pytest.raises(ValueError, match='source/state identity mismatch'):
            teacher.query_state(env, env._published_mask)
        env._raw.iloc[0, env._raw.columns.get_loc('Volume')] -= 1
        marker.write_text('changed')
        with pytest.raises(ValueError, match='Frozen quality source changed'):
            teacher.query_state(env, env._published_mask)
    finally:
        with pytest.raises(ValueError, match='Frozen quality source changed'):
            teacher.close()
        env.close()


def test_native_e_direct_trace_matches_fixed_old_observed_wrapper_every_step(tmp_path, monkeypatch):
    import ast
    import subprocess
    from gym_trading_env.research import fg_replay
    from gym_trading_env.research.fg_s3_quality_teacher_v2 import trace_native_trade
    # Bind the independent reference to the actual pre-optimization Git object.
    sha = 'b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46'
    text = subprocess.check_output(['git', 'show', sha+':src/gym_trading_env/research/fg_replay.py'],
                                   cwd=ROOT, text=True)
    tree = ast.parse(text)
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'execute_observed')
    scope = dict(vars(fg_replay))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<old_observed_'+sha+'>', 'exec'), scope)
    old_observed = scope['execute_observed']
    factory, _, _ = native_factory(tmp_path/'market', minutes=45, chunk=3)
    source, direct, wrapped = factory(), factory(), factory()
    original_step = type(direct).step
    seen = []
    def checked_step(self, action):
        if self is not direct: return original_step(self, action)
        if wrapped.truncated and wrapped.status == 'CHUNK_END': wrapped.reset()
        obs, row = old_observed(wrapped, action)
        result = original_step(self, action)
        for key in obs: np.testing.assert_array_equal(result[0][key], obs[key])
        assert float(np.float32(result[1])) == float(np.float32(row['official_reward']))
        assert result[2:4] == (row['is_terminal'], row['is_last'] and not row['is_terminal'])
        assert self.export_state() == wrapped.export_state()
        seen.append(dict(cursor=self.current_step, fills=copy.deepcopy(row['fills']),
                         rewards=copy.deepcopy(row['reward_events']), status=row['status']))
        return result
    monkeypatch.setattr(type(direct), 'step', checked_step)
    try:
        for cursor in (0, 8, 39):
            while source.current_step < cursor:
                if source.truncated: source.reset()
                source.step(0)
            if source.truncated: source.reset()
            state = source.export_state(); wrapped.import_state(state)
            # Budget gives a genuine censored trace alongside complete TP/SL.
            result = trace_native_trade(source, direct, state=state,
                max_market_steps=2 if cursor == 39 else None)
            assert source.export_state() == state
            assert result['final_state_hash'] == fg_replay.fingerprint(wrapped.export_state())
        assert len(seen) > 3 and any(x['fills'] for x in seen)
        assert any(x['status'] == 'CHUNK_END' for x in seen)
    finally:
        source.close(); direct.close(); wrapped.close()


def test_exact_dp_against_fixed_old_scalar_across_release_invalid_minute_and_sessions(tmp_path):
    import ast
    import subprocess
    from fractions import Fraction
    import pandas as pd
    from gym_trading_env.research.fg_s3_e_quality_strict import QualityStrictTeacher, GAMMA
    from gym_trading_env.research.fg_s3_exact_witness import encode
    sha = 'b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46'
    text = subprocess.check_output(['git', 'show', sha+
        ':src/gym_trading_env/research/fg_s3_e_quality_strict.py'], cwd=ROOT, text=True)
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'QualityStrictTeacher')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_value')
    scope = dict(Fraction=Fraction, GAMMA=GAMMA)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<old_scalar_'+sha+'>', 'exec'), scope)
    factory, _, profile = native_factory(tmp_path/'market', minutes=365, chunk=7)
    path = tmp_path/'market'/'market.csv'
    frame = pd.read_csv(path)
    frame['bob'] = (pd.to_datetime(frame.Date)-pd.Timedelta(minutes=1)).astype(str)
    frame.loc[194, 'bob'] = frame.Date.iloc[194]  # real session-closing non-minute marker
    frame.to_csv(path, index=False)
    profile.source_receipt['sources']['CZCE.FG2701']['market']['sha256'] = digest(path)
    env = factory()
    teacher = QualityStrictTeacher(env, source=dict(fold='TRAIN'))
    old = SimpleNamespace(query=teacher.query, clock=teacher.clock,
                          _values={teacher.query.end: Fraction(0)})
    scalar = lambda cursor: scope['_value'](old, cursor)
    try:
        assert not env.valid_fill_clock[194]
        assert len(set(env._raw.trading_date)) == 2
        roots = set(range(teacher.query.end))
        expected = {cursor: scalar(cursor) for cursor in roots}
        for cursor in roots: assert teacher._value(cursor) == expected[cursor]
        for cursor, row in teacher.query.candidates.items():
            if not row.get('filled') or row.get('right_censored'): continue
            wait = GAMMA**teacher.clock.advance(cursor, cursor+1)*scalar(cursor+1)
            add = sum((GAMMA**teacher.clock.reward_exponent(cursor, s)*Fraction.from_float(v)
                for s, v in row['rewards']['C1']), Fraction(0))
            add += GAMMA**teacher.clock.advance(cursor, row['release_cursor'])*scalar(row['release_cursor'])
            pair = teacher._numeric_pair(cursor, row)
            assert pair['value_rational_witness'] == [encode(wait), encode(add)]
            assert pair['strict_action'] == int(add > wait) and pair['tie'] == (add == wait)
        teacher.compile_value_store(tmp_path/'exact_values')
        for cursor in roots: assert teacher._value(cursor) == expected[cursor]
    finally:
        if teacher._value_store is not None: teacher._value_store.close()
        env.close()

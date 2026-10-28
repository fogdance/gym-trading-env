"""Graded entry teacher (graded/) on the synthetic Native market."""
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from wp02_fixture import GRADES, ts_corpus

from gym_trading_env.research import ts_v1_corpus as C
from gym_trading_env.research import ts_v1_grades as T
from gym_trading_env.research.fg_s3_quality_teacher_v2 import official_day_index
from gym_trading_env.research.fg_single3_oracle import CandidateIndex, ThresholdIndex

# Thresholds that give the synthetic trades every grade, with a shorter room
# window: held 1-3 / 4-8 / 9-12, mae_full 1 / 15, room 8.
VARIED = dict(time_minutes=[3, 8], mae_points=[3, 5, 20], room_points=[8, 7, 6],
              room_after_tp_minutes=30)


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    # Two official trading days: windows cut at the day end and a TP on the next day.
    folder, _ = ts_corpus(tmp_path_factory.mktemp('ts_grades'), minutes=400)
    env = C.corpus_env(folder, market=False)
    return dict(folder=folder, env=env, facts=C.load_arrays(folder / 'facts'),
                graded=C.load_arrays(folder / 'graded'), days=official_day_index(env)['days'])


def walk(env, days, i, minutes, after=None):
    """(room, last cursor, complete) of the window of `minutes` market minutes
    from the fill i, or after the cursor `after`, cut at the end of the fill's
    trading day: walked minute by minute on the quotes env.step takes profit
    at (the fill minute's close when 3 lots of volume remain, then the high of
    a valid minute with 3 lots)."""
    raw, valid = env._raw, env.valid_fill_clock
    side = env.direction_at(env.bobs[i], row_index=i)
    sign = 1 if side == 'LONG' else -1
    price = float(env._price(raw.Open.iloc[i], side, env.bobs[i], opening=True))
    volume, close = raw.Volume.to_numpy(), raw.Close.to_numpy()
    extreme = raw['High' if sign == 1 else 'Low'].to_numpy()
    best, count, c = -np.inf, 0, i
    while c < len(raw) and days[c] == days[i]:
        if valid[c] and volume[c] >= (6 if c == i else 3):
            quote = env._price(close[c] if c == i else extreme[c], side, raw.index[c])
            best = max(best, sign * float(quote))
        if valid[c] and (after is None or c > after):
            count += 1
            if count == minutes:
                return max(0., best - sign * price), c, True
        c += 1
    return max(0., best - sign * price), c - 1, c < len(raw)


def expected_grade(kind, crossed, held, mae, room, complete, grades):
    if kind in ('SL30', 'OTHER_EXIT') or (kind == 'TP5' and crossed):
        return T.POOR
    if kind != 'TP5' or not complete:
        return -1
    (t1, t2), (m1, m2, m3), (r1, r2, r3) = (grades[k] for k in T.KEYS[:3])
    time = 3 if held <= t1 else 2 if held <= t2 else 1
    pullback = 3 if mae <= m1 else 2 if mae <= m2 else 1 if mae <= m3 else 0
    space = 3 if room >= r1 else 2 if room >= r2 else 1 if room >= r3 else 0
    return min(time, pullback, space)


def test_every_fill_is_the_native_entry(corpus):
    facts, graded = corpus['facts'], corpus['graded']
    index = CandidateIndex(C.corpus_env(corpus['folder'], market=False))  # it steps its env
    filled = np.flatnonzero(facts['filled'])
    assert len(filled) > 300
    for t in filled:
        native = index.candidate(int(t))
        assert native['entry_cursor'] == facts['entry_cursor'][t]
        assert graded['entry_price'][t] == float(native['entry_price'])
        assert graded['sign'][t] == (1 if native['side'] == 'LONG' else -1)
    assert np.isnan(graded['entry_price'][~facts['filled']]).all()


def test_mfe_and_room_windows_walked_minute_by_minute(corpus):
    env, facts, graded, days = (corpus[k] for k in ('env', 'facts', 'graded', 'days'))
    cut = 0
    for t in np.flatnonzero(facts['filled']):
        i, exit_at = int(facts['entry_cursor'][t]), int(facts['exit_cursor'][t])
        for h in T.MFE_MINUTES:
            room, end, complete = walk(env, days, i, h)
            assert graded[f'mfe_{h}'][t] == np.float32(room), (t, h)
            assert graded[f'mfe_{h}_complete'][t] == complete, (t, h)
            cut += complete and days[end] == days[i] and end + 1 < len(days) and days[end + 1] != days[i]
        if T.KINDS[facts['kind'][t]] == 'TP5' and facts['crossed_days'][t] == 0:
            room, end, complete = walk(env, days, i, 60, after=exit_at)
            assert graded['room'][t] == np.float32(room) and graded['room_complete'][t] == complete
            assert room >= 5 + float(env.terms.tick)  # the TP is in the window
            assert graded['ready_at'][t] == (end if complete else -1)
        else:
            assert np.isnan(graded['room'][t])
    assert cut > 0  # some windows end at the end of the trading day, complete


def test_the_fill_minute_close_counts_only_with_volume_left_for_the_exit():
    # env.step closes on the fill minute at its close, from the volume left
    # after the 3 lots it opened; later minutes at the high (TP index).
    close, high = np.array([1012., 1000., 1003.]), np.array([0., 1004., 1006.])
    index = SimpleNamespace(quotes=dict(LONG=dict(close=close)),
                            indices=dict(LONG=dict(tp=ThresholdIndex(-high))))
    assert T._room(index, 'LONG', 1001., 0, 2, np.array([6, 3, 3])) == 11.
    assert T._room(index, 'LONG', 1001., 0, 2, np.array([5, 3, 3])) == 5.
    assert T._room(index, 'LONG', 1001., 0, 1, np.array([5, 3, 3])) == 3.
    assert T._room(index, 'LONG', 1001., 0, 0, np.array([5, 3, 3])) == 0.


def test_mfe_reaches_the_tp_exactly_when_the_trade_takes_profit_in_time(corpus):
    facts, graded = corpus['facts'], corpus['graded']
    tp = np.flatnonzero(facts['filled'] & (facts['kind'] == C.KINDS.index('TP5')))
    assert (facts['crossed_days'][tp] > 0).any()
    reach = 5 + float(corpus['env'].terms.tick)
    for h in T.MFE_MINUTES:
        in_time = (facts['crossed_days'][tp] == 0) & (facts['held_minutes'][tp] <= h)
        assert np.array_equal(graded[f'mfe_{h}'][tp] >= reach, in_time)


@pytest.mark.parametrize('grades', [GRADES, VARIED], ids=['formal', 'varied'])
def test_grades_are_the_worst_dimension_and_label_add_and_wait(corpus, grades):
    env, facts, days = corpus['env'], corpus['facts'], corpus['days']
    graded, summary = T.grade(env, facts, grades)
    seen = set()
    for t in np.flatnonzero(facts['filled']):
        kind, i = T.KINDS[facts['kind'][t]], int(facts['entry_cursor'][t])
        room, _, complete = walk(env, days, i, grades['room_after_tp_minutes'],
                                 after=int(facts['exit_cursor'][t]))
        want = expected_grade(kind, facts['crossed_days'][t] > 0, facts['held_minutes'][t],
                              facts['mae_full'][t], room, complete, grades)
        assert graded['grade'][t] == want, (t, kind)
        seen.add(want)
    assert seen >= {T.POOR, T.GOOD, -1} and (grades is GRADES or seen >= {T.FAIR, T.EXCELLENT})
    g = graded['grade']
    assert np.array_equal(graded['valid'], (g == T.POOR) | (g >= T.GOOD))
    assert np.array_equal(graded['action'][graded['valid']], (g[graded['valid']] >= T.GOOD))
    assert summary['demo'] == dict(add=int((g >= T.GOOD).sum()), wait=int((g == T.POOR).sum()))
    assert summary['tp_checked'] == int((facts['filled'] & (facts['kind'] == 0)).sum())


def test_a_censored_trade_has_no_grade_and_a_late_tp_is_refused(corpus):
    env, facts = corpus['env'], dict(corpus['facts'])
    tp = np.flatnonzero(facts['filled'] & (facts['kind'] == C.KINDS.index('TP5')))
    kind = facts['kind'].copy()
    kind[tp[:5]] = C.KINDS.index('RIGHT_CENSORED')
    graded, summary = T.grade(env, dict(facts, kind=kind), GRADES)
    assert (graded['grade'][tp[:5]] == -1).all() and not graded['valid'][tp[:5]].any()
    assert summary['ungraded_filled'] >= 5
    late = facts['exit_cursor'].copy()
    late[tp[0]] += 1
    with pytest.raises(ValueError, match='reaches TP first'):
        T.grade(env, dict(facts, exit_cursor=late), GRADES)


def test_thresholds_are_checked():
    assert T.check_grades(GRADES)['room_points'] == [10., 7.5, 6.]
    for bad in (dict(GRADES, extra=1), dict(GRADES, time_minutes=[120, 15]),
                dict(GRADES, room_points=[6, 7.5, 10]), dict(GRADES, mae_points=[3, 5]),
                dict(GRADES, room_after_tp_minutes=0)):
        with pytest.raises(ValueError):
            T.check_grades(bad)


def test_produce_writes_graded_and_regrading_needs_replace(corpus, tmp_path):
    folder = corpus['folder']
    summary = json.loads((folder / 'graded' / 'summary.json').read_text())
    assert summary['grades'] == T.check_grades(GRADES) and summary['label'] == C.LABELS['graded']
    assert summary['contract'] == 'CZCE.FG2701' and summary['mfe_minutes'] == [15, 60, 120]
    assert sum(summary['strict_vs_graded'].values()) == sum(summary['strict'].values())
    source = json.loads((folder / 'source.json').read_text())
    assert source['graded']['demo'] == summary['demo'] and 'graded' in source['labels']
    with pytest.raises(FileExistsError, match='--replace'):
        T.write(folder, GRADES)
    copy = tmp_path / folder.name
    shutil.copytree(folder, copy)
    path = tmp_path / 'grades.yaml'
    path.write_text(yaml.safe_dump(VARIED))
    T.main(['--grades', str(path), '--corpus', str(copy), '--replace', '--workers', '1'])
    regraded = C.load_arrays(copy / 'graded')
    want, _ = T.grade(corpus['env'], corpus['facts'], VARIED)
    assert set(regraded) == set(want)
    for key in want:
        assert np.array_equal(regraded[key], want[key], equal_nan=want[key].dtype.kind == 'f'), key
    assert json.loads((copy / 'graded' / 'summary.json').read_text())['grades'] == T.check_grades(VARIED)
    assert not (copy / 'graded.tmp').exists()

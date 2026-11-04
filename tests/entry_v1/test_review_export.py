"""review-v1 export (user 2026-11-04). The corpus teacher: the spec's two
FG205 records, the bars of lunch and night records, determinism, publishing
without overwriting, and validate's rejections; one export of the real V2.1
corpus into a temporary review/ serves these tests. The run part: the
student trades of pi_0000 and pi_0009 of the real V2.1 run, its manifest's
revisions and its fixed corpus, and the Native ledger checks."""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from gym_trading_env.research import review_export as X

ROOT = Path('/data/logdir/fg_single3/thinking_student/corpus_v2_1_20261104')
pytestmark = pytest.mark.skipif(not (ROOT / 'CZCE.FG205' / 'source.json').exists(),
                                reason='V2.1 corpus not present')
NOW = '2026-11-04T16:00:00+08:00'


@pytest.fixture(scope='module')
def review(tmp_path_factory):
    out = tmp_path_factory.mktemp('corpus') / 'review'
    result = X.export_corpus(ROOT, 'V2.1 teacher', 18, out=out, now=NOW, log=lambda m: None)
    assert result['manifest'] == 'published'
    return out


def records(review, symbol):
    return {r['id']: r for r in map(json.loads, (review / 'teacher' / f'{symbol}.jsonl').read_text().splitlines())}


def test_the_spec_records_of_fg205_and_the_totals(review):
    assert X.validate_review(review) == dict(files=18, records=12620, add=10110, wait=2510,
                                             excellent=3786, good=6324)
    fg205 = records(review, 'CZCE.FG205')
    add = fg205['teacher:CZCE.FG205:941']
    assert {k: add[k] for k in X.FIELDS if k not in ('details', 'note')} == dict(
        id='teacher:CZCE.FG205:941', symbol='CZCE.FG205', action='ADD', direction='SHORT',
        decision_time='2021-12-14T11:27:00+08:00', entry_time='2021-12-14T11:27:00+08:00',
        bar_bob='2021-12-14T11:27:00+08:00', bar_eob='2021-12-14T11:28:00+08:00', trading_day='2021-12-14',
        entry_price=1846, volume=3, stop_price=1855, target_price=1828, grade='EXCELLENT',
        label_ready_at='2021-12-14T21:01:00+08:00',
        outcome=dict(status='CLOSED', exit_time='2021-12-14T21:01:00+08:00', exit_price=1828,
                     exit_reason='TAKE_PROFIT', net_pnl=1062))
    assert {k: add['details'][k] for k in ('Positive ID', 'Prior high H', 'Touch', 'Risk R (points)',
                                           'Held (market minutes)', 'MAE (points)', 'MFE (points)')} == {
        'Positive ID': 'CZCE.FG205@941:A', 'Prior high H': 1850, 'Touch': 1, 'Risk R (points)': 9,
        'Held (market minutes)': 94, 'MAE (points)': 2, 'MFE (points)': 19}
    assert add['details']['Line spike RV'] == pytest.approx(3.165053622809312) and add['note'] == X.NOTE_ADD
    wait = fg205['teacher:CZCE.FG205:956']
    assert (wait['action'], wait['grade'], wait['entry_price'], wait['entry_time'], wait['outcome']) == (
        'WAIT', None, None, None, None)
    assert (wait['bar_bob'], wait['bar_eob'], wait['label_ready_at']) == (
        '2021-12-14T13:41:00+08:00', '2021-12-14T13:42:00+08:00', '2021-12-14T21:08:00+08:00')
    # Directions come from the corpus and its env, never from prices.
    fg001 = records(review, 'CZCE.FG001').values()
    assert {r['direction'] for r in fg001} == {'LONG'} and all('Prior low L' in r['details'] for r in fg001)
    assert {r['direction'] for r in fg205.values()} == {'SHORT'}


def test_an_add_is_marked_on_its_fill_bar_across_lunch_and_at_night(review):
    every = [r for f in sorted((review / 'teacher').glob('*.jsonl')) for r in records(review, f.stem).values()]
    lunch = [r for r in every if r['action'] == 'ADD' and r['decision_time'].endswith('11:30:00+08:00')]
    assert lunch and all((r['entry_time'][11:16], r['bar_bob'][11:16], r['bar_eob'][11:16]) == ('13:30', '13:30', '13:31')
                         for r in lunch)
    night = [r for r in every if r['bar_eob'][11:13] in ('21', '22')]
    assert night and all(r['trading_day'] > r['bar_eob'][:10] for r in night)
    # The trading day is the market CSV's own field.
    source = json.loads((ROOT / night[0]['symbol'] / 'source.json').read_text())['pool_row']['path']
    raw = pd.read_csv(source, usecols=['Date', 'trading_date'], dtype=str)
    eob = pd.Timestamp(night[0]['bar_eob']).strftime('%Y-%m-%d %H:%M:%S')
    assert raw.trading_date[raw.Date == eob].tolist() == [night[0]['trading_day']]


def test_a_rerun_is_byte_identical_and_a_changed_file_is_a_conflict(review, tmp_path):
    before = {p.name: p.read_bytes() for p in [review / 'manifest.json', *(review / 'teacher').glob('*')]}
    again = X.export_corpus(ROOT, 'V2.1 teacher', 18, out=review, log=lambda m: None)
    assert again['manifest'] == 'unchanged' and again['files'] == dict(published=0, reused=18)
    assert {p.name: p.read_bytes() for p in [review / 'manifest.json', *(review / 'teacher').glob('*')]} == before
    copy = tmp_path / 'review'
    shutil.copytree(review, copy)
    f = copy / 'teacher' / 'CZCE.FG205.jsonl'
    f.write_bytes(f.read_bytes().replace(b'"entry_price":1846', b'"entry_price":1847', 1))
    manifest = (copy / 'manifest.json').read_bytes()
    with pytest.raises(X.ConflictError, match='exists with other content'):
        X.export_corpus(ROOT, 'V2.1 teacher', 18, out=copy, log=lambda m: None)
    assert (copy / 'manifest.json').read_bytes() == manifest and not list(copy.rglob('*.tmp'))
    # Another label is another publication: refused, the manifest kept.
    with pytest.raises(X.ConflictError, match='a new teacher needs a new corpus id'):
        X.export_corpus(ROOT, 'V2.2 teacher', 18, out=review, log=lambda m: None)
    with pytest.raises(X.ReviewError, match='expected 17'):
        X.export_corpus(ROOT, 'V2.1 teacher', 17, out=tmp_path / 'short', log=lambda m: None)


def _broken(review, tmp_path, change):
    copy = tmp_path / 'review'
    shutil.copytree(review, copy)
    m = json.loads((copy / 'manifest.json').read_text())
    change(copy, m)
    (copy / 'manifest.json').write_text(json.dumps(m, ensure_ascii=False))
    return copy


def test_validate_rejects_the_schema_an_escaping_path_a_hash_and_nan(review, tmp_path):
    def schema(copy, m):
        m['schema'] = 'review-v0'

    def escape(copy, m):
        m['datasets'][0]['files'][0]['path'] = '../teacher/CZCE.FG001.jsonl'

    def tampered(copy, m):
        f = copy / 'teacher' / 'CZCE.FG001.jsonl'
        f.write_bytes(f.read_bytes().replace(b'"volume":3', b'"volume":4', 1))

    def nan(copy, m):
        f = copy / 'teacher' / 'CZCE.FG001.jsonl'
        data = f.read_bytes().replace(b'"volume":3', b'"volume":NaN', 1)
        f.write_bytes(data)
        m['datasets'][0]['files'][0]['sha256'] = hashlib.sha256(data).hexdigest()

    for i, (change, why) in enumerate([(schema, 'schema is not review-v1'), (escape, 'inside review/'),
                                       (tampered, 'sha256 differs'), (nan, 'NaN is not a JSON number')]):
        with pytest.raises(X.ReviewError, match=why):
            X.validate_review(_broken(review, tmp_path / str(i), change))


# ---- the run part: student trades of a formal run (user 2026-11-04) ----

RUN = Path('/data/logdir/fg_single3/thinking_student/FORMAL_V2_1_1X_20261104')


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """A run root of the V2.1 run's pi_0000 and pi_0009 (their results by
    link), its state with the status formal_train records from V2.2 on."""
    if not (RUN / 'policies' / 'pi_0009' / 'policy.json').exists():
        pytest.skip('V2.1 run not present')
    root = tmp_path_factory.mktemp('runs') / 'FORMAL_TEST'
    (root / 'policies').mkdir(parents=True)
    for k in ('pi_0000', 'pi_0009'):
        (root / 'policies' / k).symlink_to(RUN / 'policies' / k)
    for x in ('experience', 'eval'):
        (root / x).symlink_to(RUN / x)
    shutil.copy(RUN / 'config.yaml', root / 'config.yaml')
    state = json.loads((RUN / 'run_state.json').read_text())
    state.update(status='completed', step='done')
    (root / 'run_state.json').write_text(json.dumps(state))
    out = X.export_run(root, now=NOW, log=lambda m: None)
    assert out['manifest'] == 'published' and out['revision'] == 1
    return root


def test_a_run_exports_each_finished_result_of_its_frozen_policies(run):
    review = run / 'review'
    m = json.loads((review / 'manifest.json').read_text())
    assert [d['id'] for d in m['datasets']] == ['student:pi_0000:TRAIN:r0001', 'student:pi_0000:TRAIN_SMOKE:r0001',
                                                'student:pi_0000:DEV:r0001', 'student:pi_0009:DEV:r0001']
    assert (m['status'], m['progress'], m['strategy']) == (
        'completed', dict(phase='iteration', step='done', round=9), 'V2_1_STRUCTURAL_R10')
    assert (m['latest_policy_id'], m['best_dev_policy_id']) == ('pi_0009', 'pi_0000')
    corpus = ROOT / 'review' / 'manifest.json'
    assert m['corpus'] == dict(id='corpus_v2_1_20261104', manifest=os.path.relpath(corpus, review),
                               sha256=hashlib.sha256(corpus.read_bytes()).hexdigest())
    assert X.validate_review(review) == dict(files=7, records=1048, entries=1048, closed=1046, open=0, partial=0,
                                             censored=2, datasets=4, revision=1, status='completed')
    dev = [json.loads(x) for x in (review / 'student/pi_0009/DEV/r0001/CZCE.FG409.jsonl').read_text().splitlines()]
    # Native: the ADD of bar 0 (eob 21:01) fills at the open of bar 1; the
    # night bar belongs to the next trading day.
    assert {k: dev[0][k] for k in X.FIELDS if k != 'details'} == dict(
        id='student:CZCE.FG409:1:1', symbol='CZCE.FG409', action='ADD', direction='SHORT',
        decision_time='2024-04-08T21:01:00+08:00', entry_time='2024-04-08T21:01:00+08:00',
        bar_bob='2024-04-08T21:01:00+08:00', bar_eob='2024-04-08T21:02:00+08:00', trading_day='2024-04-09',
        entry_price=1488, volume=3, stop_price=1498, target_price=1468, grade=None, label_ready_at=None,
        outcome=dict(status='CLOSED', exit_time='2024-04-08T21:03:00+08:00', exit_price=1498,
                     exit_reason='STOP_LOSS', net_pnl=-618), note='')
    assert dev[0]['details']['Fees (CNY)'] == 18 and dev[0]['details']['Gross PnL (CNY)'] == -600
    # Each file is every committed entry of its Native account.
    f = m['datasets'][3]['files'][0]
    exam = json.loads((RUN / 'eval/pi_0009/DEV/CZCE.FG409/exam.json').read_text())['student']
    assert f['summary'] == dict(entries=exam['adds'], closed=exam['closed'], open=0, partial=0, censored=0)
    assert f['market_complete'] is True
    lunch = [r for r in dev if r['decision_time'].endswith('11:30:00+08:00')]
    assert all((r['entry_time'][11:16], r['bar_bob'][11:16], r['bar_eob'][11:16]) == ('13:30', '13:30', '13:31')
               for r in lunch)
    # The 6000-step smoke ended with an entry open: CENSORED, no exit.
    smoke = m['datasets'][1]['files']
    assert [(x['symbol'], x['summary']['censored'], x['market_complete']) for x in smoke] == [
        ('CZCE.FG001', 1, False), ('CZCE.FG005', 1, False)]
    rows = [json.loads(x) for x in (review / smoke[0]['path']).read_text().splitlines()]
    censored = [r for r in rows if r['outcome']['status'] == 'CENSORED']
    assert [r['id'] for r in censored] == ['student:CZCE.FG001:1:15'] and censored[0]['outcome'] == dict(
        status='CENSORED', exit_time=None, exit_price=None, exit_reason=None, net_pnl=None)
    assert (censored[0]['stop_price'], censored[0]['target_price']) == (1368, 1398)


def test_a_run_manifest_changes_with_its_state_only_and_stays_fixed_to_its_corpus(run, tmp_path):
    review = run / 'review'
    before = (review / 'manifest.json').read_bytes()
    assert X.export_run(run, log=lambda m: None)['manifest'] == 'unchanged'
    assert (review / 'manifest.json').read_bytes() == before
    # A new state is a new revision; the files stay as they are.
    copy = tmp_path / 'FORMAL_TEST'
    shutil.copytree(run, copy, symlinks=True)
    state = json.loads((copy / 'run_state.json').read_text())
    state['status'] = 'failed'
    (copy / 'run_state.json').write_text(json.dumps(state))
    out = X.export_run(copy, now='2026-11-04T17:00:00+08:00', log=lambda m: None)
    m = json.loads((copy / 'review' / 'manifest.json').read_text())
    assert (out['manifest'], m['revision'], m['status'], m['updated_at'], out['new']) == (
        'updated', 2, 'failed', '2026-11-04T17:00:00+08:00', [])
    # The corpus is fixed at the first publish; a registered file is never rewritten.
    m['corpus']['sha256'] = '0' * 64
    (copy / 'review' / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(X.ConflictError, match='is fixed to the corpus'):
        X.export_run(copy, log=lambda m: None)
    shutil.copy(review / 'manifest.json', copy / 'review' / 'manifest.json')
    f = copy / 'review/student/pi_0009/DEV/r0001/CZCE.FG409.jsonl'
    data = f.read_bytes()
    f.unlink()
    f.write_bytes(data.replace(b'"entry_price":1488', b'"entry_price":1489', 1))
    with pytest.raises(X.ReviewError, match='a registered file changed'):
        X.export_run(copy, log=lambda m: None)
    # validate: a summary that is not its records, a policy without results.
    f.unlink()
    f.write_bytes(data)
    for change, why in ((lambda m: m['datasets'][3]['files'][0]['summary'].update(closed=0), 'summary differs'),
                        (lambda m: m.update(latest_policy_id='pi_0005'), 'has no published result')):
        m = json.loads((review / 'manifest.json').read_text())
        change(m)
        (copy / 'review' / 'manifest.json').write_text(json.dumps(m))
        with pytest.raises(X.ReviewError, match=why):
            X.validate_review(copy / 'review', corpus=False)


def test_a_student_file_needs_a_native_ledger_that_agrees():
    collect = RUN / 'eval/pi_0009/DEV/CZCE.FG409/collect'
    if not (collect / 'state.pkl').exists():
        pytest.skip('V2.1 run not present')
    env = X.native_state(collect)
    row = json.loads((RUN / 'eval/pi_0009/DEV/CZCE.FG409/exam.json').read_text())['source']['row']
    m = X._pool_market(row)
    assert len(X.student_records(env, m, 'CZCE.FG409')) == 340

    def broken(change):
        bad = copy.deepcopy(env)
        change(bad)
        return bad
    for change, why in (
            (lambda e: e['fills'][0].update(time='2024-04-08 21:02:00+08:00'), 'is not the open of bar'),
            (lambda e: e['attributions'][0].update(net_trade_pnl='-600.0'), 'attributed net'),
            (lambda e: e['attributions'].pop(0), 'disagree'),
            (lambda e: e['trade_history'][0]['meta'].pop('entry_id'), 'a trade without its entry')):
        with pytest.raises(X.ReviewError, match=why):
            X.student_records(broken(change), m, 'CZCE.FG409')

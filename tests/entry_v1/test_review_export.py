"""review-v1 export of the corpus teacher (user 2026-11-04): the spec's two
FG205 records, the bars of lunch and night records, determinism, publishing
without overwriting, and validate's rejections. One export of the real
V2.1 corpus into a temporary review/ serves all tests."""
import hashlib
import json
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

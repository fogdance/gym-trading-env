import json
import os
from pathlib import Path

import pytest

from test_wp08_contract import plan
from wp08_fixture import native_source,controlled_actor,ARTIFACTS


def test_native_actor_1024_multiday_trace_and_account(tmp_path,monkeypatch):
    from gym_trading_env.research.fg_s3_e_quality_exam_session import ExamSession
    from gym_trading_env.research.fg_replay import iter_sidecar
    root=Path(os.environ.get('FG_WP08_C_ROOT',ARTIFACTS/'C_native_02')).absolute()
    source=native_source(root/'source');actor=controlled_actor(tmp_path/'model')
    def forbidden(*a,**k):raise AssertionError('TEACHER_OR_UPDATE_USED_DURING_EXAM')
    monkeypatch.setattr(actor.agent,'train',forbidden)
    monkeypatch.setattr('gym_trading_env.research.fg_s3_e_quality_strict.QualityStrictTeacher',forbidden)
    monkeypatch.setattr('gym_trading_env.research.fg_s3_quality_teacher_v2.NativeQualityTeacher',forbidden)
    p=plan(root);session=ExamSession(p,source,actor,root/'session',policy_id='F3_FIXED_SYNTHETIC')
    try:
        session.advance(1024)
        assert session.valid_minutes==1024 and len(session.chunks)>=4
        assert session.driver.carry is not None
        result=session.finish()
        rows=list(iter_sidecar(session.root/'trace'))
        assert result['reconciliation']['status']=='PASS'
        assert len({r['trading_date'] for r in rows})>=3
        assert len({r['session'] for r in rows})==2
        assert any(r['lots_after'] for r in rows if r['ordinal']%256==0)
        assert any(r['exits'] and any(e['reason']=='STOP_LOSS' for e in r['exits']) for r in rows)
        assert any(r['exits'] and any(e['reason']=='TAKE_PROFIT' for e in r['exits']) for r in rows)
        assert all('teacher' not in r for r in rows)
        assert sum(r['reset'] for r in rows)==1
        assert len({r['account_owner'] for r in rows})==1
        Path('docs/receipts/wp08/C_native_account.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:session.close()

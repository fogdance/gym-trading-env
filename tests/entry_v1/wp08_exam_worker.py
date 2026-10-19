"""New-interpreter native EvalSession, deliberately no training constructor."""
import json
import pickle
from pathlib import Path
import sys

from wp08_fixture import native_source,controlled_actor
from test_wp08_contract import plan
from gym_trading_env.research.fg_s3_e_quality_exam_session import ExamSession
from gym_trading_env.research.fg_s3_e_quality_exam_state import snapshot
from gym_trading_env.research.fg_replay import iter_sidecar
from gym_trading_env.research.fg_s3_e_quality_exam_metrics import analyze_trace

base=Path(sys.argv[1]);root=Path(sys.argv[2]);operation=sys.argv[3];target=int(sys.argv[4]);source_root=Path(sys.argv[5])
actor=controlled_actor(root.parent/(root.name+'_model'))
source=native_source(source_root)
p=plan(base)
session=ExamSession(p,source,actor,root,policy_id='F3_FIXED_SYNTHETIC',existing=operation=='resume')
try:
    if operation=='resume':session.restore()
    else:session.commit()
    session.advance(target);session.commit()
    if target==1024:
        receipt=session.finish();report=analyze_trace(root/'trace',source,root/'metrics')
        evidence=dict(state=snapshot(session),rows=list(iter_sidecar(root/'trace')),
            parameters=actor.before,policy=actor.save(),receipt=receipt,metrics=report)
        (root/'evidence.pkl').write_bytes(pickle.dumps(evidence,protocol=5))
        session.commit()
    print(json.dumps(dict(status='SYNTHETIC_EVAL_ONLY',valid_minutes=session.valid_minutes,
        native_account=session.raw.current_step,teacher_calls=0,training_updates=0)))
finally:session.close()

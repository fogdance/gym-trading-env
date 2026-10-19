"""Reproduce every declared policy action, independently of financial replay."""
from pathlib import Path
import uuid

import numpy as np

from .fg_replay import iter_sidecar,iter_replay_rows,trace_manifest,read_manifest,signed,E_QUALITY_SCHEMA


def verify_policy(plan,job,directory,model_output):
    from .fg_s3_e_quality_exam_actor import ExamFrozenActor
    from .fg_s3_e_quality_exam_compare import ExamBaseline
    root=Path(directory);name=job['policy']
    if name in plan['actors']:
        a=plan['actors'][name]
        loader=ExamFrozenActor.from_f3_checkpoint if a['kind']=='F3' else ExamFrozenActor.from_f4_checkpoint
        policy=loader(a['run'],a['checkpoint'],Path(model_output)/(job['label']+'_'+uuid.uuid4().hex),
            engineering=plan['exam_mode']=='SYNTHETIC_DEV',evaluation_seed=plan['seed'])
        policy.require_mode(plan);expected=policy.receipt
        fields=policy.agent.obs_space
    else:
        policy=ExamBaseline(name,seed=plan['seed'],contract=job['contract'],random_seed=job['random_seed'])
        expected=signed(dict(protocol='FG_S3_E_QUALITY_EXAM_BASELINE_V1',policy=policy.spec,
            teacher_used=False,training_updates=0))
        from .fg_s3_e_observation import E_FIELDS_V2
        fields={k:v for k,v in E_FIELDS_V2.items() if k!='action'}
    if read_manifest(root/'MODEL_READONLY_RECEIPT.json')!=expected:
        raise ValueError('EXAM_READONLY_MODEL_OR_BASELINE_RECEIPT_CHANGED')
    manifest=trace_manifest(root/'trace',schema=E_QUALITY_SCHEMA);rows=iter_sidecar(root/'trace')
    carry=None;count=0
    for _,step in iter_replay_rows(root/'trace',manifest):
        row=next(rows,None)
        if row is None:raise ValueError('EXAM_POLICY_SIDECAR_MISSING')
        obs={k:np.asarray(step[k])[None] for k in fields}
        if not row['reset']:policy.observe_row(row,legal_after=bool(step['action_mask'][1]))
        carry,action,_=policy(carry,obs)
        intended=0 if step['is_last'] else int(action['action'][0])
        if intended!=int(step['action']) or intended!=row['action']:
            raise ValueError('EXAM_FROZEN_POLICY_ACTION_TRACE_MISMATCH')
        recorded=row.get('policy_diagnostics',{}).get('p_add')
        if recorded is None or recorded!=policy.last_diagnostics['p_add']:
            raise ValueError('EXAM_FROZEN_POLICY_PROBABILITY_TRACE_MISMATCH')
        count+=1
    if next(rows,None) is not None:raise ValueError('EXAM_POLICY_SIDECAR_EXTRA_ROWS')
    frozen=policy.verify_frozen() if hasattr(policy,'verify_frozen') else dict(training_updates=0)
    return dict(status='PASS_ALL_POLICY_ACTIONS_REPRODUCED',contract=job['contract'],policy=job['label'],
        records=count,real_actor=name in plan['actors'],model_frozen=frozen,teacher_used=False,training_updates=0)

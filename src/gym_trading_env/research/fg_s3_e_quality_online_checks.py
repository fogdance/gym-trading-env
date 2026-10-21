"""Read-only, native teacher-forced multi-step WM diagnostics; no admission."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint


_CALIBRATIONS={}


@dataclass(frozen=True)
class Calibration:
    report: dict


def require_calibration(record):
    proof=_CALIBRATIONS.get(id(record))
    if proof is None or proof[0] is not record or proof[1]!=fingerprint(record.report):
        raise ValueError('ONLINE_ACTUAL_CALIBRATION_EXECUTION_REQUIRED')
    return copy.deepcopy(record.report)


def calibration(agent,data,*,source_identity,horizon=3,seed=202):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from dreamerv3.fg_s3_e_quality_foundation import causal_context
    from dreamerv3.fg_s3_e_quality_course_contract import WM_GROUPS
    from dreamerv3.fg_s3a_actor import parameter_digests
    k=agent.config.quality_f3_burnin_steps;t=agent.config.quality_f3_target_steps
    if type(horizon) is not int or not 0<horizon<t:
        raise ValueError('ONLINE_CALIBRATION_HORIZON_INVALID')
    def forward(raw):
        m=agent.model
        carry,obs,prev,ids=causal_context(m,raw,k,t)
        wm=m.world_model(carry,obs,prev,False,diagnostics=True)
        n=t-horizon
        entries=jax.tree.map(lambda x:x[:,:n],wm[4])
        starts=m.dyn.starts(entries,wm[1],n)
        actions=raw['action'][:,1+k:]
        known=jnp.stack([actions[:,i:i+horizon] for i in range(n)],1).reshape((-1,horizon))
        _,imag,_=m.dyn.imagine(starts,dict(action=known),horizon,False)
        tensor=m.feat2tensor(imag)
        mask,fallback=m._predicted_action_mask_info(imag,2)
        def targets(name):
            x=obs[name]
            return jnp.stack([x[:,i+1:i+1+horizon] for i in range(n)],1).reshape((-1,horizon,*x.shape[2:]))
        return dict(post_mask=wm[-1]['f1_raw/availability_post'],true_mask=obs['action_mask'],
            imag_mask=mask,imag_fallback=fallback,imag_true_mask=targets('action_mask'),
            imag_reward=m.rew(tensor,2).pred(),true_reward=targets('reward'),
            imag_continue=m.con(tensor,2).prob(1),
            true_continue=(~targets('is_terminal'))*(np.float32(332/333)**targets('market_advance')),
            stepid=ids)
    before=parameter_digests(agent.params)
    _,out=nj.pure(forward)(agent.params,{key:jnp.asarray(value) for key,value in data.items()},
        seed=agent._seeds(seed,agent.train_mirrored),create=False,modify=True)
    out=jax.tree.map(np.asarray,out)
    if before!=parameter_digests(agent.params):raise ValueError('ONLINE_CALIBRATION_CHANGED_MODEL')
    post=out['post_mask'].astype(bool);truth=out['true_mask'].astype(bool)
    pred=out['imag_mask'].astype(bool);target=out['imag_true_mask'].astype(bool)
    positives=int(target[...,1].sum());negatives=int((~target[...,1]).sum())
    def recall(a,b):return float((a[...,1]&b[...,1]).sum()/max(1,b[...,1].sum()))
    metrics={'avail/post_accuracy':float((post==truth).mean()),'avail/post_exact_accuracy':float((post==truth).all(-1).mean()),
        'avail/post_add_recall':recall(post,truth),'avail/post_add_positives':int(truth[...,1].sum()),
        'avail/post_add_true_positives':int((post[...,1]&truth[...,1]).sum()),
        'avail/prior_accuracy':float((pred==target).mean()),'avail/prior_exact_accuracy':float((pred==target).all(-1).mean()),
        'avail/prior_fpr':float((pred[...,1]&~target[...,1]).sum()/max(1,negatives)),
        'avail/prior_add_recall':recall(pred,target),'avail/prior_add_positives':positives,
        'avail/prior_add_true_positives':int((pred[...,1]&target[...,1]).sum()),
        'avail/img_fallback_rate':float(out['imag_fallback'].mean())}
    report=dict(protocol='FG_S3_E_QUALITY_ONLINE_CALIBRATION_V1',scope='READ_ONLY_NOT_ADMISSION',
        source_identity=source_identity,wm_identity=fingerprint({key:value for key,value in before.items() if key.startswith(WM_GROUPS)}),
        horizon=horizon,seed=seed,metrics=metrics,add_positives=positives,
        reward_mae=float(np.abs(out['imag_reward']-out['true_reward']).mean()),
        continue_mae=float(np.abs(out['imag_continue']-out['true_continue']).mean()),
        finite=all(np.isfinite(x).all() for x in out.values()),
        native_stepids=[x.tobytes().hex() for x in out['stepid'].reshape(-1,20)],
        actual_actions_sha256=hashlib.sha256(data['action'].tobytes()).hexdigest(),
        arrays={key:value.tolist() for key,value in out.items() if key!='stepid'},
        all_model_and_optimizer_bytes_equal=True,training_admitted=False,research_admitted=False)
    record=Calibration(report);_CALIBRATIONS[id(record)]=(record,fingerprint(report))
    return record


def formal_mask_policy(settings):
    from embodied.run.fg_s3_gate import gate_policy
    from .fg_s3_e_quality_foundation import model_config
    return gate_policy(model_config(settings,controlled_small=False),diagnostic=False)


def cpu_probe(output,mode='ONLINE_SUPERVISED'):
    """Fixed bounded native synthetic source, never an editable TRAIN bypass."""
    import json
    import random
    import sys
    from pathlib import Path
    import numpy as np
    from .fg_s3_e_quality_foundation import PROJECT
    from .fg_s3_e_quality_course_preflight import _local_path
    from .fg_replay import write_json,signed
    from .fg_s3_e_quality_actor_checks import evaluate_native
    from .fg_s3_e_quality_multi_actor import MultiCorpusF3Stream
    from dreamerv3.fg_s3_e_quality_online import MODES
    from dreamerv3.fg_s3a_actor import parameter_digests
    from embodied.run.fg_s3_e_quality_online_run import OnlineRun,engineering_permit
    out=_local_path(output)
    if out.exists() or not any(out.is_relative_to(p) for p in
            (PROJECT/'artifacts/fg_s3_v2/e_quality_wp07',PROJECT/'docs/receipts/wp07')):
        raise ValueError('ONLINE_PROBE_NEW_ISOLATED_OUTPUT_REQUIRED')
    if mode not in MODES:raise ValueError('ONLINE_EXECUTION_MODE_INVALID')
    random.seed(202);np.random.seed(202)
    sys.path.insert(0,str(PROJECT/'tests/entry_v1'))
    from wp07_fixture import execution
    from wp02_fixture import native_factory
    from data_multi_fixture import build_sources,descriptors
    factory,_,_=native_factory(out/'market',minutes=100,chunk=35)
    spec=execution(out/'run',mode);spec.update(env_steps=64,max_consumed_batches=0 if mode=='SHADOW' else 16)
    history=None
    if mode!='SHADOW':
        config=json.loads(build_sources(out/'history').read_text())
        history=MultiCorpusF3Stream.open(descriptors(config,'ACTOR_STRICT'),settings=dict(batch_size=1,burnin_steps=4,target_steps=8),
            per_source_sampling_spec=config['per_source_sampling_spec'],source_mix_weights=config['source_mix_weights'],seed=202,
            aux_per_source_sampling_spec={s['corpus_id']:dict(mode='MIXED',natural_weight=.5,
                strata_weights=dict(STRICT_ADD=1,STRICT_WAIT=1)) for s in config['sources']})
    permit=engineering_permit(spec,numeric_rl=mode=='ONLINE_RL_CONTROLLED')
    run=OnlineRun(spec,factory,dict(fold='TRAIN',scope='BUILTIN_SYNTHETIC_ONLY'),history,_probe=permit)
    try:
        before=parameter_digests(run.agent.params);result=run.run_to();after=parameter_digests(run.agent.params)
        proof=run.account_verification.state['final_evidence']['receipt']['reconciliation']
        evaluation=evaluate_native(run.agent,factory,out/'actor_only',steps=32,
            initial_state=run.collector.initial_state,
            source=dict(execution=run.identity['protocol']),scope='BUILTIN_SYNTHETIC_NOT_RESEARCH')
        repeated=evaluate_native(run.agent,factory,out/'actor_only_repeat',steps=32,
            initial_state=run.collector.initial_state,
            source=dict(execution=run.identity['protocol']),scope='BUILTIN_SYNTHETIC_NOT_RESEARCH')
        if evaluation['trace_identity']!=repeated['trace_identity']:
            raise AssertionError('ONLINE_ACTOR_ONLY_EVALUATION_NOT_DETERMINISTIC')
        result.update(scope='BUILTIN_SYNTHETIC_ENGINEERING_ONLY',before=before,after=after,
            changed_keys=[k for k in before if before[k]!=after[k]],
            complete_account_replay=proof,actor_only=evaluation,actor_only_repeat_trace=repeated['trace_identity'],
            actor_only_deterministic=True,real_TRAIN_updates=0,
            formal_rl_admitted=False,research_admitted=False)
        write_json(out/'probe.json',signed(result));return result
    finally:run.close()

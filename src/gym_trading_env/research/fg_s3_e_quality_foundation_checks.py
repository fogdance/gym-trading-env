"""Read-only F1 diagnostics over the original WM; no optimizer or admission."""
from __future__ import annotations

import numpy as np


def forward(agent, data, *, gradients=False, diagnostics=True):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from dreamerv3.fg_s3_e_quality_foundation import context,loss
    def fn(raw):
        carry,obs,actions,ids=context(agent.model,raw)
        value,(_,_,out,mets)=loss(agent.model,carry,obs,actions,False,diagnostics=diagnostics)
        return value,(ids,out,mets)
    def call(raw):
        if gradients:
            value,_,grad,aux=nj.grad(fn,agent.model.modules,has_aux=True)(raw)
            return value,grad,aux
        return fn(raw)
    # Private Ninjax state is discarded, including any RNG/state writes.
    _,result=nj.pure(call)(agent.params,{k:jnp.asarray(v) for k,v in data.items()},
                          seed=202,create=False,modify=True)
    return jax.tree.map(np.asarray,result)


def support(reward,mask,*,candidate_count,right_censored_count):
    """Pathology is a coverage finding, never market learning permission."""
    blockers=[]
    if not np.any(reward):blockers.append('ZERO_REWARD_ONLY')
    if not np.any(mask[...,1]):blockers.append('NO_LEGAL_ADD')
    if candidate_count==0 or candidate_count==right_censored_count:
        blockers.append('NO_UNCENSORED_CANDIDATES')
    return dict(status='NO_SUPPORT' if blockers else 'COVERAGE_PRESENT',blockers=blockers,
                model_research_go=False,training_admitted=False)


def summarize(agent,data,metadata,*,candidate_count=0,right_censored_count=0,gradients=True):
    """Events refer to committed native feedback on each target row only."""
    result=forward(agent,data,gradients=gradients)
    if gradients:value,grad,(ids,out,mets)=result
    else:value,(ids,out,mets)=result;grad={}
    start=1+agent.model.config.quality_f1_burnin_steps
    reward=data['reward'][:,start:];mask=data['action_mask'][:,start:]>.5
    prediction=mets['f1_raw/reward_prediction']
    rows=[m['rows'][start:] for m in metadata]
    groups={'zero':reward==0,'nonzero':reward!=0}
    for name,reason in (('TP','TAKE_PROFIT'),('SL','STOP_LOSS'),('FIRST_ADVERSE10','FIRST_ADVERSE10')):
        # Native event strings are independent of teacher counterfactuals.
        groups[name]=np.asarray([[any(e['reason']==reason for e in row['reward_events'])
            for row in window] for window in rows])
    events={}
    for name,selected in groups.items():
        n=int(selected.sum())
        events[name]=dict(status='SUPPORTED' if n else 'NO_SUPPORT',count=n,
            loss=float(out['losses']['rew'][selected].mean()) if n else None,
            bias=float((prediction-reward)[selected].mean()) if n else None,
            target_mean=float(reward[selected].mean()) if n else None)
    availability={}
    for phase in ('post','prior'):
        pred=mets['f1_raw/availability_'+phase];positive=mask[...,1];negative=~positive
        availability[phase]=dict(legal_add_count=int(positive.sum()),only_wait_count=int(negative.sum()),
            add_recall=float(pred[...,1][positive].mean()) if positive.any() else None,
            false_positive_rate=float(pred[...,1][negative].mean()) if negative.any() else None,
            add_support='SUPPORTED' if positive.any() else 'NO_SUPPORT')
    names=('enc/','enc/minute/','enc/context15/','enc/macd5/','enc/fusion/','dyn/','dec/','rew/','con/','avail/')
    report=dict(loss=float(value),finite=bool(np.isfinite(value) and all(np.isfinite(v).all() for v in grad.values())),
        losses={k:dict(mean=float(v.mean()),scale=float(agent.model.scales[k]),targets=int(v.size))
                for k,v in out['losses'].items()},reward_events=events,availability=availability,
        gradient_norms={p:float(np.sqrt(sum(np.square(v.astype(np.float64)).sum() for k,v in grad.items()
                      if k.startswith(p)))) for p in names},
        target_positions=int(reward.size),target_market_minutes=int(data['market_advance'][:,start:].sum()),
        burnin_positions=int(len(reward)*(start-1)),
        burnin_market_minutes=int(data['market_advance'][:,1:start].sum()),
        source_rows_from_reset=[m['distance_from_reset_market_minutes'] for m in metadata],
        warmup_interpretation='FINITE_HISTORY_APPROXIMATION',
        support=support(reward,mask,candidate_count=candidate_count,right_censored_count=right_censored_count),
        actual_optimizer_updates=0,training_admitted=False,model_research_go=False)
    return report,(out,mets,grad)


def cpu_probe(output):
    """Fixed built-in synthetic engineering test, never a user Corpus flag."""
    import sys
    from pathlib import Path
    from .fg_s3_e_quality_foundation import PROJECT,preflight
    from .fg_s3_e_quality_course_preflight import _local_path
    from .fg_replay import write_json,signed
    out=_local_path(output)
    if out.exists() or not any(out.is_relative_to(p) for p in
            (PROJECT/'docs/receipts/wp04',PROJECT/'artifacts/fg_s3_v2/e_quality_wp04')):
        raise ValueError('F1_PROBE_OUTPUT_MUST_BE_NEW_AND_ISOLATED')
    # Reuse the existing original-Env engineering fixtures. The installed
    # formal lifecycle has no fixture import or synthetic-source switch.
    sys.path.insert(0,str(PROJECT/'tests/entry_v1'))
    from wp04_fixture import multiscale_reader
    from embodied.run.fg_s3_e_quality_foundation import FoundationRun,synthetic_probe_permit
    out.mkdir(parents=True)
    s,r=multiscale_reader(out/'source');index=r.index_path;r.close()
    prep=preflight(s)
    run=FoundationRun(prep,index,out/'run',_probe=synthetic_probe_permit(prep))
    before=__import__('dreamerv3.fg_s3a_actor',fromlist=['parameter_digests']).parameter_digests(run.agent.params)
    result=run.run()
    result.update(scope='SYNTHETIC_CPU_ONLY',before=before,
        after=__import__('dreamerv3.fg_s3a_actor',fromlist=['parameter_digests']).parameter_digests(run.agent.params),
        reader_coverage=run.reader.coverage(),real_TRAIN_updates=0)
    write_json(out/'probe.json',signed(result))
    return result


def read_check(config,output,*,purpose='development',producer_gym=None):
    """Finite TRAIN read-only F1 loss. G3 is optional and explicit."""
    import gzip,json,time
    from pathlib import Path
    from .fg_s3_e_quality_foundation import PROJECT,read_settings,preflight,create_agent,F1WindowAdapter
    from .fg_s3_e_quality_course_preflight import _local_path
    from .fg_s3_e_quality_reader import build_join_index,EQualityCourseReader
    from .fg_replay import write_json,signed
    from dreamerv3.fg_s3a_actor import parameter_digests
    s=read_settings(config);out=_local_path(output)
    if s['device']!='cpu' or out.exists() or not any(out.is_relative_to(p) for p in
            (PROJECT/'docs/receipts/wp04',PROJECT/'artifacts/fg_s3_v2/e_quality_wp04')):
        raise ValueError('F1_READ_CHECK_CPU_NEW_ISOLATED_OUTPUT_REQUIRED')
    began=time.monotonic()
    p=preflight(s,purpose=purpose,producer_gym=producer_gym,log_path=out/'native.log' if purpose=='native-verification' else None)
    index=out/'join.sqlite';joined=build_join_index(p.source,index)
    a=create_agent(s,out/'model',controlled_small=True)
    before=parameter_digests(a.params);policies={};candidates={}
    for policy in sorted(joined['policies']):
        candidate_count=right=0
        with gzip.open(Path(s['corpus_root'])/'policies'/policy/'steps.jsonl.gz','rt') as stream:
            for line in stream:
                facts=json.loads(line)['teacher']['L1']
                candidate_count+=int(facts['actual_legal_add'])
                right+=int(bool(facts.get('censor_reason')) and facts['actual_legal_add'])
        candidates[policy]=dict(legal_add_candidates=candidate_count,right_censored=right)
        spec={**s['sampling_spec'],'policies':[policy]}
        r=EQualityCourseReader.open(p.source,stage='WM_NATURAL',sampling_spec=spec,seed=s['model_init']['seed'],index_path=index)
        try:
            adapter=F1WindowAdapter(r,{**s,'sampling_spec':spec});reports=[]
            for _ in range(2):
                data,meta=adapter.sample()
                report,_=summarize(a,data,meta,candidate_count=candidate_count,right_censored_count=right,gradients=False)
                if not report['finite']:raise FloatingPointError('F1_REAL_READ_NONFINITE')
                reports.append(report)
            policies[policy]=dict(windows=reports,coverage=r.coverage())
        finally:r.close()
    after=parameter_digests(a.params)
    if before!=after or int(a.n_updates)!=0:raise AssertionError('F1_READ_CHECK_CHANGED_TRAIN_STATE')
    receipt=signed(dict(status='F1_REAL_READ_ONLY_PASS',source=p.receipt,join=joined,
        policies=policies,candidate_coverage=candidates,controlled_small=True,
        parameter_and_optimizer_before=before,parameter_and_optimizer_after=after,
        actual_optimizer_updates=0,training_admitted=False,model_research_go=False,
        seconds=time.monotonic()-began))
    write_json(out/'read_check.json',receipt)
    return receipt

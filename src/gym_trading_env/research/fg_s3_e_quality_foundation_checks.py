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

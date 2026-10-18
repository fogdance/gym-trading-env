"""Read-only four-head diagnostics using original WM and prediction metrics."""
from __future__ import annotations
import copy,json
from pathlib import Path
import numpy as np

def forward(agent,data):
    from dreamerv3.fg_s3_e_quality_heads import read_only_forward
    return read_only_forward(agent,data)


def score_batch(data,result,start,*,prior=None):
    from scipy.special import softmax,expit
    from scipy.stats import rankdata
    from .fg_s3_quality_prediction_evaluation import quality_scores
    p=result['predictions'];labels={n:data['quality_'+n][:,start:] for n in ('event','touch','adverse','score')}
    valid={n:data['quality_'+n+'_valid'][:,start:] for n in labels}
    rows=[]
    for b,t in np.ndindex(labels['score'].shape):
        label={k:labels[k][b,t].tolist() for k in labels}
        label.update({k+'_valid':valid[k][b,t].tolist() for k in valid})
        rows.append(dict(label=label,prediction={'quality/outcome_probs':softmax(p['event'][b,t],axis=-1).tolist(),
            'quality/touch_probability':expit(p['touch'][b,t]).tolist(),
            'quality/adverse_probability':expit(p['adverse'][b,t]).tolist(),
            'quality/score_prediction':float(100*expit(p['score'][b,t]))}))
    metrics=quality_scores(rows,prior,role='F2_OFFLINE_TRAIN_DIAGNOSTIC',include_decision=False,include_account=False)
    metrics['event_confusion']={};metrics['binary_discrimination']={}
    for h in range(7):
        y=labels['event'][...,h][valid['event'][...,h]];pred=p['event'][...,h,:].argmax(-1)[valid['event'][...,h]]
        metrics['event_confusion'][str(h)]=np.bincount(y*4+pred,minlength=16).reshape(4,4).tolist()
    for name in ('touch','adverse'):
        dims=(4,2) if name=='touch' else (4,)
        for ix in np.ndindex(dims):
            index=(slice(None),slice(None),*ix);v=valid[name][index];y=labels[name][index][v];prob=expit(p[name][index][v]);pos=int((y==1).sum());neg=int((y==0).sum())
            auc=pr=None
            if pos and neg:
                ranks=rankdata(prob);auc=float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg))
                # Threshold groups preserve ties, rather than inventing order.
                thresholds=np.unique(prob)[::-1];last=0.;pr=0.
                for threshold in thresholds:
                    selected=prob>=threshold;tp=int(((y==1)&selected).sum());recall=tp/pos
                    pr+=(recall-last)*tp/int(selected.sum());last=recall
            metrics['binary_discrimination'][name+'_'+str(ix)]=dict(positive=pos,negative=neg,auc=auc,average_precision=pr,status='SUPPORTED' if pos and neg else 'UNSUPPORTED')
    selected=valid['score']
    if selected.any():
        import optax,jax.numpy as jnp
        from .fg_s3_prediction_evaluation import calibration
        pp=expit(p['score'][selected]);yy=labels['score'][selected]/100
        metrics['score_huber_normalized']=float(np.asarray(optax.huber_loss(jnp.asarray(pp),jnp.asarray(yy),delta=.1)).mean())
        metrics['score_calibration']=calibration(pp,yy)
    else:
        metrics['score_huber_normalized']=None;metrics['score_calibration']=None
    metrics['score_high_count']=int((valid['score']&(labels['score']>=80)).sum())
    metrics['invalid_counts']={n:int((~v).sum()) for n,v in valid.items()}
    metrics['loss_terms']={k:np.asarray(v).tolist() for k,v in result['metrics'].items()}
    metrics['horizon_names']=['5_MARKET_MINUTES','15_MARKET_MINUTES','60_MARKET_MINUTES','120_MARKET_MINUTES','240_MARKET_MINUTES','OFFICIAL_DAY_END','NEXT_OFFICIAL_DAY_END']
    metrics['prediction_finite']=all(np.isfinite(x).all() for x in p.values())
    metrics['research_admitted']=False
    return metrics

def ablations(agent,data,baseline=None):
    baseline=forward(agent,data) if baseline is None else baseline
    report={}
    for name,key in (('1m','market_seq_e_shared'),('15m','market_context_15m'),('partial5m_macd','market_macd_5m')):
        changed=copy.deepcopy(data);value=changed[key]
        visible=True if name=='1m' else bool(np.any(value[...,-1]>.5))
        if name=='1m':value[...]=0
        else:value[...,:-1]=0  # Preserve validity and causal geometry.
        out=forward(agent,changed)
        delta={n:float(np.max(np.abs(out['predictions'][n]-baseline['predictions'][n]))) for n in out['predictions']}
        report[name]=dict(status='SUPPORTED' if visible else 'UNSUPPORTED',valid_visible=visible,max_abs_logit_delta=delta,
            encoder_max_abs_delta=float(np.max(np.abs(out['encoder_tokens']-baseline['encoder_tokens']))),
            posterior_logits_max_abs_delta=float(np.max(np.abs(out['posterior_logits']-baseline['posterior_logits']))),
            realized_features_max_abs_delta=float(np.max(np.abs(out['features']-baseline['features']))))
    return report

def index_coverage(reader):
    """Full projected-label denominators, independent of sampled exposure."""
    from dreamerv3.fg_s3_e_quality_heads import TARGETS
    counts={n:dict(valid=0,invalid=0) for n in TARGETS};event=np.zeros((7,4),np.int64)
    touch=np.zeros((4,2,2),np.int64);adverse=np.zeros((4,2),np.int64);score_high=0
    for value, in reader.db.execute('SELECT labels FROM rows'):
        label=json.loads(value)
        for n in TARGETS:
            valid=np.asarray(label['quality_'+n+'_valid'],bool);y=np.asarray(label['quality_'+n])
            counts[n]['valid']+=int(valid.sum());counts[n]['invalid']+=int((~valid).sum())
            if n=='event':
                for h in range(7):
                    if valid[h]:event[h,int(y[h])]+=1
            elif n=='score':score_high+=int(valid and y>=80)
            else:
                dest=touch if n=='touch' else adverse
                for ix in np.ndindex(valid.shape):
                    if valid[ix]:dest[(*ix,int(y[ix]))]+=1
    return dict(counts=counts,event_counts=event.tolist(),touch_counts=touch.tolist(),adverse_counts=adverse.tolist(),score_ge80=score_high,
        contracts=[reader.contract],official_days=[r[0] for r in reader.db.execute('SELECT DISTINCT day FROM rows ORDER BY day')],
        source=reader.coverage(),evaluation_scope='ELIGIBLE_HEADS_VALID_WINDOWS',research_support='LIMITED')


def evaluate_batch(reader,agent,data):
    if reader.mode!='NATURAL':raise ValueError('F2_NATURAL_EVALUATION_REQUIRED')
    return score_batch(data,forward(agent,data),1+agent.config.quality_f2_burnin_steps)

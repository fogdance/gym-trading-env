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

def _output(path):
    from .fg_s3_e_quality_course_preflight import _local_path
    from .fg_s3_e_quality_foundation import PROJECT
    out=_local_path(path)
    if out.exists() or not any(out.is_relative_to(p) for p in (PROJECT/'docs/receipts/wp05',PROJECT/'artifacts/fg_s3_v2/e_quality_wp05')):
        raise ValueError('F2_DIAGNOSTIC_NEW_ISOLATED_OUTPUT_REQUIRED')
    return out

def probe(output,*,device='cpu'):
    """Bounded built-in synthetic handoff/update; no user-supplied Corpus."""
    import sys
    from .fg_s3_e_quality_foundation import PROJECT
    from .fg_s3_e_quality_heads import preflight,validate_settings
    from .fg_s3_e_quality_reader import build_join_index
    from .fg_replay import write_json,signed
    from embodied.run.fg_s3_e_quality_heads import HeadsRun,engineering_permit
    from dreamerv3.fg_s3a_actor import parameter_digests
    out=_output(output)
    if device not in ('cpu','gpu'):raise ValueError('F2_PROBE_DEVICE_INVALID')
    # Prefer the preserved, validated WP04 built-in source; never a real F1 run.
    base=PROJECT.parent.parent/'fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0'
    if (base/'settings.json').is_file():
        s=json.loads((base/'settings.json').read_text());f1=base/'continuous';origin='PRESERVED_WP04_SYNTHETIC'
    else:
        sys.path.insert(0,str(PROJECT/'tests/entry_v1'))
        from wp04_fixture import multiscale_reader
        from .fg_s3_e_quality_foundation import preflight as f1_preflight
        from embodied.run.fg_s3_e_quality_foundation import FoundationRun,synthetic_probe_permit
        s,r=multiscale_reader(out/'source');index=r.index_path;r.close();s['device']=device
        p=f1_preflight(s);run=FoundationRun(p,index,out/'foundation_run',_probe=synthetic_probe_permit(p));run.run()
        f1=run.root;origin='BUILTIN_SYNTHETIC_F1'
    s.update(schema='FG_S3_E_QUALITY_F2_CONFIG_V1',stage='QUALITY_HEADS',sampling_stage='HEADS_VALID',
        source_f1_run=str(f1),source_f1_checkpoint=str(f1/'ckpt/step_000000000004'),
        prepared_dir=str(out/'prepared'),run_dir=str(out/'run'),device=device,
        target_steps=8,burnin_steps=4,batch_size=1,optimizer_updates=4,report_every_updates=1,save_every_updates=2)
    s=validate_settings(s);p=preflight(s);index=out/'join.sqlite';build_join_index(p.source,index)
    run=HeadsRun(p,index,out/'run',_probe=engineering_permit(p));before=parameter_digests(run.agent.params)
    result=run.run();after=parameter_digests(run.agent.params)
    frozen=[k for k,v in run.agent.model.f2_state_owner.items() if v is None]
    if not all(before[k]==after[k] for k in frozen):raise AssertionError('F2_PROBE_FROZEN_STATE_CHANGED')
    result.update(scope='SYNTHETIC_ENGINEERING_ONLY',device=device,source_origin=origin,real_TRAIN_updates=0,
        before=before,after=after,frozen_keys=frozen,frozen_byte_equal=True,
        head_changes={n:sum(before[k]!=after[k] for k in before if k.startswith('quality_'+n+'/')) for n in ('event','touch','adverse','score')})
    import jax
    result['execution_devices']=[str(x) for x in jax.devices()]
    write_json(out/'probe.json',signed(result));return result

def read_check(config,output,*,purpose='development',producer_gym=None):
    """Actual TRAIN Reader/forward only. Fresh diagnostic weights, no handoff."""
    from .fg_s3_e_quality_heads import read_settings,create_agent,F2WindowAdapter
    from .fg_s3_e_quality_supervision import SupervisionSource,QualitySupervisionAdmission
    from .fg_s3_e_quality_reader import build_join_index,EQualityCourseReader
    from .fg_replay import write_json,signed
    from dreamerv3.fg_s3a_actor import parameter_digests
    from .fg_s3_e_quality_foundation import model_config
    from gym_trading_env.envs.entry_contract import fingerprint
    s=read_settings(config);out=_output(output)
    if s['device']!='cpu':raise ValueError('F2_REAL_READ_CHECK_CPU_ONLY')
    source=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(s['corpus_root'],stage='QUALITY_HEADS'),
        purpose=purpose,producer_gym=producer_gym,log_path=out/'native.log' if purpose=='native-verification' else None)
    index=out/'join.sqlite';joined=build_join_index(source,index)
    a=create_agent(s,out/'model',controlled_small=True);before=parameter_digests(a.params)
    reports={};market=set();accounts=set();coverage=None
    for policy in sorted(joined['policies']):
        spec=dict(mode='NATURAL',policies=[policy])
        r=EQualityCourseReader.open(source,stage='HEADS_VALID',sampling_spec=spec,seed=s['model_init']['seed'],index_path=index)
        try:
            if coverage is None:coverage=index_coverage(r)
            rows=[]
            for _ in range(2):
                data,meta=F2WindowAdapter(r,{**s,'sampling_spec':spec}).sample()
                result=forward(a,data);metrics=score_batch(data,result,1+s['burnin_steps'])
                if not metrics['prediction_finite'] or not np.isfinite(result['loss']):raise FloatingPointError('F2_REAL_READ_NONFINITE')
                rows.append(metrics)
                for m in meta:
                    for row in m['rows'][1+s['burnin_steps']:]:market.add((m['contract'],row['cursor']));accounts.add((row['policy'],row['stepid']))
            reports[policy]=dict(windows=rows,reader=r.coverage())
        finally:r.close()
    after=parameter_digests(a.params)
    if before!=after or int(a.n_updates)!=0:raise AssertionError('F2_READ_CHECK_MODIFIED_MODEL')
    result=dict(status='F2_REAL_DATA_READ_ONLY_LIMITED',foundation_status='FOUNDATION_NOT_TRAINED',
        model_origin='FRESH_CONTROLLED_DIAGNOSTIC',source_f1_consumed=False,
        blockers=['REAL_TRAINED_F1_NOT_PROVIDED','INDEPENDENT_RESEARCH_SPLIT_UNSUPPORTED'],
        training_admitted=False,research_admitted=False,real_TRAIN_updates=0,actual_optimizer_updates=0,
        model_hash=fingerprint(model_config(s,controlled_small=True)),source=source.receipt,
        course_identity=source.source.course_identity_hash,corpus_identity=source.source.corpus_content_hash,
        coverage=coverage,policies=reports,unique_target_market_cursors=len(market),target_account_rows=len(accounts),
        before=before,after=after,all_parameters_byte_equal=True,
        evaluation_split=dict(status='UNSUPPORTED',scope='FIXED_EXISTING_TRAIN_SMOKE_CORPUS',reason='NO_INDEPENDENT_MULTI_CONTRACT_SUFFIX_PARTITION'),
        data_reads='IMMUTABLE_TRAIN_CORPUS_ONLY')
    write_json(out/'read_check.json',signed(result));return result

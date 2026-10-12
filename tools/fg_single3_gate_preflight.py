#!/usr/bin/env python3
"""CPU-only frozen-model Gate calibration; never performs optimizer updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '24')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fg_gate_matplotlib')

import argparse
import collections
import gc
import json
import pickle
from pathlib import Path
import resource
import sys
import time

from fg_single3_prior_audit import ROOT, DREAMER, load_replay, make_predict, model_config, metrics, write_json
sys.path[:0] = [str(ROOT / 'src'), str(DREAMER)]
THRESHOLDS = (.50, .35, .30, .25, .20)


def native_batch(data, starts):
    import embodied
    import numpy as np
    offsets = np.asarray(starts)[:, None] + np.arange(65)[None]
    batch = {k:v[offsets] for k,v in data.items()}
    batch = embodied.Replay._annotate_batch(None, batch, None, True)
    batch['consec'] = np.zeros((16,65),np.int32)
    return batch


def make_imagination(config, data, threshold):
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    from dreamerv3.agent import Agent, IMAG_ACTION_MASK, IMAG_MASK_FALLBACK
    obs_space = {k:elements.Space(data[k].dtype, data[k].shape[1:]) for k in (
        'market_seq','position_state','action_mask','reward','is_first','is_last','is_terminal')}
    act_space = {'action':elements.Space(np.int32, (), 0, 2)}
    cfg = model_config(config).update(avail_threshold=threshold)
    model = object.__new__(Agent)
    Agent.__init__(model, obs_space, act_space, cfg)
    assert cfg.imag_length == 15 and cfg.imag_last == 0
    def run(starts):
        policy = lambda feat:model._sample_imag_action(feat, 1)
        _, feat, acts = model.dyn.imagine(starts, policy, cfg.imag_length, False)
        last = policy(jax.tree.map(lambda x:x[:, -1], feat))
        acts = jax.tree.map(lambda x,y:jnp.concatenate([x,y[:,None]],1),acts,last)
        mask, fallback = acts[IMAG_ACTION_MASK], acts[IMAG_MASK_FALLBACK]
        chosen = jnp.take_along_axis(mask,acts['action'][...,None],-1)[...,0]
        return dict(mask=mask, fallback=fallback, action=acts['action'], valid=chosen)
    pure=nj.pure(run)
    return jax.jit(lambda params, starts, seed:pure(params,starts,seed=seed)[1])


def seeds(index, imagination=False):
    import numpy as np
    # Calibration RNG only; frozen model training seed and RNG states are unchanged.
    return np.random.default_rng([202,index,int(imagination)]).integers(
        0,np.iinfo(np.uint32).max,2,dtype=np.uint32)


def original_audit_seeds(index):
    import numpy as np
    return np.random.default_rng([202,index]).integers(0,np.iinfo(np.uint32).max,2,dtype=np.uint32)


def imagination_counts(result, groups=None):
    import numpy as np
    mask, fallback, valid = (np.asarray(result[k]) for k in ('mask','fallback','valid'))
    assert mask.shape[1:] == (16,2)
    assert mask[...,0].all() and valid.all()
    assert (mask.sum(-1) >= 1).all()
    assert np.all(result['action'][fallback] == 0)
    groups = groups or {'all':np.ones(len(mask),bool)}
    out={}
    for name, group in groups.items():
        n=int(np.sum(group))*16
        out[name]=dict(decisions=n, fallback=int(fallback[group].sum()),
            invalid=int((~valid[group]).sum()), wait_invalid=int((~mask[group,:,0]).sum()),
            cardinality_sum=int(mask[group].sum()), add_enabled=int(mask[group,:,1].sum()))
    return out


def add_counts(dest, counts):
    for name, row in counts.items():
        for key, value in row.items():
            dest.setdefault(name,collections.Counter())[key] += value


def finalize_imagination(counts):
    return {name:{**dict(row), 'fallback_rate':row['fallback']/row['decisions'] if row['decisions'] else None,
        'cardinality':row['cardinality_sum']/row['decisions'] if row['decisions'] else None}
        for name,row in counts.items()}


def main(args):
    import elements
    import embodied
    import jax
    import numpy as np
    from dreamerv3.configuration import read_config_yaml
    from embodied.run.fg_checkpoint import validate_bundle, file_hash, load_replay as restore_replay
    from embodied.run.action_mask_warmup import ActionMaskWarmup, METRICS, ADD_RECALL_METRICS
    started=time.monotonic()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    write_json(output/'status.json',dict(status='RUNNING',phase=args.phase))
    meta=validate_bundle(args.source)
    before=file_hash(args.source/'manifest.json')
    config=read_config_yaml(args.source/'config.yaml')
    train=json.loads(Path(config['fg_entry']['manifest']).read_text())
    assert all(r['role']=='train' for r in train['files'])
    data=load_replay(args.source,meta,12)
    state=pickle.loads((args.source/'agent.pkl').read_bytes())
    counters=state['counters']
    assert not bool(state['params']['avail_actor_gate/value'])
    selected={k:v for k,v in state['params'].items() if k.startswith(('enc/','dyn/','avail/','pol/'))}
    params=jax.device_put(selected)
    del state,selected;gc.collect()
    predict=make_predict(config,data,return_features=True)
    imag={t:make_imagination(config,data,t) for t in THRESHOLDS}
    gate_config=elements.Config(config['action_mask_warmup'])
    gates={t:ActionMaskWarmup.for_single3_1m(gate_config) for t in THRESHOLDS}
    ends=np.r_[np.flatnonzero(data['is_terminal'])+1,len(data['action'])]
    contract_rows=[r for r in train['files'] if r['role']=='train' and r['eligible_for_training']]
    def groups_for(indices):
        flat=indices.reshape(-1)
        held=data['position_state'][flat-1,1]>.5
        reset=data['is_first'][flat]
        groups={'all':np.ones(len(flat),bool),
            'sustained_flat_wait':~reset & ~held & (data['action'][flat-1]==0),
            'held_then_released':~reset & held,
            'flat_add_then_exit_or_reject':~reset & ~held & (data['action'][flat-1]==1),
            'reset':reset}
        for i,end in enumerate(ends):
            begin=int(ends[i-1]) if i else 0
            groups[contract_rows[i]['contract']]=(flat>=begin)&(flat<end)
        return groups
    result=dict(phase=args.phase,mode='FROZEN_READ_ONLY_CPU',source=str(args.source),
        source_manifest_sha256=before,source_counters=counters,thresholds=list(THRESHOLDS),
        actual_quality_thresholds=gates[.3].opening_thresholds,required_passes=20,
        batch_shape=[16,65],horizon=333,imag_length=15,imag_starts_per_batch=1024,
        imagination_includes_final_action=True,training_modified=False,dev_test_accessed=False,
        source_hashes={str(p):file_hash(p) for p in (Path(__file__),
            ROOT/'tools/fg_single3_prior_audit.py',DREAMER/'dreamerv3/agent.py',
            DREAMER/'dreamerv3/rssm.py',DREAMER/'embodied/run/action_mask_warmup.py')})
    if args.phase=='scan':
        starts=np.arange(0,len(data['action'])-64,64)
        if args.limit_batches:
            starts=starts[:16*args.limit_batches]
        probs={'post':[],'prior':[]}
        img_counts={t:{} for t in THRESHOLDS}
        positions=[]
        imag_batches=set(np.linspace(0,(len(starts)-1)//16,min(args.imag_batches,(len(starts)+15)//16),dtype=int))
        for offset in range(0,len(starts),16):
            current=starts[offset:offset+16]
            padded=np.pad(current,(0,16-len(current)),mode='edge')
            batch=native_batch(data,padded)
            i=offset//16
            posterior=predict(params,batch,original_audit_seeds(i))
            p=jax.device_get(posterior['probabilities'])
            for head in probs:
                probs[head].append(p[head][:len(current)].reshape(-1,2))
            indices=(current[:,None]+np.arange(1,65)[None]).reshape(-1)
            positions.append(indices)
            if i in imag_batches:
                full_indices=(padded[:,None]+np.arange(1,65)[None]).reshape(-1)
                origin_groups=groups_for(full_indices)
                # Padded duplicate trajectories must not enter any statistics.
                for key in origin_groups:
                    origin_groups[key][len(current)*64:]=False
                for t in THRESHOLDS:
                    imagined=jax.device_get(imag[t](params,posterior['starts'],seeds(i,True)))
                    add_counts(img_counts[t],imagination_counts(imagined,origin_groups))
            if i==0 or (i+1)%20==0:
                print('SCAN batches',i+1,'/',(len(starts)+15)//16,'seconds',round(time.monotonic()-started,1),flush=True)
        probs={k:np.concatenate(v) for k,v in probs.items()};indices=np.concatenate(positions)
        target=data['action_mask'][indices]>.5
        np.savez_compressed(output/'probabilities.npz',indices=indices,target=target,**probs)
        all_groups=groups_for(indices)
        result['candidates']={}
        for t in THRESHOLDS:
            candidate=dict(metrics={h:metrics(p,target,t) for h,p in probs.items()},
                imagination=finalize_imagination(img_counts[t]),groups={})
            for name,g in all_groups.items():
                candidate['groups'][name]=dict(positions=int(g.sum()),positives=int(target[g,1].sum()),
                    metrics={h:metrics(p[g],target[g],t) for h,p in probs.items()} if g.any() else None)
            result['candidates'][str(t)]=candidate
        result['probability_quantiles']={head:{label:np.quantile(p[target[:,1]==truth,1],[0,.1,.5,.9,1]).tolist()
            if np.any(target[:,1]==truth) else None for label,truth in [('legal_add',True),('illegal_add',False)]}
            for head,p in probs.items()}
        result['unique_positions']=len(indices);result['imagination_batches']=len(imag_batches)
        result['imagination_scope']='stratified batches across the entire chronological TRAIN Replay; not all scoring positions'
        if not args.limit_batches:
            with np.load(args.audit/'online_probabilities.npz') as reference:
                result['prior_audit_max_abs_error']={k:float(np.abs(v-reference[k]).max()) for k,v in probs.items()}
                assert all(v==0 for v in result['prior_audit_max_abs_error'].values())
    else:
        saved=pickle.loads((args.source/'replay.pkl').read_bytes())
        replay=embodied.Replay(length=saved['length'],capacity=saved['capacity'],
            chunksize=saved['chunksize'],online=saved['online'],seed=202)
        restore_replay(replay,saved,args.source/'replay')
        runtime=pickle.loads((args.source/'runtime.pkl').read_bytes())
        from dreamerv3.main import make_stream
        streams=[embodied.streams.Prefetch(make_stream(elements.Config(config),replay,mode)) for mode in ('train','report')]
        for s,ss in zip(streams,runtime['streams']):
            s.load_full(ss);s.enable_deterministic();iter(s)
        lookup={v.tobytes():i for i,v in enumerate(data['stepid'])}
        first_batches=int(round((runtime['schedule']['next_report']-meta['step'])*config['run']['train_ratio']/1024))
        later_batches=int(round(config['run']['fg_report_every_steps']*config['run']['train_ratio']/1024))
        windows=[];batch_starts=[];batch_seeds=[];window_ids=[]
        for window in range(args.windows):
            count=first_batches if window==0 else later_batches
            aggs={t:elements.Agg() for t in THRESHOLDS}
            pool={t:{'post':[],'prior':[]} for t in THRESHOLDS}
            targets=[]
            for j in range(count):
                batch=next(streams[0]);batch={k:v for k,v in batch.items() if k!='seed'}
                start=np.array([lookup[v.tobytes()] for v in batch['stepid'][:,0]])
                expected=native_batch(data,start)
                assert all(np.array_equal(v,expected[k]) for k,v in batch.items())
                index=len(batch_starts);seed=seeds(index)
                batch_starts.append(start);batch_seeds.append(seed);window_ids.append(window)
                posterior=predict(params,batch,seed)
                p=jax.device_get(posterior['probabilities'])
                target=batch['action_mask'][:,1:].reshape(-1,2)>.5;targets.append(target)
                for t in THRESHOLDS:
                    for head in ('post','prior'):
                        met=metrics(p[head].reshape(-1,2),target,t)
                        aggs[t].add({f'avail/{head}_accuracy':met['accuracy'],
                            f'avail/{head}_exact_accuracy':met['exact_accuracy'],
                            f'avail/{head}_fpr':met['fpr']})
                        pool[t][head].append(met)
                # Full real imagination is collected only for the specified candidate;
                # other candidates cannot be certified from its different trajectories.
                if args.imag_threshold is not None:
                    imagined=jax.device_get(imag[args.imag_threshold](params,posterior['starts'],seeds(index,True)))
                    cm=imagination_counts(imagined)['all']
                    aggs[args.imag_threshold].add({'avail/img_fallback_rate':cm['fallback']/cm['decisions'],
                        'avail/img_cardinality':cm['cardinality_sum']/cm['decisions'],
                        'avail/img_action_invalid_rate':cm['invalid']/cm['decisions']})
            step=runtime['schedule']['next_report']+window*config['run']['fg_report_every_steps']
            measured={}
            for t in THRESHOLDS:
                mets=aggs[t].result()
                for head in ('post','prior'):
                    positives=int(sum(m['true_positives']+m['false_negatives'] for m in pool[t][head]))
                    tp=int(sum(m['true_positives'] for m in pool[t][head]))
                    fp=int(sum(m['false_positives'] for m in pool[t][head]))
                    mets.update({f'avail/{head}_add_positives':positives,
                        f'avail/{head}_add_true_positives':tp,
                        f'avail/{head}_add_false_negatives':positives-tp,
                        f'avail/{head}_false_positives':fp,
                        f'avail/{head}_negatives':int(sum((~x).sum() for x in targets)),
                        f'avail/{head}_add_recall':tp/positives if positives else None})
                supported=gates[t]._has_add_support(mets)
                criteria={**METRICS,**ADD_RECALL_METRICS}
                failures=[];not_run=[]
                for name,(key,mode) in criteria.items():
                    if key not in mets:
                        not_run.append(name);continue
                    v=mets[key]
                    if v is None or (v<gates[t].opening_thresholds[name] if mode=='min' else v>gates[t].opening_thresholds[name]):
                        failures.append(name)
                # Never turn missing imagination evidence into an artificial zero.
                gates[t].update(mets,step,allow_activation=False)
                measured[str(t)]=dict(metrics=mets,positive_support_sufficient=supported,
                    failed_conditions=failures,not_run_conditions=not_run,
                    passed=gates[t].last_passed,consecutive_passes=gates[t].consecutive_passes)
            windows.append(dict(report_step=step,train_batches=count,positions=count*1024,candidates=measured))
            for _ in range(config['consec_report']*config['run']['report_batches']):
                next(streams[1])
            write_json(output/'windows.json',windows)
            np.savez_compressed(output/'window_batches.npz',starts=np.array(batch_starts),
                seeds=np.array(batch_seeds),windows=np.array(window_ids))
            print('WINDOW',window+1,step,{t:{k:v[k] for k in ('failed_conditions','not_run_conditions','consecutive_passes')} for t,v in measured.items()},
                'seconds',round(time.monotonic()-started,1),flush=True)
        result['windows']=windows
        result['imagination_threshold']=args.imag_threshold
        result['source_replay_rng_restored']=True;result['source_prefetched_batches_restored']=True
        result['old_gate_aggregates_included']=False
        result['scope']='frozen 180k Replay; exact report cadence and batch shape, no new market data or optimizer updates'
    assert all(file_hash(Path(p))==h for p,h in result['source_hashes'].items())
    assert file_hash(args.source/'manifest.json')==before;validate_bundle(args.source)
    result.update(status='READ_ONLY_PREFLIGHT_COMPLETE',devices=[str(d) for d in jax.devices()],
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_unchanged=True)
    assert all(d.platform=='cpu' for d in jax.devices())
    write_json(output/'analysis.json',result);write_json(output/'status.json',dict(status=result['status']))
    write_json(output/'manifest.json',dict(files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}))
    print('PREFLIGHT_COMPLETE',round(result['elapsed_seconds'],1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--audit',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--phase',choices=['scan','windows'],default='scan')
    parser.add_argument('--limit-batches',type=int,default=0)
    parser.add_argument('--imag-batches',type=int,default=20)
    parser.add_argument('--windows',type=int,default=20)
    parser.add_argument('--imag-threshold',type=float,choices=THRESHOLDS)
    args=parser.parse_args()
    existed=args.output.exists()
    try:
        main(args)
    except BaseException as exc:
        if not existed and args.output.is_dir():
            write_json(args.output/'status.json',dict(status='FAILED',error_type=type(exc).__name__,error=str(exc)))
        raise

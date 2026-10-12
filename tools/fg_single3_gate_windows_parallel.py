#!/usr/bin/env python3
"""Parallel frozen CPU batches, preserving native Replay order and real report windows."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/fg_gate_matplotlib')

import argparse
import concurrent.futures
import json
import multiprocessing
import pickle
from pathlib import Path
import sys
import time

from fg_single3_prior_audit import ROOT,DREAMER,load_replay,make_predict,metrics,write_json
from fg_single3_gate_preflight import native_batch,make_imagination,seeds,imagination_counts,THRESHOLDS
sys.path[:0]=[str(ROOT/'src'),str(DREAMER)]


def initialize(source,shared,threshold):
    """Each process runs the unchanged model on its own read-only mmap view."""
    global DATA,PARAMS,PREDICT,IMAGINE,THRESHOLD
    import numpy as np
    import jax
    from dreamerv3.configuration import read_config_yaml
    DATA={p.stem:np.load(p,mmap_mode='r') for p in Path(shared).glob('*.npy')}
    config=read_config_yaml(Path(source)/'config.yaml')
    state=pickle.loads((Path(source)/'agent.pkl').read_bytes())
    assert not bool(state['params']['avail_actor_gate/value'])
    params={k:v for k,v in state['params'].items() if k.startswith(('enc/','dyn/','avail/','pol/'))}
    PARAMS=jax.device_put(params)
    PREDICT=make_predict(config,DATA,return_features=True)
    IMAGINE=make_imagination(config,DATA,threshold)
    THRESHOLD=threshold
    assert all(d.platform=='cpu' for d in jax.devices())


def score(job):
    import numpy as np
    import jax
    index,starts=job
    batch=native_batch(DATA,starts)
    posterior=PREDICT(PARAMS,batch,seeds(index))
    probs=jax.device_get(posterior['probabilities'])
    assert all(np.isfinite(v).all() for v in probs.values())
    target=batch['action_mask'][:,1:].reshape(-1,2)>.5
    result={str(t):{h:metrics(p.reshape(-1,2),target,t) for h,p in probs.items()} for t in THRESHOLDS}
    imagined=jax.device_get(IMAGINE(PARAMS,posterior['starts'],seeds(index,True)))
    flat=(starts[:,None]+np.arange(1,65)[None]).reshape(-1)
    held=DATA['position_state'][flat-1,1]>.5
    reset=DATA['is_first'][flat]
    legal=target[:,1]
    groups={'all':np.ones(len(flat),bool),
        'positive_sustained_flat_wait':legal&~reset&~held&(DATA['action'][flat-1]==0),
        'positive_held_then_released':legal&~reset&held,
        'positive_flat_add_then_exit_or_reject':legal&~reset&~held&(DATA['action'][flat-1]==1),
        'positive_reset':legal&reset}
    return dict(index=index,candidates=result,negatives=int((~target).sum()),
        imagination=imagination_counts(imagined,groups))


def run(args):
    import elements
    import embodied
    import numpy as np
    from dreamerv3.configuration import read_config_yaml
    from dreamerv3.main import make_stream
    from embodied.run.fg_checkpoint import validate_bundle,file_hash,load_replay as restore_replay
    from embodied.run.action_mask_warmup import ActionMaskWarmup,METRICS,ADD_RECALL_METRICS
    started=time.monotonic();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING'))
    sources=(Path(__file__),ROOT/'tools/fg_single3_gate_preflight.py',
        ROOT/'tools/fg_single3_prior_audit.py',DREAMER/'dreamerv3/agent.py',DREAMER/'dreamerv3/rssm.py')
    source_hashes={str(p):file_hash(p) for p in sources}
    import shutil
    shutil.copyfile(Path(__file__),out/'audited_script.py')
    meta=validate_bundle(args.source);before=file_hash(args.source/'manifest.json')
    config=read_config_yaml(args.source/'config.yaml')
    train=json.loads(Path(config['fg_entry']['manifest']).read_text())
    assert all(r['role']=='train' for r in train['files'])
    data=load_replay(args.source,meta,12)
    shared=out/'shared_replay';shared.mkdir()
    for key,value in data.items():np.save(shared/(key+'.npy'),value)
    shared_hashes={p.name:file_hash(p) for p in shared.iterdir()}
    state=pickle.loads((args.source/'replay.pkl').read_bytes())
    replay=embodied.Replay(length=state['length'],capacity=state['capacity'],chunksize=state['chunksize'],online=state['online'],seed=202)
    restore_replay(replay,state,args.source/'replay')
    runtime=pickle.loads((args.source/'runtime.pkl').read_bytes())
    streams=[embodied.streams.Prefetch(make_stream(elements.Config(config),replay,mode)) for mode in ('train','report')]
    for stream,saved in zip(streams,runtime['streams']):
        stream.load_full(saved);stream.enable_deterministic();iter(stream)
    lookup={v.tobytes():i for i,v in enumerate(data['stepid'])}
    first=int(round((runtime['schedule']['next_report']-meta['step'])*config['run']['train_ratio']/1024))
    later=int(round(config['run']['fg_report_every_steps']*config['run']['train_ratio']/1024))
    starts,window_ids=[],[]
    for window in range(args.windows):
        for _ in range(first if window==0 else later):
            batch=next(streams[0]);batch={k:v for k,v in batch.items() if k!='seed'}
            indices=np.array([lookup[v.tobytes()] for v in batch['stepid'][:,0]])
            expected=native_batch(data,indices)
            assert all(np.array_equal(v,expected[k]) for k,v in batch.items())
            starts.append(indices);window_ids.append(window)
        for _ in range(config['consec_report']*config['run']['report_batches']):next(streams[1])
    np.savez_compressed(out/'window_batches.npz',starts=np.array(starts),
        seeds=np.array([seeds(i) for i in range(len(starts))]),windows=np.array(window_ids))
    print('NATIVE_BATCHES_FROZEN',len(starts),'workers',args.workers,flush=True)
    # Uniform sampling is serial and frozen above. Only independent neural
    # inference is parallel; every batch retains its original index and seed.
    gates={t:ActionMaskWarmup.for_single3_1m(elements.Config(config['action_mask_warmup'])) for t in THRESHOLDS}
    windows=[];completed=0;current_window=0
    if args.continue_from:
        windows=json.loads((args.continue_from/'windows.json').read_text())
        completed=sum(w['train_batches'] for w in windows);current_window=len(windows)
        with np.load(args.continue_from/'window_batches.npz') as saved:
            assert np.array_equal(saved['starts'],np.array(starts[:completed]))
            assert np.array_equal(saved['seeds'],np.array([seeds(i) for i in range(completed)]))
            assert np.array_equal(saved['windows'],np.array(window_ids[:completed]))
        for w in windows:
            for t in THRESHOLDS:
                gates[t].update(w['candidates'][str(t)]['metrics'],w['report_step'],allow_activation=False)
                assert gates[t].consecutive_passes==w['candidates'][str(t)]['consecutive_passes']
        write_json(out/'continued_windows.json',dict(source=str(args.continue_from),windows=current_window,
            batches=completed,files={p.name:file_hash(p) for p in args.continue_from.iterdir() if p.is_file()}))
        print('VERIFIED_CONTINUATION',current_window,'windows',completed,'batches',flush=True)
    continued_count=current_window
    batch_rows=[];imag_groups={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(str(args.source),str(shared),args.threshold)) as pool:
        for row in pool.map(score,enumerate(starts[completed:],start=completed),chunksize=1):
            batch_rows.append(row);completed+=1
            for name,values in row['imagination'].items():
                dest=imag_groups.setdefault(name,{k:0 for k in values})
                for k,v in values.items():dest[k]+=v
            due=completed==len(starts) or window_ids[completed]!=current_window
            if not due:continue
            measured={}
            for t in THRESHOLDS:
                mets={}
                for head in ('post','prior'):
                    head_rows=[r['candidates'][str(t)][head] for r in batch_rows]
                    for k,name in [('accuracy','accuracy'),('exact_accuracy','exact_accuracy'),('fpr','fpr')]:
                        mets[f'avail/{head}_{k}']=float(np.mean([r[name] for r in head_rows]))
                    tp=sum(r['true_positives'] for r in head_rows);fn=sum(r['false_negatives'] for r in head_rows)
                    mets.update({f'avail/{head}_add_positives':tp+fn,f'avail/{head}_add_true_positives':tp,
                        f'avail/{head}_add_false_negatives':fn,f'avail/{head}_add_recall':tp/(tp+fn) if tp+fn else None,
                        f'avail/{head}_false_positives':sum(r['false_positives'] for r in head_rows),
                        f'avail/{head}_negatives':sum(r['negatives'] for r in batch_rows)})
                if t==args.threshold:
                    n=sum(r['imagination']['all']['decisions'] for r in batch_rows)
                    mets.update({'avail/img_fallback_rate':sum(r['imagination']['all']['fallback'] for r in batch_rows)/n,
                        'avail/img_cardinality':sum(r['imagination']['all']['cardinality_sum'] for r in batch_rows)/n,
                        'avail/img_action_invalid_rate':sum(r['imagination']['all']['invalid'] for r in batch_rows)/n})
                step=runtime['schedule']['next_report']+current_window*config['run']['fg_report_every_steps']
                gate=gates[t];gate.update(mets,step,allow_activation=False)
                failures=[];missing=[]
                for name,(key,mode) in {**METRICS,**ADD_RECALL_METRICS}.items():
                    if key not in mets:missing.append(name)
                    elif mets[key] is None or (mets[key]<gate.opening_thresholds[name] if mode=='min' else mets[key]>gate.opening_thresholds[name]):failures.append(name)
                measured[str(t)]=dict(metrics=mets,positive_support_sufficient=gate._has_add_support(mets),
                    failed_conditions=failures,not_run_conditions=missing,passed=gate.last_passed,consecutive_passes=gate.consecutive_passes)
            windows.append(dict(report_step=step,train_batches=len(batch_rows),positions=1024*len(batch_rows),candidates=measured))
            write_json(out/'windows.json',windows)
            print('WINDOW',current_window+1,'streak',gates[args.threshold].consecutive_passes,'seconds',round(time.monotonic()-started,1),flush=True)
            current_window+=1;batch_rows=[]
    assert file_hash(args.source/'manifest.json')==before;validate_bundle(args.source)
    assert all(file_hash(Path(p))==h for p,h in source_hashes.items())
    assert all(file_hash(shared/name)==h for name,h in shared_hashes.items())
    result=dict(status='READ_ONLY_PREFLIGHT_COMPLETE',mode='FROZEN_READ_ONLY_CPU',phase='windows',source=str(args.source),
        source_manifest_sha256=before,source_counters=pickle.loads((args.source/'agent.pkl').read_bytes())['counters'],
        thresholds=list(THRESHOLDS),actual_quality_thresholds=gates[args.threshold].opening_thresholds,required_passes=20,
        batch_shape=[16,65],horizon=333,imag_length=15,imag_starts_per_batch=1024,imagination_includes_final_action=True,
        imagination_threshold=args.threshold,windows=windows,positive_cohort_imagination=imag_groups,
        source_replay_rng_restored=True,source_prefetched_batches_restored=True,old_gate_aggregates_included=False,
        source_unchanged=True,training_modified=False,dev_test_accessed=False,workers=args.workers,
        elapsed_seconds=time.monotonic()-started,scope='frozen native Replay population, real report cadence, no Env steps or optimizer updates',
        source_hashes=source_hashes,continued_windows=continued_count,
        positive_cohort_imagination_scope='only newly computed windows; earlier windows retain full aggregate imagination metrics')
    write_json(out/'analysis.json',result);write_json(out/'status.json',dict(status=result['status']))
    write_json(out/'manifest.json',dict(files={p.name:file_hash(p) for p in out.iterdir() if p.is_file()},
        shared_replay_files=shared_hashes))
    print('PARALLEL_PREFLIGHT_COMPLETE',round(result['elapsed_seconds'],1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--windows',type=int,default=20)
    parser.add_argument('--threshold',type=float,choices=THRESHOLDS,default=.3)
    parser.add_argument('--continue-from',type=Path,help='Verified completed windows from the same native sequence')
    args=parser.parse_args();existed=args.output.exists()
    try:run(args)
    except BaseException as exc:
        if not existed and args.output.is_dir():write_json(args.output/'status.json',dict(status='FAILED',error_type=type(exc).__name__,error=str(exc)))
        raise

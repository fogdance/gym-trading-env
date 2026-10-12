#!/usr/bin/env python3
"""Actual frozen B16x65 loss/metric regression at the original 0.5 threshold."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import subprocess
import sys
import time
from fg_single3_prior_audit import ROOT,DREAMER,model_config
sys.path[:0]=[str(ROOT/'src'),str(DREAMER)]


def run(source,output):
    import elements
    import jax
    import ninjax as nj
    import numpy as np
    from dreamerv3.configuration import read_config_yaml
    from dreamerv3.agent import Agent
    from embodied.jax import internal
    from embodied.run.fg_checkpoint import validate_bundle,file_hash
    started=time.monotonic();output.mkdir(parents=True,exist_ok=False)
    meta=validate_bundle(source);before=file_hash(source/'manifest.json')
    cfg=read_config_yaml(source/'config.yaml');assert cfg['agent']['avail_threshold']==.5
    internal.setup(platform='cpu',compute_dtype='bfloat16',transfer_guard=False,gpuflags=False,prealloc=False)
    old_file=output/'legacy_agent.py'
    old_file.write_bytes(subprocess.check_output(['git','show','09e2e5c:dreamerv3/agent.py'],cwd=DREAMER))
    spec=importlib.util.spec_from_file_location('dreamerv3._gate_metric_legacy',old_file)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    runtime=pickle.loads((source/'runtime.pkl').read_bytes())
    batch={k:v for k,v in runtime['streams'][0]['pending'][0][0].items() if k!='seed'}
    obs={k:elements.Space(batch[k].dtype,batch[k].shape[2:]) for k in (
        'market_seq','position_state','action_mask','reward','is_first','is_last','is_terminal')}
    acts={'action':elements.Space(np.int32,(),0,2)}
    state=pickle.loads((source/'agent.pkl').read_bytes());params=jax.device_put(state['params'])
    results=[]
    for cls in (legacy.Agent,Agent):
        model=object.__new__(cls);cls.__init__(model,obs,acts,model_config(cfg))
        def evaluate(data):
            carry,observation,previous_action,_=model._apply_replay_context(model.init_train(16),data)
            loss,aux=model.loss(carry,observation,previous_action,False)
            return loss,aux[-1]
        pure=nj.pure(evaluate)
        fn=jax.jit(lambda p,b,s:pure(p,b,seed=s)[1])
        results.append(jax.device_get(fn(params,batch,np.array([202,17],np.uint32))))
    old_loss,old_mets=results[0];new_loss,new_mets=results[1]
    differences={k:float(np.max(np.abs(np.asarray(v)-new_mets[k]))) for k,v in old_mets.items()}
    assert float(old_loss)==float(new_loss)
    assert all(v==0 for v in differences.values()),differences
    assert file_hash(source/'manifest.json')==before;validate_bundle(source)
    result=dict(status='REAL_BASELINE_METRICS_ZERO_DIFF_PASS',threshold=.5,batch_shape=[16,65],
        legacy_commit='09e2e5c',legacy_agent_sha256=file_hash(old_file),source_manifest_sha256=before,
        loss=float(old_loss),checked_original_metrics=len(differences),metric_max_abs_errors=differences,
        new_diagnostic_metrics=sorted(set(new_mets)-set(old_mets)),devices=[str(d) for d in jax.devices()],
        source_unchanged=True,source_gradient_updates_performed=0,elapsed_seconds=time.monotonic()-started)
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.source,args.output)

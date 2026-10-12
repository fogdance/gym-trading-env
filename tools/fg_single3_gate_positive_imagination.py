#!/usr/bin/env python3
"""Stratified real imagination from each mutually exclusive legal ADD cohort."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import argparse
import json
from pathlib import Path
import time
import numpy as np
import fg_single3_gate_windows_parallel as window_audit
from fg_single3_gate_preflight import make_imagination,THRESHOLDS
from fg_single3_prior_audit import ROOT,DREAMER


def run(args):
    from dreamerv3.configuration import read_config_yaml
    from embodied.run.fg_checkpoint import validate_bundle,file_hash
    started=time.monotonic();args.output.mkdir(parents=True,exist_ok=False)
    validate_bundle(args.source);before=file_hash(args.source/'manifest.json')
    with np.load(args.audit/'samples.npz') as samples:
        selected=[]
        for name,n in [('reset',7),('flat_add_then_exit_or_reject',3),('held_then_released',3),('sustained_flat_wait',3)]:
            positive=samples['indices'][samples[name]&samples['target'][:,1]]
            chosen=positive[np.linspace(0,len(positive)-1,n,dtype=int)]
            selected.extend(chosen.tolist())
    assert len(selected)==16
    starts=np.clip(np.array(selected)-32,0,179935)
    window_audit.initialize(str(args.source),str(args.shared),.3)
    cfg=read_config_yaml(args.source/'config.yaml')
    rows={}
    for t in THRESHOLDS:
        window_audit.IMAGINE=make_imagination(cfg,window_audit.DATA,t)
        window_audit.THRESHOLD=t
        result=window_audit.score((10000000,starts))
        rows[str(t)]=result['imagination']
    validate_bundle(args.source);assert file_hash(args.source/'manifest.json')==before
    result=dict(status='POSITIVE_COHORT_IMAGINATION_PASS',source=str(args.source),source_manifest_sha256=before,
        batch_shape=[16,65],imag_length=15,decisions_per_trajectory=16,seed=[202,10000000,1],
        starts=starts.tolist(),selected_positive_indices=selected,scope='one stratified causal batch; not a report-window replacement',
        candidates=rows,source_unchanged=True,dev_test_accessed=False,gpu_training_started=False,
        elapsed_seconds=time.monotonic()-started)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(dict(files={'analysis.json':file_hash(args.output/'analysis.json')},
        source_script_sha256=file_hash(Path(__file__))),indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','audit','shared','output'):parser.add_argument('--'+key,type=Path,required=True)
    run(parser.parse_args())

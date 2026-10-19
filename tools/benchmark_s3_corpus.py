#!/usr/bin/env python3
"""Bounded real TRAIN benchmarks using the existing Producer/G3/Reader CLIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PYTHON = '/home/v/miniconda3/envs/dreamerv3/bin/python'
POOL = '/home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/train_pool.json'
ENV_RECEIPT = '/data/logdir/fg_s3_market_minute/20261013_v2/env/v1'
SAMPLES = dict(short=('2021-12-09','2021-12-09',128), long=('2021-12-08','2021-12-09',None))
ENV = dict(PYTHONPATH='.:src:../dreamerv3', JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
    PYTHONDONTWRITEBYTECODE='1', PYTHONINTMAXSTRDIGITS='0', MPLCONFIGDIR='/tmp/fg_data_perf_mpl',
    OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()


def run(repo,out,sample,native):
    repo=Path(repo).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    start_day,end_day,steps=SAMPLES[sample]
    if native and steps is not None:raise ValueError('Native G3 requires an untruncated DATA benchmark')
    records=[];began=time.perf_counter()
    report=dict(protocol='FG_S3_DATA_PERF_BENCHMARK_V1',sample=sample,repo=str(repo),
        repositories={name:dict(sha=git(root,'rev-parse','HEAD'),dirty=bool(git(root,'status','--porcelain')))
            for name,root in [('gym',repo),('dreamer',repo.parent/'dreamerv3')]},
        contract='CZCE.FG201',start_day=start_day,end_day=end_day,max_market_steps=steps,
        seed=202,requested_cpu_workers=20,phases=records,optimizer_updates=0,
        production='BENCHMARK_ONLY',native_g3='NOT_RUN')

    def phase(name,args):
        argv=['taskset','-c','0-19','/usr/bin/time','-v','-o',str(out/(name+'.time.txt')),PYTHON,'-B',*args]
        record=dict(name=name,argv=argv,cwd=str(repo),env_overrides=ENV,
                    start_utc=datetime.now(timezone.utc).isoformat())
        start=time.perf_counter()
        with (out/(name+'.log')).open('x') as log:
            result=subprocess.run(argv,cwd=repo,env={**os.environ,**ENV},stdout=log,stderr=subprocess.STDOUT)
        record.update(wall_seconds=time.perf_counter()-start,exit_code=result.returncode,
            end_utc=datetime.now(timezone.utc).isoformat(),log_sha256=sha(out/(name+'.log')))
        records.append(record);write(out/(name+'.command.json'),record)
        print(json.dumps(dict(phase=name,seconds=record['wall_seconds'],exit_code=result.returncode)),flush=True)
        if result.returncode:raise RuntimeError(f'{name} failed: {out/(name+".log")}')

    base=['--source-pool',POOL,'--env-receipt',ENV_RECEIPT,'--contract','CZCE.FG201',
          '--start-day',start_day,'--end-day',end_day]
    try:
        phase('validate',['tools/fg_s3_e_quality_corpus.py','validate',*base])
        phase('generate',['tools/fg_s3_e_quality_corpus.py','generate',*base,'--workers','20',
            '--seed','202','--mode','dev' if steps else 'data','--debug','--out',str(out/'corpus'),
            *(['--max-market-steps',str(steps)] if steps else [])])
        if native:
            pin=dict(protocol='FG_S3_E_QUALITY_PRODUCER_PIN_V1',repositories=report['repositories'])
            pin['content_hash']=hashlib.sha256(json.dumps(pin,sort_keys=True,default=str).encode()).hexdigest()
            write(out/'PRODUCER_PIN.json',pin)
            phase('native_g3',['tools/fg_s3_e_quality_reader.py','source','--corpus',str(out/'corpus'),
                '--out',str(out/'native'),'--purpose','native-verification','--producer-gym',str(repo),
                '--producer-lock',str(out/'PRODUCER_PIN.json'),'--producer-lock-sha256',sha(out/'PRODUCER_PIN.json'),
                '--workers','20'])
            report['native_g3']='PASS'
        phase('reader',['tools/fg_s3_e_quality_reader.py','read','--corpus',str(out/'corpus'),
            '--out',str(out/'reader'),'--length','64','--seed','202','--without-model'])
        phase('coverage',['tools/fg_s3_e_quality_corpus.py','coverage','--corpus',str(out/'corpus')])
        report.update(status='PASS_BOUNDED_TRAIN_BENCHMARK',generation=read(out/'corpus/performance.json'),
            corpus_identity=read(out/'corpus/corpus_manifest.json')['content_hash'])
        with gzip.open(out/'corpus/policies/P0/steps.jsonl.gz','rt') as stream:
            minutes=sum(bool(json.loads(line)['valid_bar']) for line in stream)
        report['unique_valid_market_minutes']=minutes
    except Exception as error:
        report.update(status='FAILED',error=str(error));raise
    finally:
        report['wall_seconds']=time.perf_counter()-began
        if 'unique_valid_market_minutes' in report:
            report['valid_market_minutes_per_second']=report['unique_valid_market_minutes']/report['wall_seconds']
        write(out/'BENCHMARK_RESULT.json',report)
    print(json.dumps(dict(status=report['status'],seconds=report['wall_seconds'],native_g3=report['native_g3'])))


# These fields bind each artifact to its own validated code, account and
# physical Replay identity. They are checked by native replay/Reader, not
# compared as equal between two different Producers or output directories.
PROVENANCE = frozenset(('content_hash','config_hash','teacher_identity','oracle_identity',
    'state_id','account_snapshot_hash','state_proof','native_event_hash','final_state_hash',
    'e_observation_identity','observation_identity','e_profile_identity','data_identity',
    'stepid','trace_hash','snapshot_identity','value_witness_encoding'))


def canonical(value,key=None):
    if key=='source' and isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value):
        return 'OWN_VALIDATED_TEACHER_SOURCE_IDENTITY'
    if key=='value_rational_witness':
        values=[]
        for text in value:
            if text.startswith('h:'):
                n,d=text[2:].split('/');number=Fraction(int(n,16),int(d,16))
            else:number=Fraction(text)
            values.append([format(number.numerator,'x'),format(number.denominator,'x')])
        return values
    if isinstance(value,dict):return {k:canonical(v,k) for k,v in value.items() if k not in PROVENANCE}
    if isinstance(value,list):return [canonical(v) for v in value]
    if value=='FG_S3_E_QUALITY_TEACHER_V3':return 'FG_S3_E_QUALITY_TEACHER_V2'
    if value=='FG_S3_E_QUALITY_STRICT_SUFFIX_V3':return 'FG_S3_E_QUALITY_STRICT_SUFFIX_V2'
    return value


def compare(left,right,out):
    import numpy as np
    from gym_trading_env.research.fg_replay import iter_sidecar,iter_replay_rows,trace_manifest,E_QUALITY_SCHEMA
    left,right=Path(left),Path(right);policies=[]
    sys.set_int_max_str_digits(0)
    for name in ('P0','P1','P2_low','P2_mid','P3','P4'):
        a,b=left/'policies'/name,right/'policies'/name
        ma,mb=trace_manifest(a,schema=E_QUALITY_SCHEMA),trace_manifest(b,schema=E_QUALITY_SCHEMA)
        count=0
        for (row_a,row_b) in zip(iter_sidecar(a),iter_sidecar(b),strict=True):
            for row in (row_a,row_b):
                facts=row['teacher']['L1']
                for layer in ('L1','L2'):
                    assert facts['layers'][layer]['source']==facts['teacher_identity']
            ca,cb=canonical(row_a),canonical(row_b)
            if ca!=cb:
                differing=[k for k in ca.keys()|cb.keys() if ca.get(k)!=cb.get(k)]
                raise ValueError(f'Semantic sidecar mismatch {name}/{count}: {differing}')
            count+=1
        for (i,aa),(j,bb) in zip(iter_replay_rows(a,ma),iter_replay_rows(b,mb),strict=True):
            assert i==j and aa.keys()==bb.keys()
            for key in aa:
                if key!='stepid':np.testing.assert_array_equal(aa[key],bb[key],err_msg=f'{name}/{i}/{key}')
        assert canonical(read(a/'account_state.json'))==canonical(read(b/'account_state.json')),name
        policies.append(dict(policy=name,rows=count,all_labels='EXACT',all_native_arrays='EXACT',final_account='EXACT'))
    value=dict(status='PASS_FULL_NATIVE_AND_TEACHER_EQUIVALENCE',left=str(left),right=str(right),
        policies=policies,exact_rational_values=True,excluded_provenance_fields=sorted(PROVENANCE),
        teacher_layer_sources_checked_against_own_identity=True,
        financial_or_risk_fields_excluded=False,optimizer_updates=0)
    write(out,value);print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run');p.add_argument('--repo',required=True);p.add_argument('--out',required=True)
    p.add_argument('--sample',choices=SAMPLES,required=True);p.add_argument('--native-g3',action='store_true')
    p=sub.add_parser('compare');p.add_argument('--left',required=True);p.add_argument('--right',required=True)
    p.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='run':run(args.repo,args.out,args.sample,args.native_g3)
    else:compare(args.left,args.right,args.out)

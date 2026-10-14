"""CPU numerical/causal cross-checks against actual read-only FG TRAIN bytes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import time
from types import ModuleType

import numpy as np
import pandas as pd

from .core import MarketData, build_e_data_v1
from .cli import _worker_budget
from .io import git_identity, independent_output, stable_identity, write_json
from .schema import Decision, EDataConfig, RAW_FIELDS
from .sources import pool_entries, read_fg_source


def _validate(entry,pool_record):
    start=time.monotonic()
    source=read_fg_source(entry,pool_record)
    config=EDataConfig(audit_1m=True)
    data=MarketData(source.bars,source.schedule,config)
    # An independent pandas aggregation keyed by FG quarter-hour eob checks
    # every accepted bucket, not just the contexts selected for export.
    f=data.frame.loc[data.frame._legal].copy()
    f['bucket_end']=f.index.ceil('15min')
    grouped=f.groupby('bucket_end').agg(Open=('Open','first'),High=('High','max'),
        Low=('Low','min'),Close=('Close','last'),Volume=('Volume','sum'),OpenInterest=('OpenInterest','last'))
    for bucket in data.complete_buckets:
        row=grouped.loc[pd.Timestamp(bucket['bucket_end'])]
        assert np.array_equal(row[list(RAW_FIELDS)].to_numpy(dtype=float),np.array([bucket[k] for k in RAW_FIELDS]))
    td=data.frame.trading_date.to_numpy()
    changes=np.flatnonzero(td[1:]!=td[:-1])+1
    indices=sorted(set([min(14,len(data.frame)-1),int(changes[0])+14 if len(changes) else min(345,len(data.frame)-1),min(4*345,len(data.frame)-1)]))
    indices=[min(i,len(data.frame)-1) for i in indices]
    facts=[]
    for i in indices:
        cutoff=data.frame.index[i]
        d=Decision(source.contract,cutoff)
        full=data.build([d])
        prefix=build_e_data_v1(data.frame.loc[data.frame.index<=cutoff],[d],source.schedule,config)
        altered=source.bars.copy(deep=True)
        t=pd.DatetimeIndex(pd.to_datetime(altered.Date))
        t=t.tz_localize('Asia/Shanghai') if t.tz is None else t.tz_convert('Asia/Shanghai')
        future=t>cutoff
        for k in RAW_FIELDS[:4]: altered.loc[future,k]*=3
        altered.loc[future,'Volume']*=7;altered.loc[future,'OpenInterest']*=2
        # Prepare complete altered data, rather than relying on the public
        # builder's early cutoff alone, to expose accidental suffix dependence.
        tampered=MarketData(altered,source.schedule,config).build([d])
        for name in ('context_15m','market_seq','market_seq_e_shared','price_anchor','audit_1m_ohlc_shared'):
            assert np.array_equal(getattr(full,name),getattr(prefix,name)),name+' prefix'
            assert np.array_equal(getattr(full,name),getattr(tampered,name)),name+' future'
        assert full.decision_meta==prefix.decision_meta==tampered.decision_meta
        assert full.bucket_manifest==prefix.bucket_manifest==tampered.bucket_manifest
        facts.append(dict(raw_cursor=i,visible_cutoff=cutoff.isoformat(),status='PASS',
                          observation_sha256=hashlib.sha256(full.market_seq.tobytes()).hexdigest()))
    return dict(contract=source.contract,source=source.provenance,
                aggregated_buckets_checked=len(data.complete_buckets),causal_checks=facts,
                seconds=time.monotonic()-start,status='PASS')


def _probe_real(source,interpreter,out):
    td=source.bars.trading_date.astype(str).to_numpy()
    changes=np.flatnonzero(td[1:]!=td[:-1])+1
    indices=sorted(set([0,14,15,int(changes[0])+14 if len(changes) else 345,len(td)-1]))
    project=Path(__file__).resolve().parents[1]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
             MPLCONFIGDIR=str(out/'mpl'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([interpreter,'-B',str(project/'e_data_v1/env_probe.py'),'--env',source.env_config,
        '--indices',*[str(i) for i in indices]],cwd=project,env=env,capture_output=True,text=True,timeout=180)
    if result.returncode: raise ValueError('Original Env probe failed: '+result.stderr)
    probe=json.loads(result.stdout)
    bundle=build_e_data_v1(source.bars,[Decision(source.contract,x['visible_cutoff'],
        market_seq=np.array(x['market_seq'],dtype=np.float32),original_provenance=probe['observation_identity'])
        for x in probe['observations']],source.schedule)
    return dict(contract=source.contract,observation_identity=probe['observation_identity'],
                probe_indices=indices,hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in bundle.market_seq],
                original_hashes=[x['sha256'] for x in probe['observations']],status='PASS')


def _benchmark(source,baseline_file):
    """Compare frozen pre-reuse E snapshot code with the current original-helper path."""
    code=Path(baseline_file).read_bytes()
    module=ModuleType('e_data_v1._benchmark_before_reuse')
    module.__package__='e_data_v1';module.__file__=str(baseline_file)
    exec(compile(code,str(baseline_file),'exec'),module.__dict__)
    config=EDataConfig(audit_1m=True)
    before=module.MarketData(source.bars,source.schedule,config)
    after=MarketData(source.bars,source.schedule,config)
    indices=np.linspace(0,len(after.frame)-1,2000,dtype=int)
    decisions=[Decision(source.contract,after.frame.index[i]) for i in indices]
    start=time.monotonic();baseline=[before.snapshot(d) for d in decisions];old_seconds=time.monotonic()-start
    start=time.monotonic();optimized=[after.snapshot(d) for d in decisions];new_seconds=time.monotonic()-start
    digest=hashlib.sha256()
    for a,b in zip(baseline,optimized):
        for i in (0,1,2,3,4):
            assert np.array_equal(a[i],b[i]),'snapshot optimization changed numeric values'
            digest.update(np.asarray(b[i]).tobytes())
        for key in a[5]: assert a[5][key]==b[5][key],key
    return dict(contract=source.contract,decisions=len(decisions),before_seconds=old_seconds,
                after_seconds=new_seconds,speedup=old_seconds/new_seconds,numeric_sha256=digest.hexdigest(),
                baseline_source=stable_identity(baseline_file),original_metadata_fields_equal=True,
                note='snapshot loop only; before lacks added per-decision baseline provenance, current path includes it',status='PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int)
    p.add_argument('--reference-python',required=True)
    p.add_argument('--snapshot-baseline')
    args=p.parse_args()
    if args.workers is not None and args.workers <= 0:
        p.error('--workers must be positive')
    entries,pool_record=pool_entries(args.source)
    out=independent_output(args.out,inputs=[args.source]);out.mkdir(parents=True)
    start=time.monotonic();reports=[]
    budget=_worker_budget(entries,args)
    with ProcessPoolExecutor(max_workers=budget['workers'],mp_context=get_context('spawn')) as executor:
        tasks=[executor.submit(_validate,e,pool_record) for e in entries]
        for task in as_completed(tasks):
            result=task.result();reports.append(result)
            print(json.dumps(dict(contract=result['contract'],status=result['status'],seconds=result['seconds'])),flush=True)
    reports.sort(key=lambda x:x['contract'])
    source=read_fg_source(entries[0],pool_record)
    probe=_probe_real(source,args.reference_python,out)
    benchmark=_benchmark(source,args.snapshot_baseline) if args.snapshot_baseline else None
    project=Path(__file__).resolve().parents[1]
    report=dict(status='PASS',source_pool=pool_record,contracts=reports,
        aggregated_buckets_checked=sum(x['aggregated_buckets_checked'] for x in reports),
        historical_points_checked=sum(len(x['causal_checks']) for x in reports),
        actual_Env_reference=probe,snapshot_reuse_benchmark=benchmark,
        repositories=dict(gym=git_identity(project),dreamer=git_identity(project.parent/'dreamerv3')),
        seconds=time.monotonic()-start,worker_budget=budget)
    write_json(out/'real_validation.json',report)
    print(json.dumps(dict(status='PASS',report=str(out/'real_validation.json'))))


if __name__=='__main__':
    main()

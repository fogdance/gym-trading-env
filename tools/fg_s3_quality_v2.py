#!/usr/bin/env python3
"""CPU quality catalog diagnostics; training uses fg_s3_v2_experiment.py."""
import argparse
import json
import os
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT.parent/'dreamerv3')]
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/fg_s3_quality_mpl')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['build-train','inspect','validate','summarize'])
    p.add_argument('--config');p.add_argument('--prepared');p.add_argument('--output-dir')
    p.add_argument('--journal');p.add_argument('--catalog');p.add_argument('--steps',type=int,default=100)
    p.add_argument('--beta',type=float,default=1.)
    a=p.parse_args()
    from gym_trading_env.research.fg_replay import read_manifest,write_json,signed,digest
    from gym_trading_env.research.fg_s3_quality_catalog_v2 import coverage,write_catalog_receipt
    if a.command=='build-train':
        from gym_trading_env.research.fg_s3_v2_experiment import prepare,read_settings
        if not a.config or read_settings(a.config)['mode']!='quality_v2': p.error('build-train needs a quality config')
        result=prepare(a.config)
    elif a.command in ('summarize','validate'):
        if a.command=='validate':
            if not a.catalog: p.error('validate requires --catalog for source and ownership admission')
            from gym_trading_env.research.fg_s3_quality_catalog_v2 import validate_catalog
            result=validate_catalog(a.catalog)
            print(json.dumps(result,indent=2,ensure_ascii=False));return
        if not a.journal: p.error('Select a quality journal')
        records=[json.loads(s) for s in Path(a.journal).read_text().splitlines()]
        result=coverage(records)
    else:
        if not a.prepared or not a.output_dir or not 65<=a.steps<=1000: p.error('inspect requires prepared, new output and 65..1000 steps')
        from gym_trading_env.research.fg_s3_v2_experiment import settings_from_prepared
        from gym_trading_env.research.fg_s3_quality_experiment_v2 import load_quality_pool
        from gym_trading_env.research.fg_s3_v2_collector import V2Collector
        from embodied.run.fg_s3_v2_train import execution_code
        settings=settings_from_prepared(a.prepared)
        prepared=read_manifest(Path(a.prepared)/'manifest.json')
        if prepared['code']!=execution_code(): raise ValueError('Quality diagnostic source changed after prepare')
        output=Path(a.output_dir).resolve()
        output.mkdir(parents=True,exist_ok=False)
        query,env=load_quality_pool(settings,a.prepared)
        collector=V2Collector(env,query,output/'rollout',settings['dreamer_root'],beta=a.beta,
            actor_probability=1. if a.beta==0 else 0.,seed=settings['replay_seed'],
            max_env_steps=a.steps,quality_catalog=query)
        try:
            collector.advance(a.steps)
            records=[json.loads(s) for s in collector.journal.read_text().splitlines()]
            write_catalog_receipt(output/'quality_catalog.json',dict(
                fold='TRAIN',score_policy='FG_S3_QUALITY_TP_MARKET_DAY_V1',
                prepared_dir=str(Path(a.prepared).resolve()),
                prepared_manifest_sha256=digest(Path(a.prepared)/'manifest.json'),
                journal_prefix_bytes=collector.journal.stat().st_size,
                journal_prefix_sha256=digest(collector.journal),
                supervision='L1/L2 auxiliary facts; L3 independent strict auditor',
                prepared_sha256=digest(Path(a.prepared)/'manifest.json'),
                journal=str(collector.journal),journal_sha256=digest(collector.journal),
                teacher_identities=sorted(query.teacher_identities),
                source_git=prepared['versions'],source_code=prepared['code']),records)
            result=signed(dict(status='PASS_READ_ONLY_TRAIN_NATIVE_QUALITY_DIAGNOSTIC',
                fold='TRAIN',env_steps=collector.counts['env_steps'],model_updates=0,
                beta=a.beta,actor_kind='SCRIPTED_DIAGNOSTIC; not learned Actor',
                coverage=coverage(records),prepared_sha256=digest(Path(a.prepared)/'manifest.json'),
                catalog_sha256=digest(output/'quality_catalog.json'),
                journal=str(collector.journal),journal_sha256=digest(collector.journal),
                below_300000_rows=sum(float(r['equity_before'])<300000 for r in records),
                gpu='NOT_RUN',counterfactual_pnl='overlapping candidates; not an executable sum'))
            write_json(output/'receipt.json',result)
        finally: collector.close()
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__': main()

"""Publish observed pilot status until budgets and admitted evaluations finish."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_entry_v1 import write_json


def publish(path):
    manifest=json.loads(path.read_text());digest=manifest.pop('content_hash')
    if fingerprint(manifest)!=digest:
        raise ValueError('Manifest changed')
    rows=[];done=True;failed=False
    for instance in manifest['instances']:
        if instance['budget']!='pilot':
            continue
        stages=[]
        for phase in instance['phases']:
            filename=Path(phase['logdir'])/'launch_metadata.json'
            meta=json.loads(filename.read_text()) if filename.exists() else dict(status='NOT_RUN')
            stages.append(meta)
            failed|=meta['status']=='FAILED'
            done&=meta['status']=='COMPLETED_BUDGET'
        audit=stages[-1].get('learning_audit',{})
        evaluation=path.parent/'pilot_evaluation'/instance['method']/str(instance['seed'])/'comparison.json'
        result=None
        if evaluation.exists():
            evaluated=json.loads(evaluation.read_text())
            result=next(r for r in evaluated['results'] if r['strategy']=='dreamerv3')
        if audit.get('actual_learning') and result is None:
            done=False
        status=('EVALUATED' if result else 'NO_POLICY_LEARNING_VERIFIED' if stages[-1]['status']=='COMPLETED_BUDGET'
                and not audit.get('actual_learning') else stages[-1]['status'])
        rows.append(dict(method=instance['method'],seed=instance['seed'],status=status,
                         learning_audit=audit,validation=result,stages=stages))
    write_json(path.parent/'pilot_comparison.json',dict(manifest_hash=digest,results=rows,
        complete=done,failed=failed,formal_status='NOT_RUN',verdict='NO_GENERAL_SUPERIORITY_CLAIM_FROM_SINGLE_SEED'))
    lines=['# Pilot A/B/C comparison','','Single seed engineering pilot; final blind Test remains sealed.','',
        '| Method | Status | Reward/day | TP batches | Open lots | Max floating loss | Equity change |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        result=row['validation']
        values=[result[k] for k in ('reward_per_day','completed_batches','open_lots_at_end','max_floating_loss','net_equity_change')] if result else ['NOT_RUN']*5
        lines.append('| '+ ' | '.join(map(str,[row['method'],row['status'],*values]))+' |')
    lines.append('\nFailed or unverified policy learning is not a timing-performance result. Matched-random partial matches and market correlation prevent unconditional superiority claims.')
    (path.parent/'comparison.md').write_text('\n'.join(lines)+'\n')
    return done or failed


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',default='artifacts/fg_abcd_v1/v1/experiment_manifest.json')
    parser.add_argument('--interval',type=int,default=60)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.interval<1:
        raise ValueError('interval must be positive')
    path=Path(args.manifest).resolve()
    while True:
        subprocess.run([sys.executable,str(Path(__file__).with_name('report_fg_abcd.py')),'--manifest',str(path)],check=True)
        terminal=publish(path)
        if args.once or terminal:
            return
        time.sleep(args.interval)


if __name__=='__main__':
    main()

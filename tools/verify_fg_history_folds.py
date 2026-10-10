"""Standalone integrity check for generated forward research datasets."""
import json
from pathlib import Path
import pandas as pd
from gym_trading_env.research.fg_history_study import digest
from gym_trading_env.research.fg_entry_v1 import write_json
import argparse
parser = argparse.ArgumentParser(description='Verify forward FG event bounds, rewards and source/output hashes.')
parser.add_argument('--run', required=True)
run = Path(parser.parse_args().run)
plan=json.loads((run/'plan.json').read_text()); summary=json.loads((run/'summary.json').read_text()); datasets=json.loads((run/'datasets.json').read_text())
assert len(summary['jobs']) == len(plan['tasks'])
assert plan['signature'] == summary['signature'] == datasets['signature']
assert plan['blind_sealed'] and summary['blind_sealed'] and datasets['blind_sealed']
assert len(datasets['datasets']) == summary['datasets']
lookup={Path(t['directory']).name:t for t in plan['tasks']}
rows=success=censored=terminal=0
for i, task in enumerate(plan['tasks']):
    dates=[]
    for item in task['files']:
        assert digest(item['path']) == item['sha256']
        f=pd.read_csv(item['path'],usecols=['Date','trading_date'])
        dates.extend(f.loc[(f.trading_date>=task['start'])&(f.trading_date<=task['end']),'Date'])
    upper=pd.to_datetime(dates).max()
    lower=pd.to_datetime(dates).min()
    done=json.loads((Path(task['directory'])/'complete.json').read_text())
    assert all(digest(Path(task['directory'])/k)==v for k,v in done['output_hashes'].items())
    for f in pd.read_csv(Path(task['directory'])/'events.csv.gz',chunksize=20000,low_memory=False):
        rows+=len(f)
        assert f.role.eq(task['role']).all()
        assert f.contract.eq(task['contract']).all()
        assert f.source_manifest_sha256.eq(task['manifest_sha256']).all()
        assert f.trading_date.between(task['start'],task['end']).all()
        for col in ['decision_timestamp','entry_timestamp','first_touch_timestamp','observation_end']:
            t=pd.to_datetime(f[col].dropna())
            assert (t>=lower).all() and (t<=upper).all(), (i,col)
        good=f.completion_status.eq('success')
        success+=int(good.sum())
        assert ((f.loc[good,'reward']-60/(60+f.loc[good,'duration_minutes'])).abs()<1e-12).all()
        c=f.completion_status.eq('right_censored'); censored+=int(c.sum())
        assert f.loc[c,'reward'].isna().all()
        other=~good&~c; terminal+=int(other.sum())
        assert f.loc[other,'reward'].eq(0).all()
    print(f'verified {i+1}/{len(plan["tasks"])}',flush=True)
for d in datasets['datasets']:
    train=[lookup[k] for k in d['jobs']['train']]
    val=[lookup[k] for k in d['jobs']['validation']]
    assert max(t['end'] for t in train)<=d['cutoff']<min(t['start'] for t in val)
    assert all(t['end'] < t['blind_start'] for t in val)
result=dict(datasets=len(datasets['datasets']),jobs=len(plan['tasks']),unique_event_rows=rows,success=success,right_censored=censored,terminal=terminal,
    timestamp_bounds_passed=True,reward_identity_passed=True,output_hashes_passed=True,blind_sealed=True)
write_json(run/'verification.json',result)
print(result)

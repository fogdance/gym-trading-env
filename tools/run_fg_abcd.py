"""Print or run the approved pilot; formal runs remain command templates."""
import argparse
import json
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import sys

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.entry_pipeline import file_hash
from gym_trading_env.research.fg_entry_v1 import write_json


def command(instance, phase, dreamer_root, transfer=''):
    return [sys.executable,'-u','-m','dreamerv3.main','--configs','defaults','fg_entry_v1','fg_entry_dataset_v1',
        '--fg_entry.manifest',phase['dataset']['manifest'],'--env.gymnasium.config_path',phase['dataset']['selector'],
        '--logdir',phase['logdir'],'--experiment_seed',str(instance['seed']),'--dreamer.seed',str(instance['seed']),
        '--run.steps',str(phase['steps']),'--jax.platform','cuda','--jax.prealloc','False',
        '--logger.outputs','jsonl','tensorboard','--run.fg_learning_audit','True',
        '--run.fg_stage_transfer',transfer]


def transfer_receipt(instance, source, target):
    root=Path(source['logdir'])
    checkpoint=root/'ckpt'/(root/'ckpt/latest').read_text().strip()
    if pickle.loads((checkpoint/'step.pkl').read_bytes())!=source['steps']:
        raise ValueError('Pretrain budget has not completed')
    files=[root/'config.yaml',Path(source['dataset']['manifest']),Path(target['dataset']['manifest'])]
    files.extend(p for p in checkpoint.iterdir() if p.is_file())
    receipt=dict(method='C',fold=instance['fold'],seed=instance['seed'],source_steps=source['steps'],
        checkpoint=str(checkpoint),source_config=str(root/'config.yaml'),target_manifest=target['dataset']['manifest'],
        input_hashes={str(p):file_hash(p) for p in files},local_recent_steps=0,total_step_offset=source['steps'],
        replay_protocol='EMPTY_RECENT_ONLY',accounts_protocol='NEW_RECENT_WITH_PRETRAIN_ACCOUNTS_ARCHIVED')
    receipt['content_hash']=fingerprint(receipt)
    path=root/'c_transfer_receipt.json'
    write_json(path,receipt)
    return str(path)


def training_coverage(phase):
    import pandas as pd
    from gym_trading_env.envs.entry_contract import DirectionSchedule,read_yaml,timestamp
    root=Path(phase['logdir'])
    checkpoint=root/'ckpt'/(root/'ckpt/latest').read_text().strip()
    driver=pickle.loads((checkpoint/'driver.pkl').read_bytes())
    state=driver['envs'][0]['env']
    if state['schema']!=2:
        raise ValueError('Pilot did not save the complete scheduler bank')
    manifest=json.loads(Path(phase['dataset']['manifest']).read_text())
    records=[r for r in manifest['files'] if r['role']=='train' and r['eligible_for_training']]
    schedule=DirectionSchedule(read_yaml(manifest['direction_path'])['directions'])
    coverage=[]
    for i,row in enumerate(records):
        advanced=state['advanced'][i]
        frame=pd.read_csv(row['path'])
        cycle,prefix=divmod(advanced,len(frame)-1)
        groups={}
        for part,weight in ((frame.iloc[1:],cycle),(frame.iloc[1:prefix+1],1)):
            if not weight:
                continue
            for _,bar in part.iterrows():
                year=str(pd.Timestamp(str(bar.trading_date)).year) if not str(bar.trading_date).isdigit() else str(bar.trading_date)[:4]
                direction=schedule.at(row['contract'],timestamp(bar.Date),str(bar.trading_date))
                key=f'{year}/{direction}'
                groups[key]=groups.get(key,0)+weight
        coverage.append(dict(contract=row['contract'],advanced_bars=advanced,full_passes=cycle,
            current_prefix_bars=prefix,year_direction_bars=groups,visited=advanced>0,
            saved_account=i in state['accounts'] or str(i) in state['accounts']))
    audit=json.loads((root/'learning_audit.json').read_text())
    result=dict(driver_interactions=phase['steps'],actual_advanced_bars=sum(state['advanced']),
        contracts=coverage,visited_contracts=sum(r['visited'] for r in coverage),
        learning_updates=audit['updates'],training_samples=audit['updates']*16*64,
        replay_sample_to_advanced_bar_ratio=audit['updates']*16*64/max(1,sum(state['advanced'])),
        accounting='Independent per-contract training accounts; not pooled-investor performance')
    write_json(root/'training_coverage.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',default='artifacts/fg_abcd_v1/v1/experiment_manifest.json')
    parser.add_argument('--budget',choices=['pilot','formal'],default='pilot')
    parser.add_argument('--method',choices=['A','B','C','all'],default='all')
    parser.add_argument('--fold')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--create-transfer',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--dreamer-root',default='/home/v/Documents/work/dreamerv3')
    args=parser.parse_args()
    if args.execute and args.budget=='formal':
        raise ValueError('Formal 1M launch is not authorized by this preparation task')
    manifest=json.loads(Path(args.manifest).read_text())
    digest=manifest.pop('content_hash')
    if fingerprint(manifest)!=digest:
        raise ValueError('Experiment manifest changed')
    selected=[r for r in manifest['instances'] if r['budget']==args.budget and args.method in (r['method'],'all')]
    selected=[r for r in selected if (args.fold is None or args.fold in (r['fold'],r['fold'].split('.')[-1]))
              and (args.seed is None or args.seed==r['seed'])]
    if not selected:
        raise ValueError('No matching frozen experiment instances')
    if args.create_transfer and (len(selected)!=1 or selected[0]['method']!='C' or args.execute):
        raise ValueError('Receipt preparation requires one C instance and no training execution')
    env=dict(os.environ,JAX_PLATFORMS='cuda',PYTHONDONTWRITEBYTECODE='1',
        MPLCONFIGDIR='/tmp/fg-matplotlib',PYTHONPATH=args.dreamer_root+':'+str(Path(__file__).resolve().parents[1]/'src'),
        JAX_COMPILATION_CACHE_DIR='/data/logdir/fg_abcd_v1/jax_cache')
    if args.execute:
        subprocess.run([sys.executable,'-c','import jax; d=jax.devices(); assert d and all(x.platform=="gpu" for x in d); print(d)'],env=env,check=True)
    for instance in selected:
        source=None
        for phase in instance['phases']:
            transfer=''
            if source:
                if args.execute:
                    transfer=transfer_receipt(instance,source,phase)
                else:
                    transfer=(transfer_receipt(instance,source,phase) if args.create_transfer else
                              str(Path(source['logdir'])/'c_transfer_receipt.json'))
            cmd=command(instance,phase,args.dreamer_root,transfer)
            print(shlex.join(cmd),flush=True)
            if args.execute:
                root=Path(phase['logdir']);root.mkdir(parents=True,exist_ok=True)
                metadata=dict(instance=instance,phase=phase,transfer_receipt=transfer,
                    code_sha={name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,text=True).strip()
                              for name,directory in [('env',Path(__file__).resolve().parents[1]),('dreamer',args.dreamer_root)]},
                    experiment_manifest_hash=digest,status='RUNNING')
                write_json(root/'launch_metadata.json',metadata)
                try:
                    with (root/'launch.log').open('a') as log:
                        subprocess.run(cmd,cwd=args.dreamer_root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                except subprocess.CalledProcessError as error:
                    metadata.update(status='FAILED',returncode=error.returncode,log=str(root/'launch.log'))
                    write_json(root/'launch_metadata.json',metadata)
                    raise
                metadata['status']='COMPLETED_BUDGET'
                metadata['learning_audit']=json.loads((root/'learning_audit.json').read_text())
                metadata['training_coverage']=training_coverage(phase)
                write_json(root/'launch_metadata.json',metadata)
            source=phase
        if args.execute:
            final=instance['phases'][-1]
            learning=json.loads((Path(final['logdir'])/'learning_audit.json').read_text())
            if learning['actual_learning']:
                fold=next(f for f in manifest['folds'] if f['validation']==instance['fold'])
                root=Path(final['logdir'])
                checkpoint=root/'ckpt'/(root/'ckpt/latest').read_text().strip()
                evaluator=Path(__file__).resolve().with_name('evaluate_fg_abcd.py')
                result_root=Path(args.manifest).resolve().parent/'pilot_evaluation'/instance['method']/str(instance['seed'])
                evaluation=[sys.executable,str(evaluator),'--manifest',str(Path(args.manifest).resolve()),
                    '--env-config',fold['validation_env'],'--model-config',str(root/'config.yaml'),
                    '--checkpoint',str(checkpoint),'--output',str(result_root),'--seed',str(instance['seed'])]
                subprocess.run(evaluation,env=env,check=True)
            else:
                print(f"{instance['method']}: policy learning gate failed; no formal comparison claimed",flush=True)


if __name__=='__main__':
    main()

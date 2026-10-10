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


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',default='artifacts/fg_abcd_v1/v1/experiment_manifest.json')
    parser.add_argument('--budget',choices=['pilot','formal'],default='pilot')
    parser.add_argument('--method',choices=['A','B','C','all'],default='all')
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
                    transfer=str(Path(source['logdir'])/'c_transfer_receipt.json')
            cmd=command(instance,phase,args.dreamer_root,transfer)
            print(shlex.join(cmd),flush=True)
            if args.execute:
                root=Path(phase['logdir']);root.mkdir(parents=True,exist_ok=True)
                with (root/'launch.log').open('a') as log:
                    subprocess.run(cmd,cwd=args.dreamer_root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            source=phase


if __name__=='__main__':
    main()

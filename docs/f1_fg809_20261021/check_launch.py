"""Check the frozen F1 selection and production certificate; no Agent/GPU."""
import argparse
import json
from pathlib import Path
import subprocess

from gym_trading_env.research.fg_replay import digest, read_manifest
from gym_trading_env.research.fg_s3_e_quality_foundation import read_settings, preflight


CONTROL = Path('/data/logdir/fg_single3/s3_v2/F1_FG809_12M_U10000_20261021')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-only',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    config=repo/'configs/experiments/fg_s3_v2/e_quality_f1_fg809_size12m_u10000_20261021.yaml'
    lock=read_manifest(CONTROL/'LAUNCH_LOCK.certified.json')
    settings=read_settings(config)
    if digest(config)!=lock['config_sha256'] or settings['corpus_certification']!=lock['corpus_certification']:
        raise ValueError('F1_FROZEN_SELECTION_CHANGED')
    for role,root in [('gym',repo),('dreamer',repo.parent/'dreamerv3')]:
        head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        if head!=lock['repositories'][role]['sha']:
            raise ValueError('F1_FROZEN_TRAINING_HEAD_CHANGED: '+role)
        if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
            raise ValueError('F1_FROZEN_TRAINING_WORKTREE_CHANGED: '+role)
    for key in ('prepared_dir','run_dir'):
        if not Path(settings[key]).is_relative_to('/data/logdir'):
            raise ValueError('F1_OUTPUT_MUST_USE_DATA_LOGDIR')
    if args.config_only:
        return dict(status='F1_FROZEN_CONFIGURATION_VERIFIED',source_purpose='certified-corpus')
    prep=preflight(settings,purpose='certified-corpus')
    return dict(status='F1_CERTIFIED_SOURCE_READY',settings=settings,
        compatibility=prep.receipt['compatibility'],source_receipt=prep.source.receipt,
        verification_mode=prep.source.native_result['verification_mode'],
        fresh_native_g3=False,agent_created=False,optimizer_updates=0)


if __name__=='__main__':
    print(json.dumps(main(),indent=2,ensure_ascii=False))

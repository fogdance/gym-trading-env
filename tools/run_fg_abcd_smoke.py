"""Formal-size GPU smoke: 1120 pretrain interactions + 920 recent = 2040."""
import argparse
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

from run_fg_abcd import command,transfer_receipt


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--logdir',default='/data/logdir/fg_abcd_v1/v1/smoke_v2')
    args=parser.parse_args()
    root=Path(args.logdir)
    if root.exists():
        raise FileExistsError('Use a new smoke directory; existing results are preserved')
    manifest=json.loads(Path('artifacts/fg_abcd_v1/v1/experiment_manifest.json').read_text())
    instance=next(r for r in manifest['instances'] if r['budget']=='pilot' and r['method']=='C')
    instance=dict(instance,phases=[dict(p) for p in instance['phases']])
    source,target=instance['phases']
    source.update(logdir=str(root/'pretrain'),steps=1120)
    target.update(logdir=str(root/'recent'),steps=920)
    dreamer='/home/v/Documents/work/dreamerv3'
    env=dict(os.environ,JAX_PLATFORMS='cuda',PYTHONDONTWRITEBYTECODE='1',
        MPLCONFIGDIR='/tmp/fg-matplotlib',PYTHONPATH=dreamer+':'+str(Path(__file__).resolve().parents[1]/'src'),
        JAX_COMPILATION_CACHE_DIR='/data/logdir/fg_abcd_v1/jax_cache')
    subprocess.run([sys.executable,'-c','import jax; d=jax.devices(); assert d and all(x.platform=="gpu" for x in d); print(d)'],env=env,check=True)
    def run(phase,transfer=''):
        folder=Path(phase['logdir']);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'launch.log').open('a') as log:
            subprocess.run(command(instance,phase,dreamer,transfer),cwd=dreamer,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    run(dict(source,steps=1110))
    checkpoint=Path(source['logdir'])/'ckpt'/(Path(source['logdir'])/'ckpt/latest').read_text().strip()
    before=dict(step=pickle.loads((checkpoint/'step.pkl').read_bytes()),
                driver=pickle.loads((checkpoint/'driver.pkl').read_bytes()))
    (root/'before_resume.json').write_text(json.dumps(before,default=str))
    run(source)  # Same-logdir full checkpoint resume, only ten more interactions.
    receipt=transfer_receipt(instance,source,target)
    run(target,receipt)
    print('GPU smoke completed: formal batch16 x length64; resume + C transfer; total2040',flush=True)


if __name__=='__main__':
    main()

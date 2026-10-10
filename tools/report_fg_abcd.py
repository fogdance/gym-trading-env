"""Materialize reviewable configuration matrix and observed run status."""
import argparse
import json
from pathlib import Path
import pickle

from gym_trading_env.envs.entry_contract import fingerprint


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',default='artifacts/fg_abcd_v1/v1/experiment_manifest.json')
    args=parser.parse_args()
    path=Path(args.manifest)
    manifest=json.loads(path.read_text());digest=manifest.pop('content_hash')
    if fingerprint(manifest)!=digest:
        raise ValueError('Manifest integrity mismatch')
    matrix=['# ABC configuration matrix','','Manifest: '+digest,'',
        '| Fold | Budget | Method | Seed | Phase | Interactions | Dataset | Status |',
        '|---|---|---|---:|---|---:|---|---|']
    pilot=['# Pilot observed status','','No market-performance claim from smoke or a single seed.','',
        '| Method | Phase | Budget | Checkpoint step | Updates | Actor gate | Status |',
        '|---|---|---:|---:|---:|---|---|']
    for instance in manifest['instances']:
        for phase in instance['phases']:
            root=Path(phase['logdir']);metadata=root/'launch_metadata.json'
            status=json.loads(metadata.read_text())['status'] if metadata.exists() else 'NOT_RUN'
            matrix.append('| '+ ' | '.join(map(str,[instance['fold'],instance['budget'],instance['method'],instance['seed'],
                phase['phase'],phase['steps'],phase['dataset']['selector'],status]))+' |')
            if instance['budget']=='pilot':
                step=0;latest=root/'ckpt/latest'
                if latest.exists():
                    checkpoint=root/'ckpt'/latest.read_text().strip()
                    step=pickle.loads((checkpoint/'step.pkl').read_bytes())
                audit=root/'learning_audit.json'
                observed=json.loads(audit.read_text()) if audit.exists() else {}
                pilot.append('| '+ ' | '.join(map(str,[instance['method'],phase['phase'],phase['steps'],step,
                    observed.get('updates','NOT_VERIFIED'),observed.get('actor_gate_enabled','NOT_VERIFIED'),status]))+' |')
    (path.parent/'train_config_matrix.md').write_text('\n'.join(matrix)+'\n')
    (path.parent/'pilot_results.md').write_text('\n'.join(pilot)+'\n')
    assumptions=['# Frozen ABC assumptions','',
        '- Human directions: conditional_direction_only; ex-ante availability not established.',
        '- Main qualification exit is not physical delivery expiry; advance announcement not established.',
        '- Calendar comes from observed user-confirmed bars; not independently exchange-verified.',
        '- Fees/margin constant from unified master, not a historical exchange rate schedule.',
        '- Blind prices excluded before loading; sealed Test metadata only.',
        '- B24 requires whole lifecycles; latest B has five contracts, actual start 2022-08-11.',
        '- Separate training accounts per contract; their returns are not pooled-investor performance.',
        '- Formal templates are NOT_RUN unless a later explicit launch is recorded.',
        '- Training steps count reset observations; actual advanced bars are separately audited.',
        '- Market uncertainty cannot be inferred from matched-random seed variability.',
        '', 'Manifest: '+digest]
    (path.parent/'assumptions.md').write_text('\n'.join(assumptions)+'\n')
    print(path.parent/'pilot_results.md')


if __name__=='__main__':
    main()

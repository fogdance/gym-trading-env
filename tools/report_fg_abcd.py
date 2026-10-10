"""Materialize reviewable configuration matrix and observed run status."""
import argparse
import json
from pathlib import Path
import pickle

from gym_trading_env.envs.entry_contract import fingerprint
from run_fg_abcd import phase_metadata, checkpoint_path


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',default='artifacts/fg_abcd_v1/v3_shared_prefix/experiment_manifest.json')
    args=parser.parse_args()
    path=Path(args.manifest)
    manifest=json.loads(path.read_text());digest=manifest.pop('content_hash')
    if fingerprint(manifest)!=digest:
        raise ValueError('Manifest integrity mismatch')
    matrix=['# ABC configuration matrix','','Manifest: '+digest,'',
        '| Fold | Budget | Method | Seed | Phase | Interactions | Dataset | Status |',
        '|---|---|---|---:|---|---:|---|---|']
    pilot=['# Pilot observed status','','No market-performance claim from smoke or a single seed.','',
        '| Method | Phase | Budget | Checkpoint step | Logged step | Updates | Actor gate | Status |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for instance in manifest['instances']:
        for phase in instance['phases']:
            root=Path(phase['logdir']);metadata=root/'launch_metadata.json'
            record=phase_metadata(instance,phase)
            status=record['status']
            matrix.append('| '+ ' | '.join(map(str,[instance['fold'],instance['budget'],instance['method'],instance['seed'],
                phase['phase'],phase['steps'],phase['dataset']['selector'],status]))+' |')
            if instance['budget']=='pilot':
                step=0;latest=root/'ckpt/latest'
                checkpoint=checkpoint_path(phase,required=False)
                if checkpoint:
                    step=pickle.loads((checkpoint/'step.pkl').read_bytes())
                audit=root/'learning_audit.json'
                observed=record.get('learning_audit',{}) if phase.get('full_checkpoint') else json.loads(audit.read_text()) if audit.exists() else {}
                logged_step=step
                metrics=root/'metrics.jsonl'
                if metrics.exists():
                    for line in metrics.read_text().splitlines():
                        try:
                            record=json.loads(line)
                        except json.JSONDecodeError:
                            continue  # A live writer may have an incomplete final line.
                        logged_step=max(logged_step,min(phase['steps'],int(record.get('step',0))))
                pilot.append('| '+ ' | '.join(map(str,[instance['method'],phase['phase'],phase['steps'],step,logged_step,
                    observed.get('updates','NOT_VERIFIED'),observed.get('actor_gate_enabled','NOT_VERIFIED'),status]))+' |')
    (path.parent/'train_config_matrix.md').write_text('\n'.join(matrix)+'\n')
    (path.parent/'pilot_results.md').write_text('\n'.join(pilot)+'\n')
    smoke_path=path.parent/'smoke_audit.json'
    smoke_status=json.loads(smoke_path.read_text()).get('status','UNKNOWN') if smoke_path.exists() else 'NOT_RUN'
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
        '- A/C share an identical protected prefix; compare their same-seed results as paired observations.',
        f'- Full checkpoints use fixed 5000-step reports and 10000-step saves; observed GPU smoke: {smoke_status}.',
        '- Market uncertainty cannot be inferred from matched-random seed variability.',
        '', 'Manifest: '+digest]
    (path.parent/'assumptions.md').write_text('\n'.join(assumptions)+'\n')
    print(path.parent/'pilot_results.md')


if __name__=='__main__':
    main()

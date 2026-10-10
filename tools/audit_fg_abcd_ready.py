"""Audit prepared ABC configurations without creating a model or starting training."""
import argparse
import json
from pathlib import Path
from tools.run_fg_abcd import command, resolve_command
from gym_trading_env.envs.entry_contract import fingerprint
from dreamerv3.fg_entry_guard import validate_fg_entry
from embodied.run.fg_checkpoint import provenance


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', default='artifacts/fg_abcd_v1/v3_shared_prefix/experiment_manifest.json')
    args = parser.parse_args()
    path = Path(args.manifest)
    manifest = json.loads(path.read_text())
    checksum = manifest.pop('content_hash')
    if fingerprint(manifest) != checksum:
        raise ValueError('Experiment manifest changed')
    identities, physical, count = {}, set(), 0
    for instance in manifest['instances']:
        for phase in instance['phases']:
            cfg = resolve_command(command(instance, phase, '/home/v/Documents/work/dreamerv3'), '/home/v/Documents/work/dreamerv3')
            # C's actual parent cannot be validated before its checkpoint exists.
            # Audit common data/learning admission here; launch binds the receipt.
            cfg = cfg.update({'run.fg_source_steps': 0, 'run.fg_stage_transfer': ''})
            validate_fg_entry(cfg)
            identity = provenance(cfg)
            key = phase['logdir']
            if key in identities and identities[key] != identity:
                raise ValueError(f'Shared training identity mismatch: {key}')
            identities[key] = identity
            count += 1
            if not phase.get('shared_source'):
                physical.add(key)
    result = dict(status='PREPARED_NOT_RUN', manifest_hash=checksum,
        logical_instances=len(manifest['instances']), logical_phases=count,
        unique_training_lines=len(physical), same_line_identity_checks='PASS',
        guard_checks=count, C_origin='VALIDATED_AT_LAUNCH', gpu_smoke='NOT_RUN', formal='NOT_RUN')
    (path.parent / 'readiness_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run the frozen S3 evaluator with native E DEV input and isolated output.

The trained model, checkpoint validators and evaluation loop are imported from
the run's exact frozen source. Only the DEV Env factory is extended for E.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--eval-steps', type=int, default=5000)
    a = p.parse_args()
    if not 1 <= a.eval_steps <= 10000:
        p.error('Actor-only evaluation is bounded to 1..10000 market minutes')
    root, prepared, output = (Path(x).resolve() for x in
                             (a.run_dir, a.prepared, a.output_dir))
    if output.is_relative_to(root) or output.is_relative_to(prepared):
        p.error('Evaluation output must be independent of the trained run and preparation')
    identity = json.loads((root/'run_identity.json').read_text())['identity']
    candidates = [Path(x) for x in identity['code']
                  if Path(x).name == 'fg_s3_v2_experiment.py' and Path(x).parent.name == 'tools']
    if len(candidates) != 1:
        raise ValueError('Expected exactly one authenticated frozen trainer entry')
    cli = candidates[0]
    if hashlib.sha256(cli.read_bytes()).hexdigest() != identity['code'][str(cli)]:
        raise ValueError('Frozen trainer entry changed')
    gym = cli.parents[1]
    sys.path[:0] = [str(gym/'src'), str(gym), str(gym.parent/'dreamerv3')]
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ.pop('JAX_PLATFORM_NAME', None)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Match the existing frozen CLI's CUDA library handling.
    if os.environ.get('LD_LIBRARY_PATH'):
        clean = dict(os.environ)
        clean.pop('LD_LIBRARY_PATH')
        os.execve(sys.executable, [sys.executable, *sys.argv], clean)

    from gym_trading_env.research.fg_replay import read_manifest, signed, write_json, check_files
    from gym_trading_env.envs.entry_contract import fingerprint
    from gym_trading_env.research import fg_single3 as single3
    from gym_trading_env.research import fg_s3_v2_evaluation as evaluator
    from gym_trading_env.research.fg_single3_evaluation import require_dev
    from gym_trading_env.research.fg_s3_e_observation import observation_contract
    from embodied.run.fg_checkpoint import file_hash
    import pandas as pd

    if Path(evaluator.__file__).resolve() != gym/'src/gym_trading_env/research/fg_s3_v2_evaluation.py':
        raise ValueError('Evaluation did not import the trained frozen runtime')
    identity = read_manifest(root/'run_identity.json')['identity']
    settings = read_manifest(prepared/'settings.json')['settings']
    name = (root/'ckpt/latest').read_text().strip()
    if Path(name).name != name or not name:
        raise ValueError('Invalid checkpoint pointer')
    node = root/'ckpt'/name
    if not (node/'done').is_file():
        raise ValueError('Latest checkpoint is incomplete')
    checkpoint_hash = file_hash(node/'manifest.json')
    protected = {str(f): file_hash(f) for f in
        (root/'run_identity.json', root/'training_receipt.json', root/'ckpt/latest',
         node/'manifest.json', prepared/'manifest.json')}
    source = read_manifest(settings['dev_source_manifest'])
    sealed = {Path(r['path']).resolve() for r in source['files'] if r['role'] == 'test'}
    oracle = Path(settings['oracle_receipt']).resolve()
    denied = []

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path in sealed or path.is_relative_to(oracle):
            denied.append(str(path))
            raise PermissionError('Actor-only evaluation refuses TEST or Oracle data: '+str(path))
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if writing and (path.is_relative_to(root) or path.is_relative_to(prepared)):
            raise PermissionError('Actor-only evaluation refuses writes to the trained run')

    sys.addaudithook(guard)
    output.mkdir(parents=True, exist_ok=False)
    view = output/'run_view'
    (view/'ckpt').mkdir(parents=True)
    (view/'run_identity.json').symlink_to(root/'run_identity.json')
    (view/'ckpt'/name).symlink_to(node, target_is_directory=True)
    (view/'ckpt/latest').write_text(name)
    write_json(output/'launch.json', signed(dict(
        trained_run=str(root), prepared=str(prepared), checkpoint=str(node),
        checkpoint_manifest_sha256=checkpoint_hash, eval_steps=a.eval_steps,
        frozen_runtime=str(gym), adapter=str(Path(__file__).resolve()),
        adapter_sha256=file_hash(__file__), protected_files=protected,
        env_extension='native E DEV profile; same authenticated bytes passed as df',
        model_and_evaluation_loop='unchanged frozen runtime', test='SEALED_NOT_READ')))
    original_load_env = evaluator.load_env
    original_make_agent = evaluator.make_agent
    original_env_class = single3.FixedDirectionEntryEnv
    profiles, models = [], []

    def make_agent(*args, **kwargs):
        model = original_make_agent(*args, **kwargs)
        models.append(model)
        return model

    def load_env(row, frozen):
        if not settings.get('resolved_composition', {}).get('use_e'):
            return original_load_env(row, frozen)
        dev = require_dev(Path(row['env_config']).parent/'dev_manifest.json')
        if dev['rows'] != [row]:
            raise ValueError('E evaluation requires the authenticated DEV member')
        path = Path(row['path']).resolve(strict=True)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row['sha256']:
            raise ValueError('E DEV source changed before CSV parsing')
        receipt = dict(protocol='FG_S3_E_DEV_OBSERVATION_V1', role='DEV',
            sources={row['contract']: {'market': {'path': str(path), 'sha256': row['sha256']}}},
            dev_manifest_sha256=file_hash(Path(row['env_config']).parent/'dev_manifest.json'),
            observation=observation_contract(), training_authorization=False)
        profile = SimpleNamespace(identity=fingerprint(receipt),
            contracts=frozenset([row['contract']]), source_receipt=receipt)
        frame = pd.read_csv(io.BytesIO(payload))

        def factory(**kwargs):
            return original_env_class(df=frame, e_profile=profile, **kwargs)

        # Retain the original load_env path/frozen-file checks. Its constructor
        # receives the existing native E profile and one authenticated frame.
        with patch.object(single3, 'FixedDirectionEntryEnv', factory):
            env = original_load_env(row, frozen)
        profiles.append(dict(identity=profile.identity, source_receipt=receipt,
            observation_shapes={k:list(v.shape) for k,v in env.observation_space.spaces.items()},
            loaded_model_updates=int(models[0].n_updates)))
        write_json(Path(row['env_config']).parent/'e_dev_profile.json', signed(profiles[-1]))
        return env

    with patch.object(evaluator, 'load_env', load_env), patch.object(evaluator, 'make_agent', make_agent):
        result = evaluator.evaluate(prepared, view, a.eval_steps)
    check_files(protected)
    if Path(result['checkpoint']).resolve() != node or result['checkpoint_manifest_sha256'] != checkpoint_hash:
        raise ValueError('Evaluation checkpoint differs from the selected latest checkpoint')
    if profiles and int(models[0].n_updates) != profiles[0]['loaded_model_updates']:
        raise ValueError('Actor-only evaluation changed the model update counter')
    proof = signed(dict(status='COMPLETE', receipt=str(Path(result['output'])/'receipt.json'),
        receipt_sha256=file_hash(Path(result['output'])/'receipt.json'),
        canonical_checkpoint=str(node), checkpoint_manifest_sha256=checkpoint_hash,
        native_e_dev_profiles=profiles, denied_oracle_or_test_accesses=denied,
        protected_training_inputs_unchanged=True, model_updates=int(models[0].n_updates),
        launch_sha256=file_hash(output/'launch.json')))
    write_json(output/'verification.json', proof)
    write_json(output/'result.json', result)
    print(json.dumps(dict(receipt=result, verification=proof), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

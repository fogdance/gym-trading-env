#!/usr/bin/env python3
"""CPU-only before/after account rotation benchmark; never starts a model."""
import argparse
import cProfile
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()


def baseline_classes(ref):
    """Read the two original classes from an explicit immutable Git commit."""
    commit = subprocess.check_output(
        ['git', 'rev-parse', '--verify', ref+'^{commit}'], cwd=PROJECT, text=True).strip()
    modules, hashes = {}, {}
    for name in ('entry_env', 'entry_dataset_env'):
        relative = 'src/gym_trading_env/envs/'+name+'.py'
        source = subprocess.check_output(['git', 'show', commit+':'+relative], cwd=PROJECT)
        module = ModuleType('gym_trading_env.envs._benchmark_baseline_'+name)
        module.__package__ = 'gym_trading_env.envs'
        module.__file__ = 'git:'+commit+':'+relative
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        modules[name] = module
        hashes[relative] = hashlib.sha256(source).hexdigest()
    modules['entry_dataset_env'].FixedDirectionEntryEnv = modules['entry_env'].FixedDirectionEntryEnv
    return commit, modules['entry_dataset_env'], hashes


def baseline_e_market(commit):
    relative = 'e_data_v1/core.py'
    source = subprocess.check_output(['git', 'show', commit+':'+relative], cwd=PROJECT)
    module = ModuleType('e_data_v1._benchmark_baseline_core')
    module.__package__ = 'e_data_v1'
    module.__file__ = 'git:'+commit+':'+relative
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module.MarketData, {relative: hashlib.sha256(source).hexdigest()}


@contextmanager
def e_market_class(market):
    if market is None:
        yield
        return
    from e_data_v1 import core
    original = core.MarketData
    core.MarketData = market
    try:
        yield
    finally:
        core.MarketData = original


def trace_row(digest, contract, action, result):
    import numpy as np
    obs = result[0]
    digest.update(json_bytes([contract, action, result[1:]]))
    for name, value in sorted(obs.items()):
        array = np.asarray(value)
        digest.update(json_bytes([name, str(array.dtype), list(array.shape)]))
        digest.update(array.tobytes())


def advance(env, obs, count, digest):
    """Fixed causal actions, identical across versions; original Env executes."""
    steps, resets, elapsed = 0, [], []
    while steps < count:
        before = env.env.current_step
        action = int(bool(obs['action_mask'][1]) and before % 23 == 0)
        started = time.perf_counter()
        result = env.step(action)
        elapsed.append(time.perf_counter()-started)
        trace_row(digest, env.env.contract, action, result)
        delta = env.env.current_step-before
        if delta not in (0, 1) or env.env.status in ('DATA_GAP', 'DATA_END'):
            raise ValueError('Benchmark reached an unsupported data/account boundary')
        steps += delta
        obs = result[0]
        if (result[2] or result[3]) and steps < count:
            previous = env.env.contract
            cache = getattr(env, '_market_cache', None)
            hits = cache.hits if cache is not None else 0
            started = time.perf_counter(); reset = env.reset()
            seconds = time.perf_counter()-started
            trace_row(digest, env.env.contract, None, reset)
            obs = reset[0]
            resets.append(dict(previous=previous, next=env.env.contract, seconds=seconds,
                cache_hit=cache is not None and cache.hits > hits))
    return obs, dict(advances=steps, reset_seconds=resets,
                    step_seconds=sum(elapsed), native_calls=len(elapsed))


def benchmark(name, module, selector, count, seed, *, cache_market=None,
              e_profile=None, profile_path=None):
    import pandas as pd
    original = module.FixedDirectionEntryEnv
    constructors, signatures = [], {}

    def measured(*args, **kwargs):
        started = time.perf_counter(); env = original(*args, **kwargs)
        constructors.append(dict(contract=env.contract, seconds=time.perf_counter()-started))
        if env.contract not in signatures:
            raw = env._raw
            digest = hashlib.sha256(pd.util.hash_pandas_object(raw, index=True).values.tobytes())
            digest.update(json_bytes([list(raw.columns), [str(x) for x in raw.dtypes],
                                      str(raw.index.dtype), raw.index.name]))
            signatures[env.contract] = dict(rows=len(raw), config_hash=env.config_hash,
                observation_identity=env.observation_identity, raw_sha256=digest.hexdigest(),
                features_sha256=hashlib.sha256(env._features.tobytes()).hexdigest(),
                shape=list(env._features.shape), dtype=str(env._features.dtype))
        return env

    module.FixedDirectionEntryEnv = measured
    env = None
    try:
        started = time.perf_counter()
        kwargs = {} if cache_market is None else dict(cache_market=cache_market)
        if e_profile is not None:
            kwargs['e_profile'] = e_profile
        env = module.FGEntryDatasetEnv(selector, **kwargs)
        obs, info = env.reset(seed=seed)
        trace = hashlib.sha256(); trace_row(trace, env.env.contract, None, (obs, info))
        profiler = cProfile.Profile() if profile_path else None
        if profiler:
            profiler.enable()
        try:
            obs, timing = advance(env, obs, count, trace)
        finally:
            if profiler:
                profiler.disable()
                profiler.dump_stats(profile_path)
        total = time.perf_counter()-started
        state = env.export_state()
        cache = getattr(env, '_market_cache', None)
        result = dict(name=name, total_seconds=total, trace_sha256=trace.hexdigest(),
            state_sha256=hashlib.sha256(json_bytes(state)).hexdigest(), constructors=constructors,
            market_signatures=signatures, timing=timing,
            cache=dict(hits=cache.hits, misses=cache.misses, entries=len(cache), nbytes=cache.nbytes)
                  if cache is not None else None,
            account_summary={str(k):dict(contract=v['contract'], current_step=v['current_step'],
                fills=len(v['fills']), closed_trades=len(v['trade_history']),
                total_reward=v['total_reward'], balances=v['balances']) for k,v in state['accounts'].items()})
        return result, env, obs
    except Exception:
        if env is not None: env.close()
        raise
    finally:
        module.FixedDirectionEntryEnv = original


def restore_check(env, obs, module, selector, continuation, *, e_profile=None):
    """Restore the original financial schema into a cold, independent cache."""
    from gym_trading_env.envs.entry_contract import fingerprint
    saved = json.loads(json.dumps(env.export_state(), default=str))
    kwargs = {} if e_profile is None else dict(e_profile=e_profile)
    restored = module.FGEntryDatasetEnv(selector, **kwargs)
    try:
        restored.reset(options={'resume': saved})
        if fingerprint(restored.export_state()) != fingerprint(env.export_state()):
            raise ValueError('Cold cache restore changed the financial state')
        first, second = hashlib.sha256(), hashlib.sha256()
        if env.env.truncated or env.env.terminated:
            a, b = env.reset(), restored.reset()
            trace_row(first, env.env.contract, None, a)
            trace_row(second, restored.env.contract, None, b)
            obs, other = a[0], b[0]
        else:
            other = restored.env._get_obs()
        advance(env, obs, continuation, first)
        advance(restored, other, continuation, second)
        equal_trace = first.hexdigest() == second.hexdigest()
        equal_state = fingerprint(env.export_state()) == fingerprint(restored.export_state())
        if not equal_trace or not equal_state:
            raise ValueError('Cold cache continuation differs from uninterrupted collection')
        return dict(status='PASS', cache_was_cold=True, financial_schema=saved['schema'],
            continuation_advances=continuation, trace_sha256=first.hexdigest(),
            state_sha256=fingerprint(env.export_state()), exact_trace=equal_trace, exact_state=equal_state)
    finally:
        restored.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selector', type=Path, required=True)
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--contracts', nargs='+', required=True)
    parser.add_argument('--steps', type=int, default=4000, help='Raw Env advances, not gradient updates')
    parser.add_argument('--chunk-bars', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=202)
    parser.add_argument('--restore-steps', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--e-data-collection', type=Path,
                        help='Verified real E collection; enables native E before/after comparison')
    parser.add_argument('--e-source-pool', type=Path,
                        help='Original E TRAIN source pool, required with --e-data-collection')
    parser.add_argument('--profile', action='store_true', help='Profile CPU Env steps separately for each variant')
    args = parser.parse_args()
    if bool(args.e_data_collection) != bool(args.e_source_pool):
        parser.error('E collection and original source pool must be supplied together')
    if (args.steps <= 0 or args.chunk_bars <= 0 or args.restore_steps <= 0
            or len(args.contracts) < 2 or len(set(args.contracts)) != len(args.contracts)):
        parser.error('Positive budgets and at least two unique contracts are required')
    output = args.output.resolve()
    if not (output.is_relative_to(PROJECT/'artifacts') or output.is_relative_to(Path('/tmp'))):
        parser.error('Output must be under this worktree artifacts/ or /tmp; formal Runs are read-only')
    if output.exists():
        parser.error('Output already exists; use a new isolated benchmark directory')
    # Set before importing Gym dependencies. No JAX/model/Oracle/W&B work here.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ.pop('DREAMER_RUN_DIR', None)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/s3-collection-benchmark-mpl')
    sys.path.insert(0, str(PROJECT/'src'))
    sys.path.insert(0, str(PROJECT))
    from gym_trading_env.envs import entry_dataset_env as current
    from gym_trading_env.envs.entry_contract import read_yaml, fingerprint
    selector = read_yaml(args.selector)
    manifest_path = Path(selector['manifest_path'])
    manifest = json.loads(manifest_path.read_text())
    unsigned = {k:v for k,v in manifest.items() if k != 'content_hash'}
    if fingerprint(unsigned) != manifest['content_hash']:
        raise ValueError('Input pool manifest identity differs')
    rows = [r for r in manifest['files'] if r['contract'] in args.contracts]
    if (len(rows) != len(args.contracts) or {r['contract'] for r in rows} != set(args.contracts)
            or any(r['role'] != 'train'
            or not r['eligible_for_training'] for r in rows)):
        raise ValueError('Benchmark contracts must be distinct eligible TRAIN members')
    if any(read_yaml(r['env_config'])['entry'].get('state_path') for r in rows):
        raise ValueError('Read-only benchmark refuses Env state_path writes')
    commit, baseline, code_hashes = baseline_classes(args.baseline_ref)
    inputs = {str(manifest_path):sha256(manifest_path), str(args.selector):sha256(args.selector)}
    for row in rows:
        config = Path(row['env_config']); raw = read_yaml(config)
        for path in (config, Path(raw['trading']['data_path']),
                     *(config.parent/Path(raw['entry'][k]) for k in
                       ('contract_config', 'strategy_config', 'direction_config') if raw['entry'].get(k))):
            inputs[str(path.resolve())] = sha256(path)
    output.mkdir(parents=True)
    subset = dict(unsigned); subset['files'] = rows; subset['content_hash'] = fingerprint(subset)
    (output/'pool.json').write_text(json.dumps(subset, indent=2)+'\n')
    import yaml
    settings = dict(selector, manifest_path=str(output/'pool.json'),
                    scheduler=dict(mode='least_progress', chunk_bars=args.chunk_bars))
    local_selector = output/'selector.yaml'
    local_selector.write_text(yaml.safe_dump(settings))
    e_profile, original_market = None, None
    if args.e_data_collection:
        from e_data_v1.provider import ETrainAdmission
        from e_data_v1.core import MarketData
        e_profile = ETrainAdmission.from_train_pool(args.e_source_pool,
            args.e_data_collection, output/'pool.json', contracts=args.contracts)
        original_market, e_hashes = baseline_e_market(commit)
        code_hashes.update(e_hashes)
        inputs[str(args.e_source_pool.resolve())] = sha256(args.e_source_pool)
        inputs[str((args.e_data_collection/'index_manifest.json').resolve())] = sha256(
            args.e_data_collection/'index_manifest.json')
    results, active, obs = [], None, None
    try:
        variants = ([('original', baseline, True, original_market),
                     ('cached', current, True, MarketData)] if e_profile is not None else
                    [('original', baseline, None, None),
                     ('single_features', current, False, None), ('cached', current, True, None)])
        for name, module, cache, market in variants:
            print('CPU benchmark: '+name, flush=True)
            with e_market_class(market):
                result, env, current_obs = benchmark(name, module, local_selector,
                    args.steps, args.seed, cache_market=cache, e_profile=e_profile,
                    profile_path=output/(name+'.prof') if args.profile else None)
            results.append(result)
            (output/(name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'variant':name,'seconds':result['total_seconds'],
                              'constructors':result['constructors'],'cache':result['cache']}), flush=True)
            if name == 'cached': active, obs = env, current_obs
            else: env.close()
        for candidate in results[1:]:
            for field in ('trace_sha256', 'state_sha256', 'market_signatures'):
                if candidate[field] != results[0][field]:
                    raise ValueError(candidate['name']+' changed '+field)
        restored = restore_check(active, obs, current, local_selector, args.restore_steps,
                                 e_profile=e_profile)
        if any(sha256(path) != digest for path,digest in inputs.items()):
            raise ValueError('Source inputs changed during benchmark')
        report = dict(schema='S3_COLLECTION_PERF_CPU_V1', status='PASS',
            baseline_commit=commit, baseline_code_sha256=code_hashes,
            gym_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
            dreamer_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT.parent/'dreamerv3',text=True).strip(),
            current_code_sha256={str(p.relative_to(PROJECT)):sha256(p) for p in
                [PROJECT/'src/gym_trading_env/envs'/n for n in
                 ('entry_env.py','entry_dataset_env.py','entry_market_cache.py')]
                +[PROJECT/'src/gym_trading_env/utils/bar_source.py']},
            original_inputs_sha256=inputs, inputs_unchanged=True, model_initialized=False,
            e_profile_identity=e_profile.identity if e_profile is not None else None,
            profiled=args.profile,
            gpu_used=False, oracle_loaded=False, gradient_updates=0, original_run_modified=False,
            contracts=args.contracts, advances=args.steps, chunk_bars=args.chunk_bars,
            exact_observation_reward_info_trace=True, exact_account_state=True,
            exact_full_market_features=True, restore=restored, variants=results,
            total_seconds_speedup=results[0]['total_seconds']/results[-1]['total_seconds'],
            step_seconds_speedup=results[0]['timing']['step_seconds']/results[-1]['timing']['step_seconds'])
        if e_profile is not None:
            report['current_code_sha256']['e_data_v1/core.py'] = sha256(PROJECT/'e_data_v1/core.py')
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'status':'PASS','report':str(output/'report.json'),
                          'speedup':report['total_seconds_speedup'],'restore':restored}), flush=True)
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(status='FAILED', error=str(error)), indent=2)+'\n')
        raise
    finally:
        if active is not None: active.close()


if __name__ == '__main__':
    main()

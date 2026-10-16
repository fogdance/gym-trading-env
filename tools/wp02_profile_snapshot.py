"""CPU-only A/B measurement using a preserved native TRAIN snapshot."""
import ast
import cProfile
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import pstats
import time
from unittest.mock import patch

from e_data_v1.provider import ETrainAdmission
from gym_trading_env.envs import entry_env
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import read_manifest, write_json
from gym_trading_env.research.fg_s3_e_quality_corpus_cli import (
    _factory, _code_lock, _expected_binding_files)
from gym_trading_env.research.fg_s3_quality_teacher_v2 import NativeQualityTeacher


def main():
    root = Path('artifacts/fg_s3_v2/e_quality_wp02')
    evidence = root/'evidence'
    baseline = evidence/'perf_baseline_source/entry_env.py'
    tree = ast.parse(baseline.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FixedDirectionEntryEnv')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('_prefix_hash', 'import_state')]
    namespace = dict(vars(entry_env))
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(baseline), 'exec'), namespace)
    corpus = (root/'real_small_03').resolve()
    lock = read_manifest(corpus/'source_lock.json')
    profile = ETrainAdmission.from_train_pool(lock['source_pool']['path'], corpus/'e_source',
        corpus/'selected_train_pool.json', contracts=[lock['contract']])
    row = dict(lock['row'], env_config=str(corpus/'inputs'/lock['contract']/'env.yaml'))
    initial = json.loads((corpus/'initial_account.json').read_text())
    files = _expected_binding_files(corpus, lock, _code_lock()['files'])
    results, labels, states = {}, {}, {}
    for mode in ('baseline', 'optimized'):
        with ExitStack() as stack:
            if mode == 'baseline':
                for name in ('_prefix_hash', 'import_state'):
                    stack.enter_context(patch.object(entry_env.FixedDirectionEntryEnv, name, namespace[name]))
            factory = _factory(row, profile)
            env = factory()
            env.import_state(initial)
            teacher = NativeQualityTeacher(env, factory, fold='TRAIN', source=dict(
                fold='TRAIN', files=files, state_cache_entries=0, calendar_complete=True))
            began = time.perf_counter()
            exported = env.export_state()
            export_seconds = time.perf_counter()-began
            began = time.perf_counter()
            teacher.clone.import_state(exported)
            import_seconds = time.perf_counter()-began
            values, durations = [], []
            for i in range(3):
                if i:
                    for _ in range(15):
                        env.step(0)
                began = time.perf_counter()
                values.append(teacher.query_state(env, env._published_mask))
                durations.append(time.perf_counter()-began)
            profiler = cProfile.Profile()
            profiler.enable()
            teacher.query_state(env, env._published_mask)
            profiler.disable()
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumtime').print_stats(20)
            print(mode, stream.getvalue(), flush=True)
            results[mode] = dict(export_state_seconds=export_seconds, import_state_seconds=import_seconds,
                query_seconds=durations, mean_query_seconds=sum(durations)/len(durations))
            labels[mode] = values
            states[mode] = env.export_state()
            teacher.close()
            env.close()
    assert labels['baseline'] == labels['optimized'], 'Full L1 labels differ'
    assert states['baseline'] == states['optimized'], 'Complete native financial history differs'
    report = dict(scope='ONE_CERTIFIED_TRAIN_CONTRACT_LAST_TWO_DAYS; THREE_READ_ONLY_STATES',
        status='PASS', measurements=results, full_l1_labels_equal=True, complete_states_equal=True,
        baseline_source_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        optimized_source_sha256=hashlib.sha256(Path(entry_env.__file__).read_bytes()).hexdigest(),
        final_account_snapshot_hash=fingerprint(states['optimized']),
        query_speedup=results['baseline']['mean_query_seconds']/results['optimized']['mean_query_seconds'])
    write_json(evidence/'perf_snapshot_comparison.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

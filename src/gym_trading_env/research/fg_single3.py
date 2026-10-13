"""TRAIN-only Single3 experiment inputs, original-Env evidence and CPU receipts."""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

from gym_trading_env.envs.entry_contract import fingerprint, read_yaml
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv, SINGLE3_PROTOCOL
from gym_trading_env.rewards.single3_reward import Single3RewardConfig
from .fg_history_study import digest
from .fg_replay import (check_files, provenance, read_manifest, signed,
                        validate_training, write_json)


PROJECT = Path(__file__).resolve().parents[3]
SCHEMA = 'FG_SINGLE3_CPU_V1'


def load_config(path):
    config = read_yaml(path)
    if config.get('clock_protocol') != 'S3_MARKET_MINUTE':
        raise ValueError('New Single3 preparation requires S3_MARKET_MINUTE')
    if config.get('schema') != SCHEMA or config.get('oracle_required_status') != 'exact_proven':
        raise ValueError('Explicit Single3 CPU/exact Oracle configuration required')
    for key in ('training_manifest', 'dreamer_root', 'output_root'):
        config[key] = str((PROJECT / config[key]).resolve())
    if (config['candidates'] != ['C0','C1','C2','C3'] or config['collection_candidate'] != 'C2'
            or config['env_chunk_steps'] != 1024 or config['batch_size'] != 16
            or config['train_length'] != 64 or config['replay_context'] != 1
            or config['oracle_gamma'] != 332/333):
        raise ValueError('Single3 CPU baseline configuration changed')
    config['config_path'] = str(Path(path).resolve())
    config['config_sha256'] = digest(path)
    return config


def execution_provenance(config,stage='E'):
    result = provenance(config['dreamer_root'])
    dependencies={'E':[], 'R':['fg_single3_replay.py','fg_single3_reward.py'],
        'O':['fg_single3_replay.py','fg_single3_oracle.py'],
        'D':['fg_single3_replay.py','fg_single3_dataset.py','fg_single3_coverage.py'],
        'H':['fg_single3_replay.py','fg_single3_dataset.py','fg_single3_horizon.py']}
    files=[Path(__file__),*(Path(__file__).parent/p for p in dependencies[stage])]
    files += [PROJECT/'src/gym_trading_env/rewards/single3_reward.py',
              PROJECT/'tools/fg_single3.py']
    result['code_files'].update({str(p): digest(p) for p in files if p.exists()})
    return result


def source_witness(output,code):
    """Archive actual dirty/committed bytes, so later CLI development is harmless."""
    directory=Path(output)/'source_witness'
    directory.mkdir()
    snapshots={}
    for filename,expected in code['code_files'].items():
        check_files({filename:expected})
        target=directory/(expected+'.py')
        if not target.exists():target.write_bytes(Path(filename).read_bytes())
        snapshots[str(target.resolve())]=expected
    return snapshots


def required_tests(stage):
    """Run semantic regressions before long work or any formal completion marker."""
    paths={'E':['tests/entry_v1/test_fg_single3_env.py','tests/entry_v1/test_entry_contract.py'],
           'R':['tests/entry_v1/test_fg_single3_replay.py','tests/entry_v1/test_fg_replay.py'],
           'O':['tests/entry_v1/test_fg_single3_oracle.py'],
           'D':['tests/entry_v1/test_fg_single3_replay.py','tests/entry_v1/test_fg_single3_coverage.py'],
           'H':['tests/entry_v1/test_fg_single3_horizon.py']}[stage]
    hashes={str(PROJECT/p):digest(PROJECT/p) for p in paths}
    command=[sys.executable,'-m','pytest','-p','no:cacheprovider',*paths,'-q','--tb=short']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
             PYTHONPATH=str(PROJECT/'src')+':'+str(PROJECT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    result=subprocess.run(command,cwd=PROJECT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(result.stdout,flush=True)
    check_files(hashes)
    if result.returncode:
        raise RuntimeError(f'{stage} required semantic tests failed; no completion may be published')
    return dict(command=command,returncode=result.returncode,output=result.stdout,test_files=hashes)


def make_output(path, config):
    import shutil
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite experiment evidence: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < config['minimum_free_disk_gib']*2**30:
        raise RuntimeError('Insufficient free disk; historical artifacts will not be deleted')
    path.mkdir()
    return path


def prepare_inputs(config, directory):
    """Derive strict max3 snapshots from frozen TRAIN inputs; never resplit CSVs."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    original, rows, frozen, source = validate_training(config['training_manifest'])
    frozen[config['config_path']] = config['config_sha256']
    base_master = read_yaml(original['direction_path'])
    views = {}
    for candidate in config['candidates']:
        target = directory/candidate
        target.mkdir()
        master = copy.deepcopy(base_master)
        master.update(schema=2, protocol=SINGLE3_PROTOCOL,
                      reward=asdict(Single3RewardConfig.for_candidate(candidate)))
        master['environment']['trading'].update(max_long_position=3, max_short_position=3,
                                                stop_loss_enabled=True)
        master['environment']['entry']['chunk_steps'] = config['env_chunk_steps']
        master['data']['split']['train_end_inclusive'] = max(original['roles']['train'])
        master_path = target/'single3.yaml'
        master_path.write_text(yaml.safe_dump(master, allow_unicode=True, sort_keys=False))
        frozen[str(master_path)] = digest(master_path)
        view = {k: copy.deepcopy(v) for k,v in original.items() if k not in ('content_hash','files')}
        view.update(single3_protocol=SINGLE3_PROTOCOL, reward_candidate=candidate,
                    clock_protocol='S3_MARKET_MINUTE',
                    original_manifest=config['training_manifest'], original_content_hash=original['content_hash'],
                    direction_path=str(master_path), direction_sha256=digest(master_path), files=[])
        for row in rows:
            raw = read_yaml(row['env_config'])
            raw['entry'].update(protocol=SINGLE3_PROTOCOL, strategy_config=str(master_path),
                contract_config=row['contract_config'], chunk_steps=config['env_chunk_steps'])
            raw['trading'].update(master['environment']['trading'])
            path = target/(row['contract'].replace('.','_')+'_env.yaml')
            path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
            frozen[str(path)] = digest(path)
            view['files'].append(dict(row, env_config=str(path), env_sha256=digest(path)))
        view_path = target/'train_manifest.json'
        write_json(view_path, signed(view))
        frozen[str(view_path)] = digest(view_path)
        # validate_training opens only the TRAIN members of this derived view.
        validated, new_rows, new_files, _ = validate_training(view_path)
        frozen.update(new_files)
        views[candidate] = dict(path=str(view_path), content_hash=validated['content_hash'], rows=new_rows)
    check_files(frozen)
    write_json(directory/'inputs.json', signed(dict(original_content_hash=original['content_hash'],
        source=source, views=views, frozen_files=frozen)))
    return dict(source=source, views=views, frozen_files=frozen,
                original_content_hash=original['content_hash'])


def load_env(row, frozen_files):
    """Reject mutable/foreign price references before the Env opens a CSV."""
    raw = read_yaml(row['env_config'])
    price_path = str(Path(raw['trading']['data_path']).resolve())
    if (raw['entry'].get('protocol') != SINGLE3_PROTOCOL or
            raw['entry']['contract'] != row['contract'] or
            price_path != str(Path(row['path']).resolve()) or
            frozen_files.get(price_path) != row['sha256']):
        raise ValueError('Env would open a price path outside frozen TRAIN inputs')
    check_files({p:frozen_files[p] for p in (price_path, row['env_config'], row['contract_config'])})
    return FixedDirectionEntryEnv(config_path=row['env_config'])


def walk_env(row, frozen_files, *, restore=False):
    """Independent original-Env P1 replay, with complete accounting invariants."""
    began = time.monotonic()
    env = load_env(row, frozen_files)
    obs, _ = env.reset(seed=0)
    hasher = hashlib.sha256()
    counts = Counter()
    lots = set()
    maximum = 0
    def capture(action, obs, reward):
        hasher.update(str(action).encode())
        for key in sorted(obs):
            hasher.update(key.encode()+np.asarray(obs[key]).tobytes())
        hasher.update(np.float64(reward).tobytes())
        hasher.update(fingerprint(dict(cursor=env.current_step, status=env.status,
            fills=env.fills[-2:], ledger=env.ledger.snapshot())).encode())
    capture(None, obs, 0.)
    while True:
        action = int(obs['action_mask'][1])
        fills, rewards = len(env.attributions), len(env.reward_events)
        obs, reward, terminal, truncated, _ = env.step(action)
        capture(action, obs, reward)
        counts['steps'] += 1
        lots.add(int(env.net_lots))
        maximum = max(maximum, len(env.entries))
        if env.net_lots not in (0,3) or len(env.entries)>1:
            raise RuntimeError('Single3 position invariant failed')
        env._assert_ledger_conservation()
        for event in env.attributions[fills:]:
            counts[event['reason']] += 1
        counts['FIRST_ADVERSE10'] += sum(e['reason']=='FIRST_ADVERSE10' for e in env.reward_events[rewards:])
        if counts['steps']%16384==0:
            print(f"Env {row['contract']} restore={restore}: {counts['steps']} steps", flush=True)
        if terminal or env.status in ('DATA_END','DATA_GAP'):
            break
        if truncated:
            if env.status != 'CHUNK_END':
                raise RuntimeError('Unexpected nonterminal interruption')
            if restore:
                before = env.export_state()
                env.import_state(json.loads(json.dumps(before)))
                if env.export_state()!=before:
                    raise RuntimeError('Single3 complete state changed on restore')
                counts['restores'] += 1
            obs, _ = env.reset()
            capture(None, obs, 0.)
    state = env.export_state()
    financial = sum((r.pnl-r.fee for r in env.trade_record_manager.trade_history), Decimal(0))
    if env._calculate_equity() != Decimal(300000)+financial+env.user_accounts.unrealized_pnl:
        raise RuntimeError('Ledger equity differs from real native trade PnL and fees')
    report = dict(contract=env.contract, counts=dict(counts), lots=sorted(lots), max_entries=maximum,
        actual_rows=len(env._raw), advanced=env.current_step, terminal_status=env.status,
        open_lots=float(env.net_lots), net_pnl=float(env._calculate_equity()-300000),
        realized_less_fees=str(financial), total_reward=env.total_reward,
        step_hash=hasher.hexdigest(), state_hash=fingerprint(state), elapsed_seconds=time.monotonic()-began)
    env.close()
    return report, state


def _env_audit_pair(job):
    row, frozen = job
    first, state = walk_env(row, frozen)
    second, restored = walk_env(row, frozen, restore=True)
    if state != restored or first['step_hash'] != second['step_hash']:
        raise RuntimeError('Independent native replay or restored financial history differs')
    return dict(original=first, independent_restore=second, reconciliation='PASS'), state


def env_audit(config, output, contracts=('CZCE.FG809','CZCE.FG909')):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    tests=required_tests('E')
    output = make_output(output, config)
    code = execution_provenance(config)
    inputs = prepare_inputs(config, output/'inputs')
    snapshots=source_witness(output,code)
    reports = []
    available = {r['contract']:r for r in inputs['views']['C2']['rows']}
    if not contracts or len(set(contracts))!=len(contracts) or set(contracts)-set(available):
        raise ValueError('Env audit requires unique TRAIN contracts')
    jobs=[(available[c],inputs['frozen_files']) for c in contracts]
    with ProcessPoolExecutor(max_workers=min(2,len(jobs)),
            mp_context=multiprocessing.get_context('spawn')) as pool:
        for contract,(report,state) in zip(contracts,pool.map(_env_audit_pair,jobs)):
            write_json(output/(contract.replace('.','_')+'_account.json'),state)
            reports.append(report)
    receipt = dict(schema=SCHEMA, clock_protocol='S3_MARKET_MINUTE', stage='E', status='PASS',
        training_content_hash=inputs['original_content_hash'], config=config, provenance=code,tests=tests,
        frozen_inputs={**inputs['frozen_files'],**snapshots}, source_snapshots=snapshots,reports=reports,
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    write_json(output/'manifest.pending.json', signed(receipt))
    check_files(inputs['frozen_files'])
    check_files(code['code_files'])
    check_files({str(output/p):h for p,h in receipt['files'].items()})
    (output/'manifest.pending.json').replace(output/'manifest.json')
    return receipt

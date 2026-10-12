#!/usr/bin/env python3
"""Reconcile frozen TRAIN Replay against the existing FG009 financial Env."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fg_gate_matplotlib')

import argparse
import collections
import csv
import json
import pickle
from pathlib import Path
import sys
import time

from fg_single3_prior_audit import ROOT, DREAMER, load_replay, write_json
sys.path[:0] = [str(ROOT / 'src'), str(DREAMER)]


def run(source, audit, output):
    import numpy as np
    from dreamerv3.configuration import read_config_yaml
    from embodied.run.fg_checkpoint import validate_bundle, file_hash
    from embodied.run.action_mask_warmup import ActionMaskWarmup, METRICS, ADD_RECALL_METRICS
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    import elements
    started = time.monotonic()
    output.mkdir(exist_ok=False, parents=True)
    meta = validate_bundle(source)
    before = file_hash(source / 'manifest.json')
    config = read_config_yaml(source / 'config.yaml')
    train = json.loads(Path(config['fg_entry']['manifest']).read_text())
    records = [r for r in train['files'] if r['role'] == 'train' and r['eligible_for_training']]
    driver = pickle.loads((source / 'driver.pkl').read_bytes())
    saved = driver['envs'][0]['env']['account']
    assert saved['contract'] == 'CZCE.FG009'
    data = load_replay(source, meta, 12)
    with np.load(audit / 'samples.npz') as z:
        indices, target = z['indices'], z['target']
        prev = indices - 1
        reset = data['is_first'][indices]
        held = data['position_state'][prev, 1] > .5
        groups = {'sustained_flat_wait': ~reset & ~held & (data['action'][prev] == 0),
            'held_then_released': ~reset & held,
            'flat_add_then_exit_or_reject': ~reset & ~held & (data['action'][prev] == 1),
            'reset': reset}
        assert np.array_equal(sum(g.astype(int) for g in groups.values()), np.ones(len(indices), int))
        assert all(np.array_equal(z[k], v) for k, v in groups.items())
        positives = {k: int((v & target[:, 1]).sum()) for k, v in groups.items()}
        assert sum(positives.values()) == 1016
    boundaries = np.r_[0, np.flatnonzero(data['is_terminal']) + 1, len(data['action'])]
    begin, end = map(int, boundaries[6:8])
    assert begin == 159109 and end == 180000 and records[6]['contract'] == saved['contract']
    assert file_hash(records[6]['path']) == records[6]['sha256']
    env = FixedDirectionEntryEnv(config_path=records[6]['env_config'])
    rows, exits = [], []
    nonreset, resets = 0, 0
    for i in range(begin, end):
        before_exits = len(getattr(env, 'attributions', []))
        previous_cursor = env.current_step if env._initialized else None
        if data['is_first'][i]:
            obs, _ = env.reset()
            reward, term, trunc = 0., False, False
            resets += 1
            if previous_cursor is not None:
                assert env.current_step == previous_cursor
        else:
            assert i > begin
            obs, reward, term, trunc, _ = env.step(int(data['action'][i - 1]))
            nonreset += 1
            assert env.current_step == previous_cursor + 1
        for key in ('market_seq', 'position_state', 'action_mask'):
            np.testing.assert_array_equal(obs[key], data[key][i], err_msg=f'{i}: {key}')
        assert np.float32(reward) == data['reward'][i], f'{i}: reward'
        assert (term or trunc) == bool(data['is_last'][i]), f'{i}: is_last'
        assert term == bool(data['is_terminal'][i]), f'{i}: terminal'
        assert obs['action_mask'][int(data['action'][i])] == 1, f'{i}: illegal action'
        rows.append(dict(index=i, stepid=data['stepid'][i].tobytes().hex(),
            cursor=env.current_step, timestamp=str(env._raw.index[env.current_step]),
            is_first=bool(data['is_first'][i]), is_last=bool(data['is_last'][i]),
            action=int(data['action'][i]), previous_action=int(data['action'][i - 1]) if i > begin else None,
            held=float(obs['position_state'][1]), add_legal=bool(obs['action_mask'][1]),
            reward=float(data['reward'][i]), entries=list(env.entries)))
        for event in env.attributions[before_exits:]:
            row = env._raw.iloc[env.current_step]
            exits.append(dict(event=event, index=i, stepid=data['stepid'][i].tobytes().hex(),
                cursor=env.current_step, bar_open=str(env.bobs[env.current_step]),
                bar_end=str(env._raw.index[env.current_step]),
                ohlc={k:float(row[k]) for k in ('Open','High','Low','Close')},
                post_add_legal=bool(obs['action_mask'][1]),
                previous_held=float(data['position_state'][i-1,1]),
                previous_action=int(data['action'][i-1]), reward=float(data['reward'][i])))
        if (i - begin + 1) % 5000 == 0:
            print('FG009 reconciled', i - begin + 1, '/', end - begin, flush=True)
    actual = env.export_state()
    compared = set(saved) - {'rng'}
    assert all(actual[k] == saved[k] for k in compared), [k for k in compared if actual[k] != saved[k]]
    assert nonreset == saved['current_step'] == 20870
    assert len(exits) == len(saved['attributions'])
    for item in exits:
        item['surrounding_frames'] = rows[max(0,item['index']-begin-2):item['index']-begin+3]
    with (output / 'fg009_frames.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    write_json(output / 'fg009_exits.json', exits)
    metrics_path = source.parent.parent / 'metrics.jsonl'
    gate = ActionMaskWarmup.for_single3_1m(elements.Config(config['action_mask_warmup']))
    windows = []
    for line in metrics_path.read_text().splitlines():
        record = json.loads(line)
        prefix = 'action_mask_gate_eval/'
        mets={k[len(prefix):]:v for k,v in record.items() if k.startswith(prefix)}
        if not mets:
            continue
        support = gate._has_add_support(mets)
        failures=[name for name,(k,mode) in {**METRICS,**ADD_RECALL_METRICS}.items()
            if k not in mets or not np.isfinite(mets[k]) or
            (mets[k] < gate.opening_thresholds[name] if mode=='min' else mets[k] > gate.opening_thresholds[name])]
        gate.update(mets, record['step'], allow_activation=False)
        logged = {k: (v if np.isfinite(v) else None) for k, v in mets.items()}
        windows.append(dict(step=record['step'], metrics=logged, add_support_sufficient=support,
            failed_quality_conditions=failures, passed=gate.last_passed,
            consecutive_passes=gate.consecutive_passes,
            historical_fp_counts='NOT_RECORDED'))
    write_json(output / 'historical_gate_windows.json', windows)
    assert file_hash(source / 'manifest.json') == before
    validate_bundle(source)
    result=dict(status='CAUSE_AUDIT_PASS', elapsed_seconds=time.monotonic()-started,
        source=str(source), source_manifest_sha256=before,
        groups=positives, fg009_replay_range=[begin,end], fg009_rows=len(rows),
        fg009_nonreset_advances=nonreset, fg009_resets=resets, exits=len(exits),
        exit_reasons=dict(collections.Counter(e['event']['reason'] for e in exits)),
        financial_state_equal_fields=sorted(compared), reconstruction_rng_not_compared=True,
        historical_reports=len(windows), historical_qualifying_windows=sum(w['passed'] for w in windows),
        historical_max_pass_streak=max(w['consecutive_passes'] for w in windows),
        insufficient_support_windows=sum(not w['add_support_sufficient'] for w in windows),
        historical_fp_counts='NOT_RECORDED', latest_gate_step=windows[-1]['step'],
        action_obs_reward_reset_alignment=True, source_unchanged=True,
        dev_test_accessed=False, gpu_training_started=False,
        train_input_files={str(records[6][k]):file_hash(records[6][k]) for k in ('path','env_config','contract_config')})
    write_json(output / 'analysis.json', result)
    write_json(output / 'manifest.json',dict(files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--audit',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.source,args.audit,args.output)

"""Strict C1_P4 outgoing labels; model observations never contain labels."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
import json
import subprocess

import numpy as np

from .fg_replay import (SINGLE3_SCHEMA, check_files, digest, iter_replay_rows,
                        iter_sidecar, read_manifest, signed, trace_manifest, write_json)
from gym_trading_env.rewards.single3_reward import Single3RewardConfig

PROTOCOL = 'FG_S3A_LABELS_V1'
INDEX_DTYPE = np.dtype([
    ('ordinal', '<u8'), ('cursor', '<i8'), ('stepid', 'u1', (20,)),
    ('label', 'u1'), ('mask', '<f4', (2,)), ('is_first', '?'),
    ('decision_valid_minute', '?'), ('year', '<i2'),
    ('direction', 'i1'), ('session', 'u1')])


def provenance():
    path = Path(__file__).resolve()
    return dict(code_sha256={str(path): digest(path)}, git_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=path.parent, text=True).strip())


def sources(dataset, oracle_receipt, candidate='C1'):
    """Inspect only TRAIN metadata/prices and selected P4 accounts, never Test."""
    if candidate != 'C1':
        raise ValueError('S3-A only accepts C1_P4')
    root, oracle_root = Path(dataset).resolve(), Path(oracle_receipt).resolve()
    m, inputs = read_manifest(root/'manifest.json'), read_manifest(root/'inputs.json')
    oracle = read_manifest(oracle_root/'manifest.json')
    identity = m['identity']
    if (m.get('schema') != SINGLE3_SCHEMA or m.get('stage') != 'D_GENERATED'
            or m.get('status') != 'PASS' or not m.get('full_training_fold')
            or identity.get('fold') != 'TRAIN' or not identity.get('conditional_direction_only')
            or identity != inputs['identity'] or m['input_manifest_sha256'] != digest(root/'inputs.json')):
        raise ValueError('S3-A requires a signed complete Single3 TRAIN dataset')
    if (oracle.get('stage') != 'O' or oracle.get('status') != 'PASS'
            or not oracle.get('full_training_fold') or oracle.get('oracle_optimality_status') != 'exact_proven'
            or oracle['identity']['training_hash'] != identity['original_training_hash']):
        raise ValueError('S3-A requires matching exact full TRAIN Oracle O receipt')
    view = inputs['training_views']['C1']
    train = read_manifest(view['path'])
    if (train['content_hash'] != view['content_hash'] or train.get('reward_candidate') != 'C1'
            or train.get('single3_protocol') != 'fg_single3_v1'
            or train['original_content_hash'] != identity['original_training_hash']):
        raise ValueError('TRAIN observation/Reward identity mismatch')
    whitelist = {r['contract'] for r in train['files']
                 if r['role'] == 'train' and r['eligible_for_training'] and r['complete_lifecycle']}
    if whitelist != set(identity['selected_contracts']):
        raise ValueError('Dataset escaped complete TRAIN whitelist')
    frozen = {str(root/'manifest.json'): digest(root/'manifest.json'),
              str(root/'inputs.json'): digest(root/'inputs.json'),
              str(oracle_root/'manifest.json'): digest(oracle_root/'manifest.json'),
              str(Path(view['path']).resolve()): digest(view['path'])}
    # Existing frozen inputs contain TRAIN files and receipts. Explicitly check
    # TRAIN row membership before opening their price/config bytes.
    for row in view['rows']:
        if row['contract'] not in whitelist or row['role'] != 'train' or not row['eligible_for_training']:
            raise ValueError('Foreign observation TRAIN member')
        for key, hashkey in [('path', 'sha256'), ('env_config', 'env_sha256'),
                             ('contract_config', 'contract_sha256')]:
            frozen[str(Path(row[key]).resolve())] = row[hashkey]
    if train.get('direction_path'):
        frozen[str(Path(train['direction_path']).resolve())] = train['direction_sha256']
    check_files(frozen)
    selected = sorted((r for r in m['traces'] if r['candidate'] == 'C1' and r['policy_id'] == 'P4'),
                      key=lambda r: r['contract'])
    if (len(selected) != len(whitelist) or {r['contract'] for r in selected} != whitelist
            or [m['traces'][i] for i in m['pool_members']['C1_P4']] != selected):
        raise ValueError('C1_P4 pool/contract membership mismatch')
    reward_hash = Single3RewardConfig.for_candidate('C1').hash
    for rec in selected:
        directory = root/rec['directory']
        if not directory.resolve().is_relative_to(root):
            raise ValueError('Trace path escaped dataset')
        frozen.update({str(directory/'manifest.json'): rec['manifest_sha256'],
                       str(directory/'reconciliation.json'): rec['receipt_sha256']})
        check_files({str(directory/'manifest.json'): rec['manifest_sha256'],
                     str(directory/'reconciliation.json'): rec['receipt_sha256']})
        t = trace_manifest(directory, schema=SINGLE3_SCHEMA)
        receipt = read_manifest(directory/'reconciliation.json')
        policy = t['policy']
        if (t.get('protocol') != 'fg_single3_v1' or t.get('obs_version') != 'FG_SINGLE3_MARKET_MINUTE_60x18_9_2'
                or t['reward_params_hash'] != reward_hash or t['reward_candidate'] != 'C1'
                or t['contract'] != rec['contract'] or t['identity'] != rec['trace_identity']
                or t['records'] != rec['records'] or t['replay_trace_hash'] != rec['replay_trace_hash']
                or t['observation_identity'] != rec['observation_identity']
                or rec['reward_params_hash'] != reward_hash
                or policy['id'] != 'P4' or policy['kind'] != 'oracle_schedule' or policy['candidate'] != 'C1'
                or receipt['status'] != 'PASS' or receipt['trace_manifest_sha256'] != rec['manifest_sha256']
                or t['identity']['training_hash'] != identity['original_training_hash']):
            raise ValueError('Foreign/unreconciled C1_P4 trace identity')
        schedule_path = Path(policy['schedule_path']).resolve()
        if not schedule_path.is_relative_to(oracle_root):
            raise ValueError('P4 schedule escaped Oracle receipt')
        rel = str(schedule_path.relative_to(oracle_root))
        if oracle['files'].get(rel) != policy['schedule_sha256']:
            raise ValueError('Oracle schedule is not bound to signed O receipt')
        plan = read_manifest(schedule_path)
        if (digest(schedule_path) != policy['schedule_sha256'] or plan['candidate'] != 'C1'
                or plan['gamma'] != 332/333 or plan.get('gamma_rational') != [332, 333]
                or plan['oracle_optimality_status'] != 'exact_proven' or not plan['training_only']
                or plan['contract'] != rec['contract'] or plan['identity'] != oracle['identity']):
            raise ValueError('P4 requires the exact frozen C1 H333 schedule')
        frozen[str(schedule_path)] = policy['schedule_sha256']
        frozen.update({str(directory/p): h for p, h in t['files'].items()})
    return dict(dataset=str(root), oracle_receipt=str(oracle_root),
                dataset_sha256=frozen[str(root/'manifest.json')],
                oracle_sha256=frozen[str(oracle_root/'manifest.json')],
                training_hash=identity['original_training_hash'], reward_params_hash=reward_hash,
                traces=selected, frozen_inputs=frozen)


def aligned_rows(directory, manifest, schedule):
    """Validate current observation and next-row execution without re-running Env."""
    previous = None
    for row, native in zip_longest(iter_sidecar(directory), iter_replay_rows(directory, manifest)):
        if row is None or native is None:
            raise ValueError('Sidecar/Replay length mismatch')
        ordinal, step = native
        mask = np.asarray(step['action_mask'])
        if (row['ordinal'] != ordinal or row['stepid'] != step['stepid'].tobytes().hex()
                or row['action'] != int(step['action']) or row['legal_add'] != bool(mask[1])
                or not np.isin(mask, [0, 1]).all() or not mask[0]
                or row['lots_after'] not in (0, 3) or row['lots_before'] not in (0, 3)
                or step['position_state'][1] != row['lots_after']/3
                or step['position_state'][6:].any()
                or row['direction'] not in ('LONG', 'SHORT', 'NO_TRADE')
                or step['position_state'][0] != {'LONG': 1, 'SHORT': -1, 'NO_TRADE': 0}[row['direction']]
                or not np.isfinite(step['market_seq']).all()
                or not np.isfinite(step['position_state']).all()
                or not np.isfinite(step['reward'])
                or float(np.float32(row['official_reward'])) != float(step['reward'])):
            raise ValueError(f'Outgoing action/observation/stepid mismatch at {ordinal}')
        for key in ('is_first', 'is_last', 'is_terminal'):
            if bool(step[key]) != row[key]:
                raise ValueError(f'Replay flag mismatch: {ordinal}/{key}')
        action = row['action']
        if action not in (0, 1) or not mask[action] or action != int(not row['is_last'] and row['cursor'] in schedule):
            raise ValueError(f'Illegal or shifted Oracle outgoing action at {ordinal}')
        if row['reset'] != row['is_first'] or (ordinal == 0 and not row['is_first']):
            raise ValueError('Reset flag mismatch')
        if previous:
            old, _ = previous
            incoming = None if old['is_last'] else old['action']
            if (row['incoming_action'] != incoming or row['is_first'] != (incoming is None)
                    or old['decision_executed'] != row['advanced']
                    or old['decision_valid_minute'] != row['valid_bar']
                    or (row['advanced'] and row['cursor'] != old['cursor']+1)
                    or (not row['advanced'] and row['cursor'] != old['cursor'])):
                raise ValueError(f'Next feedback/previous outgoing alignment mismatch at {ordinal}')
            yield previous
        elif row['incoming_action'] is not None:
            raise ValueError('First observation has an incoming action')
        previous = row, step
    if previous:
        if previous[0]['decision_executed'] or previous[0]['decision_valid_minute']:
            raise ValueError('Final outgoing action was not executed')
        yield previous


def exclusion(row, step):
    if row['is_last'] or row['is_terminal']:
        return 'administrative_last' if row['status'] == 'CHUNK_END' else 'terminal'
    if not row['decision_executed']:
        return 'unexecuted'
    if row['direction'] == 'NO_TRADE':
        return 'no_trade'
    if row['lots_after'] != 0:
        return 'held_forced_wait'
    if not np.asarray(step['action_mask']).all():
        return 'restricted_wait'
    return None


def build_labels(dataset, oracle_receipt, output, *, candidate='C1', seed=0, run_id='labels'):
    source = sources(dataset, oracle_receipt, candidate)
    root = Path(output).resolve()
    if any(root.is_relative_to(Path(p).resolve()) for p in (dataset, oracle_receipt)):
        raise ValueError('Output cannot be inside immutable sources')
    root.mkdir(parents=True, exist_ok=False)
    counts, strata, traces = Counter(), defaultdict(Counter), []
    for rec in source['traces']:
        directory = Path(dataset)/rec['directory']
        t = trace_manifest(directory, schema=SINGLE3_SCHEMA)
        plan = read_manifest(t['policy']['schedule_path'])
        rows, seen_cursors, seen_times, seen_stepids, local = [], set(), set(), set(), Counter()
        for row, step in aligned_rows(directory, t, set(plan['add_cursors'])):
            local['total_rows'] += 1
            why = exclusion(row, step)
            if why:
                local[why+'_excluded'] += 1
                continue
            if (row['cursor'] in seen_cursors or row['time'] in seen_times
                    or row['stepid'] in seen_stepids):
                raise ValueError('Duplicate supervised market minute')
            seen_cursors.add(row['cursor'])
            seen_times.add(row['time'])
            seen_stepids.add(row['stepid'])
            label = row['action']
            year = int(row['time'][:4])
            local['eligible'] += 1
            local['label_ADD' if label else 'label_WAIT'] += 1
            local['eligible_reset'] += int(row['reset'])
            local['eligible_invalid_feedback_minute'] += int(not row['decision_valid_minute'])
            local['lots_before_filter_would_lose'] += int(row['lots_before'] != 0)
            for dim, value in [('contract', rec['contract']), ('year', str(year)),
                               ('direction', row['direction']), ('session', row['session'])]:
                strata[dim+'/'+value]['ADD' if label else 'WAIT'] += 1
            rows.append((row['ordinal'], row['cursor'], step['stepid'], label, step['action_mask'],
                         row['is_first'], row['decision_valid_minute'], year,
                         {'LONG': 1, 'SHORT': -1, 'NO_TRADE': 0}[row['direction']], int(row['session'] == 'night')))
        if local['total_rows'] != t['records'] or local['label_ADD'] != len(plan['add_cursors']):
            raise ValueError('Teacher schedule/full trace count mismatch')
        name = rec['contract']+'.labels.npy'
        np.save(root/name, np.array(rows, dtype=INDEX_DTYPE), allow_pickle=False)
        traces.append(dict(rec, labels_file=name, samples=len(rows), counts=dict(local)))
        counts.update(local)
        print(f"labels {rec['contract']}: {len(rows)} (ADD {local['label_ADD']})", flush=True)
    check_files(source['frozen_inputs'])
    summary = dict(counts= dict(counts), strata={k: dict(v) for k, v in sorted(strata.items())},
                   add_rate=counts['label_ADD']/counts['eligible'],
                   always_wait_accuracy=counts['label_WAIT']/counts['eligible'],
                   always_wait_pr_auc=counts['label_ADD']/counts['eligible'])
    write_json(root/'audit.json', summary)
    source.pop('traces')
    result = signed(dict(protocol=PROTOCOL, status='PASS', candidate='C1', seed=seed, run_id=run_id,
                         source=source, traces=traces, summary=summary, provenance=provenance(),
                         files={p.name: digest(p) for p in root.iterdir() if p.is_file()}))
    write_json(root/'manifest.json', result)
    return result


def load_labels(directory, *, verify_sources=True):
    root = Path(directory).resolve()
    m = read_manifest(root/'manifest.json')
    if m.get('protocol') != PROTOCOL or m.get('status') != 'PASS' or m.get('candidate') != 'C1':
        raise ValueError('Foreign labels cache')
    if list(m['provenance']['code_sha256'].values()) != [digest(Path(__file__).resolve())]:
        raise ValueError('Labels generator implementation identity changed')
    check_files({str(root/p): h for p, h in m['files'].items()})
    if verify_sources:
        check_files(m['source']['frozen_inputs'])
    for rec in m['traces']:
        a = np.load(root/rec['labels_file'], mmap_mode='r', allow_pickle=False)
        if (a.dtype != INDEX_DTYPE or a.shape != (rec['samples'],)
                or not np.isin(a['label'], [0, 1]).all() or not np.all(a['mask'] == 1)):
            raise ValueError('Invalid labels index dtype/shape/mask')
    return m

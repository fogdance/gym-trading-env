"""CPU acceptance of the supplied Stage 1 patch on authenticated TRAIN FG201.

No training, TEST reads, reward edits or existing-run writes. The reference
replay constructs a new native Env; it does not use the patch's fork helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import check_files, digest, signed
from gym_trading_env.research.fg_single3 import load_env
from gym_trading_env.research.fg_s3_oracle_state_query import load_train_query
from gym_trading_env.research import fg_s3_quality_teacher as teacher


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def observations(env):
    return {k: hashlib.sha256(v.tobytes()).hexdigest()
            for k, v in env._get_obs().items()}


def reference(env):
    """One ADD, then only WAIT, across native chunk continuation."""
    initial = env.export_state()
    old = len(env.attributions)
    feedback = []
    first = True
    while not env.terminated and env.status not in ('DATA_END', 'DATA_GAP'):
        if env.truncated:
            env.reset()
        cursor = env.current_step
        _, reward, *_ = env.step(int(first))
        first = False
        feedback.append([int(env.current_step), float(np.float32(reward))])
        if len(env.attributions) > old or env.current_step == cursor:
            break
    env._assert_ledger_conservation()
    return dict(initial_state_hash=fingerprint(initial), feedback=feedback,
                events=env.attributions[old:], final_state=env.export_state())


def run(args):
    began = time.monotonic()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    query, live, row, frozen, _ = load_train_query(
        args.env_receipt, args.oracle_receipt, 'CZCE.FG201')
    source = query.identity['source']
    inputs = {p: h for group in ('receipt_files', 'candidate_files',
                               'checked_env_code', 'strategy_files')
              for p, h in source.get(group, {}).items()}
    inputs.update({p: frozen[p] for p in
                   (row['path'], row['env_config'], row['contract_config'])})
    check_files(inputs)
    live.reset(seed=0)
    states = [(0, live.export_state())]
    prefix_actions = []
    sl = 0
    while sl < 2 and len(prefix_actions) < 3000:
        assert not live.terminated and live.status not in ('DATA_END', 'DATA_GAP')
        if live.truncated:
            live.reset()
        old = len(live.attributions)
        action = int(live._published_mask[1])
        prefix_actions.append(action)
        live.step(action)
        for event in live.attributions[old:]:
            if event['reason'] == 'STOP_LOSS':
                sl += 1
                assert not live.entries and live._published_mask.tolist() == [1.0, 1.0]
                if live.truncated:
                    live.reset()
                states.append((sl, live.export_state()))
    assert sl == 2, 'No complete two-SL real-account prefix within the budget'
    write(root / 'native_prefix.json', dict(policy='ADD whenever actually legal',
                                           actions=prefix_actions))
    # A separate WAIT-only trajectory gives 300k at the SAME cursors; no wallet
    # field is hand-edited and no CandidateIndex.empty_at() is used.
    wait_only = load_env(row, frozen)
    wait_only.reset(seed=0)
    comparisons = []
    for count, state in states[1:]:
        while wait_only.current_step < state['current_step']:
            if wait_only.truncated:
                wait_only.reset()
            wait_only.step(0)
        if wait_only.truncated:
            wait_only.reset()
        comparisons.append((f'wait_only_at_sl_{count}', wait_only.export_state()))
    cases = [(f'after_{n}_sl', s) for n, s in states] + comparisons
    summaries = []
    market_hash = fingerprint(live._raw.to_dict(orient='split'))
    feature_hash = hashlib.sha256(live._features.tobytes()).hexdigest()
    for name, state in cases:
        live.import_state(state)
        sentinel = root / f'{name}.state_path.sentinel'
        sentinel.write_text('MUST REMAIN UNCHANGED\n')
        live.entry_config['state_path'] = str(sentinel)
        sentinel_hash = digest(sentinel)
        before = live.export_state()
        before_obs = observations(live)
        captured = []
        native_fork = teacher.fork_account

        def capture(*items):
            obj = native_fork(*items)
            captured.append(obj)
            return obj

        # Capture the exact object used by the unmodified patch, to compare
        # its entire final account against an independently constructed Env.
        with patch.object(teacher, 'fork_account', side_effect=capture):
            result = teacher.quality_record(live, fold='TRAIN', strict_query=query)
        assert len(captured) == 1
        fork_final = captured[0].export_state()
        independent = load_env(row, frozen)
        independent.import_state(state)
        direct = reference(independent)
        assert fork_final == direct['final_state'], 'Native full account mismatch'
        assert live.export_state() == before, 'Live account mutation'
        assert observations(live) == before_obs, 'Live observations changed'
        assert fingerprint(live._raw.to_dict(orient='split')) == market_hash
        assert hashlib.sha256(live._features.tobytes()).hexdigest() == feature_hash
        assert digest(sentinel) == sentinel_hash, 'Fork wrote the live state path'
        assert result['decision_snapshot_hash'] == fingerprint(state)
        assert result['training_action_label_allowed'] is False
        assert result['add_better_than_wait'] is None
        events = direct['events']
        assert len(events) == 1, 'Selected FG201 case must have a committed exit'
        event = events[0]
        assert result['exit_event_hash'] == fingerprint(event)
        assert result['net_trade_pnl'] == str(event['net_trade_pnl'])
        assert result['market_minutes'] == event['held_market_bars']
        assert result['old_reward_undiscounted'] == sum(x[1] for x in direct['feedback'])
        # Same post-hoc API on DEV is diagnostic-only and cannot mutate state.
        dev = teacher.quality_record(live, fold='DEV')
        assert dev['quality_score'] == result['quality_score']
        assert not dev['training_action_label_allowed']
        assert live.export_state() == before
        if float(result['starting_equity']) < 300000:
            assert not result['strict_decision_valid']
            assert result['strict_oracle']['reason'] == 'FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE'
            assert result['strict_oracle']['action'] is None
            assert result['quality_valid'], 'Low funds wrongly suppressed fact supervision'
        strict = query.query(live, live._published_mask)
        assert strict['q_add'] is None and strict['q_wait'] is None
        write(root / f'{name}.json', dict(starting_state=state, quality=result,
              independent_reference=direct, patch_final_state_hash=fingerprint(fork_final),
              original_state_unchanged=True, observations_unchanged=True,
              market_and_features_unchanged=True, state_path_unchanged=True))
        summary = dict(case=name, cursor=state['current_step'],
            equity=result['starting_equity'], outcome=result['outcome'],
            quality_valid=result['quality_valid'], score=result['quality_score'],
            net_trade_pnl=result['net_trade_pnl'], market_minutes=result['market_minutes'],
            strict_valid=result['strict_decision_valid'],
            strict_reason=result['strict_oracle']['reason'],
            snapshot_hash=fingerprint(state),
            full_native_final_state_match=True, source_unchanged=True)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    assert sum(float(s['equity']) < 300000 for s in summaries) >= 2
    check_files(inputs)
    code = {str(repo / p): digest(repo / p) for p in (
        'src/gym_trading_env/research/fg_s3_quality_score.py',
        'src/gym_trading_env/research/fg_s3_quality_teacher.py',
        'tools/fg_s3_quality_patch_acceptance.py')}
    receipt = signed(dict(schema='FG_S3_QUALITY_FIRST_PATCH_REAL_ACCOUNT_ACCEPTANCE_V1',
        status='PASS_REAL_TRAIN_FG201_ACCOUNT_REPLAY', fold='TRAIN',
        scope='Stage1 native account outcome teacher only', cases=summaries,
        gym_base_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        dreamer_base_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo.parent/'dreamerv3', text=True).strip(),
        code_files=code, frozen_inputs=inputs, oracle_identity=query.identity_hash,
        numeric_oracle_q='UNAVAILABLE', quality_is_action_label=False,
        model_updates=0, gpu='NOT_USED', test_fold='NOT_READ',
        elapsed_seconds=time.monotonic()-began,
        files={str(p.relative_to(root)): digest(p) for p in root.iterdir() if p.is_file()}))
    write(root / 'receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-receipt', required=True)
    parser.add_argument('--oracle-receipt', required=True)
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    output_existed = Path(arguments.output).exists()
    try:
        receipt = run(arguments)
        print(receipt['status'], flush=True)
    except BaseException as error:
        output = Path(arguments.output)
        if not output_existed and output.is_dir() and not (output/'receipt.json').exists():
            write(output/'failure.json', dict(status='FAIL', error=repr(error)))
        raise

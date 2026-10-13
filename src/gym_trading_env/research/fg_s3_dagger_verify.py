"""Independent original-Env counterfactuals at real TRAIN suffix states."""
import time

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_single3 import load_env
from .fg_single3_oracle import empty_at


def replay_add(env):
    """One original ADD interval, preserving the starting financial history."""
    start, old_fills, old_attrs = env.current_step, len(env.fills), len(env.attributions)
    rewards = []
    obs, reward, *_ = env.step(1)
    if reward:
        rewards.append([env.current_step, float(np.float32(reward))])
    filled = any(f['reason'] == 'ENTRY' for f in env.fills[old_fills:])
    while filled and env.entries and not env.terminated and env.status not in ('DATA_END', 'DATA_GAP'):
        if env.truncated:
            env.reset()
        before = env.current_step
        obs, reward, *_ = env.step(0)
        if reward:
            rewards.append([env.current_step, float(np.float32(reward))])
        if env.current_step == before:
            break
    event = env.attributions[old_attrs] if len(env.attributions) > old_attrs else None
    return dict(start_cursor=start, filled=filled, exit=event,
                exit_cursor=env.current_step if event else None, rewards=rewards)


def verify_real_oracle(query, row, frozen, candidates):
    started = time.perf_counter()
    by_cursor = {r['cursor']: r for r in candidates}
    initial_schedule = set(query.plan['add_cursors'])
    # Explicitly include states not on the initial P4 optimal path.
    adds = [t for t in range(query.end) if query.first[t] == t and t not in initial_schedule]
    if len(adds) < 3:
        adds = [t for t in range(query.end) if query.first[t] == t]
    waits = [t for t in range(query.end) if query.first[t] != t and by_cursor[t].get('eligible')]
    cursors = adds[:3] + waits[:3]
    cases = []
    env = load_env(row, frozen)
    for t in cursors:
        obs = empty_at(env, t)
        before = env.export_state()
        answer = query.query(env, obs['action_mask'])
        assert answer['valid'] and answer['certified'] and env.export_state() == before
        requested = answer['action']
        # Recommended WAIT is checked independently, including its cashflow.
        if requested == 0:
            obs, reward, *_ = env.step(0)
            assert reward == 0 and not env.fills
            env.import_state(before)
        actual = replay_add(env)
        expected = by_cursor[t]
        assert actual['filled'] == expected.get('filled', False)
        if actual['filled']:
            assert actual['rewards'] == expected['rewards']['C1']
            assert actual['exit_cursor'] == expected['exit_cursor']
            if actual['exit']:
                assert all(actual['exit'][k] == v for k,v in expected['exit'].items() if k != 'entry_id')
        cases.append(dict(kind='FRESH_SUFFIX_COUNTERFACTUAL', cursor=t, answer=answer,
            on_initial_P4_schedule=t in initial_schedule, actual=actual,
            state_id=fingerprint(before), status='PASS'))

    # Financial history is produced by actual original-account fills, not an
    # overwritten cash balance. Query after two profitable intervals.
    obs = empty_at(env, query.plan['add_cursors'][0])
    for number in range(2):
        while env.current_step < query.end:
            if env.truncated:
                obs, _ = env.reset()
            answer = query.query(env, obs['action_mask'])
            if answer['valid'] and answer['action'] == 1:
                break
            obs, *_ = env.step(0)
        actual = replay_add(env)
        assert actual['exit']['reason'] == 'TAKE_PROFIT'
        if env.truncated:
            env.reset()
        obs = env._get_obs()
        before = env.export_state()
        answer = query.query(env, obs['action_mask'])
        assert answer['valid'] and env.export_state() == before
        cases.append(dict(kind='ACTUAL_PROFIT_HISTORY', completed_TP=number+1,
            cursor=env.current_step, cash=str(env._calculate_equity()),
            answer=answer, status='PASS'))

    loss = next(r for r in candidates if r.get('exit',{} ) and r['exit']['reason']=='STOP_LOSS')
    empty_at(env, loss['cursor'])
    replay_add(env)
    if env.truncated and env.status=='CHUNK_END':
        env.reset()
    answer = query.query(env, env._get_obs()['action_mask'])
    assert not answer['valid'] and not answer['certified']
    cases.append(dict(kind='ACTUAL_LOSS_HISTORY_SKIPPED', cursor=env.current_step,
        cash=str(env._calculate_equity()), answer=answer, status='PASS'))

    obs = empty_at(env, len(env._raw)-1)
    answer = query.query(env, obs['action_mask'])
    cursor = env.current_step
    _, reward, _, truncated, info = env.step(0)
    assert not answer['valid'] and truncated and env.current_step == cursor and reward==0
    cases.append(dict(kind='REAL_CONTRACT_END', cursor=cursor, answer=answer,
                      execution_status=info['status'], status='PASS'))
    env.close()
    reject = [r for r in candidates if r.get('eligible') and not r.get('filled')]
    return dict(status='PASS_REAL_TRAIN_ENV_COUNTERFACTUALS', oracle_identity=query.identity_hash,
        contract=row['contract'], cases=cases, seconds=time.perf_counter()-started,
        scope='Original suffix proof plus independent Env interval checks; not exhaustive full-contract action search',
        real_next_bar_reject_candidates=len(reject),
        real_gap='NOT_RUN_COMPLETE_CALENDAR', real_tie='NOT_RUN_USE_ORIGINAL_ENV_CPU_FIXTURE',
        production_training='NOT_RUN', dev='NOT_RUN', test='NOT_RUN')

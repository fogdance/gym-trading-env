"""Closed-loop ABC evaluation and trajectory-conditioned random diagnostics."""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from gym_trading_env.envs.entry_contract import timestamp
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from .fg_entry_v1 import baseline_action, write_json


def slot_key(env, index):
    """Known execution clock and manual direction only; no future prices/volume."""
    time=env.bobs[index+1]
    minute=time.hour*60+time.minute
    if minute>=21*60:
        clock=minute-21*60
        session='night'
    else:
        session='day'
        clock=(120+minute-9*60 if minute<10*60+15 else
               195+minute-(10*60+30) if minute<11*60+30 else
               255+minute-(13*60+30))
    day=env._trading_day(env._raw.index[index+1])
    return (str(day),env.direction_at(time,index+1),session,int(clock//60))


def replay_policy(config_path, output, *, strategy=None, policy=None, seed=0, quotas=None):
    """A policy callback receives (obs, first, previous_reward); one account only."""
    if sum(x is not None for x in (strategy,policy,quotas))!=1:
        raise ValueError('Select one baseline, model, or matched quota policy')
    output=Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite an evaluation')
    env=FixedDirectionEntryEnv(config_path=str(config_path))
    output.mkdir(parents=True)
    rng=np.random.default_rng(seed)
    keys=[slot_key(env,i) for i in range(len(env._raw)-1)]
    slots=Counter(keys)
    target=Counter(quotas or {})
    filled=Counter()
    actions=[]
    obs=env._get_obs();first=True;previous_reward=0.0
    capacity=0;observed=0
    while not env.terminated and env.status not in ('DATA_END','DATA_GAP'):
        i=env.current_step
        if i>=len(keys):
            env.step(0)
            break
        key=keys[i]
        legal=bool(obs['action_mask'][1])
        if policy is not None:
            action=int(policy(obs,first,previous_reward))
        elif quotas is not None:
            probability=min(1,max(0,target[key]-filled[key])/max(1,slots[key]))
            action=int(legal and rng.random()<probability)
        else:
            action=baseline_action(env,strategy,rng,.5)
        count=len(env.fills)
        # Denominator counts observable minutes, not overnight closures.
        if key[1]!='NO_TRADE':
            observed+=1
            capacity+=int(env.net_lots>=6)
        obs,reward,terminal,truncated,info=env.step(action)
        new=[f for f in env.fills[count:] if f['reason']=='ENTRY' and f['committed']]
        filled[key]+=len(new)
        slots[key]-=1
        actions.append(dict(signal_timestamp=str(env._raw.index[i]),execution_clock=str(env.bobs[min(i+1,len(env.bobs)-1)]),
            trading_date=key[0],direction=key[1],session=key[2],minute_bin=key[3],action=action,
            legal_add=legal,add_fill=len(new),entry_ids=';'.join(f['entry_id'] for f in new),reward=reward,
            status=info['status'],net_lots=float(env.net_lots)))
        previous_reward=reward;first=False
        if truncated and env.status=='CHUNK_END':
            obs,_=env.reset();first=True
    events={e['entry_id']:e for e in env.attributions}
    rewards={e['entry_id']:e for e in env.reward_events}
    trades=[]
    horizon=[5,15,60]
    trading_days=sorted({str(env._trading_day(t)) for t in env._raw.index})
    end=env._raw.index[env.current_step]
    for fill in env.fills:
        if fill['reason']!='ENTRY':
            continue
        event=events.get(fill['entry_id'])
        reward=rewards.get(fill['entry_id'])
        start=timestamp(fill['time'])
        stop=timestamp(event['exit_time']) if event else end
        duration=(stop-start).total_seconds()/60
        status='success' if reward else 'terminal_failure' if event else 'right_censored'
        bars=env._raw.loc[start:stop]
        price=float(fill['price'])
        adverse=bars.Low.min() if fill['side']=='LONG' else bars.High.max()
        mae=max(0,price-adverse if fill['side']=='LONG' else adverse-price)
        trade=dict(contract=env.contract,entry_id=fill['entry_id'],direction=fill['side'],entry_timestamp=str(start),
            entry_price=price,exit_timestamp=str(stop) if event else None,completion_status=status,
            duration_minutes=duration,duration_trading_bars=max(0,len(bars)-1),MAE_points=float(mae),
            reward=reward['reward'] if reward else 0,exit_reason=event['reason'] if event else env.status)
        for h in horizon:
            trade[f'success_within_{h}_minutes']=bool(reward and duration<=h)
            trade[f'observable_{h}_minutes']=bool(event or duration>=h)
        first_day=trading_days.index(str(env._trading_day(start)))
        last_day=trading_days.index(str(env._trading_day(stop)))
        for h in (1,5):
            trade[f'success_within_{h}_trading_days']=bool(reward and last_day-first_day<h)
            # Horizon closes at the end of its h-th known trading day.
            trade[f'observable_{h}_trading_days']=bool(event or last_day-first_day>=h or
                first_day+h-1<len(trading_days) and stop==env._raw.index[-1] and last_day-first_day==h-1)
        trades.append(trade)
    calculated=sum(60/(60+e['elapsed_minutes']) for e in env.reward_events)
    if not np.isclose(calculated,env.total_reward,rtol=0,atol=1e-10) or len(rewards)!=len(env.reward_events):
        raise RuntimeError('Reward reconciliation failed')
    if env.ledger.total_balance()!=300000 or any(a['net_lots']>6 for a in actions):
        raise RuntimeError('Account conservation or capacity reconciliation failed')
    direction_days={str(env._trading_day(t)) for i,t in enumerate(env._raw.index) if env.direction_at(t,i)!='NO_TRADE'}
    all_days={str(env._trading_day(t)) for t in env._raw.index}
    equity=np.array([r['equity'] for r in env.equity_history])
    success=[t['duration_minutes'] for t in trades if t['completion_status']=='success']
    summary=dict(env._get_info(),strategy=strategy or ('matched_random' if quotas is not None else 'dreamerv3'),seed=seed,
        total_reward=env.total_reward,eligible_trading_days=len(direction_days),raw_trading_days=len(all_days),
        reward_per_day=env.total_reward/len(direction_days) if direction_days else None,
        completed_batches=len(rewards),terminal_failure_batches=sum(t['completion_status']=='terminal_failure' for t in trades),
        right_censored_batches=sum(t['completion_status']=='right_censored' for t in trades),
        median_tp_minutes=float(np.median(success)) if success else None,p90_tp_minutes=float(np.percentile(success,90)) if success else None,
        open_batches_at_end=len(env.entries),open_lots_at_end=float(env.net_lots),
        full_capacity_time_fraction=capacity/observed if observed else None,
        max_equity_drawdown=float((np.maximum.accumulate(equity)-equity).max()),
        net_equity_change=float(equity[-1]-300000),end_equity=float(equity[-1]),
        fees=sum(float(r.fee) for r in env.trade_record_manager.trade_history),
        realized_pnl=float(env.user_accounts.realized_pnl),
        ADD_request=sum(a['action']==1 for a in actions),ADD_fill=sum(a['add_fill'] for a in actions),
        WAIT_voluntary=sum(a['action']==0 and a['legal_add'] for a in actions),
        WAIT_masked=sum(a['action']==0 and not a['legal_add'] for a in actions),
        action_mask_error=sum(a['action']==1 and not a['legal_add'] for a in actions),reward_reconcile_error=0,
        simulated_fills=True,conditional_direction_only=True,calendar_basis='user_confirmed_observed_clock')
    if quotas is not None:
        differences=[dict(trading_date=k[0],direction=k[1],session=k[2],minute_bin=k[3],
            requested=target[k],filled=filled[k],difference=filled[k]-target[k]) for k in sorted(set(target)|set(filled)) if target[k] or filled[k]]
        summary['match_status']='exact_match' if all(r['difference']==0 for r in differences) else 'partial_match'
        pd.DataFrame(differences).to_csv(output/'quota_differences.csv',index=False)
    for unit,limits in [('minutes',horizon),('trading_days',(1,5))]:
        for h in limits:
            eligible=sum(t[f'observable_{h}_{unit}'] for t in trades)
            wins=sum(t[f'success_within_{h}_{unit}'] for t in trades)
            summary[f'success_count_{h}_{unit}']=wins
            summary[f'observable_count_{h}_{unit}']=eligible
            summary[f'censored_count_{h}_{unit}']=len(trades)-eligible
            summary[f'success_rate_{h}_{unit}']=wins/eligible if eligible else None
    for name,rows in [('trades',trades),('policy_actions',actions),('fills',env.fills),('reward_events',env.reward_events),('account_equity',env.equity_history)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    pd.DataFrame([summary]).to_csv(output/'episode_metrics.csv',index=False)
    write_json(output/'account_state.json',env.export_state())
    write_json(output/'summary.json',summary)
    env.close()
    return summary


def matched_quotas(actions_path):
    actions=pd.read_csv(actions_path)
    actions['trading_date']=actions['trading_date'].astype(str)
    return Counter({tuple(k):int(v) for k,v in actions.groupby(['trading_date','direction','session','minute_bin']).add_fill.sum().items() if v})

#!/usr/bin/env python3
"""Compute FG V0/V2/V3 OFFLINE; never replaces the Env's training reward.

Modes:
  --trace events.jsonl                 already-observed causal bar/fill records
  --env-config CFG --eval-dir DIR      replay previously logged real policy actions,
                                      WITHOUT loading the policy or predicting
  --env-config CFG --policy always_add|always_wait|random|delay5|delay15|delay30
                                      independent CPU closed-loop Env baseline

  --study-config FILE                 frozen FG409 offline parameter study

See docs/FG_Reward_V2_V3_Implementation_Codex_Guide.md for provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from decimal import Decimal
from math import isclose, isfinite

MODULE = Path(__file__).resolve().parents[1] / 'src/gym_trading_env/research/fg_reward_candidates.py'
spec = importlib.util.spec_from_file_location('fg_reward_candidates_standalone', MODULE)
lib = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lib
spec.loader.exec_module(lib)


def _hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path, rows, keys=None):
    rows = list(rows)
    keys = keys or list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh,fieldnames=keys,extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def _write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,
                                   default=str,allow_nan=False),encoding='utf-8')


def _read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as fh:
        return list(csv.DictReader(fh))


def _reconcile(label, old, actual, *, text=(), numbers=(), booleans=()):
    """Compare event identities and amounts; equal totals are insufficient."""
    if len(old)!=len(actual):
        raise RuntimeError(f'{label} record count differs: {len(old)} != {len(actual)}')
    for index,(expected,received) in enumerate(zip(old,actual)):
        for key in text:
            if key not in expected or str(expected[key])!=str(received.get(key)):
                raise RuntimeError(f'{label} row {index}: nonidentical {key}')
        for key in numbers:
            if key not in expected or key not in received:
                raise RuntimeError(f'{label} row {index}: missing {key}')
            a,b=Decimal(str(expected[key])),Decimal(str(received[key]))
            if not a.is_finite() or not b.is_finite() or abs(a-b)>Decimal('1e-9'):
                raise RuntimeError(f'{label} row {index}: nonidentical {key}')
        for key in booleans:
            if key not in expected or lib._true_flag(expected[key])!=lib._true_flag(received.get(key)):
                raise RuntimeError(f'{label} row {index}: nonidentical {key}')


def opportunity_clock(env, current):
    """Known execution clock/manual directions; no capacity, money or OHLC."""
    if current<1 or not env.valid_fill_clock[current]:
        return False
    opening=env.bobs[current]
    signal=env.direction_at(env._raw.index[current-1],current-1)
    execution=env.direction_at(opening,current)
    return bool(signal!='NO_TRADE' and signal==execution and
                opening<env.terms.last_open and
                (env.main_exit_at is None or opening<env.main_exit_at))


def delay_counter(counter, *, legal, eligible, opened):
    if opened or not legal:
        return 0
    return counter+int(eligible)


def _original_summary(env, actions, results):
    import numpy as np
    eligible_days={env._trading_day(env._raw.index[i]) for i in range(1,len(env._raw))
                   if opportunity_clock(env,i)}
    observed_days={r['trading_date'] for r in results if r['eligible_market_minute']}
    opportunities=[r for r in results if r['eligible_market_minute']]
    legacy_minutes=[r for r in results if r['direction']!='NO_TRADE']
    equity=np.array([r['equity'] for r in env.equity_history])
    completed=[(lib._clock(e['exit_time'])-lib._clock(e['entry_time'])).total_seconds()/60
               for e in env.attributions if e['reason']=='TAKE_PROFIT']
    return dict(env._get_info(),eligible_trading_days=len(eligible_days),
        observed_eligible_trading_days=len(observed_days),
        eligible_market_minutes=len(opportunities),valid_market_minutes=sum(r['valid_bar'] for r in results),
        metadata_rows=sum(not r['valid_bar'] for r in results),
        fees=sum(float(r.fee) for r in env.trade_record_manager.trade_history),
        realized_pnl=float(env.user_accounts.realized_pnl),
        max_equity_drawdown=float((np.maximum.accumulate(equity)-equity).max()),
        net_equity_change=float(equity[-1]-300000),end_equity=float(equity[-1]),
        median_tp_minutes=float(np.median(completed)) if completed else None,
        p90_tp_minutes=float(np.percentile(completed,90)) if completed else None,
        full_capacity_interval_fraction=(sum(r['active_slots_for_interval']==2 for r in opportunities)
                                         /len(opportunities) if opportunities else None),
        full_capacity_time_fraction=(sum(r['lots_before']>=6 for r in legacy_minutes)
                                     /len(legacy_minutes) if legacy_minutes else None),
        ADD_request=sum(r['action']==1 for r in actions),ADD_fill=sum(r['add_fill'] for r in actions),
        WAIT_voluntary=sum(r['action']==0 and r['legal_add'] for r in actions),
        action_mask_error=sum(r['action']==1 and not r['legal_add'] for r in actions),
        terminal_failure_unfilled=len(env.entries) if env.pending_reason else 0,
        right_censored=len(env.entries) if not env.pending_reason else 0)


def _replay_existing_env(args, scorer):
    # Existing Env is the sole authority for fills, TP, ledger and fees.
    # The module is deliberately loaded only in this mode; the standalone core
    # needs no numpy, pandas, gym, JAX or Dreamer model.
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    frozen_hashes={}
    if args.manifest is not None:
        from gym_trading_env.research.fg_abcd_evaluation import validate_validation_inputs
        frozen_hashes=validate_validation_inputs(args.manifest, args.env_config)
    env = FixedDirectionEntryEnv(config_path=str(args.env_config))
    actions = None
    if args.eval_dir:
        actions = _read_csv(Path(args.eval_dir)/'policy_actions.csv')
    original_hashes={str(p.resolve()):_hash(p) for p in Path(args.eval_dir).iterdir()
                     if p.is_file() and p.suffix in ('.csv','.json')} if args.eval_dir else {}
    rng=random.Random(args.seed)
    counter=0
    action_cursor=0
    results=[]
    logged=[];events=[];chunk_checks=0
    try:
        obs=env._get_obs()
        while not env.terminated and env.status not in ('DATA_END','DATA_GAP'):
            index=env.current_step
            if index+1>=len(env._raw):
                env.step(0)  # marks DATA_END, preserving pending exits and positions
                break
            now=env._raw.index[index]
            eligible=opportunity_clock(env,index+1)
            legal=bool(obs['action_mask'][1])
            if actions is not None:
                if action_cursor>=len(actions) or str(now)!=str(actions[action_cursor]['signal_timestamp']):
                    raise RuntimeError('Frozen action sequence does not match executable market clock')
                action=int(actions[action_cursor]['action'])
                action_cursor+=1
            elif args.policy=='always_add':
                action=int(legal)
            elif args.policy=='always_wait':
                action=0
            elif args.policy=='random':
                action=int(legal and rng.random()<0.5)
            elif args.policy.startswith('delay'):
                limit=int(args.policy.removeprefix('delay'))
                action=int(legal and counter>=limit)
            else:
                raise ValueError('Policy or previous actions required')
            before_fills=len(env.fills);before_attrs=len(env.attributions)
            lots_before=float(env.net_lots)
            obs, official_reward, terminated, truncated, _=env.step(action)
            new_fills=env.fills[before_fills:]
            entries=[f for f in new_fills if f['reason']=='ENTRY']
            recorded=dict(signal_timestamp=str(now),execution_clock=str(env.bobs[index+1]),
                          action=action,legal_add=legal,add_fill=len(entries),
                          entry_ids=';'.join(f['entry_id'] for f in entries),
                          reward=official_reward,status=env.status,net_lots=float(env.net_lots))
            logged.append(recorded)
            if env.current_step==index:
                break
            current=env.current_step
            row=env._raw.iloc[current]
            exits=env.attributions[before_attrs:]
            if args.policy and args.policy.startswith('delay'):
                counter=delay_counter(counter,legal=legal,eligible=eligible,opened=bool(entries))
            unit=dict(bar_open=str(env.bobs[current]),bar_end=str(env._raw.index[current]),
                      open=float(row.Open),high=float(row.High),low=float(row.Low),close=float(row.Close),
                      valid_bar=bool(env.valid_fill_clock[current]),
                      eligible_market_minute=bool(eligible and env.valid_fill_clock[current]),
                      entries=entries,exits=exits,chunk_end=bool(truncated and env.status=='CHUNK_END'),
                      true_terminal=bool(terminated))
            result=scorer.step(unit)
            result.update(trading_date=env._trading_day(env._raw.index[current]),
                          direction=env.direction_at(env.bobs[current],current),
                          lots_before=lots_before,net_lots=float(env.net_lots),
                          action=action,legal_add=legal,official_reward=official_reward)
            results.append(result)
            events.append(unit)
            if abs(result['v0']-official_reward)>1e-9:
                raise RuntimeError('Old TP Reward disagrees with official Env on actual bar')
            if truncated and env.status=='CHUNK_END':
                state=(env.total_reward,float(env.net_lots),tuple(env.entries),env.ledger.snapshot())
                restored=lib.FGRewardScorer.restore(json.loads(json.dumps(scorer.snapshot())),scorer.config)
                if restored.snapshot()!=scorer.snapshot():
                    raise RuntimeError('Scorer chunk state changed during restore')
                scorer.__dict__.update(restored.__dict__)
                obs,_=env.reset()
                if state!=(env.total_reward,float(env.net_lots),tuple(env.entries),env.ledger.snapshot()):
                    raise RuntimeError('Chunk reset changed the financial account')
                chunk_checks+=1
        if actions is not None and action_cursor!=len(actions):
            raise RuntimeError('Not all logged policy decisions were replayed')
        if abs(scorer.totals['v0']-env.total_reward)>1e-8:
            raise RuntimeError('Replayed V0 does not reconcile with business Reward')
        if actions is not None:
            _reconcile('Action',actions,logged,text=('signal_timestamp',),numbers=('action',))
            _reconcile('Fill',_read_csv(Path(args.eval_dir)/'fills.csv'),env.fills,
                       text=('entry_id','side','reason','time'),numbers=('quantity','price'),booleans=('committed',))
            _reconcile('Reward',_read_csv(Path(args.eval_dir)/'reward_events.csv'),env.reward_events,
                       text=('entry_id','side','reason','entry_time','exit_time'),
                       numbers=('elapsed_minutes','reward','entry_price','exit_price','closed_quantity'),
                       booleans=('committed',))
            archived=json.loads((Path(args.eval_dir)/'account_state.json').read_text())
            _reconcile('Attribution',archived['attributions'],env.attributions,
                       text=('entry_id','side','reason','entry_time','exit_time'),
                       numbers=('entry_price','exit_price','closed_quantity'),booleans=('committed',))
            if (Path(args.eval_dir)/'account_equity.csv').is_file():
                _reconcile('Equity',_read_csv(Path(args.eval_dir)/'account_equity.csv'),env.equity_history,
                           text=('time',),numbers=('equity','unrealized_pnl','net_lots'),booleans=('intrabar',))
            # Older hand-made traces may have only signal/action; actual pilot
            # CSVs have all decision/fill/account fields and must match them.
            for key in ('execution_clock','entry_ids','status'):
                if actions and key in actions[0]:
                    _reconcile('Action',actions,logged,text=(key,))
            for key in ('add_fill','reward','net_lots'):
                if actions and key in actions[0]:
                    _reconcile('Action',actions,logged,numbers=(key,))
            if actions and 'legal_add' in actions[0]:
                _reconcile('Action',actions,logged,booleans=('legal_add',))
        metrics=_original_summary(env,logged,results)
        if actions is not None:
            old_summary=json.loads((Path(args.eval_dir)/'summary.json').read_text())
            for key in ('tp_count','non_tp_exits','max_floating_loss','account_net_pnl',
                        'fees','realized_pnl','max_equity_drawdown','full_capacity_time_fraction',
                        'ADD_request','ADD_fill','WAIT_voluntary'):
                if key in old_summary and not isclose(float(old_summary[key]),float(metrics[key]),abs_tol=1e-8,rel_tol=0):
                    raise RuntimeError(f'Frozen account summary differs: {key}')
        for path,checksum in (frozen_hashes|original_hashes).items():
            if _hash(path)!=checksum:
                raise RuntimeError(f'Read-only input changed during replay: {path}')
        state=env.export_state()
        return results, dict(original_env=metrics,policy=args.policy or 'logged_actions',
                             actions_replayed=action_cursor,real_fills=len(env.fills),
                             real_tp_attributions=sum(x['reason']=='TAKE_PROFIT' for x in env.attributions),
                             terminal_status=env.status,pending_reason=env.pending_reason,
                             seed=args.seed,chunk_restores_verified=chunk_checks,
                             frozen_input_hashes=frozen_hashes,original_evaluation_hashes=original_hashes,
                             _artifacts=dict(events=events,policy_actions=logged,fills=env.fills,
                                 reward_events=env.reward_events,account_state=state,account_equity=env.equity_history))
    finally:
        # FixedDirectionEntryEnv.close() can WRITE a configured account state
        # file. This is a strictly read-only research runner; do not call it.
        pass


def run_single(args):
    """One immutable trace/Env replay; write only its new research directory."""
    cfg=lib.CandidateConfig(beta=args.beta,rho=args.rho,lam=args.lam,alpha=args.alpha,gamma=args.gamma)
    scorer=lib.FGRewardScorer(cfg)
    if args.out.exists():
        raise FileExistsError(f'Output exists, refusing overwrite: {args.out}')
    # Reserve the directory before simulation; failures remain reviewable.
    args.out.mkdir(parents=True,exist_ok=False)
    software={str(p.resolve()):_hash(p) for p in (MODULE,Path(__file__))}
    _write_json(args.out/'input_receipt.json',dict(config=vars(cfg),source_hashes=software,
        env_config=str(args.env_config) if args.env_config else None,
        manifest=str(args.manifest) if args.manifest else None,
        eval_dir=str(args.eval_dir) if args.eval_dir else None,policy=args.policy,seed=args.seed))
    results=[];metadata={};artifacts={}
    if args.trace:
        checksum=_hash(args.trace)
        with open(args.trace,encoding='utf-8') as fh:
            for line_number,line in enumerate(fh,1):
                if line.strip():
                    try:
                        results.append(scorer.step(json.loads(line)))
                    except Exception as err:
                        raise RuntimeError(f'Trace line {line_number}: {err}') from err
        if _hash(args.trace)!=checksum:
            raise RuntimeError('Read-only trace changed during scoring')
        metadata={'source':'event_trace','trace_sha256':checksum}
    else:
        results,metadata=_replay_existing_env(args,scorer)
        artifacts=metadata.pop('_artifacts')
        metadata.update(frozen_manifest_verified=bool(args.manifest),
                        manifest_sha256=_hash(args.manifest) if args.manifest else None,
                        env_config_sha256=_hash(args.env_config),
                        policy_actions_sha256=_hash(Path(args.eval_dir)/'policy_actions.csv') if args.eval_dir else None)
    for path,checksum in software.items():
        if _hash(path)!=checksum:
            raise RuntimeError('Scoring source changed during replay')
    trades=scorer.trade_rows()
    for trade in trades:
        if trade['status']=='right_censored' and metadata.get('pending_reason'):
            trade['status']='terminal_failure_unfilled'
            trade['exit_reason']=metadata['pending_reason']
    _write_csv(args.out/'step_components.csv',results)
    _write_csv(args.out/'trade_components.csv',trades,keys=(list(trades[0]) if trades else
        ['entry_id','side','entry_time','entry_price','status','original_tp','timeboxed_tp',
         'early_penalty','capacity_basis','capacity_cost']))
    _write_json(args.out/'snapshot.json',scorer.snapshot())
    if artifacts:
        with (args.out/'events.jsonl').open('w',encoding='utf-8') as fh:
            for event in artifacts.pop('events'):
                fh.write(json.dumps(event,ensure_ascii=False,default=str,allow_nan=False)+'\n')
        for name in ('policy_actions','fills','reward_events','account_equity'):
            _write_csv(args.out/f'{name}.csv',artifacts[name])
        _write_json(args.out/'account_state.json',artifacts['account_state'])
    summary=scorer.summary()
    summary['input_provenance']=dict(metadata,source_hashes=software)
    if results:
        discounted_potential=sum(cfg.gamma**i*r['potential'] for i,r in enumerate(results))
        expected=cfg.alpha*(cfg.gamma**len(results)*results[-1]['phi_after']-results[0]['phi_before'])
        summary['pbrs']=dict(discounted_potential=discounted_potential,
                             endpoint_term=expected,residual=discounted_potential-expected,
                             constant_gamma_only=True)
        if abs(discounted_potential-expected)>1e-10:
            raise RuntimeError('PBRS constant-discount telescoping failed')
    _write_json(args.out/'summary.json',summary)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--trace',type=Path)
    src.add_argument('--env-config',type=Path)
    src.add_argument('--study-config',type=Path)
    p.add_argument('--eval-dir',type=Path,help='Previously frozen policy_actions.csv / fills.csv; no model inference')
    p.add_argument('--manifest',type=Path,help='ABC frozen development experiment manifest (strongly recommended)')
    p.add_argument('--policy',choices=['always_add','always_wait','random','delay5','delay15','delay30'])
    p.add_argument('--seed',type=int,default=0)
    for name in ('beta','rho','lam','alpha','gamma'):
        p.add_argument(f'--{name}',type=float,default=None)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args(argv)
    if args.study_config:
        if args.eval_dir or args.manifest or args.policy or any(
                getattr(args,k) is not None for k in ('beta','rho','lam','alpha','gamma')):
            p.error('Study config is the sole input/parameter source; use --study-config and --out')
        from gym_trading_env.research.fg_reward_study import run_study
        run_study(args.study_config,args.out)
        return
    if (args.eval_dir and not args.env_config) or (args.policy and not args.env_config) or (args.manifest and not args.env_config):
        p.error('--eval-dir / --policy require --env-config')
    if args.env_config and bool(args.eval_dir)==bool(args.policy):
        p.error('Specify exactly one of --eval-dir / --policy for Env replay')
    defaults=lib.CandidateConfig()
    for name in ('beta','rho','lam','alpha','gamma'):
        if getattr(args,name) is None:
            setattr(args,name,getattr(defaults,name))
    print(json.dumps(run_single(args),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()

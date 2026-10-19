"""Executed-trade quality and accounting; never feeds an Actor observation."""
from collections import Counter,defaultdict
from decimal import Decimal
import csv
import json
from pathlib import Path

import numpy as np

from .fg_replay import iter_sidecar,trace_manifest,write_json,E_QUALITY_SCHEMA
from .fg_s3_eval_metrics import quantile,unique_events


def _table(path,rows,fields=()):
    fields=list(dict.fromkeys([*fields,*(k for row in rows for k in row)]))
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def analyze_trace(directory,source,output):
    """Only original trace/fills and native executable quotes define results."""
    directory=Path(directory);output=Path(output)
    if output.exists():raise ValueError('EXAM_METRICS_OUTPUT_MUST_BE_NEW')
    manifest=trace_manifest(directory,schema=E_QUALITY_SCHEMA)
    rows=list(iter_sidecar(directory))
    state=json.loads((directory/'account_state.json').read_text())
    initial=json.loads((directory/'initial_state.json').read_text())
    if initial['entries']:raise ValueError('EXAM_EXPECTS_FRESH_FLAT_INITIAL_ACCOUNT')
    contract=manifest['contract'];attrs={e['entry_id']:e for e in unique_events(state['attributions'])}
    rewards=Counter()
    for event in unique_events(state['reward_events']):rewards[event['entry_id']]+=float(event['reward'])
    first_native={e['entry_id']:e for e in state['reward_events'] if e['reason']=='FIRST_ADVERSE10'}
    total_reward=float(state['total_reward'])-float(initial['total_reward'])
    if abs(sum(rewards.values())-total_reward)>1e-8:raise ValueError('EXAM_REWARD_EVENT_RECONCILIATION_FAILED')
    env=source.factory();trades={};decisions=[];coverage=defaultdict(lambda:Counter());daily_reward=Counter()
    row_days={r['time']:r['trading_date'] for r in rows}
    try:
        for row in rows:
            if row['advanced']:
                key=(contract,row['trading_date'],row['session'])
                coverage[key]['market_advances']+=1;coverage[key]['valid_minutes']+=int(row['valid_bar'])
                daily_reward[row['trading_date']]+=row['official_reward']
            if row['decision_executed']:
                diag=row.get('policy_diagnostics',{})
                decisions.append(dict(contract=contract,account_owner=row['account_owner'],stepid=row['stepid'],
                    cursor=row['cursor'],signal_time=row['time'],official_day=row['trading_date'],session=row['session'],
                    action=row['action'],legal_add=row['legal_add'],valid_market_minute=row['decision_valid_minute'],
                    active_wait=row['action']==0 and row['legal_add'],forced_wait=row['action']==0 and not row['legal_add'],
                    illegal_add=row['action']==1 and not row['legal_add'],p_add=diag.get('p_add')))
            for fill in row['fills']:
                if fill['reason']!='ENTRY' or not fill['committed']:continue
                eid=fill['entry_id']
                if eid in trades:raise ValueError('EXAM_DUPLICATE_REAL_ENTRY')
                trades[eid]=dict(contract=contract,account_owner=row['account_owner'],entry_id=eid,
                    side=fill['side'],entry_time=fill['time'],entry_price=float(fill['price']),
                    entry_official_day=row['trading_date'],entry_session=row['session'],
                    adverse=[],underwater=0,longest_underwater=0,underwater_run=0,first_adverse10=None)
                coverage[(contract,row['trading_date'],row['session'])]['entries']+=1
            if not row['valid_bar']:continue
            exiting={e['entry_id']:e for e in row['exits']}
            live={e['entry_id'] for e in row['active_entries']}|set(exiting)
            for eid in live:
                if eid not in trades:raise ValueError('EXAM_TRADE_ACCOUNT_JOIN_FAILED')
                trade=trades[eid];bar=row['bar'];side=trade['side'];sign=1 if side=='LONG' else -1
                quote=float(env._price(bar['low'] if sign==1 else bar['high'],side,row['time']))
                adverse=max(0.,-sign*(quote-trade['entry_price']));trade['adverse'].append(adverse)
                age=len(trade['adverse'])
                if adverse>=10-1e-10 and trade['first_adverse10'] is None:trade['first_adverse10']=age
                # Same-bar extremes carry native risk-first OHLC ambiguity.
                # Exit closes are never counted as a still-held close.
                if eid not in exiting:
                    close=float(env._price(bar['close'],side,row['time']))
                    wet=sign*(close-trade['entry_price'])<0
                    trade['underwater']+=int(wet);trade['underwater_run']=trade['underwater_run']+1 if wet else 0
                    trade['longest_underwater']=max(trade['longest_underwater'],trade['underwater_run'])
                if eid in exiting:trade['exit_official_day']=row['trading_date']
    finally:env.close()
    result=[]
    for eid,trade in trades.items():
        adverse=trade.pop('adverse');trade.pop('underwater_run');event=attrs.get(eid)
        closed=event is not None;reason=event['reason'] if closed else 'RIGHT_CENSORED';age=len(adverse)
        if closed and event['held_market_bars']!=age:raise ValueError('EXAM_HOLDING_MARKET_MINUTES_MISMATCH')
        trade.update(exit_reason=reason,exit_time=event['exit_time'] if closed else None,
            exit_price=float(event['exit_price']) if closed else None,held_market_minutes=age,
            observed_risk_minutes=age,mae_full=max(adverse,default=0.),mae_full_valid=closed,
            mae_observed_valid=bool(age),mae_5=max(adverse[:5],default=0.),mae_15=max(adverse[:15],default=0.),
            mae_5_valid=closed or age>=5,mae_15_valid=closed or age>=15,
            risk_price_basis='native_executable_adverse_quote; native_risk_first_OHLC',
            underwater_observed_valid=True,underwater_full_valid=closed,
            underwater_definition='native_executable_close_while_held_at_eob; settled_exit_bar_excluded',
            censored=not closed,same_official_day_tp=reason=='TAKE_PROFIT' and trade.get('exit_official_day')==trade['entry_official_day'],
            crossed_official_day_tp=reason=='TAKE_PROFIT' and trade.get('exit_official_day')!=trade['entry_official_day'],
            reward=rewards[eid],net_pnl=float(event['net_trade_pnl']) if closed else None,
            open_fee=float(event['open_fee']) if closed else float(state['entries'][eid]['open_fee']),
            close_fee=float(event['close_fee']) if closed else 0.,
            native_reward_adverse10_minute=first_native.get(eid,{}).get('effective_bar'),
            intrabar_ambiguous=bool(event and (event.get('intrabar_ambiguous',False) or event.get('ambiguous_tp_sl',False))))
        for h in (5,15,30,60):trade['tp_le_'+str(h)]=reason=='TAKE_PROFIT' and age<=h
        for h in (5,15):trade['adverse10_le_'+str(h)+'_then_sl30']=reason=='STOP_LOSS' and trade['first_adverse10'] is not None and trade['first_adverse10']<=h
        result.append(trade)
    n=len(result);closed_pnl=sum(Decimal(str(t['net_pnl'])) for t in result if t['net_pnl'] is not None)
    open_fees=sum(Decimal(e['open_fee']) for e in state['entries'].values())
    equity=state['equity_history'][max(0,len(initial['equity_history'])-1):]
    values=np.array([e['equity'] for e in equity],np.float64)
    if not len(values) or not np.isfinite(values).all():raise ValueError('EXAM_EQUITY_HISTORY_REQUIRED')
    pnl=Decimal(str(values[-1]))-Decimal(str(values[0]))
    reconciled=closed_pnl+Decimal(str(state['unrealized_pnl']))-open_fees
    if abs(pnl-reconciled)>Decimal('0.000001'):raise ValueError('EXAM_ACCOUNT_NET_PNL_RECONCILIATION_FAILED')
    illegal=sum(d['illegal_add'] for d in decisions)
    if illegal:raise ValueError('EXAM_ILLEGAL_ACTION_FOUND')
    probabilities=[d['p_add'] for d in decisions if d['legal_add'] and d['p_add'] is not None]
    summary=dict(contract=contract,entries=n,closed=sum(not t['censored'] for t in result),
        right_censored=sum(t['censored'] for t in result),main_roll=sum(t['exit_reason']=='MAIN_ROLL' for t in result),
        all_market_advances=sum(r['advanced'] for r in rows),valid_minutes=sum(r['valid_bar'] for r in rows),
        eligible_add_states=sum(d['legal_add'] for d in decisions),active_wait=sum(d['active_wait'] for d in decisions),
        forced_wait=sum(d['forced_wait'] for d in decisions),add_proposals=sum(d['action']==1 for d in decisions),
        entry_fills=n,illegal_add=illegal,mask_conflicts=illegal,
        tp=sum(t['exit_reason']=='TAKE_PROFIT' for t in result),sl30=sum(t['exit_reason']=='STOP_LOSS' for t in result),
        reward=total_reward,net_pnl=float(pnl),closed_net_pnl=float(closed_pnl),unrealized_pnl=float(state['unrealized_pnl']),
        open_position_open_fees=float(open_fees),fees=sum(t['open_fee']+t['close_fee'] for t in result),
        max_drawdown=float((np.maximum.accumulate(values)-values).max()),
        tp_rate=None if not n else sum(t['exit_reason']=='TAKE_PROFIT' for t in result)/n,
        sl_rate=None if not n else sum(t['exit_reason']=='STOP_LOSS' for t in result)/n,
        mae_15_valid_count=sum(t['mae_15_valid'] for t in result),
        mae_15_mean=None if not n else float(np.mean([t['mae_15'] for t in result if t['mae_15_valid']])) if any(t['mae_15_valid'] for t in result) else None,
        p_add_quantiles=None if not probabilities else {str(q):quantile(probabilities,q) for q in (0.,.1,.5,.9,1.)},
        market_blocks=len({(r['trading_date']) for r in rows if r['valid_bar']}),
        denominator='all_actual_ENTRY_fills_including_SL_roll_and_right_censored',
        account_reconciled=True,reward_reconciled=True,
        entry_evidence='NO_ENTRY_SKILL_EVIDENCE' if n==0 else 'ACTUAL_ENTRY_SUPPORT')
    for h in (5,15,30,60):
        summary['tp_le_'+str(h)]=sum(t['tp_le_'+str(h)] for t in result)
        summary['tp_le_'+str(h)+'_rate']=None if not n else summary['tp_le_'+str(h)]/n
    daily=[];previous=float(values[0]);previous_unreal=float(initial['unrealized_pnl']);previous_open_fees=0.
    day_equity=defaultdict(list)
    for e in equity:
        day=row_days.get(e['time'])
        if day is not None:day_equity[day].append(e)
    scored_days=sorted({r['trading_date'] for r in rows if r['valid_bar']})
    for day in scored_days:
        group=day_equity[day]
        if not group:raise ValueError('EXAM_OFFICIAL_DAY_EQUITY_MISSING')
        mark=float(group[-1]['equity']);unreal=float(group[-1]['unrealized_pnl'])
        ds=[previous,*[float(e['equity']) for e in group]];peak=ds[0];dd=0.
        for value in ds:peak=max(peak,value);dd=max(dd,peak-value)
        ts=[t for t in result if t['entry_official_day']==day]
        exits=[t for t in result if not t['censored'] and t['exit_official_day']==day]
        last_row=next(r for r in reversed(rows) if r['trading_date']==day and r['advanced'])
        outstanding=sum(float(e['open_fee']) for e in last_row['active_entries'])
        closed_day=sum(t['net_pnl'] for t in exits)
        account_delta=closed_day+unreal-previous_unreal-outstanding+previous_open_fees
        if abs(account_delta-(mark-previous))>1e-6:raise ValueError('EXAM_OFFICIAL_DAY_LEDGER_RECONCILIATION_FAILED')
        daily.append(dict(contract=contract,official_day=day,start_equity=previous,end_equity=mark,
            net_pnl=mark-previous,closed_net_pnl=closed_day,unrealized_pnl=unreal,
            unrealized_delta=unreal-previous_unreal,open_position_open_fees=outstanding,
            ledger_reconciled=True,reward=daily_reward[day],max_drawdown=dd,
            entries=len(ts),tp=sum(t['exit_reason']=='TAKE_PROFIT' for t in ts),
            sl30=sum(t['exit_reason']=='STOP_LOSS' for t in ts),
            tp_le_5=sum(t['tp_le_5'] for t in ts),mae_15_mean=None if not ts else sum(t['mae_15'] for t in ts)/len(ts),
            fees=sum(t['open_fee'] for t in ts)+sum(t['close_fee'] for t in exits),
            trade_count_basis='entry_cohort; cashflow_and_exit_fees_at_actual_official_day'))
        previous=mark;previous_unreal=unreal;previous_open_fees=outstanding
    if abs(sum(d['net_pnl'] for d in daily)-float(pnl))>1e-6:raise ValueError('EXAM_DAILY_PNL_RECONCILIATION_FAILED')
    coverage_rows=[dict(contract=k[0],official_day=k[1],session=k[2],**v) for k,v in sorted(coverage.items())]
    output.mkdir(parents=True)
    _table(output/'DECISIONS.csv',decisions,('contract','stepid','action','legal_add'))
    _table(output/'TRADES.csv',result,('contract','entry_id','exit_reason'))
    _table(output/'RISK.csv',[{k:v for k,v in t.items() if k not in ('reward','net_pnl','open_fee','close_fee')} for t in result],('entry_id',))
    _table(output/'DAILY.csv',daily);_table(output/'CONTRACT.csv',[summary]);_table(output/'EQUITY.csv',equity)
    _table(output/'COVERAGE.csv',coverage_rows)
    document=dict(summary=summary,trades=result,daily=daily,coverage=coverage_rows)
    write_json(output/'METRICS.json',document)
    return document

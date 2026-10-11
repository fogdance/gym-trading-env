"""TRAIN-only indexed candidates and certified Single3 semi-Markov optimization.

The original Env supplies first-bar execution, atomic financial settlement, and
all reference replays. Indexes only skip bars with no possible execution/event.
The optimizer uses rational gamma and exact float32 reward coefficients. It
never claims funds are irrelevant on arbitrarily losing account paths: a
value-preserving dominance proof first removes those paths.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from decimal import Decimal
from fractions import Fraction
import gzip
import math
import multiprocessing
from pathlib import Path
import resource
import time

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.rewards.single3_reward import Single3Reward, Single3RewardConfig
from .fg_replay import (SINGLE3_SCHEMA, check_files, digest, iter_sidecar, read_manifest,
                        signed, trace_manifest, write_json)
from .fg_single3 import (execution_provenance, load_env, make_output, required_tests, source_witness)
from .fg_single3_replay import require_e, trace_job

GAMMA = Fraction(332, 333)
OBJECTIVE = 'native_float32_feedback_rational_gamma_env_advance_v1'


class ThresholdIndex:
    """Range minimum tree: first value <= threshold, and range extrema."""
    def __init__(self, values):
        values = np.asarray(values, np.float64)
        self.n = len(values)
        self.size = 1 << (max(1, self.n) - 1).bit_length()
        self.tree = np.full(2*self.size, math.inf)
        self.tree[self.size:self.size+self.n] = values
        for i in range(self.size-1, 0, -1):
            self.tree[i] = min(self.tree[2*i], self.tree[2*i+1])

    def first(self, start, stop, threshold):
        def visit(node, lo, hi):
            if hi <= start or lo >= stop or self.tree[node] > threshold:
                return stop
            if hi-lo == 1:
                return lo
            mid=(lo+hi)//2
            left=visit(node*2,lo,mid)
            return left if left != stop else visit(node*2+1,mid,hi)
        return visit(1, 0, self.size)

    def minimum(self, start, stop):
        lo,hi=start+self.size,stop+self.size
        result=math.inf
        while lo<hi:
            if lo&1:result=min(result,self.tree[lo]);lo+=1
            if hi&1:hi-=1;result=min(result,self.tree[hi])
            lo//=2;hi//=2
        return result


def empty_at(env, cursor):
    """Fresh original account at a known cursor; immutable market is shared."""
    env._initialized=False
    env.completed_accounts=[]
    env.reset(seed=0)
    env.current_step=cursor
    env.episode_step_count=cursor
    env.chunk_count=cursor % env.chunk_steps if env.chunk_steps else 0
    env.current_price=Decimal(str(env._raw.iloc[cursor].Close))
    env.equity_history=[]
    env._sample_equity()
    return env._get_obs()


def funds_certificate(env):
    """Prove funding for every retained TP-only reachable account, not all paths."""
    periods=[env.terms.at(t) for t in env._raw.index]
    costs={(p['margin_rate'],p['open_fee'],p['close_fee'],p['close_today_fee'],
            p['spread'],p['slippage']) for p in periods}
    # The proof is deliberately conservative; unfamiliar schedules fail closed.
    if len(costs)!=1:
        raise ValueError('NOT_PROVEN: dominance funding proof requires constant TRAIN terms')
    rate,op,cl,ct,spread,slippage=next(iter(costs))
    if (min(op,cl,ct,spread,slippage)<0 or env.config.trading.intraday_single_position
            or env.config.trading.max_entries_per_day < len(env._raw)):
        raise ValueError('NOT_PROVEN: unhandled cashflow or position rule')
    m=env.terms.multiplier
    lo=Decimal(str(env._raw.Low.min()))-spread-slippage-env.terms.tick
    hi=Decimal(str(env._raw.High.max()))+spread+slippage+env.terms.tick
    bound=3*m*((hi-lo)+hi*rate)+3*(op+max(cl,ct))
    tp_net_min=3*m*5-3*(op+max(cl,ct))
    initial=Decimal(str(env.config.trading.initial_balance))
    if bound>=initial or tp_net_min<=0:
        raise ValueError(f'NOT_PROVEN: funds bound {bound} / initial {initial}, TP net {tp_net_min}')
    return dict(status='exact_proven',proof='nonpositive_interval_dominance_then_TP_only_funds_v1',
        arbitrary_losing_paths_funds_irrelevant=False,
        retained_paths_cash_lower_bound=str(initial),required_equity_upper_bound=str(bound),
        minimum_TP_net_cash_gain=str(tp_net_min),constant_terms=True,
        reasoning=['WAIT has zero reward and no cashflow; clocks are exogenous.',
          'Non-TP and censored intervals have only nonpositive feedback (alpha=lambda_capacity=0).',
          'Remove such intervals and nonpositive total TP intervals, keeping later decisions at the same cursor.',
          'Every retained TP increases cash; worst observed quote envelope keeps funds/margin nonbinding.',
          'The relaxed interval DP upper bound is attained by its original-Env TP-only account replay.'],
        scope='observed TRAIN feedback to real terminal or right-censored boundary')


class CandidateIndex:
    def __init__(self, env):
        self.env=env
        self.proof=funds_certificate(env)
        raw=env._raw
        self.n=len(raw)
        self.prefix=np.concatenate(([0],np.cumsum(env.valid_fill_clock)))
        self.liquid=np.flatnonzero(env.valid_fill_clock & (raw.Volume.to_numpy()>=3))
        deadlines=[v for v in (env.main_exit_at,env.terms.liquidate) if v is not None]
        self.global_at=min([int(raw.index.searchsorted(v)) for v in deadlines]+[self.n])
        self.global_reason=('MAIN_ROLL' if self.global_at<self.n and env.main_exit_at is not None
            and raw.index[self.global_at]>=env.main_exit_at else 'CONTRACT_EXPIRY')
        self.gap_at=self.n
        mapping=env.terms.calendar.get_indexer(raw.index)
        broken=np.flatnonzero(np.diff(mapping)!=1)
        if len(broken):self.gap_at=int(broken[0])+1
        self.flat_end=min(self.n-1,self.global_at,self.gap_at-1)
        self.stop=min(self.n,self.gap_at)
        self.indices={}
        self.quotes={}
        for side,sign in (('LONG',1),('SHORT',-1)):
            quotes={}
            for key,column,opening in (('open','Open',False),('adverse','Low' if sign==1 else 'High',False),
                    ('favorable','High' if sign==1 else 'Low',False),('close','Close',False)):
                times=env.bobs if key=='open' else raw.index
                quotes[key]=np.array([float(env._price(value,side,t,opening=opening))
                                      for value,t in zip(raw[column],times)])
            self.quotes[side]=quotes
            valid=env.valid_fill_clock
            liquid=valid & (raw.Volume.to_numpy()>=3)
            self.indices[side]=dict(
                sl=ThresholdIndex(np.where(valid,sign*quotes['adverse'],math.inf)),
                tp=ThresholdIndex(np.where(liquid,-sign*quotes['favorable'],math.inf)),
                early=ThresholdIndex(np.where(valid,sign*raw['Low' if sign==1 else 'High'].to_numpy(),math.inf)))

    def first_liquid(self,start):
        j=int(np.searchsorted(self.liquid,start))
        return int(self.liquid[j]) if j<len(self.liquid) and self.liquid[j]<self.stop else self.stop

    def candidate(self,t):
        env=self.env
        if t>=self.flat_end:
            return dict(cursor=t,eligible=False,reject_reason='INACTIVE_TRAIN_BOUNDARY')
        obs=empty_at(env,t)
        if not obs['action_mask'][1]:
            return dict(cursor=t,eligible=False,reject_reason='KNOWN_MASK')
        _,_,_,_,info=env.step(1)
        opening=next((f for f in env.fills if f['reason']=='ENTRY'),None)
        if opening is None:
            return dict(cursor=t,eligible=True,filled=False,reject_reason=info.get('reject_reason'),
                        exit_cursor=t+1,rewards={c:[] for c in ('C0','C1','C2','C3')})
        entry=dict(opening)
        side=entry['side'];sign=1 if side=='LONG' else -1
        price=Decimal(entry['price'])
        i=t+1;idx=self.indices[side]
        s=idx['sl'].first(i,self.stop,float(sign*(price-sign*30)))
        first_early=idx['early'].first(i,self.stop,sign*float(price)-10+1e-10)
        event=env.attributions[-1] if env.attributions else None
        if event is None:
            item=next(iter(env.entries.values()))
            pending=item['sl_pending']
            sl_exit=self.first_liquid(i+1 if pending else s) if pending or s<self.stop else self.stop
            tp_exit=idx['tp'].first(i+1,self.stop,-float(sign*(price+sign*5))-float(env.terms.tick))
            global_exit=self.first_liquid(max(i+1,self.global_at))
            u=min(sl_exit,tp_exit,global_exit)
            if u<self.stop:
                env.current_step=u
                row=env._raw.iloc[u];bob=env.bobs[u];end=env._raw.index[u]
                if s<u and not pending:
                    item.update(sl_pending=True,sl_trigger_time=str(env._raw.index[s]))
                if u>=self.global_at:
                    env.pending_reason=self.global_reason
                _,positions=env._positions()
                request=env._exit_request(item,positions[item['slot']],row,bob,end)
                if request is None:raise RuntimeError('Index proposed an exit without a real Env request')
                code,event=env.settle_exit(item['entry_id'],request[1],end,request[0])
                if event is None:raise RuntimeError(f'Indexed atomic close rejected: {code}')
        u=env.current_step if event is not None else self.stop-1
        end=u+1
        early=first_early if first_early<=u else None
        rewards={}
        for name in ('C0','C1','C2','C3'):
            # All skipped bars have no reward/event and only increment bars_seen.
            scorer=Single3Reward(Single3RewardConfig.for_candidate(name))
            sparse=[]
            for cursor in sorted({i,u,*([early] if early is not None else [])}):
                row=env._raw.iloc[cursor]
                if cursor>i:
                    scorer.scorer.active[entry['entry_id']].bars_seen=int(self.prefix[cursor]-self.prefix[i])
                bar=dict(bar_open=str(env.bobs[cursor]),bar_end=str(env._raw.index[cursor]),
                    **{k.lower():float(row[k]) for k in ('Open','High','Low','Close')},
                    valid_bar=bool(env.valid_fill_clock[cursor]),eligible_market_minute=False,
                    entries=[entry] if cursor==i else [],exits=[event] if event and cursor==u else [],
                    true_terminal=False)
                value=scorer.step(bar,env.terms.multiplier)['total']
                if value:sparse.append([cursor,float(np.float32(value))])
            rewards[name]=sparse
        ma=idx['early'].minimum(i,end)
        mae=max(0.,sign*float(price)-ma)
        return dict(cursor=t,eligible=True,filled=True,side=side,entry_cursor=i,
            entry_time=entry['time'],entry_price=entry['price'],exit_cursor=u if event else None,
            release_cursor=u if event else None,right_censored=event is None,
            observed_until_cursor=u,exit=event,effective_bars=int(self.prefix[end]-self.prefix[i]),
            natural_minutes=(env._raw.index[u]-env.bobs[i]).total_seconds()/60,
            first_adverse_cursor=early,first_adverse_effective_bar=int(self.prefix[early+1]-self.prefix[i]) if early is not None else None,
            MAE_points=mae,rewards=rewards)


def _down(x):return math.nextafter(x,-math.inf)
def _up(x):return math.nextafter(x,math.inf)

def _add(a,b):return (_down(a[0]+b[0]),_up(a[1]+b[1]))

def _mul(a,b):
    products=[x*y for x in a for y in b]
    return (_down(min(products)),_up(max(products)))


def optimize(candidates,end,candidate,gamma=GAMMA,*,max_states=2000000):
    """Certified interval comparisons, exact rational fallback, WAIT on ties."""
    if end+1>max_states:raise ValueError('NOT_PROVEN: exact search state budget exceeded')
    gamma=Fraction(gamma)
    if not 0<gamma<=1:raise ValueError('Unsupported discount')
    g=float(gamma)
    gb=(_down(g),_up(g)) if Fraction.from_float(g)!=gamma else (g,g)
    powers=[(1.,1.)]
    for _ in range(end):powers.append(_mul(powers[-1],gb))
    bounds=[(0.,0.)]*(end+1)
    first=[None]*(end+1)
    next_trade={}
    chosen={}
    exact_checks=ties=pruned=0
    by_cursor={r['cursor']:r for r in candidates if r.get('filled')}

    def chain(head):
        result=[]
        while head is not None:
            result.append(head);head=next_trade[head]
        return result

    def exact_difference(add,wait,origin):
        # Identical continuation after the first common trade cancels exactly.
        left=[add]+chain(first[by_cursor[add]['release_cursor']])
        right=chain(wait)
        rset=set(right)
        common=next((v for v in left if v in rset),None)
        events=Counter()
        for route,sign in ((left,1),(right,-1)):
            for cursor in route:
                if cursor==common:break
                for step,reward in by_cursor[cursor]['rewards'][candidate]:
                    events[step]+=sign*Fraction.from_float(reward)
        return sum((gamma**(step-origin-1)*value for step,value in events.items()),Fraction(0))

    for t in range(end-1,-1,-1):
        wait=_mul(gb,bounds[t+1]);best=wait;head=first[t+1]
        row=by_cursor.get(t)
        if row and row['exit'] and row['exit']['reason']=='TAKE_PROFIT' and row['release_cursor']<=end:
            local=(0.,0.)
            for step,reward in row['rewards'][candidate]:
                local=_add(local,_mul(powers[step-t-1],(reward,reward)))
            # Removing a nonpositive complete interval preserves every later decision.
            keep=local[1]>0
            if keep and local[0]<=0:
                keep=sum((gamma**(s-t-1)*Fraction.from_float(v) for s,v in row['rewards'][candidate]),Fraction(0))>0
            if keep:
                u=row['release_cursor']
                option=_add(local,_mul(powers[u-t],bounds[u]))
                take=option[0]>wait[1]
                if not take and option[1]>=wait[0]:
                    exact_checks+=1
                    diff=exact_difference(t,head,t)
                    take=diff>0;ties+=int(diff==0)
                if take:
                    best=option;head=t
                    next_trade[t]=first[u];chosen[t]=row
            else:pruned+=1
        else:
            pruned+=int(row is not None)
        bounds[t]=best;first[t]=head
    schedule=chain(first[0])
    if any(by_cursor[t]['exit']['reason']!='TAKE_PROFIT' for t in schedule):
        raise RuntimeError('Dominance proof failed to retain only cash-positive TP trades')
    return dict(oracle_optimality_status='exact_proven',candidate=candidate,
        gamma=float(gamma),gamma_rational=[gamma.numerator,gamma.denominator],
        objective=OBJECTIVE,tie_break='WAIT',add_cursors=schedule,
        value_interval=list(bounds[0]),search_states=end+1,exact_comparisons=exact_checks,
        exact_ties=ties,dominance_pruned=pruned,selected_trades=len(schedule),
        selected_cash_gain=sum(float(by_cursor[t]['exit']['net_trade_pnl']) for t in schedule))


def reference_candidate(env,t):
    empty_at(env,t)
    rewards=[]
    _,r,_,_,_=env.step(1)
    if r:rewards.append([env.current_step,float(np.float32(r))])
    if not env.fills:
        return dict(filled=False,reject_reason=getattr(env,'reject_reason',None),rewards=rewards)
    while env.entries and not env.terminated and env.status not in ('DATA_END','DATA_GAP'):
        if env.truncated:env.reset()
        old=env.current_step
        _,r,_,_,_=env.step(0)
        if r:rewards.append([env.current_step,float(np.float32(r))])
        if env.current_step==old:break
    return dict(filled=True,exit=env.attributions[-1] if env.attributions else None,
                exit_cursor=env.current_step if env.attributions else None,rewards=rewards)


def verify_candidates(index,rows,*,seed=202,count=128):
    filled=[r for r in rows if r.get('filled')]
    eligible=[r for r in rows if r.get('eligible')]
    rng=np.random.default_rng(seed)
    sampled=set(int(v) for v in rng.choice([r['cursor'] for r in eligible],min(count,len(eligible)),replace=False))
    # Explicit event classes plus longest observed paths and chunk boundaries.
    groups={}
    for row in filled:
        event=row.get('exit') or {}
        key=(row['side'],event.get('reason','CENSORED'),bool(event.get('ambiguous_tp_sl')),
             bool(event.get('same_entry_bar')),bool(event.get('sl_pending_retry')),
             bool(event.get('gap_slippage_points')),row['first_adverse_effective_bar'] is not None)
        groups.setdefault(key,row['cursor'])
    sampled.update(groups.values())
    sampled.update(r['cursor'] for r in sorted(filled,key=lambda r:r['effective_bars'],reverse=True)[:3])
    sampled.update(r['cursor'] for r in rows if r.get('eligible') and r['cursor']%1024 in (1023,0))
    expected={r['cursor']:r for r in rows}
    for t in sorted(sampled):
        row=expected[t];ref=reference_candidate(index.env,t)
        if row.get('filled',False)!=ref['filled'] or row['rewards']['C2']!=ref['rewards']:
            raise RuntimeError(f'Indexed/reference reward mismatch at {index.env.contract}/{t}: {row} / {ref}')
        if ref['filled']:
            if row['exit_cursor']!=ref['exit_cursor']:
                raise RuntimeError(f'Indexed/reference exit cursor mismatch at {t}')
            a,b=row['exit'],ref['exit']
            if bool(a)!=bool(b):raise RuntimeError('Censor mismatch')
            if a:
                for key in a:
                    if key=='entry_id':continue
                    if a[key]!=b[key]:raise RuntimeError(f'Indexed/reference exit mismatch {t}/{key}: {a[key]} / {b[key]}')
    return dict(status='PASS',seed=seed,sampled_cursors=sorted(sampled),event_class_representatives=len(groups),
                final_H333_Driver_verification='SEPARATE_REQUIRED_GATE',all_candidates_exhaustively_replayed=False)


def require_r(path):
    path=Path(path).resolve();m=read_manifest(path/'manifest.json')
    if m.get('stage')!='R' or m.get('status')!='PASS' or m.get('selected_training_reward') is not None:
        raise ValueError('CPU Oracle requires signed R comparison Gate without a selected model Reward')
    check_files({str(path/p):h for p,h in m['files'].items()});check_files(m['frozen_inputs'])
    return m


def _oracle_job(job):
    root=Path(job['directory'])
    if job.get('resume') and (root/'manifest.json').exists():
        m=read_manifest(root/'manifest.json')
        check_files({str(root/p):h for p,h in m['files'].items()})
        if m['identity']!=job['identity']:raise ValueError('Oracle resume identity changed')
        return m
    if root.exists():raise FileExistsError('Incomplete Oracle attempt retained; use a new output')
    root.mkdir(parents=True)
    check_files(job['frozen_files'])
    began=time.monotonic()
    env=load_env(job['row'],job['frozen_files']);index=CandidateIndex(env)
    rows=[];benchmark_begin=time.monotonic()
    for t in range(index.flat_end):
        rows.append(index.candidate(t))
        if t+1==min(2048,index.flat_end):
            write_json(root/'performance.json',dict(candidates=t+1,seconds=time.monotonic()-benchmark_begin,
                projected_candidate_seconds=(time.monotonic()-benchmark_begin)*index.flat_end/(t+1),
                complexity='O(N log N) index plus O(N) original atomic settlements',
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        if t and t%8192==0:print(f"Oracle candidates {env.contract}: {t}/{index.flat_end}",flush=True)
    with gzip.open(root/'candidates.jsonl.gz','wt') as f:
        import json
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    verified=verify_candidates(index,rows)
    plans={};diagnostics={};traces=[]
    by_cursor={r['cursor']:r for r in rows}
    for name in ('C0','C1','C2','C3'):
        plan=optimize(rows,index.flat_end,name,max_states=job['max_states'])
        plan.update(contract=env.contract,training_only=True,identity=job['identity'],
                    candidates_sha256=digest(root/'candidates.jsonl.gz'),funds_certificate=index.proof)
        path=root/f'{name}_H333.json';write_json(path,signed(plan));plans[name]=str(path)
        diagnostic=optimize(rows,index.flat_end,name,Fraction(1),max_states=job['max_states'])
        diagnostic.update(contract=env.contract,diagnostic_only=True,training_only=True)
        write_json(root/f'{name}_gamma1.json',signed(diagnostic));diagnostics[name]=diagnostic
        spec=dict(id='P4',kind='oracle_schedule',candidate=name,seed_components=[0,job['index'],5],
                  schedule_path=str(path),schedule_sha256=digest(path),hindsight_behavior=True)
        trace=trace_job(dict(directory=str(root/'traces'/name),row=job['candidate_rows'][name],spec=spec,
            identity=job['identity'],dreamer_root=job['dreamer_root'],frozen_files=job['frozen_files']))
        actual=[];cash=0.;weight=1.;total=0.
        expected_feedback={s:v for t in plan['add_cursors'] for s,v in by_cursor[t]['rewards'][name]}
        for row in iter_sidecar(trace['directory']):
            if row['advanced']:
                if float(np.float32(row['official_reward']))!=expected_feedback.get(row['cursor'],0.):
                    raise RuntimeError('Oracle candidate and full Driver sparse feedback differ')
                total+=weight*float(np.float32(row['official_reward']));weight*=332/333
            for f in row['fills']:
                if f['reason']=='ENTRY':actual.append(row['cursor']-1)
            for event in row['exits']:
                if event['reason']!='TAKE_PROFIT':raise RuntimeError('Exact Oracle retained a non-TP trade')
                expected=by_cursor[actual[-1]]['exit']
                for key in expected:
                    if key!='entry_id' and expected[key]!=event[key]:raise RuntimeError(f'Oracle actual trade mismatch: {key}')
                cash+=float(event['net_trade_pnl'])
        lo,hi=plan['value_interval']
        # A separately rounded float64 displayed total is diagnostic; its tiny
        # aggregation error is not used for action comparisons or the proof.
        if actual!=plan['add_cursors'] or cash!=plan['selected_cash_gain']:
            raise RuntimeError('Exact plan was not attained by the real account')
        trace['objective_replay']=dict(status='PASS',actual_discounted_float64=total,
            certified_objective_interval=[lo,hi],actual_add_cursors=actual,
            every_trade_matched=True,every_feedback_matched=True,actual_cash_gain=cash)
        traces.append(trace)
    env.close();check_files(job['frozen_files'])
    receipt=dict(schema='FG_SINGLE3_CPU_V1',stage='O_CONTRACT',status='PASS',
        oracle_optimality_status='exact_proven',contract=job['row']['contract'],identity=job['identity'],
        candidates=len(rows),flat_end_cursor=index.flat_end,funds_certificate=index.proof,
        candidate_verification=verified,H333_schedules=plans,gamma1_diagnostics=diagnostics,
        collection=traces,elapsed_seconds=time.monotonic()-began,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        files={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()})
    check_files({str(root/p):h for p,h in receipt['files'].items()})
    write_json(root/'manifest.json',signed(receipt))
    print(f"Oracle {receipt['contract']}: exact_proven, {len(rows)} candidates, full Driver PASS",flush=True)
    return receipt


def oracle_study(config,output,env_receipt,reward_receipt,*,workers=4,contracts=None,resume=False):
    if not 1<=workers<=min(config['oracle_workers'],config['max_workers']):raise ValueError('Oracle worker budget exceeded')
    tests=required_tests('O');inputs,e=require_e(env_receipt);r=require_r(reward_receipt)
    rows=inputs['views']['C2']['rows'];selected=set(contracts or [v['contract'] for v in rows])
    if not selected or selected-{v['contract'] for v in rows}:raise ValueError('Oracle TRAIN boundary violation')
    output=Path(output).resolve()
    if not resume:output=make_output(output,config)
    code=execution_provenance(config,'O')
    if resume:
        metadata=read_manifest(output/'inputs.json')
        if metadata['provenance']['code_files']!=code['code_files']:raise ValueError('Oracle resume code changed')
        snapshots=metadata['source_snapshots']
    else:snapshots=source_witness(output,code)
    live={**inputs['frozen_files'],**code['code_files']}
    for path in (Path(env_receipt)/'manifest.json',Path(reward_receipt)/'manifest.json'):live[str(path.resolve())]=digest(path)
    identity=dict(schema='FG_SINGLE3_ORACLE_V1',objective=OBJECTIVE,training_hash=inputs['original_content_hash'],
                  code_hash=fingerprint(code['code_files']),selected_contracts=sorted(selected))
    write_json(output/'inputs.json',signed(dict(config=config,identity=identity,provenance=code,
        source_snapshots=snapshots,frozen_files=live,training_views=inputs['views'])))
    jobs=[]
    for i,row in enumerate(rows):
        if row['contract'] in selected:
            jobs.append(dict(directory=str(output/'contracts'/row['contract']),row=row,index=i,
                candidate_rows={c:next(v for v in inputs['views'][c]['rows'] if v['contract']==row['contract']) for c in config['candidates']},
                identity=identity,dreamer_root=config['dreamer_root'],frozen_files=live,
                max_states=config['max_account_states'],resume=resume))
    results=[]
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(_oracle_job,job) for job in jobs]):results.append(future.result())
        receipt=dict(schema='FG_SINGLE3_CPU_V1',stage='O',status='PASS',oracle_optimality_status='exact_proven',
            identity=identity,full_training_fold=len(selected)==len(rows),tests=tests,contracts=sorted(results,key=lambda v:v['contract']),
            provenance=code,source_snapshots=snapshots,
            frozen_inputs={p:h for p,h in live.items() if p not in code['code_files']},
            files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
        write_json(output/'manifest.pending.json',signed(receipt));check_files(live)
        check_files({str(output/p):h for p,h in receipt['files'].items()})
        (output/'manifest.pending.json').replace(output/'manifest.json')
        return receipt
    except BaseException as error:
        write_json(output/'failure.json',signed(dict(stage='O',status='FAIL',oracle_optimality_status='NOT_PROVEN',
            error_type=type(error).__name__,error=str(error))))
        raise

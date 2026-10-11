"""CPU event credit and fair ADD/WAIT original-account counterfactuals."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import copy
import gzip
import json
import multiprocessing
from pathlib import Path
import resource
import time

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import check_files,digest,execute_observed,iter_sidecar,read_manifest,signed,write_json
from .fg_single3 import execution_provenance,load_env,make_output,required_tests,source_witness
from .fg_single3_dataset import validate_dataset
from .fg_single3_coverage import require_coverage
from .fg_replay_coverage import distribution,table

HORIZONS=(333,1000,3000,None)


def gammas():return {str(h) if h else 'gamma1':1-1/h if h else 1. for h in HORIZONS}


def discounted_feedback(feedback,decision_cursor):
    if any(step<=decision_cursor for step,_ in feedback):raise ValueError('Feedback must follow the decision')
    return {h:sum(g**(step-decision_cursor-1)*r for step,r in feedback) for h,g in gammas().items()}


def fork_account(template,state):
    """Original account restore; only immutable features/market/terms are shared."""
    fork=copy.copy(template)
    fork.config=copy.deepcopy(template.config)
    fork._np_random=copy.deepcopy(template.np_random)
    fork.import_state(state)
    if fork.export_state()!=state:raise ValueError('Fork changed the complete starting account')
    return fork


def follow_p1(template,state,first_action):
    env=fork_account(template,state)
    if env.net_lots or not env._published_mask[1] or env.status!='ACTIVE':
        raise ValueError('Fair fork requires a genuinely legal empty observed state')
    initial_hash=fingerprint(env.export_state())
    initial_equity=float(env._calculate_equity())
    feedback=[];counts=Counter();old_fills=len(env.fills);old_exits=len(env.attributions)
    before=env.current_step;first=True
    while not env.terminated and env.status not in ('DATA_END','DATA_GAP'):
        if env.truncated:env.reset()
        action=first_action if first else int(env._get_obs()['action_mask'][1])
        first=False;cursor=env.current_step
        _,reward,_,_,info=env.step(action)
        if env.current_step>cursor:
            counts['advanced']+=1
            if reward:feedback.append([env.current_step,float(np.float32(reward))])
        if info.get('reject_reason'):counts['ADD_REJECTED']+=1
        for event in env.attributions[old_exits:]:counts[event['reason']]+=1
        old_exits=len(env.attributions)
        if env.current_step==cursor:break
    fills=env.fills[old_fills:]
    result=dict(first_action=first_action,starting_account_hash=initial_hash,
        decision_cursor=before,feedback=feedback,returns=discounted_feedback(feedback,before),
        subsequent_policy='frozen_causal_P1',counts=dict(counts),
        new_entry_fills=sum(f['reason']=='ENTRY' for f in fills),
        first_new_entry_time=next((f['time'] for f in fills if f['reason']=='ENTRY'),None),
        final_cursor=env.current_step,terminal_status=env.status,open_lots=float(env.net_lots),
        right_censored=bool(env.entries),
        account_equity_delta=float(env._calculate_equity())-initial_equity,
        final_state_hash=fingerprint(env.export_state()))
    env.close();return result


def fair_pair(env,state):
    initial=fingerprint(state)
    add=follow_p1(env,state,1);wait=follow_p1(env,state,0)
    if add['starting_account_hash']!=initial or wait['starting_account_hash']!=initial:
        raise ValueError('ADD and WAIT did not start from exactly the same financial/reward state')
    return dict(snapshot_hash=initial,cursor=state['current_step'],time=state['cursor_time'],
        add=add,wait=wait,delta={h:add['returns'][h]-wait['returns'][h] for h in gammas()},
        definition='one different first action, then identical causal P1 through real terminal/TRAIN boundary')


def _horizon_job(job):
    began=time.monotonic();env=load_env(job['row'],job['frozen_files'])
    selected=set(job['cursors']);snapshots={};previous=None
    for row in iter_sidecar(job['source']):
        incoming=None if previous is None or previous['is_last'] else previous['action']
        _,actual=execute_observed(env,incoming,seed=0 if previous is None else None)
        for key,value in actual.items():
            if value!=row[key]:raise ValueError(f'Counterfactual prefix changed actual source {row["ordinal"]}/{key}')
        if row['cursor'] in selected and row['decision_executed'] and row['legal_add']:
            snapshots[row['cursor']]=env.export_state()
        previous=row
        if len(snapshots)==len(selected):break
    if set(snapshots)!=selected:raise ValueError('Missing fair initial account states')
    results=[]
    for cursor in sorted(selected):
        result=fair_pair(env,snapshots[cursor]);result.update(candidate=job['candidate'],contract=env.contract,
            source_trace_hash=job['source_trace_hash'])
        results.append(result)
    check_files(job['frozen_files'])
    receipt=dict(status='PASS',candidate=job['candidate'],contract=env.contract,pairs=results,
        selected_cursors=sorted(selected),selection='seed202 uniform known legal-empty P1 observations; same cursors across Rewards',
        elapsed_seconds=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    env.close()
    path=Path(job['directory']);path.mkdir(parents=True,exist_ok=False)
    with gzip.open(path/'pairs.json.gz','wt') as f:json.dump(receipt,f,separators=(',',':'))
    print(f"Horizon fair forks {receipt['candidate']} {receipt['contract']}: {len(results)} pairs",flush=True)
    return dict(directory=str(path),receipt=receipt)


def _event_analysis(trades):
    strata={}
    for candidate in ('C0','C1','C2','C3'):
        selected=[t for t in trades if t['candidate']==candidate]
        for policy in ('P1','P2a','P2b','P3','P4'):
            selected_policy=[t for t in selected if t['policy_id']==candidate+'_'+policy]
            for reason in ('TAKE_PROFIT','STOP_LOSS','MAIN_ROLL','CONTRACT_EXPIRY','MARGIN_CALL','CENSORED'):
                events=[t for t in selected_policy if (t['exit_time'] is not None and t['exit_reason']==reason
                    if reason!='CENSORED' else t['right_censored'])]
                if not events:continue
                name=candidate+'_'+policy+'_'+reason
                strides=[t['exit_cursor']-t['decision_cursor'] for t in events if t.get('exit_cursor') is not None]
                values={h:[] for h in gammas()}
                for event in events:
                    for h,value in discounted_feedback(event['reward_feedback'],event['decision_cursor']).items():values[h].append(value)
                strata[name]=dict(count=len(events),effective_bars=distribution([t['observed_bars'] for t in events]),
                    natural_minutes=distribution([t['duration_minutes'] for t in events]),
                    env_feedback_steps=distribution(strides),
                    cross_env_chunk=sum(t['cross_env_chunk'] for t in events),
                    cross_replay_chunk=sum(t['cross_replay_chunk'] for t in events),
                    mean_local_discounted_return={h:float(np.mean(v)) for h,v in values.items()},
                    mean_direct_exit_discount={h:float(np.mean([g**(k-1) for k in strides])) if strides else None for h,g in gammas().items()})
    return strata


def _report(dataset,coverage,hresults,event_stats,root):
    pairs=[p for result in hresults for p in result['receipt']['pairs']]
    lines=['# FG Single3 Horizon 与长期信用分配 CPU审计','',
        f"H-Gate PASS；{len(pairs)}个同账户ADD/WAIT对照，四候选、18个TRAIN合约，每合约每候选8个冻结seed202快照。",
        f'数据manifest SHA256：`{digest(Path(dataset)/"manifest.json")}`；覆盖manifest SHA256：`{digest(Path(coverage)/"manifest.json")}`。','',
        '每对起始完整财务/Reward/当前观察快照一致；只改变第一步动作，此后两个分支都执行同一冻结因果P1，直到真实终局或TRAIN右删失。Δ是该有限后续政策的反事实回报差，不是最优Q，不是Critic预测误差。未创建Agent，没有梯度更新。','',
        '## 折扣定义与真实退出时间','',
        '第一步真实推进反馈指数0。γ_H=1−1/H；metadata推进行计一个Env步，chunk重复reset行不计推进。自然分钟只计算TP曲线；有效bar只计算early10窗口。所有统计使用已发布的float32整步反馈。','']
    lines+=table(['candidate_policy_exit','count','自然分钟','有效bar','Env反馈步','跨Envchunk','跨Replaychunk','H333/1000/3000/γ1局部回报均值','真实退出直接权重'],[(k,v['count'],v['natural_minutes'],v['effective_bars'],v['env_feedback_steps'],v['cross_env_chunk'],v['cross_replay_chunk'],v['mean_local_discounted_return'],v['mean_direct_exit_discount']) for k,v in sorted(event_stats.items())])
    lines+=['','## 同状态第一步ADD/WAIT排序','']
    groups=[]
    for candidate in ('C0','C1','C2','C3'):
        subset=[p for p in pairs if p['candidate']==candidate]
        for h in gammas():
            values=np.array([p['delta'][h] for p in subset]);reference=np.array([p['delta']['333'] for p in subset])
            groups.append((candidate,h,len(subset),float(values.mean()),float(np.median(values)),int((values>0).sum()),int((values<0).sum()),int((values==0).sum()),int((np.sign(values)!=np.sign(reference)).sum())))
    lines+=table(['candidate','H','pair count','mean Δ','median Δ','ADD较好','WAIT较好','平分','与H333排序不同'],groups)
    lines+=['','排序差异来自真实事件延迟和目标折扣变化，不能推断模型概率或价值校准已经出错。γ1还包含更晚的新入场机会，不能将总分较大解释为某H更适合训练。','',
        '## 原Dreamer代码信用路径','',
        '`dreamerv3/agent.py`的真实代码：cont目标为(1−is_terminal)，contdisc=True时乘γ；imag_loss的显式disc=1，预测continuation已携带γ；repl_loss使用显式γ与真实terminal，λ-return从rew[:,1:]开始、最后一个值bootstrap。第一次反馈指数与本报告Env步目标一致。','',
        'CHUNK_END为is_last=True、is_terminal=False：lambda_return在该边界停止λ轨迹传播，仍保留γ×bootstrap；不能将它说成terminal零价值或完全没有长期信用。RSSM的重复is_first与财务不重置是另一层边界。','',
        '网络imag_length=15不是333步显式展开；更长信用由value/bootstrap承担。batch_length64+context1不提供完整长期交易结果，每个抽样序列起点会被原Replay标is_first，Env1024也会重置RSSM观察流；财务持仓/SL pending/首次逆行状态由真实Env保留，但网络隐状态有截断。position_state给当前仓位和年龄，不能替代已经验证的长信用模型表现。','',
        'H333保持第一组基线。改变H会同时改变continuation训练目标与Actor/Critic价值目标，不能仅重算报告分数就称为模型换H训练。',
        '', '## 限制与复现','',
        '每合约8快照是固定工程抽样，不是所有空仓状态全集；覆盖极端持仓年龄的证据来自完整Replay事件统计。ADD/WAIT后冻结P1不是预见未来Oracle；同状态最优动作仍未被因果模型学会。未计算模型预测MAE、Critic误差或部署收益。',
        f'运行目录：`{root}`；每个pair保留源trace哈希、完整快照哈希、两个分支反馈和最终账户哈希，输入/源码/产物哈希在manifest。验证/封存Test价格未读取。']
    return '\n'.join(lines)+'\n'


def horizon_audit(config,dataset,coverage,output,*,workers=12,snapshots_per_contract=8):
    if not 1<=workers<=config['max_workers'] or snapshots_per_contract!=8:raise ValueError('H baseline resource/snapshot budget changed')
    tests=required_tests('H');data=validate_dataset(dataset);cov=require_coverage(dataset,coverage)
    if not data['full_training_fold']:raise ValueError('H requires full TRAIN dataset')
    root=Path(dataset).resolve();inputs=read_manifest(root/'inputs.json');output=make_output(output,config)
    code=execution_provenance(config,'H')
    for p in (Path(__file__).with_name('fg_single3_coverage.py'),Path(config['dreamer_root'])/'dreamerv3/agent.py',Path(config['dreamer_root'])/'dreamerv3/configs.yaml'):
        code['code_files'][str(p)]=digest(p)
    snapshots=source_witness(output,code)
    frozen={**inputs['frozen_inputs'],**code['code_files'],str(root/'manifest.json'):digest(root/'manifest.json'),
        str(Path(coverage).resolve()/'manifest.json'):digest(Path(coverage)/'manifest.json')}
    jobs=[]
    for index,base_row in enumerate(inputs['training_views']['C2']['rows']):
        contract=base_row['contract']
        record=next(r for r in data['traces'] if (r['candidate'],r['policy_id'],r['contract'])==('C2','P1',contract))
        eligible=sorted({r['cursor'] for r in iter_sidecar(root/record['directory']) if r['decision_executed'] and r['legal_add'] and r['lots_after']==0})
        if len(eligible)<8:raise ValueError('Insufficient real legal-empty snapshots')
        rng=np.random.default_rng(np.random.SeedSequence([202,index]))
        chosen=sorted(map(int,rng.choice(eligible,8,replace=False)))
        for candidate in ('C0','C1','C2','C3'):
            source=next(r for r in data['traces'] if (r['candidate'],r['policy_id'],r['contract'])==(candidate,'P1',contract))
            row=next(r for r in inputs['training_views'][candidate]['rows'] if r['contract']==contract)
            jobs.append(dict(directory=str(output/'forks'/candidate/contract),row=row,candidate=candidate,
                source=str(root/source['directory']),source_trace_hash=source['replay_trace_hash'],
                frozen_files=frozen,cursors=chosen))
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(_horizon_job,job) for job in jobs]):results.append(future.result())
    trades=json.loads((Path(coverage)/'trade_diagnostics.json').read_text());stats=_event_analysis(trades)
    write_json(output/'event_discount.json',signed(stats))
    validate_dataset(root);require_coverage(root,coverage);check_files(frozen)
    (output/'horizon_analysis.md').write_text(_report(root,coverage,results,stats,output))
    receipt=signed(dict(schema='FG_SINGLE3_CPU_V1',stage='H',status='PASS',tests=tests,
        dataset_manifest_sha256=digest(root/'manifest.json'),coverage_manifest_sha256=digest(Path(coverage)/'manifest.json'),
        provenance=code,source_snapshots=snapshots,horizons=gammas(),snapshots_per_contract=8,
        pairs=sum(len(r['receipt']['pairs']) for r in results),counterfactual_policy='first ADD/WAIT then frozen causal P1',
        gradient_updates=0,critic_prediction_error='NOT_MEASURED',
        frozen_inputs={p:h for p,h in frozen.items() if p not in code['code_files']},
        forks=[dict(directory=r['directory'],candidate=r['receipt']['candidate'],contract=r['receipt']['contract'],
            selected_cursors=r['receipt']['selected_cursors'],elapsed_seconds=r['receipt']['elapsed_seconds'],
            peak_rss_bytes=r['receipt']['peak_rss_bytes']) for r in results],
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))
    write_json(output/'manifest.pending.json',receipt)
    try:
        check_files(frozen);check_files({str(output/p):h for p,h in receipt['files'].items()})
        (output/'manifest.pending.json').replace(output/'manifest.json')
    except BaseException:
        (output/'horizon_analysis.md').replace(output/'horizon_analysis.failed.md')
        raise
    return receipt


def require_h(dataset,coverage,path):
    root=Path(path).resolve();m=read_manifest(root/'manifest.json')
    if (m.get('stage')!='H' or m.get('status')!='PASS' or m.get('pairs')!=576
            or m['dataset_manifest_sha256']!=digest(Path(dataset)/'manifest.json')
            or m['coverage_manifest_sha256']!=digest(Path(coverage)/'manifest.json')):
        raise ValueError('Full-fold fair-counterfactual H Gate mismatch')
    check_files(m['frozen_inputs']);check_files({str(root/p):h for p,h in m['files'].items()})
    return m

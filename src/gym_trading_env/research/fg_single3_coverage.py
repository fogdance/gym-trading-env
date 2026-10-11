"""Single3 event hooks over the original full-corpus and native Uniform audit."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import math
import resource

import numpy as np

from . import fg_replay_coverage as base
from .fg_replay import SINGLE3_SCHEMA,cpu_runtime,check_files,digest,read_manifest,signed,write_json
from .fg_single3 import execution_provenance,make_output,required_tests,source_witness
from .fg_single3_dataset import validate_dataset

COLUMNS=base.COLUMNS+('sl','other_exit','sl_pending','negative_reward','tp_le5','tp_5_15','tp_15_60','tp_60_480','tp_ge480')
INDEX={name:i for i,name in enumerate(COLUMNS)}


def group(record):return record['candidate']+'_'+record['policy_id']


def row_labels(row):
    labels=np.zeros(len(COLUMNS),np.uint16)
    labels[:len(base.COLUMNS)]=base.row_labels(row)
    labels[INDEX['full_wait']]=int(row['decision_executed'] and row['action']==0 and row['lots_after']==3)
    labels[INDEX['tp']]=sum(e['reason']=='TAKE_PROFIT' for e in row['exits'])
    labels[INDEX['sl']]=sum(e['reason']=='STOP_LOSS' for e in row['exits'])
    labels[INDEX['other_exit']]=sum(e['reason'] not in ('TAKE_PROFIT','STOP_LOSS') for e in row['exits'])
    labels[INDEX['sl_pending']]=sum(bool(e.get('sl_pending')) for e in row['active_entries'])
    labels[INDEX['negative_reward']]=int(row['official_reward']<0)
    for e in row['exits']:
        if e['reason']=='TAKE_PROFIT':
            duration=base.minute_age(e['exit_time'],e['entry_time'])
            key=('tp_le5' if duration<=5 else 'tp_5_15' if duration<=15 else
                 'tp_15_60' if duration<=60 else 'tp_60_480' if duration<480 else 'tp_ge480')
            labels[INDEX[key]]+=1
    return labels


def row_hook(row,opened,active,record):
    if row['lots_before'] not in (0,3) or row['lots_after'] not in (0,3) or len(row['active_entries'])>1:
        raise ValueError('Every canonical account row must have atomic 0/3 lots and at most one batch')
    for fill in row['fills']:
        if fill['reason']=='ENTRY':
            item=opened[fill['entry_id']]
            item.update(candidate=record['candidate'],decision_cursor=row['cursor']-1,
                        reward_feedback=[],pending_rows=0,first_adverse_effective_bar=None)
    for event in row['reward_events']:
        item=opened[event['entry_id']]
        if event['reason']=='FIRST_ADVERSE10':
            item['first_adverse_effective_bar']=event['effective_bar']
            item['first_adverse_cursor']=row['cursor']
        else:item['actual_exit']=event
    for item in row['active_entries']:
        if item.get('sl_pending'):opened[item['entry_id']]['pending_rows']+=int(row['advanced'])
    if row['advanced'] and row['official_reward']:
        ids={e['entry_id'] for e in row['reward_events']}
        if len(ids)!=1:raise ValueError('Single3 nonzero feedback has no unique real entry identity')
        item=opened[next(iter(ids))]
        item['reward_feedback'].append([row['cursor'],float(np.float32(row['official_reward']))])
    for event in row['exits']:
        item=opened[event['entry_id']]
        item.update(exit_cursor=row['cursor'],net_trade_pnl=float(event['net_trade_pnl']),
                    open_fee=float(event['open_fee']),close_fee=float(event['close_fee']),actual_exit=event)


def finalize_trade(item,record):
    if item['exit_reason']=='STOP_LOSS':item['status']='stop_loss'
    item['hindsight_behavior']=record['policy_id']=='P4'
    item['reward_params_hash']=record['reward_params_hash']
    item['right_censored']=item['exit_time'] is None


def require_coverage(dataset,path):
    dataset=Path(dataset).resolve();path=Path(path).resolve()
    m=read_manifest(path/'manifest.json')
    if (m.get('stage')!='D_AUDIT' or m.get('status')!='PASS' or not m.get('full_training_fold')
            or m['dataset_manifest_sha256']!=digest(dataset/'manifest.json') or m['batches']!=1000):
        raise ValueError('Full-fold 1000-batch signed Single3 D Gate required')
    check_files({str(path/p):h for p,h in m['files'].items()});check_files(m['frozen_inputs'])
    return m


def _coverage_lines(dataset,corpus,samples,root):
    totals=Counter()
    for metrics in corpus['summaries'].values():totals.update(metrics)
    lines=['# FG Single3 Replay R0–R2 完整覆盖审计','',
        f'数据：`{Path(root).resolve()}`；manifest SHA256：`{digest(Path(root)/"manifest.json")}`。','',
        'D-Gate工程数据正确性PASS；模型学习充分性NOT_ESTABLISHED。原Driver、Replay、reconcile与Uniform抽样全部复用；没有创建Agent或GPU训练。','',
        f"完整账户{len(dataset['traces'])}条，Replay行{totals['records']}，推进交互{totals['advanced']}，有效起点{totals['sequence_starts']}。",
        f"实际访问唯一(contract,time)有效分钟{corpus['unique_advanced_minutes']}；多候选/策略模拟是相关账户反事实，不能作为独立市场样本。",
        f"量化约束：非法ADD={totals['illegal_add']}，6手决策={totals['lots6']}；所有账户0/3手，每个训练池只有一个Reward；P0单列，未混入训练池。",
        f"累计账户恢复{corpus['account_restores']}；trace文件字节{corpus['trace_bytes']}（复用源与硬链接视图不视为新增市场证据）。",
        '', '## 分策略/候选','']
    metrics=('records','decision','legal_add_decision','add_requested','add_fill','add_rejected','voluntary_wait','restricted_wait','full_wait','tp','sl','other_exit','early_first5','early_next10','sl_pending','negative_reward')
    lines+=base.table(['group',*metrics],[(name,*[values[k] for k in metrics]) for name,values in sorted(corpus['summaries'].items())])
    lines+=['','TP以真实TAKE_PROFIT退出计数，不以reward_events数量计数；SL是可恢复的交易失败，不是全局terminal。SL奖励只在完整真实成交反馈出现。sl_pending是观察行标志（包含重复reset行），不是SL成交次数。','', '## TP自然时长及风险条件','']
    statistics=[]
    for name in sorted(corpus['summaries']):
        trades=[t for t in corpus['trades'] if t['policy_id']==name]
        early=[t for t in trades if t['first_adverse_effective_bar'] is not None and t['first_adverse_effective_bar']<=15]
        statistics.append((name,len(trades),len(early),sum(t['exit_reason']=='TAKE_PROFIT' for t in early)/len(early) if early else None,
            sum(t['exit_reason']=='STOP_LOSS' for t in early)/len(early) if early else None,
            sum(t['right_censored'] for t in trades),base.distribution([t['MAE_points'] for t in trades]),
            base.distribution([t['duration_minutes'] for t in trades if t['exit_reason']=='TAKE_PROFIT'])))
    lines+=base.table(['group','trades','early10','P(TP|early10)','P(SL|early10)','censored','MAE分布','TP分钟分布'],statistics)
    reward_rows=[]
    for name in sorted(corpus['summaries']):
        trades=[t for t in corpus['trades'] if t['policy_id']==name]
        native=sum(value for t in trades for _,value in t['reward_feedback'])
        raw=equity=0.
        for record in dataset['traces']:
            if group(record)!=name:continue
            directory=Path(root)/record['directory']
            manifest=read_manifest(directory/'manifest.json')
            account=json.loads((directory/'account_state.json').read_text())
            raw+=manifest['total_reward']
            equity+=float(account['balances']['user_cash'])+float(account['balances']['user_margin'])+float(account['unrealized_pnl'])-300000
        reward_rows.append((name,raw,native,equity,sum(t.get('net_trade_pnl',0.) for t in trades)))
    lines+=['','## 全Fold Reward与真实账户比较','',
        *base.table(['group','未折扣raw Reward','未折扣native float32','账户权益净变化(含尾部mark)','已成交交易净PnL'],reward_rows),
        '', '自然策略Reward和现金PnL应分别评估；P4是后验参照，不能用于反选Reward或当作部署收益。训练Reward仍未选定。']
    bins=('tp_le5','tp_5_15','tp_15_60','tp_60_480','tp_ge480')
    lines+=['',*base.table(['group',*bins],[(g,*[s[k] for k in bins]) for g,s in sorted(corpus['summaries'].items())]),
        '', 'MAE是持仓已观察OHLC的保守上界；退出bar内路径先后未知。早期逆行在有效bar计数，休市自然分钟不增加早期窗口。','', '## 年份/合约/方向/日夜盘','']
    strata_metrics=('advanced','legal_add_decision','add_fill','voluntary_wait','tp','sl','other_exit','early_first5','early_next10','sl_pending')
    lines+=base.table(['group','dimension','value',*strata_metrics],[(g,d,v,*[s[k] for k in strata_metrics]) for (g,d,v),s in sorted(corpus['strata'].items())])
    lines+=['','## 真实Uniform Replay批次','', '每池1000个B16、length65真实批次，context1与train64单列；实际完整capacity，没有淘汰。复现检查额外一个batch不计样本。','']
    lines+=base.table(['pool','capacity','RAM数组字节','累计峰值RSS','唯一起点','重复起点','跨chunk序列','首batch复现','来源序列'],[(s['pool_id'],s['sequence_starts'],s['ram_bytes'],s['audit_process_peak_rss_bytes'],s['unique_sampled_starts'],s['repeated_starts'],s['cross_chunk_sequences'],s['first_batch_reproduced'],s['sampled_policy_sequences']) for s in samples])
    batchrows=[]
    for sample in samples:
        for section,part in (('context1',0),('train64',1)):
            for metric in ('legal_add_decision','voluntary_wait','add_fill','tp','sl','early_first5','early_next10'):
                values=sample['counts'][:,part,INDEX[metric]]
                batchrows.append((sample['pool_id'],section,metric,int(values.sum()),float(values.mean()),
                    float((values>0).mean()),float((values==0).mean())))
    lines+=base.table(['pool','section','event','总数','mean/batch','至少1事件batch比例','零事件batch比例'],batchrows)
    lines+=['','## 预算投影（CPU分析，未发生训练）','',
        '下表100K/1M是离线梯度更新预算；B16×64。不是在线交互预算，也不是模型已见到的事件。2000更新先导统一为2,048,000训练时间位置、32,000连续序列。','']
    projection=[]
    for sample in samples:
        for updates in (100000,1000000):
            n=sample['sequence_starts'];draws=updates*16
            unique=-n*math.expm1(draws*math.log1p(-1/n)) if n>1 else 1
            projection.append((sample['pool_id'],updates,updates*16*64,draws,unique,draws-unique))
    lines+=base.table(['pool','gradient updates','模型train位置','序列抽样','预期不同起点','预期重复起点'],projection)
    lines+=['','自然+P4改善历史正TP曝光，同时会改变状态分布。是否改善网络须在同预算模型先导中检验；本轮不选Reward、不新增Actor监督Loss、不启动GPU。','', '每份产物/训练输入采用字段白名单；未来候选、Oracle目标与政策ID只在Sidecar/manifest，不进入模型Replay张量。验证及封存Test价格未读取。']
    return lines


def audit_dataset(config,dataset,output,*,batches=1000,seed=202):
    if batches!=1000 or seed!=config['sample_seed']:raise ValueError('D formal baseline requires 1000 batches, seed202')
    tests=required_tests('D');cpu_runtime(config['dreamer_root']);data=validate_dataset(dataset)
    if not data['full_training_fold']:raise ValueError('D audit requires all TRAIN contracts')
    root=Path(dataset).resolve();output=make_output(output,config)
    code=execution_provenance(config,'D')
    for module in (__file__,base.__file__):code['code_files'][str(Path(module).resolve())]=digest(module)
    snapshots=source_witness(output,code)
    frozen={str(root/'manifest.json'):digest(root/'manifest.json'),**code['code_files']}
    corpus=base.corpus_statistics(root,data,columns=COLUMNS,label_fn=row_labels,trace_schema=SINGLE3_SCHEMA,
        group_fn=group,row_hook=row_hook,trade_finalize=finalize_trade)
    totals=Counter()
    for s in corpus['summaries'].values():totals.update(s)
    if totals['lots6'] or totals['illegal_add']:raise ValueError('Single3 capacity or action contract violated')
    samples=[]
    for pool in data['views']:
        sample=base.sample_statistics(root,data,corpus,pool,batches=batches,batch_size=16,seed=seed,
                                      columns=COLUMNS,group_fn=group)
        if sample['audit_process_peak_rss_bytes']>config['rss_budget_gib']*2**30:
            raise RuntimeError('Audit RSS budget exceeded')
        np.savez_compressed(output/f'{pool}_batch_counts.npz',counts=sample['counts'],columns=np.array(COLUMNS))
        samples.append(sample)
        print(f"Single3 Uniform {pool}: {batches} real batches / capacity {sample['capacity']}",flush=True)
    write_json(output/'trade_diagnostics.json',corpus['trades'])
    import jax
    platforms=[d.platform for d in jax.devices()]
    if not platforms or set(platforms)!={'cpu'}:raise ValueError('Non-CPU audit')
    validate_dataset(root);check_files(frozen)
    # No PASS report is emitted until every final source validation passes.
    report='\n'.join(_coverage_lines(data,corpus,samples,root))+'\n'
    (output/'replay_coverage.md').write_text(report)
    receipt=signed(dict(schema='FG_SINGLE3_CPU_V1',stage='D_AUDIT',status='PASS',full_training_fold=True,
        dataset_manifest_sha256=digest(root/'manifest.json'),batches=batches,batch_size=16,length=65,context=1,seed=seed,
        tests=tests,gradient_updates=0,jax_platforms=platforms,provenance=code,source_snapshots=snapshots,
        frozen_inputs={str(root/'manifest.json'):digest(root/'manifest.json')},totals=dict(totals),
        unique_advanced_minutes=corpus['unique_advanced_minutes'],account_restores=corpus['account_restores'],
        pools=[{k:v for k,v in s.items() if k!='counts'} for s in samples],
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))
    write_json(output/'manifest.pending.json',receipt)
    try:
        check_files(frozen);check_files({str(output/p):h for p,h in receipt['files'].items()})
        (output/'manifest.pending.json').replace(output/'manifest.json')
    except BaseException:
        (output/'replay_coverage.md').replace(output/'replay_coverage.failed.md')
        raise
    return receipt

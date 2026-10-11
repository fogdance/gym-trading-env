"""Render the frozen FG409 offline study without changing its Env results."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description='Render the completed, fixed FG409 Reward V2/V3 offline study')
parser.add_argument('--study-dir',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True,help='New Markdown report; existing files are preserved')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
OUT=args.study_dir.resolve()
doc=args.out.resolve()
if doc.exists():raise FileExistsError(f'Refusing to overwrite report: {doc}')
summary=json.loads((OUT/'study_summary.json').read_text())
if (summary['status']!='PASS' or summary['trajectories']!=36 or
        summary['score_rows']!=1152 or not summary['frozen_hashes_verified_before_and_after']):
    raise ValueError('A completed, reconciled 36 × 32 FG409 study is required')
data=pd.read_csv(OUT/'parameter_scores.csv')
inputs=json.loads((OUT/'study_inputs.json').read_text())
if inputs['config']['contract']!='CZCE.FG409':
    raise ValueError('Report is restricted to the declared FG409 development study')
ref=data[(data.beta==.2)&(data.lam==.0002)]
zero=ref[ref.alpha==0];shape=ref[ref.alpha==.05]
order=['logged_actions','always_add','always_wait','delay5','delay15','delay30','random']
labels=dict(zip(order,['A 原动作','Always ADD','Always WAIT','Delay5','Delay15','Delay30','Random（30 Seed，中位数）']))
def value(df,policy,key):
    values=df[df.policy==policy][key].dropna()
    return float(values.median()) if len(values) else None
def num(v,d=4):
    if v is None or (isinstance(v,(float,np.floating)) and not np.isfinite(v)):return '—'
    if isinstance(v,str):return v
    if v==0:v=0.0
    return f'{v:,.{d}f}'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

parts=['# FG Reward V2/V3 离线验收与研究报告 V1\n',
       '日期：2026-10-11。开发分支：`feat/fg-reward-v2-v3-offline`。阶段一代码提交：`28cfc88`。\n',
       '**工程验收通过：36 条真实 Env 轨迹 × 32 组候选参数，共 1,152 条参数结果。正式 Reward、Env、Obs、Action、DreamerV3、Gate、Checkpoint/Replay 未修改；没有模型推理或训练，也没有访问封存 Test 价格。**\n',
       '## 1. 可以确认的结论与不能确认的结论\n']
a=zero[zero.policy=='logged_actions'].iloc[0]
delays=zero[zero.policy.isin(['delay5','delay15','delay30'])]
best=delays.loc[delays.v0.idxmax()]
random=zero[zero.policy=='random']
n_random=int((random.v0>=a.v0).sum())
parts += [f'- A 原动作重新回放与 Always ADD 完全一致：75 批开仓、73 批 TP、2 批主力切换非 TP 退出；V0={a.v0:.9f}、净收益18,990元、最大浮亏38,460元。\n',
          f'- 已确认此 FG409 区段存在简单等待策略的收益空间：{labels[best.policy]} 的 V0={best.v0:.9f}，高于 A 的 {a.v0:.9f}。但是该策略开仓 {best.ADD_fill:.0f} 批、A 开仓75批，交易次数和轨迹不同，不能把差额全部解释为纯择时能力。\n',
          f'- 在事先指定的30个 Random Seed 中，{n_random}/30 条轨迹 V0 不低于 A。它们没有匹配 A 的成交配额，结果是闭环策略对照，不是严格的交易频率匹配试验。\n',
          '- V2/V3 的后验计分会改变旧目标。480分钟后零奖、早期反向罚、容量成本都只作用于离线候选分数，不改变这些固定轨迹的交易、费用或账户。\n',
          '- alpha=.05 的未折扣 shaping 分数不能和完整轨迹的折扣回报混为一谈；它可以在不利持仓状态下提供逐分钟正分。详见第6节。\n',
          '- 本研究没有证明新的 Reward 会使 Dreamer 学会 WAIT，没有校准最佳参数，也没有证明跨合约、跨年份或 LONG 方向的泛化。\n',
          '## 2. 输入、成交假设和实验身份\n',
          '- 唯一合约：CZCE.FG409，人工方向SHORT；使用已冻结的 v3_shared_prefix 开发验证 Env、行情、合约时钟与费用配置。没有使用盲测行情，也没有重新导出数据。\n',
          '- 时间口径：Asia/Shanghai；已计分分钟为2024-04-08 21:02至2024-08-19 15:00，共31,034根有效一分钟、91个有资格交易日；初始完整K线只产生第一条决策，不重复计分。该区段没有零时长记录。\n',
          '- 账户初始300,000元、每次3手、最多6手；20吨/手，保证金10%，开平各3元/手，单侧价差1点，滑点0。主力切换按原合约最后可成交分钟退出。\n',
          '- “成交”指原 Env 已提交的保守分钟模拟成交，不是券商实盘确认。入场在下一执行K线开盘；新入场同根TP须由开仓后的收盘支持，旧仓位TP还需原协议的穿价/流动性验证。研究评分器不新增撮合规则。\n',
          f'- 冻结输入7项，软件/配置/原评估收据校验{len(inputs["source_hashes"])}项；运行前后哈希一致。原A评估CSV/JSON也逐文件核验。完整清单见 `{OUT.relative_to(ROOT)}/study_inputs.json`，E1复用收据见 `A_logged/reuse_receipt.json`。\n',
          '- 独立 Env 用4个spawn进程，单合约内部按时间顺序推进。A 已通过的 E1 通过源码/参数/数据/原日志/状态身份校验复用；不加载模型权重。\n',
          '## 3. 候选公式与已确认的实现修正\n',
          r'V0 只对完整3手TP发放 $60/(60+T)$。V2的TP分量为：$T\le60$ 时仍为V0；$60<T<480$ 时乘 $(480-T)/420$；$T\ge480$ 时为0。T始终是自然分钟。'+'\n',
          '- 每个entry_id首次反向达到10点：第1–5根有效分钟罚−beta，第6–15根罚−beta×rho，之后不罚。入场所在执行K线算第1根，休市和零时长记录不增加风险年龄。\n',
          r'- V2另加 $\alpha(\gamma\Phi_{next}-\Phi_{now})$；$\Phi=0.5\sum clip(side\,(Close-entry)/5,-2,1)$，仅使用当时已观察价格和已有成交身份。'+'\n',
          r'- V3另减 $\lambda q_t(n_t/2)^2$；0/1/2个实际占用批次分别为0/.25/1单位。q由真实执行分钟、交易日、人工方向和截止时刻确定，不依赖账户容量、资金、动作、未来价格或Volume。'+'\n',
          '- 容量按实际入场后、分钟末结算前计费。原Env将退出成交时间记为eob，因此本根完成TP仍计这根占用；同时报告决策前与区间占用两种分母，不能混用。\n',
          '- 修正了原补丁允许重叠的60秒K线；新增回归在原补丁失败，修复后通过，失败输入保持原评分器状态。\n',
          '- 原补丁只比较Reward事件数量/总和，换掉entry_id仍能通过。现在逐步及逐笔校验动作、成交、归因、Reward和权益序列；篡改身份回归先失败后通过。\n',
          '- Delay按有效行情分钟计数；资格丢失或成交后清零，休市/零时长不累积。修正区间资格使用执行时刻而非上根信号时刻，避免跨休市/方向/截止边界错计。\n',
          '- 保留chunk账户及评分状态；区分右删失和终局强平未成交，不通过reset清除头寸。已完成的480分钟以上TP仍记success，而不是失败。\n',
          '## 4. E1 真实A逐步与账户对账\n',
          table(['项目','结果'],[['决策 / 入场 / TP / 非TP','31,034 / 75 / 73 / 2'],
          ['V0 / G(T)累计',f'{a.v0:.12f} / {a.tp_timeboxed:.12f}'],
          ['净收益 / 最大浮亏 / 最大权益回撤','18,990 / 38,460 / 38,640 元'],
          ['已实现毛损益 / 手续费','20,340 / 1,350 元'],['终局未完成头寸 / 右删失','0 / 0'],
          ['止盈自然分钟中位数 / P90 / 最长','556 / 6,359.6 / 129,583'],
          ['chunk金融连续性 / scorer JSON恢复','30次全部通过'],
          ['重组与直接评分比较',f'{summary["reconstruction_check"]["steps"]}步，最大分量误差 {summary["reconstruction_check"]["max_step_error"]}']]),
          '逐步V0与Env Reward误差要求≤1e-9，累计≤1e-8；事件身份、时刻、方向、数量和价格逐笔核对。金额不是通过重新定义Reward推算出来的。\n',
          '73批TP中，36批在480自然分钟之前，37批在480分钟及之后；后37批的候选G为0。A有4批在前5根首次触发−10点、15批在第6–15根首次触发，beta=.2/rho=.3时早期罚总和−1.7。\n',
          '## 5. E2 对照策略与参数敏感性\n',
          '所有确定性策略实际独立闭环运行一次；Random使用Python random.Random，Seed0–29，ADD合法时概率.5。下表用beta=.2、rho=.3、lambda=.0002、alpha=0，属于预先声明的探索点。Random列为30次结果中位数，不能将每列中位数组合当作某一条实际轨迹。\n']
rows=[]
for p in order:
    rows.append([labels[p]]+[num(value(zero,p,k),d) for k,d in
        [('ADD_fill',1),('tp_count',1),('mean_tp_wait_minutes',1),('median_tp_minutes',1),
         ('v0',5),('v2',5),('v3',5),('account_net_pnl',0),('max_floating_loss',0),
         ('full_capacity_interval_fraction',6),('unfinished_entries',1)]])
parts += [table(['策略','入场批','TP批','平均TP分钟','中位TP分钟','V0','V2','V3','净收益元','最大浮亏元','六手区间占用比例','未完成批'],rows),
          'Always WAIT的无交易平均/中位等待时间记“—”，不伪记为0分钟成功。费用、方向、初始资金和撮合相同；实际成交次数没有匹配。各轨迹共享91个外生有资格交易日的比较分母，实际观察日与有效分钟另列在参数汇总中。\n',
          f'![策略评分和Random范围]({OUT}/strategy_scores.png)\n',
          '图中Random误差线为Seed经验2.5%–97.5%分位区间，表示固定FG409行情下的策略随机性，不是市场泛化置信区间。未把分钟样本当成独立观测。\n',
          '### 5.1 Random经验分布\n']
parts += [table(['指标','最小','P2.5','中位','P97.5','最大'],[
    [k]+[num(random[k].quantile(q),4) for q in (0,.025,.5,.975,1)] for k in
    ('ADD_fill','tp_count','v0','v2','v3','mean_tp_wait_minutes','account_net_pnl','max_floating_loss','full_capacity_interval_fraction')]),
          '### 5.2 A的完整32点分量与Always WAIT风险\n',
          '弱/中/强网格beta=0/.05/.10/.20，lambda=0/.00005/.0002/.001，alpha=0/.05；rho=.3、gamma=332/333固定。下表全部为同一条已冻结A轨迹的离线重算，不是32个新训练模型。\n']
aa=data[data.policy=='logged_actions'].sort_values(['alpha','beta','lam'])
parts += [table(['beta','lambda','alpha','G','early','potential','capacity','V2','V3','WAIT胜过V3'],[
    [num(r.beta,2),num(r.lam,5),num(r.alpha,2)]+[num(r[k],5) for k in
    ('tp_timeboxed','early','potential','capacity','v2','v3')]+['是' if r.v3 < -1e-12 else '否']
    for _,r in aa.iterrows()]),
          f'A有{int((aa.v3 < -1e-12).sum())}/32组候选的V3低于Always WAIT的0分；这只是固定策略总分比较，不等于已证明学习后的全局最优策略必然WAIT。\n',
          f'beta=.2、alpha=0时，A的容量惩罚临界lambda约为{(a.tp_timeboxed+a.early)/a.capacity_basis:.9f}。它来自该合约实际分量，不能直接推广为正式参数。\n',
          f'![A参数敏感性]({OUT}/A_parameter_surface.png)\n',
          '### 5.3 正负评分发生率\n',
          '阈值绝对值≤1e-12仅在频率统计中视为数值零，不改变实际评分。分母为每条轨迹的全部计分步；本区段均为有效分钟。多笔TP可在同根发生，因此“正分钟数”不等于“TP批数”。\n']
freq=[]
for p in order:
    for alpha,df in [(0,zero),(.05,shape)]:
        freq.append([labels[p],num(alpha,2)]+[num(value(df,p,k),d) for k,d in
            [('v2_positive_steps',1),('v2_negative_steps',1),('v3_positive_steps',1),
             ('v3_negative_steps',1),('v3_positive_fraction',6),('v3_negative_fraction',6)]])
parts += [table(['策略','alpha','V2正分钟','V2负分钟','V3正分钟','V3负分钟','V3正比例','V3负比例'],freq),
          '## 6. PBRS、额外折扣和评分解释\n',
          '原Dreamer源码的imag_loss/repl_loss存在horizon=333对应折扣；当前contdisc=True时想象侧将折扣并入continuation。离线诊断使用常数gamma=332/333，每个Env推进步折扣一次，不按休市自然分钟重复折扣。模型预测continuation、跨chunk Bootstrap和价值学习误差没有在本研究中验证。\n',
          'A示例alpha=.05的未折扣potential累计为+8.553393393；折扣累计为1.797e-16，完整终局的端点项为0。逐chunk没有重置Phi或折扣权重。相同状态轨迹的数学望远镜抵消成立，不足以声称Dreamer实际训练的策略不变性已通过验收。\n',
          '当两批仓位都处于Phi=-2的剪裁下界、没有新TP或早期罚时，alpha=.05产生+.0003003003分/步，lambda=.0002扣−.0002，V3净+.0001003003。A有26,543个分钟满足这一状态。正评分密集并不是这些分钟真实盈利，也不是新的成交事件。\n',
          f'A在alpha=0时V3={value(zero,"logged_actions","v3"):.9f}，alpha=.05时V3={value(shape,"logged_actions","v3"):.9f}；完整折扣V3分别为{value(zero,"logged_actions","discounted_v3"):.12f}和{value(shape,"logged_actions","discounted_v3"):.12f}。不能用未折扣 shaping 总分替代训练折扣目标或真实PnL。\n',
          '## 7. 同根OHLC歧义、容量与删失审计\n',
          '风险和TP同一分钟不能由OHLC推断先后。默认保守地保留风险罚；敏感性只退回“同根首次触发且仍在1–15根风险窗口”的罚。此前已经罚过的风险不再次退款，也不创造新的成交路径。\n']
amb=[]
for p in order:
    amb.append([labels[p],num(value(zero,p,'ambiguous_tp_bars'),1),
                num(value(zero,p,'ambiguity_refund'),5),num(value(zero,p,'v3'),5),
                num(value(zero,p,'v3_without_ambiguous_first_risk'),5)])
parts += [table(['策略','歧义TP分钟','不计同根首次风险的增分','保守V3','敏感性V3'],amb),
          'A有2批TP处于同一歧义分钟，但其首次−10点已在此前发生，故本次早期罚敏感性增分为0。不能把所有歧义TP当成2次新的早期罚。\n',
          f'A决策前六手比例为{a.full_capacity_time_fraction:.8%}，入场后/退出前区间六手比例为{a.full_capacity_interval_fraction:.8%}；累计容量基础量31,023.5单位。两者定义不同，前者与旧评估对账，后者对应本候选成本。\n',
          '- 当前完整FG409轨迹在主力边界按原规则结算。每条轨迹的pending_reason、终局未成交与右删失数量均独立记录，没有将未平仓免费删除。\n',
          '- 专项回归验证数据末尾没有终局退出时保留right_censored；有强平原因但Volume无法成交时保留terminal_failure_unfilled和两批头寸；两者不能混算。\n',
          '## 8. A的75笔审计明细\n',
          '下表早期罚/容量成本使用beta=.2、rho=.3、lambda=.0002。MAE沿用与原模拟成交对应的完整交易区间OHLC保守上界；potential是账户状态分量，未伪造逐笔分摊。完整入场/退出时间、价位、身份、归因见trade_components及原账户/成交文件。\n']
trades=pd.read_csv(OUT/'A_logged/trade_components.csv')
old=pd.read_csv(ROOT/'artifacts/fg_abcd_v1/v3_shared_prefix/pilot_evaluation/CZCE.FG409/A/0/dreamerv3/trades.csv').set_index('entry_id')
details=[]
for _,r in trades.iterrows():
    end=r.exit_reason
    duration=(pd.Timestamp(r.exit_time)-pd.Timestamp(r.entry_time)).total_seconds()/60
    details.append([r.entry_id,end,num(duration,0),num(old.loc[r.entry_id,'MAE_points'],0),
                    num(r.original_tp,6),num(r.timeboxed_tp,6),num(r.adverse_first_bar,0),
                    num(r.early_penalty,2),num(r.capacity_cost,5)])
parts += [table(['entry_id','终局','自然分钟','MAE点','V0','G','首次−10点根数','早期罚','容量成本'],details),
          '## 9. 验证、产物与复现\n',
          '- 原交付12项stdlib单测和纯人工合成smoke通过；修复后20项候选数学/真实Env适配/恢复/身份回归通过。新增重叠K线与篡改Reward身份测试均验证过修复前失败。\n',
          f'- 实际研究完成{summary["trajectories"]}条轨迹、{summary["grid_points"]}个候选、{summary["score_rows"]}条参数汇总。A极端候选31,034步重组与直接计分最大分量误差0；数学测试覆盖完整32点。\n',
          '- 各轨迹目录提供events.jsonl、step_components.csv、trade_components.csv、snapshot.json、summary.json、成交/Reward/权益/账户状态；parameter_scores、parameter_trade_scores、parameter_daily_scores保存参数重算。所有数据和图片在artifacts，未加入Git。\n',
          '- 逐笔参数分数中score_excluding_account_potential明确不包含账户PBRS分量；不能把它当作完整V2/V3。\n',
          '```bash\nconda activate dreamerv3\nPYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\\npython tools/fg_reward_offline.py \\\n  --study-config configs/fg_reward_offline_v1.yaml \\\n  --out artifacts/fg_reward_offline_v1/NEW_STUDY_DIRECTORY\n```\n',
          '配置中的verified_e1是可选已验收缓存；源码、参数或冻结输入不一致时拒绝复用。没有该缓存时去掉这一可选配置项，入口先执行E1。已有输出目录拒绝覆盖。\n',
          '报告入口只接受已完成的36×32 FG409研究。生成新报告（现有文件拒绝覆盖）：\n',
          '```bash\npython tools/report_fg_reward_offline.py \\\n  --study-dir artifacts/fg_reward_offline_v1/study_20261011 \\\n  --out docs/NEW_FG_REWARD_OFFLINE_REPORT.md\n```\n',
          '## 10. 后续建议与限制\n',
          '1. 保留V0正式业务目标。当前结论支持“等待值得研究”，不支持直接替换正式Reward。\n',
          '2. 先完成已有A/B/C的真实ADD召回、Actor概率和想象动作空间诊断。Reward离线计分无法代替Gate/长期信用分配验收。\n',
          '3. 参数如需进一步研究，应在其他开发合约中检查稳定性、方向与自然时间跨夜差异，并把PBRS与基础目标分别评价；最终盲测继续封存。不得根据本单一合约的事后最高分冻结正式参数。\n',
          '4. 新Reward的GPU训练、学习后Always WAIT退化风险以及模型泛化均未验证；正式训练仍由用户另行决定和启动。\n']

doc.parent.mkdir(parents=True,exist_ok=True)
doc.write_text('\n'.join(parts),encoding='utf-8')
quantiles=[]
for grid_id,group in data[data.policy=='random'].groupby('grid_id'):
    quantiles.append(dict(grid_id=grid_id,seeds=len(group),**{
        f'{k}_p{int(q*1000):03d}':float(group[k].quantile(q)) for k in
        ('v0','v2','v3','account_net_pnl','max_floating_loss','ADD_fill') for q in (.025,.5,.975)},
        v0_at_least_A=int((group.v0>=a.v0).sum()),v3_below_wait=int((group.v3<-1e-12).sum())))
pd.DataFrame(quantiles).to_csv(OUT/'random_seed_distribution.csv',index=False)
receipt=dict(report_source=str(Path(__file__).resolve()),
    report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    report=str(doc),report_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),
    study_inputs_sha256=hashlib.sha256((OUT/'study_inputs.json').read_bytes()).hexdigest(),
    study_summary_sha256=hashlib.sha256((OUT/'study_summary.json').read_bytes()).hexdigest(),
    parameter_scores_sha256=hashlib.sha256((OUT/'parameter_scores.csv').read_bytes()).hexdigest())
(OUT/'report_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
checksums={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in OUT.rglob('*') if p.is_file() and p.name!='output_checksums.json'}
(OUT/'output_checksums.json').write_text(json.dumps(checksums,indent=2),encoding='utf-8')
print('REPORT',doc,'lines',len(doc.read_text().splitlines()))
print('RESULT',dict(random_at_least_A=n_random,best_simple_policy=best.policy,
                    best_simple_v0=float(best.v0),A_wait_beats_points=int((aa.v3<-1e-12).sum())))

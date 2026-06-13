# Dreamer 入场能力隔离验证实施方案

日期：2026-06-13

状态：Phase 0 - Phase 3 已实现并完成首轮验证

当前批准范围：

```text
Phase 0：冻结实验契约
Phase 1：实现 Counterfactual Entry Evaluator
Phase 2：生成 Opportunity 与基线报告
Phase 3：训练监督入场基线并生成非重叠策略报告
```

当前明确不实现：

```text
Entry-only Dreamer
ATR ablation
动态 5m MACD
完整陈大哥识别器
智能退出
```

关联文档：

- `docs/obs_action_profitability_assessment.md`
- `docs/cdg_trading_system_integration_plan.md`
- `docs/action_mask_full_chain_implementation_plan.md`

## 1. 要回答的唯一问题

当前 Dreamer 同时负责：

```text
选择是否入场
+ 选择方向
+ 持仓
+ 平仓
+ 反手
```

训练和 Monte Carlo 结果不能区分亏损究竟来自错误入场，还是正确入场后的错误退出。
当前两个最终 checkpoint 在 Monte Carlo 中又都表现为每个 episode 固定开满 3 次，
因此系统尚未证明 Agent 能主动筛选正期望入场点。

本阶段只回答：

> 在固定、确定、与 Agent 无关的退出规则下，当前 observation 能否识别出扣除全部
> 交易成本后具有样本外正期望的 LONG / SHORT 入场点，并在没有优势时选择不交易？

本阶段不解决：

```text
智能平仓
移动保本
利润回吐保护
反手
仓位大小
完整陈大哥识别系统
实盘部署
```

这些能力只有在入场能力被独立证明后才值得继续开发。

## 2. “会入场”的工程定义

“会入场”不能定义为方向准确率超过 50%，也不能定义为训练期 score 上升。

正式定义为：

```text
1. Agent 可以不交易，而不是每天消耗完全部开仓额度。
2. 被 Agent 选中的入场点，在固定退出规则下样本外净 expectancy > 0。
3. 被选中的入场点明显优于随机入场、固定方向和简单规则基线。
4. 预测置信度越高，真实净 expectancy 越高。
5. 结果不能由单个月份、少数大盈利或低交易数量支撑。
6. 扣除手续费、点差和保守成交假设后仍然成立。
```

核心指标不是所有时间点的分类准确率，而是：

```text
selective entry expectancy
```

即模型只交易自己认为最有优势的一部分机会时，这部分交易是否具有稳定正期望。

## 3. 为什么先做离线 Counterfactual Entry Evaluator

对每个机械上允许开仓的时刻 `t`，离线评估三个选择：

```text
SHORT
FLAT
LONG
```

LONG 和 SHORT 使用完全相同、提前冻结的退出规则。FLAT 的收益恒为零。未来行情只用于
生成离线结果标签，绝不进入时刻 `t` 的 observation 或模型输入。

这样能够把问题拆为三个可诊断层次：

```text
层次 1：固定退出规则下，市场中是否客观存在足够的正收益机会？
层次 2：当前 observation 是否包含可识别这些机会的信号？
层次 3：Dreamer 是否能学会利用已经被监督模型证明存在的信号？
```

如果受约束 oracle 都没有足够机会，应调整固定退出规则或数据范围。

如果 oracle 有机会，但简单监督模型无法样本外识别，说明当前 observation 信息不足，
不应继续依赖 Dreamer 自动发现信号。

如果监督模型可以识别，但 Entry-only Dreamer 不行，问题才明确位于 RL 训练链路。

## 4. 第一版冻结范围

第一版使用当前基线分支：

```text
branch: dev/action-mask-full-chain
market_seq: float32[60, 18]
actions: [SHORT, FLAT, LONG]
product: DCE JM
direction: LONG 和 SHORT 都评估
```

动态 5m MACD 当前保留在独立实验分支，不进入第一版基线。ATR、MACD、日线和 1h
结构后续都作为独立 ablation 加入，必须用同一份 outcome 和同一份时间切分比较，
不能更换数据后直接宣称改进。

第一版固定退出策略：

```text
入场：看到时刻 t 的完整 observation 后，在下一根有效 1m bar 开盘成交
止损：固定绝对价格距离，并与 tick 数双重校验
止盈：固定 reward/risk 倍数
最大持仓：固定有效 1m bar 数
日内约束：不得跨交易日，EOD 强制退出
跨休市入场：默认禁止
反手：禁止
加仓：禁止
移动止损：禁止
保本：禁止
```

建议初始 canonical 参数：

```yaml
entry_evaluator:
  entry_delay_bars: 1
  stop_distance_price: 5.0
  stop_distance_ticks: 10
  take_profit_rr: 1.5
  max_hold_bars: 60
  max_hold_unit: valid_1m_bars
  allow_entry_across_break: false
  force_flatten_eod: true
  intrabar_collision: stop_first
```

`stop_distance_price` 的单位是产品价格单位；对焦煤可理解为元/吨报价点数。
`stop_distance_ticks` 的单位是最小价格变动 tick 数。产品配置中的 `tick_size` 是权威
来源，加载配置时必须 fail fast 校验：

```python
stop_distance_price == stop_distance_ticks * tick_size
```

以 `tick_size=0.5` 为例：

```text
stop_distance_price = 5.0
stop_distance_ticks = 10
```

参数值必须进入带版本的配置文件。Canonical 参数用于主报告；允许在验证集上比较有限
的参数网格，但测试集结果只能使用验证集提前选定的参数。

建议敏感性参数网格：

```text
stop_distance_price: 3 / 5 / 8
take_profit_rr: 1.0 / 1.5 / 2.0
max_hold_bars: 15 / 30 / 60
```

不得根据测试集选择最优组合。最终报告必须同时展示：

```text
canonical 参数结果；
validation 选参结果；
不同参数下受约束 oracle 的机会分布；
不同参数下监督模型和可执行策略的稳定性。
```

如果盈利只出现在单一参数组合，而相邻参数全部失效，结果应标记为脆弱，不得直接进入
后续 Dreamer 阶段。

## 5. 严格时序与成交语义

### 5.1 决策和成交

在时刻 `t`：

```text
可见：截至 t 完整收盘后的 observation
不可见：t+1 以及之后的任何行情
成交：下一根有效 1m bar 的 Open
```

不能继续使用“看完 Close[t] 后按同一个 Close[t] 成交”的乐观假设。

`entry_delay_bars` 和 `max_hold_bars` 都按有效 1m bar 计数：

```text
mask_t=0 不计数；
休市期间不增加 holding_bars；
EOD 仍按交易日边界强制退出。
```

第一版默认：

```yaml
allow_entry_across_break: false
```

如果决策时刻与下一根有效 bar 之间跨越休市或交易时段边界，该时刻不作为候选。后续
可以单独实验开启，但必须独立报告隔夜或休市跳价风险，不能与 canonical 结果混合。

### 5.2 成交成本

必须使用当前产品配置中的单一成本定义：

```text
spread
trading_fee_per_lot
is_round_turn
lot_size
trade_lot
```

每个 outcome 必须记录：

```text
gross_pnl
spread_cost
fee_cost
net_pnl
```

验收以 `net_pnl` 为准。

第一版离线 evaluator 可以实现独立、可审计的成本计算，但必须通过抽样 parity test
证明其结果与环境 ledger 对同一笔固定交易完全一致。若出现差异，环境 ledger 为权威，
不得通过放宽断言解决。

### 5.3 同一分钟同时触发止损和止盈

只有 1m OHLC 时，无法知道同一分钟内止损和止盈谁先发生。第一版统一采用保守规则：

```text
intrabar_collision = stop_first
```

报告必须统计 collision 数量和占比。后续如果接入 tick 数据，可以单独复核，但不能
在第一版中使用对策略更有利的顺序。

### 5.4 EOD 和无下一根行情

候选点必须满足：

```text
存在下一根有效 bar 可成交；
机械 action_mask 允许目标方向；
入场后仍可在日内完成退出；
不跨交易日持仓。
```

到 EOD、数据边界或最大持仓时间仍未触发 barrier 时，按当时可执行的保守平仓价格
退出，并记录明确的 `exit_reason`。

## 6. Counterfactual Outcome 数据契约

每个合法候选时刻分别生成 LONG 和 SHORT outcome。

```python
EntryOutcome(
    outcome_version,
    dataset_version,
    exit_policy_version,
    product,
    contract,
    decision_timestamp,
    entry_timestamp,
    direction,                 # LONG / SHORT
    entry_reference_price,
    entry_execution_price,
    stop_price,
    target_price,
    exit_timestamp,
    exit_execution_price,
    exit_reason,              # STOP / TARGET / MAX_HOLD / EOD / DATA_END
    holding_bars,
    entry_crosses_break,
    gross_pnl,
    spread_cost,
    fee_cost,
    net_pnl,
    return_on_initial_equity,
    mfe_gross,
    mae_gross,
    mfe_net_after_entry_cost,
    mae_net_after_entry_cost,
    stop_target_collision,
)
```

同一候选时刻再生成一条模型样本：

```python
EntryCandidate(
    candidate_id,
    decision_timestamp,
    trading_day,
    split,
    market_seq_t,
    flat_agent_state_t,
    action_mask_t,
    long_outcome,
    short_outcome,
    flat_net_pnl=0,
)
```

`flat_agent_state_t` 使用统一的空仓账户状态，避免之前的交易路径污染纯入场判断：

```text
当前仓位 = FLAT
未实现收益 = 0
已实现收益 = 0
entries_used_today = 0
账户资金足以开一手
时间、session phase、minutes_to_eod 等保持真实值
```

第一版监督诊断主要使用 `market_seq_t`。`flat_agent_state_t` 仅保留时间和机械状态，
不得让随机历史账户状态成为入场标签的捷径。

时间和 session 特征不属于未来信息泄漏，但可能让模型只记住某些时段或日期偏差。
报告必须增加以下诊断：

```text
仅时间特征模型；
移除时间特征模型；
完整特征模型；
按 session phase、月份、合约、交易日、LONG / SHORT 分组表现。
```

如果仅时间特征可以解释大部分收益，或者移除时间特征后优势完全消失，结果必须明确
标记为时间偏差驱动，不能解释为模型识别了行情形态。

## 7. 标签和建模方式

### 7.1 不以三分类作为主任务

直接把每个时刻标为 LONG / FLAT / SHORT 存在两个问题：

```text
接近零收益的样本标签会因微小价格变化频繁翻转；
FLAT 样本数量巨大，模型可能通过永远 FLAT 获得高准确率。
```

第一版主任务应直接预测两个 counterfactual value：

```text
Q_long(t)  = 固定退出规则下 LONG 的预期净收益
Q_short(t) = 固定退出规则下 SHORT 的预期净收益
```

模型决策：

```python
best_direction = argmax(pred_q_long, pred_q_short)
best_value = max(pred_q_long, pred_q_short)

if best_value <= threshold:
    action = FLAT
else:
    action = best_direction
```

`threshold` 只能在验证集选择，并且必须高于交易成本和预测噪声所要求的安全边际。

### 7.2 辅助任务

辅助输出可以包括：

```text
P(net_pnl > 0 | LONG)
P(net_pnl > 0 | SHORT)
P(target before stop | LONG)
P(target before stop | SHORT)
expected MFE
expected MAE
```

辅助任务用于诊断和校准，不替代净收益回归与最终策略模拟。

### 7.3 监督模型顺序

从简单到复杂：

```text
1. 线性 Ridge / Logistic Regression
2. LightGBM 或 XGBoost
3. 小型时序网络
4. Entry-only Dreamer
```

如果 LightGBM/XGBoost 无法通过样本外门禁，不进入 Entry-only Dreamer。

## 8. 两种不同的评估表必须分开

### 8.1 Opportunity Table

Opportunity Table 独立评估每个候选时刻。它允许候选相互重叠，用于研究：

```text
正收益机会密度
LONG / SHORT 条件分布
不同时间段和市场状态下的 outcome
模型分数与真实 outcome 的单调关系
```

它不能直接当作可执行策略收益，因为现实中不能同时持有大量相互重叠的交易。

### 8.2 Executable Strategy Simulation

最终验收必须运行非重叠策略模拟：

```text
一次只能持有一个仓位；
持仓期间忽略新的 entry signal；
退出后可以设置 cooldown；
每日最多开仓数量作为风险上限，而不是交易配额；
允许整日 0 次交易；
使用与 outcome 完全相同的成交和退出规则。
```

策略模拟必须报告每日交易数分布：

```text
0 次交易日占比
1 次交易日占比
2 次交易日占比
3 次交易日占比
```

如果模型仍然每天恰好开满 3 次，则入场筛选能力门禁失败。

## 9. 必须比较的基线

### 9.1 不交易基线

```text
永远 FLAT
净收益 = 0
```

这是任何可交易策略必须超过的最低基线。

### 9.2 随机和固定策略

```text
随机选择 LONG / SHORT / FLAT
随机选择方向，但交易数量与模型相同
oracle_trade_count_matched_random
固定时间入场
永远 LONG
永远 SHORT
```

`oracle_trade_count_matched_random` 和模型 matched-random 必须分别匹配对应策略的：

```text
总交易数量
每日最大交易数
一次一仓约束
非重叠持仓约束
固定退出规则
```

随机基线必须重复多个 seed，并报告均值和区间。

### 9.3 简单规则基线

至少包括：

```text
当前价格相对 VWAP 的方向
最近 N 分钟收益方向
session open drift 方向
rolling volatility 阈值过滤
仅时间和 session phase 特征模型
```

这些基线用于判断复杂模型是否只是复现一个简单规则。

### 9.4 受约束 Oracle

Oracle 可以读取真实 LONG/SHORT outcome，但仍必须遵守：

```text
一次一仓
每日最多开仓数
固定退出规则
不能选择相互重叠交易
```

Oracle 只用于估计当前固定退出规则下的机会天花板，不能作为可实现策略。

Oracle 结果解释：

```text
Oracle 也弱：固定退出或数据中机会不足；
Oracle 强、模型弱：observation 或模型不足；
Oracle 强、监督模型强、Dreamer 弱：RL 训练链路不足。
```

## 10. 数据切分和防泄漏

禁止随机打散候选样本后切分。

按交易日顺序生成 walk-forward split：

```text
train -> validation -> test
```

至少执行 3 个 walk-forward fold。每个 fold 的 threshold、退出参数和模型超参数只能
使用 train + validation 确定，test 只运行一次最终评估。

由于每个候选标签会读取未来最多 `max_hold_bars`，split 边界必须执行 purge：

```text
任何 outcome 的 exit_timestamp 跨越 split 边界，该候选从较早 split 删除。
```

建议额外添加 embargo：

```text
embargo_bars >= max(window_size, max_hold_bars)
```

置信区间和 bootstrap 必须以交易日为采样单位，不能以高度相关的逐分钟候选为独立
样本，否则会严重高估显著性。

最终报告必须按以下维度分组：

```text
walk-forward fold
月份
合约
LONG / SHORT
session phase
预测分数分位组
时间特征开启 / 移除
```

测试集完整交易少于 30 笔时，只能判定“证据不足”，不能判定策略有效。

## 11. 关键指标

### 11.1 Opportunity 指标

```text
合法候选点数量
LONG 正收益机会率
SHORT 正收益机会率
best-direction 正收益机会率
受约束 oracle expectancy
受约束 oracle trades/day
固定退出各 exit_reason 分布
stop/target 同 bar collision 比例
```

### 11.2 预测与排序指标

```text
Q_long / Q_short MAE
收益方向 balanced accuracy
P(net_pnl > 0) calibration
score decile -> realized expectancy
Top 5% / 10% / 20% entry expectancy
不同 threshold 下的 coverage / expectancy 曲线
```

高置信度分组的真实 expectancy 必须总体高于低置信度分组。若分数与收益不单调，
即使某个阈值偶然盈利，也不能判定模型掌握了入场能力。

### 11.3 可执行策略指标

```text
net expectancy per trade
aggregate net pnl
return_pct
profit_factor
win_rate
max_drawdown
VaR / CVaR
trades per day distribution
no-trade day ratio
LONG / SHORT 分方向收益
盈利月份比例
最大单笔盈利占总盈利比例
相对于随机入场的超额收益
canonical 与 validation 选参结果
参数敏感性稳定性
时间特征消融结果
```

### 11.4 后续持仓研究指标

虽然第一版退出固定，仍需保存：

```text
MFE
MAE
最终净收益 / MFE
从 MFE 到退出的利润回吐
```

这些指标用于入场能力通过后，判断移动保本、利润回吐保护和学习型平仓的价值。

## 12. ATR 和其他新特征如何评估

当前阶段不直接假设 ATR 能提高效果，而是把它作为可证伪的 feature ablation。

严格顺序：

```text
Experiment A：当前 18 维 observation + 固定绝对距离退出
Experiment B：A + 因果 ATR observation
Experiment C：B + ATR 尺度退出
Experiment D：C + 其他单一候选特征
```

每次实验只改变一个变量，并复用：

```text
相同原始行情
相同 split manifest
相同成交成本
相同评估脚本
相同基线
```

ATR 的可能价值是提供波动尺度：

```text
当前止损距离相对正常波动是否过小；
当前机会的预期幅度是否足以覆盖成本；
同样的价格移动在不同波动状态下是否具有不同意义。
```

ATR 本身不能证明行情即将结束。行情衰竭、利润回吐和智能退出属于入场门禁通过后的
Position Manager 阶段。

## 13. 验收门禁

### Gate 0：Evaluator 正确性

必须全部通过：

```text
1. 修改 decision_timestamp 之后的未来数据，不改变 observation。
2. 修改 entry_timestamp 之后的数据，只能改变 outcome，不能改变输入。
3. 同一固定交易的成本和净收益与环境 ledger 一致。
4. stop/target 同 bar 时严格 stop-first。
5. 不产生跨交易日持仓。
6. `holding_bars` 只按有效 1m bar 计数，休市和 `mask_t=0` 不推进计数。
7. `allow_entry_across_break=false` 时，所有跨休市或跨 segment 候选均被排除。
8. `stop_distance_price` 与 `stop_distance_ticks * tick_size` 不一致时 fail fast。
9. split purge 和 embargo 无跨边界标签。
10. 同一配置重复运行结果完全一致。
```

### Gate 1：机会存在性

受约束 oracle 必须证明固定退出规则下存在足够的正收益机会，并且机会不是只集中在
单一月份或少数交易中。

若 Gate 1 失败：

```text
调整固定退出参数；
扩大数据范围；
重新检查成交成本和品种；
不进入监督模型和 Dreamer。
```

### Gate 2：当前 Observation 入场可预测性

监督模型必须满足：

```text
1. 测试集 aggregate net expectancy > 0。
2. 明显优于等交易数量随机基线。
3. Top-score 组 expectancy 高于低分组，并具有总体单调关系。
4. 至少 2/3 walk-forward test fold 为正。
5. 不能依赖单一月份或单笔大盈利。
6. 测试集完整交易不少于 30 笔，否则证据不足。
7. 不再每天固定开满最大交易次数。
8. 存在合理比例的 0 次交易日。
```

Gate 2 使用三档结论：

```text
PASS:
  满足全部基础条件；
  按交易日 bootstrap 的 expectancy 95% CI 下界 > 0。

PROMISING:
  测试集均值 > 0；
  至少 2/3 folds 为正；
  分数分位与真实 expectancy 总体单调；
  但 95% CI 下界 <= 0，或交易数量不足以形成强证据。

FAIL:
  测试集均值 <= 0；
  或收益集中于单一月份 / 少数大盈利；
  或分数与真实 expectancy 不单调；
  或结果主要由时间特征偏差解释。
```

`PROMISING` 用于指导扩大数据或后续特征研究，但不等同于 Gate 2 通过。只有 `PASS`
允许继续 Gate 3 和 Entry-only Dreamer。

### Gate 3：成本鲁棒性

通过 Gate 2 后，再执行：

```text
标准成本
1.5 倍成本
2.0 倍成本
额外滑点
```

标准成本必须为正；成本上升后允许退化，但不能立即完全失效。

Gate 3 报告还必须展示 canonical 参数、validation 选参参数及其相邻参数组合。只有单一
组合通过而相邻组合全部失败时，Gate 3 判定为脆弱，不进入 Entry-only Dreamer。

### Gate 4：允许进入 Entry-only Dreamer

只有 Gate 0 至 Gate 3 通过，才实现和训练 Entry-only Dreamer。

## 14. Entry-only Dreamer 目标模式

Entry-only 环境中：

```text
空仓时：Agent 决策 SHORT / FLAT / LONG
入场后：Agent 不再负责退出
退出：由固定 Position Manager 执行
反手：禁止
加仓：禁止
```

第一版 reward 使用交易关闭后的最终净收益：

```text
空仓选择 FLAT：0
开仓时：0
持仓期间：0
交易关闭时：扣除全部成本后的最终净 PnL
```

必须验证：

```text
episode_reward_sum == final_equity - initial_equity
```

允许使用归一化后的终局净收益训练，但日志和验收必须始终保留真实货币净收益。

Entry-only Dreamer 必须使用与离线 evaluator 相同的：

```text
成交时序
成本模型
固定止损止盈
最大持仓
EOD 处理
```

由于交易关闭时才产生最终净收益，Entry-only reward 非常稀疏。若监督模型已经通过
Gate 2，Entry-only Dreamer 第一版必须使用监督 teacher 进行 behavior cloning warm
start：

```text
teacher 输入：与 Dreamer 相同的 observation；
teacher 输出：SHORT / FLAT / LONG action 及 calibrated score；
BC 阶段：先复现 teacher 的选择性入场行为；
RL 阶段：在固定退出和终局净收益 reward 下 fine-tune；
验收：RL 不得退化为每天固定开满最大交易次数。
```

BC 数据只能由 train split 生成；validation 用于选择 BC 和 fine-tune 参数；test
仍然只允许最终评估。

如果 Entry-only Dreamer 无法复现监督模型已证明存在的入场优势，重点排查 replay、
reward、world model 和 actor 训练，不继续添加更多指标掩盖问题。

## 15. 实施产物

建议新增：

```text
configs/entry_eval_jm_v1.yaml
src/gym_trading_env/research/entry_outcome.py
src/gym_trading_env/research/entry_evaluator.py
src/gym_trading_env/research/entry_dataset.py
src/gym_trading_env/research/entry_strategy_simulator.py
tools/build_entry_dataset.py
tools/train_entry_baseline.py
tools/report_entry_capability.py
tests/unit/test_entry_outcome.py
tests/unit/test_entry_cost_parity.py
tests/unit/test_entry_intrabar_collision.py
tests/unit/test_entry_causality.py
tests/unit/test_entry_split_purge.py
tests/integration/test_entry_evaluator_env_parity.py
```

输出目录结构：

```text
artifacts/entry_eval/<dataset_version>/
  manifest.json
  config.yaml
  split_manifest.json
  candidates.parquet
  outcomes.parquet
  baseline_predictions.parquet
  strategy_trades.parquet
  opportunity_report.json
  model_report.json
  strategy_report.json
  report.md
```

`manifest.json` 至少记录：

```text
git commit
数据文件 hash
数据时间范围
产品配置 hash
observation schema
exit policy version
成本参数
split manifest hash
模型参数
随机 seed
```

## 16. 实施顺序

### Phase 0：冻结实验契约

```text
冻结数据、成本、成交时序、canonical 退出策略和 split manifest；
定义 EntryOutcome / EntryCandidate schema；
定义主指标和门禁。
```

### Phase 1：实现 Counterfactual Entry Evaluator

```text
生成每个合法候选时刻的 LONG / SHORT outcome；
实现 stop-first、next-open、EOD 和成本计算；
补齐因果、成本和环境 parity 测试。
```

### Phase 2：生成 Opportunity 报告

```text
统计固定退出规则下的机会分布；
运行受约束 oracle；
运行随机、trade-count-matched random、固定方向和简单规则基线；
输出 canonical 与参数敏感性报告；
确认 Gate 1。
```

### Phase 3：训练监督入场基线

```text
训练 Ridge / Logistic / LightGBM 或 XGBoost；
验证预测分数与真实 expectancy 的单调性；
运行时间特征消融；
运行非重叠策略模拟；
确认 Gate 2 和 Gate 3。
```

### Phase 4：单特征 Ablation

```text
首先评估因果 ATR；
再评估其他单一特征；
每次只改变一个变量。
```

### Phase 5：Entry-only Dreamer

```text
固定退出；
终局净收益 reward；
监督 teacher behavior cloning warm start；
禁止反手和学习型退出；
与监督模型和随机基线比较。
```

### Phase 6：Position Manager

只有 Entry-only Dreamer 样本外通过后，才开始评估：

```text
移动保本
MFE / MAE observation
利润回吐保护
ATR trailing
学习型 HOLD / EXIT
```

## 17. 最终决策树

```text
Counterfactual evaluator 是否正确？
  否 -> 修 evaluator，不训练模型
  是
  |
固定退出下受约束 oracle 是否有足够机会？
  否 -> 调整退出规则、数据或产品
  是
  |
当前 observation 的监督模型是否样本外正期望？
  否 -> observation 信息不足，做单特征 ablation
  是
  |
Entry-only Dreamer 是否复现监督模型优势？
  否 -> 修 RL 训练链路
  是
  |
再研究 Position Manager 和智能退出
```

本方案的第一交付物不是新的 Dreamer checkpoint，而是一份可复现、可审计的入场能力
报告。只有报告证明当前 observation 可以筛选正期望入场点，才允许继续扩大 RL 任务。

## 18. Phase 0 - Phase 3 实施状态

日期：2026-06-13

当前分支：`feat/entry-capability-evaluator`

已实现范围严格限定为：

```text
Phase 0：冻结 entry_eval_jm_v1 实验配置和数据契约
Phase 1：Counterfactual Entry Evaluator
Phase 2：Opportunity、oracle 和基线报告
Phase 3：Ridge window-summary 监督入场基线、时间特征消融和非重叠策略报告
```

未实现：

```text
Entry-only Dreamer
ATR 或其他新 observation
动态 5m MACD
智能退出或 Position Manager
```

实现入口：

```bash
/home/v/miniconda3/envs/forex/bin/python tools/run_entry_capability.py \
  --config configs/entry_eval_jm_v1.yaml \
  --output artifacts/entry_eval/entry_eval_jm_v1
```

开发 smoke 可追加 `--skip-sensitivity`；正式报告不得追加该参数。

主要输出：

```text
candidates.csv
features.npz
opportunity_report.json
oracle_trades.csv
baseline_predictions.csv
model_report.json
strategy_trades.csv
strategy_report.json
parameter_sensitivity.csv
split_manifest.json
manifest.json
report.md
```

当前 JM2601 canonical 实测结论：

```text
候选点：41141
交易日：122
受约束 oracle：366 笔，expectancy 442.69，Gate 1 PASS
27/27 固定退出参数组合的受约束 oracle 净收益为正
Ridge window-summary 非重叠样本外策略：54 笔，net PnL -4014，expectancy -74.33
Ridge Gate 2：FAIL
Gate 3：BLOCKED_BY_GATE2
```

这里的 `Ridge FAIL` 只表示当前冻结的 Ridge window-summary 监督基线没有通过 Gate 2。
该基线输入是 `60x18 market_seq` 的 `latest / mean / std / delta` 汇总，不能解释为完整
时序 observation 或所有监督模型必然失败。按照本方案门禁，当前不允许进入
Entry-only Dreamer。

验证命令：

```bash
/home/v/miniconda3/envs/forex/bin/python -m pytest tests/unit -q
/home/v/miniconda3/envs/forex/bin/python -m pytest \
  tests/integration/test_entry_evaluator_ledger_parity.py -q
```

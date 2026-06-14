# DreamerV3 Observation / Action 盈利能力评估报告

日期：2026-06-12

## 1. 评估目标

本报告评估以下问题：

> 基于当前环境提供的 observation 和 action，DreamerV3 是否有可能根据当前行情以超过
> 50% 的概率判断正确，并通过交易获得稳定盈利？

这里必须区分三个不同目标：

1. **方向准确率超过 50%**：预测未来价格上涨或下跌是否正确。
2. **单笔交易具有正期望**：正确时的盈利能够覆盖错误损失、点差和手续费。
3. **策略稳定盈利**：经过入场、持仓、平仓、仓位约束和市场状态变化后，样本外净利润仍为正。

方向准确率超过 50% 不等于策略可以盈利。策略也可能在低于 50% 的胜率下，通过更大的平均盈利实现正期望。

## 2. 当前环境契约

### 2.1 Observation

当前 `obs_feature_mode: obs` 配置提供：

```text
market_seq:  (60, 18)
agent_state: (14,)
action_mask: (3,)
```

`market_seq` 包含最近 60 根 1 分钟行情的 18 个特征：

```text
volume surprise
open interest
cumulative VWAP
price versus VWAP
session high / low / range
session open drift
current bar direction
minute index / session phase / weekday
rolling volatility
open-interest change
price change from reference close
market-data-valid mask
```

这些行情特征的计算目前没有发现直接使用未来行情的问题。大部分特征用于描述当前日内状态、趋势位置和波动环境。

`agent_state` 包含：

```text
market open status
open / close availability
current position side
remaining daily entries
minutes to EOD
holding duration
unrealized / realized PnL
equity / drawdown
previous action result
```

这些特征主要帮助 Dreamer 管理仓位、账户状态和执行约束，本身不直接提供未来行情方向。

### 2.2 Action

当前动作空间保持为三个目标仓位：

```text
0 = SHORT
1 = FLAT
2 = LONG
```

动作代表目标仓位，而不是单纯的涨跌预测：

- 空仓时选择 `LONG` 或 `SHORT` 会开仓；
- 持仓时保持相同目标代表继续持仓；
- 选择 `FLAT` 代表平仓；
- 选择相反仓位代表反手；
- `action_mask` 限制当前状态下可以到达的目标仓位。

因此 Dreamer 实际需要同时学习方向、入场、持有和平仓，而不是完成一个简单的三分类任务。

## 3. 成本与盈利门槛

当前 JM 配置：

```text
spread = 0.5
trading_fee_per_lot = 3
is_round_turn = true
lot_size = 60
trade_lot = 1
```

一笔完整开平仓的近似成本：

```text
开仓点差 0.5
+ 平仓点差 0.5
+ 手续费现金 6 / 合约乘数 60
= 1.1 个价格点
```

即使模型正确判断未来上涨，如果有利价格移动小于约 `1.1` 点，完整交易仍可能亏损。

盈利要求实际是：

```text
方向判断 + 行情幅度判断 + 入场时机 + 持有时间 + 平仓时机
```

而不是仅要求方向正确率超过 50%。

## 4. 可预测性诊断

### 4.1 数据与方法

使用数据：

```text
合约：DCE JM2601
周期：1 分钟
范围：2024-06-02 至 2024-12-31
有效分钟：48,840
交易日：144
```

诊断方法：

1. 仅使用当前环境的 18 个 `market_seq` 特征；
2. 从最近 60 根行情构造末值、均值、标准差和变化量；
3. 使用线性 Ridge 模型，避免依赖复杂模型获得虚假优势；
4. 严格按照交易日先后顺序训练和测试；
5. 使用三个 walk-forward 样本外区间；
6. 使用验证区间选择交易阈值；
7. 每日最多选择 3 笔交易；
8. 每笔交易扣除 `1.1` 点完整开平成本。

该诊断用于判断当前 observation 是否存在基础可预测信号。它不是 DreamerV3 的完整策略回测，也没有模拟动态持仓和平仓决策。

### 4.2 Walk-forward 结果

| 预测周期 | 样本外方向准确率 | 样本外 balanced accuracy | 盈利区间 | 扣成本总结果 |
|---|---:|---:|---:|---:|
| 1 分钟 | 53.6% | 53.6% | 1 / 3 | -70.9 点 |
| 5 分钟 | 52.1% | 52.1% | 0 / 3 | -116.5 点 |
| 15 分钟 | 52.4% | 52.5% | 1 / 3 | -2.4 点 |
| 30 分钟 | 53.7% | 53.9% | 1 / 3 | -42.7 点 |
| 60 分钟 | 54.8% | 55.2% | 1 / 3 | -143.7 点 |

结论：

- 当前 observation **确实包含轻微、可检测的方向预测信号**；
- 在简单模型下，样本外方向准确率可以超过 50%；
- 但该优势暂时不能稳定覆盖点差和手续费；
- 所有预测周期的三个样本外区间合计结果均为负；
- 没有任何周期在三个测试区间中实现超过一个区间盈利。

对于较长周期，测试数据存在明显方向不平衡。仅使用多数方向也可能获得较高原始准确率，因此必须同时检查 balanced accuracy、净收益和不同市场阶段表现，不能只看 accuracy。

## 5. 昨晚训练结果能够证明什么

检查日志：

```text
/data/logdir/action-mask-overnight-20260611/
```

该训练已经证明：

```text
availability prior accuracy 接近 99.8%
invalid_action_total = 0
action_rejected = 0
execution_failed = 0
chosen_action_was_valid = 1
```

但该训练不能用于判断正式 actor 是否盈利，因为整个训练期间：

```text
train/loss/policy = 0
train/loss/value = 0
agent.avail_actor_enabled = false
```

该运行只训练了 world model 和 availability 相关能力，actor 没有进行正式策略优化。

同时，该日志中的训练期环境统计整体为负：

```text
965 个统计窗口中，仅 70 个平均 return_pct 为正
平均 return_pct 约 -1.86%
最后 100 个窗口平均 return_pct 约 -2.21%
平均 profit factor 约 0.864
平均 expectancy 约 -59
```

这些负收益主要反映 warm-up 阶段行为，不应被解释为正式 masked actor 的最终能力；但它们也不能提供任何盈利证据。

## 6. 当前关键风险

### 6.1 训练与评估没有真正的数据隔离

`action_mask_formal` 当前默认使用：

```text
script: train
env.gymnasium.config_path: data/trading_stage1.yaml
```

`script: train` 只创建训练环境。虽然配置中存在 `eval_envs`，但该运行入口不会自动创建独立时间范围的评估环境。

训练 episode 又会从同一份六个月数据随机选择起点。因此训练期或日志中的盈利结果不能证明样本外泛化能力。

### 6.2 当前成交时点可能过于乐观

当前 observation 已经包含第 `t` 根 K 线完整信息，包括 `Close[t]`、该分钟成交量和最高最低价衍生状态。

随后 action 仍按第 `t` 根 K 线的 `Close[t]` 成交，再从第 `t+1` 根 K 线获得收益变化。

回测内部的时序没有直接读取未来数据，但真实交易通常无法做到：

```text
看到完整 Close[t] 后，无延迟地按同一个 Close[t] 成交
```

正式盈利验证应至少支持：

- 下一根 K 线开盘成交；或
- 明确的决策延迟；或
- 额外滑点和成交不确定性。

否则回测结果可能高估真实收益。

### 6.3 Reward 不完全等价于最终净利润

当前 `FuturesIntradayReward` 除净值变化外，还包含：

```text
alpha_unrealized = 0.5 的未实现收益折扣
盈利平仓 shaping
ATR 平仓 shaping
EOD 持仓惩罚
```

Dreamer 优化的是 shaped reward。该 reward 可能有助于训练，但不能直接替代真实账户净利润作为验收指标。

必须同时记录：

```text
episode shaped reward
真实净利润
return_pct
profit factor
expectancy
max drawdown
交易次数与胜率
```

### 6.4 数据规模和市场覆盖不足

当前正式配置只使用约 144 个交易日、单一合约的数据。

对于约 50M 参数的 DreamerV3，该数据规模较小，随机重复采样容易记忆特定市场阶段。当前数据也不足以覆盖更多趋势、震荡、极端波动和合约切换状态。

## 7. 当前判断

### 7.1 是否能够超过 50% 判断正确

**有可能。**

简单线性模型已经在严格时间切分下获得约 `52.1%~54.8%` 的样本外方向准确率，说明当前 observation 不是完全没有预测信息。

### 7.2 是否能够依靠当前设计稳定盈利

**当前没有证据。**

已有基础信号尚不足以稳定覆盖交易成本。DreamerV3 可能通过动态持有和平仓获得比固定周期线性基准更好的结果，但也可能因为模型复杂度、reward shaping 和数据不足产生更严重的过拟合。

当前对“现有 observation/action 链路能够训练出稳定样本外盈利策略”的主观把握约为：

```text
20% ~ 30%
```

该判断不是对 DreamerV3 算法本身的否定，而是对当前数据规模、成交模型和验证体系的综合评估。

## 8. 正式盈利验证优先级

### P0：必须完成

1. 按日期建立完全隔离的 `train / validation / test` 数据集；
2. 正式训练过程中不得从 validation/test 数据采样；
3. 增加下一根 K 线成交或可配置执行延迟；
4. 样本外验收以真实净利润为准，而不是方向准确率或 shaped reward；
5. 对正式 actor 运行 walk-forward 和 Monte Carlo 验证。

### P1：强烈建议

1. 同时记录 shaped reward 与真实净利润，检查二者相关性；
2. 建立无交易、随机策略、固定多头、固定空头和简单线性策略基准；
3. 报告按月份、趋势/震荡阶段和波动率分组后的结果；
4. 检查策略是否只是消耗每日 3 次开仓额度；
5. 增加手续费、点差和额外滑点敏感性测试。

### P2：根据 P0/P1 结果决定

1. 增加日线或更长期市场状态；
2. 增加多个年份、多个合约或连续合约数据；
3. 加入更有预测价值的成交量、持仓变化或盘口信息；
4. 调整 reward，使其与最终净利润和风险目标更一致。

## 9. 正式训练验收门槛

正式训练可以开始，但不能仅凭训练集收益宣布成功。建议设置以下样本外门槛：

```text
所有 test 数据严格未参与训练和阈值选择
扣除手续费、点差和额外滑点后净利润为正
profit_factor > 1
expectancy > 0
多个 walk-forward 区间中大多数区间盈利
收益不能由极少数异常交易贡献
最大回撤处于可接受范围
交易活性正常，不是永远 FLAT
结果显著优于简单基准策略
```

方向准确率可以作为诊断指标，但不应成为正式训练或实盘准入门槛。

## 10. 最终结论

当前 observation 对行情方向具有轻微预测能力，超过 50% 的样本外方向判断在技术上可实现。

但目前最关键的事实是：

> 当前可预测优势尚未稳定覆盖交易成本，现有训练和评估流程也无法证明样本外盈利。

下一阶段的重点不应是继续追求更高的训练集方向准确率，而应首先建立严格的数据隔离、现实成交模型和以净利润为核心的样本外验收体系。

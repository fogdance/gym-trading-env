# 陈大哥交易系统接入 Gym Trading Env + DreamerV3 实施方案

日期：2026-06-12

状态：设计评审修订稿，暂不修改运行代码

本轮评审后冻结的关键调整：

```text
Phase 1 拆为因果基础 1A 与最小 A/D 识别器 1B
A 类改为硬门槛 + 软确认分数
新增确定性 ExitPlan
1h 关键位拆分 confirmed/provisional 生命周期
teacher 增加最少交易数、分组报告、机会覆盖率和盈利集中度门禁
Dreamer 推迟到 teacher、人工抽检和监督评级器全部通过之后
```

输入材料：

- `/home/v/Downloads/ChatGPT-陈大哥交易系统行情识别.html`
- 当前仓库：`~/Documents/work/gym-trading-env`
- DreamerV3：`~/Documents/work/dreamerv3`
- 当前盈利能力评估：`docs/obs_action_profitability_assessment.md`

## 1. 结论

这套交易系统不能通过“增加几个指标，再继续让 Dreamer 自由选择
SHORT / FLAT / LONG”完整落地。

系统的核心不是预测下一根 K 线，而是分层过滤：

```text
日线方向
  -> 1h 结构与关键位
  -> 路线图和失效位
  -> 5m MACD 确认
  -> 1m 量仓价与价格停滞确认
  -> 结构止损、目标空间和盈亏比检查
  -> A/B/C/D 起爆点评级
  -> 候选点上的入场、持仓和退出
```

当前项目主要只有最近 60 根 1m 特征。它缺少日线方向、1h 结构、关键位距离、
5m MACD、结构止损、目标空间和候选点评级。Dreamer 当前需要从低层 1m 数据中
同时学习行情识别、入场、持仓、退出和风控，任务定义过宽，这也是策略容易退化为
永远 FLAT 的重要原因。

推荐目标架构：

```text
因果多周期特征
  -> 最小行情识别规则引擎
  -> 结构化 EntryPlan + ExitPlan
  -> 规则 teacher 回测与人工抽检
  -> 监督学习起爆点评级器
  -> 最后才允许 Dreamer 在系统内候选点学习参与、持有和提前退出
```

实施时保留当前三个目标仓位动作：

```text
0 = SHORT
1 = FLAT
2 = LONG
```

第一阶段不扩展为 5 个动作，也不让 actor 自己输出止损价。结构止损和目标位由
当前行情对应的 `EntryPlan` 提供；保本、移动保护、路线失效和强制退出由
`ExitPlan` 提供，环境负责执行保护。

第一轮实现必须刻意收窄。目标不是建设完整的主观行情识别平台，也不是立即盈利，
而是回答一个可证伪的问题：

> 能否因果、稳定地识别“焦煤日线下跌中，等价格反弹到 1h 阻力并失败后再做空”？

## 2. 对“真实可以盈利”的工程判断

HTML 记录了完整的方法、正确题、错题和实盘经验，但它不是可直接验证的审计账单，
也不是已经完成时间隔离的量化回测。

因此工程上采用以下边界：

1. 把这套系统视为高价值交易先验和候选策略。
2. 不在编码前直接假定规则翻译后一定盈利。
3. 必须先构建确定性规则基线，并在严格样本外、扣除成本后验证。
4. 规则基线没有样本外正期望前，不允许用 reward shaping 强行让 RL 多交易。
5. Dreamer 的职责是改进候选点选择和持仓管理，不是证明原始规则必然盈利。

## 3. 交易系统的可编程核心

### 3.1 总纲

```text
顺日线趋势；
等关键位置；
看量仓价合力；
用 5m MACD 确认；
用结构小止损试错；
入场后应快速脱离成本；
做对后按结构移动保护；
连续失败停止。
```

### 3.2 当前优先模式

第一版只实现最明确、最容易验证的顺趋势反弹空，并将条件拆成硬门槛与软确认。

做空：

```text
硬门槛：
  日线下跌
  + 1h 反弹到最近 swing high 或原支撑变阻力
  + 不是低位追空
  + 结构高点上方可放小止损
  + 下方目标空间和 RR 足够

软确认：
  高位放量但涨不动
  + 持仓量异常
  + 5m 死叉
  + price no progress / failed break
```

后续做多为镜像，但不进入第一版：

```text
日线上涨
+ 1h 回调到前低、支撑或原阻力变支撑
+ 低位放量但跌不动
+ 持仓量异常
+ 5m 金叉
+ 结构低点下方可放小止损
+ 上方目标空间足够
```

### 3.3 必须优先识别的 D 类禁止交易

```text
方向不明
逆日线趋势
非关键位或悬空位置
下跌后追空
下跌趋势低位放量做空
上涨后追多
上涨趋势高位放量追多
结构止损不存在
止损距离超过品种上限
目标空间不足或盈亏比不足
连续失败后继续开仓
```

第一版的主要价值应是稳定拦截 D 类，而不是追求每天产生交易。

第一版运行时只输出 `A / D`。同时保留通过硬条件但软确认不足的
`near_a_candidate` 分数和数量，用于衡量候选稀疏程度；它不是可交易的 B 类评级。

## 4. 当前项目缺口

| 系统要求 | 当前实现 | 结论 |
|---|---|---|
| 日线趋势 | 可选的 7 日日线 close/volume/OI，目前关闭 | 信息不足，且没有趋势状态 |
| 1h 结构与关键位 | 没有 1h 序列和因果结构识别 | 缺失 |
| 15m/30m/4h 路线图 | 没有多周期结构与目标投影 | 缺失 |
| 5m MACD | 未进入正式 observation | 缺失 |
| 分时量仓价 | 1m volume surprise、OI、VWAP 已存在 | 可复用，但需要扩展 |
| 品种级放量/OI 阈值 | 没有品种参数注册表 | 缺失 |
| 结构止损 | 当前 JM 配置关闭 stop loss | 不符合系统 |
| 目标位与盈亏比 | 没有动态目标和 RR | 缺失 |
| A/B/C/D 评级 | 没有 | 缺失 |
| 路线图失效位 | 没有 | 缺失 |
| 连续失败停止 | 仅有每日开仓数量约束 | 不完整 |
| 样本外验证 | `script: train` 没有独立 eval 数据 | 不合格 |
| 成交时序 | 观察完整 Close[t] 后仍按 Close[t] 成交 | 过于乐观 |

当前 `FeatureEngineer.compute_zigzag_features()` 不得用于正式标签或 observation。
它在完整价格序列上调用 `peak_valley_pivots()`，历史 pivot 可能依赖未来价格确认，
会导致未来信息泄漏。

## 5. 目标架构

```text
Raw OHLCVI 1m
  |
  +-- CausalMultiTimeframeBuilder
  |     +-- 1m closed bars
  |     +-- 5m closed bars
  |     +-- 15m/30m/1h/4h closed bars
  |     +-- daily closed bars
  |
  +-- MarketRecognitionEngine
  |     +-- DailyTrendRecognizer
  |     +-- StructureRecognizer
  |     +-- RouteMapRecognizer
  |     +-- MacdRecognizer
  |     +-- TapeRecognizer
  |     +-- SetupGrader
  |     +-- EntryPlanBuilder
  |     +-- ExitPlanBuilder
  |
  +-- Env observations and logs
  |     +-- recognition_state
  |     +-- entry_gate
  |     +-- setup_mask
  |     +-- setup_grade
  |     +-- setup_reason_bits
  |     +-- action_mask
  |
  +-- Dreamer
        +-- recognition auxiliary heads
        +-- setup-mask prediction head
        +-- behavior-cloning warm start
        +-- masked actor/value fine-tuning
```

## 6. 两种 Mask 必须分开

### 6.1 `action_mask`：执行合法性

现有 `action_mask` 继续只表达：

> 当前目标仓位是否能被环境机械执行。

它覆盖资金、仓位、每日开仓次数、闭市、near-EOD 和反手可达性。

### 6.2 `entry_gate`：原始入场资格

新增：

```text
entry_gate = [short_entry_allowed, long_entry_allowed]
```

它表达：

> 根据陈大哥系统，当前是否存在允许新开或反手进入该方向的合格 setup。

`entry_gate` 只描述新入场机会，不负责持仓保持、平仓或机械执行安全。

### 6.3 `setup_mask`：最终 transition-aware 策略资格

环境根据 `entry_gate` 和当前持仓生成最终三动作策略 mask：

```text
setup_mask = expand_entry_gate(entry_gate, current_position)
```

组合规则：

1. `FLAT` 始终保留，只要执行层允许平仓或保持空仓。
2. 保持当前已有仓位不需要新的 setup。
3. 从 FLAT 新开 LONG/SHORT，需要对应 setup。
4. 反手包含新开仓，需要目标方向 setup。
5. D 类 setup 永远不允许开仓。
6. 第一版只允许 A 类；只有 B 类经过独立样本外验证后才允许配置启用。

replay 保存最终 `setup_mask`，Dreamer 的 `setup_mask_head` 直接预测这个
transition-aware 三动作 mask。imagination 阶段不在 policy 外部重建当前持仓。

### 6.4 最终 policy mask

最终 actor 使用：

```text
policy_mask = action_mask AND setup_mask
```

`action_mask` 和 `setup_mask` 都必须 strict nonempty。真实 mask 不允许 fallback；
只有 predicted mask 允许 fallback，并必须记录 fallback rate。

## 7. 识别数据契约

新增不可变快照：

```python
MarketRecognitionSnapshot(
    timestamp,
    product,
    contract,
    source_version,
    daily_state,                 # DOWN / NOT_DOWN
    resistance_level_id,
    resistance_level_type,
    resistance_price,
    resistance_is_provisional,
    resistance_created_at,
    resistance_confirmed_at,
    resistance_invalidated_at,
    resistance_distance_ticks,
    at_1h_resistance,
    macd_5m_state,
    low_position_chase_short,
    high_volume_no_up_progress_score,
    oi_abnormal_score,
    price_no_progress_score,
    hard_gate_results,
    soft_score,
    short_grade,
    reason_bits,
    short_entry_plan,
    short_exit_plan,
)
```

这是第一版最小契约，只服务于单方向反弹空。完整路线图、双方向计划和高级 tape
状态不能提前塞入该契约；后续通过版本化字段扩展。

`EntryPlan`：

```python
EntryPlan(
    plan_id,
    direction,
    grade,
    entry_zone_low,
    entry_zone_high,
    structural_stop,
    stop_distance_ticks,
    target_price,
    target_distance_ticks,
    reward_risk_ratio,
    invalidation_price,
    reason_bits,
    source_version,
)
```

`ExitPlan`：

```python
ExitPlan(
    plan_id,
    initial_stop,
    initial_target,
    breakeven_trigger_r,
    trail_mode,
    trail_reference,
    partial_take_profit,
    structure_invalidation,
    route_invalidation,
    force_exit_reasons,
    max_hold_bars,
    source_version,
)
```

Allowed decision 必须携带 execution-ready `EntryPlan` 和对应 `ExitPlan`。开仓提交时，
两个计划与仓位原子绑定；`step()` 不得重新计算另一份止损、目标或退出规则。

`plan_id` 必须由识别快照版本、规则版本、方向和关键位 ID 确定性生成，保证离线回测、
环境回放和训练日志可以关联同一份计划。

第一版 `ExitPlan` 必须是确定性的：

```text
结构止损触发 -> 强制退出
达到保本触发阈值 -> 按计划移动保护
结构/路线失效 -> 强制退出
达到最大持有 bars -> 强制退出
actor 可以提前退出，但不能绕过保护性退出
partial_take_profit = disabled，直到环境明确支持部分仓位
route_invalidation = disabled，直到完整路线图阶段
```

## 8. 因果多周期特征

### 8.1 时间对齐硬约束

在 1m 时刻 `t` 做决策时：

1. 可以使用已经完整关闭的 1m bar `t`。
2. 只能使用已经完整关闭的 5m/15m/1h/4h bar。
3. 未关闭的高周期 bar 不得进入正式特征；如需要，必须显式标记为 partial。
4. 日线只能使用前一完整交易日及更早数据。
5. 正式成交至少使用下一根 bar 的 open，或配置明确的延迟和滑点。

### 8.2 日线趋势状态

完整版本可以输出：

```text
UP
DOWN
RANGE
UP_DAMAGED
DOWN_DAMAGED
REVERSAL_UP_CONFIRMED
REVERSAL_DOWN_CONFIRMED
```

使用因果确认的 swing、前高前低、ATR 和收盘状态。不要用全序列 ZigZag。

第一版只输出：

```text
DOWN
NOT_DOWN
```

只服务于焦煤下跌趋势反弹空，不在第一轮同时解决完整趋势阶段分类。

建议初始规则：

```text
UP:
  最近两个已确认 swing high 抬高
  且最近两个已确认 swing low 抬高

DOWN:
  最近两个已确认 swing low 降低
  且最近两个已确认 swing high 降低

DAMAGED:
  原趋势关键 swing 被收盘破坏，但反向 lower-high/lower-low
  或 higher-high/higher-low 尚未完整确认
```

规则阈值必须按产品和 ATR 归一化，不使用固定百分比套所有品种。

### 8.3 1h 结构与关键位

第一版候选关键位只包含：

```text
最近 1h swing high
跌破后的旧支撑
最近下方因果支撑，仅用于目标与 RR
```

每个关键位必须包含：

```text
level_id
level_type
price
strength
created_at
confirmed_at
invalidated_at
is_provisional
source_timeframe
touch_count
distance_ticks
distance_atr
```

关键位必须在当时可知。未来触碰次数不能回填给历史 observation。

关键位分为两类：

```text
confirmed_level:
  确认较慢但可靠，用于强关键位、离线标签和验证。

provisional_level:
  只使用当时可见的局部高低点，可实时使用，也允许后续失效。
```

任何 observation、规则判断和计划都必须保存当时使用的 `level_id` 与生命周期状态，
不能在 level 后续确认后回填历史判断。

### 8.4 路线图

第一版不实现完整路线图。只使用最近下方因果支撑或固定 `kR` 形成保守目标，
用于计算 RR 和构建确定性退出。

后续版本再实现可验证的结构投影：

```text
结构幅度投影
M 头/W 底高度投影
支撑跌破后反抽确认
阻力突破后回踩确认
方向切换和失效位
```

路线图只提供候选目标、压力、支撑和失效位，不直接产生交易动作。

### 8.5 5m MACD

输出：

```text
ABOVE_ZERO_GOLDEN_CROSS
BELOW_ZERO_DEAD_CROSS
GOLDEN_CROSS
DEAD_CROSS
NO_CONFIRMATION
```

MACD 由已关闭 5m bar 计算。它是过滤器，不是独立入场信号。

### 8.6 1m 量仓价

第一版只在现有 volume surprise、OI、VWAP 基础上增加可复算的软评分输入：

```text
volume ratio: 1/3/5 分钟
OI delta: 1/3/5 分钟
price delta: 1/3/5 分钟
price progress per volume
price progress per OI change
distance to local high/low
high-volume but no-progress score
```

第一版只输出：

```text
high_volume_no_up_progress_score
oi_abnormal_score
price_no_progress_score
```

这些分数只能作为 A 类软确认，不能单独允许开仓。攻击、离场、转手等高级分类推迟。

### 8.7 低位追空的第一版定义

“低位”不能依赖肉眼或未来最低点。第一版使用当时可见信息构建硬禁止项：

```text
low_position_chase_short =
  price 接近最近因果 1h swing low / rolling low zone
  OR 最近下跌幅度超过 product.chase_short_atr
     且价格尚未反弹到有效 1h 阻力
```

所需窗口、ATR 距离和容忍区间进入品种参数注册表，只能在 train/validation 上确定。
每次禁止必须输出命中的子条件和所用 level ID。

## 9. 品种参数注册表

交易系统明确要求不同品种使用不同阈值。新增：

```yaml
products:
  DCE.JM:
    tick_size: 0.5
    normal_stop_ticks: ...
    max_stop_ticks: ...
    volume_baseline_window: ...
    oi_alert_quantile: ...
    key_level_tolerance_atr: ...
    chase_short_atr: ...
    low_zone_tolerance_atr: ...
    min_reward_risk: ...
    false_break_buffer_ticks: ...
```

禁止直接把笔记里的玻璃、纸浆、烧碱参数套到焦煤。

初始阈值应从训练区间的历史分布估计，并在 validation 区间冻结。

## 10. 第一版 A/D 评级

### 10.1 因果规则评级

运行时规则评级只能使用当前和过去信息。

做空硬条件：

```text
daily_trend == DOWN
at_1h_resistance == true
not_low_position == true
structural_stop_exists == true
stop_distance <= product.max_stop
reward_risk_ratio >= configured minimum
```

任一硬条件失败，评级直接为 `D / NO_TRADE`。

软确认项：

```text
high_volume_no_up_progress
oi_abnormal
macd_5m_state == DEAD_CROSS
price_no_progress
failed_break_or_rejection
```

每个软确认项产生独立 reason bit 和分数。第一版：

```text
硬条件全部通过且 soft_score >= A_THRESHOLD -> A
其他情况 -> D / NO_TRADE
```

软项权重和 `A_THRESHOLD` 必须在 train/validation 上确定并随规则版本冻结，禁止根据
最终 test 收益反向调节。

为了避免 A 类极度稀疏却无法被发现，必须额外记录：

```text
hard_gate_candidate_count
near_a_candidate_count
A_count_per_month
A_selection_rate_within_hard_gate_candidates
soft_score_distribution
```

`near_a_candidate` 仅用于训练、统计和人工抽检，不允许开仓，也不等同于正式 B 类。
后续数据足够后，才决定是否引入 B/C 运行时评级。

### 10.2 未来结果标签

未来结果只能用于离线训练和验收，绝不能进入 observation 或 mask。

每个方向候选点使用 triple-barrier/路径标签：

```text
stop barrier = EntryPlan.structural_stop
target barrier = EntryPlan.target or kR
time barrier = configured horizon

label:
  TARGET_FIRST
  STOP_FIRST
  TIMEOUT
```

同时记录：

```text
MFE
MAE
time_to_0.5R
time_to_1R
是否快速脱离成本
扣成本后 realized R
```

### 10.3 评级与机会覆盖指标

不能只看 A 类 precision。必须同时报告：

```text
A precision / recall
A 数量/月
A 交易按月份和合约分布
near-A 候选数量
D recall
D 误入率
候选机会覆盖率
盈利贡献集中度
```

第一版机会集合定义为所有通过硬条件的做空候选。机会覆盖率至少包含：

```text
A selection coverage =
  A 数量 / 硬条件候选数量

positive-opportunity capture（仅离线验收）=
  被 A 选中的 TARGET_FIRST 候选 / 全部 TARGET_FIRST 硬条件候选
```

第二项使用未来结果，只能用于离线验收，不能成为 observation、mask 或运行时规则。

### 10.4 人工标签

人工标注必须包含原因，不只包含 A/B/C/D：

```text
LOW_POSITION_CHASE_SHORT
NOT_AT_KEY_LEVEL
TREND_MISMATCH
HIGH_VOLUME_FAILED_BREAK
MACD_CONFIRMED
STOP_TOO_WIDE
INSUFFICIENT_TARGET_SPACE
```

## 11. Observation 设计

本节只描述 Phase 5 之后的环境接入目标，不属于第一轮离线识别器实现。

后续环境接入时新增：

```text
recognition_state: float32 vector
entry_gate:         float32 (2,)
setup_mask:        float32 (3,)
setup_grade:       float32 encoded grades
setup_reason_bits: float32/bool vector
```

保留：

```text
market_seq
agent_state
action_mask
```

建议同时启用并扩展：

```text
daily_seq
hourly_seq
five_minute_seq
```

不要只把多个周期的最终规则结果交给 Dreamer。Dreamer 需要低层序列来判断规则引擎
没有覆盖的差异，但高层识别状态必须显式提供，避免完全从零学习。

## 12. 环境接入

建议新增模块：

```text
src/gym_trading_env/recognition/contracts.py
src/gym_trading_env/recognition/timeframes.py
src/gym_trading_env/recognition/causal_swings.py
src/gym_trading_env/recognition/trend.py
src/gym_trading_env/recognition/structure.py
src/gym_trading_env/recognition/macd.py
src/gym_trading_env/recognition/setup_grader.py
src/gym_trading_env/recognition/engine.py
src/gym_trading_env/envs/entry_plan.py
src/gym_trading_env/envs/exit_plan.py
```

`route_map.py`、高级 `tape.py` 和完整 A/B/C/D grader 不进入第一轮实现。

修改边界：

```text
utils/market_store.py
  保存与每个 1m row 对齐的多周期和 recognition tensors

envs/trading_env.py
  observation 暴露 recognition_state/entry_gate/setup_mask
  _get_obs() 缓存当前 RecognitionSnapshot、EntryPlan 和 ExitPlan
  step() 原样消费对应计划

envs/target_transition.py
  继续只负责机械执行合法性
  不承载市场策略判断

envs/position.py
  保存结构止损、目标、EntryPlan、ExitPlan、plan id 和 setup grade
```

## 13. DreamerV3 接入

本章节不是第一轮实现范围。只有离线识别器、规则 teacher、人工抽检和监督评级器
全部通过各自门禁后，才允许修改 env 和 Dreamer。

### 13.1 不建议继续纯 RL 从零训练

当前正式训练已经证明：合法性学得很好，但 actor 会选择永远 FLAT。

这套系统本身强调绝大多数时刻应该等待，因此不能用“空仓惩罚”或“开仓奖励”
强行解决 FLAT。那会把模型推向过度交易，违背系统。

正确路径是：

1. 先完成离线规则识别器与 teacher 回测。
2. 人工抽检 A/D 与计划是否符合原系统。
3. 用 LightGBM/XGBoost 等可解释模型完成监督评级。
4. 再接入 env，并用系统 teacher 做行为克隆 warm start。
5. 最后用 RL 在候选点上优化是否参与、持有和提前退出。

### 13.2 新增监督 heads

建议增加：

```text
trend_head
structure_head
macd_head
setup_grade_head
setup_mask_head
```

监督目标来自 replay 中的因果规则标签和人工标签。

`setup_mask_head` 与 availability head 分离：

- availability head 学机械执行可达性；
- setup head 学最终 transition-aware 三动作策略资格。

二者都不接受 actor loss 反向梯度。

### 13.3 Imagination

真实 policy：

```text
true action_mask + true setup_mask
```

Imagination：

```text
predicted action_mask + predicted setup_mask
```

两个 hard mask 均需 stop-gradient，并且采样时使用的最终 mask 必须原样携带到
actor loss。当前已修复的 imagination mask 一致性机制可以复用。

### 13.4 行为克隆

新增 teacher policy 数据：

```text
A 类候选：按计划方向入场
D 类候选：FLAT
持仓后：按 ExitPlan 的结构止损/失效/移动保护生成 HOLD 或 FLAT
```

只用样本外验证通过的 teacher 生成训练目标。

行为克隆损失应先单独训练或高权重 warm-up，随后逐步降低权重：

```text
loss = world_model_loss
     + recognition_losses
     + bc_weight * behavior_cloning_loss
     + actor_critic_loss
```

### 13.5 Reward

不要增加全局“不交易惩罚”。

建议正式 reward 收敛为：

```text
以 R 为单位的净值变化
- 风险越界
- 超出结构止损后的异常损失
```

规则遵守优先通过 setup gate 和环境保护实现，不使用大量人工 bonus。

以下作为指标而不是 reward：

```text
A 类候选参与率
D 类误入率
机会捕获率
快速脱离成本率
趋势单持有长度
计划内退出率
```

## 14. 数据要求

当前正式数据只有约 6 个月 JM2601，不能支撑这套系统的完整验证。

最低要求：

1. 先只做一个产品族和一个方向，例如焦煤下跌趋势做空。
2. 至少覆盖多个趋势、震荡和反转阶段。
3. train / validation / test 按时间和合约严格隔离。
4. 处理换月和主力合约映射，不能把不同合约直接无标记拼接。
5. 所有品种阈值只在 train/validation 上确定。
6. 最终 test 不参与规则阈值、模型参数或 reward 调整。
7. test 中 teacher 完整交易少于 30 笔时，只能判定为“证据不足”，不能判定有效。
8. 必须按月份、合约和行情阶段分组报告，不能只报告总收益。
9. 必须报告硬条件候选数、A 类数量和机会覆盖率。
10. 必须报告盈利贡献集中度；不能由单月、单合约或少数异常交易支撑全部利润。

建议切分：

```text
train:      较早年份和合约
validation: 后续连续时间段，用于阈值和模型选择
test:       最后完整时间段，完全冻结
```

30 笔只是最低否决线，不代表统计充分。最终是否进入 Dreamer，还需要结合交易分布、
walk-forward 稳定性和置信区间判断。

## 15. 分阶段实施

### Phase 0：数据与契约审计

目标：

- 冻结系统规则版本；
- 建立产品参数注册表；
- 建立时间对齐和无未来数据测试；
- 建立现实成交模型。

必须完成：

```text
下一根 bar open 成交或明确延迟
手续费、点差、滑点
多周期 closed-bar 对齐
因果 pivot/关键位
严格时间切分
```

准入门槛：任何 causality 测试失败，不进入下一阶段。

### Phase 1A：因果识别基础

只实现：

```text
closed-bar 多周期对齐
日线 DOWN / NOT_DOWN
5m MACD
低位追空 D 类禁止
因果局部高点
结构止损距离检查
```

本阶段不输出交易动作，不修改 env 或 Dreamer。它只输出离线特征、状态和 reason bits。

准入门槛：

```text
5m/1h/daily closed-bar 对齐准确率 = 100%
5m MACD 对齐准确率 = 100%
未来数据截断不改变历史输出
DOWN / NOT_DOWN 和低位追空规则具备合成行情单测
结构止损距离完全可复算
```

### Phase 1B：最小 A/D 识别器

只在 Phase 1A 可信后增加：

```text
1h 最近 confirmed/provisional swing high
跌破后的旧支撑变阻力
当前价格是否接近 1h 阻力
做空硬条件
做空软确认分数
A / D 二分类
near-A 候选统计
reason_bits
execution-ready EntryPlan + ExitPlan
```

明确不进入 Phase 1B 的内容：

```text
B/C 运行时评级
完整 route map
主力进攻/离场/转手高级分类
多品种
多头镜像
env 与 Dreamer 修改
```

准入门槛：

```text
关键位生命周期和未来截断测试 = 100%
所有 A/D 判断都有可复算 reason bits
同一 snapshot 重建的 EntryPlan/ExitPlan 完全一致
A 数量/月、near-A 数量和软分数分布可报告
人工抽检发现的问题可以定位到具体规则版本
```

### Phase 2：确定性规则 teacher 回测

把 A 类规则转成确定性 teacher：

```text
A 类入场
EntryPlan 入场
ExitPlan 结构止损、保本、失效和最大持有期限
固定 kR 或最近因果支撑目标
```

准入门槛：

```text
扣除成本后 validation/test 具有正期望
profit factor > 1
expectancy > 0
大多数 walk-forward 区间盈利
D 类误入 = 0
结果不是少数异常交易贡献
test 完整交易数 >= 30；否则结论为证据不足
按月份/合约/行情阶段报告
报告 A 数量、near-A 数量和机会覆盖率
报告最大单笔、最大单月和最大单合约盈利贡献占比
```

如果 teacher 不通过，返回 Phase 1 修规则。规则调整必须生成新版本并重新跑完整
validation/test 流程，不允许直接进入监督模型、env 或 Dreamer。

### Phase 3：人工抽检与规则冻结

按月份、合约和结果分层抽检：

```text
A 类盈利样本
A 类亏损样本
near-A 候选
D 类禁止样本
confirmed/provisional level 切换样本
ExitPlan 强制退出样本
```

每个样本必须能还原当时可见行情、关键位生命周期、reason bits、EntryPlan 和
ExitPlan。抽检不通过则回到 Phase 1；通过后冻结 teacher 规则版本。

### Phase 4：监督学习评级器

先使用可解释基线：

```text
规则特征 + LightGBM/XGBoost
```

再比较小型多周期时间序列模型。

这是 Dreamer 之前的必经阶段，不允许跳过或直接用 Dreamer heads 代替。

准入门槛：

```text
D 类 recall > 80%
A 类 precision > 55%
A 类 recall、A 数量/月和选择覆盖率达到预先冻结门槛
非关键位识别 > 70%
低位追空识别 > 80%
校准后高置信 A 类在样本外具有正期望
按月份/合约报告 PR-AUC、混淆矩阵和候选数量
```

准确率和 precision 不能作为唯一指标。模型如果靠极度保守获得高 precision，但
候选数量、recall 或机会覆盖率过低，不得通过。

### Phase 5：环境接入

新增 observation、`entry_gate`、最终 `setup_mask`、execution-ready `EntryPlan`
和 `ExitPlan`。

准入门槛：

```text
action_mask 与 setup_mask 语义完全分离
obs_t 的 RecognitionSnapshot/EntryPlan/ExitPlan 被 step 原样消费
stale snapshot fail fast
结构止损、目标和保护性退出执行原子
D 类 setup 无法新开仓
保持仓位和平仓不被错误屏蔽
```

### Phase 6：Dreamer 监督 warm-up

训练 recognition heads、setup head 和 behavior cloning，不启用正式 actor fine-tune。

准入门槛：

```text
prior setup-mask 预测达到门槛
imagined setup fallback 接近 0
BC policy 在样本外不退化为永远 FLAT
D 类误入率满足门槛
```

### Phase 7：Dreamer 正式 fine-tune

只在前述全部门禁通过后启用 masked actor/value。

正式验收：

```text
样本外扣成本净利润为正
profit factor > 1
expectancy > 0
多个 walk-forward 区间多数盈利
D 类误入率接近 0
A 类候选参与率非 0
不是永远 FLAT
结果优于 teacher 和简单基准
最大回撤满足要求
```

## 16. 测试计划

### 16.1 Causality

对任意时间 `t`：

1. 复制数据；
2. 修改 `t+1` 之后全部行情；
3. 重新计算；
4. 断言 `t` 及之前的所有特征、关键位生命周期、评级、EntryPlan 和 ExitPlan 不变。

### 16.2 多周期对齐

测试 5m/1h bar 未关闭前不得提前出现 MACD、high、low 和 close。

### 16.3 规则单测

第一版用合成行情分别覆盖：

```text
日线 DOWN / NOT_DOWN
confirmed/provisional 关键位创建、确认和失效
跌破反抽
高位放量涨不动
低位追空
5m 死叉有/无
止损过大
RR 不足
软分数刚好位于 A 阈值两侧
```

### 16.4 环境契约

```text
action_mask 只管执行合法性
entry_gate 只管新仓资格
setup_mask 是结合当前持仓后的最终三动作策略资格
EntryPlan/ExitPlan 与 step 一致
保护性退出不能被 actor 绕过
真实 policy 不选择最终 mask 外动作
imagination 采样 mask 与 actor loss mask 一致
```

### 16.5 回测与训练

必须包含基准：

```text
永远 FLAT
随机合法动作
固定多头
固定空头
纯规则 teacher
监督评级器
Dreamer fine-tune
```

每个回测必须同时输出交易数、按月份/合约分组结果、机会覆盖率和盈利贡献集中度。

## 17. P0/P1/P2 优先级

### P0：第一轮立即实现

1. 多周期因果对齐和未来数据泄漏测试。
2. 现实成交时序。
3. 严格 train/validation/test 隔离与品种参数注册表。
4. 日线 `DOWN / NOT_DOWN`、5m MACD、低位追空和结构止损距离。
5. confirmed/provisional 1h 关键位生命周期。
6. A/D 硬条件、软评分、reason bits。
7. execution-ready EntryPlan 和 ExitPlan。
8. 规则 teacher 回测、人工抽检和样本外门禁。

### P1：teacher 通过后、Dreamer 接入前必须解决

1. 冻结 A/D 数据集、原因标签和 teacher 版本。
2. LightGBM/XGBoost 监督评级器及样本外门禁。
3. `action_mask` 与 `setup_mask` 分离。
4. RecognitionSnapshot、EntryPlan 和 ExitPlan 环境接入。
5. 保护性退出原子执行与 stale snapshot fail fast。
6. env 接入后的规则 teacher 回放一致性。

### P2：只有 P0/P1 全部通过后再做

1. Dreamer recognition/setup 监督 heads。
2. behavior cloning warm start。
3. setup head imagination 门禁。
4. masked actor/value fine-tune。
5. B/C 运行时评级、完整路线图、多产品和多方向扩展。

## 18. 明确不建议

1. 不建议只加“不交易惩罚”解决永远 FLAT。
2. 不建议只调整 reward 后立即重新跑 1M steps。
3. 不建议把策略资格直接混入现有执行 `action_mask`。
4. 不建议用未来确认 ZigZag 生成训练 observation。
5. 不建议第一版同时做所有品种、所有方向和全部路线图。
6. 不建议让 LLM 直接看图输出买卖动作。
7. 不建议在规则 teacher 样本外不盈利时，用 RL 掩盖规则问题。
8. 不建议第一轮同时修改 env 或 Dreamer。

## 19. 推荐第一版范围

### 19.1 首个交付里程碑

为了最快验证整个方向，首个交付只包含 `Phase 0 -> Phase 1A -> Phase 1B -> Phase 2`：

```text
产品：DCE 焦煤
方向：只做日线下跌中的反弹空

必须实现：
1. 日线 DOWN / NOT_DOWN
2. 1h 最近 confirmed/provisional swing high
3. 旧支撑变阻力
4. 当前价格是否接近 1h 阻力区
5. 是否属于低位追空
6. 5m MACD dead cross
7. 结构止损距离与 RR
8. 硬条件 + 软确认分数
9. A / D 二分类、near-A 统计和 reason_bits
10. execution-ready EntryPlan + ExitPlan
11. 离线规则 teacher 回测与报告

暂时不做：
1. B/C 运行时评级
2. 完整 route map
3. 主力转手高级分类
4. 多品种
5. 多头镜像
6. env 接入
7. Dreamer warm-up / fine-tune
```

这版可以最直接验证 HTML 中最明确的修正：

> 下跌趋势中，不是看到跌了才空；而是等它涨上去，在高位失败时空。

首个交付的产物是离线识别数据、逐样本解释、teacher 交易记录和样本外报告，不是新的
正式训练配置。只有它通过后，才进入人工冻结、监督评级器、env 和 Dreamer。

### 19.2 后续环境动作范围

后续接入 env 时仍保持：

```text
动作：SHORT / FLAT / LONG
策略：只允许 A 类新开 SHORT；LONG 新开禁用
保护：ExitPlan 强制执行结构止损、失效和最大持有期限
```

该单方向版本整体通过后，再镜像增加上涨趋势回调做多，随后扩展 B 类运行时评级和
更完整路线图。

## 20. 最终判断

这套系统适合应用到当前项目，但应用方式必须从“纯 RL 预测交易”改成“系统先验驱动的
分层决策”。

最重要的工程顺序是：

```text
先证明识别正确
-> 再证明规则 teacher 样本外有正期望
-> 再完成分层人工抽检并冻结规则
-> 再证明监督评级器样本外有效
-> 再把识别、EntryPlan 和 ExitPlan 接入 env
-> 再监督预训练 Dreamer
-> 最后才允许 RL fine-tune
```

如果跳过这些门禁直接继续正式 Dreamer 训练，最可能再次得到两个失败结果之一：

```text
永远 FLAT
或
为了逃避 FLAT 而过度交易
```

这两者都不符合陈大哥交易系统。

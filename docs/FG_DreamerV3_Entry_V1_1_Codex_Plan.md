# FG-DreamerV3-Entry-V1.1 — 复用既有加减仓能力的 Codex 实施方案

> 修订日期：2026-10-10。**本文替代 V1 的仓位架构、双槽位/EntryBatch 重构和相应 Codex 任务**。其余目标（固定方向、每次 3 手、上限 6 手、成交后赚 5 点止盈、速度唯一 Reward、允许跨日、只用权益进行独立账务/风控审计）保持不变。
>
> 取证：从上传的 `gy.tar(1).bz2` 与 `d3.tar(1).bz2` 的 Git 快照恢复，`gym-trading-env` HEAD `9699286`、`dreamerv3` HEAD `92af879`。已阅读相关代码；当前并未修改项目源码、运行正式测试或模型训练。旧总审阅文档中关于“必须重建双槽位持仓子系统”的建议撤回。

## 1. 修正后的核心结论

**原有 Env 的底层仓位模块支持同向多次开仓、持仓汇总及按仓位记录平仓。两个仓位不是新系统必须独立开发的两套账户或交易引擎；对 UI 可以只暴露净仓位。**

真正的改造点是：**当前 Dreamer 可见的目标仓位动作 `[SHORT, FLAT, LONG]` 不等于“增加/减少 3 手”。** 现有 `CustomTradingEnv._plan_exec_action()` 在“SHORT → SHORT”时返回 `HOLD`，`CustomTradingEnv.__init__()` 又强制 `intraday_single_position=True`。这限制了当前 RL 接口的重复同向开仓，并不否定底层 `PositionManager` 的能力。V1.1 应增设轻量的策略动作适配层，而不是重写 Ledger、PositionManager 或一套双槽位账户。

目标架构：

```text
人工 direction = SHORT / LONG / NO_TRADE
              │
DreamerV3 → WAIT(0) / ADD_3(1)
              │
轻量 EntryActionAdapter + DirectionGuard + Mask
              │
现有 Env 的开仓报价/提交、PositionManager、Ledger、UserAccounts
              │
真实净持仓：0 / 3 / 6 手（对 UI 可只显示一条）
              │
现有 Position 记录和开仓成交事件（用于确定各次加仓对应的价格与时间）
              │
逐次入场 +5 点 → 自动减仓 3 手 → 现有平仓/账务流程
              │
TP_FULL_FILL 事件 → Reward: 60/(60+T_minutes)
```

**特别说明**：为每次 `ADD_3` 保存唯一 `entry_id`、成交价、成交时间、数量、关联 `Position`/slot 与平仓完成信息，是交易事件/Reward 归因，不是建立第二套 PositionManager，也不是要求交易所或实盘账户拥有两个物理分仓。因为如果只保存 6 手的加权平均价，则无法判断每次 3 手各自是否已达到 +5 点，以及分别用了多久。

## 2. 从源码确认的仓位能力与接口差异

| 源码位置 | 已有能力/事实 | V1.1 决策 |
|---|---|---|
| `src/gym_trading_env/envs/position_manager.py`, `add_long_position/add_short_position` | 对 `long_positions`/`short_positions` 多个现有槽位插入 `Position`；可自动找到空槽，也可指定 slot。 | **原样复用**，不开发新的双仓位管理器。 |
| 同文件，`total_long_position/total_short_position` | 返回同向所有 `Position.size` 汇总。 | 使用汇总值判定 0/3/6 及最大 6 手。 |
| 同文件，`quote_close_long/short`、`commit_close_long/short` | 指定 slot 计算/提交一条 `Position` 的完整平仓，支持减少整体净仓位。 | 复用原子报价、账务、平仓；**注意目前完整关闭指定 Position，不是任意数量的部分平仓 API**。本策略每次开仓正好 3 手，可对每条 3 手记录整笔平仓。 |
| `envs/position.py`, `Position` | 已记录 `size, entry_price, initial_margin, open_step, stop_loss_price, take_profit_price`。 | 复用；仅为跨日真实时间戳/订单唯一标识补充关联记录，先确认已有 TradeRecord/Action Logger 能否覆盖。 |
| `envs/trading_env.py:85-120` | 当前 Dreamer 的 env 是 `Discrete(3)` 目标仓位，强制 `intraday_single_position=True`。 | **只在新策略模式**切换 `Discrete(2)` 的 WAIT/ADD_3 语义，不更改旧模式行为。 |
| `envs/trading_env.py:_plan_exec_action` | 当前 `SHORT → SHORT` 返回 HOLD。 | Entry 专属 action mapper 应每次合法的 ADD_3 生成同向加仓意图，不经过旧 target-to-target 简化。 |
| `envs/trading_env.py:_preflight_open/_quote_target_open` | 已有可用仓位、保证金及 `trade_lot` 计算，但 `intraday_single_position` 条件会阻止已有持仓时再开。 | 引入显式 Entry 策略分支/adapter，复用报价/记账组件，放开**同向有仓**拒绝，仍禁止反向和超过 6 手。不能简单删除原有拒绝逻辑，避免旧策略回归失败。 |
| `envs/action.py` `Action.POSITION_UP/POSITION_DOWN` | 枚举已有两个名称。 | **不能把“枚举存在”等同于“当前 TargetTransitionTable 已接入这些动作”**；先验证现有真实执行路径。如果没有适配，新增轻量映射。 |
| `envs/trading_env.py:_apply_take_profits` | 已遍历已有 `Position`，按该条的 TP 价调用 `_long_close/_short_close(slot=...)`。 | **优先复用这个机制**。改动集中在固定 5 点 TP 定价、成交时间/填单模型和完成事件。 |
| `envs/trading_env.py:_compute_take_profit_price` | 当前只支持基于 SL 的 RR TP。 | Entry 模式新增 `absolute_points=5`，不启用策略型 SL。旧 RR 模式保持不变。 |
| `envs/trading_env.py:step` | 使用已完成的当前 bar 信息后同 bar close 成交。 | Entry 模式引入因果性正确的 `t` 信号 → 最早 `t+1` 有效 bar 执行，不能继承旧同 bar 成交。 |
| `envs/accounting/ledger.py`、`envs/user_accounts.py` | 复式记账、保证金及权益核算已存在。 | 原样复用；只做账户审计及安全边界，不参与训练 Reward。 |
| DreamerV3 `dreamerv3/agent.py` | 支持 `action_mask` 动态 shape 和真实 MaskedCategorical、imagination availability head。 | 优先只换 Gym env 的 `Discrete(2)` 与 observation，保留 Agent 核心。 |

### 仓位的实际行为示例（指定 SHORT）

```text
t0：净持仓 0；Dreamer ADD_3，成交 3 手空 @900
    原 PositionManager 中增加一条 size=3，entry=900，目标平仓价=895（按平仓可成交价口径）

t1：净持仓 3；Dreamer ADD_3，成交 3 手空 @910
    原 PositionManager 增加另一条 size=3，entry=910，目标平仓价=905
    对外净持仓 = 6 手；不要求 UI 显示两条

t2：行情实际可成交价触及第二条的 +5 点目标，完成 3 手减仓
    原 PositionManager 指定该 Position quote_close/commit_close
    对外净持仓 6 → 3；只奖励第二次 ADD_3 的用时

t3：Dreamer 可再次 ADD_3，净持仓 3 → 6；第一条不强制止损
```

如果经纪商持仓视图仅提供同一合约净仓位，那么两次开仓的 `entry_id` 是**内部成交归因记录**。实盘委托和成交回报必须与净持仓一致，不可以假设经纪商也维护两条可直接指定平仓的独立仓位。

## 3. 必须冻结的策略契约

- 资金 `300000 CNY`，合约首选 `FG2701`；真实手续费/合约到期/保证金由当日有效配置或外部合约元数据提供。
- 外部方向只允许 `SHORT` 或 `LONG`，另允许 `NO_TRADE`；模型不能改变方向。逆向委托在交易 Guard 拒绝并可标记训练约束错误。
- 模型动作 `WAIT` 或 `ADD_3`；单次严格申请 3 手，最多持仓 6 手，不增加第三次同时在仓的 3 手。
- 每次完整成交的 3 手入场记录，以**该次成交均价**为依据，顺方向净价格差达到 5 个价格点时由 Env 自动平掉对应的 3 手，释放额度后继续允许新加仓。
- 未达到 +5 点则继续持仓，不设置 60 分钟策略强平、不附加亏损/权益/持仓时间负奖励。交易所规则、期货公司风控、合约可持有期限仍强制生效。
- 未实际完成平仓成交则不领取 Reward；部分开平仓需要逐笔追踪，不能用触价或虚构 3 手成交代替完整入场/平仓事件。
- Reward 仅为每个**真实完成止盈的入场记录** `60/(60+T)`，`T` 为开仓实际成交至止盈实际成交的自然分钟（含隔夜），其他时刻 Reward=0。
- 收益、权益、保证金、回撤和期货公司可能产生的强平与减仓记录继续核算、用于风控/回测报告，但**不进入 Reward**。

## 4. 新策略的状态、动作与代码复用原则

### 4.1 Action Mask

| 方向有效 | 当前净持仓 | WAIT | ADD_3 | 原因 |
|---|---:|---:|---:|---|
| SHORT/LONG | 0 | 1 | 1 | 可首开 3 手 |
| SHORT/LONG | 3 | 1 | 1 | 可继续加 3 手 |
| SHORT/LONG | 6 | 1 | 0 | 到达持仓上限 |
| NO_TRADE | 任意 | 1 | 0 | 无方向 |
| 任意 | 0/3 | 1 | 0 | 非法交易时段、保证金不足、合约临近禁开、订单未定等 |

`ADD_3` 表示**净仓位增加 3 手**，不是选择 UI 上的“第一个仓位/第二个仓位”。不能选择逆向、任意减仓或自由平仓。止盈动作由 Env 自动触发，走现有减仓/平仓路径。

### 4.2 Observation 最少需要的信息

继续沿用已经验证没有未来数据的 `market_seq` 与必要 `agent_state`；新增/整理：

```yaml
direction: SHORT | LONG | NO_TRADE
net_position_lots: 0 | 3 | 6
free_add_capacity_lots: 6 - net_position_lots
entry_lots:           # 最多两条，按成交时间排序，内部 reward 归因用，不是 UI 分仓
  - entry_age_minutes, distance_to_tp, quantity, pending_exit_flag
  - entry_age_minutes, distance_to_tp, quantity, pending_exit_flag
action_mask: [WAIT_allowed=1, ADD_3_allowed=(0|1)]
```

每次加仓的价格、时间戳来自现有 Position / TradeRecord 及最小新增事件元数据，不创建第二套账户、头寸或强行复制 `Position` 对象。建议视为只读特征投影，`net_position_lots` 始终以 `PositionManager.total_*_position()` 为真值来源。

### 4.3 TP 和 Reward 事件

- 在 Entry 专用模式中，`Position.take_profit_price` 使用 5 点绝对价差，而不是旧 RR 模式。
- `TP_FULL_FILL` 应携带稳定的 `entry_id`、`entry_fill_time`、`tp_fill_time`、`entry_fill_price`、`tp_fill_price`、`closed_quantity`、`close_reason`。
- 在现有 `_long_close/_short_close` 中完成记账和减仓，然后产生一次性完成事件；不要让 Reward 组件直接修改 PositionManager。
- `r_t = Σ[60/(60+T_i)]`，只对本 step 首次完整 `TP_FULL_FILL` 的记录计数。已完成并领奖的 `entry_id` 不能再次领奖。
- Reward 函数不得读取或使用 MTM/equity/drawdown/fee；历史账务依然正确记录，以供独立评估。

### 4.4 成交时序和跨日

观察到完整 bar `t` 后不能再次按 bar `t` 收盘价开仓；Entry 策略应提交请求，最早由下一根真实可交易 `t+1` 的 open（或可证明的真实成交模型）执行。不能在填充休市分钟成交。1 分钟 OHLC 不能证明限价订单必然成交；应使用明示的保守规则，后续用 Tick 验证。

策略不因收盘自动清仓；合约到期、强制清算及 Env 人为截断均要有清楚的账户记录，不得通过 reset 免费抹除亏损仓位。跨自然日分钟数参与 Reward 时间衰减。

## 5. Codex 最小改造路线（替代旧“重建双槽位 Env”）

**Phase 0：先核对能力，不修改代码**

1. 在当前 HEAD 逐一追踪 `POSITION_UP/POSITION_DOWN`、`add_short_position()`、`_short_open()`、`_short_close()`、`_apply_take_profits()` 的完整调用/账务和 Action Mask 路径。
2. 列出可直接复用的开仓、减仓、手续费、保证金、TP、TradeRecord、回放与跨日模块。记录不能复用之处，不预设一定新建独立交易 Env。
3. 先写 0→3→6→3→6 的 synthetic 验收测试，严格定义单笔 +5 点的成交价格和完成时间。保留旧模式 regression。

**Phase 1：增加轻量 Entry 策略模式（而不是平行仓位系统）**

1. 优先通过明确的 `entry_only` 模式及专用 action adapter/strategy component 接入现有 `CustomTradingEnv` 交易和账户引擎。若目标仓位 `step()` 强耦合难以隔离，可使用轻量 env 包装/薄适配类，**但不得复制 PositionManager、Ledger 或重写整个交易撮合与账户系统**。
2. 行为 `WAIT=0, ADD_3=1`，采用已有 PositionManager 的同向加仓和指定 Position 完整平仓能力。配置 `trade_lot=3, max_position=6`；旧 `intraday_single_position` 约束只对 Entry 新模式解除，旧模式保持不变。
3. 新增 5 点绝对止盈模式；复用 `_apply_take_profits()` / `_long_close()` / `_short_close()`，必要时在共用组件中最小修改使撮合因果性正确。每次加仓的 `entry_id` 是成交关联元数据，不是第二套账本。
4. 只在 `TP_FULL_FILL` 时生成 Reward `60/(60+T)`，其他一律 0。

**Phase 2：规则基线 + 回归及真实性测试**

- 既有多次加仓/按指定 Position 平仓测试必须继续通过；新增 Action Mask/保证金/6 手上限、交易方向、不重复 Reward、TP 只成交一次、跨日、相同分钟多笔 TP、部分成交、合约期限、Ledger 守恒、no-lookahead 测试。
- 初期用 synthetic 测试，然后使用真实 FG 合约行情回放；实验基线为 `always_add`、固定间距、随机合法入场。对外交付 `fills.csv`, `entry_tp_attribution.csv`, `reward_events.csv`, `account_equity.csv`, `summary.json`。

**Phase 3：接入 DreamerV3**

- 新建配置 profile，尽量不改 Agent/RSSM 主体；使用 `[WAIT, ADD_3]` 的 MaskedCategorical，测试真实与 imagination 的 Action Mask。
- 比较 `+1` 每次真实止盈（D1）与 `60/(60+T)`（D2），两者都是纯事件奖励，不含权益。
- 审计长时间等待正奖励的 credit assignment、无交易策略、持仓上限占用以及历史合约到期；跨年份/跨 regime 多 seed 样本外验证，禁止用训练集来选最终阈值。

**Phase 4：实盘前置条件**

- 模型输出只能是 `WAIT/ADD_3`，真实下单程序负责 side、交易时段、仓位上限与重复委托检查；自动 5 点平仓由稳定确定性执行引擎负责。
- Broker 可能只显示单份净持仓；本地 entry attribution 与券商委托/成交/净仓位必须严格对账。交易所的今昨仓、平仓优先规则、限价触及但未成交及断网重连需要单独审计。
- 训练运行禁止启用 live RPC/真实下单。正式风险与监管报备单独处理。

## 6. Codex 可直接使用的指令（以本 V1.1 为准）

```text
你可以访问 ~/Documents/work/gym-trading-env 与 ~/Documents/work/dreamerv3。
先确认 HEAD，分别预期为 9699286 / 92af879；如不同先分析 diff，不直接复写。

以 FG_DreamerV3_Entry_V1_1_Codex_Plan.md 为唯一权威设计，旧 V1 文档关于“重建两个独立仓位槽位、开发平行 EntryBatch 账户/全新交易 Env”的部分已废弃。

用户已确认现有 Env 能实现加减仓，两个仓位仅是可能的 UI 展示。必须先逐层复核现有 PositionManager、Position、_long_open/_short_open、_long_close/_short_close、_apply_take_profits、Account/Ledger 的真实可复用能力，不能重复造轮子。

业务：外部方向已定；Dreamer 仅 WAIT/ADD_3；每次增加同向 3 手；净持仓最多 6 手；每次加仓以该次实际成交价为基准，在顺向 +5 点真实止盈成交时自动减少 3 手，可继续新增；不设置策略止损或人工持仓时限，真实到期/保证金风控继续生效。

只允许 Reward = 60/(60 + T_minutes)，且仅在完整成交 TP 的 3 手归因记录领取一次；未完成和亏损浮动期间无奖励/惩罚，账户净权益只做审计和风控，不进入 Reward。

开发时优先新增轻量 entry-only mode/action adapter，复用已有仓位管理、交易撮合、Ledger 和 TP 平仓系统。不要重构出两套 PositionManager，也不要因为 UI 看起来有两笔仓位而建立独立的 Broker 分仓语义。为了独立计算每次 3 手的获利时间，可利用既有 Position 对象和最少量的 entry_id / fill timestamps 事件关联。

请先完成 Phase 0 的调用链审计和 synthetic 测试设计，再开展实现。每个阶段独立提交并说明测试、回归、财务守恒、实际/模拟撮合时序。保留旧交易模式和旧 Dreamer 实验可回归。禁止启用真实下单或使用未来行情。
```

## 7. 评审结论

此次更正显著缩小实现范围：**仓位管理与净持仓汇总基本现成；要解决的是 RL 动作语义、5 点固定止盈、真实成交时间归因、因果性成交及稀疏 Reward。** 不应因为两个 3 手入场在 UI 上可呈现为两行，就推导出必须有两个独立账户、两套持仓引擎或必须重写 Env 的全部 step 逻辑。保留必要的“逐次入场记录”，是为了计算你规定的速度得分，而不是引入额外策略目标。

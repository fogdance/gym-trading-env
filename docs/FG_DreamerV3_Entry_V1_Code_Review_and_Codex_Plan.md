# FG-DreamerV3-Entry-V1：源码审阅与 Codex 实施方案

> 版本：2026-10-10 / DESIGN REVIEW / 尚未修改原项目源代码、尚未训练  
> 目标：已知多空方向，DreamerV3 **只学习何时开仓**；每次 3 手、最多同时两批共 6 手；每批以成交价格差盈利 5 个玻璃价格点后自动平仓；成功用时越少，Reward 越高；未盈利可继续持有直至真实风控/合约持有期限约束。**权益和浮亏不得进入 Reward。**

## 0. 取证与边界

从用户上传的两个仅含 `.git/` 的包恢复：

| 仓库 | 工作分支 | HEAD | Python 模块遍历 | 审阅范围 |
|---|---|---|---:|---|
| `dreamerv3` | `dev/action-mask-full-chain` | `92af879`（2026-06-18） | 79 个 Python 文件 / 14,384 行 | 整体 AST/语法遍历；重点审查 agent、RSSM、mask、Gym wrapper、train/eval、checkpoint、config |
| `gym-trading-env` | `feat/reward-profit-lock-atr` | `9699286`（2026-07-04） | 154 个 Python 文件 / 40,825 行 | 整体 AST/语法遍历；重点审查 Env、target transition、position、ledger、reward、行情、合约/时间、测试、entry evaluator |

合计 233 个 Python 文件，AST 均可解析；`compileall` 无语法错误。已恢复工作树（`git checkout -f .`）。静态遍历不等同于逐行人工形式化证明，重点交易路径已逐段审阅。当前容器缺 `gymnasium`、`elements`、`talib`、`pymysql` 等运行依赖，**没有运行既有测试、训练或实盘执行**。两个上传包未包含实际历史 FG 行情、交易方向日志、训练数据、checkpoint 或数据库快照。因此本文是可执行设计/风险审计，不声称已有策略收益。

**重要：先建立独立研究环境，旧项目保持原状。** Codex 完成各阶段并使测试通过后才可考虑联通仿真或真实交易。禁止从训练代码中打开实盘 RPC。

## 1. 用户业务契约（冻结）

| 条件 | V1 最终定义 |
|---|---|
| 标的 | 郑商所玻璃期货真实合约，首要目标 FG2701；历史训练使用 FG 系列真实合约 |
| 方向 | **外部输入，已确定**；模型不预测或修改方向 |
| 初始本金 | 300,000 CNY，用于 Env 真实保证金与风险核算，不参与 Reward |
| 开仓量 | 每次严格 3 手，禁止部分交易意图造成 1/2 手的策略仓；真实部分成交必须追踪 |
| 持仓上限 | 任意时刻最多同时 6 手 = 两个独立 3 手批次 |
| 止盈 | 每个批次独立，**实际买卖成交价格差**沿方向达到 5 元/吨（5 个价格点）时结算；不是未扣价差的中间价碰一下就算完成 |
| 平仓后 | 对应 3 手槽位释放；可立即进入 WAIT，之后再找机会开仓，无每日 1 笔限制 |
| 没盈利 | 持续 HOLD；**无 60 分钟时限、不施加策略型时长止损，也不因浮亏扣 Reward** |
| 方向错误 | 系统性不可能：策略动作只有 `WAIT`、`OPEN_3`；侧向由外部 direction 决定；绕过策略层逆向下单视为违反业务契约，直接拒绝/失败 |
| 速度 | 从实际开仓成交时刻到实际止盈成交时刻，V1 使用自然时间分钟（含闭市、隔夜），另记录交易分钟作为诊断 |
| 止盈未成交 | 不算成功、不发 Reward，保留批次仓位 |
| 到期/强平 | 按交易所/期货公司真实规定处理，真实 PnL 入审计，但无成功速度奖励 |
| Reward | **仅** `sum(60/(60+T_i))`，每个 `TAKE_PROFIT_FILLED` 事件发一次；其他交易状态发零 |
| 交易评价 | Reward 排名与净权益回撤等经济审计**分离**；不将审计指标反写进 Reward |

玻璃合约乘数 20 吨/手，3 手价格走 5 点毛利 `3 * 20 * 5 = 300 CNY`，6 手每点 120 CNY；手续费、滑点另行计入账户，而不改变 5 价格点的业务目标。价格最小变动单位、交易所手续费、当日保证金、涨跌停、自然人持仓期限须用对应交易日的合约参数，不靠常数猜测。

### 1.1 奖励的形式化定义

对于第 `i` 个完成的独立批次：

```text
T_i = (take_profit_fill_timestamp_i - entry_fill_timestamp_i) / 60 seconds
R_i = 60 / (60 + T_i)   if and only if exit_reason == TAKE_PROFIT and fully filled 3 lots
R_i = 0                otherwise
r_t = sum(R_i for all fully completed TP batches on this environment step)
EpisodeScore = sum_t r_t
```

同一步两批都成功止盈，可发两次奖励。`T >= 0`；时钟不能被 reset、休市空白行、回测进度计数替代。`r_t` 不包含 equity/MTM、手续费、持仓浮亏、持仓天数、每日盈亏等。违规动作应由不变量和失败测试阻止，不能用负分代替。Reward 函数不得凭未成交的 `low <= target` 就直接给分。

**重要的强化学习语义：** 原 DreamerV3 本身还有折扣因子，它会偏好更早获得的奖励。新 Reward 又显式按单笔速度衰减，因此二者存在叠加的时间偏好。V1 必须跑两个严格对照：A. 每次真实 5 点完成固定 `+1`；B. 每次真实 5 点完成 `60/(60+T)`。业务首选 B；A 用来确认 B 是否提供额外学习价值。两组使用同一动作、Env 和验证程序。

## 2. 源码现状与必须修改的真实位置

| 文件及位置 | 已确认现状 | V1 处理 |
|---|---|---|
| `gym-trading-env/src/gym_trading_env/envs/trading_env.py:85-150` | `CustomTradingEnv` 定义 `action_space=Discrete(3)`，`SHORT/FLAT/LONG` 目标仓位；构造时要求 `intraday_single_position=True` | **新增独立 Gym Env**，不要直接把旧 `max_short_position` 改为 6；旧策略保持可回归 |
| 同文件 `:585-618` | `_current_target()`、`_plan_exec_action()`：已是 SHORT 再选 SHORT 会变成 HOLD；不能同方向第二次开仓 | 独立的 `WAIT/OPEN_3` 增量开仓契约 |
| 同文件 `:741-820` | `_quote_target_open()` 对已有任何持仓会拒单；创建 Position 时计算 RR 型 TP/SL | 新独立 quote/commit；允许同向槽位 1、2，指定 `abs=5` TP，无策略 SL |
| 同文件 `:1029-1165` | `TargetTransitionTable` 按目标仓位建立 mask/quote，共享状态键，消费一次即失效 | 新 `EntryTransitionTable`/mask 以合法槽位数、方向、订单中状态、价格、保证金、合约期限等计算；要求发布/消费同一决策表 |
| 同文件 `:1165-1356` | `step()` 同 bar 先按当前 `row_C` 交易，再推进下一 bar；TP 在推进后的高/低价检查 | 使用已完成 bar `t` 的观察，行动提交至**下一根有效可交易 bar 的 open/实际回报成交**，随后处理止盈订单；事件时序需单测 |
| 同文件 `:353-560` | reset 重建 ledger、PositionManager；episode_length/start_clock 选起点；多日策略可配置，但近 EOD/截断有默认限制 | 新 Env 初始化/恢复两个批次及方向、事件队列；随机训练起点必须剩余有意义的合同历史；禁止重置悄悄抹去亏损仓位 |
| 同文件 `:1698-1770` | 会话结束/数据截断/max steps 可 `_force_flatten_if_any()`；保证金、回撤可导致 terminated | 多日无策略时限；仅真实风控、合约到期、明确仿真截止触发清算；明确区分 terminated/truncated，不能靠 Episode reset 免债 |
| 同文件 `:3120-3190` | 自动 TP 按 high/low；SL 独立；TP close path 写成交、ledger 与 TradeRecord | TP 每批独立使用真实成交事件，并对 bar 内是否可成交、gap、价差和排队采用独立保守撮合 |
| 同文件 `:3390-3460` | `_compute_take_profit_price()` 仅 RR 模式，且要求已有 SL | 不能简单配 `take_profit_enabled=True`；新增 `abs_points=5` 专属配置与实现 |
| 同文件 `:1389-1636` | RPC 的订单字段和 live replay 都针对旧枚举；日志用 `TargetPos(action_index)`；同日 JSONL 恢复 | V1 **不直接复用为实盘**；新增指令日志，批次 ID、方向、数量、slot、订单 ID、成交 ID、成交时间；重启后以券商真实持仓及订单查询对账 |
| `envs/position.py` | Position 有 `size,entry_price,initial_margin,open_step,SL,TP` | 扩展或配套 `EntryBatch` 保存 `batch_id,side,entry_fill_ts,target,exit_order_id,completion`；保留旧对象兼容 |
| `envs/position_manager.py` | 已支持多个 long/short slots，`quote_close_*/commit_close_*` 可指定 slot；默认各 10 个 | 可复用 quote/commit 与数组，但 V1 **精确分配 2 个有效槽位**，确保任意一批释放可重复利用；不可依赖默认 10 槽位 |
| `envs/accounting/ledger.py`、`envs/user_accounts.py` | 现金、保证金、佣金和实现盈亏通过复式记账核算 | 保留账户作为交易正确性与风控，不参与 Reward；`sum(slot margin)=ledger user_margin`、账务守恒必须回归 |
| `envs/config.py:91-243` | 参数混杂 Forex/Futures、日内最大开仓、`take_profit_mode='rr'`、RR 需要 SL | 新独立模式配置校验，别改变旧默认，明确费率/撮合和 5 点定义 |
| `utils/bar_source.py`、`utils/market_store.py` | 已支持 Juejin 数据 `fut_bar_1m_v2`、CSV、严格期货 345 分钟重索引、时区及 `row_mask` | 复用特征和存储；成交时只在真实有效分钟，禁止在补空 bar 成交，保留合约/日期元数据 |
| `utils/market_features.py`、`utils/daily_features.py` | 多分钟/日线特征；有向前填充、历史滚动以及辅助日线 | 全部纳入无未来泄漏测试；观测只读 t 之前已形成的信息，未完成日线不可带未来全日统计 |
| `research/entry_evaluator.py:17-72,340+` | 已有 `canonical_next_open` 和 `signal_on_close_plus_spread` 两种执行协议 | 新 Env 与 evaluator 使用 canonical next open **同一 timing contract**；旧同 bar close 仅留诊断用 |
| `rewards/reward_functions.py`、`rewards/reward_audit.py` | Rewards 需 `RewardAuditMixin` 且 `debug.total` 与返回值一致；旧 reward_audit_v1 字段含 pnl/dd 等 | 新 `FastTakeProfitReward` 仅成功事件加分；字段 `pnl/dd` 固定 0 代表未参与奖励，另加 `tp_speed/closed_batches/hold_minutes_*` 诊断；遵守 audit 契约 |
| `dreamerv3/dreamerv3/agent.py:44-57` | 要求 `action_mask.shape==(action_space.n,)`，初始化 World Model/Actor 与 avail head | 将新 Env 动作缩为 `Discrete(2)`，理论上无需改核心 Agent 形状逻辑 |
| 同文件 `:143-169,210-278,328-355` | 真实行动使用 true mask；imagination 用 availability-head **预测 mask**；`hard_nonempty_mask()` 无预测可用动作时会取概率最高者 | 重点验证 `WAIT` 永远合法、imagined mask 极低 fallback 率；测试 0/3/6 手动态门禁 |
| 同文件 `:491-608` | imagined actor 使用折扣回报（`imag_length`、`horizon`）；标准训练 rewards 由世界模型预测 | 5 点成功是稀疏事件且可能跨日，严格评估训练能否学习；不要靠增加权益奖励“补救” |
| `dreamerv3/dreamerv3/main.py:404-459` | 通过 Gymnasium factory、wrapper 构造环境 | 新环境注册 `FixedDirectionEntryEnv-v0`；统一 run config 与唯一 Env constructor |
| `dreamerv3/embodied/envs/from_gymnasium.py:59-83` | Gym reset/step 转 embodied `is_first/is_last/is_terminal`；信息 `log/` 仅 scalar | 新 Env 必须遵循 terminated/truncated，输出所有新增 log 标量；不要修改 wrapper 的一般合同 |
| `dreamerv3/embodied/run/train.py`、`train_eval.py` | 记录 reward sum 为 `episode/score`；旧体系其他财务指标从 Env 日志来 | `episode/score` 在此任务只代表速度积分，不能重命名为利润；分别报告 TP event 和真实 account outcome |
| `dreamerv3/dreamerv3/configs.yaml:483+` | `action_mask_formal`、warm-up 门槛、测试已有支持 | 新独立 profile `fg_entry_v1`；复用 formal warm-up、checkpoint、seed/split 约束，不覆盖旧 profile |

### 2.1 必须优先修复的四个错位

1. **动作错位**：旧目标仓位系统下，同向第二笔会被视为 HOLD；即使 `PositionManager` 提供多槽位也不足以实现。
2. **止盈错位**：旧 RR TP 依赖止损；用户要绝对 +5 点且不启用策略止损。
3. **时间错位**：旧 `market_seq` 已包含 t 完整 bar，却按 t 的 close ± spread 开仓；旧 `docs/env_execution_timing_truth_report.md` 将其归为 `NON_CAUSAL_CURRENT_BAR`。新环境不能沿用。
4. **持仓周期错位**：旧 near-EOD / daily max opens / episode truncate 仍带明显日内策略假设；新策略允许隔夜、盈利后多次开仓、两个槽位独立清算。

### 2.2 工程边界建议

新建 `FixedDirectionEntryEnv(gym.Env)` **不要继承并覆盖巨大 `CustomTradingEnv.step()`**：其内部 `_refresh_agent_state`、mask、目标仓位 quote、日志、EOD/force flatten 强耦合，强行 subclass 更难保证正确。通过组合复用 `MarketStore / CsvBarSource / JuejinDBClient / PositionManager / Ledger / Account / UserAccounts / market_features`。对旧可复用对象做必要的有限扩展，并增加专属 test。旧 `CustomTradingEnv` 的代码路径、训练配置与测试不得被 V1 修改破坏。

## 3. 环境新结构：交易与训练责任分离

```text
ExternalDirectionProvider (SHORT/LONG/NO_TRADE; issue time & validity)
       |
       v
FixedDirectionEntryEnv-v0
  +-- EntryActionGuard [WAIT=0, OPEN_3=1]
  +-- 2 EntryBatch slots (each 3 lots, independent TP order)
  +-- EntryBrokerSimulator (signal t -> executable t+1/open; conservative fills)
  +-- Existing Ledger/PositionManager accounting adapter
  +-- ContractExpiryGuard & MarginGuard  [outside reward]
  +-- FastTakeProfitReward  [only TP_FULL_FILL events]
  +-- Obs builder [market_seq + entry_state + action_mask]
       |
       v
DreamerV3 (unchanged core: RSSM/Actor/Critic/avail)
       |
       v
WAIT / OPEN_3 (side can NEVER be chosen by the model)
```

### 3.1 为什么只用两个策略动作

`action=0`: `WAIT`。`action=1`: `OPEN_3`，由 Env 按外部 `direction` 自动生成 LONG 或 SHORT。不存在逆向动作；这是比“尝试逆向再给负奖励”更严格的设计。在 Env 接受底层订单的入口仍需断言 `order.side == external_direction`，故意构造逆向指令的测试必须报 `DirectionViolation`，且不得修改 Ledger 和仓位。若 direction 为 `NO_TRADE`，mask `[1,0]`。若 6 手/两槽已满，同样 `[1,0]`。

这样 Dreamer 只学一件事：**WAIT 还是 OPEN**。它不控制止盈，也不控制同方向平仓和加仓尺寸。

### 3.2 两批生命周期

```text
EMPTY -> ENTRY_PENDING -> OPEN(3 lots) -> TP_PENDING
    -> [TP partially filled: retain residual & track exact executed size]
    -> TP_FULL_FILL -> reward_once -> EMPTY

其他可能结束：BROKER_MARGIN_LIQUIDATION / CONTRACT_FORCED_EXIT /
DATA_SIMULATION_END (settle for audit) / FILL_ERROR.
这些事件不能误报成功，也不能获取 reward。
```

槽位的标识不能只用 `0/1`：增设永不复用的 `batch_id`，同一 slot 循环复用时确保事件没有串单。撮合状态包含 `OPEN_FILLED`, `OPEN_PENDING`, `TP_PENDING`, `TP_FILLED`, `BROKER_FORCED_EXIT`。如果交易接口不支持每批独立持仓标识，内部虚拟批次必须与经纪商净持仓及订单逐笔成交核对。

### 3.3 Observation schema（初版）

```yaml
market_seq:    [60, F_MARKET]      # 使用现有已验证分钟特征，不加未来信息
entry_state:   [2, F_SLOT]        # 位置、入场价距现价、距TP、持仓/自然时间、订单状态；空槽零填充+occupied flag
account_state: [F_ACCOUNT]        # 可用保证金、持仓/占用槽位；诊断与合法性。原则上尽量小
signal_state:  [F_SIGNAL]         # direction (-1/0/1)，信号已过去时间、有效期状态
action_mask:   [2]                # [WAIT, OPEN_3]
```

Account equity / unrealized PnL 如果作为安全可观测状态存在，不意味着进入 Reward；也可初版只给合法性摘要以减少模型学到金额依赖。没有必要引入过多新指标。高低点、量、ATR 与历史 features 必须遵循同一不泄漏时间合同。

### 3.4 同方向连续开仓示例

```text
09:10 short OPEN_3 @ 900 -> slot-0 TP target 895 (executed-side level)
09:35 short OPEN_3 @ 910 -> slot-1 TP target 905
10:02 slot-1 TP_FULL_FILL @ 905 -> reward 60/(60+27) => free slot-1
10:05 model OPEN_3 @ 908 -> reuses slot-1, TP target 903
... slot-0 remains @ 900, no negative reward for current floating loss
```

上述价格只是说明。实盘需要以实际开仓成交均价、可成交平仓价格和交易 Tick 精度计算止盈委托，不把 `bar low` 当成必然成交。

## 4. Step 的严格时间与成交协议

**决策因果性契约**：给模型的 obs_t 只包含截至已完成 bar t 的可用信息；Agent 在 t 发出 `OPEN_3` 请求，允许的最早模拟成交是下一根真实可交易 bar (`t+1`) 的 open（或有独立成交数据证明的稍后成交）。不能在 t 的 close 上成交。

每次 step 建议：

1. 验证 obs_t 发布的 action_mask 与 state_version，拒绝过期/非法请求。
2. 如果 action=OPEN_3，生成订单请求，不立即以 t 的 close 记成交。
3. 推进到下一个有效市场事件（跳过 mask=0 的缺失/休市补行），以指定成交模型撮合 pending entries、existing TP orders。
4. 新开仓使用成交均价与 tick size 计算 TP 价，立刻创建该批 TP 委托。是否可能当根成交需要时间先后证据；1m OHLC 测试中采用明确、保守的一致规则。
5. 收集 `TP_FULL_FILL` 事件，按 batch_id 去重；一次发 `60/(60+T)`，含同一 step 多批独立 TP。
6. 更新 Ledger、持仓、保证金、margin/contract guards、market features/obs、action_mask。
7. 输出 `reward`、`terminated/truncated` 和审计指标。

必须定义并测试：高低价同时跨过多个目标、价格跳空、限价委托挂单但未成交、价格触及瞬间的队列风险、订单部分成交、交易所涨跌停无对手盘、交易所交易时段/夜盘/节假日、撮合失败重试、手续费/滑点。V1 可以先用严格的 1 分钟保守 fill 协议研究，再用 Tick 或实际成交回放确认，不得把理想触价收益当实盘结果。

## 5. Reward 实现合同

建议新的 Reward class `FastTakeProfitReward`（`RewardAuditMixin`），只消费 Env 生成的 `TP_FULL_FILL` 且未曾消费的事件。它**不查询 equity delta，也不在每日结束、止损、invalid action、margin call 时发正负分**。

伪代码：

```python
class FastTakeProfitReward(RewardAuditMixin):
    def __call__(self, obs=None):
        score = 0.0
        for event in self.env.consume_unrewarded_events('TP_FULL_FILL'):
            if event.batch_id in self._rewarded_batches:
                raise AssertionError('duplicate TP reward')
            elapsed_min = (
                event.exit_fill_time - event.entry_fill_time
            ).total_seconds() / 60.0
            assert elapsed_min >= 0
            score += 60.0 / (60.0 + elapsed_min)
            self._rewarded_batches.add(event.batch_id)
        self._set_reward_debug(self._make_audit(total=score,
            tp_speed=score, tp_count=self.env.last_tp_count))
        return float(score)
```

独立 Reward 单元测试必须包含：1 分钟约 0.98360656、5 分钟约 0.92307692、60 分钟 0.5、跨夜 12 小时 `60/780`、失败平仓 0、持仓浮亏 0、两批同一时刻成功等于两次之和、重复事件异常、末尾未止盈 0、无单 WAIT 0。

**重要**：真实净收益统计不得出现在 `reward_debug.total` 之中；放到 `log/env/account_net_pnl`、`log/env/equity`、`log/env/drawdown`。`reward_audit_v1` 的既有字段如 `pnl`, `dd`, `fee` 可置零，仅表示未参与奖励；业务特有明细可作为额外 audit keys，需 `validate_reward_audit()` 通过。

## 6. 合约生命周期、账户审计与 Episode

**无用户定义持仓时限 ≠ 合约可无限持有。** 对 FG2701 和历史 FG 合约逐份读取真实最后可持有日、自然人限制、期货公司提前平仓与合约可交易日历。遇到强平/到期清算要真实结算亏损并记录终止原因，但其 Reward = 0。这是运行约束，不是另一个优化目标。

**不能利用 Episode reset 逃避坏仓位：** 当前旧代码到 `end_idx`、`max_steps` 有强平，或者 truncate 之后完全丢失仓位。新 Env 应将训练 Episode 的自然边界尽量放在真实合约/允许持有期限终点；不能任意切 345 根 bar 后免费重置。训练若采用人为 chunk 截断，必须完整持久化仓位、Ledger、订单和信号状态，并在 continuation 中恢复；否则仅准用于无仓位的 warm-up 片段。训练与验证的切片边界需说明还有多少可交易时间，不能将到期前 1 分钟开仓且未成功的交易当成完全中性的永续机会。

保留自动 margin check、最大仓位 6、方向 guard、合法交易时段/数据有效性 guard、到期强平、可用资金检查。用户明确不希望把浮亏、权益、速度以外的惩罚加入 Reward；因此这些是**外层强制约束或独立评价日志**，并不改变业务 Reward。

需报告 Reward 的盲点：一个模型可能用第一槽位长时间扛亏，只靠第二槽位频繁赚 5 点得到高分。此行为不会因 V1 Reward 扣分；评估必须清晰展示，但不偷偷修改 Reward。若模型实际选择 WAIT 到零交易，应诊断稀疏奖励问题，不应通过虚构成交或盈利奖励来强迫交易。

## 7. 数据与训练协议

1. 数据源优先使用已有 `market_data.fut_bar_1m_v2` 的**真实 FG 合约**分钟数据，并通过 `fut_continuous_map_v2` 辅助确定合同/行情上下文，绝不在换月间无成本拼接持仓。当前仓库 `JuejinBarSource` 默认使用单 `future_symbol` 和 14 天窗口，必须开发多历史合同离线训练路径，不能将这个 live 窗口当作多月训练数据。
2. 训练/验证/测试按时间和合约周期分开。相邻合同/跨市场阶段可能共享行情窗口，分区须防止相互污染；使用 manifest/hash/seed protocol。使用你给定的真实方向信号时间表为正式验证输入；可先进行全程 short-only/long-only 环境验证，但不把未来方向事后选择结果当实盘能力。
3. V1 每轮 Episode 保持外部方向固定，方向为 `NO_TRADE` 时模型只能 WAIT；以后要支持方向信号改变，必须明确已有仓位如何处理、是否允许新方向开仓，不能悄悄开反向仓。
4. 使用现有 market_seq 60-step 作为初始窗口；`entry_state` 新增最少字段；不混入未来成交、目标何时触发等标签。对 shift(-1)、backfill、当日完整日线及连续合约 rolling 重算做泄漏审计。
5. 保留 Action Mask real/imag 全链路；新 `Discrete(2)` 应由初始化自动确定 availhead 输出维度；必须检查 `WAIT` true mask 恒 1、持仓满 6 true mask `[1,0]`、actor/warmup 训练在 predicted mask 错误时不会选择非法 real action。
6. 初始实验 `batch_length`, `imag_length`, `horizon` 原样冻结一组，报告“止盈耗时 > imagination horizon 比例”。Dreamer 的 imagination 长度可能只有 15 分钟级，几十分钟至跨天的延迟正奖励容易发生信用分配困难；如不学，需要先做稀疏奖励传播诊断，而不是加浮盈/权益项。可以研究延长 imagination、经验重放采样或独立辅助事件时间预测头，但它们不能改变 Reward 的业务定义。

## 8. 实验分组与验收

固定同一合同集合、方向时间表、成本/撮合、手续费、资金/上限，至少比较：

| 组别 | 入场策略 | Reward |
|---|---|---|
| `B0` | 当前可开就开，达到 5 点自动平 | 不训练，规则基线 |
| `B1` | 固定间距/回撤规则，仅同向 | 不训练，规则基线 |
| `B2` | 按合法候选时刻随机选择入场 | 不训练，matched random |
| `D1` | DreamerV3，WAIT/OPEN_3 | 每批成功 +1 |
| `D2` | DreamerV3，WAIT/OPEN_3 | 每批成功 `60/(60+T)` |

**评估不是只看 reward：**

- **训练主目标**：累计完成 TP 速度奖励（D2）；完成次数、完成批次的 median/p90/max 等待分钟；时间越短、成功批次越多分越高。
- **真实性审计**：成交量与订单状态、成交价格差是否真有 5 点、跨夜持仓时长、双槽全满百分比、未止盈批次数、到期/强平次数、无单率、净 PnL、包括浮亏的最大权益回撤、占用保证金、账户回放 Ledger 对账。
- **统计有效性**：多个独立合同/市场状态/seed 的样本外数据；报告真实交易样本数和分布，不允许按单次最好的 checkpoint 挑选 test；验证集选模型，测试集锁定后只评一次。

**最低正确性 Gate（不代表策略赚钱）**：零非法方向提交，零超过 6 手，零重复 reward，全部 Reward 事件必须有真实完整 TP 成交，ledger 数值守恒，所有成交时间晚于该动作的可知时间，所有未结束订单在 episode 边界有解释；`sum(reward)` 严格等于 `sum(60/(60+T_i))`；D1/D2 和三个规则基线均能产生可审计结果。更高的策略 gate：D2 样本外能稳定优于 B0/B1/B2 且成本/成交假设不过分乐观，否则不进入下一阶段。

## 9. Codex 分阶段实施与文件清单

### Phase 0 — 冻结合同与环境核对（先做）

- 为两个 repo 建分支 `feat/fg-entry-v1`，记录 HEAD、依赖、数据 manifest。旧配置/模型/回测代码冻结，任何已有测试失败均要报告原因。
- 完成本文第 1/4/6 节中交易规则、事件时序、Episode 与撮合 schema 的数据类和单测。
- 验证是否有 FG 实际历史数据、包含 `open, high, low, close, volume, eob, trading_date, symbol` 以及精确分钟时钟；无外部数据时允许 synthetic fixture，但不得声称策略回测成功。

### Phase 1 — Env 账户与双槽位（不接 Dreamer）

建议新建/扩展：

```text
src/gym_trading_env/envs/fixed_direction_entry_env.py  # 新 Gym Env，不改旧 step
src/gym_trading_env/envs/entry_batch.py                # Batch/Order/Fill/TP events
src/gym_trading_env/envs/entry_execution.py            # Next-open, TP, gap, fees
src/gym_trading_env/envs/entry_mask.py                 # WAIT / OPEN_3, guards
src/gym_trading_env/rewards/fast_take_profit_reward.py # RewardAuditMixin
src/gym_trading_env/envs/entry_config.py               # 独立严格配置
configs/fg_entry_v1/fg2701_template.yaml
```

能不改现有库对象就组合使用。如果需要改 `PositionManager` / `Ledger`，必须只新增显式方法，保留旧签名与旧行为。

### Phase 2 — 基线、测试与审计

新增：

```text
tests/entry_v1/test_entry_action_mask.py
tests/entry_v1/test_two_slot_reentry.py
tests/entry_v1/test_take_profit_fill_timing.py
tests/entry_v1/test_reward_only_speed.py
tests/entry_v1/test_multiday_contract_expiry.py
tests/entry_v1/test_account_ledger_and_fees.py
tests/entry_v1/test_no_future_leakage.py
tests/entry_v1/test_episode_reset_and_partial_fill.py
tools/fg_entry_v1_baselines.py
tools/fg_entry_v1_replay_audit.py
```

完成 B0/B1/B2，输出 CSV/JSON：`entry_events.csv`、`batch_lifecycle.csv`、`reward_events.csv`、`account_equity.csv`、`summary.json`，在无真实数据时先跑 synthetic 自测。

### Phase 3 — 梦境模型接入

- 注册 `FixedDirectionEntryEnv-v0`，加入 `dreamerv3/configs.yaml` 新 profile `fg_entry_v1`，保持旧 `action_mask_formal` 不变。
- 无需优先改变 `dreamerv3/agent.py` 中的 RSSM/Actor/critic 主算法；沿用 2-action `MaskedCategorical` 与 availability head。仅当 shape/状态预测测试证实不能支持新 schema 时，以最小改动调整。
- 新增 `dreamerv3/tests/test_fg_entry_v1_mask.py`、`test_fg_entry_v1_train_smoke.py`；训练 D1/D2。检查 no-trade、imagined mask fallback、世界模型对正奖励事件的预测、event-to-reward 对齐，避免旧系统“模型一直 FLAT”和 Reward score 与 PnL 混淆。

### Phase 4 — 稳健回测及部署分离

- 多年 FG 历史、多市场阶段、数个 seed、完全样本外评估；Tick/盘口 fill 校验再决定执行落地。
- **实盘连接器必须是后续独立阶段**，先展示纸面仿真；真实接口要额外实现成交报告回写、指令去重、网络异常、Broker 仓位与本地状态对账、可撤/不可撤订单、强平和风控。

## 10. 一份供 Codex 直接使用的执行指令

> 可以把下面的整段作为 Codex 的任务说明，并附上本文件。要求 Codex 按阶段提交，不要一次性大范围重构。

```text
你可以访问两个本地 Git 仓库：
  ~/Documents/work/gym-trading-env
  ~/Documents/work/dreamerv3

首先核对 HEAD 是否分别对应 9699286 / 92af879；如 HEAD 不同，重新审阅差异，不得直接套用旧行号或提交覆盖现有修改。保持旧 CustomTradingEnv、旧 checkpoint 和 reward 实验完全可回归。

请严格按《FG_DreamerV3_Entry_V1_Code_Review_and_Codex_Plan.md》Phase 0 → Phase 1 → Phase 2 → Phase 3 顺序实施，每阶段独立提交并报告：diff 范围、测试、命令、通过/失败、阻断因素。

业务契约：方向由外部给出；Dreamer 仅 WAIT(0) / OPEN_3(1)，开单 side = 指定方向；每批 3 手、最多 2 批；每批 5 价格点真实止盈成交后自动平仓、释放槽位可循环；允许隔夜扛单，无 60 分钟或其他策略持仓时限；无策略止损；真实保证金、合约到期等为 Env Guard。

Reward 严格限制为每次真实完成 TP 事件 60/(60+T_minutes)，其他时刻 0；不要加入权益、MTM、浮亏、持仓时间单独扣分、无效操作负奖、日内次数惩罚。T 默认为实际成交自然时间分钟，含隔夜。任何强平/到期退出不能领取 TP Reward。账户与风险必须审计，但不进入 Reward。

成交与因果性：obs 使用已完成当前分钟的信息，策略下单最早在下一根有效可交易分钟处理，回测不能使用同一 bar 完整特征后再在其 close 成交。必须正确处理手续费、bid/ask、TP 委托、部分成交、同一分钟触价、交易时段、数据缺失、合约到期。

先实现新独立 Env，复用现有 Ledger、PositionManager、行情特征，不直接 subclass 并重写旧 CustomTradingEnv 大量 target-position 内部方法。保持 dreamerv3 Agent 原核心不动优先。正式训练前必须先使规则基线、合成行情单测、Reward 审计与新 Env 守恒测试全部通过。

不要直接启用 live_mode、RPC_ORDER_ENABLED 或真实下单；不自动更改数据库表结构，不自行使用不明确的真实费率/合约到期日。缺少真实 FG 数据或方向日志时只完成代码+synthetic smoke，明确报告阻断原因。

最终产出：变更清单、配置文件、测试与日志、B0/B1/B2 基线结果、D1/D2 训练入口及复现命令、待完成风险清单。任何经济结果必须注明 train/validation/test 边界及成交协议。
```

## 11. 最后评审结论

- 这次的任务比旧系统简单得多，**可复用以前扎实的持仓/保证金/复式记账/Action Mask 与 DreamerV3 基础设施**。
- 技术上最难的不是公式，而是**能否在不制造时间泄漏、不遗漏挂单和跨日状态的条件下，把两个 3 手批次的真实完成事件一致传给 Reward**。
- 训练上的最大不确定性是：**稀疏、延迟的 5 点完成奖励是否能被 DreamerV3 的短 imagination + 世界模型稳定学到**。必须通过 D1/D2 实验回答，不应提前保证盈利。
- 已实现速度 Score 与真实赚钱并非同一个数学目标。按用户明确要求，V1 保持单目标 Reward，真实权益只负责风控/审计。任何变更这个契约须另开版本、征得用户确认。

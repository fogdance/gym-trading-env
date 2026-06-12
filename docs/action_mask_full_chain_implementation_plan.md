# Gym Trading Env + DreamerV3 Action Mask 分阶段全链路实施方案

## 0. 文档状态

- 状态：方案评审稿，尚未修改运行代码
- 交易环境仓库：`~/Documents/work/gym-trading-env`
- DreamerV3 仓库：`~/Documents/work/dreamerv3`
- 前置分析：`docs/invalid_action_technical_report.md`
- 自动 warm-up 训练手册：`docs/action_mask_automatic_warmup_guide.md`
- 当前阶段：开发阶段，不要求兼容旧 observation、replay、checkpoint 或训练结果
- Phase 1A 范围：先验证非 EOD action-mask 主链路
- Phase 1B 范围：纳入 `near_eod` 和 `market_closed` 动态可达性
- 永久排除：外部 RPC/broker；`force_flatten_eod` 和 session-end cleanup 与 agent 目标动作分离
- 本文目标：把 action mask 从环境状态、真实采样、replay、world model、imagination、actor loss、评估与监控完整接通，并给出可分阶段验证的实施细节

本文取代前置分析中的粗略实施建议。前置分析仍可作为问题背景和 Monte Carlo 证据使用。

### 0.1 Review 结论处理

本版已吸收方案 review 的四项关键建议：

1. EOD 不再永久排除：
   - Phase 1A 先验证非 EOD 主链路；
   - Phase 1B 强制纳入 `near_eod` 和 `market_closed`；
   - `force_flatten_eod` 与 agent action 分离。
2. 一步 prior availability loss 从可选实验提升为 Phase 4 必选项和正式验收门槛。
3. 状态版本号作为长跑主机制；复杂 fingerprint 仅作为开发期 fail-fast/debug assert。
4. 增加 action cardinality、chosen action 分布和交易活性指标，防止“永远 FLAT”的伪修复。

---

## 1. 结论与把握

### 1.1 最终建议

采用“真实 mask + 预测 mask”的双路径设计：

1. 环境以唯一、无副作用的目标仓位预检函数生成真实 `action_mask_t`。
2. 真实采样、评估和实盘直接使用环境提供的真实 `action_mask_t`。
3. Replay 自动保存 `action_mask_t`，并与同一时刻的 `action_t` 对齐。
4. Dreamer world model 增加独立监督的 availability head，学习从 latent state 预测 mask。
5. Imagination 中没有真实环境，因此使用 availability head 预测的 hard mask。
6. 预测 hard mask 对 actor 停止梯度；availability head 和 latent representation 只通过真实 replay mask 的监督损失训练，不接受 actor loss 的梯度。
7. Policy 的采样、argmax、log-probability 和 entropy 必须全部基于同一个 masked categorical 分布。
8. 环境执行层仍保留最终校验、账务原子性和回滚；mask 不能替代执行安全检查。

核心形式是：

```python
mask_logits = availability_head(latent)
predicted_mask = stop_gradient(binarize_nonempty(mask_logits))
masked_policy = masked_categorical(actor_logits, predicted_mask)
```

这里的 `stop_gradient` 表示：

- actor 可以学习“在允许动作中如何分配概率”；
- actor 不能通过 policy loss 操纵 availability head，把不喜欢的动作伪装成不可用；
- availability 预测路径只根据环境真值 `action_mask` 学习动作是否可用。

### 1.2 当前把握

| 部分 | 把握 | 说明 |
|---|---:|---|
| 根因定位 | 97% | 无效操作来自动态可达性约束未进入 actor 分布 |
| 环境统一预检与真实 mask | 98% | mask 与执行共享同一 decision；near-EOD/闭市在第二阶段纳入 |
| Env -> Adapter -> Driver -> Replay 对齐 | 98% | 当前数据链路已确认可自动携带新 observation |
| 真实 policy masked sample/argmax | 95% | 改动局部，容易做严格单元测试 |
| Availability head 监督训练 | 90% | 可复用现有 `MLPHead(binary)`，并用独立 BCE 训练 latent 可达性表示 |
| Imagination mask 时序与 actor loss 对齐 | 82% | 是最难部分，需要 shape、时序和梯度专项测试 |
| 完整落地可行性 | 90% | 纳入 near-EOD/闭市和 prior 监督后路径完整，但改动与验证面增加 |
| 范围内真实环境无效动作归零 | 98% | 真值 mask 和统一 execution decision 可形成结构性不变量 |
| 最终交易收益改善 | 不作保证 | action mask 修复动作语义，不保证策略盈利能力 |

### 1.3 “100% 可靠”的准确边界

不能把完整 Dreamer 训练链路称为 100% 可靠。即使外部 RPC 和强制清算不在范围内，imagination 仍使用学习得到的 predicted mask，它相对真实环境规则的准确率只能测量，不能数学保证为 100%。

可以做成结构性 100% 不变量的是：

1. 真实 policy 给定 true mask 后，`MaskedCategorical` 绝不选择 mask=0 的动作。
2. 环境 true mask 与执行共享同一个 `TargetTransitionDecision`，不再维护两份业务规则。
3. 范围内、非 terminal、环境内部 transition 的 `mask_execution_mismatch` 必须为 0。
4. Imagination 中 actor 绝不选择 predicted mask=0 的动作。

不能承诺 100% 的是：

1. Predicted mask 与真实环境 mask 的语义完全一致。
2. 未发现的软件缺陷、硬件错误或执行期系统异常永不发生。
3. Action mask 修复后训练一定收敛或策略一定盈利。

因此本方案采用两层验收：

```text
真实环境结构性约束：要求 100%，违反即 P0 缺陷
imagined mask 预测质量：使用 false-positive 等指标设门槛，不能宣称 100%
```

如果要求“完整训练链路语义也必须 100%”，就不能使用学习得到的 availability head。必须在 imagination 中携带并精确更新一套符号化交易状态，再由同一环境规则确定性计算 mask。当前合法性包含价格、权益、保证金和平仓后资金，这实际上要求把精确交易模拟器嵌入 world-model rollout，复杂度和 Dreamer 的学习式 imagination 目标冲突，不建议作为本次方案。

### 1.4 为什么不是只改环境

只在 `env.step()` 内把非法动作替换成 HOLD，会造成：

- actor 记录的是动作 A，环境执行的是 HOLD；
- replay 中动作与状态转移不一致；
- world model 学到错误的动作后果；
- imagination 中 actor 仍然大量选择非法动作；
- actor loss 的 log-probability 和 entropy 仍基于完整动作空间。

因此，修复必须覆盖完整训练链路。

---

## 2. 实施边界与冻结假设

### 2.1 本次范围

本次实施覆盖：

- 三目标仓位动作空间 `SHORT / FLAT / LONG` 的动态合法性；
- 环境真实 mask；
- Dreamer 真实 policy mask；
- Replay mask 监督数据；
- World model latent availability prediction；
- Dreamer imagination mask；
- Actor loss masked distribution；
- train/eval 的动作选择模式；
- 环境与 Dreamer 的诊断指标、测试和验收。

### 2.2 明确不做

- 不迁移旧 replay；
- 不加载旧 checkpoint；
- 不兼容旧 observation schema；
- 不为其他 Gym 环境设计通用 action-mask 协议；
- 不扩展到连续动作；
- 不改变奖励目标或交易策略目标；
- 不承诺 mask 修复后收益率一定提升。
- 不处理 live 外部 RPC、broker 接单、成交确认和仓位 reconciliation；
- 不把 `force_flatten_eod`、episode 结束清算、止损、止盈或突发风控定义为 agent action mask；
- 不对上述永久排除项作可靠性承诺。

### 2.3 必须冻结的业务假设

本方案依赖以下约束：

1. 对外动作空间固定为 `spaces.Discrete(3)`。
2. 动作索引顺序固定为：

   ```text
   0 = SHORT
   1 = FLAT
   2 = LONG
   ```

3. `intraday_single_position=True`。
4. 同时存在多仓和空仓属于环境状态不变量破坏，应直接报错，不能尝试生成 mask。
5. 每次目标仓位变化只操作 slot 0。
6. 反手是单步原子操作，包含“平当前仓 + 新开反向仓”。
7. 反手消耗一次当日开仓次数。
8. Phase 1A 验证配置必须关闭 EOD 相关动作规则，并确保采样 step 可交易：

   ```yaml
   session_policy:
     block_open_near_eod: false
     force_flatten_eod: false
   ```

9. Phase 1B 必须重新开启并纳入 `block_open_near_eod`，同时纳入 `market_closed` true mask；`force_flatten_eod` 仍保持关闭或重构为独立外生动作。
10. Episode terminal cleanup、止损、止盈和突发风控属于外生转移，不纳入 action-mask 合法性契约。
11. 本方案的“执行可靠性”只指环境内部执行，不包含外部 RPC/broker。

由于当前不要求兼容，环境初始化时应对这些假设 fail fast，而不是保留模糊行为。

---

## 3. 当前链路复核结果

### 3.1 环境动作模型

环境当前暴露：

```python
self.valid_actions = [TargetPos.SHORT, TargetPos.FLAT, TargetPos.LONG]
self.action_space = spaces.Discrete(3)
```

目标仓位到执行动作的规划如下：

| 当前仓位 | 请求目标 | 规划执行动作 |
|---|---|---|
| `FLAT` | `SHORT` | `SHORT_OPEN0` |
| `FLAT` | `FLAT` | `HOLD` |
| `FLAT` | `LONG` | `LONG_OPEN0` |
| `SHORT` | `SHORT` | `HOLD` |
| `SHORT` | `FLAT` | `SHORT_CLOSE0` |
| `SHORT` | `LONG` | `FLIP_SHORT_TO_LONG` |
| `LONG` | `SHORT` | `FLIP_LONG_TO_SHORT` |
| `LONG` | `FLAT` | `LONG_CLOSE0` |
| `LONG` | `LONG` | `HOLD` |

### 3.2 当前合法性规则分散

规则目前分散在以下位置：

- `step()`：
  - 目标仓位解码；
  - 当前仓位到执行动作规划；
  - 临近收盘开仓阻断；
  - 市场开闭市判断；
  - 将请求动作改写为 HOLD。
- `_long_open()` / `_short_open()`：
  - 单仓位限制；
  - 每日开仓次数；
  - 最大仓位；
  - 保证金和手续费。
- `_flip_long_to_short()` / `_flip_short_to_long()`：
  - 持仓存在；
  - 每日开仓次数；
  - 平仓 quote；
  - 平仓后资金；
  - 新仓保证金和手续费。
- `_refresh_agent_state()` 和 `get_oracle_snapshot()`：
  - 重复计算部分 `can_open/can_close` gate；
  - 这些 gate 不是最终目标仓位 mask。

如果直接在现有代码旁再实现一份 `_get_action_mask()`，规则会出现第三份副本，后续必然漂移。

### 3.3 当前 observation gate 不能直接作为 mask

现有 observation 包含：

```text
obs_market_open_t
obs_can_long_open_t
obs_can_short_open_t
obs_can_long_close_t
obs_can_short_close_t
obs_entries_left_frac_t
```

这些 gate 有两个问题：

1. 它们描述底层开仓/平仓许可，不描述目标仓位可达性。
2. 它们未完整覆盖资金、最大仓位和原子反手。

例如持有 LONG 时，请求 SHORT 表示反手。此时 `can_short_open=0`，但反手是否允许需要根据平多后资金、开仓次数和新空仓限制重新判断。

### 3.4 Env -> Dreamer 数据对齐

当前链路如下：

```text
env state s_t
  -> env._get_obs() 产生 obs_t
  -> FromGymnasium 保留 obs_t
  -> Agent.policy(obs_t) 选择 action_t
  -> Driver 保存 {obs_t, action_t}
  -> Replay 保存全部非 log/* 字段
  -> Agent.train() 从 replay 读取 obs_t 和 action_t
```

因此，只要环境在 `obs_t` 中提供 `action_mask_t`，Replay 会自动保存，并与 `action_t` 正确对齐。

存在一个必须显式处理的 terminal 例外：

- 普通 `embodied.core.Driver` 在 `obs["is_last"] == True` 时，会把 policy 返回但不会执行的动作置零；
- 该 terminal dummy action 可能不满足 terminal observation 的 action mask；
- 它不会被环境执行，下一次 reset 时 RSSM 也会通过 reset mask 忽略该 prevact；
- 因此 replay 的 mask/action 合法性断言只适用于 `is_last == False` 的真实可执行 transition。

并行 actor 当前保留 terminal observation 上的 policy action，但该动作同样不会被执行。测试和指标必须区分“policy 选择动作”“Driver dummy action”和“环境实际执行动作”。

以下文件已复核，不需要为了传输 mask 修改：

- `~/Documents/work/dreamerv3/embodied/envs/from_gymnasium.py`
- `~/Documents/work/dreamerv3/embodied/core/driver.py`
- `~/Documents/work/dreamerv3/embodied/core/replay.py`

### 3.5 Gym space 类型约束

`FromGymnasium._convert()` 当前通过 `hasattr(space, "n")` 判断离散空间。

因此不要使用 `spaces.MultiBinary(3)`：它可能被错误转换成标量离散空间。

环境必须使用：

```python
spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
```

这样 Dreamer 收到的是：

```text
action_mask: elements.Space(float32, shape=(3,), low=0, high=1)
```

### 3.6 Dreamer 真实 policy 当前问题

当前 `Agent.policy()`：

```python
policy = self.pol(self.feat2tensor(feat), bdims=1)
act = sample(policy)
```

存在两个问题：

- 没有 action mask；
- 忽略 `mode`，因此 `eval` 仍随机采样；live 路径不在本次范围。

并行 actor 已正确把训练环境传 `mode="train"`、评估环境传 `mode="eval"`，问题只在 `Agent.policy()` 没有使用 mode。

### 3.7 Dreamer imagination 当前时序

当前时序是：

```text
start latent z_0
  -> policy(z_0) 选择 a_0
  -> dynamics(z_0, a_0) 得到 z_1
  -> policy(z_1) 选择 a_1
  -> ...
```

`RSSM.imagine()` 在当前 latent 上调用 policy，再进入下一 latent。因此 predicted mask 必须在选择动作前从当前 latent 预测。

训练代码之后构造：

```text
imgfeat: H + 1 个 latent state
imgact:  H + 1 个与 state 对齐的 action
```

`imag_loss()` 使用前 H 个 state/action。因此 actor loss 中使用的 masked policy 必须与 imagination 采样动作时使用的 mask 完全一致。

### 3.8 额外发现

以下问题与 action mask 改造直接相关，应一并处理：

1. 当前 categorical policy 的 `unimix` 配置未实际传入 `outs.Categorical`。
2. 如果后续启用探索混合，uniform 概率必须只分配给合法动作。
3. 当前 `invalid_action` 排除了 `ERROR_MARKET_CLOSED`，但闭市请求仓位变化仍然是被拒绝请求。
4. 当前反手函数宣称原子，但异常回滚主要恢复 ledger；如果平仓 commit 后新仓 commit 失败，position manager 和 realized PnL 可能未完整恢复。
5. 当前临近收盘和强制平仓逻辑会把请求动作改写或覆盖；Phase 1A 先关闭以验证主链路，Phase 1B 将 `near_eod` 纳入统一 decision，强制清算仍与 agent action 分离。

---

## 4. 冻结后的 action mask 语义

### 4.1 定义

`action_mask_t[i] == 1` 的含义：

> 在状态 `s_t`，请求目标仓位 `i` 可以按目标仓位语义被环境接受，并执行对应的目标转换。

它不表示：

- 动作一定盈利；
- 执行期间不可能出现系统异常；
- 环境之后不会因为 terminal cleanup、止损、止盈或风控发生外生状态变化。

### 4.2 Mask 顺序和类型

```text
action_mask.shape = (3,)
action_mask.dtype = np.float32
action_mask order = [SHORT, FLAT, LONG]
values            = 0.0 or 1.0
```

### 4.3 基本不变量

1. mask 永远至少有一个 `1`。
2. 当前目标仓位对应动作始终可用，因为它表示 HOLD。
3. mask 的动作索引必须与 `valid_actions` 完全一致。
4. `_get_obs()` 重复调用不得修改任何环境状态。
5. mask 中标记可用的动作，在相同状态立即执行时，不得因已知业务约束被拒绝。
6. 执行层仍可因不可预期系统错误失败，并必须安全回滚。

### 4.4 详细决策矩阵

#### 正常开市、允许开仓、资金足够

| 当前仓位 | Mask `[SHORT, FLAT, LONG]` |
|---|---|
| `FLAT` | `[1, 1, 1]` |
| `SHORT` | `[1, 1, 1]` |
| `LONG` | `[1, 1, 1]` |

持仓状态下，反向目标通过原子反手到达。

#### 当日开仓次数耗尽

| 当前仓位 | Mask |
|---|---|
| `FLAT` | `[0, 1, 0]` |
| `SHORT` | `[1, 1, 0]` |
| `LONG` | `[0, 1, 1]` |

反手包含新开仓，因此不可用。

#### Near-EOD 禁止开仓

Phase 1B 开启 `block_open_near_eod` 后：

| 当前仓位 | Mask |
|---|---|
| `FLAT` | `[0, 1, 0]` |
| `SHORT` | `[1, 1, 0]` |
| `LONG` | `[0, 1, 1]` |

允许保持当前仓位和持仓平到 FLAT；新开仓和反手包含开仓，因此不可用。

#### 闭市或当前 bar 不可交易

| 当前仓位 | Mask |
|---|---|
| `FLAT` | `[0, 1, 0]` |
| `SHORT` | `[1, 0, 0]` |
| `LONG` | `[0, 0, 1]` |

闭市时只能保持当前目标仓位。平仓也不能执行，因此 `FLAT` 在持仓状态下不可用。

#### 资金不足或目标方向达到最大仓位

- 当前目标/HOLD 仍可用；
- 平仓目标仍可用，前提是平仓 quote 和账务预检通过；
- 需要新开仓的目标不可用；
- 反手是否可用必须按平仓后资金精确计算，不能直接按当前 free margin 判断。

### 4.5 EOD 分阶段契约

Phase 1A 为降低同时改造的变量数量，暂不为 near-EOD 生成 action mask。验证配置必须关闭：

```text
block_open_near_eod = false
force_flatten_eod = false
```

Phase 1A 若检测到这些配置开启，应 fail fast，不能继续训练后再把 EOD 错误归因于 mask。

Phase 1B 必须将以下规则纳入统一 transition decision 和 true mask：

- `market_closed`：仅当前目标/HOLD 可用；
- `near_eod` 且禁止开仓：允许当前目标/HOLD 和平仓到 FLAT，禁止开仓与反手；
- unavailable reason 和 rejected reason 分别统计 `market_closed`、`near_eod`。

`force_flatten_eod` 不应继续在 agent action 执行后静默覆盖目标仓位。它应保持关闭，或重构为独立外生风险动作并单独记录，不进入 actor 请求合法性。

Episode 结束时的 terminal cleanup 不属于 agent 可再次选择的动作，不进入 mask/action 合法性统计。

### 4.6 被拒绝请求与无效动作

建议开发阶段重新定义指标：

```text
action_rejected = requested target 无法按请求执行
invalid_action  = action_rejected
execution_failed = 预检允许，但执行提交因系统或账务异常失败
mask_execution_mismatch = mask 标记可用，但立即执行时被已知业务约束拒绝
```

`ERROR_MARKET_CLOSED` 应计入 `action_rejected`，因为它代表请求目标无法执行。

如奖励层不希望惩罚某些拒绝原因，应单独通过 `reject_reason` 控制，不应修改 mask 语义或统计语义。

`execution_failed` 不应惩罚 actor。`mask_execution_mismatch` 属于环境 P0 缺陷，必须触发显著日志并在测试中归零。

---

## 5. 环境侧详细设计

### 5.1 目标

环境侧不能只是增加一个 observation 字段。必须建立单一合法性来源：

```text
同一份 transition evaluation
  -> 生成 action_mask
  -> 生成 agent_state gate
  -> 生成 oracle snapshot
  -> 驱动 step 执行
  -> 生成 reject reason 和日志
```

### 5.2 推荐新增数据结构

建议新增文件：

```text
src/gym_trading_env/envs/target_transition.py
```

推荐结构：

```python
@dataclass(frozen=True)
class TargetTransitionDecision:
    requested_index: int
    current_target: TargetPos
    requested_target: TargetPos
    planned_action: Action
    allowed: bool
    result_code: ForexCode
    reason: str
```

增加不可变 decision table：

```python
@dataclass(frozen=True)
class TargetTransitionTable:
    state_version: int
    decisions: tuple[TargetTransitionDecision, ...]
    debug_fingerprint: tuple | None = None
```

每个允许执行的 decision 必须携带 execution-ready quote：

```python
execution_quote: object | None = field(repr=False, compare=False)
```

`execution_quote`、`state_version` 和 `debug_fingerprint` 不得进入 observation 或 replay。

### 5.3 推荐新增核心接口

在 `CustomTradingEnv` 中新增：

```python
def _build_target_transition_table(self) -> TargetTransitionTable:
    ...

def _action_mask_from_table(self, table: TargetTransitionTable) -> np.ndarray:
    ...

def _execute_transition(self, decision: TargetTransitionDecision) -> ForexCode:
    ...
```

接口要求：

- `_build_target_transition_table()` 无副作用；
- 默认不写 warning 日志，因为 `_get_obs()` 会频繁调用；
- 不修改 ledger、position、counter、price cache 或 metrics；
- 一次性为三个目标仓位生成 execution-ready decision 和 quote；
- observation 缓存该 table，并从同一 table 生成 float32 mask；
- `step()` 直接消费 observation 对应 table 中的 decision，不重新计算规则；
- 所有影响动作合法性的环境状态变更必须递增 `_legality_state_version`；
- 执行前比较 `table.state_version == self._legality_state_version`，不一致则 fail fast，不能静默重算；
- 每次状态变更和下一 observation 生成时使旧 table 失效；
- `_execute_transition()` 只接受 `allowed=True` 的 decision；
- 执行函数保留最终防御校验和回滚。

开发期额外计算 `debug_fingerprint`，覆盖影响合法性的关键输入，例如：

```text
current_step / market-open flag / near-EOD flag / current target
position slots and sizes
cash / used margin / unrealized PnL
entries_used_today
trade lot / max position / leverage / fee / spread
```

开发期如果 fingerprint 或 state version 不一致，直接 fail fast。主链路和长跑训练依赖单一 decision table + 轻量状态版本号；fingerprint 保留为 debug assert 和诊断指标，不进入每步生产热路径。

该设计比“observation 计算一次 mask、step 再调用同一函数重算”更强，同时避免将 Decimal、PnL 和价格缓存的完整表示长期放入热路径。

### 5.4 统一预检顺序

构建每个 target decision 时按以下固定顺序判断：

1. 验证目标索引和单仓位状态不变量。
2. 计算 `current_target`。
3. 若 `requested_target == current_target`：
   - `planned_action = HOLD`；
   - `allowed = True`；
   - 不受闭市、次数、资金限制。
4. 读取当前 bar 是否可交易：
   - 非当前目标且闭市，返回 `ERROR_MARKET_CLOSED`。
5. 规划底层执行动作。
6. 若动作包含开仓：
   - Phase 1B 检查 near-EOD open block；
   - 检查每日开仓次数；
   - 检查目标方向最大仓位；
   - 计算开仓 lot、执行价、保证金和手续费；
   - 保证 position size 严格大于 0。
7. 若动作包含平仓：
   - 检查对应持仓存在；
   - 生成 close quote；
   - 验证 close quote 和账务结果可提交。
8. 若动作是反手：
   - 基于 close quote 计算平仓后可用资金；
   - 检查新仓资金；
   - 生成完整 flip quote。
9. 返回 allowed decision 和可选 quote。

### 5.5 开仓、平仓与反手 quote

为了避免预检和执行规则再次重复，建议将现有执行函数拆成：

```text
quote operation: 无副作用，返回 quote 或 ForexCode
commit operation: 消费 quote，执行账务和仓位变更
```

推荐私有接口：

```python
_quote_long_open(...)
_quote_short_open(...)
_quote_long_close(...)
_quote_short_close(...)
_quote_flip_long_to_short(...)
_quote_flip_short_to_long(...)

_commit_long_open(quote)
_commit_short_open(quote)
_commit_long_close(quote)
_commit_short_close(quote)
_commit_flip_long_to_short(quote)
_commit_flip_short_to_long(quote)
```

这样 mask、step 和执行函数使用同一份业务计算。

### 5.6 执行安全

即使 mask 和 quote 已通过，commit 仍必须：

- 对 ledger 使用完整事务 snapshot/restore；
- 对 position manager 使用可恢复事务或完整 snapshot/restore；
- 对 realized PnL 和相关账户状态使用完整恢复；
- 确保失败后 entries、仓位、现金、保证金、PnL、开仓次数均不变化。

当前 `Ledger.restore()` 只恢复账户余额，不会删除 `Ledger.post()` 已追加的 journal entry。实施时事务 snapshot 必须同时记录 `len(ledger.entries)`，回滚时恢复余额并截断新增 entries。

当前反手异常路径主要恢复 ledger 余额，实施时还应补齐 position slots、`closed_trade_profits`、realized PnL、trade records 和开仓次数的回滚，或重构为 commit 阶段不会在部分提交后失败。

### 5.7 `step()` 重构

当前 `step()` 中“先规划、再改写为 HOLD、再执行”的流程应改为：

```python
requested_index = validate_action(action)
requested_target = self.valid_actions[requested_index]
table = self._consume_current_transition_table()
assert table.state_version == self._legality_state_version
decision = table.decisions[requested_index]

self.requested_target = requested_target
self.planned_action = decision.planned_action

if decision.allowed:
    self.executed_action = decision.planned_action
    self.action_result = self._execute_transition(decision)
else:
    self.executed_action = Action.HOLD
    self.action_result = decision.result_code
```

要求：

- 不再静默把请求动作和执行动作混为同一个 `self.action`；
- `decision.allowed=False` 必须是严格 no-op；
- `action_result` 反映 agent 请求结果；
- 止损、止盈和 terminal cleanup 等外生动作另行记录，不进入 action-mask 合法性统计。

### 5.8 Observation 修改

在 `observation_space` 中加入：

```python
"action_mask": spaces.Box(
    low=0.0,
    high=1.0,
    shape=(3,),
    dtype=np.float32,
)
```

在 `_get_obs()` 中加入：

```python
table = self._build_target_transition_table()
self._current_transition_table = table
out["action_mask"] = self._action_mask_from_table(table)
```

必须保证：

- reset observation 含 mask；
- 每个 step 返回的下一 observation 含 mask；
- mask 对应 observation 当前状态，不对应上一步状态；
- `np.isfinite(action_mask).all()`；
- 值严格属于 `{0.0, 1.0}`；
- 至少一个动作可用。

### 5.9 Agent state 和 oracle 修改

现有 `can_long_open/can_short_open/can_long_close/can_short_close` 可暂时保留，但必须从统一 decision 派生，不能独立重复计算。

`get_oracle_snapshot()` 增加：

```text
agent_input.action_mask
agent_input.transition_decisions
```

测试 oracle 应使用统一 decision 结果，而不是复制环境规则。

### 5.10 日志和指标

环境 info/log 至少增加：

```text
log/env/action_rejected
log/env/execution_failed
log/env/mask_execution_mismatch
log/env/action_mask_valid_count
log/env/requested_target
log/env/planned_action
log/env/executed_action
log/env/reject_code
```

`reject_code` 推荐使用稳定整数。详细字符串 reason 可进入 debug 日志，不进入标量 logger。

外部 RPC/broker 不在本次范围内，不新增或修改相关日志与可靠性验收。

---

## 6. DreamerV3 侧详细设计

### 6.1 数据契约

Dreamer 收到的字段：

```text
obs["action_mask"]
shape:
  policy path: (B, 3)
  replay path: (B, T, 3)
dtype:
  float32
values:
  0.0 or 1.0
```

Actor action：

```text
act["action"]
shape:
  policy path: (B,)
  imagination: (B*K, H+1)
dtype:
  int32
```

### 6.2 Encoder 和 Decoder 决策

冻结方案：

- `action_mask` 保留在 encoder input 中；
- `action_mask` 从通用 decoder reconstruction 中排除；
- 使用独立 availability head 监督预测。

原因：

1. Posterior latent 应明确知道当前动作可达性。
2. RSSM prior 通过 KL 学习包含 mask 信息的 posterior dynamics。
3. Imagination 的 prior latent 才有机会预测未来 mask。
4. 通用 decoder 的 symlog MSE 不适合二值 mask，也会与独立 BCE 重复。

Agent 初始化改为：

```python
base_exclude = ("is_first", "is_last", "is_terminal", "reward")
enc_space = {k: v for k, v in obs_space.items() if k not in base_exclude}
dec_exclude = (*base_exclude, "action_mask")
dec_space = {k: v for k, v in obs_space.items() if k not in dec_exclude}
```

### 6.3 Availability head

使用现有 binary head：

```python
mask_space = elements.Space(bool, (3,))
self.avail = embodied.jax.MLPHead(
    mask_space,
    **config.availhead,
    name="avail",
)
```

加入优化模块：

```python
self.modules = [
    self.dyn,
    self.enc,
    self.dec,
    self.rew,
    self.con,
    self.avail,
    self.pol,
    self.val,
]
```

Posterior 监督损失：

```python
post_inp = sg(
    self.feat2tensor(repfeat),
    skip=self.config.avail_post_grad,
)
post_avail = self.avail(post_inp, 2)
losses["avail_post"] = post_avail.loss(obs["action_mask"] > 0.5)
```

默认配置：

```yaml
avail_post_grad: true
avail_prior_grad: true
avail_threshold: 0.5
availhead:
  layers: 2
  units: 512
  act: silu
  norm: rms
  output: binary
  outscale: 1.0
  winit: trunc_normal_in
loss_scales:
  avail_post: 1.0
  avail_prior: 1.0
```

默认应允许 posterior/prior availability BCE 同时训练 availability head 和 latent representation。原因是：

- `action_mask` 即使进入 encoder，如果没有相关损失梯度，encoder/RSSM 仍可能忽略它；
- availability head 最终要在 imagined prior latent 上工作，world model 必须学习可达性状态；
- 这与 reward/continue head 通过独立 world-model 目标训练 latent 的思路一致。

关闭对应 gradient 仅用于诊断和消融，不能作为默认正式训练方案。

### 6.4 一步 Prior Availability 监督

只在 `repfeat` posterior 上训练 availability 不足够。Posterior 已看到当前 observation，且 encoder 输入包含 true `action_mask_t`；它可能在 replay 上预测很好，但无法保证 imagined prior latent 可预测现实合法性。

因此一步 prior availability loss 是 Phase 4 的正式必选项，不是可选实验。

当前 `RSSM.loss()` 已计算与当前时刻对齐的 prior logits：

```python
prior_logit_t = self._prior(repfeat["deter"][:, t])
```

实施时让 `RSSM.loss()` 额外返回 `priorfeat`：

```python
priorfeat = {
    "deter": repfeat["deter"],
    "stoch": sample_from_prior(prior_logit),
    "logit": prior_logit,
}
```

`sample_from_prior()` 必须使用与 `RSSM.imagine()` 相同的 prior distribution 和 straight-through sampling 语义，不能把 posterior stochastic state 填入 `priorfeat`，否则会发生 observation 信息泄漏。可额外用 prior mode 做诊断指标，但正式 prior loss 与 imagination 的 latent 采样语义必须一致。

该 prior 表示在看到 `obs_t` 之前，仅根据前一 latent 和 `action_{t-1}` 预测的 `z_t`，目标必须与 `action_mask_t` 对齐：

```python
prior_inp = sg(
    self.feat2tensor(priorfeat),
    skip=self.config.avail_prior_grad,
)
prior_avail = self.avail(prior_inp, 2)
losses["avail_prior"] = prior_avail.loss(obs["action_mask"] > 0.5)
```

强制要求：

- `avail_post` 和 `avail_prior` 分别记录 loss、exact match、FPR、FNR；
- prior loss 必须能向 RSSM dynamics 传播梯度；
- prior target 与 `mask_t` 的时序必须通过 toy sequence 专项测试；
- imagined mask 没有真值，监控 cardinality、fallback 和分布漂移；
- prior 指标未达到门槛时，不允许进入正式 actor imagination 训练。

训练门禁建议：

1. Availability warm-up 阶段训练 world model、decoder、reward、continue、`avail_post` 和 `avail_prior`；
2. 暂停 policy/value imagination 参数更新，避免 actor 在低质量 predicted mask 上学习；
3. prior 指标连续多个报告窗口达到门槛后，再启用 masked imagination 的 policy/value loss；
4. 启用后若 prior FPR 或 empty fallback 明显恶化，停止正式训练并诊断，不能继续积累错误 actor 更新。

#### 6.4.1 自动 Warm-up 与正式训练硬门禁

正常训练只启动一次。Runner 在同一个 Agent、optimizer、replay 和 logdir 内维护
`warm-up -> formal` 状态机：

```text
fresh run
  -> actor/value/repval gate = 0
  -> 训练 world model 与 avail_post/avail_prior
  -> 连续报告窗口达到门槛
  -> gate = 1，立即保存 checkpoint
  -> 继续正式 masked actor 训练
```

自动模式硬约束：

1. fresh run 必须以 `agent.avail_actor_enabled=false` 启动，已开启 actor 时 fail fast；
2. 每个 report 周期聚合该周期全部训练 update 的 availability 指标，连续窗口同时满足
   post/prior accuracy、exact、prior FPR 和 imagination fallback 门槛后才允许切换；
3. 单个评估窗口必须达到内部最小训练样本量；不足时跨周期继续累计，不能用小样本窗口
   触发切换或正式训练退化停止；
4. 切换状态和动态 gate 必须进入 checkpoint；
5. 相同 logdir 恢复时必须恢复原阶段，formal checkpoint 不重新 warm-up；
6. formal 期间继续训练 `avail_post/avail_prior`，并持续检查相同门槛；
7. formal 连续多个报告窗口退化时 fail fast，不能继续积累 actor 更新；
8. predicted hard mask 对 actor 始终 stop-gradient。

因此不同模型尺寸、batch、replay 或环境配置不再需要用户手工建立独立 availability
资产。用户按最终正式配置启动一次，新配置会先在自身 replay 分布上自动 warm-up，再
自动进入正式训练。详细运行方式见
`docs/action_mask_automatic_warmup_guide.md`。

### 6.5 为什么预测 mask 不向 actor 反向传播

如果 actor loss 可以修改 mask head，actor 有可能通过降低某个动作的“可用概率”来逃避不利动作，而不是学习正确策略。

“独立监督损失”表示 availability 不由 actor 的收益目标训练，不表示它只能更新 head。

默认的正确梯度关系：

```text
真实 action_mask
  -> availability BCE
  -> 更新 availability head
  -> 更新 encoder/RSSM 的可达性表示

predicted hard mask
  -> stop_gradient
  -> 约束 actor distribution
  -> actor loss 只更新 actor 对合法动作的概率
```

不允许：

```text
actor loss
  -> predicted mask
  -> availability head
```

需要增加专项梯度测试证明此关系成立。

### 6.6 Predicted hard mask 与非空 fallback

推荐函数：

```python
def predicted_mask(avail_logits, threshold):
    probs = jax.nn.sigmoid(avail_logits)
    mask = probs >= threshold
    fallback = jax.nn.one_hot(
        jnp.argmax(probs, axis=-1),
        probs.shape[-1],
        dtype=bool,
    )
    mask = jnp.where(mask.any(-1, keepdims=True), mask, fallback)
    return jax.lax.stop_gradient(mask)
```

不建议 all-false 时固定回退 `FLAT`，因为闭市持仓状态下 FLAT 实际不可执行。

使用最高预测概率动作作为 fallback，且必须记录 fallback rate。fallback 是数值安全措施，不是正常训练路径。

### 6.7 Masked categorical distribution

建议在：

```text
~/Documents/work/dreamerv3/embodied/jax/outs.py
```

增加独立 `MaskedCategorical`，不要在 Agent 中零散修改 logits。

核心行为：

```python
masked_logits = jnp.where(mask, logits, -1e30)
```

不能使用 `-inf`，否则 entropy 中可能出现 `0 * -inf = NaN`。

`MaskedCategorical` 必须保证：

- `sample()` 永远返回合法动作；
- `pred()` 永远返回合法动作；
- `logp()` 基于 masked logits；
- `entropy()` 只计算合法动作；
- `unimix` 只在合法动作之间分配 uniform probability；
- mask 至少一个 true；
- 输入 logits 和 mask shape 严格匹配。

有效动作上的 unimix：

```python
policy_probs = softmax(masked_logits)
valid_uniform = mask / mask.sum(-1, keepdims=True)
probs = (1 - unimix) * policy_probs + unimix * valid_uniform
```

当前 `imag_loss()` 会在 distribution 暴露静态 `minent/maxent` 时计算 `rand/action`。Masked distribution 的最大 entropy 取决于每个状态的合法动作数量，不能继续使用静态 `log(3)`。

推荐 `MaskedCategorical` 不暴露静态 `minent/maxent`，改为单独记录：

```python
max_entropy = log(valid_count)
entropy_fraction = where(valid_count > 1, entropy / max_entropy, 0)
```

这样单合法动作状态不会产生除零，也不会被错误判断为探索不足。

### 6.8 统一 masked policy helper

在 `dreamerv3/agent.py` 中增加唯一 helper：

```python
def _policy_with_mask(self, feat, mask, bdims):
    raw = self.pol(self.feat2tensor(feat), bdims)
    return {
        "action": embodied.jax.outs.MaskedCategorical(
            raw["action"].logits,
            mask,
            unimix=self.config.policy.unimix,
        )
    }
```

当前任务只有一个标量离散动作 `action`。由于不要求兼容，本次实现应直接 assert：

```text
act_space.keys() == {"action"}
action classes == 3
action_mask last dimension == 3
```

这比实现一个未经测试的通用 dict/mixed-action mask 框架更稳妥。

### 6.9 真实 policy 路径

真实 policy 必须直接使用环境真值：

```python
mask = obs["action_mask"] > 0.5
policy = self._policy_with_mask(feat, mask, bdims=1)

if mode == "train":
    act = sample(policy)
elif mode == "eval":
    act = jax.tree.map(lambda x: x.pred(), policy)
else:
    raise ValueError(mode)
```

要求：

- train 使用 masked sample；
- eval 使用 masked argmax；
- 不使用 predicted mask 替代真实 mask；
- policy 输出前计算被选动作是否在真实 mask 中，理论值必须始终为 0；
- mode 不再被忽略。

### 6.10 Imagination 采样路径

当前 imagination policy function：

```python
policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
```

改为：

```python
def imag_policy(feat):
    mask = self._predicted_mask(feat, bdims=1)
    policy = self._policy_with_mask(feat, mask, bdims=1)
    return sample(policy)
```

`RSSM.imagine()` 会在当前 latent 调用 `imag_policy()`，因此动作和 mask 时序正确：

```text
predicted_mask(z_t) -> sample a_t -> dynamics(z_t, a_t) -> z_{t+1}
```

### 6.11 Imagination actor loss 路径

采样时 masked 还不够。`imag_loss()` 中重新计算的 policy distribution 也必须 masked。

完整流程：

```python
imgmask = self._predicted_mask(imgfeat, bdims=2)
imgpolicy = self._policy_with_mask(imgfeat, imgmask, bdims=2)

los, ..., mets = imag_loss(
    imgact,
    ...,
    imgpolicy,
    ...
)
```

对齐要求：

```text
imgfeat.shape time = H + 1
imgact.shape time  = H + 1
imgmask.shape time = H + 1
imag_loss 使用前 H 个 action/policy/mask
```

必须用测试验证：

- imagination 采样时的 mask；
- actor loss 重新计算时的 mask；
- 二者对同一 latent 生成相同 hard mask；
- `imgact_t` 在 `imgmask_t` 中合法。

### 6.12 Availability head 参数同步

真实 policy 使用环境真值 mask，不需要 availability head。

因此 `policy_keys` 不需要加入 `avail`：

```python
return "^(enc|dyn|dec|pol)/"
```

availability head 只在 learner 的 supervised loss 和 imagination 中使用，保留在 train params 即可。

如果以后希望 policy 端在缺失真实 mask 时使用预测 mask，必须另行设计并将 `avail` 加入 policy params；本次不做。

### 6.13 Replay、Driver、RSSM 和 Wrapper 的修改判断

| 文件 | 是否改代码 | 原因 |
|---|---|---|
| `embodied/envs/from_gymnasium.py` | 否 | Box float32 mask 会自动转换 |
| `embodied/core/driver.py` | 否 | 已正确保存 `{obs_t, action_t}` |
| `embodied/core/replay.py` | 否 | 已保存全部非 `log/*` 字段 |
| `dreamerv3/rssm.py` | 原则上否 | 当前 imagination 调用时序正确 |
| `embodied/run/parallel.py` | 否 | 已正确传 train/eval mode |
| `dreamerv3/agent.py` | 是 | encoder/decoder、head、真实 policy、imagination、loss |
| `embodied/jax/outs.py` | 是 | 新增 masked categorical |
| `dreamerv3/configs.yaml` | 是 | 新增 availability 配置和 loss scale |

---

## 7. 全链路时序

### 7.1 真实采样

```text
Environment state s_t
  -> evaluate all target transitions
  -> action_mask_t
  -> obs_t
  -> FromGymnasium
  -> Dreamer encoder/RSSM posterior z_t
  -> actor logits
  -> MaskedCategorical(actor logits, action_mask_t)
  -> train: sample / eval: argmax
  -> action_t
  -> env.step(action_t)
  -> transition decision
  -> execute or strict no-op
  -> state s_{t+1}
```

### 7.2 Replay 与 world model 监督

```text
Replay item at t:
  obs_t.action_mask
  action_t
  reward_t
  is_terminal_t

Encoder(obs_t including action_mask_t)
  -> posterior latent z_t

AvailabilityHead(z_t)
  -> predicted mask logits_t
  -> BCE against action_mask_t
```

### 7.3 Imagination 与 actor loss

```text
start latent z_t
  -> AvailabilityHead(z_t)
  -> threshold + nonempty fallback + stop_gradient
  -> predicted_mask_t
  -> MaskedCategorical(actor_logits_t, predicted_mask_t)
  -> sample imagined action a_t
  -> RSSM dynamics
  -> z_{t+1}

all imagined z
  -> recompute same predicted masks
  -> recompute same masked policies
  -> imag_loss(logp, entropy, advantage)
```

---

## 8. 文件级修改清单

### 8.1 `gym-trading-env`

#### 必改

| 文件 | 修改内容 |
|---|---|
| `src/gym_trading_env/envs/trading_env.py` | observation space、mask、统一 transition evaluation、step 重构、日志 |
| `src/gym_trading_env/envs/config.py` | 对 action-mask 支持范围增加明确配置校验 |
| `src/gym_trading_env/envs/action.py` | 如需要，增加稳定 reject reason/code 或动作记录结构 |
| `src/gym_trading_env/envs/metrics.py` | 增加 action_rejected、mask cardinality、reject code 指标 |
| `src/gym_trading_env/rewards/reward_functions.py` | 明确 invalid/rejected penalty 使用的新语义 |
| `tests/contract/test_observation_space.py` | 新增 action_mask shape/dtype/value 契约 |
| `tests/contract/test_invalid_actions_strict.py` | 改为验证 mask 与严格 no-op 一致 |
| `tests/contract/test_action_mask_config_scope.py` | 验证 Phase 1A 范围约束，以及 Phase 1B near-EOD 配置可运行 |
| `tests/integration/test_smoke_rollout.py` | observation 新字段与 mask-respecting rollout |
| `tests/integration/test_env_replay_determinism.py` | replay 后 mask 必须一致 |

#### 推荐新增

| 文件 | 内容 |
|---|---|
| `src/gym_trading_env/envs/target_transition.py` | decision 和 operation quote 数据结构 |
| `tests/contract/test_action_mask_contract.py` | action mask 状态矩阵与 differential contract |
| `tests/unit/test_target_transition_quote.py` | quote 无副作用与资金计算 |

#### 需要同步但不应复制规则

| 文件/模块 | 修改 |
|---|---|
| `get_oracle_snapshot()` | 暴露统一 decision 结果 |
| `_refresh_agent_state()` | gate 从统一 decision 派生 |

### 8.2 `dreamerv3`

#### 必改

| 文件 | 修改内容 |
|---|---|
| `dreamerv3/agent.py` | mask observation、availability head、真实 policy、imagination、actor loss、metrics |
| `dreamerv3/rssm.py` | 从 loss 路径返回与 `mask_t` 对齐的一步 prior feature |
| `embodied/jax/outs.py` | 新增 `MaskedCategorical` |
| `dreamerv3/configs.yaml` | availability head、threshold、gradient、loss scale |
| 自动 warm-up 配置与状态机 | 同进程动态控制 policy/value/repval 更新门禁 |
| 活跃 train/eval 环境 YAML | Phase 1A 关闭 EOD；Phase 1B 开启 `block_open_near_eod`、保持 `force_flatten_eod=false` |

#### 推荐新增测试

| 文件 | 内容 |
|---|---|
| `embodied/tests/test_masked_categorical.py` | masked distribution 数值和采样契约 |
| `dreamerv3/tests/test_action_mask_agent.py` | policy mode、availability loss、梯度、imagination shape |
| `dreamerv3/tests/test_action_mask_alignment.py` | Driver/Replay 的 mask_t/action_t 对齐 |
| `dreamerv3/tests/test_prior_availability_alignment.py` | 一步 prior feature 与 `mask_t` 的时序和监督对齐 |

#### 已验证无需修改

- `embodied/envs/from_gymnasium.py`
- `embodied/core/driver.py`
- `embodied/core/replay.py`
- `embodied/run/parallel.py`

如果实施测试证明这些文件存在未发现的 shape 或 mode 问题，再单独修改，不能预先扩大改动面。

---

## 9. 分阶段实施计划

### Phase 0：冻结语义与建立基线

#### 工作

1. Review 并确认本文第 2、4、6 节。
2. 固定动作顺序 `[SHORT, FLAT, LONG]`。
3. 固定闭市、开仓次数、资金不足和 near-EOD 语义，并准备 Phase 1A/1B 两套配置。
4. 保存当前 Monte Carlo 和 invalid action 基线。
5. 正式训练使用全新 logdir；同一进程和 replay 自动完成 warm-up 到 formal 的切换。

#### 退出条件

- 所有评审未决策项已明确；
- 不再讨论旧 checkpoint/replay 兼容；
- 测试矩阵得到确认。

### Phase 1A：非 EOD 环境单一合法性来源

#### 工作

1. 新增 transition decision/quote。
2. 将开仓、平仓、反手预检整理为无副作用 quote。
3. 生成并缓存唯一 transition decision table，mask 和 step 共同消费。
4. observation 增加 `action_mask`。
5. `_refresh_agent_state()` 和 oracle 从统一 evaluation 派生。
6. `step()` 使用同一 decision。
7. 修复反手完整回滚风险。

#### 退出条件

- 环境全部现有测试通过；
- action mask 状态矩阵测试通过；
- 重复调用 mask 无副作用；
- 对任意状态，mask 中每个允许动作立即执行不因已知业务规则失败；
- mask 永不为空。

### Phase 1B：纳入 Near-EOD 与 Market-Closed

#### 工作

1. 将 `market_closed` 纳入统一 transition decision：
   - 只允许当前目标/HOLD；
   - 非当前目标标记 unavailable reason `market_closed`。
2. 将 `block_open_near_eod` 纳入统一 transition decision：
   - 允许当前目标/HOLD；
   - 允许持仓平到 FLAT；
   - 禁止新开仓与反手；
   - 标记 unavailable reason `near_eod`。
3. 保持 `force_flatten_eod=false`，或先重构为与 agent 请求分离的外生风险动作。
4. 增加按原因统计的 unavailable/rejected 指标。

#### 退出条件

- near-EOD 和 market-closed 状态矩阵测试通过；
- `mask_execution_mismatch=0`；
- `action_rejected_reason/{market_closed,near_eod}=0` 对 mask-respecting policy 成立；
- 最终报告可分别统计 `max_entries`、`no_money`、`market_closed`、`near_eod` 和 `execution_failed`；
- Phase 1B 通过后，正式训练不得继续使用 Phase 1A 的“关闭 near-EOD”范围结论。

### Phase 2：Masked categorical

#### 工作

1. 实现 `MaskedCategorical`。
2. 支持 masked sample、pred、logp、entropy、valid-only unimix。
3. 防止 all-false 和数值 NaN。

#### 退出条件

- 采样 100,000 次不产生被屏蔽动作；
- 单合法动作时 sample/pred 固定，entropy 为 0；
- invalid action logp 极低且 finite；
- valid-only unimix 不向非法动作分配概率；
- bfloat16 actor logits 经内部 float32 处理后无 NaN。

### Phase 3：真实 Dreamer policy

#### 工作

1. `action_mask` 进入 encoder，排除通用 decoder。
2. 真实 policy 使用 true mask。
3. train 使用 masked sample。
4. eval 使用 masked argmax。
5. 增加 chosen-action violation 指标。

#### 退出条件

- 真实 policy 永不选择 mask=0 的动作；
- eval 相同输入和参数返回确定性动作；
- train 保留合法动作内随机探索；
- Driver 和 Replay 中 `action_mask_t` 与 `action_t` 对齐。

### Phase 4：Availability head 与 imagination

#### 工作

1. 增加 availability head 和 BCE loss。
2. 增加 posterior `avail_post` 监督损失。
3. 修改 RSSM loss 输出一步 prior feature，增加 `avail_prior` 监督损失。
4. 增加 availability warm-up 与 actor imagination 启用门禁。
5. 增加 hard predicted mask、nonempty fallback 和 stop-gradient。
6. imagination 采样使用 predicted mask。
7. imag actor loss 使用同一 predicted mask。
8. 增加 posterior、prior、imagined mask 预测质量和 gradient 指标。

#### 退出条件

- `avail_post` 和 `avail_prior` loss shape 均为 `(B, T)`；
- actor loss 对 availability head 参数梯度为 0；
- posterior/prior availability BCE 对 availability head 参数梯度非 0；
- prior availability loss 能训练 RSSM dynamics 的可达性表示；
- prior feature 与 `mask_t` 时序专项测试通过；
- posterior 与 prior 分别达到预测门槛后，才能启用正式 imagination actor loss；
- warm-up 阶段 policy/value imagination 参数不更新；
- actor loss 不能通过 predicted hard mask 更新 availability head；
- 每个 imagined action 在对应 predicted mask 中合法；
- imagination 和 actor loss mask 时序严格对齐；
- 无 NaN/Inf。

### Phase 5：短训练与诊断

#### 工作

1. 使用小模型、短 replay、短 imagination 进行 smoke train。
2. 检查 availability 学习曲线。
3. 检查真实 invalid/rejected action。
4. 检查 imagined mask cardinality 和 fallback。
5. 检查 eval deterministic 行为。
6. 检查 valid cardinality 下的 chosen action 分布和策略活性。

#### 退出条件

- 训练可持续运行，无 shape、JIT、checkpoint 或 replay 错误；
- 真实 chosen-action violation 为 0；
- 环境 action_rejected 接近 0；
- availability false-positive rate 达到验收阈值；
- predicted empty-mask fallback 接近 0。
- cardinality=2/3 时仍存在合理的非 FLAT 合法动作选择；
- `trades_opened`、`nonflat_action_rate`、`exposure_ratio` 未塌缩到接近 0。

### Phase 6：正式训练与回归

#### 工作

1. 使用新正式训练 logdir 启动自动 warm-up；通过门禁后同进程自动进入 formal。
2. 运行 Monte Carlo。
3. 对比：
   - invalid/rejected action；
   - 交易次数；
   - mask cardinality；
   - chosen action distribution by cardinality；
   - max_entries/no_money/market_closed/near_eod/execution_failed；
   - trades_opened/exposure_ratio/nonflat_action_rate；
   - availability 预测指标；
   - policy entropy；
   - reward 和收益风险指标。

#### 退出条件

- action mask 技术验收全部通过；
- 没有动作语义错位；
- 没有因 mask 引入的训练数值异常；
- 策略表现结果进入独立策略评审。

---

## 10. 测试设计

### 10.1 环境状态矩阵测试

至少覆盖以下状态：

| 状态 | 当前仓位 | 预期重点 |
|---|---|---|
| 正常开市、资金足够 | FLAT/SHORT/LONG | 三动作可用 |
| 开仓次数耗尽 | FLAT/SHORT/LONG | 只允许 HOLD/平仓 |
| 闭市 | FLAT/SHORT/LONG | 只允许当前目标/HOLD |
| near-EOD 禁止开仓 | FLAT/SHORT/LONG | 允许 HOLD/平仓，禁止开仓和反手 |
| 多开资金不足 | FLAT | LONG 不可用 |
| 空开资金不足 | FLAT | SHORT 不可用 |
| 反手资金不足 | SHORT/LONG | 反向目标不可用，FLAT 可用 |
| 最大多仓限制 | FLAT/SHORT | LONG 或空翻多不可用 |
| 最大空仓限制 | FLAT/LONG | SHORT 或多翻空不可用 |
| 平仓 quote 异常 | SHORT/LONG | FLAT 不可用并可诊断 |

Phase 1A 暂时跳过 near-EOD/闭市用例并对误启用配置 fail fast；Phase 1B 必须启用并通过上述完整状态矩阵。`force_flatten_eod` 仍不进入该矩阵。

### 10.2 环境 differential contract

对可构造状态执行：

1. 生成 `mask_before`。
2. 保存完整状态摘要。
3. 对每个 target 在独立同状态环境中执行。
4. 若 mask=1：
   - 请求不得因已知业务约束被拒绝；
   - requested/planned/executed 对齐。
5. 若 mask=0：
   - 请求必须严格 no-op；
   - ledger、仓位、现金、保证金、PnL、开仓次数不变。

这项测试用于防止 mask 与执行逻辑漂移。

另加 stale-table 测试：

1. 生成 observation 和 decision table。
2. 人工修改一个影响合法性的状态字段。
3. 调用 `step()`。
4. 环境必须因 `state_version` 不一致 fail fast，不能重算后继续执行。
5. 开发期 debug 测试故意漏增 version，确认 `debug_fingerprint` 能发现版本维护缺陷。

### 10.3 无副作用测试

连续调用 100 次：

```python
env._build_target_transition_table()
env._get_obs()
env.get_oracle_snapshot()
```

验证：

- ledger entries 不变；
- position 不变；
- cash/margin/PnL 不变；
- `_entries_used_today` 不变；
- action/result/metrics 不变；
- mask 每次一致。

### 10.4 Masked categorical 单元测试

必须覆盖：

- mask `[1, 0, 1]` 下采样永不返回 1；
- argmax 原本位于非法动作时，pred 返回最佳合法动作；
- 单合法动作 entropy 为 0；
- all-valid 与普通 categorical 行为一致；
- invalid logp finite 且极低；
- valid-only unimix；
- batch mask；
- time-batch mask；
- all-false 输入被显式拒绝或经过指定 fallback；
- 梯度 finite。

### 10.5 Dreamer 对齐测试

构造带有唯一标识的 toy sequence：

```text
mask_t = one_hot(t % 3)
action_t 必须等于唯一合法动作
```

通过 Driver 和 Replay 后验证：

- 对 `is_last == False` 的真实可执行 transition，replay `action_mask[:, t]` 与 `action[:, t]` 对齐；
- 对 `is_last == True` 的 terminal dummy action，不做 mask 合法性断言；
- `_apply_replay_context()` 后对齐不改变；
- prevact shift 不会错误 shift mask；
- `is_first` reset 不会使用前 episode mask。

### 10.6 Dreamer 梯度测试

必须显式验证：

1. `grad(policy_loss, availability_params) == 0`。
2. `grad(avail_post_loss, availability_params) != 0`。
3. `grad(avail_prior_loss, availability_params) != 0`。
4. 默认 gradient 配置下：
   - posterior availability loss 对 encoder/RSSM 存在有效梯度；
   - prior availability loss 对 RSSM dynamics 存在有效梯度；
   - actor loss 不能通过 hard mask 对 availability head 或其 mask 预测路径产生梯度。
5. 对应消融配置关闭时：
   - `grad(availability_loss, encoder_params) == 0`；
   - `grad(availability_loss, dyn_params) == 0`。
6. actor params 对 masked policy loss 有有效梯度。
7. 非法动作 logits 不通过 entropy 或 logp 获得错误梯度。

### 10.7 一步 Prior Availability 对齐测试

构造 action/mask 随时间变化的 toy sequence，验证：

- `priorfeat[:, t]` 只依赖 `action_{t-1}` 和此前 latent，不读取 `obs_t`；
- `priorfeat[:, t]` 的监督目标严格为 `action_mask[:, t]`；
- reset 后 prior 不携带前 episode 状态；
- 人为制造 off-by-one 时测试必然失败；
- posterior 指标与 prior 指标分别输出，不能合并掩盖差异。

### 10.8 Imagination 时序测试

使用可控 availability head，使每个 latent step 只允许一个不同动作：

```text
z_0 -> only SHORT
z_1 -> only FLAT
z_2 -> only LONG
```

验证：

- `imgact[:, 0]` 使用 `mask(z_0)`；
- `imgact[:, 1]` 使用 `mask(z_1)`；
- actor loss 的 policy mask 与采样 mask 相同；
- 不出现 off-by-one；
- last action 和 H+1 state 对齐。

### 10.9 端到端 smoke

建议分三种：

1. Mask-respecting random policy：
   - 验证环境 mask 本身；
   - 预期 action_rejected=0。
2. 未训练 Dreamer masked policy：
   - 验证真实采样路径；
   - 预期 chosen-action violation=0。
3. 小模型短训练：
   - 验证 availability/imagination/actor loss；
   - 预期无 NaN、无 shape 错误。

---

## 11. 指标与诊断

### 11.1 环境指标

```text
env/action_rejected
env/action_rejected_total
env/action_rejected_ratio
env/reject_code
env/action_mask_valid_count
env/action_mask_short
env/action_mask_flat
env/action_mask_long
env/unavailable_reason/max_entries
env/unavailable_reason/no_money
env/unavailable_reason/market_closed
env/unavailable_reason/near_eod
env/execution_failed
env/requested_target
env/planned_action
env/executed_action
```

每个原因必须同时报告两类计数：

```text
unavailable_reason/<reason>  # 有多少候选目标因该原因被 mask
action_rejected_reason/<reason>  # policy 实际请求后被拒绝的次数
```

否则 true mask 生效后 rejected reason 会自然接近 0，无法判断 `max_entries`、`no_money`、`market_closed` 和 `near_eod` 在状态空间中的真实出现频率。

### 11.2 真实 policy 指标

```text
policy/true_mask_valid_count
policy/chosen_mask_violation
policy/masked_entropy
policy/mode_train_count
policy/mode_eval_count
policy/valid_cardinality_1_rate
policy/valid_cardinality_2_rate
policy/valid_cardinality_3_rate
policy/action_short_given_cardinality_{1,2,3}
policy/action_flat_given_cardinality_{1,2,3}
policy/action_long_given_cardinality_{1,2,3}
policy/nonflat_action_rate
```

`policy/chosen_mask_violation` 必须恒为 0。

`valid_action_cardinality_distribution` 和 `chosen_action_distribution_by_cardinality` 用于识别“无效动作归零，但 policy 永远选择 FLAT”的伪修复。尤其需要检查 cardinality 为 2 或 3 时，policy 是否仍能选择非 FLAT 合法动作。

### 11.3 Availability 监督指标

```text
avail_post/loss
avail_post/bit_accuracy
avail_post/exact_match
avail_post/false_positive_rate
avail_post/false_negative_rate
avail_prior/loss
avail_prior/bit_accuracy
avail_prior/exact_match
avail_prior/false_positive_rate
avail_prior/false_negative_rate
avail_prior/predicted_valid_count
avail/true_valid_count
```

False positive 比 false negative 风险更高：

- false positive：imagination 允许环境实际不允许的动作；
- false negative：imagination 隐藏了一个本来可用的动作。

因此重点监控 `false_positive_rate`。

### 11.4 Imagination 指标

```text
imag/mask_valid_count
imag/mask_empty_fallback_rate
imag/chosen_mask_violation
imag/masked_entropy
imag/mask_false_positive_proxy
imag/valid_cardinality_1_rate
imag/valid_cardinality_2_rate
imag/valid_cardinality_3_rate
```

`imag/chosen_mask_violation` 必须恒为 0。

深层 imagined latent 没有真实环境 mask，无法计算真正 FPR。`imag/mask_false_positive_proxy` 使用 replay-root/一步 prior 上可与 true mask 对齐的预测错误作为代理；更深 horizon 主要监控 cardinality 分布漂移和 fallback，不能把 proxy 当成真实 FPR。

### 11.5 策略活性与结果指标

无效动作下降不能以“永远 FLAT、不再交易”为代价。最终报告必须同时对比：

```text
trades_opened
target_change_count
nonflat_action_rate
exposure_ratio
return_pct
max_drawdown
profit_factor
expectancy
valid_action_cardinality_distribution
chosen_action_distribution_by_cardinality
```

这些指标必须与未加 mask 的基线、mask-respecting random policy 和相同训练预算下的新 policy 对比。若无效动作归零但 `trades_opened`、`exposure_ratio` 或 cardinality=2/3 时的 non-FLAT 选择显著塌缩，不能判定修复成功。

---

## 12. 验收标准

### 12.1 环境验收

- 所有 action mask contract 测试通过；
- mask 永不为空；
- mask 与执行已知业务约束一致率 100%；
- `mask_execution_mismatch = 0`；
- mask 生成无副作用；
- 被屏蔽动作强制调用时严格 no-op；
- requested/planned/executed 可诊断；
- 反手失败回滚完整。

### 12.2 Dreamer 数值与行为验收

- 真实 policy chosen-mask violation = 0；
- imagined policy chosen-mask violation = 0；
- masked categorical 无 NaN/Inf；
- eval 确定性；
- train 仅在合法动作内探索；
- actor loss 不更新 availability head；
- posterior/prior availability loss 可更新 availability head；
- prior availability loss 可更新 RSSM dynamics；
- posterior 与 prior 指标分别达标后才启用正式 imagination actor loss；
- replay mask/action 时序无 off-by-one。

### 12.3 Availability 预测初始阈值

短训练稳定后，posterior 与一步 prior 必须分别达到：

```text
avail_post bit accuracy          >= 99.5%
avail_post exact mask match      >= 99.0%
avail_post false positive rate   <= 0.1%
avail_prior bit accuracy         >= 99.0%
avail_prior exact mask match     >= 98.0%
avail_prior false positive rate  <= 0.5%
imag empty fallback rate         <= 0.01%
```

这些是初始验收阈值，不是永久超参数。若未达到，不应直接提高 threshold 掩盖问题，应先检查 observation 是否包含足够状态、时序是否对齐、规则是否确定。

### 12.4 端到端验收

- mask-respecting random rollout 的 `action_rejected_ratio = 0`；
- Dreamer 真实 rollout 的 `action_rejected_ratio = 0`；
- 环境内部执行异常单独计入 `execution_failed`，不能混入 actor 的 `action_rejected`；
- 新训练从空 replay 和新 checkpoint 启动；
- Monte Carlo 不再出现当前 20% 至 89% 的系统性无效动作比例；
- 最终报告分别给出 `max_entries`、`no_money`、`market_closed`、`near_eod` 和 `execution_failed`；
- `valid_action_cardinality_distribution` 与 `chosen_action_distribution_by_cardinality` 无异常塌缩；
- `trades_opened`、`nonflat_action_rate` 和 `exposure_ratio` 不得在无解释情况下接近 0；
- 同时报告 `return_pct`、`max_drawdown`、`profit_factor` 和 `expectancy`；
- 交易收益和风险表现单独评审，不与 action mask 技术验收混合。

---

## 13. 风险与控制

### 13.1 最大风险：imagined mask 预测错误

真实交互使用 true mask，因此不会因预测错误产生真实非法动作。

但 imagination 中 false positive 会让 actor 学习现实中不可执行的动作。控制方法：

- availability 独立监督；
- mask 进入 encoder，使 posterior 明确包含可达性；
- dedicated BCE，不使用通用 reconstruction；
- 重点监控 false positive；
- predicted mask deterministic threshold；
- all-false fallback 单独计数；
- actor 不得修改 mask head。

### 13.2 环境预检和执行漂移

这是环境侧最大风险。控制方法：

- 单一 transition evaluation；
- quote/commit 分离；
- differential contract；
- agent_state 和 oracle 不再复制规则；
- 每次新增业务约束必须先进入统一 quote。

### 13.3 Mask 隐藏了环境 bug

如果 mask 错误地屏蔽动作，真实 policy 不会再触发对应执行路径，bug 可能更难暴露。

控制方法：

- 对 mask=0 的动作仍做独立严格 no-op 测试；
- differential contract 覆盖全部动作；
- 环境 final validation 不删除；
- reject reason 和 mask 状态保留诊断。

### 13.4 训练探索下降

某些状态只剩一个合法动作时 entropy 必然为 0。这是正确约束，不是探索退化。

在多个合法动作状态中，unimix 和 entropy 仍只应作用于合法动作。

### 13.5 Availability head 依赖状态不充分

如果 latent 无法预测精确 mask，可能是 observation 缺少决定合法性的状态，例如精确 free margin、开仓次数或 session 状态。

本方案让 action mask 进入 encoder，并允许 availability BCE 训练 latent representation，同时保留现有 agent state。若预测仍差，应增加明确、可预测的状态特征，而不是让 head 猜测隐藏状态。

必须区分 posterior、一步 prior 与 imagined latent 的预测质量。Posterior 已看到当前 observation，不能单独作为 imagination 可用性的证据。

控制方法：

1. `avail_post` 和 `avail_prior` 都是正式监督损失；
2. prior feature 与 `mask_t` 的时序必须通过专项测试；
3. 分别检查 posterior/prior FPR、FNR、exact match；
4. 检查 prior/posterior KL、imagined mask cardinality 和 fallback；
5. prior 指标未达到门槛前，不启用正式 imagination actor loss。

### 13.6 强制清算或外部执行被误纳入

`force_flatten_eod`、session cleanup 或外部执行链路如果被误纳入 actor action-mask 验收，会产生错误的可靠性结论。

控制方法：

- Phase 1B 正式训练要求 `force_flatten_eod=false`，或已经重构为独立外生风险动作；
- 所有可靠性指标只统计环境内部、非 terminal transition；
- 文档、实验名和报告明确标记“不包含强制清算/RPC”；
- 不把外部执行结果混入 mask 预测或 actor 合法性指标。

### 13.7 旧训练数据污染

旧 replay 没有 action mask，旧 checkpoint 没有 availability head，旧 actor 也未按 mask 训练。

由于当前无兼容要求，实施后必须：

- 使用新 logdir；
- 清空 replay；
- 不加载旧 checkpoint；
- 不实现默认 mask 或参数迁移。

---

## 14. Review 必须确认的决策

以下是实施前必须确认的方案决策，本文给出推荐值：

| 决策 | 推荐 |
|---|---|
| Mask 顺序 | 固定 `[SHORT, FLAT, LONG]` |
| Action mask Gym space | `Box(float32, shape=(3,), 0..1)` |
| 闭市时合法动作 | 仅当前目标/HOLD |
| Near-EOD 禁开仓 | Phase 1B 纳入 true mask |
| `force_flatten_eod` | 保持关闭，或重构为独立外生风险动作 |
| 外部 RPC/broker | 本次不处理，不进入可靠性验收 |
| Terminal cleanup/风控 | 外生转移，不进入 mask/action 合法性统计 |
| `ERROR_MARKET_CLOSED` 是否算 rejected | 是 |
| `invalid_action` 是否与 rejected 统一 | 开发阶段统一 |
| Mask 是否进入 encoder | 是 |
| Mask 是否进入通用 decoder | 否 |
| Availability 监督 | 独立 `avail_post` + `avail_prior` binary BCE |
| Availability gradient | post/prior 默认开启；关闭仅用于消融 |
| Actor 是否可反传到 predicted mask | 否 |
| Predicted mask threshold | 初始 `0.5` |
| All-false fallback | 强制最高预测概率动作可用，并记录 rate |
| 真实 policy 使用 true 还是 predicted mask | true mask |
| Eval 动作选择 | masked argmax |
| Train 动作选择 | masked sample |
| Unimix | 仅合法动作内混合 |
| 旧 replay/checkpoint | 丢弃，不迁移 |
| 单仓位假设 | 初始化 fail fast |

---

## 15. 建议实施提交顺序

为降低 review 和回归难度，建议按以下独立提交实施：

1. `env: centralize target transition evaluation and quotes`
2. `env: expose non-EOD action mask and add contract tests`
3. `env: add near-EOD market-closed decisions and reason metrics`
4. `env: separate requested planned executed action metrics`
5. `dreamer: add masked categorical distribution`
6. `dreamer: apply true mask in real policy and honor policy mode`
7. `dreamer: add posterior and one-step prior availability losses`
8. `dreamer: apply predicted mask in imagination and actor loss`
9. `tests: add mask alignment gradient activity and end-to-end coverage`
10. `training: start fresh masked-action experiment`

每个提交必须独立通过对应阶段测试。不要把环境规则重构、Dreamer imagination 和训练配置一次性混在一个不可审查的大提交中。

---

## 16. 最终实施判断

该问题可以修复，且推荐方案具备完整技术闭环。

最难的部分不是把 mask 放进 observation，也不是在真实 actor logits 上加负数，而是保证以下四件事同时成立：

1. 环境 mask 与真实执行规则永不漂移；
2. Replay 中 `mask_t` 与 `action_t` 时序正确；
3. Imagination 在选择 `action_t` 前使用 `latent_t` 的 predicted mask；
4. Actor loss 使用与 imagination 采样完全相同的 masked distribution，且不能反向操纵 mask head。

纳入 near-EOD 和 market-closed、排除强制清算与外部 RPC/broker 后，当前代码结构允许完成这些改动。对范围内完整落地的把握约为 90%。

范围内真实环境路径应按 100% 结构性不变量验收：true mask 下不选择非法动作，且 `mask_execution_mismatch=0`。任何违反都视为 P0 缺陷，不能用统计误差解释。

完整 Dreamer 训练链路不能宣称 100% 可靠。剩余不可消除的不确定性集中在 availability head 对 imagined latent 的 mask 预测质量，以及修复动作语义后策略是否收敛和盈利。

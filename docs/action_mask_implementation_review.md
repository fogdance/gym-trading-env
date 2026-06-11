# Action Mask 全链路实现 Review

日期：2026-06-11

审查范围：

- 设计方案：`docs/action_mask_full_chain_implementation_plan.md`
- 环境仓库：`gym-trading-env`，分支 `dev/action-mask-full-chain`
- DreamerV3 仓库：`~/Documents/work/dreamerv3`，分支 `dev/action-mask-full-chain`
- 动作空间保持 `[SHORT, FLAT, LONG]`
- 外部 RPC/broker 不进入本次可靠性结论

## 1. Review 结论

**结论：暂不通过正式 masked actor 训练准入。**

当前实现已经接通主要功能路径：

- 环境输出 true `action_mask`；
- 真实 Dreamer policy 使用 true mask；
- availability head 同时训练 posterior 和一步 prior；
- imagination 采样与 actor loss 使用 predicted hard mask；
- predicted mask 对 actor stop-gradient；
- active train/eval YAML 已关闭 `force_flatten_eod`。

但是环境侧仍存在两个已复现的结构性 P0 缺陷：

1. observation 对应的 transition table 不会被 `step()` 原样消费，状态变化后会静默重算；
2. 反手执行中途失败时回滚不完整，环境会发生部分状态提交。

此外，DreamerV3 侧尚未通过方案要求的 replay/prior/imagination 时序专项测试，warm-up 也没有完整冻结 value 更新。因此当前只能用于继续开发、单元测试和短 smoke，不能宣称完成全链路修复。

完成度判断：

| 范围 | 当前判断 |
|---|---|
| 环境 true-mask 功能路径 | 约 75% |
| Dreamer mask 结构接入 | 约 85% |
| 自动化验证与诊断指标 | 约 45% |
| 正式训练与端到端验收 | 未开始 |

## 2. Findings

### P0-1：反手失败回滚不完整，会产生部分提交

方案要求 commit 失败后 entries、仓位、现金、保证金、PnL、开仓次数均不变化，并明确指出反手必须补齐 position、realized PnL、closed trades 和 journal entries 的回滚。

当前实现只保存并恢复 ledger 余额：

- `src/gym_trading_env/envs/trading_env.py:2282-2297`
- `src/gym_trading_env/envs/trading_env.py:2411-2424`
- `src/gym_trading_env/envs/accounting/ledger.py:80-86`

`Ledger.restore()` 不截断 `ledger.entries`。反手路径在添加新仓之前已经：

- 移除原仓位；
- 增加 `closed_trade_profits`；
- 增加 `realized_pnl`；
- 追加 journal entry。

复现结果：持有多仓后，强制 `add_short_position()` 抛错。

```text
before: long=1, short=0, realized_pnl=0.0, closed_trades=0, ledger_entries=1
after:  long=0, short=0, realized_pnl=-59.9940, closed_trades=1, ledger_entries=2
execution_failed=True
```

此时实现把 `executed_action` 改为 `HOLD`，但真实状态已经发生变化，诊断字段也不再可信。

**必须修复：**

- 建立完整 transaction snapshot/restore，至少覆盖 ledger balances、ledger entry length、position slots、closed trade profits、realized PnL、trade records、entries counter；
- 或将 quote/commit 重构为 commit 阶段不会在部分提交后失败；
- 增加 open、close、flip 每个 commit 点的故障注入测试。

### P0-2：`obs_t.action_mask` 与 `step(action_t)` 不消费同一 decision

方案要求：

- observation 缓存 transition table；
- `step()` 消费 observation 对应 table；
- 状态版本不一致时 fail fast；
- 禁止静默重算。

当前实现使用 `state_key` 检测变化，但检测到变化后直接重建：

- `src/gym_trading_env/envs/trading_env.py:719-786`
- `src/gym_trading_env/envs/trading_env.py:819`
- `src/gym_trading_env/envs/trading_env.py:2634-2639`

`state_version` 仅等于 `current_step`，没有 `_legality_state_version`，也没有 consume/fail-fast 机制。

已复现：

```text
observed_mask = [1.0, 1.0, 1.0]
生成 observation 后修改 entries_used_today
请求原 mask 中合法的 LONG
step 静默重算并返回 action_rejected=1，terminated=False
```

这证明环境尚未满足方案要求的 100% 结构性不变量。即使标准同步 Gym 调用通常不会在 observation 与 step 之间改状态，当前实现仍会掩盖未来代码中的状态版本维护缺陷。

**必须修复：**

- 引入单调递增 `_legality_state_version`；
- `_get_obs()` 生成并缓存当前 table；
- `step()` consume 缓存 table，不调用自动重建路径；
- version/fingerprint 不一致直接抛错；
- 增加 stale-table 专项测试。

### P1-1：decision 没有 execution-ready quote，预检与执行仍是两套计算

`TargetTransitionDecision` 当前只有判定结果，没有 `execution_quote`：

- `src/gym_trading_env/envs/target_transition.py:6-21`

table 构建时 `_preflight_*()` 只返回 `ForexCode`，执行时 `_long_open()`、`_long_close()`、`_flip_*()` 再次计算 price、margin、fee、close quote 和 position：

- `src/gym_trading_env/envs/trading_env.py:602-717`
- `src/gym_trading_env/envs/trading_env.py:1872-2460`

因此 mask 和执行虽然复用了部分 helper，但尚未达到方案定义的“单一 decision + quote/commit”。这也是 `execution_failed` 和回滚问题仍然存在的根因。

**建议修复：**

- 为 open、close、flip 定义不可变 quote；
- allowed decision 必须携带完整 quote；
- `_execute_transition(decision)` 只消费 quote，不重新计算业务规则。

### P1-2：`execution_failed` 仍被归入 `invalid_action`，可能惩罚 actor

方案定义：

```text
invalid_action = action_rejected
execution_failed 不应惩罚 actor
```

当前执行失败后虽然设置了独立 `execution_failed`，但仍使用：

- `invalid_action = self.action_result != SUCCESS`
- reward 根据 `action_result` 判断 invalid

代码位置：

- `src/gym_trading_env/envs/trading_env.py:900-926`
- `src/gym_trading_env/rewards/reward_functions.py:469-486`
- `src/gym_trading_env/rewards/reward_functions.py:599-625`

Dreamer 的 `data/trading_stage1.yaml` 到 `trading_stage4.yaml` 配置了非零 `invalid_action_punish`，因此系统提交失败可能直接污染 actor reward。

**必须修复：**

- `invalid_action` 只等于 `_last_action_rejected`；
- reward 使用 `action_rejected/reject_reason`，不直接使用所有非 SUCCESS `action_result`；
- `execution_failed` 独立计数并触发环境告警。

### P1-3：availability warm-up 没有完整冻结 value，现有命令也不会进入 actor 阶段

当前 `avail_actor_enabled=False` 只将 imagination 的 `policy` 和 `value` loss 乘零：

- `~/Documents/work/dreamerv3/dreamerv3/agent.py:251-254`

但 `repval_loss` 默认仍启用，并使用 imagined return bootstrap 更新同一个 value 网络：

- `~/Documents/work/dreamerv3/dreamerv3/agent.py:260-276`
- `~/Documents/work/dreamerv3/dreamerv3/configs.yaml:89`
- `~/Documents/work/dreamerv3/dreamerv3/configs.yaml:123`

因此 warm-up 并未完全满足“暂停 policy/value imagination 参数更新”的要求，value 仍会受低质量 predicted-mask imagination 影响。

同时默认值是 `avail_actor_enabled=False`，现有 `train.sh` 和 stage command 示例没有追加 `masked_actor_training`：

- `~/Documents/work/dreamerv3/dreamerv3/configs.yaml:122`
- `~/Documents/work/dreamerv3/dreamerv3/configs.yaml:397-401`
- `~/Documents/work/dreamerv3/train.sh:3-6`

按现有命令运行会永久停留在 warm-up，actor 不会恢复训练。

**必须修复：**

- 明确 warm-up 是否应关闭 `repval`；若关闭，应随同 policy/value gate 一起控制；
- 提供可审计的 warm-up -> masked actor 启动命令或脚本；
- 切换前检查多个窗口的 prior accuracy/exact/FPR 和 fallback；
- 切换后出现退化时停止训练。

### P1-4：Dreamer 最关键的时序与梯度契约尚未自动化

当前 Dreamer 新增测试只有 `dreamerv3/tests/test_action_mask.py`，覆盖 masked categorical 的基础采样、unimix 和 finite gradient。

尚缺方案指定的：

- Driver/Replay 中 `action_mask_t` 与 `action_t` 对齐；
- `priorfeat[:, t]` 与 `action_mask[:, t]` 对齐；
- reset 后 prior 不跨 episode；
- imagination 的 `mask(z_t)`、`action_t` 和 actor loss policy 时序一致；
- actor loss 到 availability head 的梯度恒为 0；
- availability prior loss 到 RSSM dynamics 的梯度非 0；
- 每个 imagined action 在对应 predicted mask 中合法。

代码结构看起来正确：

- `~/Documents/work/dreamerv3/dreamerv3/rssm.py:120-137`
- `~/Documents/work/dreamerv3/dreamerv3/agent.py:195-209`
- `~/Documents/work/dreamerv3/dreamerv3/agent.py:220-258`

但这些是最容易出现 off-by-one 且最难靠 smoke 发现的部分。在专项测试补齐之前，不能把结构阅读等同于验收通过。

### P1-5：环境现有 contract suite 尚未迁移到 target-position 动作语义

方案 Phase 1A 退出条件要求环境现有测试全部通过。当前以下旧 contract 测试仍把底层 `Action` 当作 `env.valid_actions`：

- `tests/contract/test_invalid_actions_strict.py`
- `tests/contract/test_actions_observation_strict.py`
- `src/gym_trading_env/utils/trade_util.py:16-21`

实测上述 contract 共 7 个测试失败，原因是 `env.valid_actions` 已是 `TargetPos`，而测试仍请求 `Action.LONG_OPEN0` 等底层动作。

这些测试不是当前实现的新业务缺陷，但会让 CI 无法区分真实回归和过期测试，也意味着 strict no-op、动作观察一致性等旧契约没有完成迁移。

**必须修复：**

- 使用 target-position 请求重写旧 contract；
- 保留底层执行函数的 unit tests；
- 明确 target request 和 planned execution action 两层测试边界。

### P2-1：诊断指标未达到方案要求

环境当前只输出每步标量：

- `valid_action_cardinality`
- `chosen_action_index`
- `action_rejected`
- `execution_failed`
- `reject_code`

位置：`src/gym_trading_env/envs/trading_env.py:1675-1690`

尚缺：

- `mask_execution_mismatch`
- unavailable reason 与 action rejected reason 的分类计数
- `action_rejected_total/ratio`
- cardinality 分布与按 cardinality 的 chosen action 分布
- `nonflat_action_rate`、`target_change_count`
- oracle 中的 `agent_input.action_mask` 和 `transition_decisions`

当前 oracle 只从 table 派生了 `can_long_open/can_short_open/can_long_close/can_short_close`，没有暴露完整 decision，无法直接用于 mask/执行 differential debug：

- `src/gym_trading_env/envs/trading_env.py:1540-1547`
- `src/gym_trading_env/envs/trading_env.py:1599-1619`

Dreamer 当前输出 post/prior accuracy、exact accuracy、FPR、FNR、cardinality，以及 imagined cardinality/fallback：

- `~/Documents/work/dreamerv3/dreamerv3/agent.py:294-307`
- `~/Documents/work/dreamerv3/dreamerv3/agent.py:256-258`

尚缺：

- real `chosen_mask_violation`
- imagined `chosen_mask_violation`
- masked entropy
- imagined cardinality 分布和 drift/proxy
- policy chosen distribution by cardinality

这些指标不是装饰项；缺少它们就无法发现“无效动作归零，但策略永远 FLAT”的伪修复。

### P2-2：`force_flatten_eod` 的范围约束未在环境层 fail fast

active Dreamer train/eval YAML 已正确设置：

```text
block_open_near_eod=true
force_flatten_eod=false
```

但是环境默认仍是 `force_flatten_eod=True`，并在 agent action 后静默执行：

- `src/gym_trading_env/envs/config.py:57-66`
- `src/gym_trading_env/envs/trading_env.py:906-916`

当前没有 action-mask 模式的配置校验，也没有把 force flatten 作为独立外生动作记录。误用其他 YAML 时仍可能破坏 requested/planned/executed 诊断。

**建议修复：**

- 正式训练配置初始化时要求 `force_flatten_eod=False`；
- 或将其重构为独立外生 risk event，并从 agent action 指标中排除。

### P2-3：真实 mask 的 all-false 错误会被 Dreamer 静默 fallback

`MaskedCategorical` 对所有 all-false mask 都回退到 actor 最大 logit：

- `~/Documents/work/dreamerv3/embodied/jax/outs.py:243-258`

这对 predicted mask 是方案指定的数值安全措施，但真实环境 mask 的契约是“永不为空”。真实 policy 也使用同一个 fallback，因此 wrapper/replay/环境错误可能被隐藏。

**建议修复：**

- `MaskedCategorical` 增加 strict/fallback 模式；
- true mask 使用 strict 模式并 fail fast；
- predicted mask 使用 argmax-probability fallback 并记录 rate。

### P2-4：MaskedCategorical 测试覆盖不足

当前直接测试通过，但仅采样 1,000 次，且未覆盖：

- 单合法动作 entropy=0；
- invalid logp finite 且极低；
- all-valid 与普通 categorical 等价；
- batch/time-batch mask；
- all-false 行为；
- bfloat16 logits；
- 100,000 次采样验收。

这不会否定当前实现，但尚未满足方案 Phase 2 的退出条件。

## 3. 与方案一致的实现

以下部分与设计方案一致，可保留：

| 设计项 | 实现状态 |
|---|---|
| 动作顺序 `[SHORT, FLAT, LONG]` | 已实现 |
| `Box(float32, shape=(3,))` action mask | 已实现 |
| 单仓位假设初始化 fail fast | 已实现 |
| near-EOD/market-closed true mask | 已实现 |
| agent-state gate 从 transition table 派生 | 已实现 |
| requested/planned/executed/reject 分离 | 已实现基础字段 |
| 被 mask 请求严格 no-op | 基础路径已通过测试 |
| true policy：train sample / eval argmax | 已实现 |
| mask 进入 encoder、排除 decoder | 已实现 |
| posterior + 一步 prior availability loss | 已实现 |
| prior 使用 prior distribution sample | 已实现 |
| predicted hard mask + argmax fallback | 已实现 |
| predicted mask stop-gradient | 已实现 |
| imagination 采样使用 predicted mask | 已实现 |
| actor loss 使用 predicted masked policy | 已实现 |
| valid-only unimix | 已实现 |
| active train/eval YAML 关闭 force flatten | 已实现 |

## 4. 验证记录

### 已通过

环境目标测试：

```text
12 passed
```

覆盖：

- action mask 状态矩阵基础用例；
- near-EOD；
- market closed；
- insufficient money；
- strict no-op；
- execution_failed 分类；
- 300-step mask-respecting rollout。

Dreamer：

- `dreamerv3/tests/test_action_mask.py` 两个测试直接调用通过；
- `MaskedCategorical` 基础采样、valid-only unimix、finite gradient 通过；
- 之前的短 runner smoke 已完成启动、编译和训练步，无 shape/JIT 错误。

### 已复现缺陷

1. stale table 静默重算，原 mask 合法动作被拒绝；
2. flip commit 故障后仓位、realized PnL、closed trade 和 journal entry 未恢复。

### 未通过或未完成

- 旧 target-position contract 未迁移：7 个 contract 失败；
- replay determinism 测试依赖外部 MySQL，本次环境无法连接；
- Dreamer 环境没有安装 `pytest`，新分布测试使用直接函数调用；
- 未运行 availability warm-up 达标实验；
- 未运行 masked actor 正式短训练；
- 未运行 Monte Carlo；
- 未验证“非 FLAT 活性不塌缩”。

## 5. 阶段 Review 状态

| Phase | 状态 | 结论 |
|---|---|---|
| Phase 1A/1B 环境统一合法性 | 不通过 | true mask 已有，但同 table 消费、quote/commit、完整回滚未完成 |
| Phase 2 MaskedCategorical | 部分通过 | 实现正确方向，测试矩阵不足 |
| Phase 3 真实 Dreamer policy | 部分通过 | 结构已接通，缺 chosen violation 和 replay 对齐测试 |
| Phase 4 availability/imagination | 部分通过 | 主链路已接通，缺时序/梯度自动化，warm-up value gate 不完整 |
| Phase 5 短训练诊断 | 未通过 | 只有最小 smoke，没有预测阈值和活性验收 |
| Phase 6 正式训练回归 | 未开始 | 不具备准入条件 |

## 6. 推荐修复顺序

1. **先修环境事务与单一 decision**
   - execution-ready quote；
   - `_execute_transition(decision)`；
   - 完整 transaction rollback；
   - `_legality_state_version` + consume/fail-fast。

2. **修正环境语义和指标**
   - `invalid_action = action_rejected`；
   - execution failure 不进入 actor reward；
   - `mask_execution_mismatch`；
   - reason counters、cardinality/chosen distributions。

3. **补齐 Dreamer 安全门禁**
   - warm-up 同时处理 `repval`；
   - true mask strict nonempty；
   - 可审计的 warm-up -> actor 切换命令和阈值检查。

4. **补齐最关键自动化测试**
   - replay mask/action alignment；
   - prior availability alignment；
   - imagination mask/action/loss alignment；
   - gradient contract；
   - stale table 与 commit 故障注入；
   - 迁移旧 target-position contracts。

5. **再进入训练验收**
   - availability warm-up 达到方案阈值；
   - masked actor 短训练；
   - Dreamer real rollout rejected ratio；
   - cardinality/chosen-action 活性；
   - Monte Carlo 与收益风险独立评审。

## 7. 下一次 Review 的准入条件

下一次正式 review 至少应满足：

- flip/open/close 故障注入后状态完整不变；
- stale table 必须 fail fast；
- mask-respecting policy 下 `mask_execution_mismatch=0`；
- `execution_failed` 不进入 invalid actor reward；
- replay/prior/imagination 三类时序专项测试通过；
- warm-up 时 policy 和受 predicted imagination 影响的 value 更新均关闭；
- prior 指标达到方案阈值后，使用明确配置进入 masked actor 阶段；
- 旧 contract 测试完成 target-position 迁移。

在上述条件满足前，建议保持 `avail_actor_enabled=False`，不要开始正式策略训练。

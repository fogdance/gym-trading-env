# Gym Trading Env + DreamerV3 无效动作问题技术报告

## 1. 报告范围

本文分析以下两个仓库协同训练时出现的大量无效操作问题：

- 交易环境：`~/Documents/work/gym-trading-env`
- 强化学习实现：`~/Documents/work/dreamerv3`

本文只讨论动作合法性、执行语义以及 DreamerV3 actor/world model 与动态动作约束之间的关系，不讨论行情特征质量、奖励收益性或交易策略本身。

结论基于当前代码和已有 Monte Carlo 结果。当前对根因定位的把握约为 95%，对完整修复方案落地的把握约为 85%。

---

## 2. 执行摘要

当前环境已经将动作空间压缩为三个目标仓位：

| 动作编号 | 目标仓位 |
|---|---|
| `0` | `SHORT` |
| `1` | `FLAT` |
| `2` | `LONG` |

环境会根据当前仓位，把目标仓位转换为 `HOLD`、开仓、平仓或原子反手操作。因此，重复开多、空仓平仓等旧动作空间问题原则上已经消失。

当前大量无效动作的主要来源不是动作编号非法，而是目标仓位在当前动态约束下不可达，例如：

- 当日开仓次数已经耗尽，但 actor 仍要求开仓或反手；
- 临近收盘禁止开仓，但 actor 仍要求开仓或反手；
- 资金不足，但 actor 仍要求建立新仓位；
- 闭市时 actor 仍要求改变仓位。

环境已经把部分许可状态放入 `agent_state`，但 DreamerV3 只把这些值作为普通输入特征。Dreamer 的 categorical actor 不会据此强制屏蔽动作，仍会从三个动作中采样。

更关键的是，DreamerV3 的 actor 主要通过 world-model imagination 训练。在 imagined rollout 中不存在真实环境可供查询。因此，只在真实环境执行前增加 action mask 或把无效动作改成 `FLAT/HOLD`，不能完整修复 actor 的学习目标，反而可能造成“actor 输出动作”和“环境实际执行动作”语义不一致。

推荐方案是：

1. 环境提供唯一、可测试的目标仓位可达性函数；
2. 将三维 `action_mask` 作为独立 observation 暴露；
3. Dreamer 真实交互阶段使用真实 mask；
4. world model 增加 action-mask prediction head；
5. imagined rollout 使用预测 mask；
6. policy sample、argmax、log-probability 和 entropy 全部基于同一个 masked categorical distribution；
7. `eval/live` 模式使用 masked argmax，而不是随机采样。

---

## 3. 当前动作执行模型

### 3.1 对外动作空间

环境对外暴露 `Discrete(3)`：

```python
self.valid_actions = [TargetPos.SHORT, TargetPos.FLAT, TargetPos.LONG]
self.action_space = spaces.Discrete(3)
```

这三个动作表达的是期望目标仓位，不是底层交易指令。

### 3.2 目标仓位到执行动作的转换

环境根据当前仓位和目标仓位规划执行动作：

| 当前仓位 | 目标仓位 | 执行动作 |
|---|---|---|
| `FLAT` | `LONG` | 开多 |
| `FLAT` | `SHORT` | 开空 |
| `LONG` | `FLAT` | 平多 |
| `SHORT` | `FLAT` | 平空 |
| `LONG` | `SHORT` | 多翻空 |
| `SHORT` | `LONG` | 空翻多 |
| 任意 | 相同目标 | `HOLD` |

这个设计比直接暴露开仓、平仓等指令更适合单仓位目标控制，因为重复选择当前仓位会自然变成 `HOLD`。

### 3.3 无效动作的定义

当前环境将以下执行结果统计为无效动作：

```python
invalid_action = (
    self.action_result not in (ForexCode.SUCCESS, ForexCode.ERROR_MARKET_CLOSED)
)
```

当前错误码包括：

- `ERROR_HIT_MAX_POSITION`
- `ERROR_NO_POSITION_TO_CLOSE`
- `ERROR_NO_ENOUGH_MONEY`
- `ERROR_OPEN_POSITION`
- `ERROR_BLOCKED_NEAR_EOD`
- `ERROR_HIT_DAY_MAX_OPEN`

`ERROR_MARKET_CLOSED` 当前不计入 `invalid_action`，但它仍然代表 actor 请求了环境无法执行的仓位变化，应单独统计。

---

## 4. 问题证据

### 4.1 Monte Carlo 结果

对已有一组 200 个 Monte Carlo episode 的 `invalid_action_ratio` 汇总：

| 指标 | 数值 |
|---|---:|
| Episode 数 | 200 |
| 平均无效动作比例 | 54.1% |
| 中位数 | 42.0% |
| P90 | 80.0% |
| 最低 | 22.6% |
| 最高 | 89.0% |

这不是少量边界动作，而是系统性问题。

很多 episode 最终都恰好完成最大允许开仓次数，随后产生大量无效动作。这与“达到每日开仓上限后，部分目标仓位不可达”的判断一致。

### 4.2 环境已提供许可特征，但没有强制约束

环境的 normalized agent observation 已经包含：

```text
obs_market_open_t
obs_can_long_open_t
obs_can_short_open_t
obs_can_long_close_t
obs_can_short_close_t
obs_entries_left_frac_t
obs_action_result_t
```

这些特征可以帮助模型学习动作规则，但不能保证 actor 不采样无效动作。对于高熵 categorical policy，尤其在动作可用性高度动态时，单靠奖励和特征学习通常会保留显著无效动作概率。

### 4.3 DreamerV3 当前始终从完整动作分布采样

真实交互阶段：

```python
policy = self.pol(self.feat2tensor(feat), bdims=1)
act = sample(policy)
```

这里没有读取 action mask，也没有根据 `mode` 区分训练与评估。

因此：

- `train` 使用随机采样；
- `eval` 仍使用随机采样；
- `live` 仍使用随机采样；
- 三个动作始终拥有被采样的机会。

### 4.4 Dreamer actor 在 imagination 中训练

DreamerV3 的 actor 不是只根据真实环境 transition 训练。训练时会从 world-model latent state 开始 imagined rollout：

```python
policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
_, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
```

随后 actor loss 使用 imagined action、reward、continue 和 value：

```python
logpi = ...
entropy = ...
policy_loss = ...
```

因此，完整修复必须覆盖 imagination。只在真实环境 `step()` 前纠正动作，会导致：

1. actor 在 imagination 中继续选择无效动作；
2. actor 输出动作与环境实际执行动作不一致；
3. replay 中保存的是 actor 动作还是执行动作，会影响 world model 的转移学习；
4. actor 的 log-probability、entropy 和真实执行约束不一致。

---

## 5. 无效动作来源分析

### 5.1 达到每日最大开仓次数

这是当前最主要、最稳定的动态约束。

假设开仓次数已经耗尽：

| 当前仓位 | 合法目标仓位 | 不可达目标仓位 |
|---|---|---|
| `FLAT` | `FLAT` | `SHORT`, `LONG` |
| `LONG` | `LONG`, `FLAT` | `SHORT` |
| `SHORT` | `SHORT`, `FLAT` | `LONG` |

反手包含一次新开仓，因此也会触发 `ERROR_HIT_DAY_MAX_OPEN`。

当 `FLAT` 且开仓次数耗尽时，三个动作中只有一个可执行。如果 actor 仍接近均匀探索，理论无效概率可达到 `2/3`。

### 5.2 临近收盘禁止开仓

临近 EOD 时，环境会把所有开仓和反手动作改成 `HOLD`，并返回 `ERROR_BLOCKED_NEAR_EOD`。

这里存在两个语义：

- actor 请求的目标仓位；
- 环境实际保持的仓位。

如果只记录请求动作而不记录实际执行动作，world model 会看到错误的动作转移关系。

### 5.3 资金不足

开仓与反手在执行时都会检查保证金和手续费。资金不足依赖价格、权益、保证金和交易参数，属于动态可达性条件。

资金不足可以提前计算，因此应该进入统一的目标仓位可达性判断，而不是只在执行函数内部返回错误。

### 5.4 闭市或数据缺失

闭市时，环境将非 `HOLD` 动作强制改为 `HOLD`，返回 `ERROR_MARKET_CLOSED`。虽然当前不计入 `invalid_action`，但这仍属于请求动作无法执行。

建议分别统计：

- 策略非法/不可达动作；
- 市场不可交易动作；
- 环境内部或账务错误。

### 5.5 环境许可特征与实际执行规则可能漂移

当前 `can_long_open`、`can_short_open` 等许可特征在多个位置重复计算，而且主要描述“开/平仓许可”，不是三个目标仓位的最终可达性。

例如，当前持有多仓时：

- `can_short_open` 会因为非空仓而是 `0`；
- 但目标 `SHORT` 实际代表反手；
- 反手是否可执行还依赖剩余开仓次数和资金。

因此，现有四个 `can_*` 特征不能直接作为 `SHORT/FLAT/LONG` 的 action mask。

---

## 6. 为什么简单方案不够

### 6.1 仅增加无效动作惩罚

优点：

- 实现简单；
- 不修改 Dreamer。

缺点：

- 无法保证零无效动作；
- 高 entropy 奖励会持续鼓励探索无效动作；
- 当只有一个动作合法时，学习效率很低；
- 大量 imagined rollout 容量浪费在不可执行动作上；
- 无效动作奖励可能污染策略收益目标。

结论：只能作为诊断和辅助约束，不能作为最终方案。

### 6.2 仅把无效动作静默转换为 `HOLD/FLAT`

优点：

- 实盘不会执行危险订单；
- 无效动作指标表面上可以归零。

缺点：

- actor 输出和环境执行不一致；
- replay 动作语义不明确；
- world model 可能学习成“选择 SHORT 也会保持 FLAT”；
- actor 不会真正减少无效动作概率；
- 隐藏了策略质量问题。

结论：可以作为最后一道安全防线，但不能作为学习层修复。

### 6.3 仅在真实环境交互时使用 action mask

优点：

- 真实 rollout 不再采样无效动作；
- replay 数据质量改善。

缺点：

- imagined rollout 仍从未屏蔽分布采样；
- actor loss 与真实行为分布不一致；
- Dreamer 的主要 actor 训练路径仍未修复。

结论：适合第一阶段止损，但不是完整方案。

### 6.4 只依赖 `agent_state` 中的许可特征

优点：

- 不改 Dreamer；
- 模型理论上可学习规则。

缺点：

- 不提供硬保证；
- actor 必须自行学习精确布尔逻辑；
- world model 预测误差会放大动态约束错误；
- entropy 和 exploration 会持续产生无效动作。

结论：许可特征应保留，但不能替代 mask。

---

## 7. 方案对比

| 方案 | 真实交互无效动作 | Imagination 一致性 | 改动量 | 推荐程度 |
|---|---:|---:|---:|---|
| 增加惩罚 | 不能保证消除 | 无改善 | 小 | 不推荐作为主方案 |
| 环境静默改为 HOLD | 表面消除 | 不一致 | 小 | 仅安全兜底 |
| 真实交互 action mask | 可基本消除 | 不一致 | 中 | 推荐作为阶段一 |
| 预测 mask + 全链路 masked policy | 可消除 | 一致 | 大 | 推荐最终方案 |
| 重定义为所有目标动作永远合法 | 可消除 | 一致 | 中到大 | 可作为备选 |

---

## 8. 推荐最终架构

### 8.1 环境侧：唯一目标仓位可达性函数

新增纯逻辑接口：

```python
def target_action_mask(self) -> np.ndarray:
    """Return bool mask ordered as [SHORT, FLAT, LONG]."""
```

要求：

- shape 固定为 `(3,)`；
- dtype 为 `bool` 或 `float32`；
- 至少一个动作合法；
- 与 `step()` 使用同一套前置检查；
- 无副作用；
- 不依赖未来数据；
- `FLAT` 在有仓位时表示平仓，在空仓时表示保持空仓；
- 相同目标仓位始终合法，表示 `HOLD`；
- 反手必须同时满足平仓和新开仓条件。

建议同时返回可诊断原因：

```python
TargetAvailability(
    mask=np.array([False, True, False]),
    reasons={
        TargetPos.SHORT: "MAX_ENTRIES",
        TargetPos.LONG: "MAX_ENTRIES",
    },
)
```

### 8.2 避免许可逻辑重复

当前许可逻辑分散在：

- agent feature 构造；
- near-EOD gate；
- `_long_open()` / `_short_open()`；
- `_flip_long_to_short()` / `_flip_short_to_long()`；
- market-open gate。

建议提取统一的 preflight/quote 层：

```python
def quote_target_transition(current, target, price) -> TransitionQuote:
    # 不修改状态，只判断是否可执行，并计算执行计划。
```

`step()`、`action_mask` 和 agent feature 全部读取这个结果，避免 mask 显示可执行但实际执行失败。

账务提交仍必须保留最终检查和原子回滚，mask 不能替代执行安全校验。

### 8.3 Observation 中独立暴露 `action_mask`

推荐 observation 增加：

```python
"action_mask": spaces.Box(
    low=0.0,
    high=1.0,
    shape=(3,),
    dtype=np.float32,
)
```

不要只把 mask 嵌入 `agent_state`，原因是：

- Dreamer 需要明确识别该字段；
- policy masking 不应该依赖 agent feature 的固定列位置；
- 便于测试、日志和其他 RL 算法复用；
- mask 不应该被 symlog、归一化或普通 feature encoder 语义混淆。

### 8.4 Dreamer 真实策略：masked categorical

真实交互阶段：

```python
policy = self.pol(features)
policy = mask_policy(policy, obs["action_mask"])

if mode == "train":
    action = sample(policy)
else:
    action = policy.pred()
```

屏蔽方式建议使用大负 logits，而不是采样后替换：

```python
masked_logits = jnp.where(mask, logits, -1e30)
```

所有以下操作必须读取同一个 masked distribution：

- `sample()`
- `pred()/argmax`
- `logp()`
- `entropy()`

否则 actor loss 和实际采样仍会不一致。

### 8.5 Dreamer world model：预测 action availability

为 world model 增加三维 binary mask head：

```text
latent feature -> mask head -> predicted availability of SHORT/FLAT/LONG
```

训练目标来自 replay 中真实 observation 的 `action_mask`。

建议：

- 使用独立 binary cross-entropy loss；
- mask loss 加入 `loss_scales`；
- 报告每个动作的 precision、recall、false-positive rate；
- 尤其关注 false positive，因为它会让 imagination 选择实际上不可执行的动作；
- 推理时确保至少一个动作合法，预测全 false 时回退为当前目标仓位或 `FLAT`。

### 8.6 Imagined rollout：使用预测 mask

imagined policy function 应变为：

```python
def policyfn(feat):
    policy = self.pol(self.feat2tensor(feat), 1)
    mask = self.mask_head(self.feat2tensor(feat), 1)
    return sample(mask_policy(policy, mask))
```

actor loss 中重新计算 policy distribution 时，也必须使用对应 imagined mask：

```python
masked_policy = mask_policy(self.pol(inp, 2), imagined_mask)
```

这样可确保：

- imagined action 来自可达动作集合；
- `logp(action)` 对应真实采样分布；
- entropy 只在合法动作集合上计算；
- actor 不会因 entropy bonus 被鼓励探索无效动作。

### 8.7 训练、评估和实盘模式

当前 `Agent.policy()` 忽略 `mode`，所有模式均随机采样。建议明确：

| 模式 | 动作选择 |
|---|---|
| `train` | masked categorical sample |
| `eval` | masked argmax |
| `live` | masked argmax |
| 可选探索评估 | 显式单独模式 |

这项修改与 action mask 独立，但对降低评估和实盘中的无效动作、提升可重复性非常重要。

---

## 9. 备选方案：把所有目标仓位定义为永远合法

另一种思路是改变 MDP 语义，使 `SHORT/FLAT/LONG` 在任意状态下都有确定性合法转移。

例如：

- 达到开仓上限后，从 `LONG` 请求 `SHORT`：先平多，最终变为 `FLAT`；
- 空仓且不能开仓时请求 `LONG`：保持 `FLAT`；
- 临近 EOD 请求开仓：保持 `FLAT`；
- 闭市请求改变仓位：保持当前仓位。

这种方案不需要 Dreamer action mask，但必须重新定义动作语义：

> 动作不再表示“下一步必须达到的目标仓位”，而表示“在约束允许范围内尽量接近目标仓位”。

优点：

- 每个动作始终有合法转移；
- world model 和 actor 不需要动态动作集合；
- 工程复杂度较低。

风险：

- 一个动作在不同约束下具有不同执行结果；
- actor 可能频繁请求不可达目标而不受约束；
- 策略输出不再能直接解释为实际目标仓位；
- 实盘审计和风险控制语义变弱。

如果采用该方案，必须同时记录：

```text
requested_target
executed_target
constraint_reason
```

并确保 replay 使用实际导致状态转移的动作语义。该方案可作为低复杂度备选，但不如全链路 mask 清晰。

---

## 10. 分阶段实施计划

### 阶段 0：冻结动作契约与指标

目标：先明确什么是无效动作，避免修复后仅仅改变统计口径。

任务：

1. 定义 `requested_target`、`planned_action`、`executed_action`、`result_code`；
2. 明确 `ERROR_MARKET_CLOSED` 是否属于策略无效动作；
3. 增加按原因计数；
4. 输出每个目标动作的请求比例、可用比例和被拒绝比例；
5. 添加 action availability 契约测试。

验收：

- 每个拒绝动作都有唯一原因；
- 每个 episode 的无效动作总数等于各原因计数之和；
- action mask 与实际执行前置判断一致。

### 阶段 1：环境统一 preflight 和真实交互 mask

目标：真实 rollout 不再采样明确不可执行动作。

任务：

1. 实现 `target_action_mask()`；
2. 提取统一 preflight；
3. observation 增加 `action_mask`；
4. FromGymnasium 保持该 observation 字段；
5. Dreamer 真实 policy 使用 masked categorical；
6. `eval/live` 使用 masked argmax；
7. 保留环境最终校验与回滚。

验收：

- 随机策略在 action mask 下运行，策略无效动作比例接近 0；
- EOD、开仓上限、闭市、资金不足均有专项测试；
- replay 中 action 与环境实际执行语义一致；
- 原有账务守恒测试继续通过。

阶段 1 可以快速降低真实交互中的无效动作，但 actor imagination 仍不完整。

### 阶段 2：world model mask prediction

目标：让 imagination 拥有动作可达性信息。

任务：

1. 增加 mask prediction head；
2. replay observation 保留真实 mask；
3. 增加 mask prediction loss；
4. 添加 mask precision/recall/FPR 指标；
5. 验证 mask head 在不同仓位、剩余次数和 EOD 阶段的准确率。

验收建议：

- 总体 mask bit accuracy > 99%；
- 不可执行动作 false-positive rate < 1%；
- `FLAT` 或当前目标仓位永远至少一个被预测为合法。

### 阶段 3：imagined rollout 全链路 masked policy

目标：actor 在 imagination 和真实交互中使用相同动作约束。

任务：

1. imagined policy sample 使用预测 mask；
2. actor loss 的 policy/logp/entropy 使用预测 mask；
3. 增加 imagined valid-action-rate 指标；
4. 检查 JAX shape、JIT 和梯度稳定性；
5. 重新训练并与旧策略比较。

验收：

- imagined action 全部满足 predicted mask；
- 真实 episode 无效动作比例接近 0；
- 无效动作下降不是通过减少交易或永远 `FLAT` 实现；
- 交易次数、收益、回撤等核心指标无明显退化。

### 阶段 4：实盘安全与兼容

任务：

1. 环境继续作为最终风险控制边界；
2. live 日志同时记录 requested/executed target；
3. 非法请求在实盘中拒绝并报警；
4. checkpoint 增加 schema/version；
5. 明确旧 checkpoint 是否仅用于离线评估或迁移初始化。

---

## 11. 测试设计

### 11.1 环境单元测试

针对每种状态检查三维 mask：

| 场景 | 预期重点 |
|---|---|
| 空仓、可交易、可开仓 | 三个目标均合法 |
| 多仓、可反手 | 三个目标均合法 |
| 空仓、开仓次数耗尽 | 仅 `FLAT` 合法 |
| 多仓、开仓次数耗尽 | `FLAT`、`LONG` 合法 |
| 空仓、near EOD | 仅 `FLAT` 合法 |
| 多仓、near EOD | `FLAT`、`LONG` 合法 |
| 闭市空仓 | 仅 `FLAT` 合法 |
| 闭市持仓 | 当前目标合法；是否允许平仓需按业务规则冻结 |
| 资金不足空仓 | 仅 `FLAT` 合法 |

每个 mask 为 `True` 的动作都应在相同状态下执行成功。每个 mask 为 `False` 的动作都应返回对应拒绝原因，且不改变账务和仓位。

### 11.2 Dreamer distribution 测试

验证 masked categorical：

- 被屏蔽动作采样概率为 0；
- 被屏蔽动作不参与 entropy；
- masked argmax 不会选择屏蔽动作；
- 仅一个动作合法时，sample 和 argmax 始终返回该动作；
- 全 false mask 有明确 fallback 或直接 fail-fast；
- JIT 前后结果一致。

### 11.3 Imagination 测试

- mask head 输出 shape 与动作类别一致；
- imagined action 满足 imagined mask；
- actor loss 对屏蔽动作 logits 不产生鼓励其被采样的梯度；
- mask prediction loss 能在小型确定性数据集上过拟合；
- checkpoint 保存和加载后 mask head 输出一致。

### 11.4 集成验收

建议同时监控：

```text
log/env/invalid_action_ratio
log/env/invalid/max_entries
log/env/invalid/near_eod
log/env/invalid/no_money
log/env/invalid/market_closed
log/env/requested_target_short_ratio
log/env/requested_target_flat_ratio
log/env/requested_target_long_ratio
log/env/executed_target_match_ratio
train/mask/false_positive_rate
train/mask/false_negative_rate
train/imag_valid_action_rate
```

必须避免“无效动作降为 0，但策略永远选择 FLAT”的伪修复。至少同时比较：

- `trades_opened`
- `exposure_ratio`
- `return_pct`
- `max_drawdown_pct`
- `profit_factor`
- `expectancy`

---

## 12. 兼容性与迁移风险

### 12.1 旧 checkpoint

增加 mask prediction head 会改变模型参数结构。旧 checkpoint 通常不能完整无损恢复。

可选策略：

- 仅加载已有 encoder、RSSM、policy、value 参数，初始化新 mask head；
- 为 checkpoint loader 增加按路径忽略缺失参数能力；
- 使用旧 replay 继续训练前，确认旧 replay 是否包含 action mask；
- 如果旧 replay 没有 mask，可从 observation 中已有许可字段近似重建，但无法保证与最终统一 preflight 完全一致。

### 12.2 Replay schema

新增 `action_mask` 会改变 replay transition schema。必须：

- 增加 schema version；
- 拒绝静默混用不同 schema；
- 或为旧 replay 提供一次性迁移脚本。

### 12.3 策略分布变化

masked entropy 与原始三动作 entropy 不可直接比较。动作集合只剩一个合法动作时，entropy 应为 0，这是正确行为。

训练配置中的 entropy 系数需要重新评估，不能直接沿用为鼓励三个动作均匀探索而设置的高值。

---

## 13. 相关但独立的问题

以下问题不是无效动作的直接根因，但会影响诊断与修复效果：

1. `Agent.policy()` 当前忽略 `mode`，导致 eval/live 仍随机采样；
2. 环境许可特征和实际执行校验分散，存在规则漂移风险；
3. 当前无效动作指标没有按原因拆分；
4. `ERROR_MARKET_CLOSED` 不计入无效动作，容易低估不可执行请求；
5. 环境首步奖励初始化、训练示例版本漂移等问题应单独修复；
6. 当前测试集中仍有大量旧动作接口测试，需要统一到目标仓位动作契约。

---

## 14. 推荐决策

建议采用“阶段 1 + 阶段 2 + 阶段 3”的完整 masked-policy 方案。

短期先完成阶段 0 和阶段 1，可以快速获得：

- 可解释的无效动作原因；
- 真实 rollout 中接近零的不可执行动作；
- deterministic eval/live；
- 更干净的 replay 数据。

随后完成 world-model mask prediction 和 imagined rollout masking，解决 Dreamer actor 学习目标与真实环境约束不一致的问题。

不建议将“无效动作静默转换为 HOLD”作为最终方案。它可以保留为环境安全兜底，但必须记录请求动作、实际动作和拒绝原因，且不能替代 actor/world-model 层面的约束建模。

---

## 15. 关键代码位置

### gym-trading-env

- 动作空间：`src/gym_trading_env/envs/trading_env.py`
- 目标仓位到执行动作转换：`CustomTradingEnv._plan_exec_action`
- 动作执行与无效动作统计：`CustomTradingEnv.step`
- 开仓约束：`CustomTradingEnv._long_open`、`_short_open`
- 反手约束：`CustomTradingEnv._flip_long_to_short`、`_flip_short_to_long`
- Agent 许可特征：`src/gym_trading_env/utils/agent_features.py`
- 无效动作指标：`src/gym_trading_env/envs/metrics.py`

### dreamerv3

- 真实策略采样：`dreamerv3/agent.py::Agent.policy`
- imagined rollout：`dreamerv3/agent.py::Agent.loss`
- actor loss、logp 和 entropy：`dreamerv3/agent.py::imag_loss`
- categorical distribution：`embodied/jax/outs.py`
- policy head：`embodied/jax/heads.py`
- Gymnasium observation/action 适配：`embodied/envs/from_gymnasium.py`
- 环境包装：`dreamerv3/main.py::wrap_env`

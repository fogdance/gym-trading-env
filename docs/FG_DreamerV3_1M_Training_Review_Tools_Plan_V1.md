# FG-DreamerV3：百万步训练冻结评审工具实施方案 V1

制定日期：2026-10-10。状态：**实施方案，评审工具尚未实现，本文件不是训练结果或评估报告。**

依据：[FG_DreamerV3_1M_Training_Review_Protocol_V1.md](FG_DreamerV3_1M_Training_Review_Protocol_V1.md)。本方案将该协议落实为入口、数据流、统计口径、验收条件和实施顺序；原协议不覆盖。

用户已确认：完整规划评审工具；反事实候选采用固定抽样，每合约最多 200 个模型实际入场点；G5 仅列出实际风险数值，不自行设通过阈值。

## 1. 本轮目标及不可变业务合同

目标是判断模型是否具有超出合法即开仓、随机入场、简单回撤规则的入场择时能力。方向来自人工配置，不评价模型预测涨跌的能力。

不可变合同：

- 只使用 FG；WAIT=0，ADD_3=1；每批 3 手，最多同时 6 手。
- 唯一正奖励是独立入场批次完整实际模拟成交 +5 点 TP 后的 `60/(60+T)`，T 是实际入场成交至出场成交的自然分钟。
- 不加入浮亏、净值、回撤或额外持仓时间惩罚。不因模型表现差而调整 Reward。
- 下一合法成交事件执行信号，沿用既有保守分钟撮合、成交量及执行时方向复核。
- 换主力按旧合约最后可成交时刻退出，不产生速度奖励。数据截止保留未完成账户，不伪造到期或免费清除持仓。
- 行情 `(60,18)`、仓位 `(9,)`、动作 mask `(2,)`；不得扩展技术指标。
- 统一 345 分钟名义时钟，导出阶段已截掉 23:01～23:30，23:00 保留。评估使用本次训练的冻结版本，不能重新导出、重拟合或换标签。

实施过程不停止、重启、修改正在运行的训练。现在允许设计及开发独立工具、合成环境测试和只读证据检查；模型 GPU 评估安排在训练结束后。禁止触发额外训练及 CPU 模型训练。

## 2. 已确认的实际运行信息

以下来自本次读取的运行配置及数据清单，不是协议旧快照的推定：

| 项目 | 实际值 |
| --- | --- |
| Env 原仓库 | `/home/v/Documents/work/gym-trading-env` |
| Dreamer 原仓库 | `/home/v/Documents/work/dreamerv3` |
| 两仓库开发分支 | `feat/fg-entry-v1.1` |
| 本次运行目录 | `/home/v/Documents/work/gym-trading-env/artifacts/fg_entry_v1/train_345_seed0_20261010_160106` |
| effective config | 上述目录的 `config.yaml` |
| 当前冻结数据集 | `/home/v/Documents/work/gym-trading-env/artifacts/fg_entry_v1/dataset_v3_345_audited` |
| 数据 manifest hash | `e244dc0abfe3914f754544f13b1bf71a21f5281d467c73d02ac0331e16f2cad6` |
| experiment seed / Dreamer seed | 0 / 0 |
| 训练目标 | 1000000 个环境步，不是百万次梯度更新 |
| batch / sequence / train ratio | 16 / 64 / 64 |
| 指定步数 checkpoint 保留 | `enabled=false, steps=[0]` |
| 本次训练 seed 数 | 1；不据此宣称训练稳定性 |

实际切分：

| 角色 | 首交易日 | 末交易日 | 交易日数 | 分钟记录数 |
| --- | --- | --- | --- | --- |
| TRAIN | 2018-04-11 | 2022-12-09 | 1136 | 379305 |
| VALIDATION | 2022-12-12 | 2024-08-19 | 410 | 139890 |
| TEST | 2024-08-20 | 2025-12-31 | 332 | 112980 |

表内记录数不是全部可训练或有方向有效分钟数。FG809 没有人工方向，不准入训练；训练实际准入 13 个区段。FG301 完整进入训练，FG409 完整进入验证。最终 FG605 在数据截止时不完整，测试仍须报告其未完成持仓。

本次冻结 TEST 的方向全部为 SHORT，不能由其结果推断 LONG 泛化。

文档制定时读取的 Env HEAD 为 `ff6299e5e11f7277bbfd7b4f03c61202d533e0f9`，Dreamer HEAD 为 `557f12b828ef09e938cfb225c0e90ccf3e01162c`。**这是读取时版本，不自动等同于训练进程启动时版本。** 若没有启动日志或其他版本证据，运行时 Git 版本标为 `NOT_RECORDED`，同时保留本次读取版本及差异。

## 3. 实现位置及职责

复用两个原仓库，不新建独立 worktree：

1. Dreamer：扩展 `tools/fg_entry_evaluate.py`，新增 `tools/fg_entry_review.py` 统一编排入口；必要时新增仅供评估的诊断接口。
2. Env：扩展 `src/gym_trading_env/research/fg_entry_v1.py::run_policy()` 的可选记录能力，统计分析放在独立研究模块，避免把报告逻辑塞入交易 `step()`。
3. 复用 `FixedDirectionEntryEnv`、PositionManager、真实模拟成交、账本与退出协议。禁止另写一套更有利的撮合器用于对照或探针。

新增配置只描述评审执行参数，不修改当前训练引用的统一 `fg.yaml`。方向、成本、业务规则仍只来自冻结配置。评审参数从 CLI 解析后集中写入输出目录的 `review_plan.json`；它是冻结运行产物，不是另一份人工业务配置。

源码、测试、说明文档按阶段提交。行情、审计明细、模型、图片、日志、评审 JSON/CSV 等生成产物不提交。

## 4. 阶段入口及执行约束

拟新增 CLI，以下接口与命令是**待实现接口**，当前不可当成已有工具运行。

### 4.1 入口参数

| 参数 | 含义 / 默认 |
| --- | --- |
| `--phase` | `inspect / audit / validation / test / summarize` |
| `--run-dir` | 已存在的训练目录，必填 |
| `--output` | 新评审根目录，必填；默认建议在 `/data/logdir/fg_entry_v1/reviews/` 下 |
| `--checkpoint` | audit 可选显式完整 checkpoint；否则解析 run 的 latest。validation 枚举仍存在的完整候选 |
| `--selection` | test 必填，指向验证阶段的冻结选择文件 |

V1 固定默认参数：随机对照 seeds 0–29；bootstrap seed 0、10000 次、95% 区间；探针 seed 0、每合约最多 200 个模型点、每点一个控制点。全部进入冻结执行清单，TEST 不得临时改值。

`inspect` 创建评审根目录，写入冻结计划与证据。其他阶段要求根目录存在且冻结参数相符。每个阶段拥有独立子目录，已存在的完成或失败阶段均拒绝覆盖；失败结果保留，新实验或重试使用新目录。通过状态明确后才允许后续阶段。

统一使用 UTF-8、时间带 `+08:00`、自然分钟可为小数。金额字段写明人民币元；比例范围 0～1；缺失数值使用 JSON null 或 CSV 空值，另用 status/reason 说明，不用 0 代替缺失。JSON 禁止 NaN/Infinity。

### 4.2 inspect：现在可做的证据保护

- 读取 effective config、源 manifest、方向/成本/观察版本、两仓库 HEAD、工作树 diff、seed 和运行路径。
- 提取训练 step、实际 updates、loss、warm-up、资源与速度曲线；区分 `train/`、`report/`、`action_mask_gate_eval/`。
- 日志仍追加时只捕获读到的完整行前缀，记录字节位置、时间及 `partial=true`；末尾半行不当作损坏日志。最终审计重新读取完整日志。
- 当前正在写入的 checkpoint 不复制。只记录已有 `done` 的目录清单与可用步数；目录可能随后被训练清理，检查时不存在则记录事实。
- 明确是否保存了 25万/50万/75万等快照；没有就记 `NOT_RETAINED`，不承诺还原参数。

### 4.3 audit：完整训练终点检查

- 训练结束后确认 checkpoint 步数达到目标；正常批量步进超过目标少量步数可以接受，但必须报告实际值。
- checkpoint 必须有 `done`，并能读取 step、agent、Replay、Driver、warm-up；引用的 Replay 块须存在。
- 读取或复制完整 checkpoint 前后校验其文件集及内容哈希；发现仍变化，停止本阶段，不从半份状态生成评估结果。
- 审核模型和账户结构、有限性、金融状态及 Actor gate。禁止为了评审手动将 gate 从 0 改为 1。
- G0 失败时允许输出诊断，禁止继续正式 validation/test。

### 4.4 validation：选模

- 枚举实际仍保留的 checkpoint；Actor 未启用或结构不兼容的候选自动不合格。
- 所有候选只读取 VALIDATION，使用 deterministic policy，执行同一对照矩阵。
- 排名按总体 `速度奖励/10000 有方向有效市场分钟`；不按单笔平均 Reward、训练 score 或净收益挑选。
- 同分使用较早 checkpoint；比较主指标使用绝对容差 `1e-10`。
- 输出 `g4_validation_selection.json`，锁定模型内容哈希、配置、数据、评估实现版本、随机 seeds、参数和选择理由。
- 只有一个合格 checkpoint 时也跑验证，说明“无候选间选择”，不得编造选择曲线。

### 4.5 test：冻结验收

- 只接受 validation 生成的 selection 文件，全部哈希匹配后运行。
- 所有 TEST 区段参与；FG605 的 DATA_END 明确保留 open/censored 记录。
- 不用 TEST 挑 checkpoint、改超参或选择最佳随机 seed。
- 同一模型复跑两次只用于确定性审计，单独标记；不能把两次当成独立样本。
- 读取 selection 的操作不授权训练、不自动重建数据。

### 4.6 summarize：仅汇总已有结果

读取各阶段完成文件，生成总表、图和总结。不加载新模型、不重新评估、不根据结果重调参数。

## 5. 评估运行器和决策记录

### 5.1 最小接口扩展

`run_policy()` 保留现有参数及原始输出，新增可选审计记录开关，默认关闭。需要时返回或写出细化记录；不改变 Env 决策、成交和 Reward。

每个策略/合约开始时建立独立 30 万账户和新的模型 carry。1024-step chunk 只暂停并继续同一账户，不能清仓或重置模型时序；真实合约终止后下一独立区段才重置。

市场特征可以按冻结输入哈希缓存并复用只读数组。现金、PositionManager、ledger、持仓、carry、随机数状态不得跨策略共享。缓存不得拟合或重算不同公式。

### 5.2 action_decisions.csv

每个决策至少记录：

| 字段组 | 内容 |
| --- | --- |
| 身份 | review_id、role、checkpoint_id、strategy、random_seed、contract、account_sequence、decision_index |
| 时间与市场 | decision_time、trading_date、外部 direction、已知过去 20 根的波动/回撤上下文 |
| 决策前状态 | mask_WAIT、mask_ADD、filled_lots、entry_count、可合法 ADD 标记 |
| 模型输出 | requested_action；可获得时记录 raw logits、masked 概率；否则明确缺失 |
| 执行 | 实际下一事件时间、opened_entry_ids、opened_quantity、reject_reason、executed_ADD 标记 |
| 决策后状态 | filled_lots、entry_count、status、该步 Reward |

真实请求 ADD 与成交 ADD 分开统计。用新增 committed 开仓 fills 判断成交，不以动作值 1 代替成交。记录决策前后 fill 增量，以免把已有仓位的退出错误算作本次开仓。

终点没有下一可成交事件的决策单列 `NO_NEXT_EVENT`，不混进可执行开仓机会。强制 WAIT、主动 WAIT、请求失败各有计数。

### 5.3 entry_outcomes.csv

以 `strategy + seed + contract + account_sequence + entry_id` 为批次唯一键，记录实际成交时间/价格/手数、外部方向、全部退出归因、费用、TP 时间、Reward、实际状态、自然持仓年龄、未实现 PnL 和单笔 MAE。

状态至少区分：完整 TP、已退出但非 TP、仍持仓、数据终点右删失。部分退出或混合退出不能冒充完整 3 手 TP。单笔 MAE 沿用成交方向和可执行价差口径，在实际持有期间的 OHLC 不利价格上计算，注明这是分钟级模拟审计。

## 6. G0：工程有效性与账务审计

### 6.1 模型和暖机

- 验证 `(60,18)/(9,)/(2,)`，动作索引及 always-available WAIT。
- 检查所有可读取参数和全部日志中的 loss 是否有限；不能只抽取最后一个 loss。
- 同时验证 checkpoint 中 `avail_actor_gate/value` 与 warm-up 状态；记录切换步数、切换后的实际 updates 及 policy/value loss。
- Actor 终点未启用，或整个策略学习阶段 policy/value 始终为 0：G0 FAIL，说明不能评价择时策略。
- 现代码暖机实际使用 `action_mask_gate_eval/*` 聚合结果；普通 `report/avail/*` 与训练准确率另列，不混称 gate。
- WAIT 恒合法、满仓 ADD 被禁止；审计 imagined invalid-rate、fallback 及 posterior/prior 精度，阈值读取本次 effective config，不另设宽松值。

### 6.2 Reward 与财务

逐项复算每个可用账户和新评估账户：

1. Reward 唯一对应实际完整 3 手 TAKE_PROFIT commit。
2. 实际成交后自然分钟 T 非负，按 `60/(60+T)` 复算，累计绝对误差不超过 `1e-10`。
3. 批次键不能重复奖励；非 TP 退出和未完成头寸不得贡献速度 Reward。
4. 核对 ledger 借贷守恒、现金、保证金、PositionManager 数量、fills、退出归因和账户权益。
5. Decimal 金额按原引擎精度核对，不用浮点大容差掩盖账务差异。

历史训练若只存了当前合约账户、未存前面合约逐笔账本，只审计有记录部分并声明范围。Replay 与 TensorBoard 平均快照不能补出缺失账本。新评估必须完整输出明细。

### 6.3 确定性与执行边界

同模型、同输入重复运行：动作、成交、Reward、费用和最终持仓一致；神经网络诊断浮点值使用 `rtol=1e-6, atol=1e-6`，但实际离散动作必须完全一致。

audit 阶段的真实模型确定性复核使用按时间排序的首个完整 VALIDATION 区段，运行两次；不提前读取 TEST，也不按表现挑选复核区段。无合格完整验证区段时停止并说明原因。该复核属于 G0，不替代随后全部验证区段的选模评估。

沿用并验证：逐分钟前缀 Obs 一致、下一合法成交事件、同分钟保守 TP、执行时方向变更、跨休市信号、主力退出、保证金退出、DATA_END 与 chunk 继续。测试不得修改原撮合语义。

产物：`g0_integrity.json`、`replay_reward_audit.csv`、`mask_warmup_timeline.csv`、`financial_ledger_audit.json`。其中 replay_reward_audit 标明证据来源，不暗示只有 Replay 就能重建实际成交。

## 7. G1–G2：行为及奖励统计口径

### 7.1 行为

合法机会以决策前 `action_mask[1]==1` 且存在下一可成交事件统计。输出主动 WAIT 数、请求 ADD 数、实际成交 ADD 数及执行失败数；另外统计 mask 禁止时的强制 WAIT。

主动 WAIT 比例 = 主动 WAIT 数 / 合法机会数。分母为 0 时比例为 null，不标为 0% 或 100%。按合约、方向、年度、0/3/6 手状态分组。市场状态分组仅使用当时已有的过去 20 根波动及回撤，保留连续值，不从 TEST 重新拟合分位阈值。

### 7.2 主指标及分母

```text
speed_reward = sum(每个完整 TP 的 60/(60+自然分钟))
reward_per_10k = speed_reward / directed_valid_market_minutes * 10000
```

`directed_valid_market_minutes`：冻结区段内时间长度为 1 分钟、OHLC 有效、人工 direction 为 LONG/SHORT 的行情记录数。零时长元数据不计分钟；NO_TRADE 不计。分母来自数据，不读取策略仓位、收益、margin 或 action_mask，也不因模型提前结束账户而缩短。

总体主指标必须先汇总 Reward 和市场分钟，再做除法，不能把各合约密度简单平均。每个策略使用同一角色、同一合约集合的同一分母。

### 7.3 完成率与删失

- TP 完成比例 = 完整 TP 批次数 / 实际开仓批次数；同时给出非 TP 退出和仍持有数量。
- TP 用时仅对完成 TP 给 p10/p50/p90/mean，必须与全部开仓和删失数量同时呈现。
- 对 H=5、15、60、240、1440、4320 自然分钟：TP 在 H 内计成功；观察达到 H 且没有 H 内 TP，或在 H 前已非 TP 退出，计可判定未成功；H 前结束观察且仍持仓为右删失，不进入该 H 的可判定分母。
- 各 H 同时输出成功数、可判定数、右删失数、开仓总数。可判定比例不是生存概率估计，不将两者混称；V1 不引入额外生存模型。
- 保存按自然时间和 trading_date 的累计 Reward 曲线，不使用 chunk episode score 代替。

产物：`g1_behavior_by_contract.csv`、`actor_eval_diagnostics.json`、`g2_reward_and_completion.csv`、`tp_wait_distribution.csv`、`censored_positions.csv`、`reward_curve.png`。

## 8. G3：基线、统计区间和反事实探针

### 8.1 对照矩阵

| 策略 | 规则 | 次数 |
| --- | --- | --- |
| Dreamer | `mode='eval'`，现有 pred/argmax | 每 checkpoint、每合约 1 次；额外 1 次确定性复核 |
| Always WAIT | 始终 WAIT | 每合约 1 次 |
| Always ADD | 合法即 ADD | 每合约 1 次 |
| Rule ADD | 沿用现有 20-bar/5-point 回撤条件 | 每合约 1 次 |
| Random ADD | 合法时 Bernoulli(p=0.5) | 每合约 seeds 0–29 |
| Matched-frequency Random | 每合约 p=模型在合法机会上的请求 ADD 比例 | 每 checkpoint、每合约 seeds 0–29 |

无合法机会时 matched p=0，并标明退化情形。Matched p 按请求动作而非成交数计算，与原运行器 `1-voluntary_waits/legal_add_decisions` 一致；实际开仓数、资金占用和机会分布另列。它不是精确开仓次数或逐状态反事实匹配，不自动宣称完全公平。

同一 seed、同一合约的随机对照需要可复现，但各策略会产生不同持仓路径，不能把不同合法机会上的随机数解释为相同时间的共同动作。

### 8.2 统计方法

- 输出各随机策略 30 个 seeds 的全部结果、均值、p10/p50/p90，不能选择最差随机样本作对照。
- 主差值是模型速度奖励密度减去相应固定基线或 30-seed 随机基线均值。
- 合约块 bootstrap 作为主区间：有放回抽取合约块，双方使用相同抽样块；10000 次，seed 0。每次先加总 Reward/分钟，再相减。
- 月份块作为补充稳健性分析，按角色内自然交易月份汇总，双方配对重抽月份，参数相同；不把每一分钟当成独立样本。
- 分合约检查、去掉贡献最大的合约、去掉模型最大单笔 TP 后复算差值，披露优势是否集中。
- 区块少、TP 少时显示实际样本数、区间及局限，不隐藏零交易区段。只有一个区块时不生成伪精确区间。

产物：`g3_comparison_by_contract.csv`、`matched_random_distribution.csv`、`paired_strategy_comparison.json`、`confidence_intervals.json`。

### 8.3 反事实候选探针

每合约按模型实际 committed 入场点排序，数量超过 200 时用独立 seed 0 的抽样器均匀无放回取 200 点；否则全部纳入。固定抽样不读取未来 Reward、盈亏或 TP 时间。

每个模型点匹配一个同合约、同方向、同交易月份的控制候选。候选池只使用决策时已知数据：人工方向、已知交易时段、空仓账户当时的合法开仓 mask；排除与模型点相同的时刻及没有下一计划事件的区段终点。**不得按下一根实际 Open、Volume、未来 TP 或未来风险结果筛选候选。** 下一成交事件的价格、成交量、资金和方向复核由引擎实际执行，失败候选仍保留记录。池不足时允许不同模型点复用候选并披露唯一候选数；池为空记 `NO_MATCH`，不跨方向或月份兜底。

候选是独立空仓账户、只允许一个 3 手批次，不受原并发账户是否满仓影响；这是隔离选点质量的诊断，不是账户策略收益。

将候选以前的行情用于同版本因果特征上下文，在目标决策时刻保持空仓，发一次 ADD；随后只 WAIT，直到 TP、既有非 TP 风控退出或数据终点。不得用未来波动选择点、降低手续费、放宽成交量或使用开仓前极值。可缓存市场数组，不能绕开实际 `step()` 的成交协议。

输出每对候选的入场请求/成交时间、价格、TP 等待、退出原因、删失年龄、流动性拒绝及差值。无真正入场时不编造 TP 用时。

产物：`counterfactual_entry_quality.csv`，明确与并发账户总收益分离。

## 9. G4：冻结选模及综合解释

选择文件至少包含 review_id、选择规则、全部候选及排除原因、所选 checkpoint 路径/步数/内容哈希、effective config hash、manifest hash、方向/成本/Obs 版本、评估代码版本、参数、随机 seeds、验证主指标及未完成头寸。

TEST 检验工程正确性与择时证据，不反过来调整模型。验证/测试账户从空仓开始，不继承训练头寸。FG605 在 2025-12-31 的 open/censored 必须纳入风险表和 Reward 分解。

报告分层状态，不把一条总评级掩盖不同事实：

- 工程失败：`NO-GO`，停止正式择时结论。
- 工程通过但等同固定动作或不优于强基线：记录 `NO-EDGE` 证据。
- 单 seed、方向覆盖单一、样本量有限：泛化结论 `INCONCLUSIVE`。
- 后续至少 3 个训练 seeds 且工程、行为、样本外区间及跨合约证据充分后，才讨论 `RESEARCH PASS`；本次不自动授予该等级。
- G5 无用户风险阈值，不产生“风险通过”或 `LIVE CANDIDATE`。

## 10. A2 仓位依赖诊断

这是诊断，不是评分绝对不变的硬准入要求。V1 已选择仓位输入，架构不保证严格 A2。

- 以固定抽样入场点为诊断点，通过真实引擎回放产生可行的空仓、单批、满仓账户状态；无法构造的状态说明原因，不编造仓位字段。
- 同一市场历史、方向、模型 seed 和进入该时刻前的 carry，分别输入各真实仓位 Obs；各分支深拷贝 carry，不能依次累积状态。
- 新增仅评估使用的诊断接口输出 raw WAIT/ADD logits、raw 概率、真实 mask 后概率和动作；默认 policy/train 输出不变。
- 满仓真实 mask 必然禁止 ADD；对依赖性的判断主要比较 raw logits。可另做固定 mask 的评分计算，但绝不向交易引擎执行被禁止动作。
- 未能暴露 logits 时标为 `NOT_RECORDED`，不能用执行动作代替 raw 评分。

产物：`a2_position_sensitivity.csv`，包含各状态、评分差值、可行性和诊断口径。

## 11. G5：实际风险数值报告

**按用户确认，G5 不设阈值，不做自动 PASS/FAIL。**

| 指标 | 计算口径 |
| --- | --- |
| 期末净权益变化 | 最后账户权益减初始 300000；包含仍持有头寸的浮盈亏 |
| 已实现 / 未实现 | 分别来自账本和终点真实仓位标价，费用独立列出并说明是否已包含，禁止重复扣费 |
| 手续费、价差、滑点 | 来自 actual fills 与冻结成本；价格内含价差不能再从 net PnL 扣一次 |
| 最大回撤 | 权益曲线的历史峰值减当前权益，给元及峰值百分比；单列收盘曲线与已有不利 OHLC 审计点口径 |
| 最长回撤持续时间 | 前次权益峰值至恢复或区段终点的自然时间，未恢复标记 ongoing |
| 最大浮亏 | 沿用实际账户 equity_history/引擎 max_floating_loss，标明 OHLC 模拟属性 |
| 单笔 MAE | 实际持有期间不利可执行价格带来的最大损失；自然观察终点包含未完成单 |
| 满 6 手市场分钟占比 | 分钟区间内实际持仓为 6 手的有效一分钟数 / 区段全部有效市场分钟，不只统计有方向分钟 |
| 最长连续满仓 | 同时给连续有效市场分钟和自然分钟；休市不制造交易分钟，持仓未退出时自然计时继续 |
| 未完成仓位 | 批次、手数、成交时间、自然年龄、目标距离、未实现 PnL、DATA_END 等原因 |
| 非 TP 退出 | 换主力、保证金、合约退出等的次数和实际损益，速度 Reward 为 0 |
| 单位时间损益 | 区段净损益 / 实际交易日数；另列自然日数及自然日均值，未记录时不用外推年化 |

仓位的分钟占用尽量从 committed 开平仓时间重建，不能简单把步后仓位当成整根 K 线仓位。无法判断的分钟内先后顺序以现有撮合时刻为准并标明模拟口径。

每合约都是独立账户：总体净损益可以相加；总体风险表列出各合约值与最差区段。不能把各合约重新注资后的权益拼成真实连续账户曲线，也不能直接相加最大回撤。没有连续资金模型时总体连续回撤为 `NOT_APPLICABLE`。

手续费为用户常数，不声称逐历史交易所费率。Tick/盘口成交真实性未记录则标 `NOT_RECORDED`。本轮不自动改成本或撮合规则运行另一组“真实值”；可在后续单独授权的敏感性实验中比较。

产物：`g5_risk_by_contract.csv`、`g5_risk_summary.json`，风险状态标为 `DESCRIPTIVE_ONLY`。

## 12. 交付目录与可追溯性

默认根目录示例：`/data/logdir/fg_entry_v1/reviews/review_seed0_<timestamp>/`。

```text
review_seed0_<timestamp>/
  review_plan.json
  00_provenance/            # 配置、代码证据、输入哈希、可用 checkpoint、读取范围
  01_integrity/             # G0、Reward、账本、warm-up 审计
  02_training_diagnostics/  # step/updates/loss/资源/暖机曲线，历史缺失说明
  03_validation/            # 每候选每合约明细、验证选模文件
  04_test_frozen/           # 选定模型的正式测试、重复确定性审计
  05_baselines/             # 固定/随机对照、30-seed 分布、配对区间
  06_risk/                  # G5 实际值、open/censored、逐笔费用
  07_sensitivity/           # A2 与反事实候选；不自动改成本/撮合
  review_summary.md
```

每个策略/合约目录保留现有 fills、退出归因、reward_events、account_equity、account_state 和 summary，追加决策/批次明细。所有汇总带来源路径及哈希。

每阶段写出 completion JSON，包含状态、开始结束时间、输入哈希、输出文件清单、失败原因。可报告 `PASS / FAIL / NOT_RECORDED / NOT_APPLICABLE / INCONCLUSIVE`，不能把缺失当成通过。

## 13. 最小必要测试与验收

### 13.1 测试优先顺序

先 Env/统计纯逻辑测试，再入口集成测试，最后训练结束后的 GPU 评估 smoke。CPU 上可运行账务与统计测试，但不能在 CPU 上训练或评估 Dreamer 模型作为 GPU 降级方案。

| 测试 | 必须证明 |
| --- | --- |
| 奖励审计 | 唯一完整 3 手 TP，1/2/5 分钟、跨夜自然时间、非 TP 零 Reward，重复事件被拒绝 |
| 决策记录 | 主动/强制 WAIT、失败 ADD 与实际成交 ADD 不混淆；终点无下一事件单列 |
| 账户连续性 | chunk 不丢仓；真实主力退出后才能结束；DATA_END 保存亏损和头寸 |
| 统计口径 | 零交易、零分母、不同策略相同市场分母、总体先加总后除法、快照 sum 不冒充收益 |
| 删失 | H 前 TP、H 后 TP、H 前非 TP 退出、观察不足 open 的分母和状态正确 |
| 风险 | 用小型已知权益曲线和已知开平仓时刻验算回撤、费用、满仓时间，G5 只输出值 |
| 随机与隔离 | 固定 seeds 复现；carry 和金融状态不跨策略/合约污染 |
| 探针 | 不读未来选点、只开独立 3 手、同分钟保守 TP、无控制候选、右删失 |
| checkpoint/阶段 | gate=0、无 done、不完整文件、哈希变更、旧 Obs 版本、覆盖目录均被拒绝 |
| 冻结测试 | 只用验证选模，同分选较早；test 无 selection 不运行，参数变更不运行 |
| A2 | 同一原始 carry 复用，raw 与 masked 区分，禁止执行满仓 ADD |

不追加与本次变化无关的大规模组合测试；复用已有环境回归来保护撮合与 Reward。

### 13.2 阶段验收条件

- G0：强制工程项均通过才允许择时评估；历史缺失必须披露范围。
- G1：完整行为分解能辨认 Always WAIT、Always ADD 和真正可选时刻；不把 WAIT 多当作一定好。
- G2：逐笔 Reward、分母、完成及删失可复算，总表不靠训练 score 推断。
- G3：对照规则与 seeds 固定，跨合约和随机分布完整，局限明确。
- G4：验证选择文件冻结，独立 TEST 未参与选模，单 seed 和 SHORT-only 限制突出。
- G5：真实风险数值和来源完整，缺失不猜测，无未经用户指定的风险阈值。

## 14. 分阶段实施及提交

1. **证据与入口**：实现 inspect、冻结参数、完整 checkpoint 枚举及结果目录保护；提交 CLI 和验证说明，不提交运行产物。
2. **运行器与审计**：实现决策/批次记录、G0、G1、G2、G5 实际值；通过合成样本及既有撮合回归，提交代码与测试说明。
3. **批量对照与选模**：实现 GPU 模型加载复用、跨合约矩阵、30-seed 对照、bootstrap、validation selection/test 冻结；不执行 TEST。
4. **专项诊断**：实现固定 200 点探针与 A2 评估诊断；保持原训练 policy 默认路径不变，专项记录测试通过后提交。
5. **真实验收**：确认训练结束和完整 checkpoint，按 audit → validation → 冻结选模 → test → summarize 顺序执行。现阶段的工具实施请求不自动授权提前执行正式 TEST。

每阶段完成后给简短进度、实际测试结果和提交号。训练未收敛、暖机未通过、日志不足，都只形成评审结论，不成为修改业务规则的理由。

## 15. 实施后的调用示例

以下仅说明拟实现的入口；本文件编写时工具不存在。

```bash
conda activate dreamerv3
cd /home/v/Documents/work/dreamerv3
export JAX_PLATFORMS=cuda
export PYTHONPATH="/home/v/Documents/work/dreamerv3:/home/v/Documents/work/gym-trading-env/src${PYTHONPATH:+:$PYTHONPATH}"

fg_review_run=/home/v/Documents/work/gym-trading-env/artifacts/fg_entry_v1/train_345_seed0_20261010_160106
fg_review_out=/data/logdir/fg_entry_v1/reviews/review_seed0_$(date +%Y%m%d_%H%M%S)

# inspect 可在训练中运行；不加载模型、不复制正在写入的 checkpoint。
python tools/fg_entry_review.py --phase inspect --run-dir "$fg_review_run" --output "$fg_review_out"

# 下列步骤安排在训练结束后，G0 通过后才继续。
python tools/fg_entry_review.py --phase audit --run-dir "$fg_review_run" --output "$fg_review_out"
python tools/fg_entry_review.py --phase validation --run-dir "$fg_review_run" --output "$fg_review_out"

# 模型选择文件冻结后，明确执行独立测试。
python tools/fg_entry_review.py --phase test --run-dir "$fg_review_run" --output "$fg_review_out" \
  --selection "$fg_review_out/03_validation/g4_validation_selection.json"
python tools/fg_entry_review.py --phase summarize --run-dir "$fg_review_run" --output "$fg_review_out"
```

TensorBoard 和训练日志不迁移、不覆盖。评审使用的默认输出根目录遵循 `/data/logdir`；实际训练仍以本次已运行的 artifacts 目录为证据来源。

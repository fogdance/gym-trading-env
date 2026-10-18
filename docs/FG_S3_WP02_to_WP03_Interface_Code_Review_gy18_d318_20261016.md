# FG-S3 WP02 → WP03 接口级源码 Review（gy18/d318）

- 日期：2026-10-16
- 类型：独立静态源码审阅＋归档内报告/收据核对；**不是**本机真实行情重新回放、完整 JAX 测试或训练准入。
- 范围：只读解析用户上传 Git 归档 `gy18.tar.bz2`、`d318.tar.bz2` 中 `feat/s3-e-quality-wp02-v2-20261016` 提交，未修改原仓库、未读取 TEST、未启动训练。
- 审阅提交：Gym `e57189bad100465c9fee8d9bbb5601f3e5c376ad`；Dreamer `44a6cc62032811501631e83be977f07137dc65b8`。两仓 `integration/s3-next` 仍分别为 `6939de5b43200e6f19311804ebf1739b01f12ba5` 与 `51ed1d094c92fc52a910acda3e2a3421bc71c9e5`，**不能从 integration 直接假定 WP01/02 已合并**。
- WP01前驱：Gym `2904f3e58cd84fb383f9e5a0d2d2aa643fcb10d2`；Dreamer `347724e3207fca16907facab021302217f47cf6b`。

## 一、确认的真实生产者接口

| 维度 | 真实路径 / 函数 / Schema | 评估 |
|---|---|---|
| 总编排 | Gym `research/fg_s3_e_quality_corpus_cli.py::generate_small`, `verify_corpus`, `_publish_streams`, `_verify_publications` | 已有 CPU 生成及独立只读验证；无需 WP03 再生成 |
| 策略轨迹 | Gym `research/fg_s3_e_quality_corpus.py::behavior_specs`, `generate_behavior`, `l1_labeler`, `coverage` | P0, P1, P2_low, P2_mid, P3, P4；P2 两种概率是两条轨迹，共 6 条 |
| 原生 Replay | Gym `research/fg_replay.py::collect_trace`, `replay_fields`, `trace_manifest`, `iter_replay_rows`；`E_QUALITY_SCHEMA=FG_S3_E_REPLAY_MARKET_MINUTE_V2` | 读取 `.npz` Chunk，实际 `stepid` 为 20 字节 uint8（16 字节 chunk UUID + 4 字节 offset） |
| Sidecar | `fg_replay.py::iter_sidecar`, 原 trace `steps.jsonl.gz`；`teacher_sidecar/{policy_id}/steps.jsonl.gz` | 每个物理行附 `teacher:{protocol, fold, state_id, account_snapshot_hash, L1, L2,...}`；后验不得流入 Policy |
| 学生 Replay | `student_replay/{policy_id}/manifest.json`, `replay/*.npz` | `_publish_streams`复制字节相同 Chunk，同时故意剥离账户/老师 metadata；不能仅靠学生 manifest 恢复 teacher owner |
| 完整 trace | `policies/{policy_id}/manifest.json`, `steps.jsonl.gz`, `replay/` | 包含 policy/trace/账户来源，必须用于独立 lineage 联合认证 |
| Corpus 总清单 | `corpus_manifest.json`，`protocol=FG_S3_E_QUALITY_CORPUS_V2` | TRAIN、全局 course/source/teacher/obs/model 身份；`training_admitted=false` 不可误判为 Train GO |
| Observation | `FG_S3_E_OBSERVATION_V2`, `FG_S3_E_MACD5M_CAUSAL_V2` | 真正 E Encoder 输入含 `market_seq_e_shared[60,18]`、`market_context_15m[24,7]`、`market_macd_5m[4]`、`position_state[9]`、`action_mask[2]`；其余 action/reward/reset/stepid 是训练及核对用 |
| L1 | Gym `fg_s3_quality_teacher_v2.py::NativeQualityTeacher.query_state`；`fg_s3_quality_catalog_v2.py::validate_corpus_label` | 原生当前状态 fork 的立即 ADD 后果；score、event/touch/adverse 的有效位独立 |
| L2 | Gym `fg_s3_e_quality_strict.py::QualityStrictTeacher.query_values` 和 `QualityP4Policy` | 返回 `teacher_valid`, `strict_action`, `action_value_add`, `action_value_wait`, `value_rational_witness`, `tie`, `reason`, `suffix_boundary_identity`；资金域外严格 invalid |
| 真实 Quality TP | `FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1`; `fg_s3_quality_q_v2.py::build_quality_suffix` | 注意后缀数值内部仍以键 `'C1'` 命名 reward list，**语义以 native Quality Reward 和 objective identity 为准，不能以字符串键推断仍是旧 C1**；WP03需做原生等价测试 |
| 模型 | Dreamer `fg_s3_e_encoder.py`, `fg_s3_v2_agent.py::_S3V2Model.ext_space`, `training_loss_fn`, `quality_loss` | WP02 MACD 接入 Encoder；训练 loss 仍使用旧 `FIELDS` 和 `teacher_action/teacher_valid`，存在 WP03 必须完成的映射缺口 |
| 旧质量 Catalog | Gym `fg_s3_quality_catalog_v2.py::FIELDS`, `validate_label`, `attach_quality` | `validate_corpus_label` 新增在同文件；旧 `attach_quality` 仍依赖 legacy label/owner，不能原样附加 WP02 L1/L2 |
| 质量排序 | Dreamer `fg_s3_quality_v2.py::pair_identity`, `teaching_objective`, `active_branches` | `quality_owner uint32[8]`、相同 reset segment 才做 pair；默认 Soft 启发式的 loss 权限需显式关闭 |

## 二、接口级关键发现（修订 WP03 的必要理由）

### I-01 BLOCKER：WP02 新 Envelope 与旧 Agent ExtSpace 不相通

`validate_corpus_label(label)`消费 `label['L1']`、`label['L2']`，而 `attach_quality(batch, records)`进入的是旧 `validate_label(row['quality'])`。Dreamer `S3V2Agent.training_loss_fn`依赖扁平 `FIELDS`。WP03必须新增**同一合同内**的 V2→loss-ready projection，逐字段设值和 valid，不允许在不检查来源的情况下直接 `records` 填旧字段，也不要求 WP02 重新生成老师。

### I-02 BLOCKER：Owner 不能只由 `stepid` 推导

`stepid` 只证明物理 Chunk/offset，不等于金融账户状态。必须三方绑定：`student_replay/<P>` native row + `teacher_sidecar/<P>` posterior row + `policies/<P>` original trace/manifest。至少检查 policy id、chunk/offset、row ordinal、action/market cursor、account_snapshot_hash、L1 state_id、L2 funds domain、reward/obs/source/strict-teacher identity。**同一行情的不同账户绝不能共用老师动作**。

### I-03 BLOCKER：标签取自推进后的状态，不能错位

`fg_replay.collect_trace` 在 `capture` 中使用 `raw._get_obs()`/teacher 查询生成**已推进至当前观测的下一决策状态标签**；`decision_executed` 由下一次回调补全。Replay 本行 `action` 描述上一交易决策的执行反馈，并不自动等于本行 L2 最优动作。用真实 `stepid` 对齐未来可用于该 observation 的 actor 决策，必须验证 `decision_executed`、`is_first`/reset、chunk reset、`market_advance` 和下一个真实 step。首尾行不合法时 valid=false。

### I-04 BLOCKER：新 L2 `strict_action` ≠ L1 质量 ≠ 实际 action

L2 strict ADD/WAIT 需要同时双合法、现金证明及完整后缀。L1 即使验证立即 TP5 也不等于 strict ADD。持仓强制 WAIT、资金域外、未知结算边界均不能标 strict WAIT。行为轨迹 P1 中真实 ADD 不代表老师同意。

### I-05 BLOCKER：Q/Soft 目标语义与旧 Loss 默认不同

`fg_s3_quality_q_v2.UTILITY` 是以名义净 TP 利润与 Score 构成的独立效用，不等于本轮 `discounted_native_float32_Quality_TP_feedback`。现有 `quality_soft_lambda` 默认非零且 `teaching_objective`直接按 `quality_soft_valid` 训练，**不能只报告“不建议使用”**。WP03 要保证新 Reader 对未经授权 Soft 统一输出 `quality_soft_valid=false`、Q有效位 false，并测试 loss 权重；后续 WP06 默认也要设置 Soft/Q loss 权重为 0，任何启用需单独语义验证。

### I-06 HIGH：Pair owner 应定义为账户 episode 级，而非每行变化的账户快照

若把当前 `account_snapshot_hash`直接变成每个状态的 `quality_owner`，则即使同一账户，Pair也始终无法成立。应区分：**join key（每决策状态）**与**pair group identity（同账户、同真实 episode/reset、同教师目标）**。继承原 `pair_identity`同 owner、同 reset segment的语义；任何跨账户/跨后缀证明域/跨 session reset 的配对须拒绝。

### I-07 HIGH：Corpus G3 不是授权训练

上传 Git 归档包含代码及有限真实 TRAIN 的报告/收据引用，但没有包含完整 `real_small_04` 原生数据目录，因而本次**未独立重演**真实账户及后缀审计。Codex应在真实 worktree 先运行已实现的 `verify_corpus`，再以**实际数据文件**做 WP03 Reader END-TO-END，并提供独立签名/来源；Reader生产准入与 Train Admission 保持分离。

### I-08 HIGH：真实样本不足以证明全市场教学覆盖

WP02 报告：FG201 2021-12-08/09；六账户各 691 行，4146行；119笔实际交易（103 TP,12 SL,4 OTHER）；1551个 L1 有效立即 ADD 候选；P4 54笔 TP；93个资金域外 L2 invalid；真实右删失为0。仅支持有限工程准入，不能预测泛化训练质量。三分母为唯一市场/账户状态/优化器曝光。

### I-09 MEDIUM：性能边界不宜在 WP03 重做

现有小样生成+验证很耗时，WP03 Reader必须消除每批 `verify_corpus`/Oracle 计算，做一次授权预检→只读 hash 验证/索引→流式采样。不要修改金融 Env 或 WP02 并行实现来解决 WP03 读取速度问题。

## 三、实际验证范围与不确定性

- 已独立完成：两 Git commit/父子关系解析；抽取分支源码；关键函数、Schema、契约的静态比对；核心 4 个 Python 文件 AST 语法解析通过；报告和 receipt 内容核对。
- 根据归档报告（**未独立复测**）：最新冻结源码选定测试 `312 passed`，有限真实数据 G1/G2/G3 GO，官方训练 Gate 依旧拒绝。早期曾有一组路径配置导致 `9 setup errors`，报告通过正确 Dreamer worktree 单独复跑 13 个相关测试；不得改写成“从无失败”。
- NOT_RUN：真实完整 Corpus 重放、实际 JAX loss/grad、完整回归、GPU、TEST、全 18 合约、正式训练。

## 四、Review 结论与 GO

- **WP02→WP03 接口设计：GO**：功能和契约足以开始 WP03；不要求改造回旧 ABI。
- **WP03 立即使用旧 Catalog 接入：NO-GO**：先实施 V2 映射＋三方账户 join＋时序对齐；这是 WP03 本身的核心工作，并非要求 WP02 返工。
- **正式课程训练：NO-GO**：WP03 完成后仍需 WP04–WP07 的训练阶段和授权。

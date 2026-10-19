# FG-S3 WP06 → WP07 独立接口 Code Review（gy22 / d322）

日期：2026-10-19。性质：用户提供的双仓 Git 对象静态源码审查，结合仓库内原始 WP06 receipts 的核对；**不是重跑 GPU/真实 TRAIN 训练的验收报告**。

## 1. 被审查的代码身份和集成事实

| 仓库 | 归档内 `integration/s3-next` HEAD | WP06 最后运行代码提交 |
|---|---|---|
| Gym | `d5153268bcd33d5a4e54721767c8782f895283f9` | `53ffea2353618be99f35673ab8f4c6c544b7273d` |
| DreamerV3 | `e8c2688bc817662e5e6fa7e8cee62825bcf80b00` | `ffe6a831877ee91769f611dd16c374fb96941ebd` |

完整发布 SHA 已归档在 `docs/WP06_COMMIT_LOCK.json`。**两个 `integration/s3-next` 引用已经指向 WP06 发布提交**，不再需要把 WP06 重复 cherry-pick / merge 一次。归档主要提供 `.git` 对象，本次通过各自 `git archive integration/s3-next` 复原工作树进行审查；不能因此宣称用户本机原训练 worktree clean。

### WP06 已具备的工程链（代码与报告一致）

- 模型 `dreamerv3/fg_s3_e_quality_actor.py`：`STAGE='ACTOR_SUPERVISED'`、`GROUPS=('quality_actor/',)`、`distribution()` 复用原 `quality_actor`，原生 mask/unimix 只做一次；`actor_objective()` 接受 L2 `teacher_action/teacher_valid`，独立 Natural / auxiliary / strict-pair 计数，L1 quality_pair 仅 L2 一致才有效。
- 真实训练 `dreamerv3/fg_s3_e_quality_actor.py::train()`：`frozen_features` / 真正 K burn-in，`masked_joint_optimizer(...allowed_prefixes=GROUPS)`，使用前后 Ninjax state 恢复阻止非 owner 或非有限更改；`opt/updates` 与 batch consumption 分开计数。
- `embodied/run/fg_s3_e_quality_actor.py::ActorStreams`：由相同 Corpus 的两份 WP03 `EQualityCourseReader` 组成 Natural/Auxiliary，能保存/加载两份 Reader 的状态。**它不是跨 Corpus 调度器，也不是在线+历史 Replay 混合器。**
- `ActorRun(HeadsRun)`：直接复用 WP05 原更新循环、`FullCheckpoint`、耗用预算与冷恢复；F1 或 F2 的 non-opt 接续，fresh optimizer / RNG / counter。不应为 WP07 再发明第三条离线训练循环。
- `gym_trading_env/research/fg_s3_e_quality_actor.py`：`prepare` / `start_prepared` / `F3WindowAdapter` 与原生 V3 Join 对齐；已存在 `f3-prepare/probe/read-check/start/resume/evaluate` 入口。
- `gym_trading_env/research/fg_s3_e_quality_actor_checks.py`：Teacher-free Actor→Driver→原金融 Env→Replay→`reconcile_trace` 回放，足以成为 WP07 SHADOW/在线 Eval 的起点，但现在属于有限 CPU 工程评价，不是持续在线 Collector。

## 2. WP06 报告的证据边界

`docs/WP06_IMPLEMENTATION_REVIEW.md` 报告最终：WP06 96 passed / 共享 Guard 20 passed，全部零失败与跳过；另有 F2 cold 8 例。合成优化器两次实际更改 `quality_actor` 5 个参数 leaf 与其 10 个 slot leaf；其他 owner 状态不变。连续4更新 vs 2+冷恢复+2、首次 skip 后冷恢复均有完整状态与下一批对齐证据。有限原生合成 Env 32 分钟实际 entry fill=5、TP=2、SL=2；这不是独立真实 TRAIN 的盈利证据。

**NOT_RUN / NO_GO：**正式训练过的 F1/F2/F3 来源、完整 V3 Corpus 的实际 K32/T64/G3、真实多合约 Actor-only 交易表现、实际 GPU 参数更新。GPU 宿主曾出现 `cuInit CUDA_ERROR_UNKNOWN`，不能把 CPU PASS 说成 GPU PASS。实际小样 `f3-read-check` 只看到极少有效 WAIT，不足以代表正负监督覆盖。

## 3. WP07 前需处理的实际代码差异（按阻断优先级）

| ID | 真实源码证据 | 风险/必须修改的接口 |
|---|---|---|
| R1 | `dreamerv3/agent.py::Agent.loss` 和 `Agent._sample_imag_action` 都直接使用 `self.pol`；`fg_s3_v2_agent.py` 的 `RL_MODES=('c2','e')` | 想象RL若直接启用，会训练旧 `pol` 而实际执行 `quality_actor`；须实现 F4 严格共享的 `quality_actor` real/imag 策略选择器和梯度 owner 检查。 |
| R2 | `embodied/run/fg_s3_v2_train.py::LiveRun` 旧 Quality 仍使用旧 source/teacher/catalog/collector，`V2Collector` 的 Quality 采集逻辑依赖 query | 不能直接把旧 `LiveRun` 切成 `F4`；复用其 Driver、Collector、Replay 和 CP 原语，新增薄 F4 run coordinator。 |
| R3 | `dreamerv3/fg_s3_e_quality_course_contract.py::expected_stage_contract('ONLINE_ADAPT')` 同时要求 Prediction/Q/Decision/RL | SHADOW/在线BC不使用这些修正；不能伪造 `ONLINE_ADAPT` admitted。建议 SHADOW/ONLINE_SUPERVISED 明确认定为有唯一 Execution ID 的 `ACTOR_SUPERVISED` 在线续课；真正 RL F4 才使用新版本化 F4 执行合同与 Gate。 |
| R4 | `ActorStreams` 只有同 Corpus 的两 Reader，`FullCheckpoint` 外部 reader backend 只管其注册状态，`V2Collector` 有可变在线 replay | WP07需要只读历史 Corpus + 可变在线 Replay + 混合调度器 + teacher publication state + Env/Driver/actor policy RNG 的**原子可恢复组合状态**；不能只保存 Agent。 |
| R5 | `V2Collector(...teacher_enabled=False)` 约束 `beta=0`、无 `quality_catalog/outcome_catalog/beta_schedule`，但其父类 D0 仍有 Oracle/state/query 路径 | 必须用真实完整合成回放证明 teacher-free，不得只将 beta 设置为0便声称完全不访问后验教师或改变动作。 |
| R6 | WP03 Sidecar 使用已闭合后缀、`stepid` 与 post-step/Owner；WP06 F3 数据使用不可变教材 | 在线 Collector 新产生的状态不能立刻安全获得相同 L2：须明确待标注游标、可用完整 TRAIN 后缀、标签发布延迟、真实性与合法有效位；未证明的标签不补 WAIT。 |
| R7 | 当前 `quality_actor` F3 `distribution()` 使用真实 mask；想象态 mask 来自 `avail` 预测 | 想象预测掩码和真实 mask 可能不一致，尤其在持仓/账户状态；必须做合法性校准与 fallback 审计，不能假装真实交易状态被想象完整复制。 |
| R8 | `quality_actor` F3 只是合成有效参数更新；正式 F3 权重不存在，DATA-MULTI 独立 branch 也未显示已合并主线 | WP07可完整开发合成工程，不能凭 WP06 发布启动真实 F4；开发前复核 DATA-MULTI/EVAL-BASE 的新集成 HEAD，复用其正式 API 而不是复制。 |
| R9 | `RL` 的 `rew/con` 来源是 Quality TP WM，稀疏 SL、TP 的价格路径对 Q 值高度敏感 | 必须有多步 imagined vs real TP/SL/风险诊断和 RL numeric gate；WM默认冻结，避免在线误差把学生表征冲坏。 |

## 4. 代码级目标接线图

```text
F3 完成的 source_run/ckpt
       │ 参数级 F3→online 新Run，fresh run state
       ▼
quality_actor(policy in real Env, no teacher control)
       │
       ├── SHADOW: native Env/Driver → native Replay/trace → evaluation; 0 updates
       │
       ├── ONLINE_SUPERVISED: closed TRAIN after-the-fact L1/L2
       │           + historical EQualityCourseReader + new Online Reader
       │           → existing actor_objective/update owner → quality_actor only
       │
       └── ONLINE_RL_CONTROLLED (separate technical/formal gate)
                   → WM imagination + **same quality_actor distribution**
                   → val/slowval, real mask calibration, Quality TP Reward
                   → quality_actor + admitted value owner (WM frozen default)
```

所有模式学生观察只包含原 E+causal MACD、账户与合法 mask；教师结果、未来 suffix、Oracle、评价标签只在 TRAIN 监督 Sidecar，不能流入执行 policy。

## 5. 建议的集成验收

1. 一对纯合成 native TRAIN 合约路径，真实环境 100+分钟连续 SHADOW，无老师修改 action，完整 `reconcile_trace`。
2. F3→在线参数全键接续并新Run/optimizer，不冒充同Run resume。
3. 在线侧待标注→合法label发布→两路数据混合→仅 quality_actor 梯度更新；条件不足记为 unsupported，计数独立。
4. 旧 `pol` / `quality_actor` imagined sampling 分支差异回归：F4必须恰好使用后一者，非法 mask 不产生 ADD；RL off 价值网络逐键不变。
5. 连续N vs N1+全新解释器N2的输出、Env ledger、两路采样、全部 RNG/optimizer/log完全一致；模拟中断无重复或遗漏 label 发布。
6. WP06 / F1 / F2 / WP03 / V3 MACD 定向回归；`integration/s3-next` 完成后双仓成对记录。

## 6. Review 结论

**工程审查：WP06 模块与声明范围相符，可开始 WP07 开发。研究结论：仍 NO-GO。** 首先实现 teacher-free 在线账户闭环和在线监督；RL代码路径可以同 WP 实现，但最终启用必须在真正 `quality_actor` 梯度、Critic数值、世界模型结果、mask证据通过后。WP07 不应以旧 `LiveRun` 或`pol`开旧 RL Gate 的方式伪装交付。

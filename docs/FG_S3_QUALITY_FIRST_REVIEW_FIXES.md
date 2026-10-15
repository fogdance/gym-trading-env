# Quality-first：2026-10-16 审查修复与验收

本轮执行 [原始 Review](quality_first_code_review_20261016/S3_Quality_First_gy15_d315_Code_Design_Review_20261015.md)，继续在 `fg-s3-quality-v2-dev/{gym-trading-env,dreamerv3}` 开发。原 Tutor-CD、Tutor Full、B2 的源码工作树、Checkpoint、Replay、日志及旧收据保留。质量仍为 B(t,d)，MAE 单列；原 C1 Reward、市场分钟时钟及 Full 数学公式未重新设计。

本轮运行源码基线：Gym `dd0fe47a3bd43e33c10d6a71e16407efd6d4b50d` / Dreamer `d33985cb2989dcf2e3fe6dc06584ffb757011c3d`。211 个运行源码文件和 1 个独立诊断工具均与提交逐字节匹配；三份最终 prepared、真实 100 步 CPU Run 使用相同运行源码。后续仅补文档和原始收据，最终 Git HEAD 另在交付答复与 Run 的绑定收据报告。

## 已确认的原因与改动

| Finding | 源码证据及根因 | 本轮处理 | 状态 |
| --- | --- | --- | --- |
| F1 | 原 `fg_s3_quality_catalog_v2.require_prediction_admission` 仅要求单个有效样本和部分 horizon 改善；未完整约束风险支持、时间 fold、重复训练及事前阈值 | 移到独立 `fg_s3_quality_admission.py`，V2 gate 从封存记录重新计算全部 7H 的 NLL/Brier、TP/SL、低资金、官方日、校准、SL recall、Score 和二元窗口支持；每个留出合约×时间 fold×replay seed 必须齐全，并有独立 DEV。同 replica 的真实第 0 步模型作参考，固定模型/来源/预算，只准预定最终保存点，不允许事后挑 seed 或中间 Checkpoint | 工程合同已实现；预测研究 PENDING |
| F2 | 原 Q gate 信任 PASS、数量、short_run_verified；无法确认状态、见证、来源和实际复制的权重 | V2 Q gate 验证实际 TRAIN Catalog、完整账户快照、utility/reward/clock/code/budget、精确有理数值及完整原生转移见证；独立重跑每个采用的 Q 证明。检查两头真实参数改变和支持计数，逐留出合约/fold 复验 Q_C1、Q_V2 的 MSE 与优势准确率。SourceAdmission/transfer 绑定并复制实测 Q 权重，启动时重验全部证据 | 工程合同已实现；FULL_Q_PENDING |
| F3 | 新 Reward 严格候选原先最多 8 个均匀点加首个 TP/SL 对账 | 复用原 `verify_candidates` 的 128 个随机样本、原有退出/长路径/chunk 边界；补齐方向、官方跨日 d、B 分段、早期逆行窗口、同根/歧义/重试等风险分层。对严格最优选中路径使用原 Env 完整账户逐笔重放，逐反馈、逐结算、ADD 序列、现金及最终空仓全部核对 | 原生测试通过；真实合约实测见本轮收据 |
| F4 | 单笔高 B 分不等于 ADD 相对 WAIT 有优势，预测精度 gate 无法证明决策安全 | 增加独立决策 gate；固定同状态历史上的干预方向必须有原生 L3 相对动作证明；严格 WAIT（含平局）误 ADD、闭环 SL 率、账户回撤必须满足事前风险上界。网络与配置同时要求 `quality_decision_admitted`。允许只读诊断 delta 和原 Env 闭环，明确不计作已准入 Actor 成绩 | 工程合同已实现；DECISION_GATE_PENDING |
| F5 | 默认精确域只有最后 12 根 bar；普通合约起始区间不会有数值 Q | 保留默认 12/4096/2秒，不伪造 Q，不引入近似教师。新增实际合法空仓分母、落在精确域的潜在样本分母和候选成本；posthoc Q 取 Checkpoint 认证的原训练预算，避免后续显式扩大精确域时评价仍偷偷使用默认 12 | 覆盖不足仍 PENDING |
| F6 | 存在可调用的旧 S=B−1.5R，且新 Reward 和 Score 各实现一次 B | 删除旧函数及其旧 S 锚点测试，无兼容别名。`quality_tp_reward` 统一调用附件 `fg_s3_quality_score.tp_event_reward`；原 `timeboxed_tp` 不动，新增边界和 64 个固定随机值的一致性验证 | 已修复并验证 |
| F7 | 重复完整账户回放和固定 512MiB/worker 假设，缺真实成本依据 | 教师使用有界完整状态 LRU：先验来源/行情/规则/mask，再查 full export_state SHA；深拷贝，不跨教师/账户混用。多合约先测最大行数合约的实际峰值 RSS，再按可用内存及 CPU 配置并发，子进程 BLAS=1。保存候选持仓分布、独立持仓分钟总和、时间、RSS、日志字节和缓存计数 | 已测有限合约；18 合约/1M 性能未认证 |
| F8 | 30 万元只证明旧 L3 严格最优路径的资金，不是 L1/L2 合法性下界 | 保留 guard；低资金 L1/L2 合法评分与 L3 缺证分开统计 | 符合设计，未删除保护 |

F1–F4 属于审查确认的证据合同缺口；没有把它们描述成已经观测到的错误实盘交易。新风险上界与样本阈值没有附件给定数值，代码要求显式事前冻结，未根据现有 DEV smoke 结果编造生产默认值。

## 准入证据合同

所有文件经已有 `signed/read_manifest` 和 SHA256 校验。旧 V1 PASS 清单不兼容新 gate；不加包装器、不增加旧测试专用分支。已存在的 `fg_s3_reader_compat` 仅更新本次已审计 Reward 文件的精确哈希，仍是原历史来源权重准入，不用于放宽测试或 resume 身份。

`quality_admission_policy: <JSON 路径>` 必须在 prepare 前提供。该文件被复制到 prepared、绑定进模型来源，再在 Run 的初始 FullCheckpoint 前复制为不可变文件。其 SHA、训练合约、模型、每个 replica 的来源和最终 Env 预算都被复验。未提供 policy 的 HEADS_ONLY 只允许技术开发，不能事后补 policy 取得研究准入。

Policy 协议 `FG_S3_QUALITY_ADMISSION_POLICY_V2`，所有字段必须显式填写：

- `trained_contracts`、互斥的多个 `heldout_contracts`、多个独立 `replay_seeds`、事前选择的 `target_replay_seed`。
- `time_folds: [{id, windows: {<heldout 或 CZCE.FG409>: {start_day, end_day}}}]`：多个互不重叠的官方交易日区间；各合约可处于不同年份，不把自然日期误作共同样本窗口。
- `prediction`：`minimum_samples_per_horizon`、`minimum_score_samples`、`minimum_binary_samples`、`minimum_official_days`、`minimum_tp_per_horizon`、`minimum_sl_per_horizon`、`minimum_low_funds_per_horizon`、`maximum_event_ece`、`maximum_binary_ece`、`minimum_sl_recall`。
- `decision`：`minimum_changed_decisions`、`minimum_strict_add`、`minimum_strict_wait`、`minimum_closed_entries`、`maximum_false_add_rate_delta`、`maximum_sl_rate_delta`、`maximum_drawdown_delta`、`require_native_relative_value: true`、准确有序的 `component_sets`。风险增量上界不允许正数。
- `q`：`minimum_verified_states`、`minimum_low_funds_states`、`minimum_official_days`、`maximum_value_mse`、`minimum_advantage_accuracy`；两种 Q 均须满足。

测试中的 `fixture_policy()` 是小型合成门槛，**不是生产研究建议或批准**。现有 100 步 DEV smoke 已被观察，不能用于事后挑阈值或充当新的独立认证样本；后续研究必须预先冻结完整计划与未观察时间窗口。

三类 receipt 的公共字段为 `protocol/status`、Checkpoint/manifest SHA、policy SHA 及所引用证据文件的精确 SHA：

| Gate | 必需明细 |
| --- | --- |
| Prediction V2 | `comparisons: [{primary_scoring_dir, reference_scoring_dir, time_fold}]`、`evaluation_files`；参考必须是该 replica 自己真实的初始 Checkpoint。每条预测、动作、状态、权益与封存因果 tape 对账 |
| Decision V1 | `prediction_admission` 及 SHA、`components`、`comparisons: [{baseline_scoring_dir, candidate_scoring_dir, time_fold}]`、`evidence_files`；含 Q 时另有 `q_admission` 及 SHA。基线与候选必须评价同一个已认证预测模型、配方和 Q 权重 |
| Q V2 | `catalogs`、`holdout_evaluations: [{scoring_dir, time_fold}]`、`evidence_files`；每个数值监督均须拥有真实根账户和完整可重放见证；拒绝重复状态扩充支持 |

只读诊断复用统一 `evaluate --diagnostic-components outcome,score`。它使用模型前向中同一组有界、stop-gradient 的诊断 delta；各模型维护自己的因果 carry。`--diagnostic-q-admission <receipt>` 仅向私有评价模型加载已独立认证的 Q 参数，基线也使用该来源；不改训练 Checkpoint、不绕过决策准入。这样可先测 Q 干预效果，再决定能否用于 Actor，避免循环要求“先获得决策 PASS 才能评价 Q”。

Q 权重必须与被评价预测模型使用同一个 base NN。原 Full 默认会更新 Backbone；若 Q 来源改变了这些特征，必须在该完整来源上重新进行 HEADS_ONLY 预测训练及全部预测准入，再做 Q 私有诊断，不能把旧特征的已认证预测头与新 Q 硬拼。已有 completed S3-V2 `warm_start` 入口承担这次受控来源迁移，不新增训练入口，也不修改原 Full 的默认梯度边界。该正向研究链路尚无合格真实来源可实测。

```bash
python -B tools/fg_s3_v2_experiment.py evaluate --prepared <新准备> --run-dir <新Run> --diagnostic-components outcome,score --eval-steps 100 --output-dir <新诊断目录>
python -B tools/fg_s3_v2_experiment.py score-predictions --eval-dir <封存推理目录> --output-dir <新补评目录>
python -B tools/fg_s3_v2_experiment.py compare-predictions --eval-dir <补评> --reference-dir <参考补评> --time-fold <事前fold> --output-dir <新比较目录>
python -B tools/fg_s3_quality_v2.py validate-admission --kind prediction --receipt <receipt.json>
python -B tools/fg_s3_quality_v2.py validate-admission --kind decision --receipt <receipt.json>
python -B tools/fg_s3_quality_v2.py validate-admission --kind q --receipt <receipt.json>
```

`validate-admission` 会完整重算证据；不把填写 PASS 字段视为成功，不启动训练。诊断输出标明 `diagnostic_intervention_only`，与未干预 Actor-only 结果分开。

## 测试与实际证据

本轮原始日志与 XML 保留在 `docs/receipts/quality_review_*`，最终运行副本、具体指标及源码绑定见 [本轮收据索引](receipts/quality_review_delivery/README.md)。旧 150 项交付收据仍在 `receipts/quality_first_delivery/`，不能替代本轮验收。

| 验收 | 实际结果 |
| --- | --- |
| 数学、原生教师、Catalog、旧 Env/Oracle、来源边界、组合和共用 Tracker | 108 项通过；含 gate 合同测试 |
| 真实 JAX CPU Student | 8 项通过；四头真实更新、Full 合成监督梯度/stop-gradient、冻结、精确恢复、已评价辅助/Q 参数复制 |
| 当前共享合同回归 | 34 项通过；未新增旧测试兼容代码 |
| prepare→100→resume200 对照连续200 | 实际运行；最终收据见索引，覆盖 C0 policy 冻结、真实头参数改变、事前最终保存点及 posthoc Q 预算归属 |

最终合计 151 项。真实 FG201/FG205 分别抽验 254/248 个 cursor、覆盖 66/74 个风险分层；选中的 3255/1708 笔严格路径交易均经原 Env 完整账户逐笔回放。最终双合约候选阶段约 163.54/123.46 秒、峰值 RSS 608,239,616/493,621,248 字节。250 步 Catalog 的 12 个合法空仓候选中，5 个低资金可评分但 L3 认证为 0，真实数值 Q 为 0，默认 12 根精确域潜在样本为 0/12。

真实 CPU 第二次更新四个辅助头均有非零梯度与参数变化，另外 89 组非优化器参数摘要不变。DEV/FG401 的新封存补评仍未证明预测改善。只读 Outcome+Score 诊断 100 步全 WAIT；独立重放基线 8 个、诊断 100 个空仓状态均与原生严格查询和完整最终账户对齐。但基线 1 笔 ENTRY 未退出、诊断 0 笔 ENTRY，连最小成交闭环支持都不满足，安全支持检查明确拒绝。不能把这一段“避免一次 ADD”当成有效策略改善。

不重复累计目标重跑。首次测试命令曾使用不存在的预测测试文件名而返回 exit 4，错误日志保留为 `quality_review_native_final.log`；随后使用真实测试路径运行。开发中补充来源检查后会使之前的 prepared 失效，使用独立版本的新实验重新准备，保留旧目录及真实 Git/dirty 状态，不改旧收据伪装为最终执行。

F3 的证明范围明确为原索引/区间支配证明、分层原生抽验、以及**选中严格路径的逐笔原生重放**；`all_candidates_exhaustively_replayed: false`。没有声称所有 31K 个单独候选都逐笔原生重放。F7 的缓存只复用完全相同账户状态，不把相同 cursor 的不同资金/持仓路径混为一个结果。

## 未认证项与下一步

真实预测能力、校准、跨合约/时间/seed 的完整准入、相对 WAIT 的决策改善、真实数值 Q 覆盖、已获准来源的 Full 模型迁移和含 Q 私有诊断的正向研究链路，仍需实际合格数据和模型验证。没有合格来源时明确拒绝，不伪造 PASS 来测试生产准入。

默认 12 根精确后缀域保持不变。有限合约的候选耗时、持仓分布、峰值 RSS、250 步 Catalog 和 100 步 CPU 更新不能外推 18 合约、GPU 或 1M 性能。首次 GPU 1K 只完成 CPU prepare；GPU、10K、1M 未启动，TEST 未解封。具体对照和启动审阅材料见 [启动计划](FG_S3_QUALITY_FIRST_RESEARCH_LAUNCH_PLAN.md)。

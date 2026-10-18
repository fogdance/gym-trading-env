# E+Quality Master Roadmap V10：WP06/F3 工程交付

2026-10-19。继承已归档的官方 Review V8 和 DATA 审计 V9；历史报告保持原样。本版只汇总 WP06 的实际工程结果，不启动真实训练/考试，不改变已发布 Corpus 的课程数学或身份。

## 当前结论与公共基线

公共整合 G0 已完成：Gym `7ea95563166bf7fa03c6ce836e7484053cac0b96`，见 [基线收据](WP06_BASELINE_INTEGRATION_RESULT.json)；Dreamer `0e39dfd498f9146e10f966cfe118a24b20e5143e`。最终发布清单记录 WP06 A–F 的成对 SHA、独立集成实测版本及最终 `integration/s3-next` HEAD。

WP06 为 **ENGINEERING CODE_GO**，真实数据支持 **LIMITED**，**FORMAL_TRAIN / RESEARCH NO_GO**。完整工程报告：[WP06_IMPLEMENTATION_REVIEW.md](WP06_IMPLEMENTATION_REVIEW.md)。没有真实已训练 F1/F2/F3，不得称学生已学会交易。

## WP01–WP10

| WP | 当前实际状态 | 剩余/下游条件 |
| --- | --- | --- |
| 01 课程/权限 | 已有 V2 课程数学和显式 Draft/正式机制保留；F3 执行权限为课程上界的 quality_actor 子集，不重签 Corpus | 正式来源、权限和同版本恢复继续强校验 |
| 02 E+MACD/教材 | P0–P4、L1/L2 和原生 Quality TP Producer 已在共同基线；MACD V3，不使用旧 Tutor 1M 或 C1 Oracle 替代 P4 | 跨合约教材充分性仍需 DATA 工作流 |
| 03 Join/Reader | 唯一 V3 Join + Owner/stepid + NATURAL/STRATIFIED/MIXED + RNG；WP06 加逐字段 canonical post-step 监督核对 | V2 派生索引必须重建，不兼容、不改号冒充 |
| 04 F1 WM | 原完整模型/六组 WM 工程已集成；本轮合成 F1 作为可验证 handoff 来源，非正式受训基础 | 真实来源及正式 F1 更新 NOT_RUN |
| 05 F2 四头 | 已集成；完整非优化器接续与冻结逻辑复用，本轮再次做原 F2 冷恢复回归 | 真实 F2/Prediction Admission 仍独立 NO_GO |
| 06 F3 Actor | A–F 实现：F1/F2→F3、仅 quality_actor、原 WM K 步 burn-in、L2 主监督、Natural anchor/辅助 Reader、strict/一致性 pair、纯学生推理、原 FullCheckpoint 冷恢复、guarded CLI | 真实完整 Corpus/K32T64/G3、真实 F1/F2/F3 和研究效果 NOT_RUN；不得自动开放 Gate |
| 07 F4 接续/RL | 未实施；现有 F1/F2/F3 handoff 不是在线/RL 授权 | 后续阶段单独定义/审查，复用原 Trainer/Gate |
| 08 独立考试 | WP06 工程首版复用原 Driver/Env/reconciler，无 Teacher argmax；完整账户重播通过 | 真正隔离 DEV/TEST、成本/MAE/TP时间/稳定性考试 NOT_RUN，需独立授权 |
| 09 多合约/消融 | 合成 Natural/Balanced/strict/一致性 pair 接口及负例齐全；仅有限单合约真实 TRAIN 只读 | 多 Corpus/多合约、规模/分布充分性和真实消融仍 NOT_RUN |
| 10 冻结/研究结论 | 工程收据和版本清单归档；真实 TRAIN 更新 0，TEST 行情读取 0 | 研究版冻结与 GO 依赖上述真实验收，不以合成 PASS 代替 |

## 当前教材与真实只读边界

V9 历史 B01 审计有效保留，但其原 `/data/s3_corpus/early_train_20261018_cpu20_v2/batches/B01_FG809` 在本次现场已经不存在；已知两个 full-main 目录也未发布 corpus_manifest。本轮不读取未发布半成品、不把旧报告当当前 G3。

现存 FG809 DEV 小样通过 F3 fresh CPU read-check：K1/T1、8 次有效 WAIT 暴露、2 个独立目标 cursor、7 个账户行；P1/P2_mid 不支持，状态 NOT_RUN。原参数和 optimizer 不变；Source 为 INTEGRITY_CHECKED_ONLY / NOT_ADMITTED。完整 K32/T64 已发布真实教材验收仍 NOT_RUN/NO_GO。

## 下一步顺序

1. 提供当前可验证的已完整发布 V3 Corpus，保留来源及缺题型/独立市场分母证据。
2. 形成真实 F1/F2/F3 的训练准备和预算，用户另行授权才执行；工程 fixture 不能升级为正式来源。
3. WP07/08/09 按上述边界推进，所有 Prediction/Q/Decision/RL/正式训练 Guard 保留。

本版不启动 GPU 长训练、真实 TRAIN 长更新、TEST 考试或 18 合约批量生产。

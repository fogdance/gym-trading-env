# WP06 progress

- Independent paired integration worktrees created; Gym synchronized to the newer DATA B01 audit commit c1679a7. Dreamer remains 0e39dfd. Existing WP04/WP05 merge preserved.
- G0 V3-only source/test commit: 7ea9556. Explicit receipt/database version errors; no legacy fallback. Directed 4 PASS, exit 0.
- G0 contract/Reader/MACD/market-clock: 69 PASS, exit 0. Raw records: receipts/wp05/G0_WP06_contract_reader.*.
- G0 current real TRAIN read: exit 2, corpus_manifest missing at the earlier B01 path. Source relocation requested; not counted as PASS. Earlier planning-time read remains historical evidence only.
- G0 F1/F2 cold restore/handoff and required shared Guards: running. WP06 A–F not started; public integration not advanced.

G0 complete: 126 distinct CPU cases PASS, exit 0; real TRAIN DEV 4-minute read PASS/LIMITED, zero updates. Full earlier B01 corpus missing: read exit 2 / NOT_RUN, not formal admission. Baseline published Gym 7ea95563166bf7fa03c6ce836e7484053cac0b96 / Dreamer 0e39dfd498f9146e10f966cfe118a24b20e5143e. A implementation and directed tests underway.

06.A: explicit F3 protocol/config/complete native model and immutable course execution subset implemented. Stage mixing, legacy CONTROLLED_FULL, Q/Soft/prediction/fusion rejected. A_contract: 32 PASS, exit 0 (includes existing F2 contract regression). F3 update route deliberately fail-closed until 06.C; no real TRAIN update.
# 06.B 参数接续

- 实测 `B_transfer`：25 passed，退出码 0，442.44 秒；原始命令、配对 HEAD、源码摘要和输出见 `receipts/wp06/B_transfer.*`。
- F1、F2 均验证完整 non-opt 逐键复制、fresh optimizer/counters、policy mirror、源文件不变；包括缺键、shape、dtype、NaN、错误 stage、未完成来源、旧 Observation、损坏 bundle、冻结参数修改、F2 ledger 缺失及伪造权限拒绝。
- `B_F1_handoff.json`、`B_F2_handoff.json` 为实际合成工程接续收据。两者均 `FOUNDATION_NOT_TRAINED`，不能授权真实训练。
- 自审：保留原 F1→F2 默认接口；来源先校验后修改新 Agent，完整冻结证明包含参数专属 optimizer slots。文档字符串修正不改变已测运行源码行为。

# 06.C 原生 Actor 更新

- `C_updates_retry`：13 passed，exit 0，169.44 秒（12 个 F3 case + 1 个原 F2 allowlist 回归）。原始 `C_updates` 的字段名测试错误为 1 failed / 12 passed，已保留。
- `C_actor_delta.json`：直接使用原 Reader 未改写的 L2 标签完成两次真实原优化器更新；quality_actor 参数和专属 slots 变化，其余 owner=None 的完整参数及 slots 逐字节相同。
- 验证 K=4 / T=8，prefix 和 burn-in 标签不进入 loss；Teacher 变化不改变特征/logits，真实历史变化影响 target 表征。有效目标 stepid 等于原窗口 `[:,5:]`。
- 验证非有限候选完整回滚、已有动量的全无效 batch 不推进 optimizer、冻结非零动量/负零、错误 Owner/标签/原生 stepid、Q/Soft、单动作 mask/reset/terminal 拒绝或屏蔽。
- 当前仅 CPU 合成工程验收，真实 TRAIN 更新数为 0。D 的双 Reader 协调仍为独立待提交工作。

# 06.D Teacher、采样与同一原循环

- `D_run`：16 passed / exit 0 / 220.47 秒。包含两条原 WP03 Reader（NATURAL anchor + MIXED auxiliary）、原 Agent.stream/prefetch 和同一 HeadsRun.update/run；两次实际 Actor 更新，以及全无效批次消费 4 次后明确拒绝继续、optimizer 保持 0。
- `D_boundary`：14 passed / exit 0 / 32.66 秒。包括 NATURAL/STRATIFIED/MIXED RNG 精确恢复、辅助失败双流回滚、仅 burn-in 有标签时拒绝 target support、独立自然/辅助/平衡分母、标签翻转梯度、L1 冲突过滤、跨 Owner/reset/batch 零 Pair。
- 原始失败保留：`D_losses` 1 failed（测试把 10 分差计入 >=15），`D_loss_sampling` 5 failed（测试账户每 9 步 reset 不支持 12 步窗口）；修正 fixture 后 `D_loss_sampling_retry` 13 passed。生产准入与数学阈值未放宽。
- `D_actor_run.json` 从 D11 实际 FullCheckpoint 读取 counters/双 Reader 状态和参数 SHA；单元测试结果文件名曾与 harness 同名，已改名避免覆盖。原始 Checkpoint 未改写。
- 自审修正统计含义：无辅助流时 balanced loss 可复用主流，但 `auxiliary_*` 暴露计数为 0；新增 `balanced_count` 单独记录实际 loss 分母。

# 06.E 纯 Actor 与可学性

- `E_evaluation_retry`：4 passed / exit 0 / 198.28 秒；原始失败 `E_evaluation` 为 1 failed / 3 passed，保留。
- 修正已定位的诊断 PRNG 差异：原 Agent._seeds 使用 `[config.seed, seed_counter]` 生成 key，诊断此前直接把 202 当 key。F3 read_only_forward 现在复用原 Agent._seeds；新增第一步原 opt/loss 与诊断 Loss 一致断言。
- 512 次固定 CPU 合成更新，32 ADD / 32 WAIT 的显式合成特征分类任务，Loss 0.693907→0.687870，准确率 29/64→35/64，恒 WAIT 为 32/64；TP=25/TN=10，冻结 state 完全相同。仅证明工程可学，不是认证 L2 或真实市场效果。
- F3 推理直接调用同一 quality_actor/mask/unimix，predict() 被禁止时仍通过；三个 fusion delta 为 0，同特征训练/推理分布完全相同。
- 原 ActorPolicy/Driver/金融 Env/Replay/Quality TP Reward 的 32 市场分钟完整重播通过，36 行含 3 次账户恢复。原生 fills/TP/SL 与动作提案分开报告；最后提案未执行。详见 `E_native_actor.json`。

# 06.F Checkpoint、CLI 与交付

- `F_cold_cli` 20 passed / exit 0 / 384.20 秒：普通 4 vs 2+冷恢复+2，首次 skip 后冷恢复，双 Reader RNG/Prefetch/全部模型/optimizer/日志逐字节一致；七类身份/损坏负例、权限、prepared/read-only CLI 均覆盖。
- `F_shared` 20 passed / exit 0；`F_temporal_legacy` 121 passed / exit 0；原 `F_f2_cold_regression` 8 passed / exit 0（包括新解释器 exact restore）。不是全项目通过。
- 实际有限真实 TRAIN fresh read-check exit 0，K1/T1、8 个 WAIT 暴露、2 个独立目标 cursor、7 个账户行；所有参数相同，零更新。完整 B01 缺失，K32/T64/Native G3 NOT_RUN。
- GPU 两次尝试分别在沙箱/经批准的宿主构造前失败，宿主 cuInit CUDA_ERROR_UNKNOWN；GPU 参数更新 NOT_RUN，未改系统。
- 反向审查发现 stop_at 小于总预算仍标 COMPLETE，回归 `F_partial_before` exit 1；仅修正结果为 PARTIAL，`F_partial_after` 1 passed / exit 0，未改准入。
- 下一步封版 A–F，独立集成树运行当前版本回归和 cold restore，成功后更新成对 integration/s3-next；原始收据和 Review 都在 docs。


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


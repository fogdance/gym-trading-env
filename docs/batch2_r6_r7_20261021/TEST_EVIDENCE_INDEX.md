# 测试证据索引

最终冻结双仓共 330 次执行、330 个不重复用例通过，FAIL/ERROR/SKIP 均为 0。仅限下列集合，不是全项目测试通过。

| 最终分组 | PASS | FAIL | SKIP | 退出码 | 秒 | 原始命令/环境/源身份 |
|---|---:|---:|---:|---:|---:|---|
| FINAL_V5_f1 | 22 | 0 | 0 | 0 | 132.57 | [JSON](receipts/FINAL_V5_f1.json) / [log](receipts/FINAL_V5_f1.log) / [JUnit](receipts/FINAL_V5_f1.xml) |
| FINAL_V5_f2 | 8 | 0 | 0 | 0 | 138.30 | [JSON](receipts/FINAL_V5_f2.json) / [log](receipts/FINAL_V5_f2.log) / [JUnit](receipts/FINAL_V5_f2.xml) |
| FINAL_V5_f3 | 11 | 0 | 0 | 0 | 184.55 | [JSON](receipts/FINAL_V5_f3.json) / [log](receipts/FINAL_V5_f3.log) / [JUnit](receipts/FINAL_V5_f3.xml) |
| FINAL_V5_handoff | 39 | 0 | 0 | 0 | 300.08 | [JSON](receipts/FINAL_V5_handoff.json) / [log](receipts/FINAL_V5_handoff.log) / [JUnit](receipts/FINAL_V5_handoff.xml) |
| FINAL_V5_readers | 152 | 0 | 0 | 0 | 285.41 | [JSON](receipts/FINAL_V5_readers.json) / [log](receipts/FINAL_V5_readers.log) / [JUnit](receipts/FINAL_V5_readers.xml) |
| FINAL_V5_shared | 56 | 0 | 0 | 0 | 149.96 | [JSON](receipts/FINAL_V5_shared.json) / [log](receipts/FINAL_V5_shared.log) / [JUnit](receipts/FINAL_V5_shared.xml) |
| FINAL_V5_versions | 42 | 0 | 0 | 0 | 576.94 | [JSON](receipts/FINAL_V5_versions.json) / [log](receipts/FINAL_V5_versions.log) / [JUnit](receipts/FINAL_V5_versions.xml) |

## 研发与前序证据

以下每行绑定其真实双仓 SHA 和 dirty 状态，不能把旧版本的 PASS 自动计入最终版本。失败轮次保留，修复后通过的最终组在上表。完整逐用例结果见各 JSON/JUnit。

| 轮次 | PASS | FAIL | ERROR | SKIP | 退出码 | 原始收据 |
|---|---:|---:|---:|---:|---:|---|
| FINAL_V2_f1 | 22 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V2_f1.json) / [log](receipts/FINAL_V2_f1.log) |
| FINAL_V2_f2 | 8 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V2_f2.json) / [log](receipts/FINAL_V2_f2.log) |
| FINAL_V2_f3 | 11 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V2_f3.json) / [log](receipts/FINAL_V2_f3.log) |
| FINAL_V2_readers | 152 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V2_readers.json) / [log](receipts/FINAL_V2_readers.log) |
| FINAL_V2_shared | 56 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V2_shared.json) / [log](receipts/FINAL_V2_shared.log) |
| FINAL_V2_versions | 38 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V2_versions.json) / [log](receipts/FINAL_V2_versions.log) |
| FINAL_V3_f1 | 22 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V3_f1.json) / [log](receipts/FINAL_V3_f1.log) |
| FINAL_V3_f2 | 8 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V3_f2.json) / [log](receipts/FINAL_V3_f2.log) |
| FINAL_V3_f3 | 11 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V3_f3.json) / [log](receipts/FINAL_V3_f3.log) |
| FINAL_V3_handoff | 39 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V3_handoff.json) / [log](receipts/FINAL_V3_handoff.log) |
| FINAL_V3_readers | 152 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V3_readers.json) / [log](receipts/FINAL_V3_readers.log) |
| FINAL_V3_shared | 56 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V3_shared.json) / [log](receipts/FINAL_V3_shared.log) |
| FINAL_V3_versions | 39 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V3_versions.json) / [log](receipts/FINAL_V3_versions.log) |
| FINAL_V4_f1 | 22 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V4_f1.json) / [log](receipts/FINAL_V4_f1.log) |
| FINAL_V4_f2 | 8 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V4_f2.json) / [log](receipts/FINAL_V4_f2.log) |
| FINAL_V4_f3 | 11 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V4_f3.json) / [log](receipts/FINAL_V4_f3.log) |
| FINAL_V4_handoff | 39 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V4_handoff.json) / [log](receipts/FINAL_V4_handoff.log) |
| FINAL_V4_readers | 152 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V4_readers.json) / [log](receipts/FINAL_V4_readers.log) |
| FINAL_V4_shared | 56 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V4_shared.json) / [log](receipts/FINAL_V4_shared.log) |
| FINAL_V4_versions | 41 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_V4_versions.json) / [log](receipts/FINAL_V4_versions.log) |
| FINAL_f1 | 22 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_f1.json) / [log](receipts/FINAL_f1.log) |
| FINAL_f2 | 8 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_f2.json) / [log](receipts/FINAL_f2.log) |
| FINAL_f2_retained | 1 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_f2_retained.json) / [log](receipts/FINAL_f2_retained.log) |
| FINAL_f3 | 11 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_f3.json) / [log](receipts/FINAL_f3.log) |
| FINAL_readers | 152 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_readers.json) / [log](receipts/FINAL_readers.log) |
| FINAL_shared | 56 | 0 | 0 | 0 | 0 | [JSON](receipts/FINAL_shared.json) / [log](receipts/FINAL_shared.log) |
| MERGE_VALIDATION_V3 | 67 | 0 | 0 | 0 | 0 | [JSON](receipts/MERGE_VALIDATION_V3.json) / [log](receipts/MERGE_VALIDATION_V3.log) |
| MERGE_VALIDATION_V4 | 69 | 0 | 0 | 0 | 0 | [JSON](receipts/MERGE_VALIDATION_V4.json) / [log](receipts/MERGE_VALIDATION_V4.log) |
| R6_after | 34 | 0 | 0 | 0 | 0 | [JSON](receipts/R6_after.json) / [log](receipts/R6_after.log) |
| R6_before | 19 | 15 | 0 | 0 | 1 | [JSON](receipts/R6_before.json) / [log](receipts/R6_before.log) |
| R6_first_batch_and_shared | 83 | 0 | 0 | 0 | 0 | [JSON](receipts/R6_first_batch_and_shared.json) / [log](receipts/R6_first_batch_and_shared.log) |
| R6_transfer_real_cpu | 73 | 0 | 0 | 0 | 0 | [JSON](receipts/R6_transfer_real_cpu.json) / [log](receipts/R6_transfer_real_cpu.log) |
| R7_actual_run_assembly | 3 | 0 | 0 | 0 | 0 | [JSON](receipts/R7_actual_run_assembly.json) / [log](receipts/R7_actual_run_assembly.log) |
| R7_basis_binding_after | 2 | 0 | 0 | 0 | 0 | [JSON](receipts/R7_basis_binding_after.json) / [log](receipts/R7_basis_binding_after.log) |
| R7_basis_binding_before | 0 | 1 | 0 | 0 | 1 | [JSON](receipts/R7_basis_binding_before.json) / [log](receipts/R7_basis_binding_before.log) |
| R7_cold_and_faults | 10 | 0 | 0 | 0 | 0 | [JSON](receipts/R7_cold_and_faults.json) / [log](receipts/R7_cold_and_faults.log) |
| R7_cold_smoke | 0 | 1 | 0 | 0 | 1 | [JSON](receipts/R7_cold_smoke.json) / [log](receipts/R7_cold_smoke.log) |
| R7_cold_smoke_recheck | 1 | 0 | 0 | 0 | 0 | [JSON](receipts/R7_cold_smoke_recheck.json) / [log](receipts/R7_cold_smoke_recheck.log) |
| R7_guard_recheck | 5 | 0 | 0 | 0 | 0 | [JSON](receipts/R7_guard_recheck.json) / [log](receipts/R7_guard_recheck.log) |
| R7_hot_load_before | 0 | 1 | 0 | 0 | 1 | [JSON](receipts/R7_hot_load_before.json) / [log](receipts/R7_hot_load_before.log) |
| R7_reader_order_before | 0 | 1 | 0 | 0 | 1 | [JSON](receipts/R7_reader_order_before.json) / [log](receipts/R7_reader_order_before.log) |
| R7_transactions | 64 | 1 | 0 | 0 | 1 | [JSON](receipts/R7_transactions.json) / [log](receipts/R7_transactions.log) |
| R7_version_counter_before | 0 | 1 | 0 | 0 | 1 | [JSON](receipts/R7_version_counter_before.json) / [log](receipts/R7_version_counter_before.log) |
| R7_versions_first | 25 | 0 | 0 | 0 | 0 | [JSON](receipts/R7_versions_first.json) / [log](receipts/R7_versions_first.log) |
| R7b_all_reader_adapters | 82 | 2 | 0 | 0 | 1 | [JSON](receipts/R7b_all_reader_adapters.json) / [log](receipts/R7b_all_reader_adapters.log) |
| R7b_diagnostics_cwd_recheck | 2 | 0 | 0 | 0 | 0 | [JSON](receipts/R7b_diagnostics_cwd_recheck.json) / [log](receipts/R7b_diagnostics_cwd_recheck.log) |
| R7v2_f1 | 13 | 1 | 8 | 0 | 1 | [JSON](receipts/R7v2_f1.json) / [log](receipts/R7v2_f1.log) |
| R7v2_f2 | 6 | 2 | 0 | 0 | 1 | [JSON](receipts/R7v2_f2.json) / [log](receipts/R7v2_f2.log) |
| R7v2_f3 | 9 | 2 | 0 | 0 | 1 | [JSON](receipts/R7v2_f3.json) / [log](receipts/R7v2_f3.log) |
| R7v2_versions | 37 | 0 | 0 | 0 | 0 | [JSON](receipts/R7v2_versions.json) / [log](receipts/R7v2_versions.log) |

运行器保留绝对 argv、cwd、CPU affinity、CPU-only 环境和双仓完整 SHA。移机复现时只替换 worktree / 新输出目录；不得复用已有 pytest basetemp，因为 pytest 会清空该测试专用目录。正式 Corpus/Run 目录不得用作 basetemp。

R6 来源负例的 15 个原始失败、Guard 传播、诊断 cwd 缺失、独立 Run session 比较合同、覆盖基础摘要以及重复恢复队列的失败均单独保留；未通过跳过、xfail 或放宽金融/准入/严格 CP 消除。正式 TRAIN/G3/GPU/长期训练为 NOT_RUN，TEST 为 NOT_READ。

## 集成目录复验

`MERGE_VALIDATION_V5`：70 PASS、0 FAIL/ERROR/SKIP，exit 0，实际 214.69 秒。原始 [JSON](receipts/MERGE_VALIDATION_V5.json)、[log](receipts/MERGE_VALIDATION_V5.log)、[JUnit](receipts/MERGE_VALIDATION_V5.xml)。用例均属于上列 330 项，本轮总验收执行数为 400，不重复通过用例仍为 330。

代码合并 SHA、精确 tree 继承与 docs-only 交付的对应关系见 BATCH2_BASELINE_AND_INTEGRATION.md 及工作区 baseline-lock.json。

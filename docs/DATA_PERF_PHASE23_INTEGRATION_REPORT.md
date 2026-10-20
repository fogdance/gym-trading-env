# DATA-PERF Phase 2 / 3 集成记录

用户于 2026-10-20 明确要求先将优化合并到 `integration/s3-next`。已完成合并与合并后的定向验证。

- Gym 来源交付：`8eb8b5b126c00887368238e7a47e5784b17c5217`。
- Gym 集成合并提交：`2f00607039855236b5e3e1329a2ef2aea0c14542`；双亲为原目标 e328cf73e0e284e506e549a35743458029e353a6 与来源 8eb8b5b126c00887368238e7a47e5784b17c5217，自动合并，无目标分支冲突。
- Dreamer：`b7a693b3c7b0b45eeecdec954818be42d40e67b6`，无新增修改，无需重复合并。
- 合并后的 181 个 Gym 生产运行文件与冻结验收版本 b10b764 逐字节 SHA 相同；合并树与来源交付树完全相同。两份既有证据身份锁共 127 + 69 个文件全部核对通过。
- 集成工作树 `test_market_cache_seed.py` + `test_data_perf_phase2_guards.py` 共 **13 PASS / 0 FAIL / 0 SKIP**，15.775s，包含真实 spawn、延迟行情可见性、独立账户/冷恢复、完整 G3 门槛与原算法 Golden。运行源码前后哈希不变；未运行优化器。
- 原始合并记录、运行文件 SHA、准确测试命令、XML、日志与新身份锁：`receipts/data_perf_phase23_integration_20261020/`。最后的集成记录提交只补文档与收据，不修改运行源码；真实 G3 证据保留原冻结 Producer 身份。

合并前只读核对：Gym 目标 `e328cf73e0e284e506e549a35743458029e353a6`、Dreamer 目标 `b7a693b3c7b0b45eeecdec954818be42d40e67b6`，均位于 `fg-s3-next-integration` 且工作树 clean。

来源分支 `feat/data-perf-phase3-startup-20261020`，包含 Phase 2 的 `13b997f`→`2dd4528`、验收文档 `887386d`，以及 Phase 3 纯行情 seed 复用 `b10b764d2d1dc9ffb24c6450cef67ba0a378ec53`。此前仅在该独立分支合入 Phase 2 文档，不曾提前合并到集成分支。

本次集成包含：恢复逐查询来源检查和完整 G3 标签计数/trace 绑定；私有不可变历史与精确公共 JSON SHA；有序标签增量传输；Driver/原生对账复用；跨 spawn 的纯行情快照复用。交易金融、TP5/SL30、Quality TP Reward、严格动作/平局/资金证明及现有六行为规则保持原合同。Dreamer 没有新增运行源码修改。

原始定向、真实小样、性能、Checkpoint/公共冷恢复与独立验收证据分别在：

- `DATA_PERF_PHASE2_RESULT.md`、`receipts/data_perf_phase2_20261020/`。
- `DATA_PERF_PHASE3_REVIEW.md`、`receipts/data_perf_phase3_20261020/`。
- 新版冻结 Producer 命令 `DATA_PERF_PHASE3_COMMANDS.md`；其认证绑定 b10b764 / b7a693b，不把旧教材身份重签为集成 HEAD。

Phase 2 有 113 项定向与 106 项选定共享/Reader/多 Corpus 回归通过。Phase 3 两组各 28 项、最终选定共享集合 35 项通过（集合存在交叠，不能合计为唯一用例数）；107 条真实查询、51 次原生交易、39 个严格有效结果匹配；真实两日全部 4,146 条金融/教师语义对照通过，独立 G3 校验全部标签，Reader 10 个视图精确冷恢复通过。没有全项目通过的声明。

真实完整两日 CPU 流程：Phase 1 907.330s，Phase 2 464.315s，Phase 3 399.823s；本轮相对 Phase 2 缩短 13.89%，相对 Phase 1 为 2.269 倍。更旧版本的完整独立 G3 未完成，不声称有其全流程加速对照。

性能源码冻结到 b10b764；随后诊断工具仅补充“配置池容量 / 实际初始化进程数”两个计数说明字段，生产运行源码没有变化。合并后已核对运行源码的相对文件集合及逐字节 SHA，并在集成工作树执行最小接线 smoke；未重复运行相同源码的完整 G3。

本次不修改 TRAIN01 冻结 Producer/教材，不修改 WP04/WP06/WP07 工作树，不运行 GPU、真实优化器更新或正式 F1/F2/F3 训练，不启动 18 合约生产。正式全主力期与所有训练效果：NOT_RUN。

后续行为规则、参数及组合的配置功能单独实施。本次不会把改变数据分布当作逻辑等价性能优化，也不预设六行为比其他组合训练效果更好。

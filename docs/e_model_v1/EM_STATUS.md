# EM01–EM18 实际验收状态

本表限定 M0–M3。PASS_CPU_SYNTHETIC 表示真实 CPU 神经模型/原生组件运行，输入为明确的合成 E 数组；不等于生产 E 已准备或已解锁。正式 E 仍 waiting_e。

冻结 CPU 验收代码：Gym `da7e59fb228126102e17e264db3751656fa2f500`；Dreamer `f30d8da2523eddad53aeca1540afe1c77d47b248`，两仓执行时干净。后续文档提交不改运行代码。

| ID | 状态 | 证据与边界 |
|---|---|---|
| EM01 | PASS | before/dependencies/resources/protection；从本机实际 integration 起点建立双仓，记录期间外部integration前进 |
| EM02 | PASS | contracts_final：八组合、旧metadata、所有公开拒绝路径；describe仍waiting_e |
| EM03 | PASS_CPU_SYNTHETIC | Encoder B/BT、initial/entry_space/truncate、真实 enc三子树 |
| EM04 | PASS_CPU_SYNTHETIC | context改变tokens、posterior、同RNG Actor概率；不是交易效果或真实价格坐标证明 |
| EM05 | PASS_CPU_SYNTHETIC | shape/dtype/finite/mask、padding/all invalid、未来字段拒绝；JIT非法输入异常 |
| EM06 | PASS_CPU_SYNTHETIC | Agent policy/WM/prefix；prefix与原dyn.observe逐值对齐，历史E快照与真实prevact |
| EM07 | PASS_CPU_SYNTHETIC | WM三分支有限非零梯度与实际更新，BC Encoder梯度0；原教学函数AST不变 |
| EM08 | PASS_CPU_SYNTHETIC | opt/enc全key/policy keys/sync/Checkpoint真实哈希一致 |
| EM09 | PASS_CPU_SYNTHETIC | 仅shared1m/position重建，8个原WM loss键finite；改变旧market_seq不改变loss |
| EM10 | PASS_CPU_SYNTHETIC | 原Env/Driver/DAgger/nativeReplay；污染历史快照/执行动作拒绝；provider合约/时刻修复前失败后通过 |
| EM11 | PASS_CPU_SYNTHETIC | E+A2真实5次联合更新，Actor变化、Critic冻结 |
| EM12 | PASS_CPU_SYNTHETIC | E+Tutor真实5次更新、原Balanced/Pair与诊断、Critic冻结 |
| EM13 | PASS_CPU_SYNTHETIC | E+B2 head/adapter/schedule真实初始化/梯度/更新，2步warmup后跨ramp；未注入计数 |
| EM14 | CPU合同PASS；正式BLOCKED_BY_E_DATA_AND_REVIEW | 原生SourceAdmission私有核心、签名、same seed重现、零计数/Opt/Gate、错配拒绝；CPU Prepared使用独立协议，不称正式prepare PASS |
| EM15 | PASS_CPU_SYNTHETIC | 三教学各自原生FullCheckpoint：2次更新保存，Env70→75、继续3次更新；模型/Opt/policy/Replay/Driver/Env/Sidecar/Gate/RNG/计数及指标一致；生产resume未运行 |
| EM16 | 受影响回归PASS；BASELINE_FAIL_3保留 | 26模型/FullCheckpoint/context +60公共合同；旧SourceTransfer 2PASS/3FAIL，integration对照同三项缺S3_MARKET_MINUTE |
| EM17 | BLOCKED_BY_E_DATA_REVIEW_AND_REAL_INTEGRATION | 独立数据技术报告已21PASS，Review仍pending；本分支未做真实E consumer/三教学生产准备联调 |
| EM18 | PENDING_USER_AUTHORIZATION | GPU smoke/正式1M/DEV Actor-only/W&B online均NOT_RUN |

## 收据入口

- [全部测试汇总](evidence/test_summary.json)，保留初始失败和修复后结果，不将不同组重复测试相加。
- [冻结50项合同](evidence/contracts_final.json)、[5项真实模型](evidence/models_final.json)、[三教学完整恢复](evidence/checkpoint_final.json)。同名 `.log/.xml/.resource` 为stdout/stderr、JUnit和资源。
- [Checkpoint实际路径与文件SHA](evidence/checkpoint_index.json)，小模型完整二进制保留在对应隔离 `artifacts/fg_s3_v2/e_model_cpu/checkpoint_final/`，未上传。
- [E-Data联调阻断](E_DATA_BLOCKERS.md)。

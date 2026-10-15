# S3 E 正式训练准入执行计划（2026-10-15）

本轮任务来源：用户提供的 `FG_S3_E_Data_Model_Reviewed_Synthesis_NO_GO_20261015.zip`。目标为两仓 `feat/s3-e-model`，完成真实 TRAIN CPU、完整恢复、生产配置与 size50m GPU 1K 验收；正式 1M 只准备，等待用户验收，不启动。结果和原始失败均保留，未验证不得写 PASS。

## 已核对的基线

- Gym：`05141eefb438fd0a6038f32419e209f5fc11149f`，干净。
- Dreamer：`f30d8da2523eddad53aeca1540afe1c77d47b248`，干净。
- ZIP SHA256：`5438b719f6c0c27b6e78519c0417f66e6384336666bef65f57d11e7a3bb47225`，包内六个文件哈希全部通过。
- Bundle 校验通过，Gym 快进至 `5507ace1b923c184a2b4ed3d541f1755e4511f5c`（保留 Data→Model merge `289c6d7`），Dreamer 快进至 `03ea881b117196e7482c05ec99a2b7ee64c243b7`。
- 应用前已记录全部工作树、54 个受保护工作树 30,537 个已跟踪文件的 SHA；未 reset、未覆盖现有修改。
- 07:04 主机只读检查：Tutor 正式训练 PID 728523、在线模拟 PID 45614 正在运行；GPU 5583/16380 MiB，内存可用 12.97 GB，项目盘剩余 26 GiB。不得停止这些进程或修改其源码、配置、产物。

## 执行顺序与验收门槛

1. **包与源码 Review**：读取三份包内说明，验证 Bundle DAG 和双仓差异；审查 Provider、SourceAdmission、Collector 与训练生命周期。保留 Reward、Oracle、TP5/SL30、Action Mask、Gate、盲测 TEST；数据数学核复用 Data 分支。
2. **P0-1 真实 TRAIN CPU**：认证完整 collection、原 TRAIN pool 与 prepared subset；真实原 Env→逐分钟 E Provider→Driver/DAgger→Replay→Join→模型。A2、Tutor-CD、B2 各至少 65 帧，覆盖轮转、不同合约、1024 截断、reset，比较原 market_seq 与 stepid/账户状态。复跑 Data 原 Env 依赖测试，不接受 SKIP 代替 PASS。
3. **P0-2 来源与恢复**：三模式 direct_init、Gate CLOSED、模型/优化器/EMA/策略参数同步、数据/源码/配置/包版本身份；CPU 联合更新验证三分支梯度和权重变化；原 FullCheckpoint 保存、重载及相同下一批次/参数/调度/Sidecar/Replay/Env/RNG 等价。复核历史 BASELINE_FAIL_3 并修正过时 fixture 或明确排除依据，不放宽旧 B2 warm-start 认证。
4. **生产配置审查**：使用 canonical size50m，显式 seed、原市场分钟时钟与完整安全参数；为验收提供受限、可审计的诊断生命周期，保持公开正式 E 准入 fail-closed。不能伪造 ready 或通过环境变量绕过守卫。
5. **P0-4 GPU 1K 与恢复**：真实 TRAIN、完整 size50m，不缩网络；记录实际 Env 步与 model updates、完整配置、显存峰值、耗时/吞吐、动作合法性、ADD/WAIT、BC 有效更新、策略 pADD、三分支梯度/更新和 FullCheckpoint 恢复。每模式证据独立；失败原样留档并修复后重验。产物使用全新隔离路径，W&B 不上传例行诊断。
6. **P0-3 最终准入 Review**：只有对应模式全部 P0 通过后才修改唯一 Composition 能力映射与正式 SourceAdmission。Tutor-CD+B2 保持 NOT_IMPLEMENTED。若任一必要证据不足，该模式继续 waiting_e，结论 NO-GO。
7. **冻结交付**：分别提交两仓；记录双仓 SHA、diff、来源身份、CPU/GPU/checkpoint 原始收据、EM01–EM18 与 P0 状态表、受保护工作树对照。GO 时仅准备独立 1M 配置/快照和待用户验收命令，绝不启动正式 1M。

## 原始证据位置

本轮暂存及大文件：`../admission_20261015/`（相对 Gym 根目录为 `../admission_20261015/`）；可审查报告与小型原始收据汇入本目录。完整 checkpoint 与 Replay 留在本工作树 `artifacts/fg_s3_v2/` 的独立目录，按哈希索引，不复制历史产物。

状态：进行中。实施中发现的实际失败、修复依据和门槛变化在 `REPORT.md` 中记录。

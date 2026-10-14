# S3-NEXT E-Model V1 开发与 CPU 验收计划

用户已确认本计划，并选择 `defaults + size50m` 作为 fresh-init 默认网络规模。任务书完整来源：`~/Downloads/S3_NEXT_E_Model_V1_Composition_Integrated_Codex_Task_Reviewed_20261014.md`。

## 基线与边界

- Gym：`cc6ba78f3a2016eb280135f7730d0a1f68d4512e`；Dreamer：`eb63de710fd693fad7ac76f797f3d9f39e331b98`。均为当前干净的 `integration/s3-next`。
- Gym 比任务书参考 SHA 多一个 Tutor 配置/文档提交；模型代码未变。
- 双仓工作树：`fg-s3-e-model-dev/{gym-trading-env,dreamerv3}`，分支均为 `feat/s3-e-model`。不合并 integration/main。
- 本轮只实施 M0–M3，CPU 模型及合成 E 输入验收；正式 E 组合继续 `waiting_e`，Tutor+B2 继续 NOT_IMPLEMENTED。E-Data 状态在实施开始和交付时重新核实；真实联调与公开解锁留待 Review。
- 不启动 GPU/正式 E 训练，不改变当前 B2/Tutor、Archive 或 W&B 原 Run。

## 模型与观察链路

- 新 `S3EMultiscaleEncoder` 保持原 Encoder 的 initial/entry_space/truncate/__call__ 契约以及模块名 `enc`。
- 1m：E shared `[60,18]` + position `[9]` + action_mask `[2]`，复用三层简单 MLP；默认宽度 512。
- 15m：`[24,7] → flatten168 → Linear128 → GELU`；invalid 行前6列清零，全无效 embedding 为零。
- 融合：Concat → 常规初始化 Linear512；全部权重从头初始化，不要求旧 Encoder 权重兼容。
- 同一 enc 真正用于 policy、world_model、recompute_prefix；所有分支进入 opt、policy 参数同步及 Checkpoint。
- E Encoder/Decoder 使用显式白名单。Decoder 仅重建 shared1m 与 position，不重建旧 market_seq、15m 或 Mask；原 rew/con/avail 和教学 Loss 保持。
- 独立版本化 E_FIELDS/Obs/Replay/Decoder schema；不修改旧 FIELDS/S3_FIELDS。E Replay 保留旧 market_seq 供审计，保存每步历史 E 快照及实际执行动作。
- 共享 Collector/DAgger/observation hash/teacher join 接受明确 schema；旧默认路径不变。教师、Oracle、Outcome 和未来字段不得进入 Actor Obs。
- E provider 只传认证后的观察数组和独立 provenance；模型不重新开发 E-Data 聚合/归一化。CPU 使用确定性合成 provider。

## Composition 与来源认证

- 只沿用 encoder/teaching 双轴，支持内部构造 E+A2/E+Tutor/E+B2；不新增 E teacher。
- 公开 prepare/start/resume/evaluate、来源与模型工厂继续 require_ready；内部共享核心供 CPU 测试使用，保留真实 waiting_e，不伪造 ready 或新增解锁开关。
- fresh-init 最小配置：`source_kind: direct_init`，`model_init: {seed: 0, presets: [size50m]}`。使用原 YAML reader/elements.Config，冻结完全解析配置。
- direct_init 不解析旧 checkpoint 路径，并拒绝非空 source_checkpoint。E 拒绝旧模型 warm-start。
- 仅明确 E+B2/direct_init/可信 seed/config/schema/数据绑定条件允许创建原 OutcomeHead/Adapter/schedule；旧 B2 warm-start 严格协议保持。
- 初始化 optimizer、Replay、Gate、RNG 与计数均为新 Run；不继承旧参数来源哈希。
- Source/Prepared/Run/Checkpoint 身份绑定 Encoder/Decoder、Obs/Replay schema、E-Data manifest、Teacher/Outcome/TRAIN、Reward/clock、seed、source kind、Composition、双仓 SHA 和代码哈希。
- 受控 CPU 签名收据不等于正式 E Prepared。正式生命周期继续阻断；不扩大 ReaderCompat 或旧来源白名单。

## CPU 验收

显式禁用 CUDA，JAX CPU，沿用 Dreamer Python。小模型为 float32、1m三层宽度16、fusion16、context128、B1、T64+prefix1。保留原 optimizer 预热；B2 使用记录清楚的短测试调度并执行真实更新。

| ID | 验收 |
|---|---|
| EM01 | 双仓/Worktree/dirty/依赖/资源/E-Data/保护清单 |
| EM02 | 八组合、旧 metadata、全部公开守卫 |
| EM03 | Encoder 接口、参数与 B/BT shape |
| EM04 | 两支路/fusion/RSSM/Actor 对合法输入变化的响应 |
| EM05 | padding/mask/shape/dtype/finite/未来字段拒绝 |
| EM06 | policy/WM/recompute_prefix、历史快照与实际 prevact |
| EM07 | 分支有限非零 WM 梯度与参数更新，BC 隔离 |
| EM08 | optimizer/policy keys/sync/Checkpoint 参数完整 |
| EM09 | Decoder 白名单与 finite rew/con/avail |
| EM10 | 合成 E provider 到原生 Driver/DAgger/Replay 链路 |
| EM11 | E+A2 Actor/BC、Critic冻结 |
| EM12 | E+Tutor Balanced BC/Pair及诊断 |
| EM13 | E+B2 fresh-init、head/adapter梯度与调度 |
| EM14 | fresh-init/签名/seed/错配拒绝；正式 Prepared 单列阻断 |
| EM15 | 三教学连续 vs 中断，完整模型/Opt/Replay/Env/Driver/RNG/调度/计数和下一更新一致 |
| EM16 | Legacy A2/Tutor/B2/C2、来源与守卫回归 |
| EM17 | 等待认证 E-Data 与真实双仓联调/Review |
| EM18 | PENDING_USER_AUTHORIZATION：GPU/DEV Actor-only/W&B online |

先新接口和模型，再受影响的 Composition、DAgger、Source、B2/Tutor、FullCheckpoint、时钟和本地 Tracker 测试。已有三项 SourceTransfer 缺时钟 fixture 失败保留基线对照；不放宽认证、不冒充 PASS。日志保留所有初始失败、修复和未运行项目。

## 交付与暂停

自审并修复 Diff/合同/参数/标签/身份/恢复问题，分别提交双仓。docs 交付接口、实施报告、EM01–EM18状态表及联调阻断；外置 validation 保存完整命令、实际 SHA/dirty、stdout/stderr/JUnit、参数/梯度/policy hashes 和完整 Checkpoint 证据。外置版本锁记录最终 SHA，避免自引用。

M0–M3 完成且有 CPU 证据后暂停等待 Review；满足条件时状态为 `E_MODEL_CODE_READY_WAITING_E_DATA_AND_GPU_REVIEW`。正式 E 未解锁、EM17/EM18 未完成不标 PASS。

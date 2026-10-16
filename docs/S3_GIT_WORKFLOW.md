# S3 Git / Worktree 工作流

项目根目录：`/home/v/Documents/work/research`。Gym 与 DreamerV3 是两个独立仓库；源码必须使用同一工作区内的兄弟目录。

Quality-first 的最新合并复核与实际验收见 [集成 Review](S3_NEXT_QUALITY_FIRST_INTEGRATION_REVIEW_20261016.md)。

## 分支职责

- `main`：经过验证的稳定版本。禁止直接开发；发布前单独 Review 和授权，不自动合并、移动或重置。
- `integration/s3-next`：下一代 S3 唯一公共集成分支，保留 A2 Oracle BC、Tutor-CD、B2-Full、市场分钟、DAgger/Replay/FullCheckpoint、现有 W&B 与 Research Archive V1。
- `feat/*`：短期功能开发，从最近验收通过的 integration 双仓版本建立，完成回归和 Review 后普通 merge 回 integration；不 rebase 历史实验分支或强推。
- 旧 `feat/fg-s3-*` 分支保留为历史来源，不再作为下一项新功能的默认起点。Archive V1、已准入 E 与 Quality-first 已按各自证据进入基线；当前精确能力、限制及双仓版本以工作区版本锁与最新集成 Review 为准。
- `feat/s3-e-data`、`feat/s3-e-model`、`feat/s3-tutor-b2` 仅在对应任务实际启动时创建，不预建空分支或无用 Worktree。

## Worktree 与训练冻结

- 公共集成目录：`fg-s3-next-integration/{gym-trading-env,dreamerv3}`。
- Worktree 用于功能或源码版本隔离，不按每个模型或超参自动复制目录。创建前检查全部 refs、占用路径、HEAD 和 dirty 状态。
- 已存在分支只能安全复用；被其他 Worktree 占用、目录冲突或 dirty 时先处理事实，不强制覆盖。禁止本流程使用 `reset --hard`、分支强制更新、rebase 历史实验或删除 Worktree。
- 正式训练固定双仓完整 SHA、不可变 annotated Tag、配置/数据/Oracle/Reward 身份以及准备收据。相同 Tag 名在两个仓库分别指向各自 SHA，版本对应由独立 Manifest 明确记录；不强制改写旧 Tag。
- 正式训练使用 detached 冻结兄弟 Worktree；训练期间不修改、不合并、不重置。后续研发在功能或集成 Worktree 进行。
- 新算法或配置变化创建新 Run；同 Run 的恢复沿用原源码、准备快照与完整 Checkpoint，不改历史收据绕过身份检查。
- 当前 B2-Full 的 `fg-s3-b2-full-train` 与其正式产物受保护。本次建立集成分支不更新它们。

## 入口、验收与 W&B

- A2、Tutor、B2 共用 `tools/fg_s3_v2_experiment.py` 和 `configs/experiments/fg_s3_v2/`；三种已有模式分别运行，不将本次集成解释为联合教学。
- F0 新实验使用 `FG_S3_V2_COMPOSITION_CONFIG_V1` 和 `composition.encoder/teaching`，原 mode 配置继续兼容。当前保留 Legacy+A2/Tutor/B2、已准入 E+A2/Tutor/B2 及 Legacy+Quality。E+Quality 与 Tutor+B2 在准备/启动前拒绝；Quality 的预测、Q 和决策安全准入独立，工程合并不代表 Full 研究 GO。实现和集成证据分别见 `S3_F0_COMPOSITION_IMPLEMENTATION.md` 与 `S3_NEXT_F0_INTEGRATION_REPORT.md`。
- W&B 复用 `tools/fg_s3_wandb.py` / `research_tracking`，Project 固定 `fg-single3-research`；通过 `launch` 包装原训练命令，SDK 在独立环境，Trainer 不依赖网络。
- 研究归档复用 `tools/s3_archive.py` / `s3_archive`。`inspect` 只读；`archive` 默认 dry-run；L1/L2/L3 按真实验证结果授予。`verify` CLI 会更新归档自身的版本化元数据，校验受保护的历史归档时使用只读 `s3_archive.verification.evidence`。GPU 验证和 W&B 发布分别显式执行。
- 代理默认 `http://127.0.0.1:10808`，仅作用于 Tracker。正式指标以已完成 Checkpoint 为边界，实时数据标记 provisional；断网后保留 offline 数据，使用现有 sync 恢复。
- 功能验收优先最小 CPU 数值/模型、配置、原账户/Replay/恢复、W&B Mock/Offline 测试。GPU 或正式训练另行授权，不抢占正在运行的实验。
- 回归失败必须记录并修复直接相关问题；依赖缺失、SKIP 和 NOT_RUN 不能算 PASS。受控 CPU 测试不代表生产行情或策略效果验收。
- 测试与历史迁移要求随根目录 `AGENTS.md` 版本管理：历史测试固定两端完整 SHA，语义升级同步更新迁移边界；共享 Gate、SourceAdmission、正式 guard 及直接受影响的时钟／Replay／恢复检查必须纳入实际测试清单。旧 Gate 断言失效的根因与20项共享回归证据见兄弟 Dreamer 的 `docs/S3_NEXT_GATE_AUDIT_ROOT_CAUSE_20261016.md`。
- integration 验收通过后才可考虑发布到 main，发布不是日常功能合并的一部分。

## 双仓版本 Manifest 与清理

- 验收后的版本锁位于集成工作区根目录 `baseline-lock.json`，在两个仓库之外；提交完成后写入最终 SHA，避免交叉写 SHA 形成循环提交。
- 必须记录双仓路径/完整 SHA/clean 状态、基线名、起点、功能能力、W&B SDK 与源码版本、测试命令/结果/证据哈希以及 NOT_RUN 项。
- 任一仓库源码改变后，旧锁仍描述旧验收版本；重新验收并生成新的对应记录，不能把旧 PASS 直接沿用到新代码。
- 功能生命周期结束后，先验证提交及必要工件已包含/保存，再单独申请清理分支或 Worktree。本次不清理任何旧目录或分支。

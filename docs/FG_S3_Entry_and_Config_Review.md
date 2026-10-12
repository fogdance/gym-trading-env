# S3-A / S3-B / S3-C 入口与配置核对

## Summary

2026-10-12 按实际代码核对。S3 已有统一入口
`tools/fg_s3_experiment.py`；S3-C 接入这个管理层，新增 `s3c` 分派。
此前将参考规则里的 `fg_single3_experiment.py` 当成本任务入口，是名称误读。
该脚本实际负责另一条离线预训练 → 在线训练流程，不作为 S3-C 启动器。

实验 YAML 目前集中在 `configs/experiments/fg_single3/`。这里的目录名称
不能代替算法身份；S3 由 YAML 的 schema、trainer、mode 与准备收据区分。
优先扩展已有 S3 管理层及该配置集合，无需仅因名称重新复制启动器。

## Evidence observed

核对发生在已创建的双仓库开发 worktree：

| 仓库 | 工作目录 | 核对时 HEAD |
| --- | --- | --- |
| Gym | `/home/v/Documents/work/research/fg-s3-dev/gym-trading-env` | `1e59c27` |
| Dreamer | `/home/v/Documents/work/research/fg-s3-dev/dreamerv3` | `8de631a` |

核对前 Gym tracked diff 为空。Dreamer 的公共 source/teacher 文件有开发中修改，
S3-C Agent、Stream、runner 和测试尚未提交。它们尚未接入统一 CLI，也没有
已交付的 S3-C 实验 YAML；不能将已有代码草稿当成完整交付。

| 层次 | 实际入口或文件 | 已有职责 |
| --- | --- | --- |
| S3 管理 CLI | `tools/fg_s3_experiment.py` | `plan / prepare / start / resume / audit`；内部 `_execute` worker |
| S3 管理实现 | `src/gym_trading_env/research/fg_s3_experiment.py` | YAML 验证、路径与输入准入、不可变快照、排他锁、子进程、训练器分派、恢复审计 |
| S3-A 标签 | `tools/fg_s3a.py` | CPU `labels-build / labels-audit`，不负责联合训练 |
| S3-B 证据 | `tools/fg_s3b.py` | CPU values、weights、global normalization 构建与审计 |
| S3-A 原计算 CLI | Dreamer `scripts/fg_s3a.py` | features、train、audit、eval、CPU smoke；调用原 Actor-only 训练器 |
| S3-B 原计算 CLI | Dreamer `scripts/fg_s3b.py` | train、admission-audit、checkpoint-audit；调用原加权 Actor-only 训练器 |
| 共享数据 CLI | `tools/fg_s3_arch.py` | 完整 TeacherCatalog 构建与审计 |
| 共享来源 CLI | Dreamer `scripts/fg_s3_arch.py` | SourceAdmission、冻结源和真实参数迁移 CPU smoke |
| 其他流程 | `tools/fg_single3_experiment.py` | 原离线预训练、在线接续及评估管理，使用另一种准备协议 |

S3 管理层的 `TRAINERS`、`execute_phase` 和 `start` 已证明以下对应关系：

| 集中配置 | trainer / mode | 监督与优化器 |
| --- | --- | --- |
| `S3A_ORACLE_BC.yaml` | `s3a / S3A_FULL` | 原等权 BC；独立 BC optimizer |
| `S3A_FULL.yaml` | `s3b / S3A_FULL` | 完整合法集合的等权公平对照；独立 BC optimizer |
| `S3A_MATCHED.yaml` | `s3b / S3A_MATCHED` | 严格证书子集的等权公平对照；独立 BC optimizer |
| `S3B_CERTIFIED.yaml` | `s3b / S3B_CERTIFIED` | 同一证书子集的冻结优势权重；独立 BC optimizer |
| 待新增 S3-C1 配置 | `s3c / S3C1` | 原 RL + 等权 Oracle BC；只使用原联合 optimizer |

S3-A/B 当前准备状态为 `READY_FOR_FEATURES` 或 `READY_FOR_BC`。
启动路径是生成/复用冻结特征，再进入原 BC。这个阶段协议不能直接用于 S3-C：
S3-C 在每次当前 RSSM posterior `repfeat` 上计算监督，使用共享 Catalog 与
SourceAdmission，不依赖静态 feature cache 进行 BC，也不创建 BC optimizer。

Dreamer `configs/fg_s3b/{S3A_FULL,S3A_MATCHED,S3B_CERTIFIED}.yaml` 是原
S3-B CLI 的低层训练配置。Gym `configs/experiments/fg_s3b/certified_v1.yaml`
是 CPU 价值证据/权重配置。两者均不是新的实验管理快照来源。

## Root cause

管理规范中的示例名和实际项目入口被混用。代码证据明确表明，S3 管理层与
Single3 预训练管理层是两套不同流程；配置目录名相同不能据此合并训练流程。

另有两项实际问题须在接入时处理：

1. 四份 S3 集中配置仍指定不存在的旧 `dreamer_root`：
   `/home/v/Documents/work/dreamerv3-dev`；公共模板指向其他实验的
   `/home/v/Documents/work/dreamerv3/dreamerv3/configs.yaml`，不属于当前双 worktree。
2. 管理 CLI 和实现的实验/快照 YAML 使用 `yaml.safe_load`；只有公共模型模板
   使用 `dreamerv3.configuration.read_config_yaml`。这尚未达到用户要求的统一读取。

四份正式配置的 `step_000000062500` 来源目录也实际不存在。配置能通过 `plan`
解析，不表示来源准入通过；不得自动改用随机初始化或 CPU 诊断 Checkpoint。

## Changes made

本次只纠正实施方向并记录核对结果，没有修改训练器或冻结产物。
后续实现依据如下：

1. 扩展 `tools/fg_s3_experiment.py` 及其管理实现，复用已有锁、子进程、
   准备快照和 attempt 收据，不新增每个实验专用的训练启动脚本。
2. 在现有集中配置集合增加 S3-C1 YAML，显式指定来源、seed、lambda、
   梯度更新预算和独立输出；不替用户选正式训练来源或训练顺序。
3. 代码及模板默认定位当前开发 worktree 的兄弟 Dreamer 仓库。
   TRAIN/Oracle 只读复用用户建立的 `research/fg-data`，不导入其原仓库代码。
4. YAML 统一由 canonical reader 读取；网络 presets 与覆盖项使用
   `elements.Config` 和官方 Flags 流程解析，保存完整有效配置与来源配置。
5. S3-C 的 prepare 在 CPU 上绑定 Catalog、SourceAdmission、配置、源码及输入
   身份。start 建立新 run；resume 恢复本 run 的 FullCheckpoint、Replay、
   Stream、Gate、优化器和计数，不再次迁移模型来源。
6. A/B 保留原监督、特征缓存和独立 BC optimizer。C 只消费公共迁移结果，
   继承原 joint optimizer；`after_bc_s3a` 的独立 BC optimizer 不得迁入。
7. 日志路径沿用实际已约定的 `/data/logdir/fg_single3/`，每个 S3 实验独立。
   已冻结的有效配置、收据和 Checkpoint 不改写，修正配置后生成新准备快照。

该接入方式调整任务书第 6 节的建议 CLI 文件名，保留其来源准入、联合监督和
恢复语义。任务书明确说明具体参数名称可按实际代码确定；后续按真实入口记录命令。

## Verification performed

对四份集中配置实际执行了下列只读命令：

```bash
cd /home/v/Documents/work/research/fg-s3-dev/gym-trading-env
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
$PY tools/fg_s3_experiment.py plan --config S3A_ORACLE_BC
$PY tools/fg_s3_experiment.py plan --config S3A_FULL
$PY tools/fg_s3_experiment.py plan --config S3A_MATCHED
$PY tools/fg_s3_experiment.py plan --config S3B_CERTIFIED
```

均退出 0，实际返回 `CONFIG_ONLY_NOT_ADMITTED`、`formal_gpu_training=NOT_RUN`。
已保存完整 argv、环境、stdout、配置/代码哈希、双仓库 HEAD 与路径存在性检查：
[核对收据](../artifacts/fg_s3_arch_v1/receipts/entry_config_review_20261012.json)。

| 项目 | 本次核对结果 |
| --- | --- |
| 既有 S3 CLI 与四份配置解析 | PASS，4 份；仅配置解析 |
| 四份配置的源码定位 | FAIL，旧 Dreamer 根不存在，模板不在当前代码对中 |
| 四份配置指定的正式来源存在性 | FAIL，62500 节点不存在 |
| 来源准入、模型训练、FullCheckpoint 恢复回归 | NOT_RUN，本次只核对入口 |
| 正式 GPU / DEV / 封存 TEST | NOT_RUN |

## Risks / not verified

本报告没有对 S3-C 草稿宣布训练或恢复验收通过，也没有把四次 `plan` 当成
生产准入。配置和解析器的修正尚待实施，完成后必须跑现有管理层回归及 S3-C
真实 CPU 接线/恢复测试，再生成新的不可变准备快照。

原始 research 两仓库 HEAD 仍为 Gym `d374e93`、Dreamer `bb1d123`，均无
tracked diff；原有 untracked 文件保留。未修改 `.gitignore`、旧数据或正在运行的实验。

## Next recommended step

按上述实际入口继续 S3-C 集成。先统一 S3 配置解析和代码定位，再完成 C 分派、
新准备协议、FullCheckpoint 恢复和真实 CPU 验证；公共协议若需要调整，先完成
共享架构回归与收据，再继续依赖该接口的 C 验收。


## Final delivery update

This document's initial inventory is historical. C now uses the existing S3
entry with canonical configuration, public teacher/source services and full
restore. The final report and requirement review are in
FG_S3C1_Implementation_and_Verification.md and
FG_S3_Shared_Architecture_and_S3C1_Final_Review.md.

The user subsequently selected S3-A from the live online180k node; broader A
admission is explicitly versioned and retains the original BC math. Final
source/config/commits/CPU preparation/start commands are recorded in
FG_S3A_Online_180K_Startup.md. No formal training was started by Codex.

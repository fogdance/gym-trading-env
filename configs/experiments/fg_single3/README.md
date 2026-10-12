# FG Single3 训练配置

每个实验一份 YAML。Single3 离线/在线流程使用 `tools/fg_single3_experiment.py`，
S3 实验使用 `tools/fg_s3_experiment.py`，具体分派见下文。
公共网络定义复用 DreamerV3 presets，运行时保存完整的解析后配置和哈希。

本次使用 `C1_M1_10K_then_online_1M.yaml`：

- 离线：从随机初始化开始，10,000 次梯度更新，使用已验收 C1 Natural + P4 Replay；Actor/Critic Gate 全程关闭。
- 在线：离线完成后恢复整个 Agent 与优化器，开始 1M 环境步；使用相同 TRAIN 行情、Reward 和网络。在线 Replay、账户及 Gate 观察窗口重新开始。
- 在线 Gate 使用原有运行阈值、0.9915 / 0.004 开启余量以及连续 20 次观察；未达标期间 Actor/Critic 损失保持关闭。
- 离线 `step` 表示梯度更新，在线 `step` 表示环境步；模型累计更新计数保留。

当前 10,000 次离线更新已完成；配置已指向诊断接口修复后的在线目录，原失败目录保留。恢复准入通过后，`resume` 跳过已完成的离线阶段，保留完整模型和优化器。

2026-10-12 参数同步修复已提交至 DreamerV3 `0b3a641`。源码哈希已改变，旧准备收据与 Checkpoint 身份不能直接用于新源码恢复；需先生成并验证新的继续训练收据，保留全部旧产物。下面是统一入口形式，目前不能作为新源码已经通过恢复准入的命令。现有进程不会自动加载源码修复。见 [参数同步修复记录](../../../docs/FG_Single3_Policy_Parameter_Sync_Fix_20261012.md)。

```bash
python tools/fg_single3_experiment.py resume --config configs/experiments/fg_single3/C1_M1_10K_then_online_1M.yaml
```

新实验复制 YAML 并修改结果目录，完成 `prepare` 后使用 `start`；同一实验中断后使用 `resume`。各实验共用入口；已生成的配置快照和实验收据不修改。

`prepare` 只执行 CPU 准入。训练由用户执行 `start` 或 `resume` 后启动。当前两阶段不自动评估冻结的离线 Actor；独立 DEV 评估需使用在线训练后的模型，封存 TEST 不用于训练。

本次诊断错误及真实 GPU 验证见 [启动修复记录](../../../docs/FG_Single3_Online_Startup_Diagnostics_Fix_20261012.md)。

## S3-A / S3-B 实验管理

S3 使用 `tools/fg_s3_experiment.py` 管理 `plan / prepare / start / resume / audit`。
原有 Single3 离线预训练和在线接续继续使用 `tools/fg_single3_experiment.py`。
各实验 YAML 都集中在本目录，通过 `--config` 选择路径或下表名称。

S3-C 使用已有 S3 管理入口。本目录名不代表调用 Single3 预训练流程。
2026-10-12 已将 S3 YAML 的代码/模板定位改为兄弟 Dreamer worktree，TRAIN
输入只读复用 `research/fg-data`。原 V1 A/B 指定的正式源 62500 节点仍不存在；
`plan` 成功仅表示配置解析，实际来源准入后再生成新准备快照。初始核对记录见
[S3 入口与配置核对](../../../docs/FG_S3_Entry_and_Config_Review.md)。

| 配置名称 | 训练器 | 监督样本 / 权重 |
|---|---|---|
| `S3A_ORACLE_BC` | 原 S3-A | 全部合法空仓样本 / 1 |
| `S3A_ORACLE_BC_FROM_10K` | 原 S3-A 算法 + 公共来源准入 | 用户选定的完整 10000 来源；全部合法空仓样本 / 1 |
| `S3A_ORACLE_BC_FROM_ONLINE_180K` | 原 S3-A 算法 + 公共来源准入 | 当前用户选定的在线 180k 节点；全部合法空仓样本 / 1 |
| `S3A_FULL` | S3-B 公平对照入口 | 同一完整索引 / 1 |
| `S3A_MATCHED` | S3-B 公平对照入口 | 严格证书子集 / 1 |
| `S3B_CERTIFIED` | S3-B | 同一严格证书子集 / 冻结优势权重 |

公共网络默认配置通过 `model.template` 引用兄弟仓库
`../dreamerv3/dreamerv3/configs.yaml`，按 `model.presets` 解析。
源 Checkpoint 的模型配置和参数仍必须精确匹配；实验 YAML 不新增网络结构。
BC 复用公共优化器的兼容默认项，使用独立的常数学习率优化器且关闭原预热。

`prepared_dir` 保存请求配置、公共模板、有效配置、源模型配置、输入/代码哈希及
准备收据；`paths.features` 是独立且冻结的教师特征目录；`train.logdir` 保存各自的
日志、Checkpoint 和运行配置快照。正式日志必须在 `/data/logdir/fg_single3/` 下。

```bash
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
$PY tools/fg_s3_experiment.py plan --config S3B_CERTIFIED
# 核对并填写已完成的源 Checkpoint 后，准备与运行使用相同配置。
$PY tools/fg_s3_experiment.py prepare --config S3B_CERTIFIED
$PY tools/fg_s3_experiment.py start --config S3B_CERTIFIED
$PY tools/fg_s3_experiment.py resume --config S3B_CERTIFIED
$PY tools/fg_s3_experiment.py audit --config S3B_CERTIFIED
```

`plan` 仅解析配置，不表示数据或源模型已经准入。`prepare` 只在 CPU 上检查输入并
保存快照；`start` 根据快照生成/复用教师特征，再调用对应原 Actor-only 训练器。
正式 `start` 需要配置指定的 GPU；用 `CUDA_VISIBLE_DEVICES` 选择设备。
准备后修改中央 YAML 会使 `--config` 恢复失败。明确恢复旧快照时使用
`resume --prepared <prepared_dir>`，参数来自已保存配置。

S3-A / S3-B 原计算入口和独立 Checkpoint 协议保留；早期 `configs/fg_s3b/` YAML
仍服务原入口，新管理流程统一使用本目录。S3-C 使用同一管理入口的独立联合
训练分派。原 A/B 管理收据与说明见 `docs/FG_S3_Experiment_Management.md`。

S3-B 的 P1-B1 修复使用固定 TRAIN 平均权重，Checkpoint 为 `FG_S3B_BC_V2`。
三份对照配置使用独立的 `global_mean_v2` 实验/输出目录。新权重 manifest 是对
现有 V1 权重的冻结总体统计引用，原数组、Oracle、标签和特征均不重算：

```bash
$PY tools/fg_s3b.py normalization-build \
  --base-weights artifacts/fg_s3b_v1/weights_full_v1 \
  --output artifacts/fg_s3b_v2/weights_global_mean_reproduce
$PY tools/fg_s3b.py normalization-audit \
  --output artifacts/fg_s3b_v2/weights_global_mean_v2
```

当前配置引用已生成的 `weights_global_mean_v2`。CERTIFIED 使用 MATCHED 总体的
平均权重，FULL / MATCHED 等权对照使用 1；该尺度绑定进运行身份及恢复检查。
旧版本在两仓库的 `fg-s3b-batchmean-v1-20261012` 标签中保留，V2 不接受旧 S3-B
状态。具体公式、统计、CPU 证据和迁移边界见 `docs/FG_S3B_P1_Global_Mean_Fix.md`。

## S3-C1 原 RL + 等权 Oracle BC

S3-C 同样使用 `tools/fg_s3_experiment.py`。读取 YAML 使用 Dreamer canonical
reader，presets/覆盖项使用 `elements.Config` 和官方 Flags 解析。原 S3-A/B
训练器不改；C 不生成 BC feature cache、不使用独立 BC optimizer。

| 配置 | 来源与用途 |
| --- | --- |
| `S3C1_DIRECT_INIT` | 显式 seed 的原网络随机初始化；正式模板 |
| `S3C1_DIRECT_FULL` | 用户填写的冻结完整原模型；正式模板 |
| `S3C1_AFTER_BC_S3A` | 用户填写的已审计 S3-A 包和对应基线；正式模板 |
| `S3C1_CPU_FLOAT32` / `S3C1_CPU_BFLOAT16` | 小原网络、真实 TRAIN 链接切片、受控 READY 诊断夹具；仅 CPU |
| `S3C1_CPU_REAL_DIRECT_FULL` | 现有小原模型完整包；继承 CLOSED Gate 的 CPU 诊断 |
| `S3C1_CPU_REAL_AFTER_BC_S3A` | 现有 S3-A20 包；继承 Actor、原优化器及 CLOSED Gate 的 CPU 诊断 |

正式模板中的来源占位路径必须由用户填写。提供三个来源配置表示接口支持，
不代表选定正式训练顺序。所有预算为**梯度更新次数**；C 的 local update 从 0
开始，模型总计数为 source update offset + local update。

```bash
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
$PY tools/fg_s3_experiment.py plan --config S3C1_DIRECT_FULL
# 以下是明确的 CPU 诊断流程，每份配置拥有自己的 prepared/logdir。
$PY tools/fg_s3_experiment.py prepare --config S3C1_CPU_FLOAT32
$PY tools/fg_s3_experiment.py start --config S3C1_CPU_FLOAT32
$PY tools/fg_s3_experiment.py resume --config S3C1_CPU_FLOAT32
$PY tools/fg_s3_experiment.py audit --config S3C1_CPU_FLOAT32
```

CPU 示例使用已审计 `teacher_catalog_v1_final`；切片只链接原 C1 P1/P4 文件，
选择与哈希记录在 `artifacts/fg_s3c1_v1/input_slice_train_receipt.json`。文件缺失时
准备拒绝准入，需要按审计输入建立自己的诊断切片，不能自动回退其他数据。
全量 Catalog/來源审计继续使用现有 `tools/fg_s3_arch.py` 与 Dreamer
`scripts/fg_s3_arch.py`，C 不再提供第二套教师索引或迁移脚本。

C 准备协议是 `FG_S3C_PREPARED_V1 / READY_FOR_JOINT`，额外保存完整训练配置、
SourceAdmission 和 Catalog 身份。新 start 校验准备时的来源哈希并迁移一次；
resume 只恢复本 run 的 `FG_FULL_V1` Agent、joint optimizer、Gate/window、
Replay、train/report Prefetch、Consec.current、carry、scheduler/RNG/计数。
换来源、改 lambda/预算、改已准备 YAML 或代码不能当作同 run 恢复。

新增 BC 使用当前 posterior repfeat 的 stop-gradient 副本；每次只有原联合优化器
更新一次。Gate NOT READY 时新增 BC 为零，原 WM 继续训练。READY CPU 夹具和
真实但 CLOSED 的旧来源分别报告，CLOSED 更新不计为有效 RL+BC。
所有分类指标只描述 TRAIN imitation；不表示 DEV/TEST 或交易表现。
prior ADD 的只读 CPU 审计共用 `tools/fg_single3_prior_audit.py --config <实验YAML> --output <新目录>`；通过 `--offline-checkpoint` 和 `--online-checkpoint` 可固定已有阶段的 Checkpoint。工具复用原生 Replay 标注与模型的因果上下文、availability head，保存逐位置概率、曲线及输入哈希，不启动训练。2026-10-12 的实际结果见 [prior ADD 概率审计](../../../docs/FG_Single3_Prior_ADD_Probability_Audit_20261012.md)。

## Availability Gate 校准恢复

`C1_M1_GATE_T030_from_180K.yaml` 使用独立的新实验身份继承在线180k完整状态。训练核心仍复用原 Driver、Replay、Env 和 FullCheckpoint；只有概率判定阈值与必要统计/参数发布接线发生变化。CPU验收不能自动启动GPU。

```bash
python tools/fg_single3_experiment.py prepare --config configs/experiments/fg_single3/C1_M1_GATE_T030_from_180K.yaml
python tools/fg_single3_experiment.py publish --config configs/experiments/fg_single3/C1_M1_GATE_T030_from_180K.yaml
# CPU双路径完整恢复测试与证书生成后，用户另行授权：
python tools/fg_single3_experiment.py resume --config configs/experiments/fg_single3/C1_M1_GATE_T030_from_180K.yaml --authorize-training
```

`prepare` 校验冻结模型的全部质量条件、真实窗口及来源哈希，生成不可覆盖的恢复包；`publish` 只复制到新日志目录。正式 `resume` 必须同时具备 GO、完整状态CPU恢复证书、严格身份和显式授权。旧窗口合格 streak 清零，新Gate需重新连续20次通过；模型、优化器、交易账户、Replay、随机状态和累计180k环境步保留。

未通过完整预检时不提供绕过准入的启动路径。报告见 `docs/FG_Single3_Gate_Repair_Implementation_and_Verification.md`；GPU实际达标、DEV泛化和择时效果另行验证。

# FG Single3 训练配置

每个实验一份 YAML。Single3 离线/在线流程使用 `tools/fg_single3_experiment.py`，
S3 实验使用 `tools/fg_s3_experiment.py`，具体分派见下文。
公共网络定义复用 DreamerV3 presets，运行时保存完整的解析后配置和哈希。

本次使用 `C1_M1_10K_then_online_1M.yaml`：

- 离线：从随机初始化开始，10,000 次梯度更新，使用已验收 C1 Natural + P4 Replay；Actor/Critic Gate 全程关闭。
- 在线：离线完成后恢复整个 Agent 与优化器，开始 1M 环境步；使用相同 TRAIN 行情、Reward 和网络。在线 Replay、账户及 Gate 观察窗口重新开始。
- 在线 Gate 使用原有运行阈值、0.9915 / 0.004 开启余量以及连续 20 次观察；未达标期间 Actor/Critic 损失保持关闭。
- 离线 `step` 表示梯度更新，在线 `step` 表示环境步；模型累计更新计数保留。

当前 10,000 次离线更新已完成；配置已指向诊断接口修复后的新在线目录，原失败目录保留。使用 `resume` 跳过离线阶段，复用完整模型和优化器进入在线训练：

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

S3-C 将扩展已有 S3 管理入口。本目录名不代表调用 Single3 预训练流程。
2026-10-12 核对发现下列 S3 YAML 仍包含旧源码路径，且配置指定的正式源
62500 节点不存在；当前 `plan` 成功仅表示配置解析。修正路径并完成来源准入后
再生成新准备快照。入口、配置、待修问题及只读命令收据见
[S3 入口与配置核对](../../../docs/FG_S3_Entry_and_Config_Review.md)。

| 配置名称 | 训练器 | 监督样本 / 权重 |
|---|---|---|
| `S3A_ORACLE_BC` | 原 S3-A | 全部合法空仓样本 / 1 |
| `S3A_FULL` | S3-B 公平对照入口 | 同一完整索引 / 1 |
| `S3A_MATCHED` | S3-B 公平对照入口 | 严格证书子集 / 1 |
| `S3B_CERTIFIED` | S3-B | 同一严格证书子集 / 冻结优势权重 |

公共网络默认配置通过 `model.template` 引用
`/home/v/Documents/work/dreamerv3/dreamerv3/configs.yaml`，按 `model.presets` 解析。
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
仍服务原入口，新管理流程统一使用本目录。后续 S3-C 可以增加独立适配器和实验
配置，本轮仅支持 A/B。完整收据与说明见 `docs/FG_S3_Experiment_Management.md`。

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

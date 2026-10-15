# Quality-first：GPU 1K / 10K / 1M 启动与对照提案

状态：**提案，未启动 GPU；预测门槛未通过，Full Actor/Q 影响关闭。** 本提案不构成训练授权。

## 固定合同

固定源为完成的 Tutor Full 1M Checkpoint（40000 次模型更新）。原 Encoder/RSSM/Actor/Critic 不训练，保留原 C1 Reward；质量为 B(t,d)，MAE 独立。beta=0；辅助头梯度不回传 Backbone；新头使用来源 presets 和原优化器 warmup。所有预算按真实 Env 推进计量，4 次更新/100 Env 步，不把梯度、reset 或元数据算作 Env 步。

所有新 Run 独立配置/实验名/prepared/run 路径；恢复只用同一冻结准备。启动前从交付的双仓 SHA 建立 Gym/Dreamer 兄弟冻结 worktree，CPU prepare 后保留完整解析配置、代码/来源/数据/Reward/Oracle 身份。正式 Run 位于 `/data/logdir/fg_single3/s3_v2/`；仅用 sdd/sda。现有 dev 下的诊断 prepared 不是最终冻结训练快照。

## 预算

| 阶段 | Env 步 / 更新 | 配置 | 必须验证 |
| --- | --- | --- | --- |
| GPU 技术 1K | 1000 / 40 | `QUALITY_V2_GPU_1K.yaml`（实际实验名 QUALITY_FIRST_HEADS_GPU_1K_v2） | 实际来源 NN 继承、辅助头真实更新、原模型与 Actor/Q 冻结、显存/吞吐/覆盖、先 stop-at 500 再 resume 至 1000；不以此认定预测有效 |
| 辅助头 pilot | 每 Run 10000 / 400，save_every=2000 | `QUALITY_FIRST_{EVENT_ONLY,SCORE_ONLY,HEADS}_PILOT_10K.yaml` | 固定源、相同轨迹/预算对比任务；TRAIN holdout 与独立 DEV 的 NLL/Brier、Score MAE/排名、高分召回、风险分层、低资金覆盖 |
| Full 受控对照 | 同预算、同源、多 seed，另行授权 | `QUALITY_V2_{B1,B2,B3,FULL}_PILOT_10K.yaml` | 预测准入先通过；Actor-only 对照；每次只引入一个教学/融合变化；Q 无有效支持不得称 FULL |
| 1M | 1000000 / 40000 | 尚不发布可启动 1M 配置 | 留出、多 seed、多 fold、真实 Q/Actor 风险证据及全量 Catalog/资源验收后再决策 |

来源 warmup=1000 更新必须保留：1K 的 40 更新和 10K 的 400 更新属于早期技术/pilot 阶段，不可声称完成优化器收敛。

## 辅助头与后续 Full 消融

| 对照 | 差异 | Actor 影响 |
| --- | --- | --- |
| H0 | 固定原模型、TRAIN 先验/中位数 | 无 |
| H1 Event-only | Score lambda=0 | 无 |
| H2 Score-only | Outcome/mono lambda=0 | 无 |
| H3 Joint heads | 默认多窗口 Outcome + Score | 无 |
| B0 历史参考 | 原 Tutor-CD / Tutor Full / B2 的真实冻结 Checkpoint | 原 Actor；只读、标出预算差异，不能当作匹配预算消融 |
| B1 | 新 QualityActor，原 C1 strict 教学，无质量辅助/融合 | 预测准入后的单独受控研究 |
| B2 | 增加质量教学/辅助头，不融合预测 | 同上 |
| B3 | 加 Outcome/Score 有界 stop-gradient logit 修正 | 同上 |
| FULL | 再加经认证的 Q_V2 优势修正 | Q 证据不足则 FULL_Q_PENDING，不重复运行一个等同 B3 的“FULL” |

先固定 replay/mixing seeds **202、203、204**，来源模型 seed=0 和源 NN 完全相同；这些是训练采样 seed，不冒充多个独立预训练模型。每个 seed 使用新 YAML、实验名和路径。多模型同状态比较必须重建各自的因果 carry，上一动作来自相同的真实执行轨迹；不直接共享 RSSM latent。

首个 FG201 1K 只验工程。较大 pilot 的质量标签训练可固定 12 个 TRAIN 合约：FG001/005/009/101/105/109/201/205/209/301/305/309；留出 FG401/405/809/901/905/909。该划分只针对新质量监督；来源 Backbone 之前已使用过原 TRAIN，不声称完全未见合约。扩大前须冻结该划分，并完成各合约 holdout 评价执行/来源认证；当前共享评价已实测独立 DEV 和 FG401 留出，另外五个留出合约尚未运行。

DEV 不生成 TRAIN 标签、不用于挑 B 曲线参数；TEST 保持封存。当前 8 个 DEV smoke 候选的 Score MAE 为 24.21，TRAIN 中位数先验为 15.32；FG401 的 11 个同状态候选也未优于初始头参考。均不能作为准入。预测准入须有真实 Checkpoint、训练集与留出互斥、匹配状态参考比较、有效样本/风险/资金分层和独立 DEV 证据；不能自行把 status 字段置 PASS。

## 运行入口（授权后执行）

CPU prepare、start/resume、evaluate、report、score-predictions 均使用 `tools/fg_s3_v2_experiment.py`。预算授权后先建立最终 SHA 的冻结 worktree，并在其中重做：

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py prepare --config QUALITY_V2_GPU_1K
```

来源、配置、磁盘、GPU/显存准入通过后，GPU 启动命令为：

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py start --prepared artifacts/fg_s3_v2/prepared/QUALITY_FIRST_HEADS_GPU_1K_v2 --stop-at 500 --authorize-training
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py resume --prepared artifacts/fg_s3_v2/prepared/QUALITY_FIRST_HEADS_GPU_1K_v2 --authorize-training
```

这条命令本轮**没有执行**。中断则将 `start` 改为 `resume` 并使用同一 prepared。统一 W&B 通过已有 `tools/fg_s3_wandb.py launch --trainer-cwd <冻结 Gym 路径> --no-evals -- <同一训练命令>`，project 固定 `fg-single3-research`，例行 Actor-only 评价仅本地保留，不上传未来标签、Oracle、Replay 或完整 Checkpoint。

启动前重新测量真实 CPU/可用内存、GPU/显存、sdd/sda 空间；预检合约数、原生线程和 worker 内存，避免嵌套 BLAS 并行。当前只完成 CPU 检查：可用内存约 8.52 GiB、研究盘剩余约 21.29 GiB，来源 pickle 约 .604 GiB；这些不是 GPU 或 1M 资源认证。先量化全量 Catalog 的长持仓/右删失成本，再决定批量优化与 1M 预算。

9 个 10K Run 的 Checkpoint 不能按当前研究盘剩余量直接排队。单个真实源模型 pickle 约 .604 GiB，保存初始点及 5 个正式保存点仅模型文件就约 3.6 GiB／Run，尚未包括优化器扩展、Replay、旁路日志和评价。后续先测 1K 实际工件尺寸；冻结训练工作树与正式产物使用已允许的 sda `/data` 空间，并重新确认可用容量。保留旧工件，不以删除历史结果换取预算。

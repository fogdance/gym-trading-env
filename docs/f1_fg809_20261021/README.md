# FG809 正式 F1：独立工作树与启动命令

基线为最新双仓 `integration/s3-next`：Gym
`99e9f81dc91440bd38a04e84adacbb7c2616baa7`，Dreamer
`990f61f8b7ac6bad25ff9e41b65624ae13ccf497`。
双仓分支均为 `run/f1-fg809-20261021`，位于
`/home/v/Documents/work/research/fg-s3-f1-fg809-20261021/`。
本分支只添加本次 Run 配置、命令和核对材料；训练实现复用现有统一入口。

## 已选配置

| 项目 | 取值 |
| --- | --- |
| 阶段 | `FOUNDATION_WM` / `WM_NATURAL` |
| 唯一合约 | `CZCE.FG809`，2018-04-11 至 2018-08-08 |
| Corpus | `/data/s3_corpus/train_full_main_optimized_20261020_cpu20/batches/TRAIN01_FG809_FULL_MAIN` |
| Corpus 内容身份 | `76a856fb85ea526800bf2cfd6ce51be1ae26c8f7ac70dea0904bba6b4b629550` |
| 抽样 | 原 WP03 `NATURAL`，包含 P0/P1/P2_low/P2_mid/P3/P4 |
| 初始化 | 原 Dreamer `size12m` preset，fresh seed 202 |
| batch / 目标 / burn-in | 8 / T64 / K32；每窗口含 97 行及原始前缀 |
| 预算 | 10,000 次优化器更新，5,120,000 个目标位置预算 |
| 设备 | 单卡 GPU；本次没有运行 GPU |
| 日志 / Checkpoint | 每 100 / 500 次更新 |
| Run | `/data/logdir/fg_single3/s3_v2/E_QUALITY_F1_FG809_12M_U10000_20261021` |
| 恢复 | 原 FullCheckpoint，`STRICT_PUBLISHED_LATEST_V2` |

`size12m` 为原生预设名称，未据此声称实际扩展模型恰有 1,200 万参数。
Reader 按用户指示视为已就绪，本次不等待或重跑其生产验收。原 Corpus
及正在生产的所有其他合约均不修改。F1 来源仅为上述一个 Corpus。

## 正式命令（用户执行）

从任意目录启动：

```bash
bash /home/v/Documents/work/research/fg-s3-f1-fg809-20261021/gym-trading-env/docs/f1_fg809_20261021/RUN_F1.sh start
```

首次 `start` 自动调用原 `f1-prepare --source-purpose native-verification`，
创建新 F1 Prepared 和独立 Join index；成功后调用原 `f1-start --authorize-training`。
**现有入口在 prepare 和 start 时都会实际重验 G3，所以该命令先执行 CPU
准入，再进入 GPU 训练。**没有将历史 JSON PASS 冒充本次原生执行证明。

如需将正式准备与训练启动分开，先执行同一脚本的 `prepare`，再执行 `start`。
已有完整 Prepared 会复用，已有不完整产物会保留并报错，不删除后静默重做。
首次启动使用 `start`；中断后执行：

```bash
bash /home/v/Documents/work/research/fg-s3-f1-fg809-20261021/gym-trading-env/docs/f1_fg809_20261021/RUN_F1.sh resume
```

只读查看配置和准备状态：

```bash
bash /home/v/Documents/work/research/fg-s3-f1-fg809-20261021/gym-trading-env/docs/f1_fg809_20261021/RUN_F1.sh status
```

Verifier 使用本次独立冻结双仓。`../release/VERIFIER_PIN.json` 和
`../release/LAUNCH_LOCK.json`（相对 Gym）保存实际 SHA 与外部 Pin SHA256；
启动脚本不会重新生成 Pin，也不会根据启动时的 HEAD 自动提升信任。

## 本轮证据与未执行项

只执行 YAML/模型解析、Corpus ABI 元数据、Verifier 身份和 shell 语法检查。
实际结果见 `CONFIGURATION_CHECK.json` 和 `../release/LAUNCH_LOCK.json`。
没有运行 Agent、合成训练、正式 prepare、G3、Join 重建、GPU 或优化器更新。
正式 `F1_DATA_READY`、训练与 Checkpoint 冷恢复均为 **NOT_RUN**，由上述命令执行。

当前共用 W&B `research_tracking.identity.SourceRun` 只接受
`FG_S3_V2_LIVE_LEARNING_V1`，而本阶段使用 `FG_S3_E_QUALITY_F1_RUN_V1`。
本次保留 F1 原生日志/Checkpoint，不给出无法认证此协议的 W&B 启动命令。
后续若接 W&B，应扩展现有公共 Tracker，而不另建跟踪器。

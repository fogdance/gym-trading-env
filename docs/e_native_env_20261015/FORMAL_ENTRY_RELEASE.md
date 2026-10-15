# E + Tutor-CD Full 正式1M入口

2026-10-15用户明确要求：停止其他工作，放开正式训练入口，准备1M配置并给出启动命令。按该指令，仅解除 `e_multiscale_1m15m + tutor_cd` 的waiting_e；E+A2、E+B2和Tutor+B2保持原守卫。真实TRAIN的正式SourceBinding使用独立scope，仍完整校验来源、配置、种子、双仓SHA及源码；合成数据不能进入正式路径。

本次GPU访问已确认成功。用户要求停止后，已向本次诊断PID1027597发送SIGINT，进程退出码130；日志和原有Checkpoint保留。完整GPU1K和恢复没有完成，不记PASS。入口放开来自用户明确指令。没有启动正式1M，没有再次运行测试。

配置：`configs/experiments/fg_s3_v2/E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M.yaml`。

参数：完整size50m、全18个TRAIN合约、1000000真实市场分钟；每100分钟4次联合更新，共40000次；每100K分钟完整Checkpoint；Replay32768、batch1、beta1；Tutor Full alpha=.8、pair=.1、tau1、kappa1、cap2。原Reward、Oracle、TP5/SL30、Action Mask、交易安全Gate及盲测TEST规则不变。E从头初始化，不继承旧Encoder权重。

以下命令由用户执行。第一步只生成新的正式准备快照，第二步才启动1M；已存在的快照或Run不会被覆盖。启动后保持这对兄弟工作树源码与有效配置不变。

```bash
cd /home/v/Documents/work/research/fg-s3-e-model-dev/gym-trading-env

env PYTHONDONTWRITEBYTECODE=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare \
  --config E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M

env PYTHONDONTWRITEBYTECODE=1 FG_S3_VISIBLE_GPUS=0 \
  WANDB_ENTITY=xkm104501-none WANDB_PROJECT=fg-single3-research WANDB_MODE=online \
  /home/v/Documents/work/research/fg-s3-wandb-env/bin/python tools/fg_s3_wandb.py launch \
  --no-evals \
  --run-dir /data/logdir/fg_single3/s3_v2/E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261015 \
  --prepared "$PWD/artifacts/fg_s3_v2/prepared/E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261015" \
  --trainer-cwd "$PWD" -- \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start \
  --config E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M --authorize-training
```

同Run中断恢复：使用同一准备快照，将上述最后一条训练器子命令的 `start` 改为 `resume`，保留 `--authorize-training`。不要重新prepare或改写旧身份。原FullCheckpoint恢复模型、优化器、Replay、Sidecar、调度和累计计数。

本次仅修改入口和正式配置；只做静态语法及diff自审，不运行CPU/GPU训练或测试。正式准备快照由上面的prepare命令生成，不将其提前声称为已完成。

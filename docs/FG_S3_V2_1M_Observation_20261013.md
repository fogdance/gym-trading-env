# 同一条 A2 1M Run 的观察与固定 Actor-only 评价

2026-10-13。观察入口已运行，正式训练继续；没有拆分新训练实验，没有审批停点。
正式源码仍冻结 Gym `904bee6` / Dreamer `6356747`，未改 Loss、网络、调度、Checkpoint 或 Replay。
单一 Run：`/data/logdir/fg_single3/s3_v2/A2_GPU_FORMAL_1M_BETA_v1`。

## 节点与计量

| 环境步 | 关注内容 |
| --- | --- |
| 10K | 更新链路、Actor对认证Oracle标签的响应 |
| 50K | ADD提议识别、有效监督与实际行为 |
| 100K | 首次重点判断训练方向；独立评价 |
| 300K | 自主入场质量是否持续改善；独立评价 |
| 1M | 预算完成；独立闭环评价 |

当前运行每100K保存，因此10K/50K无模型快照，只记录准确节点的TRAIN交互和更新统计。
不能将0步Checkpoint评价标成10K/50K模型。现有0步模型先作基线，之后每个完整100K节点自动评价。
Actor-only在固定FG409 DEV起点、reset seed=0、eval argmax、5000实际环境步预算上运行；无Oracle接管。
每次调用正在训练的冻结 `tools/fg_s3_v2_experiment.py evaluate`，严格保留原身份校验。
记录实际使用Checkpoint、哈希、市场/规则身份和真实完成步数；中途真实终止则报告实际步数。
观察进程只读取训练日志和已完成Checkpoint，独立GPU进程评价，不给训练进程发停止或变更指令。
TEST及完整DEV合约生命周期评价未执行；当前固定5000步是受限闭环评价。

## 指标口径

- TRAIN：认证标签上的Actor提议ADD Recall/Precision；非法/未认证标签不计入混淆矩阵。
  这些指标来自混合控制到达的TRAIN状态，不能作为独立DEV识别能力。
- TRAIN：Actor实际接管的合法决策、自主执行ADD、实际自主成交，另列Oracle指导成交。
- TRAIN：有效标签/全部环境步、有效标签/合法决策、ADD教学支持、资金认证域外次数。
- 更新：真实WM与有效BC更新数、含ADD的BC更新数、Actor/WM确实变化的更新次数。
- 独立DEV：自主入场、TP5点达成比例、TP耗时（自然分钟p50/p90及有效bar p50）、SL30、首次逆行10点、Reward、右删失。
  TP5比例以全部实际入场为分母，包含失败和预算结束尚未完成的持仓；无入场时比例为null。
- ADD排序：保存合法已执行决策的p_add分位数及20个最高分，不将末尾未执行提议混入。

## 已观测事实

| TRAIN累计节点 | WM更新 | 有效BC | Actor变化更新 | 有效标签覆盖 | 合法覆盖 | 自主成交 | 提议Recall / Precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10K | 400 | 400 | 400 | 87.25% | 100% | 0 | 3.51% / 3.47% |
| 50K | 2000 | 1876 | 1876 | 76.80% | 100% | 8 | 4.09% / 3.93% |

两个节点资金认证域外均0。10K处beta=1，Oracle指导成交427，Actor自主成交0。
固定DEV的**0步初始化模型基线**（真实GPU、并行于正式训练）：46入场、33 TP5、12 SL30、24逆行10点、1右删失。
TP5比例71.74%，TP自然分钟p50/p90=102/1640，有效bar中位数64，Reward=-26.6923。
该基线不是当前50K模型结果，不能由TRAIN的高TP数量推出Actor独立改善。

## 已启用入口与收据

后台窗口 `d3:a2-observe`，观察进程最后核对PID2865476；训练PID2838219持续运行。
输出为 Run 内 `observations/summary.json`，各Checkpoint的`actor_only_step_*.json`及原始评价目录。
进程断开后以下命令可继续观察；独占锁防止重复运行，已有评价结果复用：

```bash
cd /home/v/Documents/work/research/fg-s3-dev/gym-trading-env
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_observe.py \
  --prepared /home/v/Documents/work/research/fg-s3-v2-train-beta/gym-trading-env/artifacts/fg_s3_v2/prepared/A2_GPU_FORMAL_1M_BETA_v1 \
  --run-dir /data/logdir/fg_single3/s3_v2/A2_GPU_FORMAL_1M_BETA_v1 \
  --eval-steps 5000 --watch
```

3项针对性测试PASS（只验证指标和日志边界，无神经计算）：
`env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fg_s3_v2_mpl /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_fg_s3_v2_observation.py --junitxml=artifacts/fg_s3_v2/receipts/observation_contracts.xml`。
首次入口校验发现同名src模块和tools文件，已限定使用tools冻结入口并加回归测试；随后实际GPU评价完成。
已验证：0步独立GPU闭环、10K/50K实时指标、后台并行运行、训练不中断、冻结树tracked clean。
尚未验证：100K及后续模型评价（等待真实Checkpoint产生），1M完成，未来回滚后的持续观察，独立交易能力改善。

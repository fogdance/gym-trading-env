# FG809 正式 F3

独立双仓：/home/v/Documents/work/research/fg-s3-f3-fg809-20261022/
分支 run/f3-fg809-20261022。基线为实际 integration/s3-next，保留 a4b3bb0 流式哈希修复。
不修改 F1、教材生产工作树或主线。

## 配置

用户选择 size12m、10,000 次更新、batch=8、T64、K32、seed202、单卡 GPU。
ACTOR_SUPERVISED 直接继承正式 F1 的 step_000000010000 非优化器参数，新建优化器与计数。
FG809 六行为 NATURAL，原生默认 Natural Loss=1，Balanced/Strict Pair/Quality Pair=0，无辅助流。
只训练 quality_actor；其他参数按原 F3 合同冻结。复用生产 G3、独立 Reader 与原 FullCheckpoint。
最多消费 40,000 个 batch；有效更新预算 10,000，数据支持不足时由原入口报错。

## 正式命令（用户执行）

~~~bash
bash /home/v/Documents/work/research/fg-s3-f3-fg809-20261022/gym-trading-env/docs/f3_fg809_20261022/RUN_F3.sh start
~~~

首次 start 自动使用原 f3-prepare --source-purpose certified-corpus，然后 f3-start --authorize-training。
已有完整 Prepared 直接复用。中断后同脚本用 resume，status 只读核对冻结配置元数据。
保留 STRICT_PUBLISHED_LATEST_V2，恢复不静默回退。F3 Run 目录由训练器创建。

## 路径

配置：configs/experiments/fg_s3_v2/e_quality_f3_fg809_size12m_u10000_20261022.yaml

Prepared、启动锁、临时文件、缓存和 launch.log：
/data/logdir/fg_single3/s3_v2/F3_FG809_12M_U10000_20261022/

模型、Checkpoint 和 updates.jsonl：
/data/logdir/fg_single3/s3_v2/E_QUALITY_F3_FG809_12M_U10000_20261022/

~~~bash
tail -f /data/logdir/fg_single3/s3_v2/E_QUALITY_F3_FG809_12M_U10000_20261022/updates.jsonl
~~~

F3 Checkpoint 位置按 consumed_batches，实际更新数看 optimizer_updates。
准备不等于 GPU 训练完成；GPU 迁移/更新、冷恢复和 Actor-only 效果评价以实际执行收据为准，未运行部分为 NOT_RUN。
不追加测试集、不重跑 G3、不读取 TEST、不自动合并主线。

# FG809 正式 F1：复用生产认证

新冻结双仓位于：
/home/v/Documents/work/research/fg-s3-f1-certified-20261021/

配置：size12m，10,000 次更新，batch=8、T=64、K=32、seed=202、单卡 GPU。
仅使用 CZCE.FG809 TRAIN，六行为自然采样保持不变。

## 启动命令（用户执行）

从任意目录执行：

~~~bash
bash /home/v/Documents/work/research/fg-s3-f1-certified-20261021/gym-trading-env/docs/f1_fg809_20261021/RUN_F1.sh start
~~~

首次启动自动准备，然后进入训练。准备、启动和恢复均复用正式生产批次的 Native G3 和独立 Reader，不执行 Teacher、Oracle 或账户回放。
仍核对批次外部 SHA256、身份锁、内容哈希、原始 Verifier Pin、六行为完整收据、TRAIN 来源和 ABI；文件检查仍需要读取教材。
复用的 Reader SQLite/JSON 逐字节复制，不重建投影或标签。

中断后的正式恢复：

~~~bash
bash /home/v/Documents/work/research/fg-s3-f1-certified-20261021/gym-trading-env/docs/f1_fg809_20261021/RUN_F1.sh resume
~~~

恢复保留原 FullCheckpoint 和 STRICT_PUBLISHED_LATEST_V2，损坏的最新检查点不静默回退。用户授权仍由正式命令显式传递。

## 数据与落盘位置

- 原 Corpus：
  /data/s3_corpus/train_full_main_optimized_20261020_cpu20/batches/TRAIN01_FG809_FULL_MAIN
- Corpus 身份：
  76a856fb85ea526800bf2cfd6ce51be1ae26c8f7ac70dea0904bba6b4b629550
- 正式生产发布批次：
  /data/s3_corpus/train_full_main_scheduler_20261021/attempts/TRAIN01_FG809_FULL_MAIN/publish/0002/BATCH_MANIFEST.json
- 批次外部 SHA256：
  3440ab119346be715a0b203a1d8e3c2cb086cff1b18695b6e4b2ffbcd39bd349
- Prepared、启动锁、临时目录与缓存：
  /data/logdir/fg_single3/s3_v2/F1_FG809_12M_U10000_20261021/
- 模型、更新日志、全部 Checkpoint：
  /data/logdir/fg_single3/s3_v2/E_QUALITY_F1_FG809_12M_U10000_20261021/
- 更新日志：
  /data/logdir/fg_single3/s3_v2/E_QUALITY_F1_FG809_12M_U10000_20261021/updates.jsonl

生产教材不修改、不重签。此前中断的准入日志保留在控制目录的 interrupted_20261021/。
无用的 3.2 GiB G3 临时值缓存已删除。

本目录中的历史 CONFIGURATION_CHECK.json/HISTORICAL_LAUNCH_LOCK.json 属于旧启动尝试；新启动只认 /data/logdir 下的 LAUNCH_LOCK.certified.json。
实际验证范围与 NOT_RUN 见 ../f1_certified_20261021/IMPLEMENTATION_REVIEW.md。

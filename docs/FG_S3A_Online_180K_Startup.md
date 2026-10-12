# S3-A 在线180k来源的启动记录

## Summary

当前选择：S3A_ORACLE_BC_FROM_ONLINE_180K；CPU prepare 已完成，READY_FOR_FEATURES。
正式 GPU 未由 Codex 启动。统一入口仍是 tools/fg_s3_experiment.py。

## Evidence observed

来源原节点：/data/logdir/fg_single3/online_after_10k_callback_fix/C1_M1_ONLINE_1M/ckpt/step_000000180000。
来源仍训练中，由用户确认。已复制完整183文件为独立只读 snapshot：
research/fg-s3-dev/gym-trading-env/artifacts/fg_s3_arch_v2/online_180k_source。
未改写来源模型、日志、Replay、Checkpoint、Gate或运行进程。

来源计数：updates=21183、batches=21399、actions=180000；Gate=false。
180000环境步和21183模型更新用于来源记录与后续联合训练的计数，不作为A的预算。
A的独立BC从0计到1000，原联合optimizer和原模型更新计数冻结。

训练worktree固定提交：Gym 5293e978d3f234d5f7c3149f326f1da831012bad；
Dreamer 93a6fb65b493b60f7c1205b8351d7024cbfc326a。
已包含最新远端分支；后续开发继续在 fg-s3-dev，不改训练源码。

## Root cause

原A V1准入限定62500离线节点。共享V2来源准入现在识别原FG_FULL在线
provenance、C1 TRAIN来源和真正Agent计数，不依赖checkpoint目录数字推测更新量。
后续新来源只指定新路径与新实验名；程序自动验证、冻结、读取计数，无须手填。
同一A运行中断用resume，不重新迁入来源。输入或代码改变使用新准备快照。

## Changes made

公共SourceAdmission和freeze_source负责在线来源格式；现有S3管理器调用它们。
原ActorBC、等权标签、Sampler、独立BC优化器和atomic checkpoint实现不修改。
当前新配置不修改旧10k准备、旧标签、旧缓存或历史checkpoint。
TRAIN/Oracle/Catalog只读复用，不重新生成；该模型对应的特征在start时生成。

## Verification performed

真实在线来源CPU载入原S3A Agent，286个完整参数/状态条目逐字节一致；
保留joint optimizer、Gate、norm与计数；没有BC optimizer导入来源Agent。
收据：online_180k_neural_transfer_{receipt,summary}.json；
在线单位反例合同1 passed；冻结来源183文件与原manifest校验；
正式准备收据在训练worktree artifacts/fg_s3_v2/experiments/S3A_ORACLE_BC_FROM_ONLINE_180K_seed0。

准备收据SHA256：4060a6d0c766c30d37384bfbc88646e0ba75db5aa34b556b14569571ace7e49d。
来源manifest SHA256：4979acfb0272ffbc3c96833e5b77fccd66c5d4d0f335bfca48810f764d927a57。

## Risks / not verified

formal GPU=NOT_RUN_BY_CODEX；用户启动结果未推断。DEV/TEST=NOT_RUN。
当前仍有来源训练，GPU启动时间由用户安排。上述286条证明参数继承，不证明
BC收敛、泛化或交易收益。原FG_FULL计数、身份和source snapshot不改写。

## Next recommended step

下面由用户执行；prepare已完成，无须再次执行。

```bash
cd /home/v/Documents/work/research/fg-s3-train/gym-trading-env
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
S3A_LOG="artifacts/fg_s3_v2/experiments/S3A_ORACLE_BC_FROM_ONLINE_180K_seed0/launcher_$(date +%Y%m%d_%H%M%S).log"
nohup env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/v/Documents/work/research/fg-s3-train/dreamerv3:/home/v/Documents/work/research/fg-s3-train/gym-trading-env/src \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" tools/fg_s3_experiment.py start --config S3A_ORACLE_BC_FROM_ONLINE_180K \
  > "$S3A_LOG" 2>&1 &
tail -f "$S3A_LOG"
```

训练结果：/data/logdir/fg_single3/s3/S3A_ORACLE_BC_FROM_ONLINE_180K_seed0。
中断后恢复（相同固定worktree/config，不变更来源）：

```bash
$PY tools/fg_s3_experiment.py resume --config S3A_ORACLE_BC_FROM_ONLINE_180K
$PY tools/fg_s3_experiment.py audit --config S3A_ORACLE_BC_FROM_ONLINE_180K
```

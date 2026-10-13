# S3-V2 P0：恢复与真实 GPU 10K 结果

2026-10-13。**P0工程PASS / GO_WITH_LIMITS；自主ADD效果FAIL（ALL_WAIT）**。
P1 A2→C2、R1、B2后置。正式1M、TEST、不中断GPU位级等价均未执行。

## 用户要求的四项结果

| 项目 | 实测结果 |
| --- | --- |
| GPU训练是否完成 | 是。指定Conda、真实42,584,132参数Replay模型、FG201 TRAIN、原Oracle；最终版1000步/40更新与独立10K/400更新完成。无CPU神经训练。 |
| 10K试训练 | 10000真实Env推进、400次WM+BC、100轮Actor/WM均变化；有效教师8360行。BC前20次均值0.4404→后40次0.3441；WM loss 24.519→22.193。最后40次ADD recall=0/223。 |
| 1M预计耗时 | 10K循环243.45秒；核心46.29步/秒，去除首1000步编译后57.30步/秒。暖态外推4.85小时，预计5–7小时。仅基于FG201短跑，18合约切换和长期额外开销未测。 |
| 真正阻断项 | 固定来源、P0恢复与配置准备无已知工程阻断；正式授权尚缺。自主策略ALL_WAIT，增加预算能否改善尚无证据。下一轮优先研究ADD采样/类别权重/接管调度；P1迁移不阻塞A2。 |

beta=1。Actor提出ADD共1040次，在896个Oracle ADD机会命中108次（随机提议召回12.05%）。
896成交/896 TP、SL=0、Reward=841.8357、TP分钟p50/p90=2/5、逆行10事件=0，**全部属于教师指导TRAIN**。
独立FG409 DEV、eval argmax、无Oracle、1000步：自主ADD/成交/Reward均0，ALL_WAIT。
损失下降和参数变化证明更新，不能证明学会优质入场。

## P0验证与修复

- GPU在500步保存退出，新进程恢复到1000步并继续20次更新；累计21183→21203→21223。
  Actor/WM在两段均改变；完整Critic、slow target、Normalizer和Critic优化器状态不变。
- 复用原FullCheckpoint，保存完整Env/Driver/carry、模型/优化器、Replay/采样、已提交与待完成教师、Gate/RNG和计数。
  最终10K的2841个Sidecar owner与全部物理Replay owner相等；10009行历史另存日志（含reset标记）。
  未完成native行不发布；恢复前归档未提交日志尾部。Native Replay无teacher/oracle/Q字段。
- 慢版10K在5000步中断，旧Checkpoint/日志保留，**不计完整10K通过**。
  真正3000步账户profiling确认完整历史导出是主要查询开销。
  当前决策描述保存真实余额/权益、cursor、Mask、持仓、状态和active reward episode；完整市场仍核对，完整历史仍保存在Checkpoint。
  同一真实账户20次查询4.58秒→0.124秒，动作/认证/理由一致。原Env、Oracle后缀数学、Reward、网络、Loss未改；教师描述不进入Actor observation/carry。
- 梯度/Critic更新检查每次执行；完整参数树检查在首轮、恢复首轮及保存边界执行。
  10K真实Gate仅1个report、CLOSED、RL更新0；20窗READY验收NOT_RUN。

进程峰值RSS3918MiB，Oracle近期查询p50/p95/p99=3.681/4.053/4.534ms。
10K保留11个完整Checkpoint，共6.69GiB。GPU只有离散采样：同模型旧短跑见2194MiB；最终10K GPU峰值未连续测量。

## 冻结版本与启动命令

当前训练树：`/home/v/Documents/work/research/fg-s3-v2-train/{gym-trading-env,dreamerv3}`，detached、tracked clean。
Gym `b71250c0708dd0f34aa61f80585f47f43ad94223`；Dreamer `e64004e3da2cfe2848e6004234847c758ec41cc0`。
原Online180k来源manifest SHA256保持`4979acfb0272ffbc3c96833e5b77fccd66c5d4d0f335bfca48810f764d927a57`，模型更新21183。
两条只读输入软链接复用dev的`teacher_catalog_v1_final`、`online_180k_source`；行情/Oracle复用授权数据别名`research/fg-data`。
原research仓库、旧S3和T030训练树未改；过渡冻结树`fg-s3-v2-train-p0`保留。

已准备冻结Gym树`artifacts/fg_s3_v2/prepared/A2_P0_GPU_FORMAL_1M_FAST_v1`：18个原TRAIN合约、1M真实Env步、40000更新、Replay容量8192、50K步保存。
正式Run `/data/logdir/fg_single3/s3_v2/A2_P0_GPU_FORMAL_1M_FAST_v1`尚未创建。
以下**仅授权后执行**，目前未执行。模板仍beta=1，不称已验证的自主探索课程；重复历史市场不称新增独立行情。

```bash
cd /home/v/Documents/work/research/fg-s3-v2-train/gym-trading-env
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config A2_P0_GPU_FORMAL_1M_FAST --authorize-training
# 同Run恢复
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config A2_P0_GPU_FORMAL_1M_FAST --authorize-training
```

## 实际验证与收据

dev Gym：6项控制行情/公共合同测试PASS，1项实际GPU工件检查PASS（只读保存结果、无CPU神经计算）。实际命令：

```bash
env PYTHONPATH=src:../dreamerv3 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fg_s3_v2_mpl /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_fg_s3_v2.py tests/entry_v1/test_fg_s3_v2_pool.py -k 'not real_gpu' --junitxml=artifacts/fg_s3_v2/receipts/p0_current_state_contracts.xml
env PYTHONPATH=src:../dreamerv3 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fg_s3_v2_mpl /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_fg_s3_v2_artifacts.py -k actual_p0 --junitxml=artifacts/fg_s3_v2/receipts/p0_actual_gpu_artifacts.xml
```
完整真实命令与终端输出存于`artifacts/fg_s3_v2/receipts/p0_fast_*_terminal.log`；统一入口调用：

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config A2_P0_GPU_SMOKE_1K_FAST
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config A2_P0_GPU_SMOKE_1K_FAST --stop-at 500
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config A2_P0_GPU_SMOKE_1K_FAST
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config A2_P0_GPU_TRAIN_10K_FAST
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config A2_P0_GPU_TRAIN_10K_FAST
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py evaluate --config A2_P0_GPU_TRAIN_10K_FAST --eval-steps 1000
```

权威收据：`p0_training_summary.json`、`p0_actual_gpu_verification.json`、`p0_10k_metrics.json`、两份XML；均在上述receipts目录。
实际模型/Replay保留在`runs/A2_P0_GPU_SMOKE_1K_FAST_v1`和`runs/A2_P0_GPU_TRAIN_10K_FAST_v1`。
PASS：真实GPU学习、实用恢复、Replay/教师绑定、来源保持、冻结准备。FAIL：自主ADD研究效果。
NOT_RUN：正式1M、TEST、20窗READY、不中断位级等价、P1/R1/B2。控制行情与代码检查不称生产验收。

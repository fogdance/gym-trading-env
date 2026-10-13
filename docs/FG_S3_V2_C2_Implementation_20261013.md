# S3-V2 C2 实施与实际验收 — 2026-10-13

## 结论与范围

C2 已具备 WM+Oracle BC 和 Gate READY 时的原 Dreamer Imagination RL 联合更新能力。
真实 CPU 小模型证明 Actor 同时获得 BC/RL 梯度、Critic 获得 RL 更新，且恢复后可以继续。
结论为 **GO_WITH_LIMITS**：这是功能验收；新版 GPU、真实市场自然开启 Gate、正式 C2 1M 均未运行。
未开发 A2→C2 迁移、B2、R1，也未修改 Reward、Oracle、网络层或原 Loss 定义。

历史 C2 GPU 500 步的 RL 更新为 0 是 Gate CLOSED 的符合规则行为。
本次缺口是完整状态切换证据、CPU 诊断入口和 C2 恢复后的联合学习证据；无需重构核心算法。

## 源码、隔离与实现

- 双仓独立开发目录：`/home/v/Documents/work/research/fg-s3-v2-c2-dev/{gym-trading-env,dreamerv3}`，分支均为 `feat/fg-s3-v2-c2`。
- 验收源码：Gym `3df593ae0055630d77eeeb293233eca4e33079a6`；Dreamer `7ff697f70c92e0a9f764ef738437415470d0e59f`。后续报告提交仅增加文档。
- A2 冻结版本保持 Gym `904bee6f851fd70d96dca48d06951bb466505561` / Dreamer `6356747d64963fb5ef55a036d837591f778c9f0a`。不修改或重启其训练和观察进程。
- 实际 Python 导入均来自新目录内的 Gym `src/gym_trading_env` 与兄弟 Dreamer `dreamerv3`、`embodied`。
- 数据共享：新 Gym 的 `artifacts/fg_s3_arch_v1/teacher_catalog_v1_final` 和 `artifacts/fg_s3_arch_v2/online_180k_source` 为只读使用的输入软链接，分别指向原 `fg-s3-dev/gym-trading-env/artifacts/` 下的同名工件。TRAIN 沿用已授权的 `research/fg-data` 数据别名。没有动态源码软链接。

保留既有 `S3V2Agent`、Collector、OracleStateQuery、Native Replay、教师 Sidecar、SourceAdmission 和 FullCheckpoint。
增加 RL Actor/Critic 分项 Loss 指标，以及仅限 C2 diagnostic 的 CPU 后端；正式训练与 A2 仍要求 GPU。
公共入口仍为 `tools/fg_s3_v2_experiment.py`，新增 `c2_check`，不增加另一套训练循环。

CLOSED 只执行 WM+BC。Actor 的共享优化状态随 BC 更新；Critic 专属动量、SlowCritic 和 RL Normalizer 冻结。
READY 使用原 WM+RL Loss 加等权 Oracle BC，BC 在原 `repfeat` 停止梯度，只更新 Actor；原联合优化器一次更新。
正式 Gate 保留 0.30、80% ADD Recall、连续 20 个合格窗口及原其他规则。
原连续退化 5 窗的行为仍是停训报错，诊断中的重新 CLOSED 不改变正式行为。

## 实际测试结果

| 项目 | 结果 | 证据与边界 |
| --- | --- | --- |
| CPU C2 后端边界、真实神经状态切换和恢复 | PASS | 新测试文件 2 passed，129.30s；54,468 参数的小模型、原 Env 受控行情 |
| Replay 行归属恢复、Oracle 过期状态拒绝 | PASS | 两项现有回归，2 passed，3.80s |
| CLOSED → READY → 恢复 READY → CLOSED | PASS | 7 次有效 BC 更新，另 1 次空 BC 更新；共 8 次 WM、3 次 RL Actor/Critic 更新 |
| 联合优化次数及状态计数 | PASS | 每次有效更新 optimizer step 增 1；空 BC 后总更新 8、采样 512 行 |
| 原 Replay 大模型初始化与跨进程恢复 | PASS | `C2_CPU_FG201_300_v2`，42,584,132 参数，200→300 步、新进程继续 2 次更新；共 6 次 WM+BC |
| Actor-only 评价入口 | PASS（功能） | v2 FG409 DEV 100 步，无 Oracle、无教师接管；All-WAIT，不证明交易能力 |
| 真实市场自然达到 20 窗 READY | NOT_RUN | 受控指标只验证控制器与 RL 接线，不冒充生产 Gate 达标 |
| 新版真实 GPU / 正式 C2 1M / TEST 数据集 | NOT_RUN | A2 占用 GPU，本任务结束后停止 |

小模型分项梯度：Actor BC `0.45352145`、Actor RL `0.02974033`、联合 `0.46447875`；
逐参数相加误差最大 `5.59e-9`，BC 对非 Actor 梯度为 0。

| 状态 | Actor 总梯度 | Critic 梯度 | SlowCritic 计数 |
| --- | --- | --- | --- |
| CLOSED 两次 | 0.442848、0.365487 | 0、0 | 0、0 |
| READY 两次 | 0.323744、0.406745 | 1.798210、1.688323 | 1、2 |
| 恢复 READY 一次 | 0.271733 | 1.513280 | 3 |
| 重新 CLOSED 两次 | 0.398617、0.368726 | 0、0 | 3、3 |

关闭前后逐字节比较 Critic、SlowCritic、Normalizer 和 Critic 动量；READY 验证实际参数、动量及统计变化。
测试使用非零继承动量。测试夹具启用有状态 valnorm/advnorm；生产配置仍为 `none`，并验证无可变参数。
READY FullCheckpoint 的模型、优化器、Gate、Replay、Sidecar 及训练计数恢复一致；恢复后真实继续更新。

## 命令与收据

以下命令从新 Gym worktree 执行，Python 为 `/home/v/miniconda3/envs/dreamerv3/bin/python`。
CPU 神经任务实际使用 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 nice -n 10` 限资源执行；未初始化 CUDA。

```bash
env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD/src:$PWD/../dreamerv3" taskset -c 20-23 nice -n 10 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  tests/entry_v1/test_fg_s3_v2_c2.py -s --basetemp=artifacts/fg_s3_v2/checks/c2_pytest_cpu_v2

# 现有数据正确性回归（已执行）：
env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD/src:$PWD/../dreamerv3" taskset -c 20-23 nice -n 10 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  tests/entry_v1/test_fg_s3_v2.py::test_live_committed_rows_restore_preserves_native_ownership \
  tests/entry_v1/test_fg_s3_v2.py::test_current_decision_descriptor_preserves_labels_and_rejects_stale_state

# 原模型诊断生命周期；每条神经命令均带上述限资源前缀执行：
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config C2_CPU_FG201_300
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config C2_CPU_FG201_300 --stop-at 200
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config C2_CPU_FG201_300
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py evaluate --config C2_CPU_FG201_300 --eval-steps 100
```

主要收据位于新 Gym 的 `artifacts/fg_s3_v2/`：

- `checks/c2_pytest_cpu_v2/test_real_cpu_c2_gradients_gat0/c2/receipt.json`、`actor_gradient_components.json`、`updates.jsonl` 和 READY/final FullCheckpoint。
- `prepared/C2_CPU_FG201_300_v2/manifest.json`：双仓版本、完整配置、TRAIN/Oracle/来源身份。
- `runs/C2_CPU_FG201_300_v2/training_receipt.json`、`source_transfer.json`、`ckpt/`：原模型实际训练与恢复。
- 历史 `C2_CPU_FG201_300_v1` 和前两轮小模型收据完整保留。自查修正空 BC 诊断采样计数后创建 v2，不修改旧快照或绕过源码校验。

原来源为 Online180k：180000 是环境步，继承模型更新计数 21183。
使用共享 `direct_full` 迁入完整原模型及联合优化器；不迁入独立 BC optimizer。
跨阈值 0.50→0.30 的 Gate 窗口按公共协议清空；同运行恢复继承自己的 Gate。

真实 FG201 TRAIN：6 次 WM+BC，模型更新计数 21189，BC NLL 从 `0.695643` 降至 `0.485712`。
恢复后的 Actor 梯度为 `5.725976`、`5.495291`，WM 也继续改变；全程真实 Gate CLOSED、RL 更新为 0。
有效教师覆盖率 `250/300`；39 次执行 ADD/TP 均为 beta=1 的 Oracle 接管成绩，不计作 Actor 自主交易。
工件核验确认 200/300 行 Replay 与 Sidecar 逐行归属一致、未完成行未发布、原来源哈希不变、RL 专属状态冻结。
收据：`receipts/c2_cpu_original_model_verification_v2.json`，这是保存结果核验，不计为新增神经计算。
Actor-only 收据：`runs/C2_CPU_FG201_300_v2/evaluations/actor_only_9aecb5acac/receipt.json`，实际 100 步、ADD=0。

## 后续入口与停止

待 A2 释放 GPU 后，若用户决定运行 C2，可使用独立 `C2_GPU_FG201_500_V2` 配置及既有 prepare/start/resume/evaluate 入口。
GPU 短跑及 `c2_check --device gpu` 本次没有执行，不声明通过。
本次不提供或执行 C2 正式 1M 启动；后续训练由 A2 实际结果和用户决定。

最终自查：核心训练数学与网络未修改，无私有 Oracle/迁移实现；输入与旧工件保留，A2 冻结源码保持不变。
完整状态测试与原模型真实运行分开报告，没有将静态或工件检查当作神经计算，也没有将 Oracle 接管 TP 算为 Actor 成绩。
最后主机只读检查：A2 PID `2838219`、观察 PID `2865476` 均仍运行；GPU 计算进程仅 A2，2194 MiB；A2 双仓 tracked 工作树干净、HEAD 与上述冻结值一致。

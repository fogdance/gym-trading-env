# S3-B2 GPU 利用率只读诊断（2026-10-14）

## Summary

当前 B2-Full 正常使用 CUDA 并持续训练。GPU 平均利用率偏低的主要瓶颈位于串行采集阶段，特别是每 1,000 环境步的合约账户切换：切换重新构造 Env，重新读取整份合约数据、校验并生成行情特征，然后恢复账户状态。采集完成后才执行 4 次梯度更新；训练 batch size 为 1。

没有修改训练代码、配置、Run、Checkpoint、Replay、历史评价或 W&B；没有发送信号、暂停或重启训练。E-DATA 按用户要求暂停。

## Evidence observed

Run：`/data/logdir/fg_single3/s3_v2/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014`。

训练 PID：`4045558`，W&B 启动器 PID：`4045557`。训练进程从 2026-10-14 15:58:35 开始运行。本次读取的最后一个完整 cycle 为 359,000 环境步。

冻结双仓库版本与 `run_identity.json.identity.versions` 一致，两个训练工作树 `git status --porcelain=v1` 均为空：

| 仓库 | Git SHA |
| --- | --- |
| Gym | `3970e6ec69f67fb2d02f30456a077ea5a31fd05d` |
| Dreamer | `5fd4ebdeb590fd8433338300bd82c7f2f520bf76` |

### 真实循环日志

只读统计 `cycles.jsonl` 最后 500 个完整 cycle（309,100–359,000 环境步）。每 cycle 收集 100 步、更新 4 次。

| 指标 | 实测 |
| --- | ---: |
| 每 cycle 采集平均耗时 | 3.3692 秒 |
| 每 cycle 更新阶段平均耗时 | 0.5689 秒 |
| 采集占采集与更新耗时之和 | 85.55% |
| 无 reset 的 cycle（450 个）采集平均耗时 | 1.5648 秒 |
| reset 增加的 cycle（49 个）采集平均耗时 | 19.6212 秒 |
| reset 增加的 cycle 采集中位耗时 | 19.5966 秒 |

分组使用相邻 cycle 的 `counts.resets` 增量，第一条无法归组。更新阶段包含日志、策略权重同步和 Gate report；该时间不是纯 GPU kernel 时间。两阶段统计没有包含阶段外的 prune、Checkpoint 保存等耗时，因此不把该比例当作整个进程绝对时间占比。

解析后的 `config.yaml`：`batch_size=1`、`batch_length=64`、`replay_context=1`、`jax.platform=cuda`、`jax.jit=true`、`jax.sync_train_metrics=true`。冻结实验配置：`collect_every=100`、`updates_per_collect=4`、`env_slice_bars=1000`。

### 主机资源采样

2026-10-14 19:54:04 起，20.576 秒主机只读采样。普通命令能够看到真实训练进程，沙箱内 Python 的进程命名空间看不到该 PID；最终 `/proc` 采样使用主机权限，结果仅写 `/tmp`。

| 指标 | 实测 |
| --- | ---: |
| GPU 利用率，20 次采样 | 平均 21.15%，最小 9%，最大 98% |
| GPU 总显存占用（含桌面） | 约 3,381 MiB / 16,380 MiB |
| B2 进程显存（`nvidia-smi`） | 2,196 MiB |
| GPU 温度 | 51–56°C |
| GPU SM 时钟 | 2,610–2,760 MHz |
| B2 主线程 CPU | 97.88%（100% 表示一个逻辑核心） |
| B2 全部线程 CPU | 107.84% |
| 系统 CPU 空闲 | 89.66% |
| 系统 I/O wait | 0.045% |
| B2 实际磁盘读取 | 0 字节 |
| B2 实际磁盘写入 | 约 52,553 字节/秒 |
| B2 major page faults | 0 |
| 系统 swap out | 0 页 |
| 系统 swap in | 37 页 |
| CPU / memory / I/O PSI avg10 | 均为 0 |

另做 30 秒 `nvidia-smi dmon` 和 `vmstat` 连续检查，GPU 利用率呈低占用与高占用脉冲，CPU 总空闲约 82–92%，I/O wait 为 0% 的显示精度，未见持续换页。初查内存 available 约 9.9 GiB；Swap 已使用 8.6 GiB，不能仅凭占用量认定正在发生换页瓶颈。

`nvidia-smi -q` 显示当前功率上限 110 W、默认上限 165 W。这是已有设置。本次检查时 SW Power Cap、HW/SW Thermal Slowdown、HW Slowdown 均未激活；样本功率低于上限。不能将累计限功率计数归因于本轮训练，也没有更改功率上限。

## Root cause

有运行数据支持的阶段瓶颈：单个主线程串行采集，周期性 Env 切换的耗时远大于更新阶段。硬件整体仍有余量，GPU 工作未形成连续的大批量计算。

对应源码路径（均为冻结训练源码，只读）：

1. Dreamer `embodied/run/fg_s3_v2_train.py:333`：`run_to()` 先暂停 stream 并调用 `collector.advance(end)`，随后才执行 4 次 `update()`；采集与更新没有重叠。
2. Gym `src/gym_trading_env/research/fg_s3_dagger_d0.py:117`：Driver 只有一个 Env，`parallel=False`。
3. Gym `src/gym_trading_env/research/fg_s3_v2_collector.py:73`：逐步调用 Driver，每次 `steps=1`。
4. Gym `src/gym_trading_env/envs/entry_dataset_env.py:80`：推进到 `chunk_bars=1000` 设置 `CHUNK_END`；`_rotate():93` 导出旧账户，切换合约时新建 `FixedDirectionEntryEnv`，随后导入账户状态。
5. Gym `src/gym_trading_env/envs/entry_env.py:45` 与 `:133`：构造 Env 调用 `_load_market()`，重新 `pd.read_csv`、创建 `CsvBarSource`、校验整份行情、遍历合约时间和生成特征。
6. Gym `src/gym_trading_env/utils/bar_source.py:102`：`CsvBarSource._build()` 调用 `build_market_features()`；真实 FG 有 `trading_date` 时 `entry_env.py:168` 还会按真实市场时钟计算特征。
7. Dreamer `embodied/jax/agent.py:221`、`:265`：policy 每次返回主机动作；同步训练指标配置要求拿到本次更新的结果后再继续。

未对正在运行的进程附加 Python/GPU profiler。因此，19.6 秒切换开销内部各函数所占百分比尚未测量；不能声称全部由某一个 `build_market_features` 调用造成。已确认的是账户切换 cycle 的耗时关联及其完整重建代码路径。

## Changes made

仅保存本报告和两份小型诊断 JSON；没有实施性能修改。

## Verification performed

- 核对真实进程、CUDA 设备、解析后的配置、selector、双仓库 SHA 和工作区状态。
- 读取实际 `cycles.jsonl`、`updates.jsonl`，最新完整更新样本 `opt/finite=1.0`，训练计数持续增长。
- 按真实 reset 增量对 cycle 分组，定位周期性约 19.6 秒采集开销。
- 完成 GPU/系统连续采样以及真实 PID 的线程、I/O、page fault 采样。
- 核对 Driver、Collector、合约调度、Env 构造、行情特征与 JAX 同步代码。

原始诊断数值：

- `/tmp/s3-b2-gpu-diagnostic-20261014.json`
- `/tmp/s3-b2-cycle-diagnostic-20261014.json`

## Risks / not verified

这是短时资源采样和过去 500 个 cycle 的日志分析，不能代表后续所有训练阶段。没有做逐函数 profiler、优化后吞吐对照或训练效果对照；没有将未验证的优化收益写成结论。

## Next recommended step

后续独立开发分支优先测量 Env 构造、特征生成、历史 prefix hash、账户导出/导入分别耗时；考虑缓存具有数据、配置和源码身份的不可变行情/特征，并保留账户轮换和恢复语义。以原始与优化版本相同动作/数据的观测、账户、Reward、Replay/Sidecar 对齐和恢复检查证明等价。

更大训练 batch 可以单独做吞吐研究，但会改变训练条件和样本预算；提高 updates_per_collect 会直接改变模型更新预算。均不应作为本次正式 Run 的现场调整。仅凭 GPU 未满不足以要求变更正式实验。

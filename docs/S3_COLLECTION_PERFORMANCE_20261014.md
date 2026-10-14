# S3 账户轮换性能修复与验证（2026-10-14）

## Summary

从 `integration/s3-next` 建立独立开发环境：

- 路径：`/home/v/Documents/work/research/fg-s3-collection-perf-dev/`
- 双仓库分支：`perf/s3-env-rotation`
- Gym 基线：`58708ceb9e10cc2a355511bdf3133710f4459e0e`
- Dreamer 基线：`d41bc80ba13b3dc3f5d2c32fcdc800613b2d0b4d`，本次没有源码修改。

修复真实市场时钟 Env 构造中的重复特征计算，并为轮换账户池增加有容量上限、按真实输入身份索引的进程内行情缓存。账户对象、账户恢复、调度顺序、Reward、Oracle、模型、Replay、训练预算和配置保持原有语义。

真实 FG809、FG901 TRAIN 的 4,000 步固定动作 CPU 对照：原基线 57.82 秒，修复后 23.83 秒，约快 **2.43 倍**。完整行情特征、逐步返回值和最终账户状态的哈希完全相同。再次访问 FG809 的 Env 构造从 17.535 秒降到 0.212 秒；包括账户导出/恢复的该次切换从 17.577 秒降到 0.290 秒。

这些是 CPU Env 对照结果，不是正式模型训练吞吐或 GPU 满载证明。运行中的 B2 冻结工作树、Run、进程与 W&B 均未调整；E-DATA 继续暂停。

## Evidence observed

先前只读诊断见 [S3_B2_GPU_DIAGNOSTIC_20261014.md](diagnostics/S3_B2_GPU_DIAGNOSTIC_20261014.md)。正式 B2 最后 500 个 cycle 中，采集占采集与更新时间之和的 85.55%；有账户切换的采集 cycle 平均 19.62 秒，没有切换的平均 1.56 秒。

本次在新 worktree 对真实 FG809 完整数据进行 CPU `cProfile`，共 28,264 行：

| 调用 | 次数 | 累计耗时 |
| --- | ---: | ---: |
| `FixedDirectionEntryEnv.__init__` | 1 | 27.245 秒 |
| `_load_market` | 1 | 26.767 秒 |
| `build_market_features` | 2 | 25.001 秒 |
| `_add_obs_features_inplace` | 2 | 23.410 秒 |

Profiler 增加了执行开销，表中耗时不直接与无 profiler 的对照时间相比。

修复前的回归测试实际失败：期望特征调用为 `[True]`（真实市场时钟一次），得到 `[False, True]`（旧时钟一次、真实时钟一次）。修复后通过。

## Root cause

`entry_dataset_env._rotate()` 每次切换合约都会新建 Env。原 `entry_env._load_market()` 先构造 `CsvBarSource`，后者计算旧时钟特征并创建 `MarketStore`；带 `trading_date` 的真实 FG 数据随后再次调用 `build_market_features(real_futures_clock=True)`，替换此前特征。第一份特征和 store 没有用于最终 Entry 观测。

此外，同一合约再次被调度时，原路径重复读取、规范化、校验和计算整段行情。正式训练先完成采集再更新，切换期间 GPU 无法收到后续更新工作。

## Changes made

### 单次真实时钟特征计算

在 `utils/bar_source.py` 抽出 `normalize_csv_frame()`，复用原 CSV schema 与时区规范化。`CsvBarSource` 的普通期货/FX 路径继续使用该公共函数。

`entry_env._load_market()` 对含 `trading_date` 的数据直接调用相同规范化，再执行原有的价格、成交量、OI、时间顺序、合约、日历、有效分钟等校验，以及原真实时钟特征生成。没有改变 `market_features.py`、价格参考、特征公式、18 维输出或 `[60,18]` 观测。

不含 `trading_date` 的既有路径继续使用 `CsvBarSource`。

### 有身份约束的行情缓存

`envs/entry_market_cache.py` 提供 `EntryMarketCache`。缓存只保存规范化行情、最终特征、bob 和有效分钟数组。

- 作用域为一个轮换账户池，容量为池成员数，使用 LRU 淘汰旧项；`close()` 清理。
- 键包含 Env 类型、完整 `config_hash`、真实 CSV 路径及当前文件字节 SHA256。
- 每次查询都重新哈希 CSV，配置/合约条款或文件字节变化会走原校验与计算路径。
- 哈希和构造前后检查设备、inode、大小、mtime、ctime 与路径解析；不稳定输入拒绝发布缓存。
- 缓存与新 Env 各持有独立的行情/特征拷贝，Env 的可变字段不会污染下一账户。
- 始终新建账户 Env，使用原 `export_state/import_state/reset`，不缓存账户、余额、交易、Reward 状态或随机数状态。
- 显式传入的 `df` 和 `continue_data()` 不从 CSV 缓存读取。
- 缓存不进入账户/Driver/FullCheckpoint 的序列化状态；冷缓存恢复重新生成行情。

`FGEntryDatasetEnv` 的轮换池默认启用缓存；`cache_market=False` 可在 CPU 对照中关闭。非轮换池的行为不变。`ObservedTrainingPool` 沿用该扩展，不修改 Collector、Oracle、Outcome 或 Replay。

## Verification performed

### CPU 回归与现有测试

新增缓存测试 6 项全部通过：

1. 真实时钟只计算一次特征。
2. 复用行情但创建独立拷贝和全新账户。
3. 重新检查 CSV 字节及配置/合约身份。
4. 构造期间源文件变化不发布缓存。
5. 显式输入帧绕过缓存，容量和清理有效。
6. 缓存/不缓存的 TP、SL、逐步状态及冷缓存恢复一致。

相关现有测试 **83 项通过**，用时 38.91 秒，包含账户银行、持仓和亏损恢复、S3 市场分钟、B2 Outcome、Driver/Replay/Sidecar 所有者与恢复，以及普通期货/FX CSV 特征。

执行命令（新 worktree 内）：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu DREAMER_RUN_DIR='' \
MPLCONFIGDIR=/tmp/s3-rotation-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=src:../dreamerv3 \
/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  tests/entry_v1/test_entry_market_cache.py

CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu DREAMER_RUN_DIR='' \
MPLCONFIGDIR=/tmp/s3-rotation-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=src:../dreamerv3 \
/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  tests/entry_v1/test_fg_abcd.py tests/entry_v1/test_fg_s3_v2_pool.py \
  tests/entry_v1/test_entry_contract.py tests/entry_v1/test_fg_single3_env.py \
  tests/entry_v1/test_s3_market_minute.py tests/unit/test_futures_sessions.py \
  tests/unit/test_build_market_fx_features.py -rA
```

自审另补充 1 项 benchmark 输入身份边界测试：带重复合约的签名 manifest 不能冒充两个不同请求合约，也不能创建输出目录，已通过（0.78 秒）。本次合计 **90 项 CPU 测试通过**。

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu DREAMER_RUN_DIR='' \
MPLCONFIGDIR=/tmp/s3-rotation-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=src:../dreamerv3 \
/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  tests/entry_v1/test_s3_collection_benchmark.py -rA
```

### 真实 TRAIN CPU 对照

独立工具：`tools/benchmark_s3_collection.py`。工具只执行原 Env 的 CPU 固定因果动作（当前 mask 允许且 cursor 为 23 的倍数时 ADD），不初始化模型、不读取 Oracle、不做梯度更新、不使用 GPU。

比较三种路径：Git 中固定原始 Env/调度器、仅单次特征计算、单次计算加缓存。仍使用原调度器、相同源配置、相同 seed、`chunk_bars=1000`。

| 路径 | 总耗时，含冷启动 | 再访问 FG809 的构造 | 再访问 FG809 的完整切换 |
| --- | ---: | ---: | ---: |
| 原基线 | 57.816 秒 | 17.535 秒 | 17.577 秒 |
| 单次特征计算 | 33.132 秒 | 9.545 秒 | 9.590 秒 |
| 加入缓存 | 23.832 秒 | 0.212 秒 | 0.290 秒 |

两合约缓存负载约 28.09 MiB（29,453,936 字节）。首次访问仍需完整计算：缓存版本 FG809 构造约 9.495 秒，FG901 约 10.014 秒。

三条路径的校验值完全一致：

- 4,000 步观测/Reward/终止/截断/info 和 reset 轨迹：`615bb184a5bf4ee3390216a7a383f0eaca2125171ca22feb0be4fcd3f937b4a9`
- 最终账户银行状态：`fe664b3523ba0f89abf011896b44a1e26b457878e224db5a5234babcd257d52c`
- FG809 完整 `[28264,18] float32` 特征：`1d68ac4467640ab857b0942e462e99dee9c54ad73b189b728df495bdbd8d5a64`
- FG901 完整 `[29416,18] float32` 特征：`e244eaa2f2d6888790c4d8c9805dfc93f16415a01029c9729f6c756ae4c09d30`

随后将缓存版本的 schema=2 财务状态恢复到独立冷缓存，继续推进 64 步；轨迹和账户状态与未中断版本完全一致。

实际结果：

`artifacts/fg_s3_v2/collection_perf_20261014/report.json`

报告 SHA256：`60e42b566e6a6c77f3e8f3304f11052f2e85fccc137872d283917aca9e815c3b`。

报告记录了原始/当前核心源码 SHA256、双仓库 HEAD、原始输入 SHA256、三个版本的耗时/市场特征身份与恢复结果。运行时实现尚未提交，因此报告中的 Gym HEAD 是基线；核心文件 SHA256 标识实际被测修复源码。最终提交可通过 `git rev-parse HEAD` 查看。

原始 selector、manifest、CSV、Env、策略和合约配置在对照前后 SHA256 均相同。正式 B2 双工作树检查仍干净，同一个训练 PID `4045558` 持续运行；GPU 计算进程检查只有原 B2。

复跑时指定新的输出目录，已有结果不会覆盖：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu DREAMER_RUN_DIR='' \
MPLCONFIGDIR=/tmp/s3-rotation-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=src:../dreamerv3 \
/home/v/miniconda3/envs/dreamerv3/bin/python tools/benchmark_s3_collection.py \
  --selector /home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/selector.yaml \
  --baseline-ref 58708ceb9e10cc2a355511bdf3133710f4459e0e \
  --contracts CZCE.FG809 CZCE.FG901 --steps 4000 --chunk-bars 1000 \
  --restore-steps 64 --output artifacts/fg_s3_v2/collection_perf_review
```

输出只允许位于本 worktree 的 `artifacts/` 或 `/tmp`，拒绝已有目录、写账户 `state_path` 的配置，以及非 TRAIN/重复/缺少合约的输入。所有预算为 CPU 环境推进，与正式训练或模型更新无关。

## Risks / not verified

- 没有修改或重启当前正式 B2；当前 Run 不会自动使用本分支优化。
- 没有初始化 GPU 模型、做联合更新吞吐对照或测量优化后 GPU 利用率；不能声称 GPU 会达到 100%。单 Env、小 batch 与串行调度仍存在。
- 本次真实对照覆盖两个 TRAIN 合约，不是完整 18 合约的长期训练性能验收；已有 B2 所有者/恢复测试覆盖原 Collector/Outcome/Replay 接口。
- 本次验证为 Env/账户与 CPU Replay/Sidecar 恢复；不宣称跨版本完整模型 Checkpoint 可直接恢复。
- 首次访问合约和冷恢复仍需构造与生成特征；缓存不是磁盘产物，不要求历史归档保存它。

## Next recommended step

将本分支提交审查并合入 `integration/s3-next` 后，按原准备入口生成新的代码身份快照，未来在新的冻结训练工作树使用。不能把修复复制到正在运行的 B2 或旧 prepared/Run 中绕过源码身份校验。正式训练或恢复另行明确授权；本次交付不启动训练。

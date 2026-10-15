# 一分钟计算优化梳理与 E-Data 复用边界

核查日期：2026-10-14。Gym 基线 `03a625c1661546b9183b9ceeb24119b516a87459`；Dreamer 基线 `d41bc80ba13b3dc3f5d2c32fcdc800613b2d0b4d`。本文件先梳理已有实现；E-Data 的优化据此复用，不新增训练缓存框架。

## 已有一分钟路径

| 层次 | 实际代码 | 方法与约束 |
|---|---|---|
| 读取和规范化 | `utils/bar_source.py:normalize_csv_frame`、`utils/ohlcvi_contract.py:normalize_ohlcvi`、`utils/time_contract.py:ensure_feature_tz_index` | 统一 OHLCVI 类型及显式时区；正式 Entry 在调用前检查时间唯一性和数字完整性。原通用 normalizer 会保留重复的最后一条、将无效 VI 填0，不能越过 Entry/E-Data 前置断言直接用于异常数据。 |
| 计算一次 | `envs/entry_env.py:_load_market` | 含 `trading_date` 的真实数据直接规范化，再调用原 `build_market_features(real_futures_clock=True)` 一次。此前先创建旧 CsvBarSource/store、随后又算真实时钟特征的重复路径已去掉。 |
| 数学计算 | `utils/market_features.py` | 沿用 pandas/NumPy 分组累计 VWAP、高低值、EMA/rolling 等原公式。最终输入取固定18列并转成 float32。源码仍会计算不进入18维输入的扩展列；已有优化没有更改这一数学实现。 |
| 重访合约 | `envs/entry_market_cache.py:EntryMarketCache` | 一个轮换池内的有界 LRU；键含 Env类型、完整配置身份、真实 CSV 路径和当前字节 SHA256。每次重访重新哈希，构造前后检查 device/inode/size/mtime/ctime；输入变化重新校验与计算。 |
| 缓存内容 | `_MarketSnapshot.capture/restore` | 仅行情、最终特征、bob、有效分钟，拷贝恢复；不保存账户、交易、Reward、随机状态，不进入 FullCheckpoint。显式 df/续传数据绕过 CSV 缓存。 |
| 每步窗口 | `utils/s3_market_clock.py:MarketClock`、`entry_env.py:_get_obs` | 初始化时累计有效分钟 prefix 和 rows；每步直接从有效行索引取末60根，再切已算好的特征数组。零成交的真实一分钟仍计数，零时长标记不计数；不每步扫描整段历史或重新计算特征。 |
| 调度恢复 | `envs/entry_dataset_env.py:_new_env/_rotate` | 复用行情缓存，但创建/恢复原账户 Env，保留原持仓与恢复语义。 |

## 已有性能证据

原 CPU 对照报告：`artifacts/fg_s3_v2/collection_perf_20261014/report.json`，位于 `fg-s3-collection-perf-dev/gym-trading-env`。

该报告 SHA256：`60e42b566e6a6c77f3e8f3304f11052f2e85fccc137872d283917aca9e815c3b`。

- FG809/FG901、4,000 次行情推进：原路径57.816秒，去重后的单次计算33.132秒，行情缓存23.832秒，原路径与缓存路径约2.426倍。
- FG809 重访构造17.535秒降至0.212秒；完整切换17.577秒降至0.290秒。
- 三条路径的观测/Reward/info轨迹、完整市场特征和最终账户状态哈希相同；冷缓存恢复后64步也相同。
- 这些是已有一分钟 CPU 证据，不能作为 E-Data 或 GPU 提速的实测结果。

详情见仓库已有 [S3_COLLECTION_PERFORMANCE_20261014.md](../S3_COLLECTION_PERFORMANCE_20261014.md)。

## E-Data 采用的复用方式

1. 直接加载未修改的原特征计算器；保留其18维数组，仅用原始辅助量重算 E 所需的六个同锚通道。原公式不重写。
2. 正常唯一且完整的输入使用原 OHLCVI/时区规范化 helper；重复、非数值等异常先保留证据并拒收，避免通用 helper 抹掉异常。
3. 直接复用原 `MarketClock` 的 prefix/rows/window；E snapshot 从缓存数组取60行，不扫描全历史，不重算一分钟特征。
4. 复用原行情缓存的稳定文件身份检查。每个只读合约准备一次，后续 E snapshot 复用这份数据；不另建 LRU 或磁盘特征缓存。现有 Env 账户缓存继续留在原系统。
5. 新增计算仅包括严格15分钟分桶、可见性筛选、24根上下文及六通道同锚变换。延迟到达采用当时可见前缀，不能用吞吐优化牺牲因果性。
6. 原纯 helper 通过隔离导入载入：只重定向这些文件之间的 import，不改函数体；记录原文件 SHA256。独立 CPU venv 无需引入 Gym 包初始化中的 DB、Torch 或训练依赖。

## 复用后的验收

- 原18维必须与真实 Env 逐字节一致；原来源身份、顺序和 mask 保留。
- 普通 CSV 合约准备只调用一次原特征计算器；反复取 E snapshot 不新增调用。
- 原 `MarketClock` 控制有效分钟窗口，包含零成交、排除零时长标记。
- E 优化前后在同一真实源数据和决策集合上比较数组及逐时点元数据；结果一致后才报告性能变化。
- 全部 F01–F11 和真实 TRAIN 分布验证继续执行；正式训练、Replay、Archive、W&B 均不改动。

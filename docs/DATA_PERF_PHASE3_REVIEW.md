# DATA-PERF Phase 3：纯行情初始化复用

状态：定向开发与真实小样验收完成；用户已于 2026-10-20 授权合并到 `integration/s3-next`。最终合并状态见 `DATA_PERF_PHASE23_INTEGRATION_REPORT.md`。

## 已观察的证据

- 起点为已验收的 Phase 2：Gym `2dd45283567398e782cbcd0a958e9be873386921`，Dreamer `b7a693b3c7b0b45eeecdec954818be42d40e67b6`。
- `initialization_baseline_01` 固定真实 TRAIN 107 次查询，51 次原生反事实交易、39 个严格 L2 有效标签；完整标签及固定原账户的 root/clone 公共 SHA 全部匹配。
- 16 个 worker 的初始化 profile 合计约 629 CPU 秒，其中 16 次 `_load_market` 约 444 CPU 秒。`cProfile` 有开销；这些数值只用于定位重复工作，不是端到端加速证据。
- 同一 CSV、配置、E 来源身份，在不同 spawn worker 重复做特征构造、HTF 分桶和行情元数据准备。原有 `EntryMarketCache` 只覆盖进程内重复构造。

## 实现与等价边界

候选 Gym 提交：`b10b764d2d1dc9ffb24c6450cef67ba0a378ec53`。Dreamer 无修改。

扩展既有 `EntryMarketCache`，通过 `export_seed()` 生成不可变 bytes 与 SHA256 的内部传输对象。父进程、账户进程、标签进程复用同一份已验证纯行情的计算结果。每次 cache load 仍重新计算实际 CSV SHA，并与 E admission 核对；配置、Env 类型、路径和 E observation identity 都仍在缓存键中。

每个 Env 从缓存恢复独立拥有的 raw/features/E MarketData；金融账户、RNG、Ledger、Reward、Teacher 标签和 Strict Oracle 值不进入 seed。完整金融恢复、原始公共 JSON SHA、每查询来源监控、原生 L1/L2 计算均保持原路径。延迟行情仍只在可见前缀调用原有 `build_market_features`，参数不变。局部 lambda 改成模块函数，仅为支持标准 spawn 序列化。

Generate 与独立 G3 分别从原始来源重新准备自己的缓存。G3 不读取 Generate 的行情 seed 或 Oracle 缓存作为预期答案。内部 pickle 仅用于同一冻结 Producer 的可信进程之间传输，不增加外部文件加载接口，不写入 Corpus 或账户 Checkpoint。

六个账户及标签 worker 分配规则保持不变。

## 调试与验证收据

原始收据保存在本工作区的 `../artifacts/`，每次运行记录双仓 SHA、dirty、完整参数、CPU 环境和运行前后源码哈希。

| 收据 | 实际结果 | 说明 |
| --- | --- | --- |
| `market_seed_targeted_01` | 25 PASS / 3 FAIL | 首轮 seed 无法 pickle 局部 lambda；保留原始失败，不属于基线失败 |
| `market_seed_targeted_02` | 28 PASS / 0 FAIL / 0 SKIP | 独立所有权、公共冷恢复、真实 spawn 标签及完整账户 SHA；此组在早期 `/data/research` 工作树执行 |
| `market_seed_causal_01` | 28 PASS / 0 FAIL / 0 SKIP | Home 工作树执行；增加 delayed-arrival seed 边界，结合既有市场缓存、MACD/日历测试；与上一组有交叠，不能相加为唯一用例数 |
| `short_market_seed_01` | PASS | 固定 DEV128，774 条记录；生成 134.246s，全流程 155.798s；Native G3 NOT_RUN，不能作为正式 DATA 认证 |
| `short_seed_equivalence_01` | PASS | 对照 Phase 2 的 774 条完整金融/教师/原生数组/最终账户语义等价 |
| `query_market_seed_01` | 预期来源身份拒绝 | 旧 E collection 的 schema 绑定旧 `entry_market_cache.py` 字节；没有放宽校验、迁移或重新签旧产物 |
| `query_market_seed_02` | PASS | 改用本轮新生成并通过独立 G3 的 E collection；107 次完整查询、51 次原生交易、39 个严格有效标签；公共 root/clone SHA 对本轮同一 profile 的固定黄金账户逐字匹配 |
| `long_market_seed_01` | PASS | 两日 4,146 条记录；生成 192.610s，独立 G3 175.288s，Reader 14.967s；全流程 399.823s |
| `long_seed_equivalence_01` | PASS | 对照已验收 Phase 2 的全部 4,146 条记录，完整有理数、金融状态、教师语义与原生数组相同 |
| `shared_market_seed_final_01` | 35 PASS / 0 FAIL / 0 SKIP | 原生 E / WP03 admission 与规定的三个共享回归文件；实际执行清单见 XML，优化器用例由准确命令中的 `-k` 显式排除；不执行优化器更新 |

早期 `/data/research` 候选的六份源文件已逐字节核对并转回 `~/Documents/work/research`。用户随后撤销新目录偏好；后续开发和新 worktree 均回到 Home 工作区。已有冻结目录和收据不迁移、不删除。

## 性能结果、自查与边界

相对 Phase 2，生成 225.624→192.610s，独立 G3 205.976→175.288s，全流程 464.315→399.823s（1.161 倍，时间减少 13.89%）。相对最初 Phase 1 的 907.330s，全流程为 2.269 倍；逐查询来源检查继续开启。此次改善主要减少一次性的多进程行情准备，并不是按合约长度持续保持固定倍数。

初始化诊断：基线实际初始化 16 个 worker，instrumented CPU 629.441s；候选池容量 18，实际初始化 14 个 worker，CPU 153.620s。不同实际进程数不能直接作同工作量加速倍数；这些数值只定位热点。候选查询 CPU 10.344s。最终诊断工具新增字段，明确区分池容量与实际初始化数；此修正不改生产运行源码，不重新伪造已有收据。

新的剩余热点为冷恢复时的 MACD 前缀重建、公共账户恢复及市场前缀指纹准备。累计时间相互包含，不能相加当作独立份额。本轮不继续修改，以先完成用户要求的集成。静态自查确认：纯行情种子无账户/教师闭包；缓存恢复仍复制；源码与 CSV 身份每次验证；源账户和 clone 不共享可变数据；G3 从自身原始来源构造缓存；金融步骤与 L1/L2 算法均未改变。14 个真实标签 worker 的缓存均为 2 hits / 0 misses。

原始 generated manifest 保持 `COLLECTED_UNVERIFIED`，独立 G3 与 Reader 是附加的认证证据。完整记录与身份锁见 `docs/receipts/data_perf_phase3_20261020/`，冻结 Producer Pin 见 `docs/locks/DATA_PERF_PHASE3_PRODUCER_PIN_20261020.json`。

行为组合的可配置化是用户另行提出的后续功能，不混入本次等价性能优化；当前默认六行为及其概率/等待规则保持原样。

## 尚未验证

全 18 合约完整主力期、合约前中段长精确后缀的生产耗时、GPU、实际优化器更新、正式 F1/F2/F3 训练、TRAIN01 冻结生产修改：NOT_RUN。共享集合通过不代表全项目通过。粗估工期见 `DATA_PERF_PHASE3_TRAIN18_ESTIMATE.md`。

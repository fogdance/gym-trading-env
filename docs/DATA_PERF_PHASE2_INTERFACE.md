# DATA-PERF 第二阶段：内部状态接口与等价边界

第一阶段审查已收尾。第二阶段代码在独立分支 `feat/data-perf-phase2-20261020`。
生产候选冻结 Gym `2dd45283567398e782cbcd0a958e9be873386921`、Dreamer
`b7a693b3c7b0b45eeecdec954818be42d40e67b6`；没有合并到 integration/s3-next。

## 公共接口

`FixedDirectionEntryEnv.export_state()` 仍返回完整、独立、可修改的 JSON 快照。
`import_state()` 仍完成原有 JSON 规范化、行情前缀、账户、Ledger、Reward、RNG、
持仓时钟及 E Observation 核验。公共 Schema、排序 JSON 的 SHA256、Replay 字段、
Teacher 隔离、L2 精确有理数及 tie→WAIT 规则均保持。

普通 Env、普通 Replay 默认继续使用可变 list 历史。内部表示不会写入 FullCheckpoint，
也不替代 Corpus/Teacher/Observation 身份。旧 Run 不修改、重签或伪装恢复。

## 内部调用链

`utils/native_account.py` 的 `NativeHistory` 持有不可变的 256 行块和所属账户的尾段。
块中只存不可变基本值或 JSON bytes。查询封存尾段后产生 `NativeHistoryVersion`；
clone 共享冻结块，复制自己的尾段。历史读访问只读，旧行修改、替换、删除明确抛错。
需要可修改历史时，调用公共 `export_state()` 取得独立副本。

`NativeAccountSnapshotView` 保存来源、账户实例和 account_sequence、游标、历史版本、
行数、内容承诺以及独立热状态 JSON bytes。来源、账户或游标不符时拒绝恢复。
`_restore_private_state(view, source_env=...)` 最终执行同一 `_restore_account_state()`，
没有跳过金融恢复校验。Ledger、positions、fills、Reward、RNG 等热对象从 JSON 重建。

`view.digest()` 按原始排序 JSON 顺序输入全部历史 bytes，输出原公共 SHA256；
私有内容承诺用于校验所有权和版本，不替换公共 SHA。同一冻结 view 可以缓存该 SHA。
新的热状态或追加产生新的 view。公共可变 list 仍走原有前缀比较和完整重算路径，
包括 `-0.0/0.0`、bool/int、非标准行、历史修改、删除和 rewind。

`NativeLabelPool` 每个有序 worker 拥有自己的初始历史版本和追加尾段。进程间传递
热状态和独立 JSON 增量，校验 base_count、实际 mask 和 Driver/Reconcile 的完整 SHA。
`begin_stream()` 恢复该账户初始版本，不能拼接不同账户或乱序增量。

Corpus Driver 与 E 原生 reconcile 显式启用 `_enable_private_history()`。
公共冷恢复仍执行完整 export/import 和完整状态比较；恢复 CHUNK_END 后仍按原流程
显式继续。新账户生成独立 epoch。这个选择不改变普通 Env 的默认行为。

## 来源监视与验收

正式 Producer 的 `source_guard_interval=1`。每次查询逐文件检查 stat 身份；绝对普通
路径共享查询前后的父目录检查，符号链接、相对路径或目录变化回到原 strict resolve
检查。SHA-on-open、批次 SHA、close guard、实际行情相等、交易条款及 mask 检查保留。
metadata 监视不冒充逐查询全文 SHA；最终独立 G3 保留完整来源哈希核验。

Generate 产物仍是 `COLLECTED_UNVERIFIED`。本地 reconcile 核验所有原生交易和
Observation；DATA 发布必须经过新进程、独立重建 Oracle/精确后缀的 Native G3。
外层还要求六行为的 PASS、非零 records、全部标签核验数量和实际 replay hash 一一对应。
分块来源监视、缺 G3、部分 G3、失败或错误 Producer 均不能取得 DATA-ready 收据。
开发 Reader 可以做完整性诊断，但不因此获得训练准入。

## 证据与限制

固定旧 Git `b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46` 的标量 Oracle 和 Observed
Wrapper 作为独立参考。已有 107 个真实账户查询对照；定向测试覆盖公开/私有恢复、
逐步动作、独立所有权、篡改、账户切换和实际 spawn worker。最终速度与 G3/Reader
状态以 `DATA_PERF_PHASE2_RESULT.md` 和原始终态收据为准。

本轮没有改变六账户调度上限，没有重写 MACD 历史加载器，没有启用 GPU/优化器/正式
训练，没有修改 TRAIN01 冻结源码及数据。全 18 合约正式生产：NOT_RUN。

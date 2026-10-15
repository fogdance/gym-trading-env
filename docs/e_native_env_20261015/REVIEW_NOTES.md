# 补丁接入 Review 与实际修复

实际基线与包内重建基线不同：保留本机所有 E Python 模块源码指纹、schema 校验、既有受限验收流程；只应用 Dreamer 的训练接线 hunk。原包、校验清单及原始日志另行归档，不使用强制 apply/reset。

## 已复现的问题

1. 原包的 Native Env 仍调用 `MarketData.normalize_bars`，并建立自己的有效分钟时钟；新增“禁止重复规范化”回归失败。修复为原 `_load_market` 一次计算原特征后，`MarketData.from_env` 使用同一规范化行情、已计算辅助特征、bob 索引和 `MarketClock`。E 仅附加完整桶/可见性元数据和已确认六通道公式；不重新解析 CSV、不重做原18维。
2. 原包的单合约 `Observed` 包装没有导出 raw Env 的 `e_snapshot`，Collector 初始化报 `Native E Env lacks snapshot provenance`。为原 observed adapter 增加只读转发；正式 Collector 接收 metadata-only `e_profile`，无 `EObserved`。
3. 原包把 E 数据身份混入原 `config_hash/observation_identity`，原 Oracle 实测拒绝 `RULE_OR_MARKET_IDENTITY_MISMATCH`。保留原金融/教师身份，新增独立 `e_observation_identity`（完整E观察合同+来源身份），缓存键显式包含该身份。原 Oracle、Reward、撮合及账户状态算法文件不改；Run/Checkpoint 继续由 E binding、完整代码和观察合同认证。该处是针对实际失败的接入修复，不采用原包“修改旧金融身份”的实现。
4. E-Data 原独立依赖测试与真实 Env 同进程运行时，错误检查了 Env 预先导入的 Gym 模块。改为子进程执行完整独立 E 构建并核对无 GPU/Gym/W&B/DB 导入，仍保留原依赖隔离要求。

## 迭代中保留的失败

- 第一次组合运行：35 PASS / 1 FAIL（独立依赖断言被真实Env导入污染）。
- 新增接入回归修复前：2 PASS / 3 FAIL，对应上述重复规范化、原生快照转发、Oracle身份。
- 复用接口第一轮：bob 原列是无时区字符串，实测报 tz-naive/tz-aware 相减；改为直接复用 Env 已规范化的 bobs。
- 第二轮：原生视图漏传 `_legal`；改为直接复用市场时钟现有 valid 数组。
- 第三轮：4 PASS / 1 FAIL，账户比较的两个 Env 使用了不同随机初始化；测试改为同 seed reset 后比较完整状态。Oracle 认证已通过，未放宽算法检查。

最终合同测试：48 PASS / 0 FAIL / 0 SKIP；旧 Single3/Replay/V2：27 PASS / 0 FAIL / 0 SKIP，1项GPU测试显式未选择。本阶段仍为正式训练 NO-GO。真实全合约和轮换证据完成后写入 REPORT。

## 用户后续澄清的边界

遵循轻量目录规范；本轮是已有功能融合，不做目录迁移或 MarketStore 重设计。`e_data_v1` 保留为原数学/研究审计代码，Env 消费已计算行情与特征，正式路径不再把它用作第二个行情加载器。新代码不增加模型功能，不改变 Encoder、RSSM、Actor Loss、Oracle、Reward 或 FullCheckpoint。

## 真实 TRAIN 全量证据与后续入口修复

- 新 collection `e63e761cc90411a6075c7ebb0056f36557427703c608decdf62fe642fbb2ab29`，schema `1d085a5db76b9ea7162bc90bb07d992dda604e412d0f896f75efbe858071c210`。
- `real_native_all_18/native_validation.json`：18合约、488160行、全原18维矩阵、逐帧原三个Observation、E两个数组、原step结果和最终完整金融状态全部精确匹配；每合约一次CSV/一次feature build、复用原MarketClock，54个前缀重建/未来污染检查通过。仅六个确认的重锚定通道出现变化。15workers，412.89秒；最大单进程RSS408.62MiB。
- 原 reader 白名单实测拒绝新的 Env 文件。新增一个四文件精确 Native E bundle，保留旧 bundle、原 producer认证和未知修改拒绝。旧B2的entry_env语义校验不放宽。新增bundle依据全量对照及实际diff；Oracle仍核对原数据、funds certificate和suffix DP。
- 旧真实CPU prepare日志确认为 `ModuleNotFoundError: e_data_v1`：统一CLI没有将Gym根目录加入sys.path。补齐根目录，保留原src与兄弟Dreamer优先级；正式路径继续复用同一入口。
- 真实native pool测试第一轮1PASS/1FAIL：原Oracle和教师标签、2200步金融状态、跨合约、冷Driver/Replay/Sidecar、2250步继续与下一随机batch已精确相等；末尾“随机batch必须有有效15m”断言失败。原Replay文件实测1574行有效，随机batch可落在20桶预热区。改为直接验证持久化Replay有效15m覆盖，保留随机冷恢复相等断言，不更改数据/采样/预热公式。第二轮结果另记。
- Source/Composition/观察/来源身份/受限准入合同重跑47项通过（原45项另加两项准入预算边界）。正式E仍waiting_e，正式1M未启动。

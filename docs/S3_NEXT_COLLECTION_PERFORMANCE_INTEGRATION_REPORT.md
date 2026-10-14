# S3 Next 账户轮换性能优化集成

日期：2026-10-14。用户明确要求合并已经验收的性能优化；本次未启动或重启正式训练，E-DATA 继续暂停。

## 来源与合并结果

- 目标 Gym：`integration/s3-next`，合并前 `e7df18931bbb384282d602de8009208ca33d1de1`，已包含 Archive V1。
- 性能来源：`c14fcdc257d84f2c9859030859898baff71b8bd0`，`perf/s3-env-rotation`。
- 普通 merge 提交：`03a72b91958c4de9cb5f98c4a8444e8df6c6d8e3`，无冲突，两个父提交分别为目标与性能来源。
- Dreamer：`d41bc80ba13b3dc3f5d2c32fcdc800613b2d0b4d`，两分支相同，没有需要合并的 Dreamer 修改。
- 集成目录：`/home/v/Documents/work/research/fg-s3-next-integration/{gym-trading-env,dreamerv3}`。

本页和总报告入口为合并后追加的文档。最终 HEAD、双仓状态和版本锁位于工作区根目录 `baseline-lock.json`，避免文档提交相互引用 SHA 的循环。

## 已集成的行为

包含性能分支全部 13 个文件：移除真实时钟 Entry Env 构造时被丢弃的旧时钟特征计算；轮换账户池缓存按配置及 CSV 字节身份验证的行情与特征；每次仍新建账户、沿用原导出/恢复；提供 CPU 对照工具、回归测试和原始诊断证据。

缓存容量受池成员数约束，不进入财务 Checkpoint，显式输入帧/继续数据绕过 CSV 缓存。训练配置、模型、Reward、Oracle、Collector、Replay 与 W&B 接口未修改。本轮保留先前 Archive 集成，不重复移植或退回旧 Tracker 解析器。

详细实现与此前真实数据对照见 [S3_COLLECTION_PERFORMANCE_20261014.md](S3_COLLECTION_PERFORMANCE_20261014.md)。该对照属于性能分支的实际验收；本轮没有重复执行 4,000 步对照，而是逐文件确认合并后的核心源码 SHA256 与被测源码一致。

## 本轮实际验证

| 验证 | 合并后结果 |
| --- | --- |
| 缓存、账户、市场分钟、B2 Driver/Replay/Sidecar、普通期货/FX 特征 | **90 PASS**，44.31 秒 |
| Archive、Tracker、V2 评价、预测评价 | **48 PASS**，21.193 秒，CPU/Mock/真实 Offline SDK |
| 已验收真实数据对照的 4 份核心源码哈希 | 全部相同 |
| B2 冻结源码与不可变 Run 文件 | 200 份，0 个哈希变化 |
| B2 训练进程 | PID 4045558、15:58:35 开始的同一进程继续运行 |
| Git 差异空白检查 | PASS |

两组测试的执行目录均为当前公共集成 Gym，测试来源提交为上述 merge 提交。最终文档跟进不改变被测源码。

本轮日志及结构化收据位于公共集成根目录：

```text
validation/collection_perf_20261014/
  collection.log
  collection.xml
  collection.json
  archive_tracking.log
  archive_tracking.json
  protection.json
  baseline-before.json
```

`collection.json` 和 `archive_tracking.json` 包含完整 argv、cwd、实际被测 SHA、退出码、耗时与 stdout SHA256。所有测试屏蔽 GPU；Offline SDK 仅执行测试数据的本地写入和 Mock 传输，不发布线上里程碑。

测试集合与性能分支验收相同；额外执行 Archive/W&B 四个既有测试模块，确认在新组合中继续兼容。早先基线的 20 个 CPU 小模型与其他合同验收保留为历史证据，没有计入本轮的 **138 PASS**。

## 版本锁与保护

在两个仓库最终提交完成后，更新工作区外置 `baseline-lock.json`。上一份锁的原始字节保存在 `validation/collection_perf_20261014/baseline-before.json`；旧测试、旧归档身份和验收范围可独立核对，不把历史 PASS 宣称为本轮重新执行。

新锁记录最终双仓完整 SHA/clean 状态、merge/来源提交、新增能力、两组实测验证及证据 SHA256、被保留的真实数据对照身份与未执行事项。

正式 B2 原 Gym/Dreamer 提交仍为 `3970e6ec69f67fb2d02f30456a077ea5a31fd05d` / `5fd4ebdeb590fd8433338300bd82c7f2f520bf76`。没有覆盖其源码、配置、Replay、Sidecar、Checkpoint 或评价，也没有改变旧 Tag、main、性能分支或历史研究分支。

## 限制与后续

- 本轮无 GPU 模型训练、联合更新吞吐测试或优化后 GPU 利用率测量；不声称 GPU 满载。
- 无在线 W&B 发布、新 L2/L3 恢复验收、E 开发或正式 Run 迁移。
- 当前 B2 仍运行原冻结版本；本次 merge 只更新公共研发基线。
- 后续功能从新版本锁指定的 `integration/s3-next` 双仓版本建立；正式训练须使用新准备快照和独立冻结工作树，不绕过旧 Run 的源码身份检查。

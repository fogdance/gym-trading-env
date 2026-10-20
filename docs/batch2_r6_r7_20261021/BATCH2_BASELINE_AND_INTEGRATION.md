# Batch 2 双仓基线与集成

## 已核实起点

任务书归档中的“第一批尚未合入”不适用于这台开发机的实际起点。读取当前双仓源码与 Git ancestry 后，确认下列干净 HEAD 已包含第一批；未重复 merge 旧归档分支。

| 仓 | 本轮起始 integration HEAD | 已包含的第一批 feature HEAD |
|---|---|---|
| Gym | `def3e0e2bf99898c0443ae7047e2a1b9f18ebebc` | `20cdedf4b1d01ab7c2f9dffd979a01d80c9f5067` |
| Dreamer | `b2287a9d892e932339857ebb36f75499163a3046` | `4c4a36636bfa1a3f6a02c6df7f7d164ca8ec7036` |

原始起点与 Corpus 文件哈希见 `receipts/BASELINE.json`；实际 ancestry 命令、stdout 与退出码见 `receipts/BASELINE_ANCESTRY_AND_R6_INTEGRATION.json`。本次只读取本任务涉及的已知提交和 worktree，没有扫描其他工作分支。配套任务书、接口 Review、Roadmap V16 的哈希记录于最终只读资产审计。

开发目录为 `/home/v/Documents/work/research/fg-s3-batch2-r6-r7-20261021/{gym-trading-env,dreamerv3}`，双仓均为 `feat/batch2-r6-r7-20261021`。没有在 WP07、冻结 Producer 或历史 Run 中开发。

## R6 独立交付

| 仓 | R6 独立代码/测试提交 | 已执行普通 merge 到 integration |
|---|---|---|
| Gym | `5119f1deb8fc17676b85454c20ce7793e190876d`；随后 `7f80221` 补收据 | `710a1332d5f9ebc1993cd7c3a5002b0d732d3533` |
| Dreamer | `cbc1343971786caec979cead3025431690f1591b` | `a5631ffadd4815efcd6caf4c9026a274ea3d6ab7` |

R6 独立 CPU 参数接续/来源回归为 73 PASS，第一批兼容/来源 pin/严格 CP 和共享回归为 83 PASS，均 exit 0；详情见 [R6 Review](R6_IMPLEMENTATION_REVIEW.md)。R7 后对当前冻结代码再次运行阶段 handoff，不把前述旧 SHA 的 PASS 自动算作 R7 PASS。

## R7 最终测试代码

| 仓 | 干净冻结代码 HEAD |
|---|---|
| Gym | `4c881d39d9a482542534ee678e53787547afae89` |
| Dreamer | `a6aa3e649ca36022b767e2d89d0392868953f5e9` |

实际 sibling worktree 为本次工作区下的 `rounds/final-v5/`。`FINAL_V5_*` 记录该双 SHA、clean 状态、全部收集用例、原始命令与实际退出码；不同轮次及 dirty DEV 结果分别保留。最终工程范围和计数见 `receipts/FINAL_ACCEPTANCE.json`，完整索引见 [TEST_EVIDENCE_INDEX.md](TEST_EVIDENCE_INDEX.md)。

R7 的事务、版本图、Run 接线、基础覆盖绑定和重复恢复拒绝各有独立提交。此处的代码 SHA 与随后报告提交、普通 merge 产生的交付 HEAD 不相同；合并只改变历史或文档时，用实际 tree/运行源码哈希证明测试代码未改变。

## 冻结资产保护

`receipts/FINAL_PROTECTED_ASSET_AND_SOURCE_AUDIT.json` 逐文件核对两份既有 Corpus 共 176 个文件，文件集合和全部 SHA-256 与起点一致。仅做只读哈希，没有修改字节、重签或重新生成。

正在正式生产所用的冻结目录为 `/home/v/Documents/work/research/fg-s3-formal-corpus-optimized-20261020/`：

- Gym `8619f04b6f996f4c4d7f169486e4d10c25a3b2b9`，clean。
- Dreamer `b7a693b3c7b0b45eeecdec954818be42d40e67b6`，clean。
- 冻结 `PRODUCER_PIN.json` SHA-256：`d87082946aaa9db7408d68b2c2d147bf6cf1d28f27c1afb36d07c32b59239425`。

未打开或改写正在生成的正式输出目录、既有训练 Run、Checkpoint 或日志。工程测试和压力测试的新输出全部在本次独立工作区 artifacts / docs 或 pytest 专用临时目录，未写 `/data`。CPU 合成小模型验证不代表获准启动长期正式训练。

源文件审计确认金融 Env、Reward、Teacher、L1/L2、独立 G3、Producer/验证器 pin、ABI 3.1.0 和 `embodied/run/fg_checkpoint.py` 未修改。原有运行源码中仅本任务允许的 Reader/Adapter/Multi 与 Dreamer stream/lifecycle/transfer 文件发生变化；新增两个模块分别承载事务和状态版本。逐文件前后摘要见审计 JSON。

## 交付身份规则

代码合并的真实命令、退出码、合并前后双 SHA、tree 一致性以及合并后的定向验证由 `receipts/INTEGRATION_COMMANDS_V5.json` 与 `receipts/MERGE_VALIDATION_V5.json` 给出。报告提交可以位于代码合并之后；最终交付双仓 HEAD 由两个仓库之外的 `/home/v/Documents/work/research/fg-s3-next-integration/baseline-lock.json` 锁定，以免互相写入 SHA 造成循环提交。

原工作区旧版本锁另存为带日期的 byte-identical 历史副本，不把其中旧能力或旧测试变成当前 PASS。没有修改 main、重置历史、强推、清理分支/worktree 或发布训练结论。真实 TRAIN 集成训练、正式 G3、GPU、长期训练为 **NOT_RUN**；TEST 为 **NOT_READ**。

## 最终代码合并与复验

| 仓 | 源码普通合并 HEAD |
|---|---|
| dreamerv3 | `a6a6ad9e18196f799c4517f370dd26a4ca68aaf5` |
| gym-trading-env | `1073542163849540ea4ac9a626310cee2f3a8932` |

两仓 complete tree 与最终冻结测试树逐字节相同。合并后独立复验 70 项 PASS，exit 0，FAIL/ERROR/SKIP 均为 0；原始命令及结果见 `receipts/MERGE_VALIDATION_V5.*`。这些用例是 330 项中的重复执行，不增加不重复用例数。

以上是代码合并 HEAD。随后只交付 docs 的提交与 merge，最终交付双仓 HEAD 另由工作区 `baseline-lock.json` 锁定，并用运行源码字节一致性证据关联到本表。

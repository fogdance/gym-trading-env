# CORPUS_COVERAGE_REPORT

**当前正式教材已完成0/18批。** 用户要求覆盖原TRAIN池全部18合约的完整主力资格期。来源只读校验已完成18/18，来源合计1455个合约官方日、488160行；这些数字不是已生产Replay、已验收课程或有效市场分钟数。

旧教材按用户要求忽略并删除：cpu20_v1失败尝试及cpu20_v2旧B01的教材和对应收据四目录均已删除、核对不存在。无完整备份，冻结Producer源码/pin及原始TRAIN行情保留。[删除记录](docs/formal_train_corpus_20261018/evidence/OLD_CORPUS_DELETION.json)。历史B01审计不再计入当前教材覆盖。

## 当前18批状态

| 批次 | 计划官方日 / 来源行 | 真实来源校验 | 原生生成 | 独立Native G3 | 独立Reader | 质量分布/成本 |
| --- | ---: | --- | --- | --- | --- | --- |
| TRAIN01_FG809_FULL_MAIN | 83 / 28264 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN02_FG901_FULL_MAIN | 86 / 29416 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN03_FG905_FULL_MAIN | 79 / 26762 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN04_FG909_FULL_MAIN | 78 / 26546 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN05_FG001_FULL_MAIN | 81 / 27464 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN06_FG005_FULL_MAIN | 75 / 20505 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN07_FG009_FULL_MAIN | 88 / 27600 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN08_FG101_FULL_MAIN | 82 / 28050 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN09_FG105_FULL_MAIN | 76 / 25620 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN10_FG109_FULL_MAIN | 73 / 24945 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN11_FG201_FULL_MAIN | 92 / 31260 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN12_FG205_FULL_MAIN | 68 / 23220 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN13_FG209_FULL_MAIN | 94 / 31950 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN14_FG301_FULL_MAIN | 81 / 27703 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN15_FG305_FULL_MAIN | 79 / 26895 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN16_FG309_FULL_MAIN | 84 / 28740 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN17_FG401_FULL_MAIN | 78 / 26790 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| TRAIN18_FG405_FULL_MAIN | 78 / 26430 | SOURCE_VALIDATED_ONLY | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

精确日期、官方日列表、源文件SHA与每批身份锁见 [生产计划](TRAIN_CORPUS_PRODUCTION_PLAN.md) 和 [完整锁](docs/formal_train_corpus_20261018/locks/IDENTITY_LOCK.json)。18份实际validate输出只证明TRAIN范围/来源/日历与主力边界，不代替生成、G3或Reader。

## 每批生成后填写的质量分母与缺口

所有以下项目当前均NOT_RUN，不预填0或沿用旧PASS：

- 市场：去重contract×cursor、有效推进分钟、合约官方日；六账户物理Replay行及账户决策数单列。
- 原始L1候选：TP5、SL30、OTHER_EXIT、快速TP≤5市场分钟、较慢TP、5/15分钟内逆势≥10及随后SL、TP且高MAE、同日/跨日、右删失；同时报告账户状态次数与去重cursor支持，重叠题型不相加为独立样本。
- 实际交易：各行为真实ADD、TP、SL、主力退出、跨日持仓和未完成交易；候选结果不冒充已执行交易。
- Reader投影：outcome/score/event/touch/adverse valid分母、严格ADD/WAIT、Quality Pair、无效原因；无效标签不补WAIT。
- 课程支持：自然/分层/混合采样、各行为及各阶段真实窗口数、缺题型表；不足保持LIMITED/UNSUPPORTED。
- 成本：各阶段GNU time和实际产物字节、Reader索引/抽样成本；全进程树PSS峰值无测量时保持NOT_RUN。全周期耗时不能由旧三日样本推定已完成预算。

发布成功以新目录对应的BATCH_MANIFEST与逐批G3/Reader原始收据为准。教材范围由完整TRAIN池确定，不按未来质量标签选日；生成后的题型缺口用于评估课程支持，不能据此改写历史身份。当前没有可据以报告的正式18批质量结果。

本轮无真实生成、无新G3/Reader、无模型/优化器更新、无GPU、无DEV/TEST价格访问。逐批命令已提供给用户，入口不自动启动后续批次，不合并到integration/s3-next。

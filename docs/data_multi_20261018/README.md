# DATA-MULTI 开发交付

状态：**DEV_COMPLETE；真实来源验收 NOT_RUN**。独立共用数据层已实现，使用两个分别采集的原生合成Corpus完成端到端验证。选定的116个不同用例通过，失败0、错误0、跳过0；这不代表全项目或正式训练验收通过。

## 核对入口

| 材料 | 内容 |
| --- | --- |
| [实施Review](IMPLEMENTATION_REVIEW.md) | 需求逐项核对、修改范围、修复与未验证项 |
| [接口说明](INTERFACE.md) | 共用采样、身份、RNG、隔离、适配器和恢复合同 |
| [训练接入说明](RUN_INTEGRATION.md) | 后续F1/F2/F3接入Run所需的最小修改 |
| [真实来源配置示例](sources.example.json) | 显式来源与1:1权重；路径为待生产示例 |
| [代码身份](evidence/SOURCE_IDENTITY.json) | 双仓完整SHA、dirty状态、测试源码哈希 |
| [材料SHA256清单](evidence/SHA256SUMS.json) | 文档、原始日志、JSON及Checkpoint文件完整性 |
| [测试汇总与完整清单](evidence/TEST_RESULTS.json) | 26个本功能用例、90个共享回归 |
| [实际执行命令](evidence/COMMANDS.json) | argv、环境变量、退出码、日志位置 |
| [原生冷恢复证据](evidence/synthetic_final/CHECKPOINT_COLD_EVIDENCE.json) | 三个独立进程、原FullCheckpoint、两批预取队列 |
| [合成诊断报告](evidence/synthetic_final/DATA_MULTI_REPORT.json) | DEV_PASS、聚合来源身份、零优化器更新 |
| [覆盖与分母](evidence/synthetic_final/coverage.json) | 各来源投递曝光及独立分母 |
| [窗口出处](evidence/synthetic_final/sample_audit.json) | 逐行Corpus/账户/segment/stepid出处 |
| [真实教材就绪情况](evidence/REAL_SOURCE_READINESS.json) | 两个正式生产根下均未发现已发布BATCH_MANIFEST |
| [未运行项](evidence/NOT_RUN.json) | 8个会更新真实优化器的共享用例及后续训练接线验收 |

## 代码与验证范围

Gym实现提交：`3abf8aaeee5e5db0060439bb18188562c958fbdd`；分支`feat/data-multi-20261018`，工作树`/home/v/Documents/work/research/fg-s3-data-multi-20261018/gym-trading-env`。

Dreamer：`0e39dfd498f9146e10f966cfe118a24b20e5143e`，未修改。测试执行于提交前，代码身份收据逐文件确认测试字节与该提交一致；后续文档/证据提交不改变运行源码。没有合并到`integration/s3-next`，没有修改WP04或WP06工作树。

新增`fg_s3_e_quality_multi.py`和只读诊断模块，在原Reader工具增加`multi-read`。每来源直接使用原WP03 Reader及已有Join索引；共用原F1/F2适配器和FullCheckpoint/RuntimeState/Prefetch。没有改动Agent、Replay、金融数学、MACD或训练入口。

冷恢复对比连续消费6批与消费3批后退出、在新解释器恢复至6批。所有原生数组dtype/shape/字节、出处、调度器/各Reader RNG、计数、已见集合、Runtime和两批待消费队列完全一致。证据目录保留原始日志、Checkpoint bundle和数据状态pickle；其中`agent.pkl`是显式`DATA_ONLY_NO_AGENT_OR_OPTIMIZER`占位对象，不含模型或优化器。

留存的两个合成来源位于`artifacts/fg_s3_v2/data_multi_delivery_20261019/sources/`，不作为正式教材；[来源引用清单](evidence/SYNTHETIC_SOURCE_REFERENCES.json)记录绝对路径、配置、来源锁与文件哈希。原生NPZ继续位于外部Corpus，未复制进Checkpoint。测试会自行建立新的两份合成来源，不依赖这套留存路径。

## 当前边界

真实多Corpus验收需至少两份已发布且通过Native G3及独立Reader的教材。目前该项为NOT_RUN；开发完成不依赖生产进度。F1/F2/F3的实际prepare/start/resume接线、真实神经模型/优化器恢复、GPU训练和正式训练准入均NOT_RUN。原共享三文件共收集20例，执行其中12例；另外8例的fixture调用真实`ActorBC.update`，按用户禁止优化器更新的要求未执行，清单没有使用skip或伪造PASS。

本次审查中的极端权重`1e-320 : 1e300`是手工构造的回归输入。实际合成验收配置和正式来源示例均为1:1；核心没有自动学习、推断或放大来源权重。数值下溢拒绝的[根因证据](evidence/WEIGHT_ROOT_CAUSE.json)与正式教材长分数故障分开记录。

后续接入仅需在已有Run准备/配置中加入来源列表、构造并绑定共用流、绑定聚合Run身份、沿用原Checkpoint注册与恢复。详细接口见[接入说明](RUN_INTEGRATION.md)；F3提供同协议阶段适配器即可，无需重构调度核心。

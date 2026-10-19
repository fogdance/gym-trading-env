# DATA-MULTI F3 交付入口

以s3-next中已合入的WP06为基线，为F1/F2/F3复用同一个多Corpus核心。F3保留必选自然流、可选辅助流和原Loss权限；本轮交付数据接口、采样统计及精确数据状态恢复。

**代码已合入integration/s3-next。** DATA-MULTI接口42例、共享回归126例、正式教材长精确分数运行时回归3例，共171例通过，最终失败/错误/跳过均为0。集成工作树的F1/F2/F3接口冒烟通过；真实教材验收仍为NOT_RUN。

| 材料 | 内容 |
| --- | --- |
| [接口与配置说明](INTERFACE.md) | API、角色语义、Natural统计口径、RNG和Checkpoint |
| [实施Review](IMPLEMENTATION_REVIEW.md) | 八项要求核对、实际测试、未运行项及源码范围 |
| [配置示例](sources.example.json) | 显式来源权重、自然spec和可选辅助spec |
| [代码身份](evidence/SOURCE_IDENTITY.json) | 双仓完整实现SHA、测试字节哈希及dirty状态 |
| [合入WP06记录](evidence/MERGE_RECEIPT.json) | s3-next基线、合并方向和实际父提交 |
| [合回s3-next记录](evidence/OUTGOING_MERGE.json) | DATA-MULTI双仓源码的实际集成 |
| [正式生产修复合并](evidence/FORMAL_PRODUCTION_MERGE.json) | 长精确分数运行环境修复及3个回归 |
| [最终集成核对](evidence/INTEGRATION_FINAL_CHECK.json) | 最终源码与集成接口冒烟的字节一致性 |
| [本对话分支核对](evidence/THIS_CONVERSATION_MERGE_AUDIT.json) | 仅核对本对话使用的分支，遗漏源码提交为0 |
| [测试清单](evidence/TEST_RESULTS.json) | 最终执行集合、准确数量、命令与未运行项 |
| [仅自然流冷恢复](evidence/natural_final/CHECKPOINT_COLD_EVIDENCE.json) | 连续6批，对比3批后新进程恢复至6批 |
| [自然＋辅助流冷恢复](evidence/auxiliary_final/CHECKPOINT_COLD_EVIDENCE.json) | 全部角色、来源、统计和预取队列恢复 |
| [Natural覆盖报告](evidence/auxiliary_final/coverage.json) | 来源级/全局分母及配置混合含义 |
| [只读原F3 Loss证据](evidence/auxiliary_final/F3_READONLY_LOSS.json) | 固定logits，无模型训练或优化器更新 |
| [真实教材状态](evidence/REAL_SOURCE_READINESS.json) | 等待真实V3生产完成，实际验收NOT_RUN |
| [材料校验](evidence/DOCUMENT_VALIDATION.json) | JSON、引用、代码身份及原Checkpoint完整性 |
| [文件哈希清单](evidence/SHA256SUMS.json) | 本次F3文档与收据的独立身份清单 |

Natural仅描述来源内部窗口采样。全局混合使用显式来源权重，报告明确不代表真实市场自然分布；来源级及全局标签曝光计数全部保存恢复。

F1/F2现有MultiCorpus接口保持。F3新增`MultiCorpusF3Stream.open(...)`，不重写训练器，不修改Loss、Teacher或Observation。源文件、账户与Teacher后缀边界继续由原Join、原Reader、原F3适配器和原Pair函数验证。

本目录是本次F3扩展的独立收据；上级目录的原DATA-MULTI验收属于历史版本，不作为当前版本自动PASS。按照用户追加授权，通过接口测试后已将本次代码合并回integration/s3-next；正式教材运行时修复也已合入，实际结果见上述合并记录。只操作DATA-MULTI及integration工作树，未覆盖WP07工作树。

仍未接通F1/F2/F3的prepare/start/resume多来源入口，未执行真实V3多Corpus验收、真实神经模型/优化器冷恢复、GPU或正式训练。后续Run接线继续沿用原准入及授权流程。

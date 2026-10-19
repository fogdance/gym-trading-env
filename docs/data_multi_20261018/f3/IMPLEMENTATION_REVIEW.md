# DATA-MULTI F3 实施Review

## 结果与范围

本轮交付F1/F2/F3共用的多Corpus数据层，复用已合入s3-next的WP06。实现、测试及集成提交的双仓完整SHA、dirty状态和测试字节绑定见[源码身份](evidence/SOURCE_IDENTITY.json)；实际测试清单见[TEST_RESULTS](evidence/TEST_RESULTS.json)。真实V3来源及神经模型/优化器恢复均保留NOT_RUN。

本轮选定集合共171例通过：DATA-MULTI接口42例、共享回归126例、正式教材运行时3例；失败、错误、跳过均为0。双仓代码已合回integration/s3-next，F1/F2/F3实际集成接口冒烟通过，见[合并记录](evidence/OUTGOING_MERGE.json)与[最终集成核对](evidence/INTEGRATION_FINAL_CHECK.json)。这不是全项目或正式训练验收。

Gym先将WP06基线`d5153268bcd33d5a4e54721767c8782f895283f9`合入DATA-MULTI，产生`b5c6f2dca3bf0396ced3f6bc519e37ae9ab4197e`；Dreamer从`0e39dfd498f9146e10f966cfe118a24b20e5143e`快进到`e8c2688bc817662e5e6fa7e8cee62825bcf80b00`。合并无冲突，未以历史WP06 PASS代替本轮测试。

## 八项要求核对

| 要求 | 实现及验证 |
| --- | --- |
| 1. Natural/辅助流及Loss权限 | 必选NATURAL、可选AUXILIARY，按原WP06顺序合并；辅助即使使用NATURAL spec，`f3_natural`仍为false。四种角色配置及原Loss分母测试验证。 |
| 2. MultiCorpus职责与原目标 | 核心只选择来源、组织窗口、保存身份/RNG/统计；新增F3薄组合复用原ActorStreams与F3WindowAdapter。原F3 Loss、Teacher和训练目标文件字节未改变。 |
| 3. 跨来源batch、单来源窗口 | 一批混合来源，只拼接batch轴；每行经原Reader审计单一Corpus/账户/segment及连续ordinal，保持原生数组字节。 |
| 4. Pair隔离 | 原Join的owner绑定Corpus、账户trace、segment、Teacher和suffix；原F3适配器核对canonical标签。实际Sidecar/Join owner正例、跨行owner碰撞、不同suffix owner负例及原WP06账户/reset/零owner回归验证。 |
| 5. RNG与精确恢复 | 角色分域种子、独立Scheduler及各Reader RNG；全部状态、采样计数、Natural统计和逐批出处随原Checkpoint与预取队列保存，三进程冷恢复逐字节比较。 |
| 6. Natural统计口径 | 保存来源级及全局自然角色曝光、ADD/WAIT/无效分母和原来源账户标签统计；明确配置权重、实际窗口份额及`represents_true_market_natural_distribution=false`。 |
| 7. 合成开发与真实V3后验收 | 两个独立六行为原生合成Corpus完成开发验证；现场正式V1/V2根无已发布BATCH_MANIFEST，真实来源验收NOT_RUN。未生成正式教材。 |
| 8. 通过后合回s3-next | PASS：接口测试通过后，Gym与Dreamer代码均已合入integration/s3-next，实际接口冒烟通过；见合并与最终集成收据。未修改WP07工作树。 |

## 修改与复用边界

Gym新增`MultiCorpusF3Stream`作为角色组合的薄接口，继续使用原多Corpus核心。共用WindowAdapter只增加reader列表、coverage和审计访问；通用coverage增加显式来源分布说明。原Reader工具新增F3诊断选项，旧native模式保留。

Dreamer对原`ActorStreams`增加已构造角色流的注入接口，将回滚/coverage/close统一委托给原生命周期对象；原单Corpus构造和V1保存格式保持。`CourseStream`仅增加三个委托访问。没有增加Replay、标签生产器、训练器、Checkpoint backend或优化器。

原训练合同核对见[未改变的源码与类](evidence/UNCHANGED_CONTRACTS.json)；实际集成工作树接口验证见[冒烟收据](evidence/INTEGRATION_SMOKE.json)。

本轮未接通F1/F2/F3 prepare/start/resume的多来源入口，未改变模型参数继承或正式准入。原F3 Loss、Pair公式/阈值、有效标签规则、Teacher、MACD和Observation没有本功能修改。仍使用原V3 Join，不为旧格式新增fallback。

## 统计与恢复的实际含义

来源内部NATURAL采样和全局市场先验是两个不同统计口径。当前显式1:1来源权重只决定来源抽取概率，不能作为真实市场分布。报告分别保留每来源原Reader分母、配置来源概率、投递份额和自然角色全局曝光，不将重叠来源或六行为账户相加为独立行情。

新增统计只累计K后目标位置的已有Teacher标签，不重写F3 Loss有效性判定。ADD+WAIT+无效数等于目标位置数；恢复额外校验这些分母与来源窗口计数。真实Loss分母由原目标函数独立给出；固定logits的只读Loss诊断不代表Actor模型前向或学习效果。

两组冷恢复分别使用“仅自然流”和“自然＋MIXED辅助流”，B=4、K=4、T=8、两批原生Prefetch。对照连续消费6批与3批后退出、在新进程恢复至6批；比较全部数组dtype/shape/字节、审计、角色/来源状态、所有RNG、计数、已见集合、Runtime及待消费队列。只排除两个输出目录的Checkpoint路径。自然统计在启用辅助流前后相同，额外计数对比见[统计冷恢复证据](evidence/STATISTICS_COLD_EVIDENCE.json)。

Checkpoint保留原bundle、原始日志和状态pickle，没有复制Corpus NPZ。`agent.pkl`是显式DATA_ONLY_NO_AGENT_OR_OPTIMIZER占位对象；不宣称神经模型或真实优化器冷恢复已经验收。外部源和索引在Checkpoint边界校验哈希，每批继续走原stat guard与缓存。

## 自审发现与修正

1. 新F3接口最初直接公开内部identity字典。运行复现外部把target_steps改为999后，内部值也变化而identity_hash不变，见[修复前证据](evidence/DEV_IDENTITY_MUTABILITY_BEFORE.json)。现改为返回深拷贝，并在接口回归中验证内部设置仍为8。
2. 新增suffix测试先误取不存在的manifest键，后又误要求未获监督支持的行具有非零owner。查阅真实trace/Sidecar及`project_label`后，测试改为核对canonical Join投影，只对非零owner验证完整suffix身份；没有放宽生产投影。两份失败JUnit保留为开发过程证据，不混入最终PASS计数。
3. 已逐段自审新增模块、原组合器、诊断分派、状态/统计回滚及测试，检查标签与Observation隔离、身份变更拒绝、缓存总预算和错误后的失效行为。最终语法、差异空白及材料完整性检查见[材料校验](evidence/DOCUMENT_VALIDATION.json)。

## 正式教材运行时修复的合并

按用户追加要求，将长精确分数运行时修复`9f0f847979e2047a5c9b2d87206c4423ecf111a3`随正式生产计划合入DATA-MULTI，并以`b0201f0b130e92674c716c5344f1bc1856d5a0c9`同步到integration/s3-next，见[独立合并收据](evidence/FORMAL_PRODUCTION_MERGE.json)。生产入口向整个子进程树传递`PYTHONINTMAXSTRDIGITS=0`，不改变精确分数数学、Teacher或冻结Producer源码。3个定向回归、脚本语法和生产包哈希均通过；本轮未启动正式教材生产。

## 测试、未运行项及接入

实际命令、环境、测试文件清单和准确通过/失败/跳过数量以[最终测试汇总](evidence/TEST_RESULTS.json)及原始JUnit为准。CPU执行，隐藏CUDA，无真实优化器更新。原WP06采样/Loss测试的相对输出写入隔离artifacts目录，未覆盖历史WP06收据。

固定共享三文件中8个用例依赖会执行`ActorBC.update`的fixture，按本轮限制保持NOT_RUN，未添加skip/xfail或篡改守卫。实际未运行清单见[NOT_RUN](evidence/NOT_RUN.json)。这不等于全项目测试通过。

真实V3多Corpus验收等待教材发布，本次仅记录[现场就绪情况](evidence/REAL_SOURCE_READINESS.json)。真实神经Agent/优化器恢复、GPU、正式训练及阶段入口接线均未运行。

后续接Run的最小修改仍为：原prepare逐来源准入并冻结索引；用本接口替换数据流构造；Run身份绑定全部来源/角色/配比；将同一流与原Prefetch注册到原FullCheckpoint/RuntimeState。阶段模型、Loss和训练循环继续复用，不以数据层DEV完成替代训练授权。

# DATA-MULTI 实施Review

## 结论与观察证据

DEV交付完成，真实来源验收NOT_RUN。实现提交`3abf8aaeee5e5db0060439bb18188562c958fbdd`，Dreamer保持`0e39dfd498f9146e10f966cfe118a24b20e5143e`。运行源码、测试文件的实际SHA256及提交字节一致性见[代码身份](evidence/SOURCE_IDENTITY.json)。

原WP03 Reader已有原生窗口采样、Sidecar Join、账户/segment校验、完整RNG状态及外部不可变来源合同。F1/F2已有阶段字段适配器；原FullCheckpoint支持`external_immutable_course_reader`，Prefetch及RuntimeState支持队列保存恢复。本轮沿这些扩展点组合，未新建Replay、标签或训练系统。

## 需求核对

| 要求 | 实现与实际证据 | 状态 |
| --- | --- | --- |
| 复用WP03 Reader | 每来源打开`EQualityCourseReader`；分组调用原sample/audit；原Reader/Join/Admission/CLI回归 | PASS |
| 共用F1/F2/F3接口 | `open/sample/bind/state_dict/load_state_dict/validate_snapshot/coverage/close`；泛化阶段适配协议；实际F1/F2适配测试 | PASS；F3模型接线NOT_RUN |
| 多来源调度与跨Corpus批次 | 显式有序sources/spec/weights；每行调度，按来源采样，再只沿batch轴合并；四种stage测试 | PASS |
| 独立RNG | 调度器与各Reader分域派生种子；对照独立原Reader及NATURAL/STRATIFIED/MIXED恢复 | PASS |
| 聚合身份 | 来源、Course/Teacher/source lock、index及收据、顺序、权重、spec、stage均绑定；改变任一来源配置拒绝恢复 | PASS |
| 账户与窗口隔离 | 每行仅一个Corpus/账户/segment；拒绝跨源splice；跨行owner碰撞不能伪造Pair；适配器不得改原生字节 | PASS |
| 完整数据状态冷恢复 | 三个独立进程，连续6批对比3批后冷恢复至6批；全状态及待消费队列字节一致 | PASS |
| 原Checkpoint系统复用 | 原FullCheckpoint、RuntimeState和Prefetch；外部不可变Reader backend；原bundle校验及相关10个回归 | PASS；神经模型/优化器NOT_RUN |
| 两个合成Corpus端到端 | 两次独立六行为原生采集，beta价格整体平移17；各40行情行；分别Join与Admission | PASS；DEV_ONLY |
| 真实来源验收 | `--require-published`核验既有批次/G3/Reader/索引/原TRAIN信任根；现场发布清单数量0 | NOT_RUN；不阻断DEV |
| 阶段入口及工作树边界 | 未修改F1/F2入口、WP04、WP06；本独立分支提交，无integration合并 | 保持 |

## 实现范围与自审

- 核心模块负责来源调度、聚合身份、窗口出处、事务回滚和保存恢复；继续复用原Reader的窗口/Pair语义。
- 通用WindowAdapter按来源应用既有适配器，校验原生Replay字段、stepid和consec字节不变，恢复原batch顺序。出处保存在模型数组之外，随原Prefetch的每批状态一并保存。
- 诊断模块复用原Reader工具及原Checkpoint；所有占位状态明确DATA_ONLY，无Agent参数或优化器。正式发布证据必须来自实际生产，不由合成测试生成。
- 所有来源每批均检查stat guard，未选中的来源被改也拒绝继续；Checkpoint边界全量检查外部哈希。默认NPZ总缓存64MiB按来源分配；本次1MiB预算下实测上界752560字节。
- 正常异常回滚所有RNG与计数；来源/索引完整性失败使流失效。缺类在消耗RNG前拒绝，不调换来源或补造目标。
- 没有修改原Reader、Replay、Teacher、Observation、MACD、FullCheckpoint、Prefetch、F1/F2适配器或模型代码；唯一既有文件修改是Reader CLI增加10行分派。

自审已逐段检查新增核心/诊断/测试及现有CLI差异，检查依赖复用、身份变更拒绝、来源篡改、原生字段保持、计数恢复、预取出处配对和NOT_RUN表述。发现的权重边界已按下节修复；最终暂存差异空白检查、5个Python文件AST解析及JSON配置语法检查通过。

## 权重问题的根因及最小修复

**观察事实：** 合成配置与正式示例均显式1:1。配置值直接传入核心，不由行情、课程分母、Reader或模型计算。极端值仅出现在本次主动添加的`underflow_weight`回归输入。

**已确认根因：** 原归一化使用float64，先除最大值。手工输入`1e-320 : 1e300`后，小值的理论比例为`1e-620`，低于最小正float64约`4.94e-324`，运行结果为`[0.0, 1.0]`。输入正数校验无法发现归一化后的下溢；核心因此可能静默排除来源。见[数值及配置证据](evidence/WEIGHT_ROOT_CAUSE.json)。

**修复与验证：** 在打开Reader前完成权重数值转换、有限正数检查、归一化及正概率检查，不能表示时抛出`MULTI_SOURCE_WEIGHTS_NOT_REPRESENTABLE`。正常1:1算法及采样顺序不变。新增同一公开合同下的一个回归输入；[修复前日志](evidence/WEIGHT_REGRESSION_BEFORE.log)实际显示`DID NOT RAISE ValueError`，[修复前JUnit](evidence/WEIGHT_REGRESSION_BEFORE.xml)为1失败；最终26例包含该回归，全部通过。失败收据明确是修复前证据，不计入最终通过总数。

正式教材V1的4300位报错来自冻结Producer精确Fraction转十进制字符串，与多Corpus权重无调用关系。其独立V2生产启动修复和重跑命令在另一生产计划工作树的`docs/FORMAL_CORPUS_WITNESS_REVIEW_20261018.md`；本功能未更改冻结Producer，也未启动正式教材生成。

## 测试与恢复证据

实际命令及环境见[COMMANDS.json](evidence/COMMANDS.json)。CPU执行，CUDA设备隐藏，JAX平台cpu，BLAS/OpenMP每进程1线程，无GPU或真实优化器更新。

| 执行范围 | 通过/失败/错误/跳过 | 原始收据 |
| --- | --- | --- |
| DATA-MULTI公开合同与端到端冷恢复 | 26 / 0 / 0 / 0 | [JUnit](evidence/TARGETED_FINAL.xml)、[日志](evidence/TARGETED_FINAL.log) |
| WP03 Reader/Join/Admission/CLI、FullCheckpoint及允许执行的Gate/来源/动作守卫共享回归 | 90 / 0 / 0 / 0 | [JUnit](evidence/SHARED_FINAL.xml)、[日志](evidence/SHARED_FINAL.log) |
| 固定共享三文件仅收集，不执行fixture | 20 collected；其中12已在上行执行、8未运行 | [完整收集日志](evidence/SHARED_REQUIRED_COLLECTION.log)、[未运行清单](evidence/NOT_RUN.json) |
| 两个留存合成Corpus的原工具CLI | exit 0；DEV_PASS | [原始CLI日志](evidence/RETAINED_SYNTHETIC_CLI.log)、[诊断报告](evidence/synthetic_final/DATA_MULTI_REPORT.json) |

最终不同执行用例共116，重复调试次数不叠加。原共享fixture在`embodied/tests/test_fg_s3_source_transfer.py:87`调用两次`ActorBC.update`，8个依赖该路径的用例按本轮禁止优化器更新的限制保留NOT_RUN，没有新增skip/xfail或放松生产守卫。完整用例名、准确数量及该限制见[TEST_RESULTS](evidence/TEST_RESULTS.json)。

留存冷恢复使用B=4、有效L=12，两批Prefetch，连续消费6批与在3批Checkpoint后新进程恢复至6批对照。三次子进程argv/退出码/日志哈希均保留；比较全部原生数组dtype/shape/字节、窗口出处、所有来源RNG/计数/已见集合、调度RNG、Runtime和待消费队列。仅排除两个不同输出目录的Checkpoint绝对路径。结果`PASS_EXACT_COLD_DATA_STATE`，bundle step=6、backend=`external_immutable_course_reader`，未复制Corpus chunk。

合成来源聚合身份：`782febb12c655405dc69169300211a84366d9dc42618a796e369075fddf7308e`。

核心源码SHA256：`5994e7ac36c460dbd1d9fe011621b8020a5c531dfe049f669b7cf7c0e5689291`。冷恢复bundle同时绑定诊断源码哈希、适配器身份、数据维度及实际包版本；已核对与实现提交相同。原始bundle、pending数组及数据状态pickle保存在[synthetic_final](evidence/synthetic_final/)。

## 风险、未验证及后续

本次只验证共用数据层，未验收F3模型/损失、实际神经Agent/优化器冷恢复或学习效果。后续F3适配器须遵守已有无独立采样RNG/训练状态的协议。采样按行独立抽取，不保证每批都含所有来源；正式配比需由后续明确配置。

截至[现场观察时间](evidence/REAL_SOURCE_READINESS.json)，正式V1/V2生产根均无BATCH_MANIFEST，真实来源验收NOT_RUN。两份教材发布后使用[接口文档命令](INTERFACE.md)加`--require-published`进行只读验收；它核验既有G3证据，不声称重新执行G3或获得训练授权。

F1/F2/F3接Run的最小修改仅为准备/配置来源列表、构造并绑定共用流、更新聚合Run身份、注册原Checkpoint与预取恢复，详见[接入说明](RUN_INTEGRATION.md)。旧单来源Checkpoint不能伪装恢复为多来源Run。无需重构MultiCorpus调度核心，本轮不自动实施这些入口修改或合并到integration。

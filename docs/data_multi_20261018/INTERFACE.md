# DATA-MULTI：共用数据层接口

本模块只组合原WP03 `EQualityCourseReader`，不新建Replay、合并NPZ/Sidecar、生成标签或授予训练权限。真实教材与合成工程验证的状态分别报告；未执行的真实验收保留NOT_RUN。

## 原生采样接口

```python
from gym_trading_env.research.fg_s3_e_quality_multi import MultiCorpusCourseStream

stream = MultiCorpusCourseStream.open(
    sources=[{"corpus_id": "a", "admission": admission_a, "index_path": index_a},
             {"corpus_id": "b", "admission": admission_b, "index_path": index_b}],
    stage="WM_NATURAL",
    per_source_sampling_spec={"a": {"mode": "NATURAL"}, "b": {"mode": "NATURAL"}},
    source_mix_weights={"a": 1., "b": 1.}, seed=202, cache_bytes=64 << 20,
)
raw, audit_rows = stream.sample(batch_size=4, effective_length=96)
state = stream.state_dict()
stream.load_state_dict(state)
coverage = stream.coverage()
stream.close()
```

`admission`是现有 `QualitySupervisionAdmission.preflight(...)` 返回的类型；不能传入伪造JSON PASS。各来源的Admission stage必须匹配：

| Reader采样阶段 | Admission课程阶段 |
| --- | --- |
| WM_NATURAL | FOUNDATION_WM |
| HEADS_VALID | QUALITY_HEADS |
| ACTOR_STRICT / ACTOR_QUALITY_PAIR | ACTOR_SUPERVISED |

来源数至少1；ID及Corpus内容身份分别唯一。来源列表有序，权重和采样spec必须显式覆盖全部ID，权重有限且大于0，归一化后也必须能表示为正float64概率。极端比例导致数值下溢时，在打开Reader前明确拒绝。没有隐式合约配比、自动排除来源、零权重隐藏来源或缺类替补。示例1:1是接口示例，不是正式训练配比结论。

每批每行独立按来源权重抽取，随后按来源分组调用一次原Reader，最后只沿batch轴恢复行顺序。因此一批可以跨来源，但不承诺每批必然出现全部来源。每行始终是同一个来源、账户、segment内连续的 `effective_length+1` 行，保持原生字段、dtype、stepid及首位置标签屏蔽。原生chunk边界由WP03处理，不按日期重新建账户。

`raw`是原WP03批次字典，包含原生Replay字段及分离的监督目标；不是可直接当作政策Observation的字典。`audit_rows`是独立审计列表，每个元素包含`corpus_id`、`source_identity`、`index_sha256`及原Reader的contract/policy/segment/ordinal/cursor/stepid等窗口出处。`audit_windows(raw, audit_rows)`按显式来源重新核对窗口，不能用全局stepid查询猜来源。

## 跨阶段适配

`stream.bind(settings, adapter_cls)`返回 `MultiCorpusWindowAdapter`，具有 `sample()`、迭代器、`save/load/save_full/load_full` 和 `validate_snapshot`。`settings`至少包含正整数`batch_size/target_steps/burnin_steps`，窗口长度为K+T+1。

通用适配器协议：构造函数 `adapter_cls(reader, settings)`；`adapt(raw)`返回`(data, metadata)`；可选`validate_support()`在绑定时检查目标区间的支持。适配器应无独立采样RNG或可变训练状态，只允许缓存静态支持查询。核心按实际来源子批次设置batch_size，保持K/T与其他设置，检查所有原生Replay字段、stepid、consec字节不变；阶段适配器负责监督字段选择及合法投影。

| 阶段 | 接法 | 边界 |
| --- | --- | --- |
| F1 | `bind(settings, F1WindowAdapter)` | 复用现有原生字段白名单，去掉所有Teacher/Quality目标；各来源必须NATURAL |
| F2 | `bind(settings, F2WindowAdapter)` | 复用四头目标、valid mask和K后目标损失支持检查；不创建Actor/Q目标 |
| F3 | WP06提供同协议适配器，选择ACTOR_STRICT或ACTOR_QUALITY_PAIR | 数据调度/合并/恢复核心不变；本轮不实现或宣称验收F3模型/损失 |

`bound.sample()`同时返回模型数组和独立审计；`next(bound)`仅返回模型数组，兼容原 `agent.stream(bound)`。政策Observation始终由既有Agent.spaces/阶段适配器确定，不添加`corpus_id`或Teacher信号。

## RNG、来源身份与失败

调度器使用单独的PCG64 RNG。每个子Reader的种子由主种子、固定域标签、corpus_id和内容身份经现有fingerprint派生；不使用Python hash，也不从调度器抽取子种子。读取另一来源不会消耗本来源RNG。

聚合identity版本化并绑定有序来源ID、Corpus/Course/Teacher/source/index/索引收据身份、各来源spec、显式权重、主种子和调度协议。按原生验证器确认合同，要求Observation、模型合同、Reward和Teacher数学兼容；不同合约的Course/Teacher内容身份正常不同，不要求相等。源绝对路径与缓存大小不决定采样身份，迁移须保留相同文件及索引收据字节。

`support(L)`查询各来源实际可采样窗口/类别数；缺少任何要求的支持时`sample`在消耗RNG前报错，不静默重分配权重。正常适配/采样异常回滚调度器、子Reader及计数；发现源/索引完整性错误则当前流失效，修复后须重新打开并从合法状态恢复，不能继续使用已失效对象。

状态保存全部子Reader RNG、计数与已见集合，以及调度器RNG、来源曝光计数和最近批次出处。改变来源列表、顺序、权重、阶段、spec或索引身份都会拒绝恢复。绑定适配器还校验适配器标识和完整设置；不支持把单来源旧Run伪装恢复为多来源Run。

## 原生Checkpoint与预取

继续使用backend `external_immutable_course_reader`：将`bound`注册为`cp.course_reader`，将原生Prefetch传给FullCheckpoint和RuntimeState。不新增backend、不把外部Corpus注册成训练Replay。

`bound.save()`在每次预取后保存该批的`last_batch_audit`，原生Prefetch同时保存`(data, state)`。消费后可从 `prefetch.state['last_batch_audit']` 获取与当前已投递批次对应的出处；不能用采样器当前最新批次冒充尚在队列中的批次。`RuntimeState.save/load`保存原队列，恢复时先恢复来源状态与队列，最后启动worker。

开启/Checkpoint保存和恢复时验证完整源文件及索引哈希；每批复用WP03 stat guard、chunk缓存与窗口查询，不全量重算源哈希。总NPZ缓存默认64MiB，按来源均分；SQLite自身缓存和必要批次数组不计入该NPZ预算。覆盖报告保留每来源独立分母，只汇总投递曝光，不把可能重叠来源相加为独立行情。

## 诊断命令与真实验收

[JSON示例](sources.example.json) 对应两份完整主力期教材，路径仅作示例，不表示这些批次已完成。仅接受显式本地配置和已有索引；用原工具构建的索引仍留在Corpus之外。直接重建含长精确分数证据的Join时，使用 `PYTHONINTMAXSTRDIGITS=0`，参见生产修复说明。

```bash
env PYTHONINTMAXSTRDIGITS=0 /home/v/miniconda3/envs/dreamerv3/bin/python -B \
  tools/fg_s3_e_quality_reader.py multi-read \
  --config docs/data_multi_20261018/sources.example.json \
  --out artifacts/fg_s3_v2/data_multi/real_read_01 \
  --batch-size 4 --length 96 --require-published
```

`--require-published`要求至少两份真实BATCH_MANIFEST，核对清单、既有G3/Reader/索引及身份关联，并通过原Producer来源校验器重新确认原TRAIN信任根。它只沿用已有G3证据，不填写Admission.native_result，不声称本进程重新执行G3或取得正式训练准入。没有该标志的工程诊断始终为DEV_PASS/REAL_NOT_RUN，不能升级为真实来源验收。

诊断输出仅写入Corpus外的新目录，生成coverage、sample_audit、DATA_MULTI_REPORT及CHECKPOINT_COLD_EVIDENCE。冷恢复使用原FullCheckpoint、RuntimeState和两批Prefetch队列，在三个独立进程比较连续6批与3批后冷恢复到6批。占位saveable显式标记DATA_ONLY_NO_AGENT_OR_OPTIMIZER；不是模型或优化器Checkpoint。真实神经训练恢复仍需后续Run接入验证。

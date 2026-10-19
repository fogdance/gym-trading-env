# F1/F2/F3 共用多Corpus数据层

## 接口与复用关系

F1/F2继续使用`MultiCorpusCourseStream.open(...).bind(settings, F1WindowAdapter/F2WindowAdapter)`。F3使用下列组合接口；内部仍由相同MultiCorpus核心选择来源、原WP03 Reader采样、原WP06 F3WindowAdapter校验标签及投影。原WP06 `ActorStreams.from_streams(...)`负责角色合并和成组回滚，没有增加Replay或训练循环。

```python
from gym_trading_env.research.fg_s3_e_quality_multi_actor import MultiCorpusF3Stream

stream = MultiCorpusF3Stream.open(
    sources=sources,  # Same descriptors: corpus_id, typed admission, index_path.
    settings=dict(batch_size=4, target_steps=8, burnin_steps=4),
    per_source_sampling_spec={name: dict(mode="NATURAL") for name in ids},
    aux_per_source_sampling_spec={
        name: dict(mode="MIXED", natural_weight=.5,
                   strata_weights=dict(STRICT_ADD=1, STRICT_WAIT=1))
        for name in ids},  # None: no auxiliary role.
    source_mix_weights={name: 1.0 for name in ids},
    seed=202, cache_bytes=64 << 20,
)
data, audit_rows = stream.sample()
saved = stream.save()
stream.load(saved)
coverage = stream.coverage()
stream.close()
```

来源Admission必须为`ACTOR_SUPERVISED`；两个角色都使用当前WP06规定的`ACTOR_STRICT` Reader阶段。Quality Pair仍由原F3 Loss、有效标签和owner判断，不切换成另一套阶段实现。

自然角色必选，每个来源的spec必须是NATURAL。辅助角色可选，spec使用原Reader的NATURAL/STRATIFIED/MIXED；启用时必须显式覆盖全部来源。两角色使用相同的显式来源权重。本轮不推断合约权重，也不根据标签数量自动改配比。

`sample()`返回`(data, audit_rows)`；`next(stream)`只返回模型训练数组。`save/save_full/state_dict`和`load/load_full/load_state_dict`互为别名，另有`validate_snapshot/coverage/close`及只读审计属性`last_batch_audit`。

## 批次与Pair边界

自然流输出B行，开启辅助流后输出2B行，前B行为自然流。每行长度为K+T+1；同批可有不同Corpus，但单行仅一个Corpus、账户和合法连续segment。只沿batch轴合并。

`f3_natural`在自然行全为true，在辅助行全为false；辅助流内部使用NATURAL采样也不会获得自然损失权限。原生Replay字段、stepid和consec保持原字节，阶段适配器按既有合同选择监督字段。`corpus_id/role`只位于`audit_rows`，不进入模型Observation。

原Join生成的owner绑定Corpus、账户trace、segment、Teacher及合法suffix边界；原F3适配器重新核对canonical post-step标签。原Loss只在同一batch行内匹配非零owner和合法reset域，不跨Corpus、账户或Teacher后缀组合。未获得监督支持的行按原投影合同保留零owner。本轮没有修改Loss、有效性公式、Pair阈值或训练目标。

缺类、K后目标区间不支持、标签篡改等错误直接拒绝，不重采样来掩盖失败。正常失败回滚全部角色、所有来源RNG与统计；外部完整性错误使流失效，须重新打开。

## Natural统计的含义

**Natural指来源内部的NATURAL窗口采样。全局输出是按显式来源权重混合后的样本分布。** 例如每Corpus 1:1，并不等于按原始行情数量加权的市场分布。

`coverage()`保留每个角色、每个来源的原WP03覆盖报告与独立分母，并新增：

- `source_selection`：配置权重形成的来源概率、实际投递窗口比例；各共用MultiCorpus流均提供。
- `sampling_statistics[role][corpus_id]`：累计窗口数、K后目标位置数，以及原投影Teacher标签的ADD/WAIT/无效数量。
- `natural_distribution.per_source/global_counts`：自然角色逐来源及全局累计数。全局只求和投递曝光，不去重，不宣称独立市场分钟。
- `source_reference_account_labels`：每个原Corpus的全部账户trace中，既有Join统计的STRICT_ADD/WAIT数量；它包括行为账户重复，不是市场自然先验。

标签统计仅使用已有`teacher_valid/teacher_action`，不重新实现F3 Loss的有效性判定。因此`loss_or_optimizer_exposure=NOT_MEASURED_BY_DATA_LAYER`；原Loss的实际分母另由原F3指标给出。所有上述混合报告明确标记`represents_true_market_natural_distribution=false`。

这些计数随Checkpoint保存，恢复时校验角色/来源集合、窗口数量、目标位置分母和标签计数总和。

## RNG、Checkpoint和兼容边界

主种子按固定协议和角色名派生Natural/Auxiliary种子；每个角色核心再独立派生Scheduler及各Corpus Reader RNG。辅助流不会消耗自然流RNG。默认64MiB为所有角色、所有来源的总NPZ预算，不是每角色预算。

F3聚合identity绑定角色、来源、权重、spec、独立种子和窗口设置；对外`identity`返回副本，不能通过修改该字典改变内部设置。快照同时保存全部子Reader状态、Scheduler、角色状态、采样统计与最近批次出处。改变角色、来源、配比、采样spec或窗口设置均拒绝恢复。

继续注册原`FullCheckpoint`的`external_immutable_course_reader` backend：`cp.course_reader=stream`，使用同一个原生Prefetch及RuntimeState。消费后从`prefetch.state['last_batch_audit']`读取本批出处，不能读取采样器已经预取到未来的审计状态。恢复保留待消费队列中的原数组和审计，不重新抽样。

原单Corpus`ActorStreams`默认构造及`FG_S3_F3_READER_GROUP_V1`状态格式保持；新组合使用独立状态协议。没有为旧模型、旧Corpus或旧Join格式新增兼容回退。真实训练Run仍需自己绑定完整代码/配置身份及原准入流程。

## 只读诊断命令

[真实来源配置示例](sources.example.json)使用两份尚待认证的生产路径，不表示实际生产完成。合成验收配置和实际命令见evidence目录。

```bash
env PYTHONINTMAXSTRDIGITS=0 /home/v/miniconda3/envs/dreamerv3/bin/python -B \
  tools/fg_s3_e_quality_reader.py multi-read \
  --config docs/data_multi_20261018/f3/sources.example.json \
  --out artifacts/fg_s3_v2/data_multi_f3/new_read_01 \
  --adapter f3 --batch-size 4 --length 12 --burnin-steps 4
```

`length=K+T`，必须显式给出正burn-in且小于length。删除配置中的`f3_auxiliary_sampling_spec`或设为null，即只保留自然流。原`multi-read`默认native模式不变；F3辅助配置不会被native模式静默忽略。

诊断调用原F3目标函数，使用固定logits检验Loss和分母，不运行模型或优化器。检测中显式启用的辅助Loss权重仅用于该只读测试，不修改F3默认训练配置。Checkpoint中的`agent.pkl`仍是显式DATA_ONLY占位对象，不含真实模型或优化器状态。

真实V3教材完成后，增加`--require-published`核对实际BATCH_MANIFEST、既有G3/独立Reader及原TRAIN来源身份。该验收是只读的，不赋予训练权限。本轮不修改F1/F2/F3 prepare/start/resume；后续只需在原F3 Run的数据流构造处接入此接口，并将聚合身份写入Run身份。

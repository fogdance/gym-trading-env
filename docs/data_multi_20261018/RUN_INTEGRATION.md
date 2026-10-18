# 后续接入F1/F2/F3的最小修改

本轮没有修改F1/F2的prepare/start/resume入口，也没有修改WP04或WP06工作树。以下是下一次显式接线所需的修改，不表示已实施或获准训练。

1. **准备配置与来源。** 在各阶段现有配置读取/准备流程增加有序sources、每来源spec和显式mix权重；逐来源复用现有Admission/模型/金融/原TRAIN校验，按原WP03建立并冻结独立Join索引。保留原单来源路径与旧Run身份，不为旧Checkpoint增加静默兼容。
2. **替换数据流构造。** 从每阶段已通过校验的Admission及索引构造共用stream，再绑定该阶段WindowAdapter；F1/F2的模型、损失及Agent代码不变。F3由WP06的阶段适配器确定监督字段，调度核心无需改动。
3. **更新Run身份及计数。** 新Run identity绑定聚合identity及全部来源/索引/配比，不再只保存单个index hash。训练消耗计数继续由原Run管理，数据层曝光不得称为优化器曝光。从Prefetch消费态读取本批审计，不使用采样器前瞻态。
4. **沿用完整恢复。** `cp.course_reader=bound`；原 `agent.stream(bound)`、FullCheckpoint、RuntimeState继续保存模型、优化器、seed、队列和数据流。恢复前调用`bound.validate_snapshot(saved)`，来源身份不符时在模型/优化器状态改变前拒绝。

```python
# F1 example; preparations must already have passed their existing checks.
stream = MultiCorpusCourseStream.open(
    sources=sources, stage="WM_NATURAL",
    per_source_sampling_spec=specs, source_mix_weights=weights, seed=seed)
bound = stream.bind(settings, F1WindowAdapter)
prefetch = agent.stream(bound)
checkpoint.course_reader = bound
# Register the same prefetch with existing RuntimeState and FullCheckpoint.
# After batch = next(prefetch):
audit_rows = prefetch.state["last_batch_audit"]
```

F2替换为HEADS_VALID与F2WindowAdapter，保留各来源K后目标有效性检查。F3使用ACTOR_STRICT/ACTOR_QUALITY_PAIR及同一适配协议，保留原生owner、reset、合法动作、Teacher/Observation隔离；禁止跨来源/跨行配对。阶段切换、改变数据范围或配比创建新Run，不伪装同Run resume。

现有共用Checkpoint的协议没有修改；本轮只验证独立数据层与原生外部Reader backend及Prefetch的完整衔接。真实F1/F2/F3模型与优化器的端到端恢复、学习效果和训练准入均NOT_RUN，需要后续接线后按原合同验证。不得因为数据层DEV_PASS跳过正式Run身份校验或用户授权。

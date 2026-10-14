# E-Model V1 Review 入口

- [任务书原文](../S3_NEXT_E_Model_V1_Composition_Integrated_Codex_Task_Reviewed_20261014.md)
- [实施计划](../S3_E_MODEL_V1_IMPLEMENTATION_PLAN.md)
- [实施与验收报告](../S3_E_MODEL_V1_IMPLEMENTATION_REPORT.md)
- [模型/配置/来源接口](../S3_E_MODEL_V1_INTERFACE.md)
- [EM01–EM18状态](EM_STATUS.md)
- [真实E-Data联调阻断](E_DATA_BLOCKERS.md)
- [全部测试结果（含失败）](evidence/test_summary.json)
- [完整Checkpoint索引](evidence/checkpoint_index.json)

原始命令、日志、JUnit、资源、来源和数值收据在evidence，SHA256SUMS可逐文件复核。完整CPU Checkpoint留在索引给出的本隔离工作区artifacts目录，不上传W&B。双仓最终SHA和原始代码Diff位于工作区根 `baseline-lock.json` / `validation/`；不将CPU契约当作正式E Run。

```bash
cd docs/e_model_v1
sha256sum -c SHA256SUMS
```

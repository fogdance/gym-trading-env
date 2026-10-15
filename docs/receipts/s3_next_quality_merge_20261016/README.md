# Quality-first 合入 s3-next 的实际证据

[完整 Review](../../S3_NEXT_QUALITY_FIRST_INTEGRATION_REVIEW_20261016.md)列出五处文本冲突、两项自动合并缺陷、测试断言／输出目录问题及逐项修复。[机器摘要](summary.json)仅授予 CPU 工程验收；Quality Full NOT GO，预测／Q／决策门禁保持 PENDING，GPU／10K／1M 未启动。

[逐项测试索引](review/final_test_index.json)按测试身份选择最后一次实测结果，共 **235 个不同用例通过**，没有未解决失败／跳过。它不是一条 235 项全绿命令：

- [Native 最终 XML](tests/native_v2.xml)：198 项通过；[首轮 XML](tests/native.xml)的真实 TRAIN E／Legacy 财务、原 Oracle 与冷 Replay 用例另外通过，未重复运行。首轮其余 12 项因临时目录落入受保护集成树失败，最终独立 `/tmp` 目录修复该执行问题。
- [模型 XML](tests/models.xml)：35 项中旧 EM16 全类 AST 断言失败；[E source 最终 XML](tests/e_source_v2.xml)六项通过，保留原八个教学／共享 Loss 方法的严格 AST 比较。其余模型数值／梯度／参数／恢复案例首轮均通过，最终模型组为 35 个不同成功用例。
- [Quality 生命周期 XML](tests/lifecycle.xml)：共享 CLI prepare→start100→resume200 与连续200 的参数摘要和所有更新指标相同。
- [修复前两项](tests/before_fix.xml)、[定向首轮](tests/targeted.xml)、[定向最终50项](tests/targeted_v2.xml)和孤立路径诊断均原样保存，不重复计入235。

每个 XML 的对应原始 `.log` 在同一 `tests/` 目录。[归档清单](archive_manifest.json)保存原路径、长度及 SHA256；副本保持原始字节。没有复制完整模型、Replay 或行情库。

[双父提交源码与来源复核](review/review_static_and_source_binding.json)：E 来源准入核心与 Reader 验证函数 AST 保持目标原样；九个 Quality 核心模块与已验收来源逐字一致；231 个运行源码文件匹配 Git，并与实际 prepared／Run 身份一致。[合并前分支／Tag／工作树](review/before.json)、[保护复核](review/git_protection_review.json)和[旧版本锁](review/baseline-lock.before.json)保留。最终双仓 SHA 位于集成工作区 `baseline-lock.json`，源码变更只在 `integration/s3-next`，旧分支／Tag 不改写。

[真实 1M 来源的 CPU prepare](quality_source_prepare/manifest.json)、[来源准入](quality_source_prepare/source_admission.json)与[构建成本](quality_source_prepare/quality_build_performance.json)证明合并后可以重新认证原正式 Tutor Full 1M 来源并构建 FG201 Quality 准备快照；未初始化或训练 GPU 网络。

[Quality CPU 生命周期](quality_lifecycle/receipt.json)为合成原生监督，200 Env／4 更新；[每次梯度](quality_lifecycle/updates.jsonl)及[训练收据](quality_lifecycle/training_receipt.json)。保留 Full 模块的实际 CPU 前向／七模块梯度、冻结／恢复及参数复制见[联合头收据](model_evidence/test_all_heads_gradient_and_ex0/joint_cpu_receipt.json)、[HEADS_ONLY](model_evidence/test_quality_first_heads_only_0/heads_only_receipt.json)、[warm-start](model_evidence/test_quality_actor_parameter_w0/warm_start_receipt.json)。三个 E FullCheckpoint 的 `checkpoint_evidence.json` 也在 `model_evidence/`。

[真实 E 共享 CLI 验收](e_cli/acceptance.json)：新小 CPU 网络、canonical Single3 Gate，FG201／FG301 共2200 Env／6更新；[独立冷恢复比较](e_cli/cold_resume_comparison.json)为 exact=true，三个 E 编码分支与 Actor 均实际更新，推理参数同步。[完整命令](commands/e_cpu_commands.json)与 prepare／start1100／resume2200 原始输出在 `commands/`。beta=1，为教师控制工程验收，不作为 Actor-only 交易成绩。

[真实原生账户／Oracle／Replay](e_native_replay/receipt.json)、[完整入场 Fold 风险](risk/entry_fold_risk_receipt.json)与[右删失](risk/censored_fold_risk_receipt.json)分别记录独立的数据、交易与风险合同。

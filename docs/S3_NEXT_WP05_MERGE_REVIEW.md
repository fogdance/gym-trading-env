# integration/s3-next：WP05 双仓合并审查

日期：2026-10-18。用户授权合并；本次为 DEV 代码集成。

## 合并范围与真实起点

| 仓库 | 集成第一父提交 | WP05 第二父提交 |
| --- | --- | --- |
| Gym | `0e4af493bcbbfddad1dfcf5eb8b30b5b81a7a65b` | `61dd64b5428588868f374389d4f3d79e0845cdac` |
| Dreamer | `fd144113ce706e2d8f1ce8e7828ece0799c8bf7a` | `be7ce01ebbe63144e4d335b1603ca73da4fa1c79` |

核对时四棵相关工作树均 clean。直接在 `fg-s3-next-integration/{gym-trading-env,dreamerv3}` 的 `integration/s3-next` 合并。WP05 的 WP01–WP04 祖先保留；没有 squash、rebase、改写旧 Run 或操作冻结生产/训练树。最终双仓 SHA 与 clean 核对保存在提交后生成的 [结果收据](S3_NEXT_WP05_MERGE_RESULT.json)。

唯一 Git 冲突为同名 V3 路线图 add/add。保留集成侧 DATA V3 原文件，将功能分支版本存为 `FG_S3_E_Quality_Master_Roadmap_20261018_V3_WP04_Baseline.md`。新统一状态见 [V8](FG_S3_E_Quality_Master_Roadmap_20261018_V8.md)。运行源码无文本冲突。

## 实测问题、根因与最小修正

1. 原 WP05 夹具和内置 Probe 优先引用机器上旧 WP04 的 MACD V2 工件。合入现有 MACD V3 后，真实 `inspect_f1` 抛出 `F2_F1_REWARD_OBSERVATION_CLOCK_MISMATCH`。保留此拒绝；测试及内置 Probe 改用当前原生 Env/Teacher/Reader 和既有 F1 Run 生成 4 次合成更新的工程来源。没有迁移、重签旧 Checkpoint 或建立第二套训练器。另覆盖显式 V2 Observation 拒绝负例。
2. WP03 CLI 正例输出到 `fg-s3-next-integration`，命中既有 `independent_output` 禁止规则。使用原集成提交的完整 `e_data_v1/io.py` 执行同一路径，复现同一错误；该文件及诊断 `_output` 函数均与基线相同。将正例输出改为临时目录，并加入集成目录拒绝负例，保护规则不变。见 [基线复现](receipts/wp05/Integration_baseline_path_reproduction.json)。
3. 历史真实小样配置同样引用 MACD V2 Corpus，当前首先在课程合同校验被拒绝。更新负例为断言实际的 `COURSE_INVALID observation.macd_5m`；另用两个当前合同下的合成来源验证 `F2_F1_CORPUS_COURSE_MISMATCH`。两个拒绝均不产生 prepared/run。

[逐字节核对](receipts/wp05/Integration_preserved_DATA.json) 确认集成侧 6 个 Gym 与 1 个 Dreamer DATA/MACD 运行源码完整保留。金融撮合、Quality TP Reward、MACD 数学、F1/F2 来源准入、公开 E+Quality Guard 没有为测试而改动。

## 实际验证

命令从兄弟 Gym 执行，固定 Dreamer Python、CPU 和单 worker BLAS/OMP 线程。精确 argv、环境、时间、双仓 HEAD/dirty、原始日志与逐例 JUnit 均在下列收据；[命令全文](S3_NEXT_WP05_MERGE_TEST_COMMANDS.md) 可核对。

| 实际收据标签 | 范围 | 结果 | 退出码 |
| --- | --- | --- | --- |
| [Integration_data_reader_v2](receipts/wp05/Integration_data_reader_v2.json) | MACD V3/partial、市场时钟、DATA/DEV 控制、Join/Reader/CLI | 102 PASS | 0 |
| [Integration_shared_v1](receipts/wp05/Integration_shared_v1.json) | 必跑 Gate/SourceTransfer/Formal Guards；FullCheckpoint/S3C 恢复 | 33 PASS | 0 |
| [Integration_f1_v1](receipts/wp05/Integration_f1_v1.json) | F1 准入、原生优化器/冻结、K/T、冷恢复、身份负例 | 53 PASS | 0 |
| [Integration_legacy_quality_v1](receipts/wp05/Integration_legacy_quality_v1.json) | 原 Quality Loss/预测/Heads-only/迁移行为 | 13 PASS | 0 |
| [Integration_wp05_v2](receipts/wp05/Integration_wp05_v2.json) | 完整 WP05 文件集合，CLI 旧预期仍加载在该进程中 | 75 PASS；1 FAIL（第 3 项旧预期） | 1 |
| [Integration_cli_v3](receipts/wp05/Integration_cli_v3.json) | 修正后的完整 CLI 文件，含旧 V2 及当前来源不匹配负例 | 9 PASS | 0 |
| [Integration_probe_v1](receipts/wp05/Integration_probe_v1.json) | 真实 CLI、当前合同合成 F1→F2、4 次 F2 更新 | PASS | 0 |

采用每个测试模块最后一次执行结果，当前选定集合为 **278 个不同 CPU 用例 PASS，0 个未解决 FAIL，0 SKIP**；不是全项目测试通过，也不是单条全套命令退出 0。CLI v3 替代旧 CLI 模块的结果，其他 WP05 模块来自 v2 的实际 PASS。[清单](receipts/wp05/Integration_final_test_inventory.json) 保留选择规则及全部用例。原始失败收据未删除：`Integration_contract_v1` 为 14 PASS/1 ERROR、exit 1；`Integration_data_reader_v1` 为 100 PASS/1 FAIL、exit 1；WP05 v2 的旧 CLI 断言失败如表所列。

测试在合并待提交树执行。除 shared_v1 启动时尚未修正内置 Probe 外，其余最终收据的运行源码与提交候选逐字节一致；shared_v1 覆盖的 Gate/迁移/Checkpoint 源码一致，变化的 Probe 模块由后续真实 CLI Probe 和 CLI 测试验证。见 [源码范围核对](receipts/wp05/Integration_test_source_audit.json)。没有把旧版本 PASS 直接当作本次 PASS。

Probe 的四个预测头各 5 个参数叶发生变化；285 个冻结叶保持字节相同，原生优化器更新 4 次；真实 TRAIN 参数更新为 0。F1/F2 冷恢复和跳过无有效标签后的 Reader/counter 恢复见 [本次证据](receipts/wp05/Integration_evidence)。测试覆盖未知参数/slot、非有限更新回滚、Owner/Stepid/动作时序、Q/Soft 隔离、burn-in-only 无目标、未授权/伪造收据无副作用等负例。

旧 WP04/WP05 收据原字节保留，本次测试覆盖写出的 5 份结果另存于 `Integration_evidence/`。功能分支提交后的历史最终复验及完整提交锁也原样复制入 docs，仍是历史版本证据。

## 自审及限制

已检查新增合并适配的完整 diff、非 docs 的 whitespace 检查及未解决冲突列表。改动仅涉及内置合成 Probe、测试来源/预期和文档；没有降低生产准入。历史 Markdown 换行和原始失败日志中的空格原样保留。

本次 **CODE_INTEGRATION_GO**；**真实 TRAIN 当前版本读取/Native G3、真实 F1/F2 训练、GPU、TEST、18 合约生产、Actor 研究效果均 NOT_RUN_THIS_MERGE**，FORMAL/RESEARCH 仍 NO_GO。没有检查或更改正在使用的生产进程，没有启动正式训练，没有 push 或导出 patch。历史 V2 工件继续冻结；后续应使用 V3 对应的新 Corpus 和真实 F1 来源独立验收，不得把本次合成来源冒充认证底座。

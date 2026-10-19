# WP07：在线接续工程实施报告

范围：任务书 V2 的 A–H。CPU 合成工程开发；没有真实 TRAIN 参数更新、GPU 训练、TEST 行情读取、18 合约生产，也没有开放公开 E+Quality 或正式 RL Gate。

## 结论与版本

SHADOW、ONLINE_SUPERVISED、ONLINE_RL_CONTROLLED 已接入同一个原生 Agent、Env/Driver、Replay、优化器和 FullCheckpoint。A–G 已逐阶段提交；H 的统一 CLI、回归和隔离发布审查进行中。最终发布结果、A–H 成对 SHA 和集成 HEAD 由 `WP07_COMMIT_LOCK.json` 补齐。此段在发布验收完成前不代表主线已更新。

**正式训练 / RL 研究：NO_GO。** 缺少本轮可验证的真实受训 F3、两份正式发布的 V3 TRAIN Corpus、WM/imagined mask 足够支持量和独立研究授权。CPU 数值更新及合成 TP/SL 不是策略有效性或盈利证明。

原始共同基线为 Gym `2129afa2d21e23f0bf915c15e41f649919f4931e`、Dreamer `208a499a7a3afa1dfd240e51de5c6a1e27dd1c47`。已经包含 WP06 和新版 DATA-MULTI 的 F1/F2/F3 Reader；没有从任务书较旧 HEAD 重做。完整 HEAD/dirty/worktree/AGENTS 审计在 [WP07_BASELINE.json](WP07_BASELINE.json)。开发分支两仓均为 `feat/s3-e-quality-wp07-20261019`，路径为 `/home/v/Documents/work/research/fg-s3-e-quality-wp07-20261019/` 下的兄弟仓库。

## 改动清单

| 阶段 | 实施 | 主要位置 |
| --- | --- | --- |
| A | 版本化 execution；明确 F3 online continuation / controlled F4；检查实际完成 F3 的完整账本、模型、上游和冻结来源；全部 non-opt 参数级继承，fresh optimizer/counters/RNG/account/Replay | Gym/Dreamer `fg_s3_e_quality_online.py`；原 `fg_s3_e_quality_transfer.py` |
| B | 原 V2Collector 的 teacher-free 适配，原 ActorPolicy、Driver、Replay 和统一 trace row；真实 mask、Actor logits、账户、E/MACD 快照、执行反馈；原 chunk 留档 | Gym `fg_s3_e_quality_online_collector.py`、`fg_replay.py` |
| C | 私有原生 L1/L2 post-step 认证和延迟发布；pending 有效位为 false；原 WP03 投影和 canonical Join；在线原 Replay 窗口 + 原 DATA-MULTI F3 Reader | Gym `fg_s3_e_quality_online_data.py`；Dreamer `fg_s3_e_quality_online_stream.py` |
| D | 直接复用原 F3 K 步因果预热、features/distribution/objective/train 和原优化器；仅 quality_actor 更新；无标签跳过、非有限候选整体回滚 | 原 `fg_s3_e_quality_actor.py`、在线显式 train dispatch |
| E | 从原 Agent 提取单一 imagination objective；只在显式 F4 RL 使用 quality_actor，real/imag 分布和 loss 共用；Val/slowval 和相关 normalizer 条件更新 | Dreamer `agent.py`、`fg_s3_v2_agent.py`、`fg_s3_e_quality_online.py` |
| F | 四个独立 Gate、原正式 mask 阈值和支持量、多步原生 reward/continue/mask 校准；真实 mask 内显式探索；独立 RL 审核和授权 | `fg_s3_e_quality_online_gate.py`、Gym `fg_s3_e_quality_online_checks.py` |
| G | 原 FullCheckpoint 原子保存 live Replay/Env/Driver/账户、历史 Reader/RNG、Prefetch、policy/exploration、publication/consumption、phase/budget/Gates、日志提交边界；strict latest 不回退 | 原 `fg_checkpoint.py`；薄 `OnlineRun(ActorRun, LiveRun)`、`fg_s3_e_quality_online_state.py` |
| H | 原统一入口 online-prepare/start/resume/probe/evaluate/report；显式 draft 非认证诊断；正式来源、权限和同版本恢复；复用原多 Corpus ABI 兼容检查并保留每份完整课程身份 | Gym 原 `tools/fg_s3_v2_experiment.py`、配置 README、隔离测试/审查 |

金融撮合、金融时钟、原生 Quality TP Reward、E/MACD 计算、Oracle 算法没有修改。原 `RL_MODES`、原 `ONLINE_ADAPT` 课程要求、Prediction/Q/Decision Admission 保留。未新增平行金融 Env、Replay、模型优化器或训练系统；在线协调器调用原 `HeadsRun.update` 和 `LiveRun.collect_to`。

## 每阶段真实 CPU 证据

准确 argv、cwd、环境、双仓 HEAD/dirty、执行源码摘要、耗时、退出码及逐 case 列表见 [H_STAGE_TEST_INDEX.json](receipts/wp07/H_STAGE_TEST_INDEX.json) 和同名 `.log/.xml`。下表是选定测试集合的结果，存在集合重叠，不能相加宣称独立测试数或全项目通过。

| 收据 | PASS | exit | 范围 |
| --- | ---: | ---: | --- |
| Baseline_shared | 20 | 0 | 未修改基线的共享 Guard |
| A_contract | 10 | 0 | source、合同、fresh 参数接续和拒绝 |
| B_shadow_archive | 21 | 0 | 两段 native SHADOW + 原 WP03 Reader 19 例 |
| C_data_stable | 38 | 0 | delayed 标签、DATA-MULTI F3 和 WP03 恢复；2 条 pytest JUnit 元数据警告 |
| D_supervised_candidate | 3 | 0 | 真实 BC 更新、slots 冻结、非有限回滚 |
| E_rl_stable_initialization | 2 | 0 | 同一 Actor、真实 imagined RL/Critic 更新、RL_OFF |
| E_shared | 20 | 0 | Agent 共用 imagination 重构后的共享 Guard |
| F_gates_identity | 2 | 0 | 原生校准、伪认证拒绝、mask 内探索和 RNG |
| G_uuid_restore_regression | 3 | 0 | native UUID 跨 hash seed、chunk/sampler、最新节点/外部身份拒绝 |
| G_native_cold_binding | 1 | 0 | 一个协调测试实际执行四类新进程冷恢复 |
| H_cli | 4 | 0 | CLI 正负例、真实 bounded 合成运行、报告及无教师评价 |
| H_four_cold | 4 | 0 | 四模式冷恢复 + core 3 例 |
| H_shared | 119 | 0 | 必跑三项 Guard、原 FullCheckpoint/Driver、市场分钟、课程 Draft/正式、WP03、MACD dated tail |
| H_macd | 14 | 0 | partial EMA、未来修改前缀不变、重复查询、跨夜/休市与 native 恢复 |
| H_f1 / H_f2 / H_f3 | 18 / 20 / 26 | 0 / 0 / 0 | 原 F1/F2/F3 实际优化器、冻结、负例、全新进程 CP 与下一 batch |

实际入口示例（从对应 Gym cwd 执行，LABEL 必须唯一）：

```bash
env FG_WP_RECEIPTS=wp07 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py LABEL tests/entry_v1/test_wp07_contract.py
env FG_WP_RECEIPTS=wp07 FG_WP07_G_ROOT=artifacts/fg_s3_v2/e_quality_wp07/NEW_COLD FG_WP07_CASE_RECEIPT=NEW_cold_cases.json /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py LABEL tests/entry_v1/test_wp07_checkpoint.py
```

验证器固定 CPU、隐藏 CUDA、BLAS/OMP 每进程 1 线程，独立场景并发。F1/F2/F3 回归在独立 cwd 生成收据，不覆盖旧报告。资源耗时来自真实命令计时，没有重复资源探测；例如 H 四冷恢复 160.78 秒、H_shared 222.98 秒，三组旧阶段回归各约 267–372 秒。

## 参数、标签和账户证据

- [D_actor_delta.json](receipts/wp07/D_actor_delta.json)：4 次实际在线监督更新；有效自然标签 ADD=12、WAIT=116。quality_actor 5 个参数 leaf 和其 10 个 slot leaf 变化，外加共用优化器计数；其余模型/slots 全字节相同，包括原 WM、四预测头、Q、旧 pol、Val。
- [E_rl_delta.json](receipts/wp07/E_rl_delta.json)：2 次实际合成 RL 更新。quality_actor 和 Val 参数/slots、slowval、获准 normalizer 状态变化；旧 pol、WM、Quality/Q heads 冻结。RL_OFF 时 Value/slowval/normalizer 不变。真实与 imagined logits 相同，任意改变旧 pol 不改变这两条策略路径。
- [C_publication.json](receipts/wp07/C_publication.json)：Actor 全部 WAIT 时，原生 strict Teacher 仍独立发布 ADD=4、WAIT=44；pending=1，保留无效位。认证使用 post-step account/cursor 和完整 TRAIN suffix，既不是本行动作，也不是本行是否 TP；prefetch 不算 optimizer exposure。
- [B_shadow_0.json](receipts/wp07/B_shadow_0.json)、[B_shadow_330.json](receipts/wp07/B_shadow_330.json)：两段各 128 个真实合成市场分钟，共有 TP=16、SL=15，含持仓/合法 WAIT/ADD 和官方交易日切换；原 native `reconcile_trace` 均 PASS，模型/slots 不变；Teacher 构造、查询/state 路径注入 raise 后仍可完整交易。
- H CLI 合成小样：64 市场分钟，消费 8 batch、有效更新 7 次、显式 disabled-loss skip 1 次。随后原无教师 argmax 评价 32 分钟：真实 entry fill=5、TP=2、SL=2，reward=-5.7，完整账户重播 PASS，参数不变。此处只证明闭环，不能视为优质策略。

不同账户/来源/episode 绝不配 Pair 或拼接时间；Natural/Aux/online 按源实曝计数，历史预取/被替换的读者行与真正消费分开。多账户同价格不是多份独立市场样本。数据身份、模型状态和 checkpoint 文件摘要在最终发布 evidence inventory 中给出；原 CP/Replay 大文件保留在隔离树 artifacts，不上传或复制旧 Run。

## 精确恢复与失败修复

[G_cold_cases.json](receipts/wp07/G_cold_cases.json) 是 G 原始四场景；[H_cold_cases.json](receipts/wp07/H_cold_cases.json) 是 H 再验。SHADOW、online BC、刚发布 7 条标签中断、RL optimizer slots：连续 64 分钟与 32+新进程恢复+32（发布场景按 publication phase 中断）的动作/交易、完整模型/slots、Reader/Prefetch/RNG、账户/Driver、chunk/item/sampler、标签账本、日志字节和下一 batch 均一致。

以下早期失败原始日志保留，没有 skip/xfail 或放宽金融准入：

1. 原 B fixture 只有 LONG，无法覆盖要求的做空 SL；改正合成方向与路径。原 reconciler 的 RNG 不一致来自初态不是实际 seeded account；保存并复用真实初态。
2. C 测试 Dreamer 路径错误；另一次冷恢复期间修改了被绑定代码，恢复正确拒绝。路径和验证时冻结策略修正，见 `C_cold_failure_review.json`。
3. Ninjax 初始化 discovery 与 grad 的创建模式不同，仅靠 `nj.creating()` 选择策略导致模型 schema 不同；完整初始化使用稳定独立作用域，实际 F4 更新始终走 quality_actor。
4. 新进程创建 Teacher 之前未加载执行初态，原 fork 证明含 RNG，造成身份不匹配；现在先导入同一真实 account snapshot，再构造原 Teacher，不删除身份字段。
5. 原 native `elements.UUID` pickle 保留另一个进程缓存的 Python bytes hash。实际新进程恢复后，同 16-byte stepid 的新 UUID 无法查字典。load_replay 重建 UUID 的派生 hash，保留每个 ID 和原容器/数组；不同 `PYTHONHASHSEED=11/22` 正例通过，移除修正的同源码 mutant 确实失败。
6. H 源码复核：完整 course identity 包含各 Corpus 的来源，不能把三份合法独立来源强制为同一个完整哈希。现在复用 DATA-MULTI 原 Observation/model/Reward/Teacher 兼容合同，分别保存每份不可变 course/source/index 身份；不修改 Corpus，不削弱正式发布/G3 要求。与源模型 ABI 不匹配仍拒绝。

最新 node/manifest、外部 Reader、原生 chunk、日志前缀、代码/来源/配置不符均拒绝；strict latest 不静默退回旧节点。未提交日志尾巴先保存在 recovery_logs，再恢复提交边界。F3→online 是 fresh 新 Run，同 Run 恢复从不重复 handoff。

## 正负准入与未执行项

显式结构合法 draft 返回 exit 0、`CONTRACT_ONLY / NOT_ADMITTED`，只产生 `contract_diagnostic.json`；来源缺失/未验证 blockers 按实际模式列出，无 Prepared、Run、Checkpoint 来源认证或训练副作用。缺少 spec_mode、空串、非法路径、TEST、未知认证字段、formal 缺源、合成 F3 冒充正式均非零退出。详见 H CLI 收据和 [验收矩阵](WP07_ACCEPTANCE_MATRIX.md)。

| 项目 | 结论 |
| --- | --- |
| CPU 合成 SHADOW / BC / 受控 RL 工程 | 按上述实际收据 PASS；最终隔离发布结果另行记录 |
| 真实受训 F1/F2/F3 权重用于 online | NOT_RUN / NO_GO；未提供可验证来源 |
| 两份正式发布真实 TRAIN Corpus 融合终验 | NOT_RUN / NO_GO；不以 WP02/06 报告替代 |
| 正式 online prepare/start/resume 正例、正式 RL 授权正例 | NOT_RUN；真实 native 证明、来源和独立审核缺失，默认关闭 |
| WM/imagined mask 研究校准充分性 | NO_GO；保持原 min_train_samples=25000、20 连续报告及原阈值，未以小样凑 GO |
| GPU 更新、真实 TRAIN 长更新、18 合约生产 | NOT_RUN；本轮不执行 |
| TEST/独立策略考试、真实 RL 改善、盈利结论 | NOT_RUN；留给 WP08/09/10，不能从 loss finite 推断 |
| 全项目测试 | NOT_RUN；仅本文与最终发布清单所列集合 |

本轮 metadata-only 检查见 [H_real_readiness.json](receipts/wp07/H_real_readiness.json)：已检查的正式 s3_v2 Run 目录没有提供 F3 来源，历史 B01 路径不存在，现存 FG809 DEV 小样仍不是两份发布教材。没有打开 TEST 价格或扫描未发布半成品的价格内容。

WP08 可复用：原 deterministic evaluate_native、exact checkpoint、per-source exposure、post-step publication、frozen WM 和 actor-only 证据。正式研究启用必须补齐真实来源、充分校准和单独授权；不能把工程 permit、JSON PASS 或旧 Tutor 1M 当作这些条件。

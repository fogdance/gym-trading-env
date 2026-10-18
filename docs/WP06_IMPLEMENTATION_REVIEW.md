# WP06 实施 Review（2026-10-19）

结论：A–F 核心工程已完成，最终独立集成 **96 个 WP06 case + 20 个共享 Guard case 全部 PASS，退出码均 0，无失败/跳过**。原始命令和逐 case 清单见 `Release_test_index.json/.tsv`。**工程可运行不等于真实策略有效，FORMAL_TRAIN / RESEARCH 仍 NO_GO。** 所有实际参数更新仅来自明确标记的合成 fixture；真实 TRAIN 更新为 0，未读取 TEST 行情、未生产 18 合约、未解除公开 E+Quality Guard。

最终测试代码为下表 F 成对 SHA；此后的发布提交只归档 docs，不改运行源码/配置/测试。发布前后 runtime_hash 相同的继承证据和成对提交清单见 `WP06_COMMIT_LOCK.json`。完整冷恢复产物保存在隔离发布树 `artifacts/fg_s3_v2/e_quality_wp06/release_cold_01`；完整无 Teacher 账户回放位于 `release_eval_01/actor_trace`。256 份持久工件逐文件 SHA 和四组 CP node 见 `Release_artifact_inventory.json`，账户/动作文本副本在 `Release_native_account_evidence/`。这些是工程证据，不是正式 Corpus/Checkpoint 认证。

## 基线、目录和提交

先只读核对实际双仓 HEAD/status/worktree 和 WP05 祖先关系，再在隔离集成树完成 G0。Gym 主线已包含 WP04/WP05 和 MACD V3，避免重复合并；保留了期间新增的 DATA B01 审计与 Roadmap V9。

- G0 公共 Gym：`7ea95563166bf7fa03c6ce836e7484053cac0b96`。
- G0 公共 Dreamer：`0e39dfd498f9146e10f966cfe118a24b20e5143e`。
- 开发分支：两仓均 `feat/s3-e-quality-wp06-20261018`。
- 开发根目录：`/home/v/Documents/work/research/fg-s3-e-quality-wp06-20261018/`，兄弟 `gym-trading-env` / `dreamerv3`。未修改旧训练树或真实旧 Run。

| 阶段 | Gym | Dreamer |
| --- | --- | --- |
| A | `389b054a3793183d139254247411c60db1da7044` | `4a662da09edab0a84291ca7c9c9e4b24d5cdbb76` |
| B | `81a1c8f019aab3ed1f95b3f21ddfa8539e94d601` | `e846d7a878530b43b1de539b17a8e673cd180835` |
| C | `035d1628757df24ae95b3b3f3e9fbdd8b55bca20` | `f4104b4f17ea2bc3110840de7bc3ae0df0866601` |
| D | `34549a3fc8369d8cb8c3ad53a3ee439621d6e2ef` | `6d045035591cd614b88853c30387951edd584f4c` |
| E | `2ac709d7344cc7e08f83cd30ccd172f514befbc7` | `085bf9ca0bd7eff11b1676d694ec5bdd686438d1` |
| F | `53ffea2353618be99f35673ab8f4c6c544b7273d` | `ffe6a831877ee91769f611dd16c374fb96941ebd` |

完整 HEAD/dirty、命令、源码摘要和每个 case 清单均保留在各次 `docs/receipts/wp06/*.json/.log/.xml`，不能用历史 PASS 代替当前版本。阶段测试的 pytest 临时目录会被框架清理；最终集成通过显式持久目录保留完整 Checkpoint/账户回放，位置在发布收据中记录。最终集成收据另记测试所用完整成对 HEAD；若最后仅补文档，明确列出运行源码哈希相同的继承证据。

## 改动和复用

| 位置 | 实现 |
| --- | --- |
| Dreamer `fg_s3_e_quality_actor.py` | 显式 F3 合同、损失参数、原 quality_actor、真实 K burn-in、冻结 WM 表征、原优化器正白名单和完整 state 回滚 |
| `fg_s3_quality_v2.py` / `fg_s3_v2_agent.py` | F1/F2/F3 互斥路由；F3 Teacher 只进 ext_space；推理不调用预测头，mask/unimix 一次，fusion delta 为 0 |
| `fg_s3_e_quality_foundation.py` / heads | 抽取既有完整 state owner 和冻结表征公共 helper；F2 行为保留 |
| `embodied/jax/agent.py` | F3 finite 检查、只在成功更新时推进 host update counter |
| `embodied/run/fg_s3_e_quality_transfer.py` | F1 或 F2 完成状态、源冻结/ledger/upstream 证明；完整 non-opt 复制，fresh optimizer/RNG/counter |
| `embodied/run/fg_s3_e_quality_actor.py` | 原 WP03 Reader 同 Corpus 双流组合；F3 仅提供阶段 hooks，继承 HeadsRun 的唯一 update/run/FullCheckpoint 生命周期 |
| `embodied/run/fg_s3_e_quality_heads.py` | 抽出原 runtime 初始化、统计和拒绝 hooks，未新增第二套训练循环 |
| Gym `fg_s3_e_quality_actor.py` | 设置/preflight、实际 native 权限、prepared/start/resume、原生目标窗口适配与 target support |
| `fg_s3_e_quality_reader.py` | G0 V3-only 拒绝；可选逐字段核对 canonical post-step 标签，F1 默认仍不读 Teacher |
| `fg_s3_e_quality_actor_checks.py` | Fresh CPU read-check；原 ActorPolicy→Driver→Env→Replay→reconcile_trace 的无 Teacher 评价；固定合成 probe |
| 既有统一 CLI / configs | `f3-prepare/probe/read-check/start/resume/evaluate`，保留授权和来源校验；集中 README 和明确只读示例 |

金融 Env、Reward、MACD、Oracle、课程数学和原 FullCheckpoint 文件格式未改。对 G0 基线执行的 `git diff --name-only` 检查，Gym envs/rewards/utils/strict teacher/corpus producer、Dreamer course contract/Encoder/fg_checkpoint 均为空。未重新签署已认证 Corpus，也不把 V2 index 改个版本号冒充 V3。

## 实际测试与命令

从开发 Gym 执行，统一记录器实际设置：

```bash
env FG_WP_RECEIPTS=wp06 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py LABEL TEST_PATH...
```

记录器的真实 pytest 命令：

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q \
  --junitxml=docs/receipts/wp06/LABEL.xml TEST_PATH...
```

| Receipt / 输入测试 | 实际结果 | 退出码 |
| --- | --- | --- |
| G0 contract/Reader/clock/MACD + F1/F2 cold + shared | 126 个不重复 CPU case PASS；另一次 4 case 是其中的重跑 | 0 |
| `A_contract`：test_wp06_contract.py + test_wp05_contract.py | 32 passed | 0 |
| `B_transfer`：test_wp06_transfer.py + 原 B01 F1→F2 case | 25 passed | 0 |
| `C_updates_retry`：test_wp06_updates.py + 原 F2 C09 allowlist | 13 passed | 0 |
| `D_run`：test_wp06_losses/sampling/run.py | 16 passed | 0 |
| `D_boundary`：losses/sampling（新增 burn-in-only 拒绝） | 14 passed；与 D_run 重合 13 个，共 17 个 D case | 0 |
| `E_evaluation_retry`：test_wp06_evaluation.py | 4 passed | 0 |
| `F_cold_cli`：test_wp06_checkpoint.py + test_wp06_cli.py | 20 passed | 0 |
| `F_shared`：指定 Gate / source_transfer / action_mask_formal_guards 三文件 | 20 passed | 0 |
| `F_temporal_legacy`：课程草案/正式、Reader、market minute、MACD、legacy Quality student/admission | 121 passed | 0 |
| `F_f2_cold_regression` | 原 F2 冷恢复回归，8 passed | 0 |
| `F_partial_after` | stop_at 未完成预算标 PARTIAL；先有失败回归再修正 | 0 |
| `F_cpu_probe`：统一 CLI 固定合成 probe | 4 次原优化器更新，5 个 Actor 参数变化，冻结 state 相同 | 0 |
| `Release_wp06`：全部九个 test_wp06_*.py | 96 passed，816.57 秒 | 0 |
| `Release_shared`：指定三文件 | 20 passed，187.74 秒 | 0 |
| `F_real_read`：统一 CLI f3-read-check | 有限 TRAIN 只读 LIMITED，零更新 | 0 |

不是全项目测试通过。完整 test ID 与计数以原始 XML/最终测试索引为准，不对重跑 case 重复计数。未运行的实际研究验收不计入 PASS。

保留失败：`C_updates` 1 failed/12 passed（测试字段名误写）、`D_losses` 1 failed/6 passed（测试错误计入 <15 分差）、`D_loss_sampling` 5 failed/8 passed（9 步 reset 不支持 12 步窗口）、`E_evaluation` 1 failed/3 passed（诊断 PRNG 与训练不同）、`F_partial_before` 1 failed（stop_at=2/预算4仍错误报告 COMPLETE）。最后一项仅修正结果状态为 PARTIAL，计数/准入不变；after 1 passed。没有跳过、xfail 或降低准入。修正后分别重跑并保留两次证据。

## 参数和优化器证据

- `B_F1_handoff.json` / `B_F2_handoff.json`：完整 non-opt keys、每键 source/target SHA 一致；新 optimizer 全部保持 fresh、seed/Reader/counters 从零，policy mirror 同步，源 agent.pkl 不变。两类源均标 `FOUNDATION_NOT_TRAINED / ENGINEERING_ONLY`。
- `C_actor_delta.json`：未改写的原 Reader L2 标签进行两次更新；quality_actor 的 5 个参数 leaf、10 个专属 slot leaf 变化，4 个共享 clocks 变化；所有 owner=None 的完整参数和专属 slots 零变化。
- 全无效批次（包括已有动量）、非零冻结动量/负零、非有限候选、未知参数/slot、注入 WM 动态 mask 均有实际负例。非有限整次状态回滚，并记录消费位置；不得不断重试同一坏批次。
- `D_actor_run.json` 从 D11 实际 FullCheckpoint 读取：2 次成功/2 次消费，Natural 有效 16（ADD 1/WAIT 15），辅助有效 15（ADD 1/WAIT 14）；strict Pair 6，L1 一致 Pair 3，候选 10/过滤冲突 7。不是独立市场样本数。

## Teacher、因果与采样

原 V3 Join/project_label 是唯一标签投影。适配器逐 stepid 将 teacher_action/valid、Owner、score/valid/outcome_valid 对照 index；prefix 为零标签。错位、Owner 替换、原生 observation/action/reward/stepid 改写在进入 JAX 前拒绝。原生行为 action 仅作为前一执行动作，不能当当前 L2 真值。

有效 L2 同时要求 teacher_valid、合法 WAIT/ADD、原生双动作 mask `[1,1]`、非 first/last/terminal；不能补造 WAIT。K 个真实历史步经过原 `causal_context/wm_only(training=False)`，T 个目标 id 对齐；Teacher 或 burn-in 标签变更不改变表征，历史输入变更会影响表征。

Natural anchor 永远存在，辅助使用同一 Corpus 的第二条原 Reader。各自 RNG、抽样状态和 prefetched input 均恢复。Natural / auxiliary 暴露、balanced loss 分母、strict Pair、L1 一致 Pair 分开记录；没有辅助流时 auxiliary 暴露为 0。不同 batch/account/segment/Corpus 不能配对。L1 Pair 默认权重 0，只允许高分 L2 ADD→低分 L2 WAIT，分差 >=15；同动作和逆序过滤均有计数/测试。Q/Soft 禁用且不会进入 loss。

Pair/count 是对应合法监督分母；权重为 0 的项只保留诊断计数，不应被读作该 loss 已提供梯度的证据。D 的明确启用消融和默认 Natural-only probe 分开报告。

## FullCheckpoint 精确冷恢复

使用原 FullCheckpoint 的 `external_immutable_course_reader` backend，完整保存原 Agent、optimizer、主/辅助 Reader 状态、Python/NumPy RNG、Agent seed counter、prefetch、schedule 和 counters；不复制 Corpus 为新 Replay。

| 场景 | 成功更新 | 消费 | 证据 |
| --- | --- | --- | --- |
| 连续 4 vs 2 + 新解释器恢复 + 2 | 4 | 4 | `F_cold_ordinary.json` |
| 首次全无效 skip 后退出，冷启动继续；对照连续运行 | 4 | 5，其中 skip 1 | `F_cold_skip_first.json` |

全部 state、下一批输入和 updates.jsonl 字节一致；K32/T64 两流 target positions 为 512 / 640，burn-in positions 为 256 / 320（消费暴露量，不是独立市场 N）。篡改 settings/stage/source/code/index/architecture、损坏最新 bundle、F1 冒充 F3 resume 均在 model/run 写入前拒绝，未静默回退旧 CP。无授权 exit 1，无 Run；错误 CLI 模式 exit 2；工程 source 即使带授权仍不能正式 start。

## Actor-only 工程小测

- 推理与训练调用同一个 quality_actor / 原 mask / unimix；预测函数被置为抛错仍能推理，三个 fusion delta 恒 0，未通过阈值或 P4 接管造 ADD。
- 显式合成特征分类：512 次原 Agent 更新，32 ADD/32 WAIT，BC 0.693907→0.687870，准确率 29/64→35/64；恒 WAIT 为 32/64。训练后 TP=25/FN=7/FP=22/TN=10。仅说明可更新/可学，不能推断真实胜率。
- 原 Env 32 市场分钟，36 行，3 次账户恢复，完整账户/ledger/Reward/E snapshot 重播 PASS。实际合法执行提案 ADD=5/WAIT=4，真实 entry fills=5，原生 TP=2/SL=2。最后提案单列为未执行；TP 是实际金融退出记录，不是 L1 后验计数。
- GPU：`F_gpu_smoke` 在沙箱缺少可见 NVIDIA 设备而于构造前失败，exit 1；经批准在宿主执行 `F_gpu_host_smoke`，真实日志 `failed call to cuInit ... CUDA_ERROR_UNKNOWN` / `No visible GPU devices`，exit 1。未发生 GPU 更新；GPU 更新验收为 NOT_RUN，初始化尝试为 FAIL_ENVIRONMENT。未降级到 CPU 冒充 GPU，也未修改驱动/服务。

## 有限真实 TRAIN 只读结果

实际执行：

```bash
env FG_WP_RECEIPTS=wp06 /home/v/miniconda3/envs/dreamerv3/bin/python -B tools/wp03_verify.py F_real_read --cmd \
  tools/fg_s3_v2_experiment.py f3-read-check \
  --config configs/experiments/fg_s3_v2/e_quality_f3_read_smoke.yaml \
  --output-dir docs/receipts/wp06/F_real_train_read
```

仅现存 FG809 的 4 分钟 DEV Corpus（真实 TRAIN 来源）可读；不是已发布正式教材。K1/T1：8 次有效 L2 暴露，全是 WAIT；2 个独立目标 market cursor、7 个账户行。P0/P2_low/P3/P4 读取，P1/P2_mid 无目标支持 NOT_RUN。模型 fresh controlled diagnostic，不消费任何 F1/F2 权重；所有参数 SHA 相同、optimizer_updates=0。Source 为 `INTEGRITY_CHECKED_ONLY / NOT_ADMITTED`。

| 身份 | SHA256 |
| --- | --- |
| 原 Corpus | `636ceada845a4cf5c73be364584147acf99dff951ca52d40b3bd3191aafc7304` |
| 原课程 | `236498452047aba8740f570d0d142728712a862ae0072a58962ba63aa59b6b47` |
| Admission content | `2adff90658da632da52ba27dbb817934097043d2ead3e5c347530088773c4056` |
| 此只读诊断模型配置 | `8415c4f74c221708ad23ba41fb408de02ba9146b3536605d0d8c9e73091c5eab` |

原 B01 路径 `/data/s3_corpus/early_train_20261018_cpu20_v2/batches/B01_FG809` 在本轮不存在，最初 read exit 2；当前已知两个 full-main 目录也没有 corpus_manifest。**真实完整 Corpus K32/T64/G3 验收 NOT_RUN，NO_GO**。历史 WP02/B01 报告和规划时的成功不替代本轮验收，也未读取未发布的半成品。

最终追加只读来源复核：`source_inputs` 实测验证了受信 TRAIN 池/原生 Env 收据、所选 FG809 行及真实 TRAIN 日期集合。完整官方日的 source_lock 与四步 DEV 锁不同；调用原生 `_execution_scope(mode='dev', max_market_steps=4)` 后，逐字段及 content_hash 完全重现原锁，差异列表为空。见 `Release_real_source_scope.json`。未改旧 Corpus 或忽略任何字段；该证明仍不是 Native G3 / Prepared / Checkpoint 认证。

## 反向源码 Review

这是本代理在实现后独立按风险清单复核和原生重播验证，未声称有第二位审阅者。

| 风险 | 代码/运行证据 | 结论 |
| --- | --- | --- |
| 错训原 pol | F3 GROUPS/owner + C 全 leaf SHA | 仅 quality_actor 参数/slots 变 |
| 冻结被 Adam/decay 改写 | 原 masked_joint_optimizer + C02/C04/C05 | 包含 slots、负零和非有限回滚 |
| Q/预测改变动作 | F3 policy_forward 不调用 predict；E01/E03 | fusion 为 0，Teacher-free |
| L1 冒充 L2 | actor_objective 一致性筛选；D02 冲突负例 | 阈值仍 >=15，默认关 |
| Balanced 冒充市场先验 | Natural role / balanced_count / auxiliary_count | 分母与实际暴露分开 |
| 跨账户 pair | 原 pair_identity + V3 canonical Owner；D03 | Owner/segment/batch 隔离 |
| 单帧代替历史 | 原 causal_context + C03 + K32/T64 cold | 真 K 步 burn-in |
| handoff 冒充 resume | 源 non-opt 参数接续；F3 run identity / F04 | fresh 与 exact resume 分开 |
| V2 冒充 V3 | G0 外/内 index protocol 拒绝、重建派生索引 | 无兼容/重签 |
| 合成/小样冒充研究成功 | 本报告、raw receipts、CLI admission | FORMAL/RESEARCH NO_GO |

自审还纠正了测试 fixture、诊断 PRNG、辅助暴露统计和一个测试结果文件与记录器同名覆盖的问题；没有改准入判断掩盖失败。原 CP 证据重新从实际 node 提取并保留路径。

## 未完成与下一条件

1. 完整当前 V3 已发布 Corpus 的真实 K32/T64/Native G3 验收：NOT_RUN（来源路径缺失）；需提供可验证的完整发布位置。
2. 正式受训 F1/F2→真实 F3 更新、真实已训练 F3 Checkpoint 的公共评价：NOT_RUN；当前没有此类来源，也未获授权启动真实长训练。
3. 真正隔离 DEV/TEST 的策略有效性、成本/MAE/稳定性/跨合约消融：NOT_RUN，归 WP08/研究授权；有限 TRAIN WAIT-only 不支持有效性结论。
4. 18 合约全量、DATA-MULTI、F4/RL：未实施，保持后续 WP 边界。
5. GPU 有效优化器更新：NOT_RUN（宿主 CUDA 初始化失败）；CPU 工程结果不继承为 GPU PASS。

下一步先补可验证教材位置和真实 F1/F2 训练准备；另行授权训练与研究。不要用合成 CODE_GO 自动开放任何正式 Gate。

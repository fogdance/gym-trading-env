# WP08 实施与验收报告

本报告由实际源码、命令输出、逐行原生回放和原始收据生成。**WP08_CODE_GO；真实受训 Actor DEV 考试 NOT_RUN；策略质量 INCONCLUSIVE；SEALED_TEST_READY=false。** 成对提交和完整测试范围见文末及 [WP08_TEST_INDEX.md](WP08_TEST_INDEX.md)。任务原文为 [WP08 V2](FG_S3_E_Quality_WP08_Actor_Only_Exam_Final_Codex_gy23_d323_20261019.md)。

## 范围及真实基线

- 独立成对 worktree：`fg-s3-e-quality-wp08-20261019/{gym-trading-env,dreamerv3}`，分支均为 `feat/s3-e-quality-wp08-20261019`。
- Gym 公共基线：`0a425534bc6a06e5bc47f9683c58a08d645d59b1`；Dreamer：`745b7e0c22165769b516f5c91545b032e9511357`。开工 HEAD、dirty、worktree 和 DATA-MULTI 分支审查见 [原始审计](WP08_BASELINE.json)。最终集成前再次查验，未从落后分支另行实现 Reader。
- 实施顺序为 A → B → C → D → G → E → F → H。复用原 Env、Quality TP Reward、MACD V3、ActorPolicy、Driver、Observed、Chunk、Trace、参数构建及 Checkpoint 来源验证。
- 未修改金融 Env、Reward、MACD、Teacher/Oracle 或正式训练/RL Gate；未读取 TEST 价格、启动 GPU 或真实 TRAIN 参数更新、修改历史 Run、生成 18 合约教材。

## 已实现的调用链

`tools/fg_s3_v2_experiment.py exam-*` → `research/fg_s3_e_quality_exam.py` → 独立 ExamSource / ExamFrozenActor → 原生 Env + Driver + ActorPolicy → 原始 Trace / reconcile_trace → 交易风险与分块对照报告。

考试合同为 `FG_S3_E_QUALITY_EXAM_PLAN_V1`。哈希纯函数不访问文件系统。显式区分 SYNTHETIC_DEV、DEV_EXAM；SEALED_TEST_FINAL 在打开行情前拒绝。可声明每合约的 `contract_windows`，不同合约不必具有重叠主力日期，同一合约所有策略的日期、费用、初始资金完全相同。

DEV 来源由原始 validation 白名单、原生 Env 收据、市场/合约文件、官方日和前缀初始化重新验证。TRAIN Corpus、自签 JSON、软链或工程模型不能代替正式 DEV 来源/模型。旧 C1 仅提供已认证的金融配置，未读旧 C1 Oracle/Checkpoint；Reward 为原生 Quality TP。

F3/F4 通过原来源核验及 `load_exact` 加载全部参数，包括 F4 Value、slowval、slots、normalizer。唯一 ActorPolicy 使用真正的 quality_actor masked argmax，无老师、探索或 Prediction/Q 修正。旧 pol 变动不改变动作。Eval 副本的 train 方法直接拒绝调用；模型、policy 参数、训练计数和原始来源文件在每次提交和结束时校验不变。推理 RNG/动作计数独立于原训练 seed，所有策略使用预注册的共同评价 seed。

每合约一个持续账户、Driver 和 RSSM carry。256 行存储块不重建账户；实际落单是上一观测提出的 action，决策、下一步执行反馈和成交 Entry ID 显式 Join。多合约分配不同 Owner/carry/RNG。指标完全来自原生成交和已观察价格，不将 Teacher 候选当成交。

EvalSession 保存独立 `FG_S3_E_QUALITY_EVAL_STATE_V1`：Env/账户/可见 MACD 构建态、Driver/RSSM carry、待执行动作、基线 RNG、原生 Chunk 标识、已提交块/日志哈希、指标聚合和冻结身份。权重与 optimizer 不放入评价状态；训练 FullCheckpoint 只是只读模型来源。严格 latest、源/代码/块校验，不回退旧状态；崩溃产生的未提交尾块移到 recovery 目录留证。

六个 CLI 为 exam-plan/run/resume/verify/compare/report。verify/compare 不仅重新做原生金融对账，还重新加载原声明策略，逐步复算动作和 p_add，拒绝合法但非该 Actor 输出的动作篡改。每个输出目录不可覆盖，resume 必须明确指定已有 Session；默认数据生产输出保护不变。

## 数据与模型实际状态

| 项目 | 实际结果 | 边界 |
| --- | --- | --- |
| E/MACD V3 validation FG409 | 原始白名单、行情、官方日、成本与 E 前缀可用；32 次真实分钟推进，并原样 reopen 核验 | 来源工程检查，未运行真实受训 Actor DEV 考试 |
| 有限 TRAIN FG809 | 32 次原生 WAIT 推进、33 行完整对账 PASS | SYNTHETIC_DEV / FINITE_TRAIN_READ_ONLY_NOT_DEV_EXAM，无训练更新 |
| F3/F4 SHADOW/BC/RL 来源 | 使用 WP07 留存的完整 CPU 合成来源，实际模型参数与 CP 核验通过 | 工程权重，不可用于正式 DEV 证据 |
| 正式受训模型 DEV 考试 | NOT_RUN | 未提供本轮经认证的正式 F3/F4 模型 |
| 策略收益/泛化、TEST | INCONCLUSIVE / NOT_RUN | 不自动授予研究、训练或 TEST 权限 |

最终真实来源命令收据为 `receipts/wp08/Release_real_dev_source.json`、`Release_real_dev_reopen.json`、`Release_real_train_readonly.json`；早期迭代保留 H_source_* 与 H_finite_train_readonly。原生来源验证会解析/核验所选 validation 或 TRAIN 文件，实际有限账户推进均为32分钟，不声称仅打开32行。完整产物留在两组 WP08 worktree 的 `artifacts/fg_s3_v2/e_quality_wp08/`；docs 保存索引/摘要/哈希，避免复制完整市场或 Checkpoint。

## 实施中观察到的问题及修正

1. 初次真实来源使用工作区软链别名，被现有 `_local_path` 正确拒绝。改为真实规范绝对路径后重新运行，未放宽路径校验。
2. 初版测试 fixture 的到期/roll 日期、相对路径和 JAX carry 比较方式不符合原接口。修正测试输入及比较工具后原生回放通过；原金融规则未改。
3. F4 RL 来源包含原 OnlineRun 合法设置的 `avail_actor_gate/value`。源码确认其为原生控制状态；按原 Gate/工程身份验证并精确保留，未把它当成新训练更新或自行开放 Gate。
4. CLI plan 后 run/resume 需重新打开已拥有的输出目录。共享 `independent_output` 增加显式 `allow_existing`，默认仍拒绝既有目录；考试调用方另验计划/Session/代码身份。默认保护和覆盖负例均保留。
5. WP07 H 回归初次将输出放入新 WP08 路径，被原 `ONLINE_PROBE_NEW_ISOLATED_OUTPUT_REQUIRED` 拒绝。未修改公共基线复现同一拒绝后，仅修正测试隔离目录到原允许位置。原 Guard 未变。
6. 最终审阅发现日分块 MAE15 平均值曾把不足 15 分钟的右删失观测混入完整风险均值。新增原生 3 分钟删失回归实际复现后修正有效分母；修复和重测收据单独保留。

失败尝试的原始日志和命令/退出码均保留。新命名的验证另存各自 XML；复用 WP06/WP07 worker 的分组 XML 为最后一次运行结果，各次命令和日志另外归档。旧 PASS 不自动等于最终版本 PASS；实际命令、源码哈希、dirty 和逐 case 记录见测试索引。

## 反向源码 Review

以任务书 12 项 NO-GO 路径单独审阅调用链，并用全新进程行为复验。源码边界审计见 `receipts/wp08/H_source_review.json`；冻结集成树另有 `Release_source_review.json`，逐文件核对实际集成源码与已审阅功能树一致。它们只证明改动归属，不替代功能测试。审阅由本执行者完成；第三方人工 Review 未执行，不冒充另一位审阅者。

- 未调用 OnlineRun/OnlineCollector 执行 DEV，未重标 TRAIN fold；旧 online-evaluate 仍为原有限 TRAIN 诊断。
- 来源阶段 F3/F4 分别校验，合成许可不能升级正式。模型 schema、课程、Observation、Reward、完整 slots 和 native 控制状态均保留。
- Teacher 访问失败探针、训练方法拒绝、旧 pol 变异和 legal-action 篡改负例均有实际测试；每步策略动作另行重算。
- 长账户不靠分段 collect_trace 拼收益。新进程恢复验证逐行 stepid、账户、carry、推理 RNG、模型 bytes 与指标。
- TP 分母为全部实际 Entry，包括 SL、roll 和右删失。每个时间窗的有效位单列；未知结果不会被当作确定负例。
- 全 WAIT 的 TP 率 N/A、NO_ENTRY_SKILL_EVIDENCE；少样本 UNDERPOWERED，多个 seed/账户不重复计行情独立样本。
- 比较只使用 WAIT/ADD/多 seed Random/Delay 与合法冻结 Actor；P4 Oracle 明确拒绝。
- 工程合成和有限 TRAIN 能继续开发，但正式 DEV 未运行、策略质量未证明。TEST 入口保持拒绝；未为测试通过降低训练或来源门槛。

## 发布判定

源码冻结版本为 Gym `09898736602c1a7d6c89899bb79fe232d7907923` / Dreamer `b7a693b3c7b0b45eeecdec954818be42d40e67b6`，独立集成 worktree 已在这对 HEAD 上全部通过最终复验。主线两仓在集成前核查仍为原始基线且 clean，发布只允许成对快进；实际发布结果与文档归档后的公共 HEAD 另记提交后生成的本机 `WP08_DELIVERY.json`，不伪造包含自身 Git SHA 的提交内容。

| 最终验证 | 通过数 | Exit | 原始收据 |
| --- | ---: | ---: | --- |
| 冻结合同、F3/F4、风险窗口、完整CLI、原生roll | 40 | 0 | Release_cli_risk.json/log/xml |
| 1024精确新进程恢复、Random恢复、严格拒绝 | 3 | 0 | Release_exact_cold.json/log/xml |
| Teacher失败探针、1024持续账户 | 1 | 0 | Release_native_teacher_isolation.json/log/xml |
| 强制共享Guard、FullCheckpoint、Driver、MACD | 51 | 0 | Release_shared.json/log/xml |
| WP07 A–H + WP06短评价 | 33 + 2 | 0 | H_final_previous.json；各H_final_regression_*命令/日志 |
| 原扩展共享Reader/课程/Clock/恢复集合 | 150 | 0 | H_shared.json；共享运行源码相同的继承证据 H_shared_source_inheritance.json |
| 每合约区间 / 质量分级等受影响模块 | 43 | 0 | H_final_targeted.json（与上面部分重复） |
| 日分块MAE有效位修复 / 跨日TP / 账务 | 9 | 0 | H_daily_risk_fixed.json |
| 10账户同源对照 / 共同评价seed | 3 | 0 | H_final_fair.json；H_fair_risk_valid.json以最终指标重做全部原生对账 |

明确选定集合去重后为 **239 个 case，0失败、0跳过**，清单见 `receipts/wp08/FINAL_SELECTED_CASES.json`。这不是全项目测试；包含已有模块的相同运行源码继承及受影响模块复验，不能把各行简单相加。H_shared 的2个 warning 来自 pytest `record_property` 与 xunit2 格式，未跳过测试。早期失败按上文保留。

六类实际 CLI 拒绝：TEST=1、空模式=1、TRAIN冒DEV=1、缺正式Actor=1、授权训练flag=2、GPUflag=2。各自输出目录均不存在、无 agent.pkl/Prepared；精确 argv/原始日志和文件检查见 `Release_negative_summary.json`。另有来源阶段混淆、坏Checkpoint、旧pol变异、合法动作篡改、损坏latest/chunk/源身份的函数级负例，均由对应 pytest 实际验证。

仅发布 WP08 工程代码。`WP08_DEV_EVIDENCE_GO=NOT_RUN`、`STRATEGY_QUALITY_GO=false/INCONCLUSIVE`、`SEALED_TEST_READY=false`，正式训练与RL Gate不开放。下一研究步骤是提供真正受训、来源可验证的F3/F4模型，并预注册独立DEV计划；本次不自动启动它。

### 八阶段真实提交

| 阶段 | Gym 完整 SHA | Dreamer 完整 SHA |
| --- | --- | --- |
| A | `32157b325df2cc5bf907b77d18e4e1439f0eb9b8` | 无改动 |
| B | `f56a4fa3490613e647d9efc6ff13053636c39e2a` | `2666d847463c46f2948fc2d9cae4e61e550019e9` |
| C | `7d7414aebc04c4ef0de383f3494d8452dd2888ab` | 无改动 |
| D | `a63c028030b9316fa213b2a34f09460ba57504eb` | 无改动 |
| E | `b08105d0eb1a6b196895d008f5244ae67d5d10bb` | `26be0506c344af139c11ea0160b65f2135a2e27b` |
| F | `cc12783bf88d748882a5979ad22fa1bb3a570b78` | 无改动 |
| G | `a84a9e595b66ff9d66e3ff18bc4fa5cc8a87dbcf` | 无改动 |
| H | `09898736602c1a7d6c89899bb79fe232d7907923` | `b7a693b3c7b0b45eeecdec954818be42d40e67b6`（接口文档） |

未制造空提交。最终文档归档提交仅增加报告/收据，运行源码与上述冻结版本相同。原 EvalSession 继续绑定冻结 HEAD；其恢复使用保留的 `integration/s3-e-quality-wp08-release-20261019` 双仓，不能把文档发布后 HEAD 的新 Run 冒充旧 Session resume。

### 已核实的账户与恢复数字

冻结版本的合成固定 ADD Actor 连续推进 1024 分钟，生成 1025 行：132 个实际 Entry、71 TP、60 SL、1 右删失。净 PnL −89,067，手续费 2,367，最大权益回撤 89,913；原生 Quality TP Reward −128.01406250000008，二者分别对账。总权益变化为闭合交易净 PnL −89,058 减仍持仓开仓费 9。该负收益压力样本证明工程账务，不是策略有效性。

1024 与 333+新进程691 对完整 snapshot、Driver/Env、stepid、RSSM carry、policy RNG、模型摘要、日志和指标做递归精确比较，无容差；实际推理状态为 `FG_POLICY_INFERENCE_STATE_V2 / seed=202 / actions=1025`。随机基线另做 64=23+新进程41。严格 latest 损坏、chunk/来源变化拒绝，未提交尾块保留。

完整证据见 [WP08_ACCOUNT_RECOVERY_EVIDENCE.json](WP08_ACCOUNT_RECOVERY_EVIDENCE.json)、[WP08_IDENTITIES.json](WP08_IDENTITIES.json)、[45项映射](WP08_ACCEPTANCE_MATRIX.md)、[命令及原始结果](WP08_TEST_INDEX.md)。实际合成比较为10账户、4个独立合约日、132条 paired block 对照；F3/F4四个工程来源在该样本交易结果相同，未证明策略提升。

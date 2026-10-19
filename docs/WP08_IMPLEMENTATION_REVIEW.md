# WP08 实施与验收报告

本报告由实际源码、命令输出、逐行原生回放和原始收据生成。当前处于最终发布复验阶段；最终成对提交和判定以文末发布记录及 `WP08_TEST_INDEX.md` 为准。任务原文为 [WP08 V2](FG_S3_E_Quality_WP08_Actor_Only_Exam_Final_Codex_gy23_d323_20261019.md)。

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

真实来源收据：`receipts/wp08/H_source_reopen_ready.json`、`H_source_exact_reopen.json`、`H_finite_train_readonly.json`。完整产物留在本 worktree 的 `artifacts/fg_s3_v2/e_quality_wp08/`；docs 保存审阅所需索引/摘要/哈希，避免复制完整市场或 Checkpoint。

## 实施中观察到的问题及修正

1. 初次真实来源使用工作区软链别名，被现有 `_local_path` 正确拒绝。改为真实规范绝对路径后重新运行，未放宽路径校验。
2. 初版测试 fixture 的到期/roll 日期、相对路径和 JAX carry 比较方式不符合原接口。修正测试输入及比较工具后原生回放通过；原金融规则未改。
3. F4 RL 来源包含原 OnlineRun 合法设置的 `avail_actor_gate/value`。源码确认其为原生控制状态；按原 Gate/工程身份验证并精确保留，未把它当成新训练更新或自行开放 Gate。
4. CLI plan 后 run/resume 需重新打开已拥有的输出目录。共享 `independent_output` 增加显式 `allow_existing`，默认仍拒绝既有目录；考试调用方另验计划/Session/代码身份。默认保护和覆盖负例均保留。
5. WP07 H 回归初次将输出放入新 WP08 路径，被原 `ONLINE_PROBE_NEW_ISOLATED_OUTPUT_REQUIRED` 拒绝。未修改公共基线复现同一拒绝后，仅修正测试隔离目录到原允许位置。原 Guard 未变。
6. 最终审阅发现日分块 MAE15 平均值曾把不足 15 分钟的右删失观测混入完整风险均值。新增原生 3 分钟删失回归实际复现后修正有效分母；修复和重测收据单独保留。

所有失败尝试的 log/json/xml 均保留。旧 PASS 不自动等于最终版本 PASS；实际命令、源码哈希、dirty 和逐 case 记录见测试索引。

## 反向源码 Review

以任务书 12 项 NO-GO 路径单独审阅调用链，并用全新进程行为复验。源码边界审计见 `receipts/wp08/H_source_review.json`，它只证明改动归属，不替代功能测试。审阅由本执行者完成；第三方人工 Review 未执行，不冒充另一位审阅者。

- 未调用 OnlineRun/OnlineCollector 执行 DEV，未重标 TRAIN fold；旧 online-evaluate 仍为原有限 TRAIN 诊断。
- 来源阶段 F3/F4 分别校验，合成许可不能升级正式。模型 schema、课程、Observation、Reward、完整 slots 和 native 控制状态均保留。
- Teacher 访问失败探针、训练方法拒绝、旧 pol 变异和 legal-action 篡改负例均有实际测试；每步策略动作另行重算。
- 长账户不靠分段 collect_trace 拼收益。新进程恢复验证逐行 stepid、账户、carry、推理 RNG、模型 bytes 与指标。
- TP 分母为全部实际 Entry，包括 SL、roll 和右删失。每个时间窗的有效位单列；未知结果不会被当作确定负例。
- 全 WAIT 的 TP 率 N/A、NO_ENTRY_SKILL_EVIDENCE；少样本 UNDERPOWERED，多个 seed/账户不重复计行情独立样本。
- 比较只使用 WAIT/ADD/多 seed Random/Delay 与合法冻结 Actor；P4 Oracle 明确拒绝。
- 工程合成和有限 TRAIN 能继续开发，但正式 DEV 未运行、策略质量未证明。TEST 入口保持拒绝；未为测试通过降低训练或来源门槛。

## 发布判定

最终检查尚在进行。本段将在冻结版本复验和成对集成后更新，不能将当前草拟报告当作发布准入收据。

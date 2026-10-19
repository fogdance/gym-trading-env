# FG-S3 WP07 → WP08 源码接口级独立 Review

**日期：2026-10-19**  
**依据：** 用户上传的 `gy23.tar.bz2`、`d323.tar.bz2` 的完整 Git 对象与 `integration/s3-next` 源码；归档内的 `docs/WP07_IMPLEMENTATION_REVIEW.md`、测试收据，以及此前的 WP08 预设计。**仅静态审阅源码、文件和既有收据；未在用户主机执行真实行情回放、CPU/JAX模型更新或GPU训练。**

## 1. 真实基线与证据边界

| 类型 | Gym | Dreamer |
|---|---|---|
| 最新 `integration/s3-next` | `0a42553`（完整SHA见 Git） | `745b7e0`（完整SHA见 Git） |
| WP07实际测试/集成代码冻结提交 | `1a28f0a20866ea2f6c2d38f31a963b42ca92155a` | `232e57f70b2751adfc9733d7e695d834b73ce439` |
| 最新提交区别 | 运行代码无修改，补充大量 docs/收据 | 运行代码无修改，补充 docs |

报告中 `247 PASS` 为 **选定 CPU 合成工程测试**，不代表所有项目测试，也不代表已受训 F1–F4、真实 DEV/TEST 或策略收益达标。报告给出了 D阶段仅 quality_actor 更新（4次在线监督）、E阶段 quality_actor + Value/slowval 等受控 RL 更新（2次）、原生账户/恢复证据，属于本次设计可用的功能基础。在线正式研究仍 NO-GO。

## 2. WP07 已实际提供的可复用接口

| 真实源码 | 可复用能力 | WP08 应如何复用 |
|---|---|---|
| Gym `research/fg_s3_e_quality_actor_checks.py::evaluate_native()` | `ActorPolicy → collect_trace → reconcile_trace`、Teacher-free Argmax、p_add、基本TP/SL、参数摘要；**1–512步** | 复用 policy/trace/account primitive；不直接把短程诊断当成考试 |
| Gym `research/fg_s3_e_quality_online.py::evaluate_checkpoint()` | F4完整节点 `validate_bundle()`、Agent `create_agent(...,execution_mode=...)`、`load_exact()`、验证模型/代码身份；**TRAIN原生来源、≤512步** | 提炼“冻结Checkpoint→只读推理实例”的加载协议；独立注入合法DEV考试Env，不调用原 `_live_inputs()` |
| Gym `research/fg_s3_e_quality_online_collector.py::OnlineCollector` | 真正Teacher-free执行、带carry和账户的完整原生trace，记录mask、p_add、stepid、chunks | **不能直接用作DEV Collector**：构造器检查 `fold == TRAIN`、Quality TP与E Profile，设计独立只读EvaluationSession复用其底层V2Collector/Driver/原生trace协议 |
| Gym `research/fg_single3_evaluation.py::ActorPolicy` | `agent.policy(...,mode='eval')`、policy carry与诊断 sidecar；不修改Observation | 唯一神经策略适配器，明确关exploration/fusion；Eval初始化同步policy快照后再做不可变性比较 |
| Gym `research/fg_replay.py::collect_trace(...,initial_state,end_cursor)`、`reconcile_trace()` | 原生单账户追踪、account state、完整金融回放对账、E_QUALITY协议 | 用于短/中段考试；长区间必须增加可续跑Session/chunk和区间拼接**账户状态**，不得拼接多份“新账户”PnL |
| Dreamer `embodied/run/fg_s3_e_quality_online_run.py::OnlineRun` | SHADOW/在线BC/在线RL、在线账户/Replay/标签发布、FullCheckpoint原子恢复 | 只复用检查与状态协议思路，**不得创建训练OnlineRun充当考试进程** |
| Dreamer `embodied/run/fg_s3_e_quality_online.py::OnlinePolicy` | 内含原始 ActorPolicy 和可选探索 RNG | 考试使用纯ActorPolicy或明确 exploration_rate=0 并验证无任何探索；不得按在线训练探索配置读取 |
| Gym `fg_s3_v2_evaluation.py::prepare_dev` / `fg_single3_evaluation.py::require_dev` | 已有独立DEV来源、历史FG409、原生检查经验 | **不能直接用现行旧C1/FG409专用准备协议**，必须为E+Quality/MACD/V3构造新DEV来源与因果E Profile入口 |
| Dreamer `embodied/run/fg_checkpoint.py::validate_bundle`、F3/F4源检查 | 完整参数/来源/同版本身份核验 | 只读加载模型参数；考试运行状态单独恢复，不复制或写入训练optimizer |
| Gym `tools/fg_s3_v2_experiment.py` | `f3-evaluate` / `online-evaluate` 等现有CLI，严格用法 | 新建独立 `exam-*` 命名空间，不暗改已有≤512的工程诊断语义 |

### 2.1 当前可确证的模型来源

- F3模型：`fg_s3_e_quality_actor.py`、`fg_s3_e_quality_actor_checks.py` / `embodied/run/fg_s3_e_quality_actor.py` 的来源身份、参数schema与已完成F3运行核验。
- F4模型：`fg_s3_e_quality_online.py::evaluate_checkpoint()` 的 `run_identity.json`、`validate_bundle(node, identity)`、`agent.pkl`、`create_agent(...execution_mode=...)`、`load_exact()`。
- F3/F4推理都必须最终使用 `quality_actor`，不能从 `pol` 读取或更新策略；两个节点的完整身份**不同**，需明确源种类，不可把F4节点交给F3同Run恢复器。
- 若F4是SHADOW，模型不更新；F4在线BC与RL具有不同策略/Value参数变化合同；考试均禁模型/slots/normalizer写入。

## 3. 阻断性接口差异与推荐修正

**I-01 [P0] DEV数据的 E+Quality 入口不存在。** `online._live_inputs()`从已认证TRAIN Corpus和TRAIN初始账户构造环境；`OnlineCollector.__init__()`拒绝非TRAIN。`fg_s3_v2_evaluation.prepare_dev()`是固定旧C1 FG409的DEV流程。WP08必须新增 `ExamSourceAdmission`：根据实际已存在的validation白名单与原生市场/合约/env配置，生成E/MACD V3兼容**只读**观察；必须记录并验证训练/DEV日期隔离；初始资金/方向/roll/官方日、合法E Profile warmup由同一金融源实现。绝不把训练Corpus Sidecar变成DEV教师。

**I-02 [P0] WP07公开在线评估仅是TRAIN有限诊断。** `online.evaluate_checkpoint()` 断言 `steps<=512`、`_live_inputs()`绑定TRAIN，并标识 `FINITE_TRAIN_READ_ONLY_NOT_RESEARCH`；F3 `evaluate_native()`也限制512。必须新增原生长区间考试Session，不得只改大步数上限或重命名 `scope`。

**I-03 [P0] WP07 OnlineRun包含更新和在线Teacher生命周期。** 不能复用 `OnlineRun.run_to()`当考试执行器；即使SHADOW不更新，它的来源仍是TRAIN，持有新Replay/OnlineRun来源语义。WP08设计只有 Env/ActorPolicy/Driver/Trace/Checkpoint ReadOnly，Teacher factory被注入raise也能考试。

**I-04 [P0] 模型推理不等于恢复训练运行。** F3/F4选中完整Checkpoint后，只复制到inference state；不resume训练计数、scheduler、Env训练账户、训练Replay；`sync_policy_from_train`若用于内部初始化，应发生在不可变快照记录之前并验证没有再更新训练参数、normalizer或policy。

**I-05 [P0] 长区间资金与carry跨段一致性。** `collect_trace()`是一次独立Driver/账户：每段若重新reset将凭空重置仓位与资金。WP08按 **同合约一个连续EvalSession** 处理，跨分块保存Env/account、RSSM carry、部分5m EMA/预热、native events、观察cursor和日志提交边界。跨合约一定要全新账户与隐藏态。必须做连续N vs N1+新进程恢复N2逐动作/资金完全相同测试。

**I-06 [P0] 金融结果不可替换成后验候选。** 交易结果只能由native fills/exit attribution、真实账户状态计算，teacher L1/L2只能在考试后作隔离分析，不能进入action或统计成交分母。TP时间/MAE早期逆行/日内效果/右删失各自有清晰有效位；不能把持仓结束的admin chunk boundary当SL/TP或重置。

**I-07 [P1] 旧DEV/质量分析不覆盖全部目标指标。** `fg_single3_evaluation.analyze_trace()`已有PnL/Reward对账实例，但其 schema 和旧C1、单合约固定DEV绑定；新评估从 `E_QUALITY_PROTOCOL` 原生流水实现TP5 ≤5/15/30/60市场分钟、MAE 5/15、early adverse10、underwater、官方日、净收益/费用/回撤及 MAIN_ROLL/RIGHT_CENSORED，有效位与trade/account双分母。

**I-08 [P1] 纯评价要有自己的Checkpoint。** 不能把 EvalSession state写入F4训练 FullCheckpoint；需要只读模型引用+可恢复的考试账户/策略carry/基线RNG/流处理偏移/已提交receipts。模型和训练checkpoint字节必须不变；严格latest、失败保留、不能默默覆盖旧考试。

**I-09 [P1] 评估与决策参数分离。** 坚持 `quality_actor` deterministic argmax、合法mask、无teacher/探索/fusion；对于无合法ADD的强制WAIT和法律上可ADD但学生主动WAIT，必须分开分母。All-WAIT、样本不足的明确失败状态不能被“低回撤”掩盖。

**I-10 [P1] 基线不能用P4 Oracle。** AlwaysWAIT、AlwaysADD、Random_p10多seed、Delay15必须与学生有相同原始市场、账户/费用/roll，执行原生mask；同一官方日/合约作配对比较，不以不同交易笔数下裸收益为唯一标准。

**I-11 [P1] 预设计对WP07仍是等待状态，应更新。** WP07公开主线已实际完成SHADOW/BC/RL工程，DATA-MULTI公共F1/F2/F3 Reader已作为WP07已核对基线。Master Roadmap应更新；真实完整V3生产、正式F1/F2/F3受训权重与DEV评价仍不能据报告预填PASS。

**I-12 [P2] DEV考试先做简单可用，不被最终TEST过度审批阻断。** 合成/有限TRAIN工程验收仅验证完整因果/金融/身份；DEV研究可使用单独验证来源；TEST必须由WP10封存。不要为开发过程每次运行 `free -h`/`nvidia-smi` 或全部正式G3；正式研究来源/金融不可降低标准。

## 4. 反向 Review 要点

1. **将 `online-evaluate` 换名复制**：只跑TRAIN和512步，NO-GO。
2. **直接实例化 TRAIN 限定 OnlineCollector 骗过role判断**：风险为DEV考试混入Teacher/训练源；NO-GO。
3. **修改旧训练环境为评估并跑 `OnlineRun`**：可能更新模型/Replay和自签训练身份；NO-GO。
4. **评估session重启从新账户开始**：阶段拼接导致PnL、carry和MAE错误；NO-GO。
5. **无教师无探索仅靠配置声明**：必须运行时注入Teacher-query失败探针和参数字节快照，NO-GO if touched。
6. **将回测胜率当交易能力**：需要真实TP速度、SL/MAE/风险、净成本、非退化ADD与跨合约日支持，且区分合成和真实。
7. **DEV/TEST混用**：DEV可迭代但必须角色清楚，TEST只在WP10固定模型与评价后明确开启。
8. **“MACD V3”不等于已拥有正确DEV E Profile**：实际源格式及可见时刻必须重新验证，不得凭训练Profile身份直接套到DEV。
9. **同日与日历午夜混淆**：夜盘属于相应官方交易日；日内完成只计质量，不改变强制TP/SL金融规则。
10. **以旧训练Checkpoint恢复评价**：训练slots、RNG与更新数不得在考试运行产生改变。

## 5. Review结论

**WP07工程基线：CODE_GO（以随归档测试收据为依据）；WP08：可以开始正式代码开发。** 首版应先实现真实的 E+Quality DEV 数据准入、只读模型/原生金融长区间考试、质量统计与可续跑评价，而不是重复WP07训练系统。真实DEV来源和F3/F4受训Checkpoint尚未在本轮提供，故WP08正式研究成绩 `NOT_RUN`，训练/TEST均无新授权。

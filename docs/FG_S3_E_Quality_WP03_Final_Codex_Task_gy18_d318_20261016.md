# FG-S3 WP03 最终 Codex 实施任务书 — V2 Teacher Supervision Admission & Single Replay Reader

**版本**：Final V2 / 2026-10-16  
**状态**：已基于 `gy18/d318` 的 WP02 实际源码完成接口级静态 Review；**可以交付 Codex 开发**，不代表 WP03 已实现、真实数据已在本审阅环境复测或正式训练获准。  
**配套**：`FG_S3_WP02_to_WP03_Interface_Code_Review_gy18_d318_20261016.md`（接口审查）、`FG_S3_E_Quality_Master_Roadmap_20261016_V1.md`（今后持续更新的 WP01–10 状态）。  
**工程主旨**：**不做旧模型兼容，不重造 Env、Oracle、Replay、MACD、训练器。一次把新WP02 Corpus→Teacher SourceAdmission→真实 Replay/Sidecar Join→统一课程 Reader→新 E+Quality Agent Loss 的输入通道打通。**

## 0. 先读这些真实仓库，锁定双仓版本

WP02真实最终功能分支均是 `feat/s3-e-quality-wp02-v2-20261016`：

| Repo | WP01提交 | WP02最终功能HEAD |
|---|---|---|
| Gym | `2904f3e58cd84fb383f9e5a0d2d2aa643fcb10d2` | `e57189bad100465c9fee8d9bbb5601f3e5c376ad` |
| Dreamer | `347724e3207fca16907facab021302217f47cf6b` | `44a6cc62032811501631e83be977f07137dc65b8` |

**重要**：本次Git归档的 `integration/s3-next` 分别仍为Gym `6939de5b...`、Dreamer `51ed1d0...`，并**未合入WP01/02**。Codex首先只读检查用户真实机器两个WP02 worktree 的 `git status --porcelain`、`rev-parse HEAD`、`log -8`、`worktree list`，打印 `base_head`。在两仓 WP02 **相匹配** 的提交上新建独立 WP03 功能 worktree，不 reset/clean/rebase 原树，不从 integration 的落后HEAD开始，不自动合并。

项目最新约束：新模型、新Obs V2、新教材不需要保持旧V1/旧E-1M Checkpoint兼容。原金融规则、Quality TP Reward、S3 MarketClock、TRAIN因果/隔离、真实准入/FullCheckpoint身份始终不能被破坏。

## 1. 已经存在的真实接口，必须优先复用

| 代码实际入口 | 本任务中用途 | 不要犯的错误 |
|---|---|---|
| Gym `src/gym_trading_env/research/fg_s3_e_quality_corpus_cli.py::verify_corpus` | 原WP02真实来源、account、suffix及全部发布文件的只读重演，签发新Reader前必须验证 | 不在每个批次重新运行昂贵Oracle/Env回放；以一次预检/证书冻结依据 |
| Gym `fg_s3_e_quality_corpus_cli.py::_publish_streams`, `_verify_publications` | WP02 发布分离的`student_replay/`与`teacher_sidecar/` | 不重新构建另一种Replay文件格式 |
| Gym `fg_replay.py::trace_manifest`, `iter_replay_rows`, `iter_sidecar`, `replay_fields`, `E_QUALITY_SCHEMA` | 读取原 `.npz`、20字节StepID、gzip posterior label | `iter_replay_rows`已验证 Chunk UUID/succ/offset；不自己凭行号编 stepid |
| Gym `fg_s3_quality_catalog_v2.py::validate_corpus_label` | 校验 `FG_S3_E_QUALITY_TEACHER_V2` 包含L1/L2的新合同 | 旧 `validate_label()`只适用于旧扁平结构，不能直接当V2验证器 |
| Gym `fg_s3_quality_teacher_v2.py::NativeQualityTeacher.query_state` | L1字段语义证据（此包不调用老师计算） | 不将 MAE 隐式并入 Quality Score |
| Gym `fg_s3_e_quality_strict.py::QualityStrictTeacher.query_values` | L2 strict与资金域、理性数值witness、Suffix身份（此包不生成老师） | 不将 missing strict WAIT 填0冒充证书 |
| Gym `fg_s3_quality_catalog_v2.py::FIELDS`, `attach_quality` | 与现有模型 loss-ready字段表对齐的参考；复用字段定义与共同校验工具 | **不能**原样用 `attach_quality` 的旧label映射为新L1/L2 |
| Gym `fg_s3_v2_collector.py::V2Collector.join` | 复用既有有效字段/teacher对齐思路；必要时抽取通用只读辅助函数 | 不把在线Collector当成Corpus生成器或附加Train准入 |
| Dreamer `dreamerv3/fg_s3_v2_agent.py::_S3V2Model.training_loss_fn`, `ext_space`, `quality_loss` | 真正的现有Agent训练批次入口，`replay_context=1`和`effective_stepid`必须保持一致 | 不为WP03新写优化器/Agent，不能为了Smoke解除公开Gate |
| Dreamer `dreamerv3/fg_s3_quality_v2.py::pair_identity`, `teaching_objective` | Owner/Reset合法Pair与BC、质量排序的真实消费路径 | 旧Qv2 utility/Soft启发式未经新目标认证，一律禁用Loss |

### WP02实际文件结构（每个policy_id均P0/P1/P2_low/P2_mid/P3/P4）

```text
CORPUS_ROOT/
  corpus_manifest.json                 # FG_S3_E_QUALITY_CORPUS_V2; training_admitted:false
  course_v2.json
  source_lock.json
  teacher_manifest.json
  model_contract.json
  coverage.json / coverage.csv
  p4_plan.json
  policies/<policy_id>/manifest.json   # 完整原生账户Trace/Policy/financial lineage
  policies/<policy_id>/steps.jsonl.gz
  policies/<policy_id>/replay/*.npz
  student_replay/<policy_id>/manifest.json  # 只读/剥离后验信息
  student_replay/<policy_id>/replay/*.npz    # 与原trace相同原生Replay Chunk字节
  teacher_sidecar/<policy_id>/steps.jsonl.gz# posterior L1/L2，绝不进Policy
```

WP02 V2输入合同：`market_seq_e_shared float32[60,18]`、`market_context_15m float32[24,7]`、`market_macd_5m float32[4]`（DIF、Signal、Histogram、valid）、`position_state float32[9]`、`action_mask float32[2]`。Replay还含`market_seq`（审计用）、action/reward/is_first/is_last/is_terminal/`market_advance`、`stepid uint8[20]`等，必须调用 `replay_fields(E_QUALITY_SCHEMA)` 获得**真实全部字段**，不得自己手写一个不完整Schema。

真实Teacher Envelope：`teacher.protocol='FG_S3_E_QUALITY_TEACHER_V2'`, `teacher.fold='TRAIN'`，`teacher.state_id`, `teacher.account_snapshot_hash`, `teacher.L1`, `teacher.L2`, `quality_q_v2_valid=false`, `prediction_admitted=false`, `decision_admitted=false`。L1保留 event/touch/adverse 各窗口的单独有效位，score_valid等；L2包含严格两动作值、rational witness、资金范围/候选覆盖/结算边界及 invalid reason。

## 2. WP03 必须一次交付的 5 个子功能与接口

### WP03-01：来源准入 / Dataset Receipt（Commit A）

以 `verify_corpus` 已证明的真实来源、完整账户和正式课程签名为基础，创建**新的 SUPERVISION_DATA_READY**数据证书，而非训练权威。推荐接口放在 `src/gym_trading_env/research/` 现有 Source/Quality 模块旁，命名尽量沿用项目风格（确切名字可按源码规范调整）：

```python
@dataclass(frozen=True)
class SupervisionSource:
    corpus_root: Path
    corpus_content_hash: str
    course_identity_hash: str
    teacher_identity_hash: str
    observation_contract_hash: str
    source_lock_hash: str
    stage: str                     # FOUNDATION_WM / QUALITY_HEADS / ACTOR_SUPERVISED

class QualitySupervisionAdmission:
    @staticmethod
    def preflight(source: SupervisionSource, *, verify_native: bool = True) -> dict: ...
```

- 校验 `fold==TRAIN`、`protocol/Schema`、Course V2/E Obs V2/MACD V2/Model/Reward、六个policy ID、L1/L2严格teacher身份、chunk与sidecar哈希、相对路径/symlink/路径穿越、来源字节真实性。
- `verify_native=True` 的首次准入必须**实际调用**既有原生 `verify_corpus` 或消费一个可验证的、同一冻结字节版本且与完整原生重演相绑定的已有可信收据。不能仅相信嵌入的 `status=CORPUS_VERIFIED_FOR_TRAIN_PREPARATION` 或哈希自签。明确记录完整 verify 执行时间与来源；缓存证书必须绑定 code/content/file inventory，一旦源码或源文件变化强制重新验证。
- 完整实验证据在用户本机 `artifacts/.../real_small_04`，当前上传的 gy18/d318 Git归档**不包含原始Corpus**。没有真实Corpus时实施/合成测试可以继续，但REAL_DATA必须NOT_RUN/NO-GO。
- 证书区分：`SOURCE_VERIFIED`、`SUPERVISION_DATA_READY`、`CURRICULUM_SUPPORT_LIMITED`、`NOT_ADMITTED`；绝不输出 TrainingReady/Prepared/ModelReady。正式公开E+Quality训练仍必须拒绝。
- 产生不可变 index of allowed policy/episode/stepid/row/account-owner；索引和hash的建立须可stream，禁止一次性把全部行情和sidecar都加载RAM。

### WP03-02：新 Teacher Envelope → Loss-Ready 投影（Commit B）

**绝不能重复训练老师或重算L1/L2。** 在既有 `fg_s3_quality_catalog_v2.py` 中扩展新版本 adapter（可拆独立 `fg_s3_e_quality_supervision.py` 归研究层），调用已存在的 `validate_corpus_label`，并返回现有Agent接口恰好需要的：

```text
teacher_action                    int32 [B,T+1]
teacher_valid                     bool  [B,T+1]
quality_score                     float32[B,T+1]
quality_score_valid               bool  [B,T+1]
quality_outcome_valid             bool  [B,T+1]
quality_tp / quality_sl / quality_early5 bool[B,T+1]
quality_event                     int32 [B,T+1,7]
quality_event_valid               bool [B,T+1,7]
quality_touch                     float32[B,T+1,4,2]
quality_touch_valid               bool[B,T+1,4,2]
quality_adverse                   float32[B,T+1,4]
quality_adverse_valid             bool[B,T+1,4]
quality_owner                     uint32[B,T+1,8]
quality_q_c1 / quality_q_v2       float32[B,T+1,2]
quality_q_c1_valid / quality_q_v2_valid bool[B,T+1]
quality_soft_target               float32[B,T+1]
quality_soft_valid                bool[B,T+1]
quality_soft_source               int32[B,T+1]
```

注意名字/维度最终需从真实 `FIELDS` 与 `ext_space`派生。**不新增另一套模型字段表**，除非当前真实V2目标在本Schema无法无歧义表达，此时需在双仓同一新版本中完成显式协议更改，并加全过程负例。所有未认证数值可以填数值0占位，但**对应 valid 必须False**，不得用数值0冒充正确WAIT/真实Q。

**严格动作投影**：`teacher_valid=true` 当且仅当L2的`teacher_valid`、双动作均合法、原真实账户状态/stepid/来源/完整后缀全证且`decision_executed=true`、非prefix/terminal/admin reset；`teacher_action=int(L2.strict_action)`，实际行为 `step.action` 不得覆盖老师答案；严格 tie 按已冻结 WAIT 规则；域外理由记录计数。

**L1投影**：独立于L2；`immediate_outcome_verified=true` 才允许已认证 Outcome字段；`quality_score_valid`必须仍取L1原始证据；event/touch/adverse各窗口各自读取 `*_valid`，不能因为L2无效而抹掉L1，不能把MAE塞进原Score；`quality_early5`含义应按既有Catalog一致解释为首次不利10点出现在前5有效分钟（若当前定义不同，写合同映射并验证）。

**L2数值Q**：新后缀的 `action_value_add/wait`、`margin`和witness保存在只读监督source sidecar/证据索引，以 `objective='discounted_native_float32_Quality_TP_feedback'` 标注；不能塞进旧 `quality_q_v2`（`UTILITY`语义不等）。本包默认 `quality_q_c1_valid=false`, `quality_q_v2_valid=false`, `quality_soft_valid=false`, `quality_soft_source=INVALID`。WP06训练配置同时强制对应Q/Soft Loss权重为0，前后做负例检查。未来若显式升级Q合同，由独立WP小任务证明目标、量纲、discount、边界一致后才启用。

### WP03-03：原生物理Replay + Sidecar 三方精确Join（Commit B）

允许一个共同Reader先读取原生 Chunk、师傅Sidecar与完整 Trace，最终返回**同一个** `[B,T+1]`模型数据Batch：

```python
class EQualityCourseReader:
    @classmethod
    def open(cls, admission_receipt, *, stage, sampling_spec, seed): ...
    def sample(self, batch_size: int, effective_length: int) -> dict: ...
    def coverage(self) -> dict: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state) -> None: ...
```

- **绝对不从相邻价格/时间推断账户身份。** 三方按 `policy_id + trace_hash + physical stepid + ordinal + state_id/account_snapshot_hash + legal mask + market cursor/date` 关联。`student_replay`的 manifest 被刻意剥离后验和账户信息，必须调用对应 `policies/<policy_id>`的真实trace作为owner根证据。原学生Chunk必须字节一致；提供反向损坏/调换Chunk测试。
- `stepid`原物理 UUID+offset，`dtype=uint8[20]`；当前Replay值 `action`对应实际Env进度，**不是当下TeacherAction**。WP02 `capture()` 先把当前Obs与L1/L2标签保存，等下一回调才填 `decision_executed`。直接按行查 Sidecar，但需以**下一次推进/当前决策已执行**证明是否允许监督；prefix/terminal/unexecuted反馈都置valid=False；不得把teacher标签整体右移或左移一行。
- 应保留P0/P1/P2低中/P3/P4的同市场多账户。必须证明episode边界，`CHUNK_END`是行政结束可合法继续而非真实交易终止；需要正确按原生reset处理及历史上下文；非法账户/时间跨越不拼接。[B,T+1]首行replay_context是前缀，`training_loss_fn`丢弃第0位置，只对有效的后T位置施加Loss。
- `quality_owner`用于Pair分组，应由同一真实账户episode/suffix/teacher和合法reset段计算，稳定于账户内多个合法状态；**不可**直接用每一行不同的`account_snapshot_hash`作为Pair group。仍需把变化的account hash保留在join key中。**严格有效或质量有效的样本不能具有全零`quality_owner`**；所有无效样本不得因为默认全零owner相同而构成合法Pair。用原 `pair_identity(labels,first)` 的源码实测 same-owner valid、different-owner invalid、跨reset invalid。
- Teacher/Q/MAE等仅在Train Reader的supervision view可见；Policy input严格由原 E Observation keys生成。Teacher Sidecar任何篡改不能改变纯 Actor-only policy前向。`market_macd_5m`需保持原 student Replay 字节，不在Reader重新计算。
- 同一行两个冲突 teacher，丢失teacher，重用stepid，错误policy/account，同官方日不同账户，来自DEV/TEST或过期Teacher的Join一律fail closed；某些原生prefix物理row合法无标签时允许读入WM，但loss-valid全为false并明确证据。

### WP03-04：统一 NATURAL / STRATIFIED / MIXED Sampler（Commit C）

```text
sampling_stage = WM_NATURAL | HEADS_VALID | ACTOR_STRICT | ACTOR_QUALITY_PAIR
sampling_mode  = NATURAL | STRATIFIED | MIXED
```

- NATURAL：保持实际账户行为和自然行情，供F1基础WM；按真实长度/episode窗口生成有效 `[B,T+1]`序列。
- STRATIFIED：按真正认证的L1 TP5快/慢、SL30、5/15min早期逆势10、同日/跨日、L2 strict ADD/WAIT、不同账户策略等题型构建惰性索引，**抽样单位为完整合法序列或合法时间窗，不破坏RSSM连续性**。对于某类没有真实例子，只报告`INSUFFICIENT_CURRICULUM_SUPPORT`，不生成伪样本。
- MIXED：按显式配置混合自然锚和分层样本；权重不在本包调优为“最优”。冻结seed、采样配置、source hash、index hash、次数和序列有效位。对同一随机状态`state_dict`/`load_state_dict`精确恢复下一批的stepid序列；restore来源变更强制fail。
- 批次显式有三种不同统计分母：独立市场时间点、真实账户决策、**实际优化器曝光**；WP03没有优化器所以第三项应0，Reader可记录被交付的批次曝光候选量但不能伪称已更新参数。
- **Pair必须实际在同一个 sampled batch & owner/reset domain 内生成**：不能因有全局两类有效标签就谎称具备每批Pair。`strict_pair`与`quality_pair`计数单列；负例检查不同真实账户在同价同时刻仍为0。

### WP03-05：真实 Agent CPU端到端 / 可见性 / 收据（Commit D）

- 使用这轮真实 **E+partial5m MACD+Quality** Agent构造 (fresh init)，通过原 `_S3V2Model.training_loss_fn` `ext_space`/对应`quality_loss`做 CPU init/forward/loss smoke；不修改公开`require_ready()`/Production Prepare/Start/Resume/Eval Guard，不启动训练/优化器参数更新。
- 证明 `data['stepid'][:,1:] == effective_stepid`，Loss shapes/有效位、ActionMask、Chunk reset、来源全部匹配；人为错位一行应得到 `S3_V2_TEACHER_REPLAY_CONTEXT_SHIFT`或等价强制失败。
- 证明修改仅Teacher未来Sidecar不会改变直接Actor-only政策输入或 logits；同一batch改变合法teacher_y/valid仅影响对应教学Loss而非学生观测或原生金融Reward；无效标签不参与损失。
- 做两层测试：**纯合成/受控CPU** 与 **用户真实本地 `real_small_04` 原生文件**。实际真实缺失则该部分 `NOT_RUN`，绝不可凭旧WP02报告替代。
- `real_small_04`启用模型Loss前仍须保持正确性而非研究优效：WP02真实P4严格动作和L1事实存在不意味着学生已学会；本包没有任何模型训练批准。

## 3. 严格工程约束与禁止事项

- 不重新计算 MACD/Oracle/L1/L2、行情图形、技术分析算法、金融撮合；不读取任何TEST行情，不将DEV标签并入监督；不生成全18合约教材。
- 不创建ParallelTrainer或新Replay存储格式；必须复用 `embodied` 原生Chunk、已冻结WP02学生Replay及Dreamer原Agent输入。允许源码级添加Reader、审计及同版本V2投影，但不得双轨随意复制输入Schema。
- 旧 E/Tutor 1M参数全部不用，无旧Checkpoint adapter；**同版本新Batch与Agent的精确身份/冷恢复仍必须有测试**。
- 不申请GPU；允许CPU合成测试/真实有限TRAIN只读核验，以及有限CPU的无优化器模型前向。不能把曾通过CPU验证当作训练已启动。
- E+Quality正式运行仍返回 `S3_QUALITY_E_NOT_IMPLEMENTED`；Prediction/Q/Decision Admission仍保持禁止。

## 4. 开发提交和每提交验收

| 开发节点 | 核心改动 | 本节点最重要的反例 | 代码责任 |
|---|---|---|---|
| **Commit A** | Source/G3复核、Supervisor Admission、数据Index schema、source/owner Receipt | 自签假证书、TEST路径、symlink、改源字节、formal≠train-authorized | 主要Gym，Dreamer只消费结果 |
| **Commit B** | V2 Teacher有效位映射、原Replay/Trace/Sidecar三方Join、完整 `[B,T+1]` Batch | 同MarketCursor不同账户、stepid错位、强制WAIT冒名、未来teacher溢入Obs | 主要Gym；必要时最小改Dreamer ext_space合同 |
| **Commit C** | NATURAL/STRATIFIED/MIXED、合法episode/pair、seed/状态恢复、coverage | 不足正例补假样、跨reset/跨owner配对、重采样身份不稳定 | Gym/现有Embodied Reader集成 |
| **Commit D** | E+MACD+Quality Agent CPU Loss-ready端到端、真实TRAIN小样证明、回归及实施Review | 失效Teacher仍产生Gradient、Q/Soft暗开、修改Teacher影响policy输出、公开Gate变READY | 双仓 |

每个Commit提交前先运行其定向测试与 `git diff --check`，报告**实际命令、真实退出码、testcase列表、被测试代码SHA**；最后合并验收运行完整选定依赖集合。任何测试路径错误必须独立更正重跑并保留失败记录，禁止删除或标记跳过来“通过”。尽量在一条新WP03开发worktree上串行交付，避免多分支在两个仓库造成不一致。

## 5. 最少32类测试（Codex需落成命名用例，不是只列文字）

1. 校验新 Corpus/Course/Obs/Reward/Model/Teacher/source 真实hash一致。
2. 修改Course、source、任何Replay Chunk、MACD字节即拒绝。
3. 任何非TRAIN或TEST路径/内层manifest穿越/symlink，在读取价格前拒绝。
4. 旧V1/旧C1/旧QualityUtility不得冒名当前严格后缀。
5. 只自签Hash不能代替native verify/G3来源。
6. StudentReplay与原policyTrace Chunk逐字节一致。
7. Sidecar与原Trace Gzip逐字节一致。
8. 六policy正确，每个独立账户恢复。
9. stepid真正是uint8[20]且UUID/offset完整链验证。
10. 同一步重复label/缺失label、错误ordinal应拒绝，允许有证明的真实prefix物理行但Loss全无效。
11. 同一MarketCursor不同policy/account owner互换强制失败。
12. L1 state_id/account_hash与L2账户hash不一致拒绝。
13. L2 strict值/rational witness/tie规则不符拒绝。
14. L2双合法失败、持仓forcedWAIT、terminal、现金域外、右删失，都不能产生合法strictWAIT。
15. Sidecar teacher_action 不得覆盖student真实action。
16. 真实捕捉后状态/下一次执行决定标签，错移一步拒绝。
17. L1 Outcome独立于L2 strict validity：L2无效仍可读合法L1窗口。
18. L1 Score原Quality TP公式与TP/SL结算事件一致，MAE不改Score。
19. event[7]/touch[4,2]/adverse[4]各自独立valid；未知窗口Loss为0。
20. 风险早期逆势10、慢TP、同日/跨日标签不伪造。
21. Reader原始MACD原字节不变，invalid=0的投影不污染Encoder。
22. Policy输入不包含任何未来Outcome/teacher/Q/Score/Risk字段。
23. Prefix `[B,T+1]`正确，`effective_stepid`错位强制失败。
24. ChunkEnd行政reset和真terminal、同账户/不同账户episode拼接严格区分。
25. Natural序列不打乱历史，不产生未来价格或补齐假K线。
26. Stratified能够选真实快TP/SL/逆势样本且不破坏episode上下文。
27. Mixed复现seed/weight，source/index改变时拒绝恢复旧sampler状态。
28. strict ADD/WAIT valid计数和自然比例真实，零有效类标`INSUFFICIENT_SUPPORT`。
29. 同owner同episode合格strictPair计数，跨owner/跨reset严格Pair为0；全零默认owner永远不能给有效Pair。
30. QualityPair只来自同Owner且分数差/valid正确，不能冒用L2作用。
31. 所有旧Q有效位、启发式soft `quality_soft_valid`为false；即便有数值也不产生Loss。
32. 无效标签若随机填NaN/巨值不能改变任一合法loss或Actor-only前向；读入端应先在安全投影中剔除/拒绝非法浮点。
33. E+Quality Fresh Agent CPU init, forward, loss均有限；必要Loss梯度路径经过严格检查（**不执行优化器更新**）。
34. WP01 draft/formal、现有准备/开始/恢复/评估 E+Quality Guard保持阻断。
35. Reader实存磁盘/RSS/吞吐记录，证明非全Corpus常驻内存。
36. 实际有限TRAIN六账户数据独立Reader全链验收（源缺失时必须REAL_DATA_NOT_RUN）。
37. 跟踪unique market cursor/unique account decision/actual optimizer exposure分母；Reader层不得伪造optimizer updates。
38. 模型输入/Teacher分离：改后验label不改变Teacher-free Actor logits。

## 6. 实际样本审查：防止验收空转

WP02报告用户机器 `real_small_04`真实范围：FG201 2021-12-08至12-09，690 market cursors、689真实推进、6个账户×691Replay记录、119笔实际交易（103 TP5,12 SL30,4 MAIN_ROLL）、1551 L1候选、93 L2资金域外invalid、P4的54笔TP。详见WP02 `docs/WP02_REAL_TRAIN_COVERAGE.md`。

Codex在真实本机 `corpus_root` 运行实际读入时，必须列出自身独立计算并核对的：

- P0–P4每条Replay实际stepid/teacher行数、L1有效Outcome和各窗口数、L2合法strict WAIT/ADD数及**真正同episode合法pair数**。
- 每个训练视图中的 `[B,T+1]`长度、P0–P4曝光、唯一账户/市场分母、真实Censor/OTHER，不把所有TP反事实当实际成交。
- 实际CPU model forward/loss时间、reader采样吞吐与内存；只验证小样无需重生成市场数据。
- 对WP02报告的任何差异明确`MISMATCH/NO_GO`，不修改准入逻辑去凑数。

**不是性能/策略GO**：两日一个合约不足以决定F1/F3训练样本量，也不许可批量18合约生产。

## 7. 验收矩阵：三个GO必须分开

| 验收项 | 需要的证据 | 通过后意味着 |
|---|---|---|
| `WP03_CODE_GO` | Commit A-D代码、CPU测试、Model Loss-Ready、无越权/无未来泄漏、完整状态恢复测试 | Reader功能已实现、接口闭合 |
| `WP03_REAL_TRAIN_DATA_GO` | 用户本机实际有限TRAIN Corpus G3独立重放/身份、三方Owner/Stepid Join、真实Reader取样、合法Loss输入 | 新训练系统已具备*有限真实数据*来源条件，不表示教材已覆盖全部品种 |
| `WP03_CURRICULUM_SUPPORT` | strict ADD/WAIT有效计数、legalPair、SL/快速逆势、TP快慢分布、Sampling每类支持 | 足够/不足分层教材有据可查；不足可报告LIMITED，不得伪造GO |
| `E_QUALITY_FORMAL_TRAIN_GO` | **继续NO-GO**，直到WP04–WP07单独实现阶段/参数/优化器/来源准入并获得用户授权 | 本WP不开放训练 |

**WP03代码可以合成GO且真实数据NO-GO；也可以真实来源GO但教学覆盖LIMITED。** 这三种状态不得揉成一句“通过训练验收”。

## 8. Review后特别强调的阻断性问题

| 严重度 | 风险 | Codex处理 |
|---|---|---|
| BLOCKER | 原Catalog旧标签与WP02新`L1/L2`不兼容 | 新V2合同投影，不原样调用旧`attach_quality` |
| BLOCKER | `stepid`只是物理位置，账户状态还须绑定 | 原Trace+StudentReplay+Sidecar联合Join、native状态/证据hash |
| BLOCKER | post-step observation/teacher与实际action时间意义不同 | next-execution对齐测试，prefix/terminal失效保护 |
| BLOCKER | Qv2 utility和strict discountedQualityTP不一致 | 默认所有Q/Soft监督LOSS禁用，不通过teacher posteriorQ冒充新QHead标签 |
| HIGH | 以每步account_hash作Pair owner会使pair计数假性为0 | episode级owner与step级JoinKey分开 |
| HIGH | 真实数据不在当前Git归档内 | 真实本机G3/Reader验收NOT_RUN不得声称PASS |
| HIGH | 只关注WP03 Reader可能漏掉未来WP06全WAIT | strict正负曝光、pair count、real actor_loss有效样本提前计数并加入路线图 |
| HIGH | 为开发方便提前解除旧E+Quality Guard | WP03必须保持现有公开Guard，模型CPU纯损失在隔离入口测试 |

## 9. 强制交付文件及Codex完成回复格式

应交付：

- `WP03_WP02_INTERFACE_AUDIT.md`（Codex在用户**当前真正HEAD**上复核本审计，列出差异）
- `WP03_IMPLEMENTATION_REVIEW.md`（最终真实状态，而非计划）
- `WP03_SOURCE_ADMISSION_RECEIPT.json`
- `WP03_TEACHER_JOIN_AUDIT.json`
- `WP03_COURSE_READER_COVERAGE.json`
- `WP03_CPU_FORWARD_LOSS_EVIDENCE.json`
- `WP03_CODE_GO_NO_GO.json`
- 本轮完整选定测试日志、testcase/exit码、目标双仓SHA和dirty状态、diff/patch/commitA-D SHA、实际CPU资源测量
- **更新后的** `FG_S3_E_Quality_Master_Roadmap_<date>_V2.md`：更新WP03子任务、WP01/02真实SHA、WP03状态、阻断项与下一步WP04，不要覆盖历史V1。

最终回复先给**双仓版本锁、CODE_GO、REAL_TRAIN_DATA_GO、CURRICULUM_SUPPORT、TRAIN_GO**五个独立状态，再列提交和原始证据路径。

## 10. 可直接发给 Codex 的执行口令

> 读取《FG_S3_E_Quality_WP03_Final_Codex_Task_gy18_d318_20261016.md》、WP02实际实施报告、接口Code Review、《FG_S3_E_Quality_Master_Roadmap_20261016_V1.md》和双仓 AGENTS.md。先在用户机器的真实 `feat/s3-e-quality-wp02-v2-20261016` 成对功能分支只读核对Git HEAD/dirty/status，并建立独立WP03 worktrees；不得从落后的integration HEAD开始或碰正在使用的训练树。只实施WP03 A–D：真实Corpus/G3权限验证、WP02 V2 L1/L2→Agent FIELDS合法投影、StudentReplay+Trace+TeacherSidecar三方Owner/Stepid正确Join、episode-safe NATURAL/STRATIFIED/MIXED统一Reader和随机状态恢复、原E+MACD+Quality Agent CPU Loss-ready与真实有限TRAIN读取验收。特别验证teacher在post-step观测上形成而不是本行action；严格WAIT只在L2双动作证明下有效；Qv2/Soft旧目标不得参与训练loss；不同账户绝不配对。仅运行CPU定向/有限真实TRAIN只读核验，不读TEST、不触发GPU、不启动正式训练、不生成18合约，不解除E+Quality公开Guard，不改金融Env/Quality Reward/MACD/Oracle，不造新Replay/Trainer，不恢复旧1M参数。每个提交给实际测试和负例，完成独立Review并提供四个提交SHA、原始Receipt/测试结果、GO/NO-GO及更新后的WP01–WP10 Master Roadmap。缺少实际Corpus、来源不一致或任何真实测试未运行必须明确NOT_RUN/NO_GO，绝不以WP02报告代替。

**结论**：这份任务书已将上版 WP03 中的“假设接口”替换为 gy18/d318 的真实接口与明确的新V2适配任务。**可以交付Codex开发；不授权正式模型训练。**

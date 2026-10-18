# FG-S3 E + Quality TP：WP04 FOUNDATION_WM 正式 Codex 开发任务书 V2（基于 gy19/d319）

**2026-10-18 | 状态：CODE_IMPLEMENTATION_READY；REAL_TRAIN/GPU：NO-GO，等待独立授权。** 由前版 `FG_S3_E_Quality_WP04_Foundation_WM_Codex_PreDesign_Reviewed_V1_20261016.md` 按最新 WP03 已实现接口修订。配套：`FG_S3_WP03_to_WP04_Interface_Review_gy19_d319_20261018.md`、`FG_S3_E_Quality_Master_Roadmap_20261018_V3.md`。先读双仓 `AGENTS.md`、当前真正代码和 WP03 实施报告；如实现不一致，先停下报告精确差异。

## 0. 冻结开发基线、总体目标和边界

- WP03成对提交：Gym `7366886f3b40bf74aa4485978ffcc97e39d0d53e`，Dreamer `446d1bc99debd2d05c313eafd7d0c414ebd2815c`；分支 `feat/s3-e-quality-wp03-v2-20261016`。源归档默认HEAD是旧 S3B，不得选错；`integration/s3-next` 尚未包含WP01–03全部交付。
- 从此**成对WP03提交**建立新的 WP04 feature worktree；只对新worktree做开发，不reset/clean/merge用户主线或训练工作树，保留原WP02/03 Corpus和审计文件。
- 唯一目标：从零训练 E-native + partial5m MACD 的 `enc/ dyn/ dec/ rew/ con/ avail/`，学习当前信息、原生账户反馈、Quality TP Reward及世界状态变化；形成可供WP05/06接续的完整模型Checkpoint。**WP04不会教Actor自主ADD/WAIT；后者在WP06。**
- 新架构一次到位，不迁移旧E 1M、不保持旧模型Observation/Checkpoint ABI；但必须保留真实金融Env、3手做空、TP5/SL30、官方交易时钟、原有Quality TP Reward、模型输入因果性、来源签名和恢复正确性。
- 本任务授权**代码修改、CPU合成有限更新和只读有限TRAIN来源审核**；不授权真实TRAIN优化器更新、GPU训练、1K/10K/1M Pilot、完整18合约数据生成、读取TEST、W&B正式发布。未经用户明确追加批准不得做这些事。
- 工作总包仍WP01–WP10，不得通过缩减WP05/06/07/08/09/10完成WP04。

## 1. 事实来源：WP03真实函数，不能再使用假设API

| 责任 | 真实接口/路径 | WP04使用方法与约束 |
|---|---|---|
| WP03来源 | Gym `fg_s3_e_quality_supervision.py::SupervisionSource.from_corpus(root,stage='FOUNDATION_WM')` | 只接受正式TRAIN、冻结Corpus+课程+Teacher+Obs+source lock内容 |
| WP03前检 | `QualitySupervisionAdmission.preflight(source,purpose='development'|'native-verification',producer_gym=...,log_path=...)` | `native-verification`真实运行G3才使 receipt.status=`SUPERVISION_DATA_READY`，但 `training_admitted=False`；development不能作为正式F1数据认证 |
| WP03三方Join | Gym `fg_s3_e_quality_reader.py::build_join_index(admission,index_path)` | index SQLite新建于Corpus外；必须保存index哈希和绑定的课程身份，不修改原Replay |
| WP03 Reader | `EQualityCourseReader.open(admission,stage='WM_NATURAL',sampling_spec={'mode':'NATURAL'},seed=...,index_path=...,cache_bytes=...)` | `stage=WM_NATURAL`是**抽样阶段**，不同于课程`FOUNDATION_WM`；正式F1起步NATURAL |
| 采样/恢复 | `sample(batch_size,effective_length) -> [B,effective_length+1,...]`；`coverage()`；`state_dict()`；`load_state_dict()`；`close()` | 时间前缀+1、stepid uint8[20]、原Replay `reward`, `action`, `is_first`, `market_advance`, `consec`；Reader RNG与index身份可恢复，**不等于训练carry/optimizer自动恢复** |
| Agent CPU先例 | Gym `fg_s3_e_quality_checks.py::cpu_model`、`cpu_forward` | 使用原 `embodied.run.fg_s3_source_transfer._make_agent`、内部 `_S3V2Model`创建真正fresh E+Quality；此文件是只读Loss smoke，无optimizer，不能把它当F1训练器 |
| Dreamer网络 | `dreamerv3/fg_s3_v2_agent.py::_S3V2Model.wm_only` + `dreamerv3/agent.py::Agent.world_model` | 原 `enc`→`dyn`→`dec/rew/con/avail`真实Loss与梯度；不再写另一个WM |
| 全模型初始化 | `_S3V2Model.quality_init_loss`与 `quality_extensions`/`head_modules` | 初始构建完整网络：WM + `pol/val/slowval` + QualityHeads + `quality_actor`；除WM之外均不在F1训练 |
| 检查点 | `embodied/run/fg_checkpoint.py::FullCheckpoint`；`embodied/jax/agent.py::save_exact/load_exact` | Agent可完整恢复；但FullCheckpoint在线`replay`分支**无法直接保存WP03 Reader**，本WP必须设计外部不可变Reader状态绑定 |
| 旧公开入口 | `dreamerv3/fg_s3_v2_agent.py::S3V2Agent`和原S3 composition/LiveRun | `waiting_quality_e`/`S3_QUALITY_E_NOT_IMPLEMENTED`仍存在；**只新增经F1特殊准入的新课程入口**，不得全局解除Guard |

WP03报告在归档 `gym/docs/WP03_IMPLEMENTATION_REVIEW.md`，证据在 `gym/docs/receipts/wp03/`。其声称177项选定CPU PASS、真实原生G3 GO、Reader LOSS/冷恢复PASS；本次静态查看了收据JSON，但没有原始168MB Corpus，不能在审查机重新执行。新F1真实训练准备必须在Codex用户本机重新认证真实Source和Corpus；旧收据仅作为可追溯历史证据。

## 2. 系统设计：一次到位的F1闭环

```text
完整 WP03 TRAIN Corpus + SourceAdmission / native G3
                 │
                 ▼
WP03 EQualityCourseReader (WM_NATURAL)  ← 只读、RNG+index+源数据身份
                 │  [B,T+K+1] native row/action/reward/context
                 ▼
F1 Context Adapter：有效前缀、K步因果burn-in、真实action对齐
                 │  禁止老师后验标签进入模型，严格识别有效T
                 ▼
fresh 原生 _S3V2Model  (init阶段完整参数)
                 │  enc + dyn + dec + rew + con + avail
                 ▼
F1 WM-only Loss + 原单优化器 + 严格正向WM白名单
                 │
                 ▼
日志/事件分层/参数-冻结-优化器审计
                 │
                 ▼
原 FullCheckpoint 扩展外部Reader后保存完整Run身份/状态
                 └── cold restart → exact next batch / model / optimizer / counts
```

课程身份 `FOUNDATION_WM`。配置必须显式携带course identity、Corpus/index源hash、双仓Git SHA、model architecture/seed、Reward schema `FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1`、E Observation V2 + MACD schema、Reader policy/spec、B、有效T、burn-in K、loss scales、优化器配置、checkpoint协议、更新预算、CPU/GPU及授权状态。禁止把F1 optimizer_updates 冒充在线Env advances。

### 2.1 模型参数与优化器权限（必须逐键验证）

| Ninjax参数前缀 | F1 | 说明 |
|---|---|---|
| `enc/` | TRAIN | 1m、15m和partial 5m MACD已进入Encoder；输入合法/有效位因果性保持 |
| `dyn/` | TRAIN | RSSM posterior/prior state，保留原随机性/复现合同 |
| `dec/` | TRAIN | 仅当前Decoder的1m共享序列、position两类重建 |
| `rew/` | TRAIN | 预测原生真实Replay反馈；不以L1/L2 teacher分数或动作覆写 |
| `con/` | TRAIN | 真实终止/延续与market_advance折扣 |
| `avail/` | TRAIN | legal WAIT/ADD Mask，后验/先验各自诊断 |
| `pol/ val/ slowval/` | FROZEN | 参考RL网络不能学习；包括其optimizer leaf/EMA状态 |
| `quality_actor/ quality_event/ quality_touch/ quality_adverse/ quality_score/ quality_q_c1/ quality_q_v2/` | FROZEN | WP05/06专门教授，不得提前更新 |
| Gate、归一化/其他未列的状态 | FROZEN或明确审计例外 | 所有新/未知前缀 fail closed；唯一允许前进的是显式标注的共享Optimizer全局step/RNG及F1调度 |

固定 `quality_q_c1_lambda=quality_q_v2_lambda=quality_soft_lambda=0`、Prediction/Decision Fusion关闭、BC/Pair/DAgger/RL关闭。**不能仅凭Loss乘0或梯度为0认定冻结**：Adam动量、weight decay和Ninjax变量写入均需审计。应在现有 `masked_joint_optimizer`／原 `self.opt`机制上实现`f1_wm_prefix_allowlist`；未知参数或优化器槽位不允许默认更新。

### 2.2 F1 Loss仅复用原 world_model

- 准确复用 `world_model()` / `wm_only()` 的 `losses` 和 `self.scales`；**Loss键不是六个参数前缀**。实际会涉及RSSM各项、Decoder重建、`rew`、`con`、`avail_post/prior`。
- 初始网络参数通过原 `_make_agent` / `_S3V2Model.quality_init_loss`一次性生成完整布局；初始化阶段可触发原完整模型构造，但在任何实际F1 optimizer step中禁止调用 `quality_loss`、imag/ActorCritic、Teacher后验监督。把`nj.creating()`时的初始化和训练阶段分别测试。
- 只让真实`obs`及已提交 `reward`进入WM；`teacher_action`, `teacher_valid`, `quality_*`, L1/L2数值未来结果只在Reader、数据审计中，F1 Loss函数签名无这些输入。测试改变后验标签的全部取值，WM Loss与全部WM梯度应逐键保持一致。
- 输入应是 `market_seq_e_shared[60,18]`、`market_context_15m[24,7]`、`market_macd_5m[4]`、`position_state[9]`、`action_mask[2]`。训练中只允许真实训练前合法可见的MACD与行情状态；不能为提高Loss添加未来指标。
- Loss报告除总loss外提供loss名/权重、事件条件分层、有效目标数、grad finite和数值范围。Reward Head训练和原生Quality TP Reward保持一致：不重新设计金融公式、不在F1擅加MAE奖励。

### 2.3 核心修订：随机窗口RSSM的K步因果warmup（不能再用1帧冒称完整历史）

**WP03现状**：`Reader.sample(B,T)`从账户段中部随机切片，`consec`初始化为零；`_apply_replay_context`的`recompute_prefix`按K=1在窗口第0帧做一次动态重建。它对WP03 Loss连接smoke足够，但**无法证明RSSM有真实历史隐状态**。

**WP04初始建议的具体可实现合同**（强制写测试，可在源码比对后做等价优化）：

1. 原Reader已经可以按任意 `effective_length` 返回`[B,L+1]`且保证不跨同一账户段。F1取样时调用 `Reader.sample(B, T+K)`，其中 `T` 为实际反向传播有效分钟数、`K` 为额外的因果预热分钟数，返回`[B,T+K+1]`；首版建议测试 `K=32` / `T=64`（配置化，勿写死为正式训练参数）。这样**不需要另造Replay，也不一定需要改Reader接口**。
2. `_apply_replay_context`仍只消耗第0帧作为首帧；随后先在同一真实账户段按时间依次处理前K帧，`training=False`或等价的纯前向状态更新，并`stop_gradient`阻断该前缀反向传播；对剩下T帧调用原生 `wm_only(training=True)`产生优化目标。
3. `actual_action`历史、`prevact`、`stepid`与`reward`按原Replay的incoming/action规则时间对齐；严禁以L2 strict action替换真实行为。burn-in真实行必须来自当前随机切窗的**前置市场信息**，绝不读采样终点之后的信息。
4. 固定K仅代表**有限历史近似**，不是完整真实episode从reset起点的隐状态。报告包括warmup长度、距离最近reset多少分钟、有效/无效样本、训练目标时间数量；另外用短episode从reset全量前向作为一致性对照。若发现K=32造成不可接受偏差，调整K或采用episode起点迭代，但必须通过完整效益/性能对照才升格。
5. 为避免首帧来自任意账户中途却误用上一批隐藏状态：**每个随机窗口重新初始化其自身carry**，通过此窗口K+1真实历史构造，不得在不同随机窗口之间传carry。对于真正的连续训练流，必须另写严格顺序和源segment identity证明，不允许把Reader随机结果当连续流。
6. 验收对照包括：相同batch改变K是否影响posterior；切掉warmup会出现预期变化；将同一账号顺序打乱、混入其他策略、把replay stepid错移必须失败；若K+T超过有效support报`INSUFFICIENT_CURRICULUM_SUPPORT`，不得静默缩短。首版若数据不足，可缩小合成T/K验证工程，**不可从TEST补样本**。

该方案避免了修改 `consec` 伪造跨批连续性；训练优化目标只计T个有效时间步，采样曝光应区分K burn-in与T supervised WM targets。

### 2.4 Checkpoint：复用原FullCheckpoint，但必须扩展真实外部课程Reader后端

**不能直接 `cp.replay=reader`。** 源码 `FullCheckpoint.save` 硬访问在线Replay并调用 `save_replay()`，其锁/Uniform/chunks与WP03 Reader完全不同。新F1只有已冻结的外部只读Corpus/SQLite index，应复用既有原子bundle与 identity/manifest/checksums，增加明确的`external_immutable_course_reader`存储方式。实现可在`FullCheckpoint`中以参数化backend/小适配器支持新语义，**不得绕过FullCheckpoint安全保证，不得把整份Corpus复制进每个Checkpoint**。

F1一次完整状态至少包含：

- `agent.save_exact()`含全部WM+Quality/Actor冻结参数、原优化器状态和Ninjax运行状态；更新计数;
- `reader.state_dict()` RNG、stage/sampling_spec、source/index identity、coverage/delivered counters；只引用外部Corpus+index，不复制训练行情文件;
- `carry`（若训练器在步间保留状态，随机独立窗口应存明确空carry/初始状态规则），JAX随机状态、Python/NumPy随机状态;
- `optimizer_update_step`、有效WM目标数、burnin曝光数、Reader采样次数、课程/时间/RNG状态、report/save调度;
- 锁定双仓代码 SHA、课程/模型/Reward/MACD、Corpus/index SHA、`T,K,B,seed,loss_scales,optimizer`和运行环境必要版本。

严格证明相同种子下连续N步与跑M步→FullCheckpoint cold restore→N-M步的**下一batch逐字节、参数逐键、优化器state、计数、指标**一致（容许已预先定义的浮点确定性边界，但不能以数值近似偷换本该完全一致的CPU证据）。Source/index/代码/Reward/K参数不同必须拒绝恢复，坏snapshot不可静默fresh init。可继续使用现有`StepSchedule`，但不能把Env step作为离线F1主计数。新离线F1正式准备身份单独命名，禁止覆盖在线旧Run。

### 2.5 事件诊断：不能只看总WM Loss

一次更新后必须能按真实事件报告（有效数量为零时写`NO_SUPPORT`而非0准确率）：

- 奖励target原值分布：`reward=0`、非零、止盈、止损等可由原生事件证明的分组，及对应`rew`预测Loss、偏差/校准；不能用teacher L1/L2结果替代真实Reward target。
- Availability：post/prior的双合法、仅WAIT、合法ADD召回、假阳性，按真实Env action_mask，对全WAIT误导进行负例。
- 多尺度：有效15m和partial MACD输入消融引起的Encoder/RSSM输出差异、分支梯度与参数变化；无效valid时输入值变化不影响输出；不能虚构15m或MACD Decoder重建Loss。
- RSSM：dyn/rep项、KL/状态有限性、连续chunk/跨账户的`is_first` reset；market_advance/continue时钟。
- 学习覆盖：unique market cursor、unique账户、Reader交付重复次数、raw corpus rows、WM目标步数、burn-in位置/分钟、optimizer updates各为独立指标。共享行情P0–P4不能算五倍独立市场经历。

## 3. Codex开发提交划分（A→E，功能小任务）

### WP04-A：真实F1课程模型与Preflight（Commit A）

**改动**：新F1配置合同与独立Prepare/StageAdmission薄适配；复用 WP03 SupervisionSource/Admission、course expected_stage_contract 和原 `_make_agent`；fresh完整E+Quality网络/身份绑定。新增F1公共入口只允许 `FOUNDATION_WM`，仍要真实G3与外部训练授权分层；旧公开E+Quality依然阻断。准备结束输出非训练 `F1_ENGINEERING_READY` 收据，或如果真实G3重验 `F1_DATA_READY`，两者均不是 `F1_TRAIN_AUTHORIZED`。

**测试**：模型全部前缀存在，参数随机初始化且不继承旧1M；Reward/Obs错误、正式身份伪造、DEV/TEST、非预期stage拒绝；未授权formal启动/恢复无副作用；旧Guard原样。

### WP04-B：原生F1 WM-only真优化器更新（Commit B）

**改动**：明确路由 `_S3V2Model.train` 新课程分支到 `wm_only`，不走 `quality_loss`/HEADS_ONLY；新增 `T+K+1`真实上下文Adapter，K步warmup与只计T目标；原 `self.opt`严格WM允许前缀、默认冻结其他全部（包含状态与slot）、单次更新；nonfinite fail closed。

**测试**：合成合法真实schema数据上2–10个CPU优化器step；六组实际非零学习信号和参数变化在对应有效样本中可证；`pol/val/slowval/quality_*` 及其optimizer槽位精确不变；只允许共享opt step递增；错误 K/长度/账号边界、标签污染和返回字段错位拒绝；没有teacher sidecar修改WM Loss/梯度。

### WP04-C：预测/数据可观测性与病理诊断（Commit C）

**改动**：原生每项Loss、Reward/Avail按真实反馈事件诊断、多尺度gradient/ablation、重复市场和账户统计、K上下文校准对照。所有诊断无未来行情输入；必要时保留只读诊断旁路，禁止将教师正确答案塞进Encoder。

**测试**：真实可见5m MACD/15m敏感、无效位一致；全零reward、全WAIT、无合法ADD、全部候选删失等都不得被错误标为学习GO；诊断禁用时训练行为不变。

### WP04-D：F1只读离线Reader + FullCheckpoint精确恢复（Commit D）

**改动**：唯一离线F1 Run协调层，原Reader、模型与原FullCheckpoint扩展外部immutable课程Reader状态。阶段运行身份锁、更新预算、计数、schedule、冷恢复，不能复制Corpora、不会开在线Env Collector、不会以RL Gate解锁训练。

**测试**：中断前后next batch全字段一致、full-state逐键相同；源/索引/双仓代码/K/seed/loss权重/optimizer schema变化拒绝；checkpoint不产生整个Corpus副本；破损拒绝不从零悄悄继续；首次写入与resume目录权限正确。

### WP04-E：双仓集成与独立Review（Commit E）

**改动**：整合A–D、离线CPU合成probe、小规模真实TRAIN**只读Loss核验**与G3（若本机可重验），完成35+类正负回归和使用文档，输出 `WP04_IMPLEMENTATION_REVIEW.md`、`WP04_F1_INTERFACE_LOCK.json`、测试命令+原stdout/stderr/exit码、全差异与完整身份收据。

**测试/退出**：来源与Replay/Reader/QualityTP/MACD/金融Env/Gate/FullCheckpoint共享回归必须通过。若缺真实Corpus/G3，标注 `REAL_DATA_NOT_RUN`，不得为推进训练伪造数据准入；本轮禁止真实TRAIN优化器更新/GPU。

## 4. 完成/验收合同（>=40项，负例必须实跑）

### 基线与身份

1. 双仓HEAD恰为本任务WP03配对祖先，分支独立、源干净且没有修改运行worktree。
2. WP03报告、G3原始收据和新实际source/index逐个核对；原真实Corpus没在审查机就不写“本次已重播”。
3. `FOUNDATION_WM`课程阶段≠ `WM_NATURAL`采样阶段；错位拒绝。
4. 课程固定Quality TP C1与原生E+MACD V2、3手/TP5/SL30、官方交易日内质量语义。
5. Fresh init；旧1M/legacy Quality参数绝不复制。
6. `F1_ENGINEERING_READY`/`F1_DATA_READY`/`F1_TRAIN_AUTHORIZED`/`F1_RESEARCH_GO`四种状态不得互冒。
7. 正式训练需真实native G3+外部用户授权，单靠自哈希/loaded JSON/contract draft不能放行。
8. 旧E+Quality `prepare/start/resume/evaluate` Guard、Prediction/Decision Fusion都保持原状。

### 时间/数据与真实Loss

9. `Reader.open(...stage='WM_NATURAL'...)`，返回 `[B,T+K+1]`，后验标签隔离。
10. `stepid`唯一有效轴与 `obs/action/prevact/reward`时间一致，不把后验`strict_action`充当行为。
11. `is_first`、`is_last`、`is_terminal`和正式`market_advance`保持原生含义，隔离账户/政策。
12. K步warmup不计Loss且不参与梯度，训练有效T，无旁观窗口数冒充更新样本数。
13. K+T不足support明确失败，不截断到另一账户/交易日之外。
14. 随机窗口carry不得串流；Reader `consec`默认零不等于整段历史carry。
15. 样本多次重复曝光与独立市场分钟分开统计。
16. Teacher后验值全部重写但不改原生Observation时，F1 Loss/梯度完全相同。
17. 原Reward来自Env、不是L1/L2 Score；改变“未来标签”不改变实际WM奖励目标。
18. 完整Loss键与原 `self.scales`一致，禁止将六个参数前缀当作Loss键清单。
19. 只存在既有两种重建目标，不声称重建了MACD或15m。
20. 原金融市场分钟折扣+终止约束保持。

### 参数更新和诊断

21. 新鲜模型一次初始化包含未来F2/3 Heads/Actor与原WM所有前缀。
22. `enc/`在真实有效数据上有非零grad和真实变化。
23. `dyn/`有有效梯度和更新。
24. `dec/`有既有两个重建目标的真实更新。
25. `rew/`在非零原生事件出现时有可学证据。
26. `con/`在合法样本有有效学习梯度。
27. `avail/`在双合法及仅WAIT case有更新，分别报告prior/post ADD recall。
28. `pol/ val/ slowval/ quality_actor/ quality_*`参数逐键bitwise冻结。
29. 上述冻结模块optimizer leaf/动量/权重衰减状态逐键不动，允许的全局step例外明示。
30. 未知前缀、新建参数、非WM Ninjax状态更新须拒绝或审计通过，不能默认allow。
31. 纯零Reward/全WAIT训练Loss有限不能标成市场识别GO。
32. MACD有效时敏感，无效时对隐藏数值不敏感；15m有效context敏感。
33. Teacher Q/Soft、Prediction、Decision、RL、BC/Pair损失和融合均为OFF。
34. GPU相关及旧1M参数全NOT_RUN，无后台训练。

### 恢复与来源

35. 相同seed N连续更新与M+冷恢复+N-M的next batch一致。
36. 对比所有Agent/optimizer/state、有效目标数、Reader RNG/coverage、StepSchedule、输出metrics与JAX随机态。
37. Checkpoint没有原Corpus大数组副本，只引用Source+Index hash并重验原文件身份。
38. source/index/Reader stage/sampling_spec/Model/Reward/代码双SHA/优化器参数/burnin K/可见特征任一变化拒绝恢复。
39. 目录损坏或身份不同不得自动从头训练。
40. 自查WP03其余课程HEADS_VALID、ACTOR_STRICT、ACTOR_QUALITY_PAIR没有被F1更改；后续WP05/06仍可用同源数据。
41. WP01/02/03已有已选CPU回归、Shared Gate/SourceAdmission/E模型/Checkpoint准入不回退。
42. 严格记录原测试、负例、失败后修复的证据、实际Commit SHA及未运行项。

## 5. 工程里程碑与测试预算（不得自动启动研究训练）

| 层级 | 环境 | 本包需要实际证据 | 禁止把它解释为 |
|---|---|---|---|
| G0 | 人工合成但符合原生Schema的 CPU Reader | 2–10真实优化器更新、冻结、warmup、full-state冷恢复 | 真实市场研究效果 |
| G1 | WP02/03有限真实TRAIN两日小样 | 真实G3/source和**只读loss/candidate事件分布**，无任何真实数据优化器更新 | 泛化能力/正式训练许可 |
| G2 | 多日多合约TRAIN | 后续WP09依据覆盖报告独立批准来源/数据生产 | 此任务自动执行18合约 |
| Pilot | 真实TRAIN + 固定超参数 | 未来用户明确授权后先1K、10K再决定100K+ | WP04任务书已经准许运行GPU |

**完成标准分开报告**：`WP04_CODE_GO`（工程链+CPU updates）、`WP04_REAL_DATA_READ_GO`（真实G3+只读F1输入）、`WP04_TRAIN_GO`（外部授权及运行准备）、`WP04_MODEL_RESEARCH_GO`（需要训练和冻结评价，WP04任务开发期不能声明）。WP04 engineering GO后继续WP05设计/代码即可；但完成真正的F1基础权重需要后续经授权运行。

## 6. 反向 Review：首版必须做好，避免事后返工

| 假失败路径 | 可预见根因 | 首版处理 |
|---|---|---|
| 无法训练WM | 旧 Quality HEADS_ONLY关闭WM | 独立 F1 Loss路由，不使用legacy Stage冒名 |
| All-WAIT 模型假成功 | World Model总Loss和mask accuracy好看，实际reward/ADD罕见 | 原生Reward事件分层、Avail ADD Recall，独立有效支持诊断 |
| 教师标签污染 | 将L2 `strict_action`或L1 Score写入Observation或Reward | Actor侧/teacher侧隔离，F1带未来反事实测试 |
| 随机Reader破坏RSSM记忆 | `consec=0` + 仅1帧prefix | 固定K因果真实warmup+截断解释，额外episode起点对照 |
| WM慢性冻结漂移 | Adam momentum/weight decay或未知Ninjax状态 | 六组白名单，参数/opt/state逐键比对 |
| 切F2/F3时随机新Head | F1只构造了WM | 一次初始化完整E+Quality模型，保留所有冻结头 |
| 模型断点不能精确继续 | FullCheckpoint在线Replay假设；没存Reader RNG | 扩展原FullCheckpoint外部Reader后端，保存完整状态 |
| 研究数据泄露 | 读TEST或以DEV统计优化 | formal仅TRAIN；DEV研究评估独立冻结，不参与梯度 |
| 耗时不合理 | F1每步重做 G3/读取全部Corpus/复制Replay | prepare阶段G3，训练热路径只做Reader chunk缓存/身份轻检，Checkpoint引用不复制 |
| “直接重训无兼容”被误用 | 删除因果、金融、时间及来源保护 | 仅新Observation不兼容旧模型；正确性仍强制 |

**发出前Review结论：WP04任务已从“假设Reader存在”升级为真实接口可执行设计。待 Codex 在用户真实机器核验当前成对WP03 HEAD和源码后即可开始WP04代码开发；正式真实优化器/GPU研究训练仍NO-GO。**

## 7. 直接交给Codex的执行指令

> 阅读此任务书、`FG_S3_WP03_to_WP04_Interface_Review_gy19_d319_20261018.md`、最新 Master Roadmap，以及双仓 `AGENTS.md`、WP03 实施报告和真实代码。以Gym `7366886f...` / Dreamer `446d1bc...` 的WP03功能分支为共同祖先，新建隔离WP04 worktrees。先输出接口锁和真实字段差异，严禁从归档默认旧S3B HEAD或者落后的`integration/s3-next`开工。按WP04-A至E五个小提交依次完成：fresh完整E+Quality F1模型与课程准入；原生wm_only的真实CPU优化器更新与六组WM白名单冻结；K步真实因果burnin及完整计数；Reward/Avail/多尺度学习诊断；FullCheckpoint外部immutableReader模式、源身份与下一batch精确冷恢复；全测试和Review。不要另写Model/Replay/Oracle/Reward，保留现有金融/时钟/来源及E+Quality非F1 Guard。只允许CPU合成更新与真实TRAIN小样只读检查，未经用户进一步批准禁止实际真实数据更新/GPU/正式研究Run/TEST/全量Corpus。输出五个精确提交SHA、完整源码差异、真实测试命令退出码、前后参数与optimizer leaf哈希、Reader/Checkpoint负例、`WP04_IMPLEMENTATION_REVIEW.md`和更新版Master Roadmap。未运行或未认证项标`NOT_RUN/NO_GO`，禁止以旧报告代替新实验成功。

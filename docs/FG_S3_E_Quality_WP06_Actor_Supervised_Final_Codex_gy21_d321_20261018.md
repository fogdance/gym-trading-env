# FG-S3 E + Quality TP — WP06/F3 Quality Actor 正式 Codex 开发任务书

**日期：** 2026-10-18  
**状态：** `IMPLEMENTATION_SPEC_REVIEWED / CODE_DEVELOPMENT_GO / FORMAL_TRAIN_NO_GO`  
**适用源码：** `gy21` Gym `61dd64b5428588868f374389d4f3d79e0845cdac`；`d321` Dreamer `be7ce01ebbe63144e4d335b1603ca73da4fa1c79`。这两个是 WP05 的**功能分支**，不是归档默认 HEAD。真实集成的 `integration/s3-next` HEAD须由Codex现场核对。WP06实施前执行第0阶段的主线整合；若WP06实际开发HEAD已不同，按真实diff去重，不重复实现相同能力。

## 0. 最高优先级：目标不可变

- 最终由 **`quality_actor`** 在仅看到决策时合法 E(1m+15m+partial5m MACD)+账户/动作mask 后，自主判断 `WAIT/ADD`。有高质量做空机会ADD3手，优先较快TP+5、少逆行、回避开仓后快速逆行10点及SL30；没有机会WAIT。
- 原金融Env、Quote、TP5/SL30、Quality TP Reward、同一官方交易日质量偏好均不变；**不因开发F3重设计教材/指标/老师**。
- 直接从新F1或新F2学生参数接续（可二选一，对照），不迁移历史 E+Tutor 1M、不保留旧模型ABI。
- **L2 strict_action为唯一动作真值；L1只描述立即入场结果，不自动把高score等同ADD。** Teacher、future、suffix只能参与离线Loss，不进policy observation。
- 工程完整做出来、可做合成有效更新；未有正式受训F1时不得称模型已学会交易。开发/数据/正式训练采用分层流程；用户未授权真实长训练和TEST考试。

## 1. 先执行的第0阶段：公共双仓集成基线

本任务不能默认WP05代码已经进`s3-next`：归档中当前Gym主线 `0e4af49`、Dreamer主线 `fd14411`，WP05仅在各自feature分支。按《WP05→WP06接口Code Review》第5节：在独立集成 worktree合入WP04/WP05，Gym处理Roadmap V3的 add/add，统一 MACD V3课程/Observation/DataProducer，跑合成F1/F2/Reader/Checkpoint/因果/身份定向测试，取得**两仓成对集成SHA**。失败停下并报告，不自动修改训练worktree。成功后才从公共成对HEAD建立新的WP06双仓feature worktrees。用户若选择先让WP06在WP05分支开发，必须把它记为暂时分叉并在发布前重复完整集成Review；**推荐先集成**。

**不能因为不维护旧模型而忽略V2/V3差别**：历史V2样本可做隔离开发诊断，不得冒用为V3正式课程。当前完整源码身份变化按新数据真实生成、真实来源进行认证；绝不通过改哈希或关闭验证让旧样本假装有效。

## 2. WP06六个大任务与建议提交

| 任务 | 目标 | 首要交付 |
|---|---|---|
| **06.A F3合同与Agent阶段** | 新 `ACTOR_SUPERVISED` + `quality_f3_protocol`，不经过旧预测准入 | 显式模型配置、合同白名单、禁用预测融合/Q/Soft/RL、调用路由与负例 |
| **06.B F1/F2→F3模型参数接续** | F1或F2完整参数（含尚未学过的quality_actor）进入新的 F3 Run | 严格source二选一、所有non-opt逐键对齐、fresh Opt/RNG/时钟、源未改写、receipt |
| **06.C F3真实Actor-only更新** | 原始 `quality_actor/` 独自接受监督梯度 | Reader、真实burn-in、停止WM梯度、原Opt+逐键冻结/finite回滚、非有限/无标签跳过 |
| **06.D Teacher与采样** | L2 Natural+Balanced BC、strict同Owner pair、可选L1一致性Quality pair | 可配置Loss消融；L1/L2冲突计数与拒绝；natural anchor、有效样本与pair分母 |
| **06.E Actor-only评估** | 真正仅学生Argmax，拒绝P4、教师/预测修改动作 | 真实合法Action mask、Teacher-free Eval、pADD/ADD/WAIT、合成可学数据与原金融回放接口 |
| **06.F CP与验收交付** | 原FullCheckpoint真恢复、CLI、测试、Review | 前后参数/Opt及Reader RNG/prefetch一致；CPU定向测试、选定小GPU验证（环境允许）、实施报告 |

鼓励将模块复用放在现有 `dreamerv3/fg_s3_quality_v2.py`、`fg_s3_v2_agent.py`、`fg_s3_e_quality_foundation.py`、`embodied/run/...` 和 Gym research/Reader。可建立**薄的** F3 配置/阶段协调模块，但禁止第二套Env、Replay、教师、优化器或训练主循环。

## 3. 06.A 课程与模型合同（细节）

1. 新协议名称建议：`FG_S3_E_QUALITY_ACTOR_SUPERVISED_V1`，stage=`ACTOR_SUPERVISED`，sampling=`ACTOR_STRICT`；显式配置 `target_steps`、`burnin_steps`、`batch_size`、`sampling_spec`、actor Loss配置、Optimizer Budget、保存间隔、来源类型(`F1`/`F2`)、来源node、corpus、run/prepared路径等。
2. 复用 `expected_stage_contract('ACTOR_SUPERVISED')` 的唯一课程上界；runtime **有效更新白名单**仅 `quality_actor/`，不改变进入 WP02/03已有 Corpus 哈希的课程数学/参数矩阵。WP05 F2白名单实现可作模板。
3. 将 `quality_f3_protocol` 注入`quality_extensions()`优先分发，不把 `CONTROLLED_FULL` 当作 F3，也不能用F2占用的 `QUALITY_HEADS` stage。 `_S3V2Model`增 `self.f3` 及`train()`和`ext_space`必要路由；F1/F2/F3互斥，完整E+Quality模型只有一个最终stage。
4. F3 Prediction、Q、Score对Action的融合均关闭，`quality_decision_admitted=False`、`quality_prediction_admitted=False`、`quality_q_admitted=False`、`quality_soft_lambda=0`，任何旧`RL gate`不得打开。Teacher Loss可使用Quality标签，但 `pol`、`val/slowval`不得参与优化。
5. 保留完整 `quality_actor`输出 + `_masked_policy` 遮罩/Unimix一次；区分原logits和最终masked分布。动作无效时mask，不给F3假正例。动态模型构造检查禁止非法差异而非静默修复。
6. F3 `HEADS` Loss不可参与Actor梯度；若使用F2预训练预测头，只作冻结只读诊断，未经DecisionAdmission不能添加到policy分布。

## 4. 06.B 来源与Checkpoint：两个不同操作

**参数接续（新Run）**：`source_stage`为合法F1或F2之一。读取源 `run_identity`、`FullCheckpoint` bundle、模型参数与source训练证据；核对工程/正式身份与模型/Observation/Reward/schema，先验证再修改新的Agent。必须逐键复制 **完整non-opt模型参数**，含Encoder/RSSM/WM、Quality四头、原Actor及`quality_actor`（此时仍是初始权重），拒绝缺键、shape/dtype非有限；目标optimizer/RNG、Reader、prefetch、step/counters fresh，源文件不变化。F2来源若仅 `ENGINEERING_ONLY` 只能进入F3工程烟测；不能当正式已训练F2。

**同Run恢复**：只有 F3 Run 的 FullCheckpoint 可以执行F3→F3 resume；完整恢复原参数、Actor optimizer slots、共享时钟、Reader/sampling RNG和prefetch、优化器updates、**消费批次**（可能与updates不同）、日志与调度身份。不得误把接续当ExactResume，或只恢复模型漏掉Sampling。

**注意**：目前 WP05 正式训练证据不完整，真实F1/F2未训练；Codex应使用既有合法built-in CPU合成fixture证明接续，生产未就绪时保留明确 `ENGINEERING_ONLY / TRAIN_NO_GO`，不能为使F3通过而降低SourceAdmission。

## 5. 06.C 学习主通路：Data→WM冻结表征→Actor更新

1. 从 WP03 `EQualityCourseReader.open(admission,stage='ACTOR_STRICT',sampling_spec=...,seed=...,index_path=...)` 获取真实账户连续窗口。`sample(B,K+T)`返 `[B,1+K+T]`；reader `t=0`仅作为历史prefix，标签在后续实盘账户状态对齐。Target Slice由 F1已有 `causal_context` 控制， K真实历史预热，输出 `ids[:,K:]`与 data target 的`stepid`一致；不能将前缀frame当成完整RSSM历史，也不能重复移动 `teacher_action` 或把 `step['action']` 当成监督。
2. 使用 `wm_only(training=False)` 从完整冻结 World Model 得到真实RSSM表征，`stop_gradient`；只连接原 `quality_actor`。Policy distribution由 `quality_actor`raw logits 经原 action_mask/unimix一次计算；禁止Teacher字段参与Encoder/Actor forward。
3. 对每个目标状态定义 `valid_L2 = teacher_valid & (teacher_action∈{WAIT,ADD}) & action_mask==[1,1] & legal_decision & !reset !terminal`。用Reader已有对齐和原生mask判断，不凭空派生合法性；对invalid仅屏蔽，不能补WAIT。
4. 以L2监督为主：Natural BC必须存在；可选 balanced BC 与 strict pair。每个loss按自身合法标签/真实Pair计数归一化。全无效，Actor参数与optimizer所有非时钟slots精确不变；当没有有效pair时pair loss=0并明确记录no_pair，而不是强行构造跨账户pair。
5. `quality_actor/` 参数及专属Optax slot按有效更新许可；`enc/dyn/dec/rew/con/avail`、所有Quality Heads、两个Q、`pol/val/slowval`与其Opt槽**逐字节不变**。复用F2 `masked_joint_optimizer` + state schema owner映射/旧值恢复；finite失败要恢复整个Candidate更新。
6. 分离 `consumed_batches`、`optimizer_updates`和Reader结果数量；训练进度以真实成功更新计，单个坏Batch不可无限循环，训练调度要能报告支持不足而非隐藏失败。

## 6. 06.D 教师损失与训练策略（L2优先）

建议数学：

`L_F3 = λ_nat L_BC,natural(L2) + λ_bal L_BC,balanced(L2) + λ_pair L_Pair,strict(L2) + λ_qpair L_Pair,consistent_L1`。

- Natural BC记录真实采样暴露/类别基准；Balanced仅用于提高少量ADD的训练机会。旧 Quality `teaching_objective()` 用 alpha混 natural 与 balanced，可复用但F3的新Loss应显式报告混合权重和采样模式；不能将全库正负平衡当成真实市场先验。
- **严格Pair**：与Reader `quality_owner`、reset/segment以及L2正ADD/负WAIT共识一致；每批明确报告pair个数。随机窗口没有Pair时只执行BC，不构造跨contract/corpus/account的Pair。
- **L1 quality_pair**：当前旧实现无条件以 `quality_score_high-score_low≥15` 排序，可能和L2矛盾。新的F3需显式冲突过滤（两端L2均有效、同Owner/Segment、动作顺序不与L2逆向）；**默认`λ_qpair=0`**。模块、开关、计数、单元测试一次实现完整，未来研究开启不需改Schema；只有经过消融验证才可打开。L1其他TP时间、MAE和早期逆势可作诊断与划分，不要偷换动作真值。
- Q/Soft监督来源在WP03尚未认证，`λ_soft=λ_q=0`，不得让L2的后验Q标量直接填入旧Quality Q loss。
- 对照组至少包括 Natural-only，Natural+Balanced，Natural+Balanced+StrictPair，以及启用一致QualityPair的可选消融（必须具足够真实pair并单独报告）。LOSS权重必须配置化并有finite、范围/非负校验；不能先把 `.8/.2/.1`当经验证的最优参数。
- 训练阶段完整开放可用的合成超参小规模调试，不按每次CPU变动做全套FORMAL入场审计。Actor只更新不意味着正确入场；业务评估另走考试。

## 7. 06.E 独立Actor-only考试（工程首版）

- F3结束即支持用 **纯学生权重** 执行无Sidecar/无Teacher的原生推理，deterministic argmax under actual action_mask/unimix。对同一个合法状态训练/推理输出一致，原生WAIT与ADD都可发生；不能通过阈值降低、手动开仓、P4接管制造ADD。
- CPU合成可学正负分类任务：记录 L2 BC Loss、ADD/WAIT准确率、`pADD`均值/分位/分布、同Owner strict Pair accuracy、各类ECE/Brier若使用概率校准、梯度与参数变化。比较fresh Actor及恒WAIT基线，防止只看整体Accuracy。
- 有限真实TRAIN read-only推理及金融Env交互可作为DEV小样；Actor-only真正的独立DEV/TEST考试与稳定性/成本/MAE/TP时间归WP08和正式研究授权。L1后验TP计数不得冒充真实成交。无合格TRAINDATA时保持 `INSUFFICIENT_CURRICULUM_SUPPORT`，不得以合成测试推断交易胜率。

## 8. 06.F 恢复、验证与验收清单

**开发必需（可在CPU合成或可信TRAIN小样完成）**：

1. stage组合/配置/来源权限拒绝错误类型、旧CONTROLLED_FULL不能绕过Prediction、F3禁Quality Fusion/Q/Soft/RL。
2. F1或F2二选一的non-opt逐键handoff，fresh Opt、种子和训练Run，新旧模型参数schema/dtype有限；正式未训练源拒绝冒充已学基础。
3. 原生 E/MACD时序、K burn-in、Teacher与action位置、reset/合法mask及禁止未来输入。
4. 仅 `quality_actor/` 的参数、Opt slots有真实有效变化；WM/四头/Q/参考Actor逐字节不变；非有限/全无效/标签错位负例。
5. L2 ADD/WAIT标签翻转导致梯度变化，Natural/Balanced正负端计数，strictPair同Owner约束，L1高分但L2 WAIT冲突过滤，zeroPairs/单类别安全退化。
6. Reader不同采样模式+新stage RNG/coverage计数，`[B,1+K+T]`target有效率和内存可控；测试同市场多账户不能当独立市场N。
7. 原 `FullCheckpoint`：连续N与 N1+冷启动+N2 严格一致；包括 skipped batch、sampling RNG、prefetch、优化器、消费/更新两计数。篡改Corpus/架构/阶段身份拒绝。
8. 完整Model forward、真实参数更新、纯Actor eval、合法Argmax ADD/WAIT，至少一项合成任务loss下降；不能将F3的CPU_PROBE称真实研究通过。
9. 以当前合并后双仓成对HEAD的定向CLI/课程/Reader/F1/F2/Quality MACD V3回归，保留历史金融Env与SOURCE内核正确性；性能异常再检查机器资源，不例行执行`free -h`/`nvidia-smi`。
10. 原始测试命令、退出码、修改清单、最终commit SHA、未运行项、冲突与修复、实施Review、与Master Roadmap同步更新。

**继续阻断直到单独授权**：实际长时间正式F1/F2/F3训练、真实DEV/TEST研究考试、直接实盘、真实交易执行；未经授权不得把CPU合成GO解读为训练准入。

## 9. 反向Review：最可能产生错误结论的十个地方

| 风险 | 必须采取的措施 |
|---|---|
| 训练到的其实是原`pol`而不是`quality_actor` | 强制输出参数前后SHA，Actor param/slot白名单；独立Eval直接引用同一头 |
| 冻结网络仍被Adam weight decay改变 | 包括动量slot与参数的逐键等值；无效梯度回滚整次状态 |
| Q/预测改变了动作 | F3 fuse delta恒为0且来源完整，只有原始Actor logits/合法mask一次 |
| L1高分误被当成L2 ADD | L2优先，L1冲突pair排除及记录；默认L1 pair Loss=0 |
| 过度Balanced导致校准扭曲 | Natural anchor与实际分母，训练和自然考试分开输出 |
| Pair跨账户/跨Corpus | owner、segment、corpus identity、reset严格匹配，未形成Pair不应假称有监督 |
| F3默认到旧legacy Quality阶段 | 显式F3分派、禁`CONTROLLED_FULL`借用预测准入 |
| F1/F2 checkpoint被当F3 same-run resume | 新Run参数接续+Fresh Optimizer，与同Run FullCheckpoint区别明确 |
| 为兼容MACD V2绕过V3身份 | 正式新课程统一V3，旧V2只隔离诊断；不能重签来源 |
| GPU失败/合成PASS被当研究效果 | WP05已有GPU初始化失败；工程CODE_GO≠真实TRAIN≠研究GO |

## 10. 最终提交形式与停止条件

- 先交付第0阶段：双仓集成 SHA、真实冲突处理、V3课程Identity与定向测试证据。如Codex发现已有人合并，则复核后不要重复merge。
- 按06.A–06.F有意义地拆小提交；每次提交只修改获准模块与相应测试，不为维护历史ABI增加冗余分支。
- 最终提交《WP06_IMPLEMENTATION_REVIEW.md》：当前两仓完整SHA、参数/Opt差异、Reader/Teacher支持分母、源handoff证据、精确冷恢复、独立Actor-only小测、测试原文/退出码、未执行项、下一步真实F1/F2/F3训练准备和Roadmap状态。
- 任何时刻发现未来泄漏、错误标签、止损撮合篡改、跨来源错配或状态无法正确恢复，停止发布并报告；**普通DEV CPU小失败可以直接修补并定向测试，不需逐项申请全G3**。

**交付等级：** 工程可交Codex开发；正式训练/研究仍独立审批。完整实现优先，但不应把未完成的来源/研究指标冒充开发失败，也不把工程完成冒充已会交易。

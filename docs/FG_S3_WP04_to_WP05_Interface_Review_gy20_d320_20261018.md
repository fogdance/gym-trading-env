# FG-S3 E+Quality：WP04 → WP05 双仓源码接口独立审查（gy20/d320）

**日期**：2026-10-18  
**审查性质**：读取两个 Git 对象归档中真实 `feat/s3-e-quality-wp04-20261018` 分支源码、WP04 实施报告和验收矩阵；属于静态独立复核，不代表在本环境复现全部 JAX/Ninjax 测试，也不代表训练效果认证。

## 1. 当前代码身份（不得将归档默认 HEAD 当成 WP04）

| 仓库 | 当前 WP04 实际 SHA | WP03 父提交 | WP04 小提交 |
|---|---|---|---|
| Gym | `b7d62113f1e8d4d7ddbdce175acd83568422420e` | `7366886f3b40bf74aa4485978ffcc97e39d0d53e` | A `fa75f01`、B `a74c698`、C `f314036`、D `22fbb52`、E `b7d6211` |
| DreamerV3 | `9715352d4e548b82ba24d3b31070b9fa20d345a6` | `446d1bc99debd2d05c313eafd7d0c414ebd2815c` | A `d7bcbcd`、B `a2cb48e`、C `09fc122`、D `c6c35b0`、E `9715352` |

工作树应建立在上述 WP04 功能分支提交之上；`integration/s3-next` 还不是本轮可靠父提交。DATA 分支 `63f2f8c`、`5873d22` 是独立 Producer 工作线，WP05 开始前应评估必要差异，不允许凭空声称它们已经合入 WP04。

## 2. WP04 已落实的真实可用接口

| 位置 | 已有接口 | WP05 正确使用方式 |
|---|---|---|
| `gym/src/gym_trading_env/research/fg_s3_e_quality_reader.py` | `EQualityCourseReader.open(admission,stage,sampling_spec,seed,index_path)`；`sample(B,T)`返回`[B,T+1]`、`state_dict/load_state_dict`；`HEADS_VALID`及NATURAL/MIXED/STRATIFIED | F2用`HEADS_VALID`，索取`sample(B,K+T)`；不要复制Reader/Join |
| `gym/src/gym_trading_env/research/fg_s3_e_quality_supervision.py` | `QualitySupervisionAdmission.preflight()`、`inspect_source()`、`project_label()` | 继续做TRAIN认证、有效标签mask及内容绑定 |
| `gym/src/gym_trading_env/research/fg_s3_e_quality_foundation.py` | `preflight()`、`prepare()`、`create_agent()`、`F1WindowAdapter`、`require_training_authority()` | 复用配置/模型构造/数据Guard思路；F2需新阶段适配器，不能直接把`WM_NATURAL`当`HEADS_VALID` |
| `dreamer/dreamerv3/fg_s3_e_quality_foundation.py` | `context(model,data)`：1原生前缀+K步`wm_only(training=False)`，`stop_gradient(carry)`；`train()`：六WM allowlist | 提取共享因果context功能，F2重新计算K+T表征但**只训练质量头** |
| `dreamer/dreamerv3/fg_s3_v2_agent.py` | `_S3V2Model.quality_init_loss()` 建完整模型；`wm_only()`；`masked_joint_optimizer()`，`train()`的F1分派 | 增加显式F2分派；复用同一个底层model和opt，不允许走legacy HEADS_ONLY；F2 allowlist须独立定义 |
| `dreamer/dreamerv3/fg_s3_quality_v2.py` | `head_modules()`、`predict()`、`prediction_objective()` | 复用四头/CE/BCE/Huber/mono数学；F2禁止Q/Soft/Actor项且独立valid；必要时抽取纯预测损失 |
| `dreamer/embodied/run/fg_s3_e_quality_foundation.py` | `FoundationRun`、`CourseStream`、`Counters`、`RuntimeState` | F2新阶段Run可以借用相同基础服务，但需要独立run identity、训练阶段与计数，不能修改F1 run |
| `dreamer/embodied/run/fg_checkpoint.py` | `FullCheckpoint(... backend='external_immutable_course_reader')`，`course_reader`状态/不可变外部index校验，run exact resume | F2同Run精确恢复可复用；F1→F2不允许按同Run完整load，必须参数级stage handoff |
| `gym/tools/fg_s3_v2_experiment.py` | `f1-prepare/start/resume/probe/read-check` 子命令 | 加`f2-*`独立授权入口；其他公开E+Quality正式Guard继续保持关闭 |

## 3. WP04 工程验收与未完成事项

归档 `docs/WP04_IMPLEMENTATION_REVIEW.md` 报告 **379个不同选定CPU用例PASS**、真实TRAIN G3和只读Reader复核，以及合成4次WM优化器更新和2+冷启动+2 bitwise恢复。报告声称349参数叶、205个可变、144个冻结，WM六组参数发生变化；本轮实际确认这些功能的对应源码路径、训练分派、快照与证据文件存在，**未重新在本容器安装依赖并复跑JAX测试**。

WP04 **未产生正式TRAIN F1权重**（正式训练更新0，GPU训练NOT_RUN，F1_TRAIN_GO=NO_GO）。可立即做F2开发、合成模型更新与fake F1来源stage-handoff测试；不能称已有受训市场基础供实际F2研究。

## 4. 已确认的阻断/设计冲突（需要先改）

**R1 — 课程声明是上界，不是本轮可实际训练集；不得草率改旧身份**：`dreamerv3/fg_s3_e_quality_course_contract.py::expected_stage_contract('QUALITY_HEADS')`当前把六头（含Q）列为`TRAIN_WITH_VALID_LABEL`。它是“有合法有效标签时才允许”的**条件上界**。WP03没有认证Q数值标签，F2必须定义更严格的**执行权限白名单四预测头**，Q及相关slot始终冻结，并用测试证明执行集是条件合同允许的子集。**尤其注意**`course_identity_payload()`包含派生`stage_contracts`，若修改该函数会改变现有Corpus的`course_identity_hash`，连已认证FG201教材也会因身份不匹配无法读取。故WP05首版不直接修改已冻结的课程矩阵；在独立F2配置/运行Receipt中冻结四头执行合同。确需改变上层课程语义时，另外明确版本升级、完整重新认证，禁止静默冒充原课程。

**R2 — 不能复用旧 Quality HEADS_ONLY**：`fg_s3_quality_v2.py::quality_extensions()`只识别`HEADS_ONLY`/`CONTROLLED_FULL`或明确F1；`_S3V2Model.train()`现有F1专门路由或legacy；需要正式的F2协议、Quality配置与训练分派，不能通过修改legacy Guard绕过C1 Reward。

**R3 — F1→F2不是`FullCheckpoint.load()`**：`FullCheckpoint.load()`恢复Agent+原优化器/Reader/计数/RNG，用于同Run、同身份。F2应对F1正式完成节点独立验证，**迁移全部非`opt/`的同结构模型参数**（包括已训WM和随机但已初始化的Heads/Actor），F2新建优化器、计数、Reader采样与Run身份；留原F1不变。来源完整性与目标结构逐键核对。不得使用`agent.load(regex=...)`偷换阶段，因为其还会恢复旧计数，亦不可静默`strict=False`。

**R4 — 冻结仅靠stop_gradient不足**：旧masked优化器只有F1固定WM allowlist，新增F2四个预测头的逐head有效mask，并核对optax动量/衰减/EMA与全套非F2状态逐键不变。所有有效标签数为零时不允许发生假优化器更新。

**R5 — F2必须保证分窗口有效与右删失**：`prediction_objective(pred,labels,...)`现有数学包含Q计算及mono诊断；可以抽共享纯项，但不能把缺失窗口当OPEN，不能让无效位置的占位值在CE、BCE、Huber、mono或0×NaN中污染计算；应有负例。

**R6 — Reader stage / Training stage须区分**：Reader的`HEADS_VALID`不是课程`QUALITY_HEADS`，T+K样本加prefix长度为T+K+1，K步burn-in不参加任何F2监督。Horizon有效率统计必须分别计数。

**R7 — 流程级授权**：WP04 `f1-*`是独立入口。WP05工程运行可synthetic/有限TRAIN只读/CPU或受控GPU烟测（用户同意时），Formal真实TRAIN参数更新必须有新的F2独立授权/来源；F2训练完成不授予Quality预测改动作权限。

**R8 — DATA状态不可误报**：在最新已给材料中，生产者另线B01曾遇`MACD incomplete previous bucket`；本轮归档并未提供解决后的数据生产新收据。WP05只读旧FG201小样可测试开发，不能把四个新批次当已生成。

## 5. 反向 Review、工作边界和结论

- 最大后验泄漏风险不在模型文件，而在错误的Reader→Loss连接：Teacher标签只做target，改变L1/L2不应改变学生的前向表征/logit，只应在有效位置改变Loss。
- 最大训练错觉是四头参数其实没有更新或某一无标签头被Adam顺带更新：必须逐head报告有效曝光、梯度/参数/slot前后差异，不把总体Loss下降当学会预测。
- 最大恢复错觉是把F1完整恢复当跨阶段接续：同Run exact resume vs 跨stage fresh-source必须是不同API和不同收据。
- 最大研发摩擦是每次合成迭代都跑正式G3或连续查询机器资源：DEV定向回归、默认资源空闲，按任务选择设备和提高利用率；DATA发布/FORMAL训练再履行其真正需要的认证。

**R9 — 现有Course内容身份绑定派生矩阵**：`validate_course_spec`附加`stage_contracts`，`course_identity_payload`将其一起哈希，而`SupervisionSource.inspect_source`再与Corpus签名匹配。因此WP05收窄F2执行白名单时不应自动修改课程V2函数或要求重新生产原Corpus。

**审查结论**：`gy20/d320` 已足以签发 WP05 正式**开发**任务书；但真实受训F1权重未提供，F2正式训练仍NO-GO。此差异是明确的依赖状态，不是WP05工程开发阻断。

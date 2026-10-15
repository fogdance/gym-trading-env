# Quality-first V1：实现与验收记录

## Summary

本实现按用户最新确认采用《Quality-first Architecture Design V1》：质量目标为 **B(t,d)**，不额外扣 MAE；先训练、评价辅助头，预测效果认证后才允许影响 Actor。原 Reviewed V1 的 Full 网络、Loss 和融合能力保留为后续受控实验。**工程接线通过 CPU 验证；预测效果、Actor 改进和 GPU 训练未认证。**

开发仓库为 `fg-s3-quality-v2-dev/{gym-trading-env,dreamerv3}`；源码始于 Gym `c3a808437ef277ee87f25b0ae8803e31fa5dca74` / Dreamer `eb63de710fd693fad7ac76f797f3d9f39e331b98`。最终提交及执行代码对应关系见验收收据索引。既有 Tutor-CD、Tutor-CD Full、B2 的训练工作树及工件未修改。

源码提交为 Gym `f71124a45d52b80b862cf7f813f1491d8a901de7` / Dreamer `29ac494b6e3cdbd405a262f66dfcc4a040219ef2`。[源码绑定收据](receipts/quality_first_delivery/implementation_git_binding.json) 验证全部 210 个运行源码文件逐字节匹配。后续交付文档提交不改变运行源码；最终 HEAD 与运行源码绑定另在原 Run 的 `final_git_binding.json` 保存。

## Evidence observed

### 先验收附件，再开发 Catalog / Student

独立 Patch 验收提交：Gym **`636d839efc45ff38cb914e9e77e99b75657fb32b`**；Dreamer 为未修改的 `eb63de710fd693fad7ac76f797f3d9f39e331b98`。另在附件注明的 Gym `871ca09` / Dreamer `730939d` 基线重复验收。两个基线各 41 项通过。完整记录、原补丁及真实完整账户对照见 [Patch 验收](quality_first_patch_review_20261015/ACCEPTANCE.md)。附件三个 Python 文件保持字节一致。

真实 TRAIN FG201：零次、一次、两次原生 SL 后，以及 WAIT-only 到相同 cursor 的五组对照，均与重新构造并恢复完整快照的独立原 Env 的**完整最终状态**一致。第一次 SL 后权益 299592 元，第二次 298338 元；旧严格 Oracle 拒证，合法入场仍分别得 95.78125、98.125 分。原账户、因果观察、行情/特征和 state_path 文件不变。旧 Oracle 数值 Q 仍是 null。

### 后续真实 Catalog 与神经更新

最新真实运行入口均为原 `tools/fg_s3_v2_experiment.py`，没有新增训练启动器：

- CPU 准备及 100 步、2 更新运行：`artifacts/fg_s3_v2/{prepared,runs}/QUALITY_FIRST_HEADS_CPU_FINAL_v5`。
- 独立 Catalog：`artifacts/fg_s3_v2/quality_checks/QUALITY_FIRST_FINAL_CATALOG_V5/quality_catalog.json`；对应原生旁路日志、状态归属、来源和覆盖统计均在同目录。
- Actor-only DEV 与封存后补评使用独立 `quality_checks` 输出，不写历史 Run。
- 真实 Tutor Full 1M 来源的 CPU 准备：`artifacts/fg_s3_v2/prepared/QUALITY_FIRST_HEADS_GPU_1K_v2`，状态 `PREPARED_NO_TRAINING`，来源 40000 次更新、1000000 Env 推进。源 Checkpoint manifest SHA256 为 `d6b03a8f2e3604885a7550c9bd217471e290c7840f91dd1ecc93bf93658e9098`；未启动 GPU。

实际 250 步 scripted 原账户 Catalog：250 个已完成原生记录中有 **12 个双动作合法空仓决策**，12 个成交/质量有效，7 个严格可认证；5 个低于 30 万元的合法空仓状态均可评分，严格可认证为 0。其余 238 条为持仓被迫 WAIT，不能当作可学习 WAIT 正例。12 个反事实结果为 10 TP / 2 SL；它们包含未来独立重放，不能相加作为可执行收益。

| cursor | 当前实际权益 | B 分 | 严格认证 |
| ---: | ---: | ---: | --- |
| 47 | 299592.00 | 95.78125 | 无 |
| 84 | 299874.00 | 90.625 | 无 |
| 187 | 298338.00 | 98.125 | 无 |
| 209 | 298620.00 | 96.09375 | 无 |
| 244 | 298902.00 | 99.84375 | 无 |

真实神经运行的 100 步样本有 12 个质量有效、9 个严格有效候选，包含 3 个低资金候选。第一次更新保持来源优化器的 LR warmup 起点，梯度非零但 LR=0；第二次更新四个辅助头实际改变。Actor、Critic、Encoder/RSSM/WM、两个 Q 头均为零梯度、零参数变化；修正后的 `wm_updates=0`。

| 分支 | 第二次真实 CPU 更新梯度范数 | 参数变化范数 |
| --- | ---: | ---: |
| Outcome event | 0.1687845 | 1.18367e-6 |
| touch | 0.0427502 | 9.88570e-7 |
| adverse | 0.0640542 | 8.93880e-7 |
| QualityScore | 0.1021377 | 8.96836e-7 |
| QualityActor / Q_C1 / Q_V2 / 原模型 | 0 | 0 |

保留的 Full 分支另做实际 JAX CPU 后端测试：七个新模块均有非零梯度、参数更新；Actor 的 Natural、Balanced、Strict Pair、Quality Pair、Soft、合计项均测得梯度；预测融合 stop-gradient、无支持分支的参数及优化器槽冻结、准确恢复下一次更新均通过。该测试使用合成监督，**不构成研究准入或预测能力证据**。

## Root cause / scientific boundary

30 万元是旧 Strict Oracle 保留路径的资金证书下界，不是原 Env 的最小合法开仓资金。单笔后果、质量分、最优动作分别认证，不能以旧 Oracle 拒证制造 WAIT 标签。质量高也不能推出当前 ADD 优于 WAIT；严格 WAIT 在数值 Q 缺失时只标作弱偏好，保留平局可能。

最新 DEV smoke 为 100 步、8 个可评分候选。Score MAE **24.20572**，TRAIN 中位数先验 MAE **15.31975**。互斥于质量监督的 FG401 有 11 个可评分候选，Score MAE **49.57939**，TRAIN 中位数先验 **57.20644**，初始头同状态参考 **49.579385**。5 分钟 Outcome NLL **1.335104**，初始参考 **1.335103**；没有证明预测优于参考。三项 logit 修正均为 0，预测对 argmax 影响为 0。该 smoke 验证推理、封存、补评链路，不能通过预测门槛；保持 `PREDICTION_GATE_PENDING / NOT_CALIBRATED`。

## Changes made

### 四层 Catalog 与来源

| 层 | 实现合同 |
| --- | --- |
| L0 | 当前原 Env 的真实 WAIT/ADD mask、有效性、原因和规则来源 |
| L1 | 完整账户快照 hash → 独立原 Env ADD→WAIT；成交/拒单/TP/SL/OTHER/RIGHT_CENSORED；原生事件、净 PnL、结算与最终状态 hash、风险事实 |
| L2 | 独立 B(t,d) 协议、分数有效位、TP 新事件奖励；原 C1 已观察奖励的离线 TP 差额替换；SL=0，OTHER/拒单/右删失=null |
| L3 | 独立严格动作认证、原因、来源身份与证书下界；无认证则 action=null；不以 L1/L2 反推最优动作 |

`NativeQualityTeacher` 只接受 TRAIN，拒绝未认证 E profile。来源文件在打开时校验 SHA256，每次查询检查 device/inode/size/mtime/ctime；真实行情与规则也核对。Query 不改原账户，拒单和异常分支也检查不变性；独立 executor 不调用可能写 state_path 的 close。

Catalog 保存原执行动作与原生 stepid，按 stepid、状态、账户和教师来源严格联结；未来标签只在旁路与训练 join 中出现，不进入 Actor obs、RSSM carry、policy cache 或物理 Replay chunk。TRAIN 来源由冻结清单及 prepare binding 认证；DEV 补评生成的数据标为 evaluation-only，禁止训练 join。

七个 Outcome 时间范围为 5/15/60/120/240 市场分钟、入场官方日末、下一官方日末；touch/adverse 也有各自有效位。`OPEN` 表示已完整观察至该 H 仍未退出，数据右删失单独保留。未知负事件窗口无监督，已观察到的触达可保留；质量未知不转成零分。持仓市场分钟含成交 bar，零时长元数据不计时；跨日按官方 trading_date，不按墙钟日期。

### Student、梯度边界与保留 Full

默认 `quality_stage=HEADS_ONLY`：原 C1 Reward、beta=0、BC lambda=0；原 Backbone/Actor/Critic、QualityActor 和 Q 头冻结，辅助表示梯度尺度为 0，仅 Outcome、touch/adverse、Score 头更新。实际优化器槽也冻结，防止残余动量/weight decay 改动。目标先验只从匹配 Checkpoint 的 TRAIN 样本计算。

保留的 `CONTROLLED_FULL` 复用原 presets，含新 QualityActor、Q_C1/Q_V2、多任务联合 Loss、Natural/Balanced/E2、两种 Pair、Soft Teacher、单一训练/推理前向和预测→Actor logit 接线。Actor 使用停止梯度的 latent；预测修正停止梯度，mask/unimix 只施加一次；辅助表示梯度尺度恢复为原方案 .1。数学实现见 `dreamerv3/fg_s3_quality_v2.py`，当前 B 分目标使用新协议。归档的旧 S=B−1.5R 纯函数保留但不生成本 Catalog 的目标。

Full 默认权重保持原方案：BC 混合 .8/.2；Strict Pair .1、Quality Pair .05、Soft .1、Outcome .2、Score .2、Q_C1 .1、Q_V2 .1、mono .02。QC1 的 WAIT hinge 允许平局；无真实数值证据不回归伪 Q。QV2 保留金融账户原生有界后缀证明；当前 HEADS_ONLY 不启用该研究分支，数值 Q 覆盖为 0，`FULL_Q_PENDING`。

启用 Full 必须提供绑定真实 Checkpoint、互斥 TRAIN holdout、先验/参考模型实测比较和独立 DEV 证据的预测准入；Q 融合还有独立证明门槛。准备与 transfer 还绑定已评价的四组预测头权重及摘要，要求与固定源 Backbone/Actor NN 一致，再原样载入已评价预测头；优化器/Q 保持独立。代码不会因单笔高分或可读旧标签自动准入。可选 Quality TP Reward Schema 保留独立开关和回归测试，**本阶段所有研究配置继续原 C1**，避免同时改变 Reward 和 Student。

### 评价与后续准入

共享 `evaluate --holdout-contract CZCE.FG401` 从原签名 TRAIN 清单构造 evaluation-only 快照，拒绝已用于质量训练的合约和清单外合约，不构造训练教师。DEV 与 TRAIN_HOLDOUT 都先封存原 Actor 的完整因果预测、动作和最终账户，再单独补评；不解封 TEST。

`evaluate-reference` 载入初始 Checkpoint，以独立模型和 carry 在同一实际动作历史上预测，每步验证 cursor、完整账户决策身份及 mask，最后核对完整账户；不把该回放账户称作参考 Actor 自主成绩。评价统一冻结配置 seed、counter=0，不改训练 RNG。`compare-predictions` 要求逐条状态／动作／原生未来标签相同，分别比较 TRAIN 先验和真实初始头参考。FG401 本轮 101 条因果记录、11 个评分候选，完整匹配通过；有效样本和校准规模仍不足。

每个保存点生成独立 Catalog 收据，绑定该时点 journal 的字节前缀、来源、teacher 白名单、Replay stepid、prepared 和 FullCheckpoint。后续 journal 追加不破坏旧前缀证明；篡改前缀必须拒绝。FullCheckpoint 恢复测试已覆盖这两条分支。

### 源码位置

- Gym：`research/fg_s3_quality_score.py`、`fg_s3_quality_teacher.py`（附件原件）；`fg_s3_quality_teacher_v2.py`（Catalog/多窗口适配）、`fg_s3_quality_catalog_v2.py`、`fg_s3_quality_q_v2.py`、`fg_s3_quality_experiment_v2.py`、`fg_s3_quality_prediction_evaluation.py`。
- Dreamer：`dreamerv3/fg_s3_quality_v2.py`；通过 `fg_s3_v2_agent.py` 与原 joint optimizer 接线。
- 原共享入口、Collector/Sidecar、SourceAdmission/transfer、FullCheckpoint、共用 Tracker 均复用；未在 artifacts 增加独立训练脚本。

## Verification performed

实际 XML、日志、梯度明细及运行收据见 [收据索引](receipts/quality_first_delivery/README.md)。主要命令使用解释器 `/home/v/miniconda3/envs/dreamerv3/bin/python`，`JAX_PLATFORMS=cpu`，BLAS/OMP 单线程，pytest 无 cacheprovider、无 pyc 写入。最终主验收 107 项、Student 后端 8 项、恢复 1 项，加现有合同回归 34 项，共 **150 项通过**；另列两个独立 Patch 基线各 41 项，不合并重复计数。

- 数学、来源/时钟、合法性、低资金、分层 join、原 Env/Oracle、组合配置与 Tracker 专项回归。
- 八项实际 CPU Student 后端测试：全头梯度、预测 stop-gradient、默认辅助头冻结边界、参数 warm-start、已评价预测权重绑定、无未来标签影响 Actor、精确更新恢复。
- 真实 Driver/Replay/Sidecar：prepare→100 步→resume 200 步，与连续 200 步所有更新指标及最终参数摘要逐项相等；FullCheckpoint 含模型、优化器、Replay、Collector、Driver、调度、RNG、计数。
- 真实 TRAIN Catalog 250 步、真实神经更新 100 步、Actor-only DEV 100 步、封存后补评；均使用新隔离目录。
- 共用 Tracker 只读准入成功，实际导入 335 项指标；未联网或上传。

现有合同初次回归的 1 项既有失败已按用户后续要求解决：统一评价提交 `34023e7f` 接入奖励对账，测试仍沿用 `19e72f88` 的旧账户夹具，遗漏 `total_reward`。修复夹具并增加奖励不一致必须拒绝的断言；生产评价器未改动。目标测试及全部 34 项现有合同回归通过，见 [根因与清单](FG_S3_QUALITY_REGRESSION_FIX.md)。[基线失败 XML](receipts/preexisting_actor_metrics_c3a8084.xml) 和修复前日志完整保留；真实 DEV 与 FG401 账户的 reward 对账另有实际收据。

## Risks / not verified

预测效果、校准、多 seed、多时间 fold、原 Full Actor 决策影响、真实 Q 有效覆盖均未认证。FG401 留出与同状态参考工程链路已实测，尚未形成研究准入。Full 从真实已通过预测准入的 Checkpoint 迁移尚无合格来源可验收；已验证复制权重的数学／schema 合同，未伪造 PASS 来源。全量 18 合约 Catalog 的吞吐、极长/右删失路径成本和 1M 资源峰值未验收；默认原生重放至退出，可选有效市场分钟上限将未结束路径保留为右删失，不能凭 250 步样本宣布大规模性能通过。极低保证金的真实账户未覆盖；MARGIN_CALL、OTHER、拒单、跨官方日/跨自然日和右删失有可控原 Env 边界测试。费用是冻结研究常数，日历是 user-confirmed observed clock，未追加交易所历史费用认证。E 明确拒绝。

CPU 测试运行在开发过程中，收据如实记录当时 Git 基线及 dirty 状态，并保存执行代码 SHA256；最终代码映射另行校验，不改写旧收据。GPU/10K/1M 均未启动；没有解封 TEST。

## Next recommended step

按 [启动与完整对照计划](FG_S3_QUALITY_FIRST_RESEARCH_LAUNCH_PLAN.md) 审阅首次 GPU 1K 技术验收及后续小预算研究。GPU 启动须另行授权；首次 1K 通过不自动放行 10K、Full Actor 或 1M。

# S3 Quality-first V2：gy15/d315 设计—源码差异与风险 Review

**日期** 2026-10-15  
**范围** 只读 Git 对象：Gym `feat/fg-s3-quality-v2` @ `f71124a45d52b80b862cf7f813f1491d8a901de7`（后续文档提交 `5371a64`）；Dreamer `feat/fg-s3-quality-v2` @ `29ac494b6e3cdbd405a262f66dfcc4a040219ef2`。对照《Quality-first Architecture Design V1》、`FG_S3_QUALITY_FIRST_IMPLEMENTATION.md`、原 Reviewed V1 补充模块和启动方案。没有修改上传归档或运行训练。

## 验收分级

- **设计一致**：L0 当前 Env 交易合法性、L1 完整账户叉路、L2 `B(t,d)` 不扣 MAE、L3 独立严格 Oracle 身份；HEADS_ONLY 不允许预测干预 Actor；旧 C1 / 新 TP Reward 两套 Schema 分离。
- **工程证据**：源文档报告 150 项 CPU 回归、250 步 Catalog/100 步神经运行、冻结源 Checkpoint 继承及恢复；此 Review 能确认源码与收据存在，**未独立在用户的 conda 环境重跑**。本容器 12 个核心模块 Python AST 成功；无 `gymnasium`、`optax`、`elements` 等完整运行依赖。`git diff --check` Dreamer 无报告，Gym 的空格提示来自存档日志/XML，不是功能 Bug。
- **研究效果未验收**：DEV 100步/8个可评分候选，Score MAE 24.20572 > TRAIN 中位数 MAE 15.31975；FG401 留出11个候选，Score MAE 49.57939 ≈ 初始参考 49.579385；Outcome NLL≈初始参考。预测门禁 `PENDING`，Full Actor、数值 Q 融合、GPU、10K、1M 均无实测成功证据。

## Findings（依据源码，按需要处理的优先级）

### F1 · P0（在 Actor 接入前）预测准入缺少充分的样本和风险覆盖约束

**代码** `src/gym_trading_env/research/fg_s3_quality_catalog_v2.py:290–374`，尤其 `355–360`、`365–373`。`require_prediction_admission()` 要求已认证 Checkpoint、TRAIN_HOLDOUT、独立 DEV、预测 NLL/Brier 和 Score MAE 同时优于先验及初始参考，这些检查有价值。但每条对照仅需 `samples >= 1`；函数未强制多个时间区段/合约、7 个 horizon 覆盖、足够 SL 支持、低资金覆盖、最小 Score 样本数、预测校准及错误 ADD/SL 风险上界。`admission['comparisons']` 可只含一个 horizon。

**影响** 有可能通过极小、偶然有利的样本，使 Full 的 `quality_fuse_outcome/score` 介入真实 Actor。这与设计文件中“先独立预测评价、风险分层及足够支持，再影响 Actor”的科学门槛存在缺口。**这不是已发生的错误开仓**；当前 gate 仍是 PENDING。

**修正建议** 把所有准入条件版本化成预先冻结的 `minimum_samples_per_horizon`, TP/SL 支持、官方日和低资金分层、合约/时间段数量、置信区间或重复试验，以及 DEV 决策安全阈值。不能在 DEV 结果出来之后临时选阈值；未达成保持 PENDING。

### F2 · P0（在 Q 融合前）数值 Q 准入验证与预测准入不对称

**代码** `src/gym_trading_env/research/fg_s3_v2_experiment.py:75–81`。当 `quality_q_admitted=true` 时，目前只读取一份 manifest，检查 `status=='PASS_TRAIN_Q_FUSION_ADMISSION'`、`fold=='TRAIN'`、`q_v2_pair_valid_count>0`、`short_run_verified`；随后只记录该 manifest 的 SHA。相比预测准入，没有独立跟踪并重验具体 Q 状态/账户快照、reward/utility 版本、完整 suffix proof、数量/代表性、来源 Checkpoint/模型和留出实测。Dreamer `fg_s3_quality_v2.py:106–112` 因此有机会启用 `delta_q`。

**影响** 一份状态标志充足但来源支撑不足的收据，可能开启数值 Q 对 Actor 的调整；原设计要求所有辅助信息“严格证明后才用于决策”。这是准入验证缺口，而非报告宣称的已发生故障。

**修正建议** 实现和 `require_prediction_admission()` 一样完整的 `require_q_admission()`，绑定原 Env/Reward/Oracle/Q 证明、样本状态、独立复验、Checkpoint、Q 风险和训练覆盖，禁止仅靠计数/标志放行。

### F3 · P1 新 Strict Oracle 候选与原 Env 的重建对账抽样过少

**代码** `src/gym_trading_env/research/fg_s3_quality_q_v2.py:148–199`，尤其 180–188。新目标的 TP 候选用 `quality_tp_reward()` 替换旧 C1 的事件项，旧候选及 Replay 来源重新签名（做法正确）；但构造出的整条合约候选列表只对最多 8 个均匀时点及首个 TP、首个 SL 调用 `reference_candidate()` 对账。随后将全体 `rows` 用于 Strict Oracle 认证。

**影响** 未覆盖大量可能不同的组合（交易日切换、早期逆行与TP同根、长时间持仓、费用/保证金边界、数据终止、gap）。不能把“样本通过”解释为“所有重算候选经原 Env 验证”。这是**验证覆盖缺口**，并未观测到具体不一致候选。

**修正建议** 在原有 `verify_candidates()` 基础上版本化分层随机+风险边界抽样，至少覆盖每一退出类、天数 d、B 分段、逆行窗口、chunk/合约边界；对于真要作为 `exact_proven` 标签发布的代表路径，逐笔认证或证明批量算法与原 Env 完全等价。

### F4 · P1 质量预测评分直接推动 ADD，仍缺相对 WAIT 价值约束

**代码** Dreamer `dreamerv3/fg_s3_quality_v2.py:100–121`：`delta_s=.5*clip((score-60)/40, -1,1)`，当 `quality_fuse_score` 时加到 ADD logit；`delta_o` 用日内 TP/SL 与早期风险做有界修正。虽然预测被 stop_gradient、设有概率准入且单步修正有界，但 **单笔 B 分高并不蕴含 ADD 优于 WAIT**（本项目曾发现很多“TP盈利、Strict WAIT”的例子）。预测 MAE/NLL 优于参考，并不能推出该 logit 修正在决策层安全。

**影响** 即便 F1 的预测效果门槛满足，仍可能增加 Strict WAIT 状态下的错误 ADD、增大 SL/资金回撤。这是研究设计上的因果目标错配风险，不是未授权的当前生产 Bug。

**修正建议** 单独加决策门禁：在冻结同状态轨迹和合法原 Env 闭环中，验证相对价值/Oracle 误报、SL、资金回撤及干预方向，再放开 score-to-logit；先允许只读诊断 delta_s，不要把预测精度门禁等同交易效果门禁。

### F5 · P1 数值 Q 默认上界限制导致 Full Q 实际监督稀疏

**代码** `fg_s3_quality_q_v2.py:49–75` 里 `max_suffix_bars=12` 默认；`solve()` 如果合约后缀超过12根就直接 `OUTSIDE_BOUNDED_PROOF_DOMAIN`，`q_c1/q_v2` 无值。只有合约末尾很窄区间可得到数值 Q。`fg_s3_quality_catalog_v2.py:124–128` 对 Q 数值有效位正确独立屏蔽，避免伪监督；但意味着完整合约多数入场状态没有数值 Q。这和“Full 网络具备 Q 头”不是同一回事。

**影响** 无法以当前 12-bar 边界和有限候选称已经完成可用于真实择时的 Q_C1/Q_V2 教学。官方报告也明确 `FULL_Q_PENDING`、真实 Q 有效覆盖为零。**属于研究覆盖不足，不是立即致错 Bug**。

**修正建议** 提前评估真实计算复杂度和 Q 支持率，再考虑安全扩大精确后缀域，或另建清晰标明 `reference_policy_relative` 的近似价值教师，不能标成严格 Q。

### F6 · P2 历史 MAE 扣分公式仍在新源码中可被调用

**代码** `fg_s3_quality_teacher_v2.py:30–41` 的 `score_policy_v2_1()` 仍返回 `B−1.5×加权MAE`，而当前 Catalog 目标和真实 Reward 使用无 MAE 额外扣分的 `B(t,d)`。函数注释已经明确 `Archived Reviewed V1 formula`，且当前 `validate_label()` 要求新的 `score_policy`；因此**现阶段没有发现该旧函数混入正式 Catalog**。

**影响** 后续调用方可能误用两个语义相反的同名“quality”打分器，形成难查的分数漂移。

**修正建议** 将旧纯函数移至 legacy/研究命名空间，显式废弃或按配置校验；减少 `fg_s3_quality_score.tp_base_points()` 与 `fg_reward_candidates.quality_tp_reward()` 对公式的重复实现，新增随机与边界跨实现一致性测试。

### F7 · P1 扩展到全量 Catalog 前的性能/内存风险

**代码** `fg_s3_quality_teacher_v2.py:303–362` 每个合法状态克隆完整账户，从当前 cursor 执行 ADD→WAIT 至退出/边界。`fg_s3_quality_q_v2.py:162–199` 还会构建全合约候选/Oracle。业务逻辑符合原 Env；但若重复状态、多合约长持仓或右删失，独立回放重叠，容易接近 O(N×平均持仓长度) 的成本，最差近 O(N²)。项目文档同样明确没有全量18合约吞吐和资源峰值验收。

**影响** 大规模训练中可能卡在 CPU Catalog、磁盘日志或超时；不能用250步 smoke 外推 1M。

**修正建议** 做各持仓区间长度分布、候选个数、峰值内存/磁盘测量；精确状态缓存、交易时段索引和分层 native Env 对账需保持来源身份，不能用未经验证的价格阈值算法替代执行器。

### F8 · P2 旧资金门槛依然存在于 L3，且被显式保留

**代码** `fg_s3_quality_catalog_v2.py:258–260`：原 `STRICT_SUFFIX_AUDITOR` 如果权益低于30万元，会被保护逻辑拒绝当成已认证。L1/L2 已取消此门槛，可以给亏损后真实账户评分。该分离符合 Quality-first，但**并不代表严格 Oracle 的资金证明已修复**。报告必须区分“亏损账户质量数据有效”与“亏损账户已有严格动作最优性证明”。

**结论** 设计符合当前决策；不应作为 Bug 去删除这项保护。若将来需要 L3 也覆盖亏损账户，应以新证书另做证明。

## 重要的非 Bug 事实

- `HEADS_ONLY` 所谓“完整实现”指核心功能、CPU 训练接线和读写恢复到位，**不包含预测效果认证或 Actor 利用率提升**。
- `FGRewardScorer` 现在有版本化新 TP 奖励；生产训练配置仍旧 C1，属于实验轴冻结，不是新奖励没实现。
- Q 辅助头和 FULL 概率/价值融合保留模块及单测；真实 Q 后验支持、合格的 predictor admission 和 FULL GPU 对照尚未完成，不得拿“代码已经存在”替代研究完成。
- 当前源码具备源指纹、Replay stepid、账户状态及旁路监督独立性等工程防线；本次静态审阅未发现明确的未来标签直接进入 Actor observation 或 Replay 的路径。

## 建议决策

1. 维持 `HEADS_ONLY` 与 `PREDICTION_GATE_PENDING`；不授权 Q/Score/Outcome 直接干预 Actor。
2. 先修 **F1/F2** 的准入证据合同，补 **F3** 新 Strict Oracle 的分层原 Env parity；把 F4 的动作安全验证加入第二道门槛。
3. 再进行 **GPU 1K 技术验收** 和 **10K 预测研究**。先确证“Score/Outcome 预测本身能学会且跨样本有效”，再考虑决定性 Actor 消融。暂不启动1M。

## 本轮独立可复核行为

- 已解包 gy15/d315，读取源码和文档；12个关键模块 Python `ast.parse` PASS。
- `git diff --check` Dreamer 通过；Gym 提示均为存档运行日志/XML 的尾随空格，无已发现功能性错误。
- 未在用户原始工作树操作，不修改 Git/Checkpoint/Replay/训练配置；未运行 GPU 或训练。无法在本容器进行 gymnasium/optax/elements 的真实端到端测试。

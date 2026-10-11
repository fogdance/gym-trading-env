# FG Single3 TP5/SL30 实施计划与进度

日期：2026-10-11。本文是已确认计划的工程执行记录；状态必须以实际运行收据为依据。

## 1. 正式依据与研究边界

- 正式任务书：`FG_DreamerV3_Single3_TP5_SL30_Reward_OracleReplay_Execution_Plan_V1.md`。
- 任务书 SHA256：`9e3287a5bbe9eec93788275ae5b3ee082677c637b266690bad08732823679744`。
- 目标：在给定历史 LONG/SHORT 方向的条件下学习 WAIT/ADD 入场择时；一次3手、最多一批，TP5追求尽快退出，SL30真实退出并释放仓位。
- 本次按 E → R → O → D → H 执行，准备模型先导配置，但不启动 GPU、不训练模型、不读取封存测试价格、不新增 Actor BC Loss、不改网络结构。
- 历史方向只具有 `conditional_direction_only` 资格。工程通过、Oracle后验上界和模型学会择时必须分开报告。
- 用户最新要求：代码直接在现有目录开发，不创建隔离 checkout。两个仓库使用实验分支 `feat/fg-single3-tp5-sl30-oracle-replay`；原 Replay、Checkpoint、日志及历史研究产物不覆盖、不重签哈希、不删除。

## 2. 真实基线与需要调整的入口

Gym 起始 HEAD：`e2263713afa173f65ab4d42e32ae19d6e8109cd5`；原 R0–R2 完成标记修复已独立提交为 `721392b`。Dreamer 起始 HEAD：`64c1a51e2e1fd3bd4442662795ca313a9ed4afe3`。

已读取本任务相关 Env、PositionManager/原子开平、价格/费用/保证金、Reward、Driver、Replay、FG准入、离线训练及折扣执行链，以及旧 R0–R2 和 Reward V2/V3 报告。引用的另一份单仓位研究参考文件未找到；不以缺失文件替代正式任务书。

实际基线测试：Env/Reward 49项通过，Dreamer相关4项通过；此前完成标记修复的Replay测试13项通过。它们不代表新单仓位已验收。

必须调整：

1. 旧 Env 严格限制 max6、两批、无SL和V1 Reward。新增明确协议而非修改旧协议默认值；配置、快照、哈希和状态恢复一致升级。
2. 新 Observation保持 `(60,18)/(9,)/(2,)`，但语义显式版本化。position_ratio按3手归一，第二槽归零，pending字段表达真实SL待成交状态；验证SL剩余距离可从TP距离推导。
3. 旧Replay的P3、容量和TP统计不适用于新实验。复用采集和审计公共代码，增加协议/策略/事件钩子。
4. FG准入目前不允许offline_pretrain；offline_pretrain调用完整agent.train且没有在线Mask Gate接线。新实验只补准入、身份检查和既有Gate复用，不宣称WM-only。
5. 旧主配置TRAIN截止与实际FG409/all Fold不一致。只以验证后的训练manifest为权威，不重分割/重导出行情。旧Reward报告使用的验证期评分不作为本次TRAIN评分。

## 3. 冻结业务协议

| 项目 | 决定 |
|---|---|
| 初始资金/交易量 | 300000 CNY；ADD3；最大3手；最多一个entry_id |
| 动作 | WAIT=0，ADD=1；沿用下一执行bar开盘撮合 |
| TP/SL | 以实际入场成交价为基准，TP+5，SL−30；LONG/SHORT镜像 |
| 双触达 | 正式路径SL优先，记录歧义；TP优先只作单列敏感性诊断 |
| 入场同bar | 保留TP收盘确认；SL按公开的保守风险优先协议，不虚构tick顺序 |
| SL跳空/pending | 跳空用更差可执行开盘价；量不足/拒单不释放、不发SL奖励；挂起后下一可成交bar优先退出，即使价格恢复 |
| 风险优先级 | 保留保证金、主力、到期的原优先级，终局风险与可恢复SL分开 |
| 时间 | 退出反馈沿用bar完成时钟，额外登记价格参考时点；早期逆行用有效bar，自然分钟仅用于TP曲线 |
| 退出数量 | 新协议自动成交只有0或3；部分成交请求先拒绝再记账；旧协议部分成交路径保持 |
| 480分钟 | TP奖励归零边界，不引入超时强平 |
| Chunk | Env1024不改变；重复reset观察不增加有效bar、不重发风险或奖励 |
| 非标准完整退出 | `−L×min(1,max(0,−net_trade_pnl)/(3×multiplier×30))`，净盈亏含真实开平费用；盈利退出不另发奖励 |
| 未成交/右删失 | 保留仓位、事件和删失标志，不虚构平仓或损益 |
| Oracle | TRAIN未来仅在Sidecar；主目标γ=332/333、Env推进步；γ=1单列诊断；正式P4只接受exact_proven |

首次逆行10点每entry一次：入场bar算1；前5根 `−beta`，第6～15根 `−beta×rho`，以后不罚但保存触发状态。保留V2的TP时间曲线；SL完整真实成交发 `−L`，可与early10叠加。PBRS和容量惩罚为0。

| 候选 | beta | rho | L | 模型训练是否选定 |
|---|---:|---:|---:|---|
| C0 | 0 | 0.4 | 3 | 否 |
| C1 | 0.5 | 0.4 | 3 | 否 |
| C2 | 1 | 0.4 | 3 | 否 |
| C3 | 1 | 0.4 | 6 | 否 |

C2只作为不依赖Reward的自然策略采集诊断配置。保存真实事件并派生四个候选的原生Replay视图；每视图重新执行对应配置的Env对账，不伪造原始身份。

## 4. 阶段、接口与验收

### E：Env

修改现有entry_env及必要的数据集/配置扩展点；不复制账户或撮合。实现限制、SL触发/待成交、可观察状态、真实事件和完整快照恢复。

验收覆盖LONG/SHORT、正常TP/SL、双触达、入场同bar、跳空、量不足/拒单、重复fill_id、部分成交拒绝、主力/资金/到期、数据尾、跨1024及恢复。真实TRAIN双方向合约逐步独立重放、费用/权益/账本对账；旧协议回归通过。

产物：`FG_Single3_Env_Execution_Contract.md`、测试和真实运行收据。

### R：Reward

复用已有TP纯函数、首次逆行状态机及事件校验，新增SL/非标准退出评分。先用真实TRAIN完整轨迹研究，D完成后更新全Fold统计。

验收：公式边界、event→reward时序、恢复/重标记幂等、费用和净盈亏一致、Reward版本隔离。报告early10后的TP/SL条件概率、TP时长、MAE、年份/方向差异及All-WAIT风险，不按Oracle高分选参数。

产物：`FG_Single3_Reward_Research.md`、不可变事件trace、四候选评分表。

### O：Oracle

新增训练候选搜索和优化器。原Env为参考，共享行情及成交判定，索引加速；先短区间实测性能，禁止每个候选加载CSV或无界二次扫描。

只有资金无关前提被证明后才省略账户状态；否则精确状态DP，仅做有证明的合并/剪枝。采用数值误差区间，比较重叠时精确复核，平分WAIT。独立短合成穷举、真实边界和固定seed分层候选对账；所有最终P4交易完整原Env重放。

若资源不足或无法证明，标为未证明，保留诊断，停止正式P4及下游发布；不自动改成近似数据。

产物：`FG_Single3_Oracle_Validation.md`、候选表、四候选H333动作表、γ1诊断、最优性证书及重放收据。

### D：新Replay

复用原Driver/Replay/Sidecar/reconcile/audit。18合约重新采集P0、P1、P2a(p=.1)、P2b(p=.5)、P3和四候选P4，所有账户完整执行；不重用旧6手账户轨迹。

P2采集seed=0，合约/策略独立稳定子seed。P3等待15个合法空仓有效分钟，成交/真实退出重新计时；reset重复观察不计数。P0独立基准，不进入默认训练混合。

每Reward提供策略池、natural only、natural+P4，默认Uniform连续序列，不预设事件加权。原生Replay每池1000个真实batch，B16、length65，训练64位置与context分别统计。汇总主动/受限WAIT、合法ADD、请求/成交/拒单、TP时长、SL/early10/非标准退出/pending/删失及0/3手；按年份、合约、方向、日夜盘和唯一市场分钟分解。

产物：`FG_Single3_Replay_R0_R2_Coverage.md`、完整manifest、原生chunk/Sidecar哈希、逐行对账和采样收据。

### H：信用分配

分析H333/1000/3000及γ1的真实事件折扣；区分自然分钟、Env推进步和Replay行；核对continuation/bootstrap、反馈指数、1024及RSSM is_first。相同真实账户快照分叉ADD/WAIT，随后都用冻结P1因果策略，到真实终局或TRAIN边界。

产物：`FG_Single3_Horizon_Credit_Assignment.md`。不把政策反事实、Oracle上界或优化目标权重称作已测量的Critic误差。

### 模型先导准备

复用现有完整离线训练及Mask warmup；增加新协议身份准入、单Reward/新Replay/新Checkpoint约束，旧FG路径不变。准备M0 natural与M1 natural+P4，均H333、B16/64+context1、2000梯度更新、模型seed0、采样seed202；只做CPU预检和FakeAgent接线测试。

产物：可复现配置/命令与`FG_Single3_Model_Pilot_Comparison.md`，明确NOT_RUN；训练Reward未选定。

## 5. 数据、资源、审计及发布纪律

TRAIN源：`artifacts/fg_abcd_v1/v3_shared_prefix/FG409/all/manifest.json`，content hash `ffc45db584106b7c348b7f3aee95096acbcb560c31a7d67fa340f30d2e10009a`。已验证56个冻结输入；18合约、488160行、488154有效分钟、6个零时长元数据行；交易日2018-04-11～2024-04-08。

- 只打开已验证TRAIN白名单中的价格文件。Observation/模型Replay字段白名单严格限制；Oracle未来字段、政策ID和hindsight标签只在Sidecar。
- 每stage记录源码提交及运行前后文件哈希、配置/输入/产物哈希、实际导入路径、seed、对账收据；运行中代码变动使发布失败。
- manifest pending → 完整验证 → 原子发布。失败不留下正式完成标记/PASS报告；不删除失败证据。
- 24逻辑核；当前约21GiB可用RAM、95GiB空闲磁盘。采集初始12 workers、Oracle初始4，实测后最多20；单worker BLAS1，总RSS预算16GiB，磁盘保留20GiB。单合约精确搜索状态预算200万，达到预算不自动降级。
- 先测性能和大小再全量。长任务保存可恢复进度并低频轮询；不以预算预测代替实际覆盖。
- 逐阶段先最小针对性测试，再真实对账、自审和必要回归；提交代码/测试/报告。总体报告明确未完成和未验证项。

## 6. 当前状态（不得把计划写成验收结果）

| 阶段 | 状态 | 实际证据 |
|---|---|---|
| 基线 | PASS | 49+4项既有测试通过；旧完成标记修复已提交721392b |
| 计划记录 | COMPLETE | 本文及原任务书纳入实验分支 |
| E | PASS | run_20261011_03；FG809/FG909完整独立重放及27/25次chunk恢复PASS |
| R | IN_PROGRESS | E已通过，开始候选事件研究 |
| O/D/H | NOT_RUN | 等待依赖Gate |
| GPU/模型 | NOT_RUN | 未授权执行；择时能力NOT_ESTABLISHED |

后续每阶段填写实际命令、耗时、测试数、数据范围、产物路径/哈希、提交和未解决项。最终总报告：`FG_Single3_CPU_Implementation_and_Verification.md`。

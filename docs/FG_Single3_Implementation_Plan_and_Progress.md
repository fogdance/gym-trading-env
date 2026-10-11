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
| E | PASS | run_20261011_04；44项测试；FG809/FG909完整独立重放及27/25次chunk恢复PASS |
| R | PASS（完整TRAIN比较） | R02双合约研究及40视图对账；D已补齐18合约四候选自然策略/Oracle实测比较；参数未选定 |
| O | PASS | 18合约488142候选、3439个独立参考路径、72条最终H333完整Driver账户；exact_proven；5项独立原Env/精确比较测试 |
| 输入因果审计 | PASS | run_20261011_01；18个TRAIN合约×11个截点，60×18 float32观察与当时前缀逐值相同 |
| D | PASS | 378账户、10261209 Replay行；29池×1000真实batch；0非法ADD/6手/不连续序列；完整D_AUDIT已发布 |
| H | IN_PROGRESS | 正式D-Gate后自动启动20 worker、576组同状态分叉；尚无正式H收据 |
| GPU/模型 | NOT_RUN | 未授权执行；择时能力NOT_ESTABLISHED |

后续每阶段填写实际命令、耗时、测试数、数据范围、产物路径/哈希、提交和未解决项。最终总报告：`FG_Single3_CPU_Implementation_and_Verification.md`。

### E刷新及R发布前回归

E04命令：`python tools/fg_single3.py env-test --output artifacts/fg_single3_v1/env/run_20261011_04`。收据content hash：`06ca881053ada31f064a7c488f3140dfdcb51c0a8cb1bf95f1fe1d07ee03ad89`。两个合约逐步/最终状态哈希与E03一致，新增TP统计缓存不改变财务或反馈；执行源码归档在每次产物内。

R01候选逐行Reward对账完成后，补充短chunk测试发现：P3读取CHUNK_END的行政mask=0，错误清空15分钟计时。已改为用当前已知状态判定恢复后的合法空仓机会；真实模型发布mask保持原协议。回归测试使用chunk7，证明首次入场16、SL后再次入场41，并验证候选视图财务和动作一致。R01正式`manifest.json`已改名为失败证据，保留所有原产物及执行源码，不能被下游当成PASS。阶段命令现在先执行所需测试，再生成和签发产物。

R02命令：`python tools/fg_single3.py reward-study --env-receipt artifacts/fg_single3_v1/env/run_20261011_04 --output artifacts/fg_single3_v1/reward/run_20261011_02 --workers 12`。最终收据已PASS；四候选尚未选定。

### 输入因果性与后台阶段依赖

实际命令：`python tools/fg_single3_input_audit.py --env-receipt artifacts/fg_single3_v1/env/run_20261011_04 --output artifacts/fg_single3_v1/input_audit/run_20261011_01`。使用4个CPU worker，只打开冻结TRAIN白名单价格。198个截点覆盖首bar、60、1024、固定seed202随机点、日边界及数据尾；完整行情计算的60×18观察窗口与仅使用当时前缀重算结果精确相同。收据content hash：`4c44ee52e79c44699c2acf9091b1bebb6608856175414cb584588d320572129b`。这是实际抽样因果审计，不能表述为逐一穷尽所有行情截点。

`tools/fg_single3_pipeline.py`复用各阶段正式CLI，等待完整签名O收据后自动启动新Replay生成、每池1000真实batch覆盖审计、576个同账户快照ADD/WAIT反事实、8份受保护的模型先导模板。任一阶段失败立即停止下游；流水线没有神经Agent/GPU训练命令。当前日志：`artifacts/fg_single3_v1/cpu_pipeline_run_20261011_02.log`。尚未发布的阶段保持NOT_RUN/IN_PROGRESS，不能因实现或单元测试通过改写成实际数据验收PASS。

已提交：Gym `c54b2af`（R及Replay扩展）、`5f49900`（精确Oracle）、`a2cdc64`（D/H/输入审计/流水线）；Dreamer `61a3987`（联合离线先导准入与完整恢复）、`adbeae5`（Agent初始化前Checkpoint身份及完整性校验）。Dreamer FakeAgent恢复与Mask Gate接线测试2项通过，模型训练仍NOT_RUN。

模型模板自审发现真实preset中`run.steps`是字符串`"2000"`，来源为原默认YAML的科学计数法字符串类型：旧的整数比较误拒绝准入，且Counter与字符串比较会抛TypeError。新Single3准入使用精确数值预算比较，联合离线循环将已准入预算规范化为整数；未更改旧模式行为。新增真实preset回归验证能抵达数据准入、2001更新仍被拒绝；FakeAgent使用字符串预算完成门控及完整恢复。配置YAML重新载入为elements.Config后与解析配置一致。此项CPU验证不初始化神经Agent。

`tools/fg_single3_reports.py`从各阶段签名收据汇总Markdown，验证被引用报告的文件哈希。缺失Gate显示IN_PROGRESS/NOT_COMPLETED；`--require-complete`拒绝在CPU阶段未结束时发布完成报告。该工具支持指定各stage运行目录，只读取收据/报告，不打开价格数据。

先导准备的发布自审复现了源码哈希校验失败却留下“准备完成”报告的路径。把首轮完整哈希校验前移到报告生成之前；已有最终发布检查失败仍保留`.failed.md`证据。新回归在旧顺序下失败（实际存在`pilot_comparison.md`），修复后通过（没有完成报告/正式收据）。先导正式CPU预检同时运行这项回归，测试源码也归档并记哈希。该验证只检验发布失败语义，不是实际模型训练验收。

CPU先导完整预检集合已实际通过7项（3个新离线/真实preset检查、3个既有FG检查、1个发布失败回归）。报告汇总进程已用`python tools/fg_single3_reports.py --wait --pipeline-log artifacts/fg_single3_v1/cpu_pipeline_run_20261011_02.log`启动：等待所有CPU Gate后自动更新对应docs；流水线失败则拒绝汇总为完成。后台生成/审计/整理报告均已安排；最终仍须读取实际结果并自审。

### O完整Gate（实际完成）

命令：`python tools/fg_single3.py oracle --env-receipt artifacts/fg_single3_v1/env/run_20261011_04 --reward-receipt artifacts/fg_single3_v1/reward/run_20261011_02 --output artifacts/fg_single3_v1/oracle/run_20261011_01 --workers 4`。签名收据content hash：`f2a8497913568d9fedec1c2ad568facc32905a948a29d65ca3b58309b3e8e678`。

488142个TRAIN候选完成索引及精确优化；3439个固定seed分层/边界候选另行完整原Env参考重放；所有四候选72条最终H333账户每笔交易、每步反馈和净现金增量均与计划一致，18个合约均为exact_proven。worker实际累计峰值RSS上界924872704字节；实际保留路径所需权益包络上界104358 CNY，小于初始300000。抽样不是全候选完整原Env穷尽重放；γ1只是精确优化诊断，未声明全部γ1账户另行Driver重放。逐合约证书、耗时与回报表见`FG_Single3_Oracle_Validation.md`。

## CPU并行度调整与同身份恢复

按用户“直接当作空闲机器用”的要求，将D及后续H从12个worker调整为20个。原流水线01的FAIL是主动停止一个已核实属于本任务的worker后ProcessPool退出的真实记录，保留原日志，不解释为行情对账失败。原12个worker配置保存在 `artifacts/fg_single3_v1/replay/run_20261011_01/inputs.initial_workers12.json`；新 `inputs.json` 的identity、frozen_inputs和source_snapshots逐值相同。

恢复命令在原流水线参数上使用 `--workers 20 --resume-dataset`，日志新建为02。已完成的R/O 82份账户和C2自然策略80份账户共162份，经同身份恢复检查后复用；12份被中断的局部产物保留，由新的attempt目录重新执行。原账户、撮合、反馈、Oracle及训练行情源码均未修改。

实际资源证据：`artifacts/fg_single3_v1/resources/cpu_workers20_verified.json`，20个worker、CPU利用率90.1%、worker总RSS 7703437312字节、最大worker RSS 415178752字节、可用RAM 17236787200字节、identity_unchanged=true。该值为一次实测，不宣称整个运行持续达到此利用率。D/H仍遵守总RSS预算16GiB和每worker原BLAS线程限制。

已完整读取当前根目录 `AGENTS.md`：CPU为12物理核/24逻辑线程，机器按空闲专用计算使用，存储仅使用 `/dev/sdd`、`/dev/sda`。其通用GPU使用建议不改变本任务明确的CPU先验收约束；未启动GPU。未使用NVMe或机械盘迁移产物。

### D生成完成时记录（当时覆盖Gate待验收）

流水线02的 `replay-generate --workers 20 --resume` 实际returncode=0，耗时3093.0419569秒。`artifacts/fg_single3_v1/replay/run_20261011_01/manifest.json` 的stage为D_GENERATED，content hash为 `2cb543e5167a4c828fb05b67cdca9fda3c0cabdc7961fc28fc7ad8d433aba123`。完整378份新协议账户、29个Replay视图；C0/C1/C3各5策略×18合约，C2另含18条独立P0。实际D worker累计峰值RSS最大707710976字节，低于20 worker时单worker预算858993459字节。

生成通过不等于D完整验收；生成结束后流水线自动启动全语料统计及每池1000个原生Uniform批次。当时D_AUDIT收据尚未发布，最终结果见下节。

### D完整覆盖Gate（实际完成）

流水线02的 `replay-audit` 实际returncode=0，耗时1205.34595275秒。完整D_AUDIT收据content hash：`a1cbec5f456a3b3629d36e55f0aa20ebd005053923b29e0630741eb8bdb9c0b9`，绑定dataset文件SHA256 `c9e325b17be7bef1c83dff4c562f22bf61bd0d2ad01beb827f40c95fa136a17c`。

实际378账户、10261209行、10250982次推进、488136个唯一推进有效分钟；18个合约初始观察不算推进行，unvisited_rows=0。9849次真实账户恢复；29池各1000个B16×65批次，首batch全部复现，0不连续序列、0非法ADD、0拒单、0六手决策。总审计进程累计峰值RSS为13611241472字节（约12.68GiB），低于16GiB预算；JAX实际平台仅cpu。阶段必需4项测试实际通过，既有公共覆盖回归此前也已验证。

四候选自然策略未折扣总分均负，P0为0，不能据此宣称模型已出现All-WAIT；模型尚未训练。C2自然池实际train64 TP均值9.408/batch，自然+P4为14.519/batch，两个池零TP batch比例均0；后验曝光变化不等于模型收益。真实TRAIN没有SL pending观察或终局未平仓，相关边界仍只有构造Env测试证据。完整表已写入 `FG_Single3_Replay_R0_R2_Coverage.md`、`FG_Single3_Reward_Research.md`；当前CPU汇总保留H和模型先导准备未完成状态。

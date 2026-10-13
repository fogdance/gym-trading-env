# S3-V2 DAgger 实施计划与进度

最新P0优先级和实测交付已完成，参见
`FG_S3_V2_P0_GPU_10K_Training_20261013.md`：真实GPU恢复、10K/400更新、吞吐和冻结长期入口。
自主DEV仍ALL_WAIT；正式1M待授权，A2→C2迁移/R1/B2后置。以下保留原开发计划供追溯。

日期：2026-10-13。执行依据是 D0 后执行修订，原 1M 详细方案仅保留不冲突的技术约束。
本文是实施计划，不是验收结果；实际命令、指标和限制写入后续实施报告。

## 基线与范围

- 开发双仓：`research/fg-s3-dev/{gym-trading-env,dreamerv3}`。
- 起点 Gym `82072058e257828d965da37999e74341910e7d71`、Dreamer `e872602fde0452467d0b1b589f6b2dd048648701`，tracked clean。
- 开发分支 `feat/fg-s3-v2-dagger`，从已提交 D0 创建；原分支、两个冻结训练 worktree 均保留。
- 来源是已冻结 Online180k，模型更新 21183；SourceAdmission 显式校准至 0.30，新 S3 环境计数从零开始。
- 只读复用 `research/fg-data` 的原 TRAIN/Oracle 输入。新输出在本开发仓库 `artifacts/fg_s3_v2/`，不覆盖来源和 V1。
- 受限诊断和模型测试仅使用 GPU，不回退 CPU；不会停止其他实验，不启动正式 1M，不使用 TEST。

## A. 实时数据闭环

复用 D0 OracleStateQuery、MixedExecutionPolicy、原 Env/Driver/Replay、标签证明校验。
V2 适配层暂存当前未执行的 outgoing row；下一次 Env 返回后完成执行结果，提交原生行和教师记录。
提交/采样共用锁，记录 commit_seq；未完成行不可采样。Native Replay 只保留真实 observation/reward/执行动作。
教师通过 UUID+offset 连接并验证 observation/state/proof，旧 P4 委托原 Catalog。
不认证资金域、强制 WAIT 等继续交互但跳过 BC。D0 CLI 的 CPU/1000 步限额不扩大。

## B. A2 学习器与首个里程碑

共享提取原 WM 前向，保留原默认 Agent 算法与网络 schema；准入校验只允许能重新展开为原 AST 的提取。
A2 使用 WM + 等权认证 Oracle BC，BC 对当前 repfeat stop_gradient，只更新原 pol/*。
不计算 imagination、不更新 RL normalizers/slow Critic；原联合优化器一次调用，冻结参数及对应动量。
无有效 BC 标签时 Actor 不受旧动量/decay 更新。训练后明确同步 policy，再进入下一次采集。
已经排队的 executed 动作及 carry 保留，记录提议时的模型版本。

首轮默认：FG201 TRAIN，原完整来源模型，seed 保留来源，mix/replay seed=202，beta=0.5，BC lambda=1。
1000 次真实 Env 推进，每250步2次更新，总8次；B1×65（64有效帧+1前缀）。
验收必须有多个周期的 WM 和 Actor 非零梯度/参数变化、Critic 不变、更新后 Actor 的后续真实提议。
成立后可在独立配置/目录进行5000步、40次更新诊断，不因效果弱提前宣布机制失败。

## C. 实用恢复与自主评价

复用 FullCheckpoint：Agent/联合优化器、Driver/Env/carry、Replay/sampler、sidecar 已提交边界及 pending native row、
Oracle身份、混合 RNG、Gate、真实 Env 与模型/BC/RL 计数、训练 carry、Consec/Prefetch 队列。
在500步节点保存，再恢复至1000步并继续训练；不重复计数或标签，不重新暖启动。
FG409 DEV 使用现有冻结划分/原 Env，单独模型实例，Actor-only，不加载教师指导或改变训练实例。
评价至少 ADD、成交、TP+5及分钟/有效bar耗时、SL-30、早期逆行10、Reward、未结束持仓。
诊断初期不依据 DEV 扫描参数；TRAIN 提议/混合成绩不称自主泛化。

## D. C2 与最少回归

C2 复用 A2 管线；Gate CLOSED=WM+BC，READY 才启用原 RL 与其 Critic/normalizers/slow target 更新。
仍是0.30、Single3连续20窗及原支持量/Recall/退化规则，不用小试验强开生产 Gate。
测试：提交边界/标签和carry时序；WM/BC参数所有权及空标签动量冻结；共享提取与准入篡改拒绝；
C2 CLOSED/受控READY、lambda=0、一次联合更新；一次真实 FullCheckpoint 短续训；Actor-only 评价隔离。
只重跑受影响的旧接口回归，受控夹具和受控READY不称真实行情/生产Gate通过。

## 入口、交付及进度

统一 V2 入口：prepare/start/resume/evaluate/report；YAML 集中在 `configs/experiments/fg_s3_v2/`，官方配置解析。
R1/B2、多合约和提前优化后置。Q仍为null，不实现伪价值权重。
按 A2、恢复/评价、C2 功能提交双仓 SHA，报告真实命令/收据和PASS/FAIL/NOT_RUN，最后整体Review。

当前：已完成并提交 GPU 学习器、实时提交/采样、独立评价和实用 FullCheckpoint。
真实 Online180k 来源模型完成 A2 1000步/8更新（500保存后恢复），另一个独立 A2 接管诊断
完成1000步/40更新，10个周期均有WM及Actor参数变化；C2 CLOSED 完成500步/4更新。
两个独立 Actor-only DEV 各1000步：前者9次ADD、Reward负；后者All-WAIT，研究效果未达成。
真实Checkpoint验收检查来源继承、Critic/慢目标/Normalizer/对应优化器状态冻结及恢复行身份。
具体数值、命令、失败及NOT_RUN写入 `FG_S3_V2_Live_Learning_Implementation_20261013.md`。
正式1M入口/预算及长期资源准备仍未完成，R1/B2按最新修订后置；不把当前短诊断标成完整1M验收。

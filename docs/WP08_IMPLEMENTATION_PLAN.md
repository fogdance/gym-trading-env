# WP08 实施计划与证据边界

日期：2026-10-19。按用户确认的方案实施；基线完整身份见 `WP08_BASELINE.json`。

顺序：08.A → B → C → D → G → E → F → H。每阶段先定向 CPU 测试、审阅 diff，再提交；按实际改动仓库记录 SHA，不制造空提交。

| 阶段 | 交付 |
| --- | --- |
| A | 纯 ExamPlan、独立验证白名单来源、E/MACD V3 Profile、TRAIN/DEV/TEST 隔离 |
| B | 原 F3/F4 来源验证及完整 Checkpoint 只读加载、唯一 ActorPolicy、冻结证明 |
| C | 原 Env/Driver/Observed/Trace 上的持续账户 ExamSession，超过512分钟 |
| D | 真实成交、TP速度、MAE/逆行/SL、费用、Reward、权益与账户对账 |
| G | 独立评价状态、原子不可变 chunks、严格 latest、全新进程精确恢复 |
| E | 同源独立账户基线和 F3/F4 比较，预注册随机种子 |
| F | 工程/DEV/入场/质量/泛化证据分级，全WAIT及支持量不足明确判定 |
| H | 原统一 CLI、45项行为映射、共享回归、反向Review、Roadmap与成对集成 |

复用原生金融、Quality TP Reward、E/partial5m MACD、ActorPolicy、Driver、Trace和原来源核验。不得创建平行金融或训练系统，不放宽 OnlineCollector 的 TRAIN 限制。模型生产身份和考试执行身份分别记录；评价不是训练 resume。

每合约一个账户及 RSSM carry，分块不 reset。默认输出块256个有效市场分钟；至少1024分钟合成验收，并比较333分钟中断后全新进程续跑。保存原 policy 动作计数作为评价随机状态，冻结训练参数、slots、normalizer和训练计数。严格 latest，不静默回退。原训练 Run 只读。

测试使用现有 Dreamer Python/CPU，独立 case 合理并行、每进程原生线程1。每阶段保留原始命令、退出码、逐 case 及失败证据。最终执行 AGENTS 强制共享 Guard、直接受影响恢复/MACD及 WP06/WP07 回归。仅宣称实际执行的选定集合通过。

源及模型真实可用才运行 DEV；否则明确 NOT_RUN，不以合成或 TRAIN 冒充。TEST 默认拒绝，不更新真实 TRAIN 参数、不启动 GPU/长训练、不改金融规则或开放 Gate。所有结论进入 `WP08_IMPLEMENTATION_REVIEW.md`。大文件留在 artifacts，docs 留报告、索引与哈希。

# S3 统一市场分钟实施方案

依据 `FG_S3_All_Versions_Market_Minute_Unification_Codex_Task_20261013.md`；原任务书位于 research/gym-trading-env/docs。本方案不改变算法、网络、成交和资金规则。

## 基线和隔离

新工作区：`research/fg-s3-clock-dev/{gym-trading-env,dreamerv3}`，双仓分支 `feat/fg-s3-market-minute`。实际基线 Gym `240d1b139d5afaf8a4c805b8430bac3be18a9f98`、Dreamer `a561b09f1c10eaa73aa2834c46a2d94e6e354a36`，创建前两者 clean。相比任务书 gy9/d39，保留已验证的 Tutor E2 有限课程权重。原仓库及冻结 worktree 不修改。

## 唯一协议

`S3_MARKET_MINUTE`：`valid_fill_clock=True` 的完整1分钟bar计1，包括零成交量；零时长metadata计0。真实Gap保持拒绝/终止。实际成交bar为第1根，退出bar计入。时间戳仅用于排序、交易时段及成交合法性。

共享前缀计数供 Reward、持仓年龄、60根行情及分钟滚动窗口、Oracle、折扣和评价使用。保留 Reward 曲线与系数、position_state 9维和归一化、market_seq 60×18、神经网络结构。新 schema/observation identity/快照不能静默读取旧规则。

## 数据和训练

全部18个TRAIN合约重新生成新版Env配置/收据、行情特征、C0–C3候选与exact证明、自然/P4真实轨迹、Replay、可信标签、Catalog、S3-B价值差标签及相关BC缓存。原始行情、日历、条款只读；历史派生产物保留，不复用为新规则真值。新输出专用目录，不覆盖任何旧结果。

DP保持gamma=332/333、区间证明、exact fallback、WAIT tie及资金域。指数使用有效bar距离，首个执行bar的即时奖励指数为0。Replay保留物理时序与实际动作；训练辅助market_advance与该行观测Reward对齐，不进入Actor Observation。continuation/replay return使用gamma**market_advance，Imagination保留原数学并避免重复折扣，lambda不变。

共享SourceAdmission增加参数warm_start；首批验收来源为已审计Online180k。只复制网络参数，优化器、Normalizer、Gate、Replay、教师和运行计数初始化为新Run。来源进度仅记录。新Run FullCheckpoint恢复模型/优化器/Env计数/Replay/Sidecar/随机状态/累计计数；跨协议不能当resume。

## 最小验收和交付

受控CPU真值：同bar/相邻bar TP，5/6/15/16/60/61，午休/隔夜，metadata/零量bar，逆行窗口，chunk/持仓恢复，Gap/拒单/未平仓，稀疏和完整Oracle及折扣一致。新入口拒绝旧教师、跨协议Checkpoint。

真实18合约数据重建；FG201抽查跨休市交易。真实GPU A2运行1000环境步：500保存、独立进程恢复后再500，证明多轮collect/train及Actor/WM更新。Tutor/CD短训练探针，Critic/RL冻结。新环境Actor-only短评价不使用Oracle接管。其他S3入口按实际完成的配置/准入/数值验证报告，不冒充真实训练验收。

最终提交两仓SHA、协议/迁移简表、新Oracle/教师身份、真实命令和收据、PASS/FAIL/NOT_RUN及简短报告。完成后停止等待Review；不启动正式1M、不开发B2、不扩展C2研究。

## 2026-10-13 执行进度与后续隔离

新版数据实际输出位于 `/data/logdir/fg_s3_market_minute/20261013_v2/`。E、18合约因果输入检查和R真实轨迹检查已通过；18合约O正在执行。此基线提交用于让独立B2 worktree继承公共时钟，不代表全部数据或GPU训练验收通过。

用户随后授权在独立双 worktree 开发B2；公共改造目录与B2目录分离。Tutor冻结训练目录仍独立使用不含B2网络的双仓版本；`/tmp/done_cd` 必须等待Tutor所需数据、真实训练及恢复验证全部通过后才能创建，不能以B2通过代替，也不能因数据生成完成单独触发。

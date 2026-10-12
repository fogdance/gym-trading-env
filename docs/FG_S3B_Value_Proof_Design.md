# S3-B 价值证明设计与开发进度

基准：Gym `97551fad6f512a571bd3028777cff18c01c5ab91`；Dreamer
`8952315be94ed543af1b2bcb73e211cba5077fbc`。两个仓库从各自当前分支创建
`feat/fg-s3b-certified-bc`。S3-A 已完成，本次只新增 S3-B 文件。

## 观察与证明范围

源码 `fg_single3_oracle.py::funds_certificate` 明确声明
`arbitrary_losing_paths_funds_irrelevant=False`。原证明首先删除非正奖励区间，
再证明 TP-only 最优路径可执行；不能直接用于任意反事实亏损路径。
`CandidateIndex.candidate` 使用 fresh account；实际 P4 的现金必须从原生轨迹
已完成交易的 Decimal 净现金变化重建，并与 `account_state.json` 对账。

S3-A 当前行是 outgoing action，本行反馈来自上一步；资格使用 `lots_after`。
直接复用 S3-A index 的顺序、stepid、mask、label；行政重复行不重新加入。

## 新资金引理

常量成本且符合 Single3 原规则时，设扩张后的 TRAIN 报价范围为 `[lo,hi]`，
`Lq=3*M*(hi-lo)+3*(open_fee+max(close_fee,close_today_fee))`，
`B=Lq+3*M*hi*margin_rate`。对于实际合法空仓现金 `C>B`：

1. 一次强制交易的入场费用、最坏浮亏和保证金需求受 B 控制，资金不会改变
   原候选的执行结果。价格、方向、流动性、截止时间与现金无关。
2. 完全平仓后用实际 `C'=C+net_trade_pnl`；只有另外证明 `C'>B` 才能将
   TP-only relaxed 最优续行当作可行路径。后续 TP 每笔净现金严格增加。
3. `C'<=B` 时，真实分叉已经验证的局部收益加一直 WAIT 是下界；局部收益加
   relaxed suffix 是上界。未验证强制分支则没有可认证的 QADD。

这是一条新的、只针对一次强制交易及其 TP-only 续行的条件引理；不宣称任意
连续亏损账户资金无关。低现金合成反例必须拒绝 EXACT_Q。

## 时间与数值

使用已有冻结候选，不重新搜索或替换教师策略。反向 DP 导出所有 suffix 的
保守区间和可行 witness，单步反馈是原 Env 的 float32，gamma 为 Fraction(332,333)。
局部指数 `step-t-1`，续行指数 `u-t`；1024 reset 不消耗市场推进步。
向外舍入区间相交时只允许有预算的 Fraction 比较；预算耗尽保持区间和
UNCERTIFIED，不用 epsilon 裁决。默认每合约 4096 次、每次 8192 个不同反馈事件。

只有证明真实 suffix 达到 relaxed 上界才标 EXACT_Q。未完结/右删失保持
CENSORED。无成交只有在 ADD 与 WAIT 的金融、奖励、时钟状态等价时才是 TIE_EXACT；
行动统计和 reject_reason 等诊断字段不进入价值状态。非标准终局退出没有虚构续行。

`D=[Alo-Whi,Ahi-Wlo]`。严格正/负端点产生 ADD_BETTER/WAIT_BETTER；
与教师冲突直接拒绝发布合约。区间重叠、严格同值、删失是不同状态。

## 数据与训练接口

同一个完整 S3-A index 保留所有行，不可信行权重为 0。TRAIN 已证正 margin 的
全局 P90 为尺度，权重 min(3,margin/P90)，不做类别/合约再均衡。
初始工程发布门槛为总 ESS>=2560、ADD/WAIT 各 ESS>=32；可显式配置，不自动放宽。
单合约贡献超过 50% 标 WEIGHT_CONCENTRATION。FULL 全行等权，MATCHED 与
CERTIFIED 使用逐 batch 相同的已证样本，只改变权重。

冻结特征继续由 S3-A 读取；证明、退出、Q、现金只用于离线权重，不能成为模型输入。
新 Actor 适配层复用原 pol 和 S3-A 已有 NLL 内核，独立优化器与 S3-B checkpoint
协议。全零 batch 跳过。非 Actor/原 opt/Norm/Gate 与推理同步逐项验证。

## 进度

| Gate | 状态 | 验收证据 |
|---|---|---|
| B0 | 完成；生产全局哈希 DRIFT 保留 | B0_baseline / B4_protection_drift；开发原文件不变 |
| B1 | 完成 | 全账户穷举、一次强制资金引理、55 次实际分叉 |
| B2 | 完成 | 完整 402096 行、396542 正证书、全局 P90 与分层 ESS |
| B3 | 完成 CPU 验收 | 原 Actor 5 键更新/203 键冻结、恢复与推理同步 |
| B4 | 完成代码与 CPU 交付 | 16 项新测试、全量 TRAIN、三模式接口与实施报告 |
| B5 | NOT_RUN | 本轮未授权正式 GPU/DEV，TEST 不访问 |

详细实际结果、命令、收据与未来运行条件见
`FG_S3B_Implementation_and_Verification.md`。S3-A 按已完成依赖复用；
Replay 正在训练不构成本次代码开发阻塞。进度估计：100%（本轮开发范围）。

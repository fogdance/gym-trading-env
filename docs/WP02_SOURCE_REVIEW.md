# WP02 源码与反向 Review

本文件记录源码证据与限制；真实样本和最终 G1/G2/G3 结论以 `WP02_IMPLEMENTATION_REVIEW.md` 为准。

## 复用路径

`source_inputs` → 原 `read_fg_source` / `ETrainAdmission` → 原 `FixedDirectionEntryEnv` → 原 `embodied.Driver` / `Replay` → 原 `reconcile_trace`。新增的 Corpus CLI 只编排行情认证、有限采集和只读核验，没有 Agent 训练、Prepared 或 Checkpoint 生产入口。

MACD 状态机位于 Gym `utils/macd_5m_causal.py`；日历与可见性使用既有 `MarketData`。原生 Env `_get_obs()` 同时产生三个 E 数组及 HOST provenance。Replay 保存真实字节，Dreamer 原 `Agent` 调用新版 E Encoder。没有平行 Env、撮合器、Replay 或训练器。

L1 使用 `NativeQualityTeacher` 的独立原生 Env fork。开仓、持有、退出、费用、资金变化和 Reward 均由原 Env 执行。每次查询绑定完整账户快照；fork 独立市场和可变指标缓存，查询后核对原账户没有变化。L2 使用既有 `CandidateIndex` 和 `OracleStateQuery` 结构，按原生 float32 Quality TP feedback 重新建立候选和完整后缀；它没有读取旧 C1 Oracle 或旧模型参数。

## 金融与 Gate 保护证据

`artifacts/fg_s3_v2/e_quality_wp02/evidence/source_review.json` 与 WP01 双仓固定完整 SHA 对比：

- `FixedDirectionEntryEnv` 仅 `__init__`、`_get_obs`、`export_state`、`import_state`、`_prefix_hash` 有修改；其余 **25 个方法 AST 相同**。这是本任务明确要求金融语义不变的定向审计，不代替行为重播。
- **13 份受保护源码逐字节相同**，包括原 Reward、Quality Score、市场特征、旧 Oracle/行为策略、统一入口、composition Gate、SourceAdmission、FullCheckpoint、Driver、Replay 与训练运行入口。
- E cache 改为独立深拷贝，因为 V2 的 MACD 查询状态可变。未通过共享可变缓存获得加速。
- 新 Catalog 仅校验新的 Corpus envelope；旧 `validate_label` / `attach_quality` 仍拒绝新 envelope。它不授予 Q、Prediction 或 Decision 权限。
- WP01 的显式 draft / formal 行为沿用：草案只能产生非认证诊断，正式预检始终非零，公开 E+Quality 训练入口继续阻断。

## 反向审查问题

| 问题 | 代码结论与证据 |
| --- | --- |
| 是否把 MACD 只放在 Sidecar？ | 否。原生 ObsSpace、实际 Replay chunk、真实 Agent Encoder 都包含 `[4] float32`；CPU 测试检查连续 token 和 RSSM posterior logits 的变化。 |
| valid=0 是否仍通过 bias 泄漏信号？ | 数值输入和投影输出各遮蔽一次；真实 Encoder/Agent 的污染值负例验证无影响。 |
| partial 是否重复推进 EMA？ | 第1–4根从上次提交状态临算，第5根提交一次；重复查询返回私有缓存的副本，冷恢复重建并核对相同快照。 |
| 是否越过未来到达时间？ | 对不可见/晚到分钟拒绝；未来 OHLC 与 arrival 改动不影响早先快照。 |
| 如何处理跨午夜/休市？ | 按冻结 FG 345分钟日历，夜盘与下个官方交易日之间跨午夜、周末和休市不补bar、不推进 EMA。当前冻结 FG 特征不允许虚构的午夜连续交易会话；此输入明确拒绝，不为测试扩张金融时钟。 |
| 开盘零时长元数据是否改写锚价？ | 不提交 EMA，保持最近可见完整1m Close；补充实际 MarketData 回归。 |
| 是否把早期逆势10点当 SL30？ | L1 分开保存真实退出种类、MAE、首次逆势及窗口有效位；Quality Score 使用原函数，不加入 MAE 惩罚。 |
| underwater 是否把已退出分钟计为持仓？ | 使用仍持有时的可执行收盘报价，排除实际结算退出分钟；OHLC MAE 独立保存。 |
| 为什么严格老师不是任意账户 Oracle？ | 原根资金下界仍要求至少300000；强制 ADD 分支另证明结算后资金超过后缀报价上界。域外、持仓、删失、未知边界均 null/invalid，不标主动 WAIT。 |
| 是否只证明 schedule 生成成功？ | 保留候选随机/风险分层慢速原生校验、全部选定计划的完整原生账户重播、每个 reward feedback 与成交事件核对；小型独立穷举再核对两动作数值。非声称所有真实候选都逐条穷举。 |
| 两日截点是否伪装成终止？ | 本次末两日结束于认证的 main qualification exit；不是交割到期，也不是任意步数。绑定真实边界及来源，未知边界拒绝。 |
| 多账户样本是否夸大样本量？ | 唯一市场 cursor/官方日、不同账户决策状态、优化器曝光分别统计；反事实 L1 不算实际成交。 |
| 哈希是否等同认证？ | 否。原项目 TRAIN 分区和原生 Env 收据须命中审阅过的信任锚，再核对全部选定市场字节、官方日、金融配置和真实执行。自签假证书负例不能创建来源权限。 |
| 内层 manifest 是否能指向 TEST？ | 新 E Replay 在任何文件哈希或 np.load 前检查 files 和 chunks；外层和内层都拒绝穿越、绝对路径及 symlink。 |
| 生成中源码变化能否继续签发？ | 不能。`real_small_02` 真实因冻结源码变化退出2；保留原始 NO_GO 输出，没有最终 Corpus 收据。 |
| 是否动了旧 Run、旧1M权重或训练 Gate？ | 新产物仅写隔离 worktree artifacts；旧资产只读。新训练、GPU、全18合约、TEST行情、W&B均不执行。 |

## 测试合同迁移

先前 broad run 的唯一失败来自把固定旧 producer 与不断变化的 HEAD 默认视为可迁移。修正测试固定历史来源 `e7df18931bbb384282d602de8009208ca33d1de1`、历史目标 `6939de5b43200e6f19311804ebf1739b01f12ba5`，实际读取两端 Git 对象并先验证正例；负例修改同一份已通过的历史目标。另增当前 V2 不继承旧迁移许可的拒绝测试。没有修改生产 whitelist、增加兼容别名或放松 SourceAdmission。

## 性能与未做项

保留真实只读剖析：完整历史 export_state 约0.53秒，单个 L1 查询约3.20秒。用户现已恢复优化并授权停止real_small_03，所有旧产物保留。前缀JSON缓存保留原SHA256字节且逐次检测行情变动；恢复使用JSON独立复制。真实三状态L1均值1.865→0.866秒，完整标签/账户一致。有界教师进程池按PSS、可用内存和CPU分配；队列只传历史增量，worker重建完整历史再经原Env恢复与来源校验。生成和独立重播均保持顺序核对、全部Future完成后才允许PASS。没有裁剪历史或去掉完整哈希/重播。该成本不能被解释为18合约吞吐认证。全量生产、GPU训练、研究效果、WP03训练Join、WP04–WP07课程训练均 NOT_RUN。

## 最终冻结核对

当前239份运行源码与真实生成、独立verify及最终312项测试时逐字节一致。`final_verified_source.json`保留命令退出码与执行源码hash。六条真实轨迹各691条记录/标签及1次原Env恢复通过，P4全部553个合法决策均在证明域；其余轨迹有93个合法空仓状态因原根资金域不满足而L2无效，认证NO-GO，没有用主动WAIT补齐。用户要求保持当前性能实现，未继续扩展并发调度或算法。

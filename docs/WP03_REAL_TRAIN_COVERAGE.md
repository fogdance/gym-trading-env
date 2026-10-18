# WP03 实际有限 TRAIN 覆盖

**REAL_TRAIN_DATA_GO；CURRICULUM_SUPPORT=LIMITED；FORMAL_TRAIN_GO=NO_GO。**
本轮重新执行原生verify/G3、三方Join、10个Reader视图和真实CPU Agent Loss，未以WP02报告替代。

输入：`/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp02/real_small_04`。FG201主力资格末两官方日2021-12-08/09；主力日历仍采用用户确认的完整市场来源，未新增交易所认证。
6策略账户，原发布84文件共168,506,837字节；原文件全部保持不变。
市场分母690 cursor、689真实推进；账户分母4134已执行决策；优化器曝光0。

| policy | 物理行 | 原L1 Outcome | 投影L1 Outcome | 投影Score | 原L2 valid | 投影strict ADD / WAIT | 资金域外invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 691 | 689 | 687 | 634 | 689 | 75 / 612 | 0 |
| P1 | 691 | 21 | 20 | 19 | 8 | 1 / 6 | 13 |
| P2_low | 691 | 89 | 88 | 85 | 72 | 15 / 56 | 17 |
| P2_mid | 691 | 39 | 38 | 37 | 24 | 5 / 18 | 15 |
| P3 | 691 | 160 | 159 | 155 | 112 | 21 / 90 | 48 |
| P4 | 691 | 553 | 551 | 500 | 553 | 54 / 497 | 0 |
| 合计 | 4146 | 1551 | 1543 | 1430 | 1458 | 171 / 1279 | 93 |

原L1与投影相差8、原L2与投影相差8：原生reset/prefix/未执行行仍有原始老师查询，但不允许教学loss；不是改写老师或计数不一致。
独立窗口有效元素：event=10801、touch=12344、adverse=6172。
原L1候选1551，其中TP1307、SL131、OTHER113、RIGHT_CENSORED=0。真实成交119笔=103TP+12SL+4MAIN_ROLL；不能将反事实TP当真实成交。
P4实际54笔TP、净收益15228 CNY、费用972 CNY。这些是老师轨迹成绩，不是学生Actor成绩。

原生G3逐账户691行/691老师核验且各1次完整恢复；6账户终态均MAIN_ROLL、cursor31259、无未平仓、equity_history保留62519行。
完整历史长度超出两日Reader窗口，因为来源账户快照保留了完整后缀轨迹；未裁剪以提速或替换账户来源。
证据：[native_verification](receipts/wp03/native_verification.json)、[完整账户历史](receipts/wp03/complete_account_history.json)、[本轮重新统计](receipts/wp03/real_native_coverage.json)。

所有视图B=1、T=64，物理stepid shape=[1,65,20]；每视图4次取样，再保存/恢复同一状态精确比较下一批；实际进行6次读取，其中恢复后的重复读不增加最终计数。
下表曝光为恢复后状态记账的5批×64=320候选位置，**不是优化器曝光**。

| 视图 | 交付候选位置 | 交付unique市场 | 交付unique账户决策 | 实际交付strict / quality pair | policy曝光 |
| --- | ---: | ---: | ---: | ---: | --- |
| WM_NATURAL:NATURAL | 320 | 224 | 311 | 0 / 14 | {"P1": 128, "P2_low": 192} |
| HEADS_VALID:NATURAL | 320 | 213 | 308 | 453 / 99 | {"P0": 64, "P1": 128, "P2_low": 128} |
| HEADS_VALID:STRATIFIED | 320 | 251 | 290 | 565 / 1395 | {"P0": 64, "P1": 64, "P2_low": 192} |
| HEADS_VALID:MIXED | 320 | 198 | 283 | 396 / 536 | {"P1": 64, "P2_low": 192, "P4": 64} |
| ACTOR_STRICT:NATURAL | 320 | 239 | 287 | 1041 / 908 | {"P0": 128, "P1": 64, "P2_low": 128} |
| ACTOR_STRICT:STRATIFIED | 320 | 215 | 231 | 1135 / 1766 | {"P0": 128, "P2_low": 192} |
| ACTOR_STRICT:MIXED | 320 | 223 | 223 | 642 / 1216 | {"P1": 64, "P2_low": 192, "P4": 64} |
| ACTOR_QUALITY_PAIR:NATURAL | 320 | 319 | 319 | 676 / 374 | {"P0": 128, "P2_low": 192} |
| ACTOR_QUALITY_PAIR:STRATIFIED | 320 | 194 | 314 | 591 / 1061 | {"P0": 64, "P1": 64, "P2_low": 192} |
| ACTOR_QUALITY_PAIR:MIXED | 320 | 235 | 298 | 433 / 706 | {"P1": 64, "P2_low": 192, "P4": 64} |

每个分层请求都有真实合法窗口；各题型支持数（TP_FAST/TP_SLOW/SL30/EARLY10_5/15/SAME_DAY/CROSS_DAY/STRICT_ADD/WAIT）见原[Coverage](WP03_COURSE_READER_COVERAGE.json)。
这是窗口数且可重叠，不是独立样本数。全episode可配strict=46743、quality=89729；全局数与实际batch pair分开报告。
实际CPU模型batch strict=295、quality=256；均同Owner/reset。
6个policy另各抽取一个T64批次经过原training_loss_fn，有限loss见[CPU证据](WP03_CPU_FORWARD_LOSS_EVIDENCE.json)。

性能：索引6.155s，实测采样3840行/6.870s=558.95行/s；
索引10,002,432字节；Reader峰值RSS330,432,512字节；含模型进程峰值2,130,935,808字节；
单Reader原chunk缓存峰值13,000,474字节，预算67,108,864字节。RSS包括Python/索引/库；不等于cache大小。
真实模型诊断49.590s；整体Reader命令65.053s，退出0。

覆盖结论LIMITED：单合约两日、strict WAIT占多数、93资金域外invalid、没有真实删失样本，不支持推广训练量或策略泛化结论。
所有请求窗口存在不等于完整研究教材充足。未生产新真实Corpus，未读取TEST，未启动优化器/GPU/正式训练。

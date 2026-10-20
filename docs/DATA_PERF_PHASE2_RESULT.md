# DATA-PERF 第二阶段结果

状态：第二阶段完成，固定真实两交易日端到端、独立 G3/Reader、共享回归及完整 SHA 对照全部通过。
阶段一风险已关闭。阶段二未合并 integration/s3-next。

冻结生产代码 Gym `2dd45283567398e782cbcd0a958e9be873386921`，Dreamer
`b7a693b3c7b0b45eeecdec954818be42d40e67b6`，双仓 clean。
冻结路径、pin 与后续用户手工命令见 [DATA_PERF_PHASE2_COMMANDS.md](DATA_PERF_PHASE2_COMMANDS.md)。
内部所有权及公共恢复合同见 [DATA_PERF_PHASE2_INTERFACE.md](DATA_PERF_PHASE2_INTERFACE.md)。

## 实测速度

固定 CZCE.FG201、2021-12-08 至 2021-12-09，六种原生账户行为，seed 202，
CPU affinity 0–19 / 20 worker 预算 / 原生线程 1。690 个唯一行情位置、689 次真实
行情推进，每行为 691 条记录，合计 4,146 条。初始账户保留真实 WAIT 前缀的
61,141 条 equity_history；严格 Teacher 使用真实完整 TRAIN 后缀。

| 同一完整基准 | 阶段一 e328cf73 | 阶段二 2dd45283 | 加速 |
| --- | ---: | ---: | ---: |
| Generate | 476.513 s | 225.624 s | 2.112× |
| 独立 Native G3 | 397.995 s | 205.976 s | 1.932× |
| Validate→Generate→G3→Reader→Coverage | 907.330 s | 464.315 s | 1.954× |

阶段二完整流程为 7 分 44 秒。阶段一曾将来源检查设为 1024 次一组，阶段二恢复逐查询
检查；这是更强的保护下的实测改善。两次均为 20 总进程预算，六账户和标签分配
`[3,3,2,2,2,2]`，没有改变用户暂缓的调度上限。

原始慢版 b0d5ef9a 的同区间 Generate 为 2940.658 秒，阶段二生成快 13.033 倍。
原始慢版 G3 被按用户指令中止，完整旧链路未完成，不能声称相对它的端到端倍数。

短样本 DEV128/774 条：阶段一生成 208.209 秒，阶段二最终 Driver 候选 172.373 秒。
短样本完整链路 195.165 秒，G3=NOT_RUN；它不冒充正式 DATA 验收。

| 107 个固定真实账户查询 | query CPU | 完整标签 |
| --- | ---: | --- |
| 恢复逐查询来源保护的基线 13b997f | 47.169 s | PASS |
| 仅 L1 private clone 6c16450 | 49.575 s | PASS；单独并未提速 |
| worker/root 共享 26cb4da | 35.223 s | PASS |
| 不可变字节 SHA 1bc608d | 19.345 s | PASS |
| 共享目录遍历 dda7f0f | 12.960 s | PASS |

CPU 指标为同一查询循环的线程 CPU 合计，函数计时有嵌套，不能相加。
进程启动和初始化仍使该微基准 worker wall 约 48–51 秒，不能把 3.64× 查询 CPU
改善冒充端到端 wall 改善。完整链路的 1.954× 是独立实测。

## 正确性和覆盖

新的独立 G3 从原始源重建 Oracle 和精确后缀，不使用生成端的数值缓存作为答案。
六行为各 691/691 标签核验、各一次完整公共账户冷恢复，全部 PASS。
独立 Reader 的 10 种采样视图精确下一批恢复全部通过，实际采样 3,840 行；
`SUPPORTED_FOR_REQUESTED_WINDOWS` 只对应此次 length=64 的请求窗口。

新旧慢版全部 4,146 行、原生数组、实际交易、教师精确有理数及最终账户对照
`PASS_FULL_NATIVE_AND_TEACHER_EQUIVALENCE`。跨 Producer 的路径/代码/物理 Replay
身份按各自身份验收；不把不同 Producer 的签名内容哈希硬说相等。
同一固定数据 profile 的 107 个真实根状态 SHA 和 51 个原生 clone 末状态 SHA 也逐字一致，
query_final_exact_sha_01 另存每个实际 clone 的完整 SHA 对；定向恢复测试独立保护公共 JSON 合同。

L1 有效后验 1,551：TP5 1,307、SL30 131、其他退出 113；快速 TP≤5 分钟 288、
较慢 TP 1,019、快速逆势10≤5分钟 311、跨官方交易日 174。
L2 有效 1,458：严格 ADD 171、WAIT 1,287；margin 正/负/零为 171/1,112/175。
93 条资金落在原证书域外，保持无效，未修改资金合同来补齐覆盖。
这些分母是六账户后验，不是 1,551 个独立市场样本。真正唯一推进分钟为 689。
RIGHT_CENSORED 在本窗口为零，跨合约覆盖及全 18 合约生产 NOT_RUN。

Corpus `dffe36e7b687e1d55ccf28e52caae02f19bb43075274b0b295f27eff94e90302`。
source lock `71d147f52f5407881ddf5f05870e4fe7a86f82b83fcb189ee75490ce7e69f37e`。
course `944bf041a8ff83ab6ebd9f8da3ac5de01cf8ab5a311b5355070387aeccafdd01`。
Teacher `1442611257c33f47f2c62faffa3b1a18a9bbf5aa03cb8f7cef7aa261408ba2ab`。

不可变生成 manifest 仍为 `COLLECTED_UNVERIFIED`，通过新增 Native G3/Reader 收据
关联认证，不改写原 manifest。训练准入 false，优化器更新 0。

最终测试：final_contracts_01 的 113 项定向测试 PASS；shared_final_01 的 106 项测试 PASS
（准确清单见终态收据）（包含共享 Gate/SourceAdmission/action mask、金融边界、Replay、Reader、
MultiCorpus F1/F2/F3）。未执行的 8 项真实优化器测试单列 NOT_RUN，不计入 PASS。

## 收据和剩余成本

原始收据根目录：
`/home/v/Documents/work/research/fg-s3-data-perf-phase2-20261020/artifacts/`。
`long_final_01/` 保存完整五阶段命令、时间、日志、Corpus、独立 G3 和 Reader；
`long_equivalence_01.json` 保存原始慢版全量比较；
`phase2_quality_summary.json` 来自本批真实 sidecar/coverage/Reader。
定向测试、失败迭代与原始 XML 均保留，未用旧 PASS 代替新代码验证。

长历史每查询的完整物化及 pickle 已从受控 worker/clone/Driver/reconcile 消除；
公共可变输入保留慢路径。原生金融 step、行情相等与源身份校验仍真实执行。
短样本还包含显著的准备/进程初始化成本；下一轮在独立工作树实测初始化调用树，
优先检查不同 spawn worker 重复准备同一纯行情的成本。具体子函数占比尚未测得，
不能先声称是某个函数的已确认热点。六账户调度重写继续暂缓。

GPU、真实优化器更新、F1/F2/F3 正式训练、完整 28,264 分钟重复验证、18 合约正式
生成、第二阶段自动合并：NOT_RUN。后续按用户新指令继续测新热点，不热改本冻结版本。

已提交的轻量原始收据与不可变批次/身份锁：
[receipts/data_perf_phase2_20261020/BATCH_MANIFEST.json](receipts/data_perf_phase2_20261020/BATCH_MANIFEST.json)
和 [IDENTITY_LOCK.json](receipts/data_perf_phase2_20261020/IDENTITY_LOCK.json)。

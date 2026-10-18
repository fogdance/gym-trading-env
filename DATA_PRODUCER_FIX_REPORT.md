# DATA Producer 阻断修复报告：2026-10-18

修复已进入独立DATA工作树及共用 `integration/s3-next`，WP04未修改。新正式批次仍NOT_RUN。用户确认的DEV/DATA/FORMAL分级已写入仓库AGENTS.md；开发不再以每次冻结SHA和完整G3作为门槛。

## 观察证据与根因

旧B01 `generate.log` 为 `NO_GO: MACD incomplete previous bucket`，exit2，42.50s/99% CPU，尚未进入多worker阶段。原冻结版本是Gym `63f2f8c` / Dreamer `446d1bc`，旧pin及失败目录保留。

实际原生Env重现首例：FG809 cursor7816，2018-05-15 21:03（官方日05-16）。既有已锁定dated_sessions为 `(21:00,21:01]` 与 `(21:02,23:00]`。旧代码无条件要求跨桶前slot=5，把认证连续区间的尾部partial当成异常。真实导出与上游副本均没有21:02记录；本轮不据此断言源损坏或交易所缺行情，也没有查询数据库确定该分钟原因。

继续诊断发现第二类边界：2018-05-16 10:15行存在，但bob=eob，现有Env合同定义为零时长mark-only；MACD跳过它后停在10:14第4槽，10:31再报上一桶不完整。FG809全主力期间共3个零时长行，均位于认证区间末端。根因是MACD结束边界处理未覆盖现有真实源和Env时间语义，不是主力资格筛选逃逸。

首例完整堆栈、槽位、前后20个实际市场位置及第二例数据见 [诊断证据](docs/data_macd_fix_20261018/evidence/first_failure.json)、[真实片段](docs/data_macd_fix_20261018/evidence/second_failure_46rows.csv)。原日历是用户确认完整来源生成的连续段，本轮没有新增交易所独立认证。

## 实现

- `CausalMACD5m`仍在可见第1～4分钟由上个已提交EMA投影当前partial，第5分钟只提交一次。未满5根不会阻止实时Observation。
- 仅在已观察到认证区间终点或可见零时长结束标记时，允许下一区间丢弃上一未提交尾部；不把partial或元数据Close提交EMA。
- 元数据结束标记必须紧跟上个真实分钟；真实同session缺分钟、重复、乱序、晚到、未证实末尾仍拒绝；不补造分钟、不偷看未来。
- MACD子合同升级V3，Gym/Dreamer课程定义一致，Observation/Course/Corpus内容身份重锁；旧V2 Snapshot和旧Corpus不在V3中冒名恢复。
- 同一Producer增加 `--mode dev --max-market-steps N`；限定真实市场推进，不截断教师TRAIN后缀。DEV产物不可Native准入；DATA按显式官方日发布。`--debug`保存完整堆栈和错误上下文。
- Reader诊断对不存在的自然/分层/单账户窗口报告UNSUPPORTED和LIMITED；实际sample仍拒绝无支持窗口，要求模型批次的诊断仍拒绝缺少Quality Pair。

## 实际验证

1. FG809主力资格2018-04-10夜盘至2018-08-08收盘的28264行全部通过MACD扫描：28261个有效分钟、5642完整桶、21次认证尾部丢弃。两处原报错各前后20个市场位置，截短同历史前缀以及修改未来OHLC/到达时间，输出和完整元数据逐字节相同。见 [真实扫描](docs/data_macd_fix_20261018/evidence/real_prefix_final.json)。
2. 264项选定CPU基础回归PASS；新增Reader缺题型回归与相关26项PASS；最终共用基线19项PASS，另补结束标记精确恢复。共265个不同用例，8项需真实优化器fixture的共享用例NOT_RUN；不是全项目通过。Reader改动后只重跑受影响测试，继承范围见 [验证清单](docs/data_macd_fix_20261018/evidence/VERIFICATION_SUMMARY.json)。
3. 单worker实际TRAIN DEV生成exit0，433.913s；六账户各5行Replay/Teacher标签原生复核PASS；独立Reader `PASS_READER_ONLY / LIMITED`，零优化器更新。小样不支持TP/SL/Pair，见 [覆盖报告](CORPUS_COVERAGE_REPORT.md)。独立Native G3为NOT_RUN。

## 集成和最终身份

已核对 `63f2f8c` 为Producer功能，`5873d22` 为计划/收据文档；后者父提交为前者。Gym共同祖先为 `6939de5b43200e6f19311804ebf1739b01f12ba5`，Dreamer为 `51ed1d094c92fc52a910acda3e2a3421bc71c9e5`。依赖完整WP01–03提交链，采用快进，未孤立摘取CLI，也未修改WP04。

最终运行Gym `b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46`，Dreamer `fd144113ce706e2d8f1ce8e7828ece0799c8bf7a`；[集成收据](docs/data_macd_fix_20261018/evidence/INTEGRATION_RESULT.json)、[最终身份锁](docs/data_macd_fix_20261018/locks/IDENTITY_LOCK.json)。真实DEV生成在修改过程中进行，manifest如实保留当时HEAD和dirty；后续Reader修复只改诊断代码，不将DEV重签成最终认证教材。

最终四批生产目录、命令、manifest与source lock见 [生产计划](TRAIN_CORPUS_PRODUCTION_PLAN.md)，路线图为 [Master Roadmap V3](docs/FG_S3_E_Quality_Master_Roadmap_20261018_V3.md)。正式生产由用户执行；20核运行验证、GPU、TEST价格、真实优化器、正式训练、全18合约生产均NOT_RUN。

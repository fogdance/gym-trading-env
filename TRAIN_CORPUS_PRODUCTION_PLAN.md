# TRAIN Corpus 生产计划：MACD V3 修复版，2026-10-18

当前代码修复和有限 DEV 小样已完成；**新 B01–B04 正式生成、逐批 Native G3、整批 Reader 与质量覆盖均 NOT_RUN，由用户执行**。旧 `/data/s3_corpus/early_train_20261018_cpu20_v1` 的 B01 是失败尝试，保留原目录与 pin，禁止覆盖重试。

## 最终运行版本

| 仓库 | 运行源码提交 | 冻结检查 |
| --- | --- | --- |
| Gym | `b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46` | clean，已快进共用 integration/s3-next |
| Dreamer | `fd144113ce706e2d8f1ce8e7828ece0799c8bf7a` | clean，已快进共用 integration/s3-next |

开发位置：`/home/v/Documents/work/research/fg-s3-data-integration-20261018/`，两仓分支 `integration/s3-data`。WP04 worktree 未修改。最终文档提交可在运行提交之后；生产脚本始终冻结上表精确运行提交。

最终 [Producer pin](docs/data_macd_fix_20261018/locks/PRODUCER_PIN.json) 文件 SHA256：`7310b2a2cdcfba7fdc8697bc46fff0d6ca39e4f7932b90710eff7de1a7179515`。242 份运行源码的相对路径/哈希、Observation 合同与模板哈希见 [完整身份锁](docs/data_macd_fix_20261018/locks/IDENTITY_LOCK.json)。[冻结源码检查](docs/data_macd_fix_20261018/evidence/FROZEN_PRODUCER_AUDIT.json) 只证明代码身份，不是 G3。

## 用户执行命令

```bash
cd /home/v/Documents/work/research/fg-s3-data-integration-20261018/gym-trading-env
bash docs/data_macd_fix_20261018/RUN_TRAIN_CORPUS.sh prepare
bash docs/data_macd_fix_20261018/RUN_TRAIN_CORPUS.sh produce B01_FG809
```

新生产根目录固定 `/data/s3_corpus/early_train_20261018_cpu20_v2`。`prepare` 创建冻结兄弟 worktree；`produce` 只执行所选单批：来源校验 → 六行为原生生成及随附复核 → 独立 Native G3 → 独立 Reader → 质量统计 → 不可变 `BATCH_MANIFEST.json`。失败立即停止并保留日志、退出码及已有工件；重试必须使用另一个新版本目录。

阅读 B01 覆盖后再手动选择 `produce B02_FG009`、`produce B03_FG209` 或 `produce B04_FG405`。脚本没有全合约模式，也不会自动启动下一批。上述 prepare/produce 本轮均未执行。

## 已锁定范围

| 批次 | TRAIN 合约 | 官方日 | cursor | 计划市场位置 | 正式生成/G3/Reader |
| --- | --- | --- | --- | ---: | --- |
| B01_FG809 | CZCE.FG809 | 2018-08-06、07、08 | 27230..28263 | 1034 | NOT_RUN |
| B02_FG009 | CZCE.FG009 | 2020-08-07、10、11 | 26565..27599 | 1035 | NOT_RUN |
| B03_FG209 | CZCE.FG209 | 2022-08-08、09、10 | 30915..31949 | 1035 | NOT_RUN |
| B04_FG405 | CZCE.FG405 | 2024-04-02、03、08 | 25515..26429 | 915 | NOT_RUN |

合计为计划 4 合约、12 合约官方日、4019 市场位置；不是已生产覆盖。每批固定 P0/P1/P2_low/P2_mid/P3/P4 六个账户和 seed=202。不可变 [计划 manifest](docs/data_macd_fix_20261018/batches) 与 [source lock](docs/data_macd_fix_20261018/locks) 已更新到 V3；未知的 Corpus、Teacher、Reader 索引身份仍为 null。

四批新 `validate` 均 `SOURCE_VALIDATED_ONLY`；精确命令和 stdout 见 [来源收据](docs/data_macd_fix_20261018/evidence/source_validation)。TRAIN 信任根、白名单和主力资格边界未改。原18来源的历史清点为488160行/1455合约官方日，见保留的 [来源清单](docs/train_corpus_20261018/evidence/SOURCE_INVENTORY.json)，本轮没有重新生成全部18合约。

显式日期限制 StudentReplay 观察窗口；计算读取该合约主力资格内的因果预热和完整 TRAIN 教师后缀，不以观察窗口末端伪造金融平仓。全部实际价格行仍须属于原始 TRAIN。原日历依据为 `user_confirmed_complete_market`，本轮没有增加交易所独立日历认证。

## 算力与开发层级

正式命令使用 `taskset -c 0-19`、`--workers 20`，账户及标签 worker 合计最多20；原生库每进程1线程，GPU 不可见。内存充足时为6个账户+14个标签进程；按实际 MemAvailable/PSS 缩减并发，Swap 不算物理内存。预热与部分建表仍是串行阶段，不承诺全程2000% CPU。**20核占用、吞吐、整批工期验证按用户要求 NOT_RUN。**

产物位于 `/data`（历史设备核查 `/dev/sda1`），开发位于工作盘 `/dev/sdd2`；不使用其他存储。用户生产脚本在开始前记录实际内存与磁盘余量，并拒绝 `/data` 设备变化。CPU和数据范围由本次明确批次命令选择，不构成代理自动生产授权。

| 层级 | 使用方式 |
| --- | --- |
| DEV | 临时目录、合成数据、有限真实 TRAIN、定向测试；失败可改后重试；不要求每次冻结SHA或完整G3；产物 DEV_ONLY/NOT_CERTIFIED |
| DATA | 最终发布版本才冻结代码与来源；每批发布前独立G3和Reader；保留失败结果身份 |
| FORMAL | 独立训练授权、阶段冻结/优化器/Checkpoint/研究验收；DATA成功不解除训练Guard |

上述规则已写入 [AGENTS.md](AGENTS.md)，兄弟 Dreamer 遵循同一规则。金融正确、MACD因果性、Teacher/Observation隔离在三个层级都必须满足。

## 当前实际验证

MACD第1～4根有效分钟持续输出 partial，完整第5根才提交一次EMA；认证区间尾部和零时长结束标记不供新EMA价格。数学子合同为 `FG_S3_E_MACD5M_CAUSAL_V3`；Observation四维形状保持，合同及Course/Corpus内容身份重新锁定，旧V2产物不能冒名准入。

实际单worker DEV：FG809 2018-08-08，4个市场推进/账户，六行为30行Replay，5个独立cursor，24个账户决策，逐行账户/Teacher复核PASS；Reader `PASS_READER_ONLY / LIMITED`。缺少题型明确UNSUPPORTED。详见 [修复报告](DATA_PRODUCER_FIX_REPORT.md) 和 [覆盖报告](CORPUS_COVERAGE_REPORT.md)。

265个不同的选定CPU用例通过（264项基础回归、Reader定向回归及共用基线19项，重复用例不累计）；非全项目通过。8项调用真实优化器fixture的共享用例、GPU、TEST价格和正式训练均NOT_RUN。原始清单及继承范围见 [验证摘要](docs/data_macd_fix_20261018/evidence/VERIFICATION_SUMMARY.json)。

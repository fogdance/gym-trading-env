# 正式 TRAIN 教材 V2：长精确分数运行环境修复

范围已按用户要求从早期四个三日窗口扩展为原始合法 TRAIN 池全部18合约，各自完整主力资格期间。每个合约为一份独立六行为 Corpus；不按末两日/三日截取，也不按未来标签挑日。这里只读校验和编制生产包，实际生成由用户执行。

**生产包发布时：18/18来源校验成功；正式教材已生成0/18。生成、Native G3、独立Reader、质量分布和耗时全部NOT_RUN。旧教材按用户指示删除并排除。** 后续真实结果更新根目录覆盖报告，原计划manifest保持不变。

## V1失败与V2修复

正式V1的TRAIN01在707.32秒后generate exit=2，P4写出`str(Fraction)`时触发CPython默认4300位整数十进制转换限制。源码同时确认Reader会用`Fraction(witness)`解析同一证据，必须同时覆盖写入与读回；不是K线/日期错误。

V2仅在用户启动脚本及其子进程导出`PYTHONINTMAXSTRDIGITS=0`，保留原来的十进制精确分数字符串、折扣、交易规则、Teacher和Observation合同，不修改系统Python设置或冻结源码。每批增加32000步精确分数的parent/spawn/verifier三端运行环境预检，并把环境实测收据及设置纳入最终BATCH_MANIFEST。18份来源校验与242份源码哈希继承未变化的V1证据；新批次仍NOT_RUN。

```bash
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh runtime-check
```

上面只做CPU精确数值预检，不启动教材生成。正式重试由用户执行下述produce命令，输出为新V2目录；V1失败现场不覆盖、不删除。以后直接调用Reader重建Join时同样使用`env PYTHONINTMAXSTRDIGITS=0`，否则大数证据反解析仍会触及Python默认限制。

## 批次定义

编号 TRAIN01–TRAIN18 按原TRAIN池的合约顺序固定；FULL_MAIN 表示该合约全部主力资格TRAIN范围，交易日以官方日而非自然日归属，包含对应夜盘。一个合约内部六个账户完整连续运行，不在每天开头重新建账户。

| 批次 | 合约 | 完整主力期官方日（含首尾） | 官方日数 | 原行情位置/行数 | 生成/G3/Reader/覆盖 |
| --- | --- | --- | ---: | ---: | --- |
| TRAIN01_FG809_FULL_MAIN | CZCE.FG809 | 2018-04-11—2018-08-08 | 83 | 28264 | NOT_RUN |
| TRAIN02_FG901_FULL_MAIN | CZCE.FG901 | 2018-08-09—2018-12-14 | 86 | 29416 | NOT_RUN |
| TRAIN03_FG905_FULL_MAIN | CZCE.FG905 | 2018-12-17—2019-04-16 | 79 | 26762 | NOT_RUN |
| TRAIN04_FG909_FULL_MAIN | CZCE.FG909 | 2019-04-17—2019-08-08 | 78 | 26546 | NOT_RUN |
| TRAIN05_FG001_FULL_MAIN | CZCE.FG001 | 2019-08-09—2019-12-09 | 81 | 27464 | NOT_RUN |
| TRAIN06_FG005_FULL_MAIN | CZCE.FG005 | 2019-12-10—2020-04-01 | 75 | 20505 | NOT_RUN |
| TRAIN07_FG009_FULL_MAIN | CZCE.FG009 | 2020-04-02—2020-08-11 | 88 | 27600 | NOT_RUN |
| TRAIN08_FG101_FULL_MAIN | CZCE.FG101 | 2020-08-12—2020-12-11 | 82 | 28050 | NOT_RUN |
| TRAIN09_FG105_FULL_MAIN | CZCE.FG105 | 2020-12-14—2021-04-07 | 76 | 25620 | NOT_RUN |
| TRAIN10_FG109_FULL_MAIN | CZCE.FG109 | 2021-04-08—2021-07-23 | 73 | 24945 | NOT_RUN |
| TRAIN11_FG201_FULL_MAIN | CZCE.FG201 | 2021-07-26—2021-12-09 | 92 | 31260 | NOT_RUN |
| TRAIN12_FG205_FULL_MAIN | CZCE.FG205 | 2021-12-10—2022-03-23 | 68 | 23220 | NOT_RUN |
| TRAIN13_FG209_FULL_MAIN | CZCE.FG209 | 2022-03-24—2022-08-10 | 94 | 31950 | NOT_RUN |
| TRAIN14_FG301_FULL_MAIN | CZCE.FG301 | 2022-08-11—2022-12-09 | 81 | 27703 | NOT_RUN |
| TRAIN15_FG305_FULL_MAIN | CZCE.FG305 | 2022-12-12—2023-04-10 | 79 | 26895 | NOT_RUN |
| TRAIN16_FG309_FULL_MAIN | CZCE.FG309 | 2023-04-11—2023-08-11 | 84 | 28740 | NOT_RUN |
| TRAIN17_FG401_FULL_MAIN | CZCE.FG401 | 2023-08-14—2023-12-07 | 78 | 26790 | NOT_RUN |
| TRAIN18_FG405_FULL_MAIN | CZCE.FG405 | 2023-12-08—2024-04-08 | 78 | 26430 | NOT_RUN |

合计18合约、1455个合约官方日、488160行原始行情。这里的行数是来源规模，包含原生时钟中不推进市场分钟的行，不能直接称为有效市场分钟；六个账户也不使独立市场规模乘六。每批精确的官方日列表、cursor=0至该来源末行、完整Teacher后缀、来源文件SHA见 [来源锁](locks) 和 [计划manifest](batches)。

## 用户逐批执行

```bash
cd /home/v/Documents/work/research/fg-s3-formal-corpus-plan-20261018/gym-trading-env
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN01_FG809_FULL_MAIN
```

每次命令只选中一个批次，固定20个逻辑CPU预算。首批及后续批次都由用户手动运行，完成后看覆盖报告再选下一批；不能同时复制执行整段。下面列出18个可独立选择的命令，不是自动顺序脚本，也没有all入口：

```bash
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN01_FG809_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN02_FG901_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN03_FG905_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN04_FG909_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN05_FG001_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN06_FG005_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN07_FG009_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN08_FG101_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN09_FG105_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN10_FG109_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN11_FG201_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN12_FG205_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN13_FG209_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN14_FG301_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN15_FG305_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN16_FG309_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN17_FG401_FULL_MAIN
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN18_FG405_FULL_MAIN
```

不用prepare。复用已有冻结的兄弟源码目录 `/data/s3_corpus/early_train_20261018_cpu20_v2/code/` 和该目录同级 `locks/PRODUCER_PIN.json`；这两项保留，仅旧教材及收据被删除。新产物统一写到 `/data/s3_corpus/train_full_main_20261018_cpu20_v2/`，与旧教材隔离。入口脚本在本独立worktree，不依赖合并到s3-next。

## 每批实际流程和发布证据

现有五阶段流程保持：来源validate → `generate --mode data` 六行为原生生成及复核 → 独立Native G3 → `read --without-model --length 64` → 原生质量统计 → 最后写不可变BATCH_MANIFEST。复用WP02/WP03原入口，没有修改Producer、Reader或金融规则。六行为为P0/P1/P2_low/P2_mid/P3/P4，seed=202；主力末端由真实环境退出，保持Teacher与Student Observation隔离。

每阶段保存准确argv、stdout/stderr日志、退出码和GNU time。失败立即停下并保留结果；不得覆盖同一批重试，不得补写PASS。成功批的输出为：

- `batches/<batch>/corpus_manifest.json`：真实Corpus/Course/Teacher身份及文件哈希。
- `receipts/<batch>/native/native_verification.json`：本批独立Native G3。
- `receipts/<batch>/reader/WP03_COURSE_READER_COVERAGE.json` 及Join索引：独立Reader/抽样/恢复和题型支持。
- `receipts/<batch>/quality_coverage.json`、原生`coverage.json/csv`：候选及执行交易分母、TP/SL/快速TP/快速逆势10后SL/高MAE、有效及无效标签、行为分布；生成后再据真实Sidecar补充跨日和全局分布审计。
- `receipts/<batch>/BATCH_MANIFEST.json`：仅在全部阶段完成并校验关联身份后创建，绑定源码pin、source/course/teacher/corpus/index/file身份。

计划manifest的null/NOT_RUN保留发布时真实状态；真实生成后用生产现场BATCH_MANIFEST报告结果，不回填重签计划文件。不能把来源校验当作G3或质量验收，也不能把G3通过当作题型足够。缺少题型、零样本和不支持窗口必须如实保留。18合约全部来源完整也不预先保证所有题型满足要求。

## 资源、冻结版本和未运行项

固定 `taskset -c 0-19`、`--workers 20`、每worker原生库1线程，GPU不可见，Reader不加载模型，无优化器更新。每次仅运行一批，不能并发启动多个20-worker批次。Producer依据真实可用内存调整worker池，不使用Swap作为物理内存预算；本轮不再测20核占用。

运行前由脚本记录内存与/data磁盘余量并确认/data设备仍为/dev/sda1；开发文件在工作盘。全周期生成耗时、G3耗时、总磁盘和全进程树内存峰值均NOT_RUN，不用旧三日样本线性外推为承诺预算。

| 身份 | 精确值 |
| --- | --- |
| Gym运行SHA | `b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46` |
| Dreamer运行SHA | `fd144113ce706e2d8f1ce8e7828ece0799c8bf7a` |
| Producer pin文件SHA256 | `7310b2a2cdcfba7fdc8697bc46fff0d6ca39e4f7932b90710eff7de1a7179515` |
| MACD数学合同 | `FG_S3_E_MACD5M_CAUSAL_V3` |

[精确身份锁](locks/IDENTITY_LOCK.json) 记录242份运行源码、18份新来源锁和计划manifest的哈希。[只读validate原始输出与准确命令](evidence/source_validation) 均为真实执行的SOURCE_VALIDATED_ONLY；未生成新Corpus。日历仍沿用已认证来源的用户确认口径，本轮未新增交易所独立日历认证。

GPU、DEV/TEST价格访问、真实优化器更新、训练准入、正式18批生成/G3/Reader均未执行。本工作树不主动合并到integration/s3-next，不修改WP04或其他正在开发的worktree。

## 旧教材删除

用户明确要求忽略并删除旧教材，已删除cpu20_v1和cpu20_v2各自的 `batches/B01_FG809` 与 `receipts/B01_FG809` 四个目录；无完整备份。原始TRAIN行情、冻结源码和pin保留。[删除命令及不存在核对](evidence/OLD_CORPUS_DELETION.json)。历史审计摘要只作历史记录，不是当前存在或可入教材的Corpus。

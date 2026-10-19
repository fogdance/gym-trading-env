# TRAIN_CORPUS_PRODUCTION_PLAN

> 正式V1的TRAIN01生成失败：Python精确分数证据超过默认4300位转换限制，707.32秒、exit=2；无完整Corpus，G3/Reader未执行。当前使用V2发布包，生产进程树设置PYTHONINTMAXSTRDIGITS=0，冻结源码及金融数学不变。V1现场保留；V2仍待用户执行。

**正式教材范围：原合法TRAIN池全部18合约、每个合约完整主力资格期间；不是此前四个三日窗口。** 每合约一批，六种原生行为连续生成。全部18份来源锁已只读校验成功，共1455个合约官方日、488160行原始行情；实际正式生成0/18，生成/G3/Reader/质量分布均NOT_RUN。

旧教材按用户要求删除并排除，当前正式教材从零统计。旧B01成功结果不再作为当前教材；旧B02–B04三日计划封存，不是当前正式生产命令。现行范围、18批定义和所有逐批命令见 [正式生产包](docs/formal_train_corpus_20261018_v2/README.md)。

## 用户执行首批

```bash
cd /home/v/Documents/work/research/fg-s3-formal-corpus-plan-20261018/gym-trading-env
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN01_FG809_FULL_MAIN
```

首批范围为CZCE.FG809，2018-04-11至2018-08-08全部83个官方交易日，cursor 0..28263；用户运行上面命令才开始真实生成。每次只能指定一个批次，固定20逻辑CPU预算；不自动运行全部18合约。完成后逐批检查质量覆盖，由用户选择下一批。

## 18批完整范围

| 批次 | 合约 | 官方日范围（含首尾） | 官方日数 | 来源行数 | 生成/G3/Reader/覆盖 |
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

[逐批不可变计划manifest](docs/formal_train_corpus_20261018_v2/batches)、[逐批来源锁及完整身份锁](docs/formal_train_corpus_20261018_v2/locks) 含全部官方日、TRAIN信任根、来源文件哈希与真实主力末端。所有Corpus/Teacher/index新结果身份仍为null；不能继承旧教材的认证身份。

## 生产和验收

复用原WP02 Producer及WP03 Admission/Reader：validate → 六行为原生generate及复核 → 独立Native G3 → 独立Reader → 覆盖报告 → 不可变BATCH_MANIFEST。每批必须有自己的G3和Reader，缺题型或采样不支持必须如实报告。P0 DATA-QUALITY仍在推进，不能用18份来源校验声称18批教材生产完成。

生产输出 `/data/s3_corpus/train_full_main_20261018_cpu20_v2`。复用保留的冻结源码 `/data/s3_corpus/early_train_20261018_cpu20_v2/code/`，Gym=`b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46`，Dreamer=`fd144113ce706e2d8f1ce8e7828ece0799c8bf7a`；[Producer pin](docs/formal_train_corpus_20261018_v2/locks/PRODUCER_PIN.json) 文件SHA256=`7310b2a2cdcfba7fdc8697bc46fff0d6ca39e4f7932b90710eff7de1a7179515`。不需要prepare，不重新冻结到当前集成HEAD。

资源为taskset 0–19、workers=20、每worker原生线程1、按可用物理内存约束并行；一次一批。不再做20核占用验证。全主力期真实生成耗时、磁盘增长、G3成本和全进程树内存峰值均NOT_RUN。运行时由脚本记录实际内存/磁盘；仅使用工作盘及/data指定存储。

本轮只改独立worktree的生产文档和发布脚本，不改Producer/Reader金融语义、不改WP04、不合并到s3-next。无GPU、无DEV/TEST价格、无真实优化器更新；DATA教材验收与FORMAL训练准入仍分开。旧产物删除及未运行项见 [覆盖报告](CORPUS_COVERAGE_REPORT.md)。

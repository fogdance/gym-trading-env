# TRAIN Corpus 生产方案：2026-10-18

**新 Corpus 生产、四批 Native G3、四批 Reader 和质量覆盖均为 `NOT_RUN`。正式生成由用户执行。** 本次仅修改新建工作树、核查 TRAIN 来源、运行 CPU 回归并重新读取历史 FG201 教材；没有操作 WP04 工作树，没有读取真实 DEV/TEST 价格，没有使用 GPU 或执行真实优化器更新。

## 已实现的入口

复用 `tools/fg_s3_e_quality_corpus.py`，直接扩展现有 Producer，没有第二套教师或采集器。

| 控制项 | 当前行为 |
| --- | --- |
| `generate --start-day ... --end-day ...` | 明确选择一个 TRAIN 合约的官方交易日闭区间；可以超过两天，也可以选择主力资格末两天以前的日期。两个端点必须存在，禁止静默裁剪或换日。 |
| `--workers 20` | 指定账户与教师进程合计的 CPU 预算。内存充足时为 6 个账户进程 + 14 个教师进程；原生库线程数为 1。生成和独立 G3 都支持。 |
| `--out /data/s3_corpus/...` | 允许独立外部目录。保留新目录、无符号链接、不覆盖来源和正式训练目录的检查。 |
| Reader `source/read --out ...` | 验收收据和 SQLite 索引也可放在 `/data/s3_corpus`，必须新建且位于 Corpus 外。 |
| `source --purpose native-verification --producer-lock ... --producer-lock-sha256 ...` | 对明确锁定的新生产者执行完整原生 G3。仍检查完整 SHA、运行代码无改动以及生成时的逐文件源码身份；不改变 TRAIN 信任根。 |

`generate-small` 保留原有默认末两日调用，旧产物与旧收据不迁移、不重签。新 `generate` 必须显式给出日期；不能默认生成全部合约。

日期区间限制的是 StudentReplay/Trace 中的观察行。L1/L2 使用同一已认证 TRAIN 合约从开始 cursor 到原生主力退出点的完整后缀，`source_lock.json` 分别记录 `end_cursor` 和 `suffix_end_cursor`。**较早的 `--end-day` 不会伪造平仓、清空持仓或把已知 TRAIN 后缀标成金融终点。** 整个价格来源的官方日仍须全部属于 TRAIN；禁止借用其他 fold 的未来。

## 隔离及代码锁

开发工作树：`/home/v/Documents/work/research/fg-s3-e-train-corpus-production-20261018/`，Gym 分支 `feat/fg-s3-train-corpus-cpu-20261018`。不修改 WP01、WP03 或 WP04 原工作树。

| 角色 | 冻结提交 | 取证时 dirty |
| --- | --- | --- |
| 新 Producer / WP03 消费入口，Gym | `63f2f8c35d4f84d9dfd11d2bab8e4b4d2c6d876c` | false |
| 兄弟 Dreamer | `446d1bc99debd2d05c313eafd7d0c414ebd2815c` | false |
| Gym 开发基线 | `7366886f3b40bf74aa4485978ffcc97e39d0d53e` | false |

代码提交与后续文档提交分开。下面的准备命令从**上述运行源码提交**创建冻结 worktree，文档提交不会改变生产者身份。完整 242 个运行源码文件的相对路径/字节哈希见 [IDENTITY_LOCK.json](docs/train_corpus_20261018/locks/IDENTITY_LOCK.json)；开发路径下的实际源码清单见 [EXECUTED_CODE_LOCK.json](docs/train_corpus_20261018/locks/EXECUTED_CODE_LOCK.json)。搬到 `/data` 后的绝对路径源码清单由真正生成时重新写入原生 manifest，目前为 `NOT_RUN`。

最终源码提交的选定 CPU 回归为 **140 PASS / 0 FAIL / 0 ERROR / 0 SKIP**，用时 219.79s，见 [完整命令与两仓状态](docs/train_corpus_20261018/evidence/release_code_63f2f8c.command.json) 和 [实际用例清单](docs/train_corpus_20261018/evidence/VERIFICATION_SUMMARY.json)。源码锁采集时两仓均 clean；最终回归记录中 Gym dirty=true，仅两份未跟踪报告，Dreamer clean，运行源码无变化。此后只调整用户要求的 20 核命令/计划，不运行 20 核复验。

[PRODUCER_PIN.json](docs/train_corpus_20261018/locks/PRODUCER_PIN.json) 的外部固定 SHA256：

```text
dc1be365f64d794ff5c28b88c4378f74561c5b64422a3ee5a7d36d285fdff0d0
```

已执行只读 [冻结生产者审计](docs/train_corpus_20261018/evidence/FROZEN_PRODUCER_AUDIT.json)：两仓 HEAD 匹配、runtime diff=0、无未跟踪运行代码。这只是代码冻结核查，不是新 Corpus G3。旧 WP02 的默认 SHA 白名单不变；新生产者必须使用上述明确 pin，不能用自签 PASS 替代 G3。

## TRAIN 来源与首批范围

只读核查结果见 [SOURCE_INVENTORY.json](docs/train_corpus_20261018/evidence/SOURCE_INVENTORY.json)：18 个 TRAIN 合约，488,160 根分钟线，合计 1,455 个合约官方日；市场/Env/合约配置哈希均与原白名单一致；symbol、行数、日期数、时间有序且无重复；所有行情日都在 TRAIN，held-out 日期交集为 0。此处只读取 fold 日期元数据，没有打开 DEV/TEST 价格文件。四批另执行了真实 `validate`，其日历检查通过；其他合约的原生日历重验为 `NOT_RUN`，不能把清单元数据当作新的交易所日历认证。

原白名单路径：

```text
/home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/train_pool.json
```

其文件 SHA256 为 `4d9aa8faa5c2ad22a5664280675067c5c18e9fb22e3fee7b9af3bf0f15e6281b`，原 TRAIN 内容身份为 `ffc45db584106b7c348b7f3aee95096acbcb560c31a7d67fa340f30d2e10009a`。原 Env 收据 `/data/logdir/fg_s3_market_minute/20261013_v2/env/v1/manifest.json` 的 SHA256 为 `cf8fbf93953142b743f718effe2f377a80eba8a0cddf196a901d5344cf3d19ad`。

TRAIN 官方日范围为 2018-04-11 至 2024-04-08；validation 从 2024-04-09 开始。生产前仍逐行检查选定来源不跨 fold。原日历依据为用户确认的完整市场来源，未新增交易所认证。

| 批次 | 合约 | 明确选择的官方日 | cursor 范围 | 待生成唯一市场位置 | 状态 |
| --- | --- | --- | --- | ---: | --- |
| B01_FG809 | CZCE.FG809 | 2018-08-06、07、08 | 27230..28263 | 1,034 | NOT_RUN |
| B02_FG009 | CZCE.FG009 | 2020-08-07、10、11 | 26565..27599 | 1,035 | NOT_RUN |
| B03_FG209 | CZCE.FG209 | 2022-08-08、09、10 | 30915..31949 | 1,035 | NOT_RUN |
| B04_FG405 | CZCE.FG405 | 2024-04-02、03、08 | 25515..26429 | 915 | NOT_RUN |

合计**计划**覆盖 4 合约、12 合约官方日、4,019 个市场位置；这些不是已生成样本。跨周末、清明假期的跳日来自认证日历，不补造分钟。每批固定 P0/P1/P2_low/P2_mid/P3/P4 六个原生账户，seed=202。按年份跨度选批，不按未来 TP/SL、波动或收益挑行情。首批仍集中于主力资格尾部，不能代表合约中段；中段追加须另列日期和新版本，不自动扩成 18 合约。

生产会读取所选**完整 TRAIN 合约**的因果预热历史及教师后缀；Replay 行只发布选定三天。不是只打开三天价格，也不会在运行时打开另外 14 个合约的价格。

四个不可变计划 manifest 与四份完整 source lock 分别位于 [batches](docs/train_corpus_20261018/batches) 和 [locks](docs/train_corpus_20261018/locks)。`corpus_identity`、`teacher_identity`、`index_sha256` 均为 null，相关执行状态为 `NOT_RUN`；不得在生产前填造这些身份。

## CPU、内存与存储预算

[主机只读证据](docs/train_corpus_20261018/evidence/HOST_RESOURCES.json)，采样时间 2026-10-18 09:38:36 +08:00：Ryzen 9 3900X，12 物理核 / **24 逻辑核，CPU 0–23**；MemAvailable=16,979,034,112 字节；`/data` 是 `/dev/sda1`，可用 413,539,557,376 字节；工作区为 `/dev/sdd2`，可用 35,101,765,632 字节。两块盘均符合允许的设备范围。Swap 不计入物理内存预算。

每次明确批次使用 `taskset -c 0-19`、`--workers 20`，BLAS/OpenMP/MKL/NumExpr 各为 1 线程，GPU 可见设备为空。按用户最终要求最多占用 20 个逻辑核，CPU affinity 固定 0–19；不再按 24 核生产预算执行。保持现有六账户顺序推进，教师查询并行；不能把 24 个 worker 解释为 24 条独立行情。

现有分配器使用真实进程 PSS 的 125%（至少 256 MiB）作为单 worker 预算，加 512 MiB 协调余量，并按 MemAvailable/CPU affinity 限制并发。已有测试确认原分配器可分配 24 个计算进程；当前 20-worker 命令按同一代码分配为 6+14。用户明确要求不再验证 20 核，因此未执行 20-worker 并发或利用率验证；不足时输出 `requested_worker_processes`、`total_worker_processes`、`worker_limit_reason=memory`，不靠 Swap 或绕过内存检查强行启动。**真实 20 核利用率、峰值进程树 PSS、吞吐和工期均为 `NOT_RUN`。** 原生 WAIT 预热、后缀构建和文件审计含串行阶段，不能承诺全程 2000%。

历史 FG201 两日六账户：生成含附带重播 2855.278s，独立 G3 历史收据 1693.454s，产物 168,506,837 字节；当时只有 13 个 worker。它们只是历史成本实例，不作为本次三日/其他合约的线性工期或容量保证。首批完成后用每阶段 `*.time.txt`、`performance.json` 和 G3 预算日志记录真实成本，再决定下一批。

## 用户执行命令

命令文件已做语法、参数和身份锁检查；下面的 `prepare/produce` **本次均未执行**。正常用户权限即可；`/data` 当前由用户 v 所有。

```bash
cd /home/v/Documents/work/research/fg-s3-e-train-corpus-production-20261018/gym-trading-env
bash docs/train_corpus_20261018/RUN_TRAIN_CORPUS.sh prepare
bash docs/train_corpus_20261018/RUN_TRAIN_CORPUS.sh produce B01_FG809
```

`prepare` 只在新的 `/data/s3_corpus/early_train_20261018_cpu20_v1/` 创建冻结兄弟 worktree、pin 和缓存，尚不生成 Corpus。`produce B01_FG809` 明确指定该批三天、最多 20 个 CPU worker，顺序执行：验证 source lock → 原生六行为生成及随附重播 → 独立 Native G3 → 独立 Reader/SQLite/精确下一批恢复（`--without-model`）→ 真实质量计数 → 新建 `BATCH_MANIFEST.json`。每阶段保存准确 argv、GNU time、stdout/stderr。没有训练器、优化器或 W&B 路径。

后续批次**各自手动选择**，先阅读 B01 覆盖及成本：

```bash
bash docs/train_corpus_20261018/RUN_TRAIN_CORPUS.sh produce B02_FG009
bash docs/train_corpus_20261018/RUN_TRAIN_CORPUS.sh produce B03_FG209
bash docs/train_corpus_20261018/RUN_TRAIN_CORPUS.sh produce B04_FG405
```

脚本没有 `all` 模式，也不会自动接续下一批。四条命令不等于代理获得生产授权。新版本目录和批次目录必须不存在；失败保留全部日志和部分产物，不能删除、覆盖或重签旧收据。重试需要独立输出版本/新收据，不能冒称 resume。

产物布局：

```text
/data/s3_corpus/early_train_20261018_cpu20_v1/
  code/{gym-trading-env,dreamerv3}/    # 固定运行源码提交
  batches/B01_FG809/                  # 原生 corpus_manifest/source_lock/六账户/Replay/Sidecar
  receipts/B01_FG809/
    native/                          # 本批重新执行的 G3 与 Admission 收据
    reader/                          # 独立索引、抽样/恢复与分母报告
    quality_coverage.json             # 读取真实标签与逐窗口 masks 的补充统计
    BATCH_MANIFEST.json               # 仅全部阶段通过后产生；含字节哈希和实际身份
  locks/PRODUCER_PIN.json
```

可另开终端运行 `vmstat 5` 并查看本批日志；GNU time 的 `(user+system)/elapsed` 反映实际平均用核。`time -v` 的最大 RSS 不等于所有进程的并发总 PSS，不能混用。需要进一步调度优化时先保留真实利用率、可用内存、I/O wait 和分钟/秒证据。

## 验收、迁移及尚未运行的部分

每批必须同时有新 G3 `status=PASS/G3=GO`、零 TEST 价格、匹配的 Corpus 身份，Reader 的原始 Replay/Trace/Sidecar Join、独立分母比较和精确下一批恢复通过。质量窗口缺支持按 `UNSUPPORTED/INSUFFICIENT_CURRICULUM_SUPPORT` 报告；invalid 不补 WAIT。质量支持和训练充分性分别判断；没有预造最低 N 或比例。

现有 Reader 一次绑定一份 Corpus、一个合约、一个 index；pair Owner 包含 Corpus/账户/segment/teacher/suffix 身份。四份独立产物可以逐份验收，**并未实现或认证跨 Corpus 的训练采样调度和完整恢复**。不拼接 NPZ/JSONL，不拼跨账户窗口，不修改 WP04。未来训练调度需独立任务，新数据身份不能沿用旧 same-run checkpoint。

新生产、首批 G3/Reader/质量矩阵、实际 20 核吞吐、全 18 合约生产、GPU、训练和优化器更新：`NOT_RUN`。只读来源验证及历史 FG201 Reader 的证据见 [CORPUS_COVERAGE_REPORT.md](CORPUS_COVERAGE_REPORT.md)。选定 CPU 回归、完整收集清单和未执行项见 [evidence](docs/train_corpus_20261018/evidence)。共享回归中 8 项的 `source_bundle` fixture 会执行 `ActorBC.update`，遵守用户禁令而 `NOT_RUN`；其余 12 项实际运行，不能宣称完整共享集合或全项目通过。

# Corpus 覆盖报告：2026-10-18

**本轮新生产 Corpus=0。B01–B04 的生产、Native G3、Reader 和质量覆盖均为 `NOT_RUN`；真实生成由用户执行。** 以下区分已核查来源、重新读取的旧 FG201 数据和待生产批次，任何历史 PASS 都不转写为新批次 PASS。

## 已观察的 TRAIN 来源

18 个来源共 488,160 根分钟线、1,455 个合约官方交易日。全部市场、Env、合约配置 SHA 匹配原 TRAIN 白名单；实际 symbol、行数、日期数、有序唯一时间戳和 TRAIN 日期归属检查通过。原始清单保留每个文件的绝对路径、SHA、大小、完整官方日列表，见 [SOURCE_INVENTORY.json](docs/train_corpus_20261018/evidence/SOURCE_INVENTORY.json)。未打开真实 DEV/TEST 价格。除四批 `validate` 外，其他来源的完整原生日历重新验收为 `NOT_RUN`。

| 合约 | 实际分钟行 | 官方日数 | 官方日起止 |
| --- | ---: | ---: | --- |
| CZCE.FG809 | 28,264 | 83 | 2018-04-11 .. 2018-08-08 |
| CZCE.FG901 | 29,416 | 86 | 2018-08-09 .. 2018-12-14 |
| CZCE.FG905 | 26,762 | 79 | 2018-12-17 .. 2019-04-16 |
| CZCE.FG909 | 26,546 | 78 | 2019-04-17 .. 2019-08-08 |
| CZCE.FG001 | 27,464 | 81 | 2019-08-09 .. 2019-12-09 |
| CZCE.FG005 | 20,505 | 75 | 2019-12-10 .. 2020-04-01 |
| CZCE.FG009 | 27,600 | 88 | 2020-04-02 .. 2020-08-11 |
| CZCE.FG101 | 28,050 | 82 | 2020-08-12 .. 2020-12-11 |
| CZCE.FG105 | 25,620 | 76 | 2020-12-14 .. 2021-04-07 |
| CZCE.FG109 | 24,945 | 73 | 2021-04-08 .. 2021-07-23 |
| CZCE.FG201 | 31,260 | 92 | 2021-07-26 .. 2021-12-09 |
| CZCE.FG205 | 23,220 | 68 | 2021-12-10 .. 2022-03-23 |
| CZCE.FG209 | 31,950 | 94 | 2022-03-24 .. 2022-08-10 |
| CZCE.FG301 | 27,703 | 81 | 2022-08-11 .. 2022-12-09 |
| CZCE.FG305 | 26,895 | 79 | 2022-12-12 .. 2023-04-10 |
| CZCE.FG309 | 28,740 | 84 | 2023-04-11 .. 2023-08-11 |
| CZCE.FG401 | 26,790 | 78 | 2023-08-14 .. 2023-12-07 |
| CZCE.FG405 | 26,430 | 78 | 2023-12-08 .. 2024-04-08 |

## 新批次：只有来源验证通过

| 批次 | 合约 | 官方日数 | 计划市场位置 | 实际新 Replay/账户/成交 | 新 G3 | 新 Reader / 质量覆盖 |
| --- | --- | ---: | ---: | --- | --- | --- |
| B01_FG809 | CZCE.FG809 | 3 | 1,034 | NOT_RUN | NOT_RUN | NOT_RUN |
| B02_FG009 | CZCE.FG009 | 3 | 1,035 | NOT_RUN | NOT_RUN | NOT_RUN |
| B03_FG209 | CZCE.FG209 | 3 | 1,035 | NOT_RUN | NOT_RUN | NOT_RUN |
| B04_FG405 | CZCE.FG405 | 3 | 915 | NOT_RUN | NOT_RUN | NOT_RUN |

四批的真实 `validate` 均 exit=0、`SOURCE_VALIDATED_ONLY`，核对了完整 TRAIN 白名单、日历、连续有效分钟和三日范围；该动作没有发布 Replay、教师标签或训练许可。完整 [逐批命令及 stdout](docs/train_corpus_20261018/evidence/source_validation) 和 [不可变计划 manifest](docs/train_corpus_20261018/batches) 已保留。四个 source lock 的内容身份分别为：

```text
B01 88ea5b0128f8636265f6d6f7fe4ed5ae901ede007139eb854c97fc6c94fce239
B02 08903e8e8a4989d67db021e0302d9982cec5bb9df85f2840bcabdac64d3c104e
B03 0e9c55b7e6c8c63a8053ffa9d8403b5c03f14042b719640d3597386f107ec13b
B04 49d3ba9da70281c52f5f32360f9ed8c3d096504d760a35ff944d8ec00cc99af8
```

4,019 个市场位置、4 合约、12 合约官方日是**计划覆盖**，不计入实际教材规模。新批次 TP/SL、逆势后亏损、长持仓/水下时长、MACD/七窗口有效率、strict ADD/WAIT/Pair、自然和分层采样支持均为 `NOT_RUN`，不是 0，也不是已有支持。

## 本轮重新观察的旧 FG201 小样

原目录：

```text
/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp02/real_small_04
```

本轮使用新工作树的真实 `tools/fg_s3_e_quality_reader.py read --without-model`，在 Corpus 外新建 SQLite，完成 Replay/Trace/Sidecar Join、10 个阶段/抽样视图、各账户读取与精确下一批恢复，exit=0、`PASS_READER_ONLY`。索引耗时 6.266s、抽样 7.097s、索引 10,002,432 字节、Reader 峰值 RSS 328,953,856 字节。收据：[命令](docs/train_corpus_20261018/evidence/baseline_reader.command.json)、[Join](docs/train_corpus_20261018/evidence/baseline_reader/WP03_TEACHER_JOIN_AUDIT.json)、[完整 Reader 覆盖](docs/train_corpus_20261018/evidence/baseline_reader/WP03_COURSE_READER_COVERAGE.json)。SQLite 原件仍保留在命令记录中的外部路径，未重复提交大型二进制。

Corpus 内容身份：`640865b4239fb483e5d98daf18d3104e670ff9b3c1486cb63ec9924cf6e089a9`；原生 manifest 文件 SHA256：`e2c0fc4004eea2bd72a33cb734c3d3a4dcab0248accd8122b46d8add6a9f97f3`；本轮只读索引 SHA256：`c145e57ca1f1d17df2bcb48ae286d5fe807d6122ad86de3807207ebda482f1e1`。原 Corpus 的文件内容检查由真实 Reader 完成，未改写旧文件。

**本轮没有重跑其 Native G3。** 当前 Reader Admission 是 `DEVELOPMENT_VALIDATED/INTEGRITY_CHECKED_ONLY/NOT_ADMITTED`；历史 WP03 Native G3 exit=0、GO、1693.454s 仅保留为历史证据，见原有 [native_verification.json](docs/receipts/wp03/native_verification.json)。不冒充本轮或新批次原生认证。

覆盖仍只有 CZCE.FG201、2021-12-08/09：690 唯一市场 cursor、689 真正推进分钟、4,146 物理 Replay 行、4,134 已执行账户决策，优化器曝光 0。六个账户共享行情，不是六倍独立市场数据。

| 已观察量 | 账户状态上的候选次数 | 去重后的合约×cursor 支持 |
| --- | ---: | ---: |
| L1 候选 | 1,551 | 689 |
| 后验 TP5 | 1,307 | 590 |
| 后验 SL30 | 131 | 46 |
| 后验 OTHER | 113 | 53 |
| TP，持有 ≤5 市场分钟 | 288 | 115 |
| TP，持有 >5 市场分钟 | 1,019 | 475 |
| 5 分钟内逆势 ≥10 点，最终 SL30 | 84 | 27 |
| 15 分钟内逆势 ≥10 点，最终 SL30 | 129 | 45 |
| 最终 TP，且完整 MAE15 ≥10 点 | 416 | 182 |

上述由 [audit_coverage.py](docs/train_corpus_20261018/audit_coverage.py) 重新逐行读取真实标签，并与原生 coverage 候选数核对；原始输出为 [baseline_quality.json](docs/train_corpus_20261018/evidence/baseline_quality.json)。指标可重叠，后验候选不是真实成交，更不能相加为独立盈利样本。该结果证明旧小样存在快速逆势后 SL 的案例，仍不能说明跨合约、跨时期的支持充分。

真实成交为 **119=103 TP+12 SL+4 MAIN_ROLL**；六账户均有 691 物理行。P0 无成交；P4 的 54 笔 TP 属于后验教师行为，不是学生 Actor 表现。真实 RIGHT_CENSORED=0，不能宣称这一题型已被真实市场覆盖。

| 账户 | L1 候选 | 实际 TP / SL / OTHER | MAE15 P50 / P90 | TP 时间 P50 / P90 | 水下分钟 P50 / P90 |
| --- | ---: | --- | --- | --- | --- |
| P0 | 689 | 0 / 0 / 0 | 8 / 16 | 18 / 88 | 16 / 116 |
| P1 | 21 | 16 / 4 / 1 | 9 / 21 | 11 / 82 | 9 / 95 |
| P2_low | 89 | 9 / 2 / 1 | 8 / 20 | 12.5 / 95.7 | 17 / 95.2 |
| P2_mid | 39 | 17 / 4 / 1 | 6 / 21.2 | 6 / 69.9 | 7 / 71.6 |
| P3 | 160 | 7 / 2 / 1 | 6 / 20 | 9 / 78.8 | 7.5 / 87.5 |
| P4 | 553 | 54 / 0 / 0 | 8 / 16 | 18 / 89 | 18 / 131.6 |

表中质量分布的分母为各账户 L1 反事实候选。MAE 单位为报价点，TP 持有与水下时长单位为有效市场分钟；分位数不能跨账户相加或平均后当作全局分位数。

本轮 Join 得到投影 L1 outcome=1,543、score=1,430，strict ADD=171、WAIT=1,279，93 个资金域外 invalid 保持无动作标签。原 L1 候选 1,551 与投影相差 8，是 reset/prefix/未执行位置不得用于教学；不修改原标签或补 WAIT。

事件七窗口（5/15/60/120/240 市场分钟、入场官方日末、下一官方日末）各有 1,543 个有效投影元素，统一物理行分母 4,146，**各为 37.2166%**。七窗口合计 10,801；touch 八个元素合计 12,344；adverse 四个元素合计 6,172。原始 MACD valid=4,146/4,146，包含完整 TRAIN 因果预热，不代表整个合约起始阶段也已覆盖。

四个阶段的 NATURAL/STRATIFIED/MIXED 实际读取及恢复详见原收据；WM 只用 NATURAL。各请求窗口在这个旧小样内有支持，Reader 返回 `SUPPORTED_FOR_REQUESTED_WINDOWS`，研究规模仍判断为 **LIMITED**：单合约两天、WAIT 占多、账户/市场重复曝光、无真实删失、不能据此宣称泛化或 TRAIN GO。全 episode pair 数与实际抽样曝光分开保留；真实优化器曝光始终为 0。

## 下一批必须填入的覆盖矩阵

按合约×官方日保留自然分母，再分别报告六账户行为状态、实际 ADD/TP/SL/OTHER/未平仓、L1 后验结果与 TP 时间/MAE/水下分布、短时逆势再 SL、低质量 TP、同日/跨日、夜盘/白盘、MACD 与逐个窗口有效率、L2 ADD/WAIT/invalid，以及各 Reader 阶段、自然/分层/混合的合法窗口与 Pair 支持。

原生 `coverage.json/coverage.csv` 和独立 Reader 提供原统计；补充审计只读取已生成标签与索引，不造标签。某窗口没有支持时标 `UNSUPPORTED`；未运行标 `NOT_RUN`。按实际 missing class 决定新的真实 TRAIN 日期追加，不预造比例、不拼账户、不借未来筛选可盈利 Observation。

## 验证范围与未验证项

新源码在独立 CPU worktree 运行控制项、WP02 CLI/事实/并行标签/恢复、WP03 Admission/Join/Reader、市场分钟及共享安全子集回归。准确命令、用例清单、源码身份和结果保存在 [evidence](docs/train_corpus_20261018/evidence)。共享集合完整收集 20 项，其中 12 项执行；另 8 项会使用实际优化器更新的 `source_bundle` fixture，按用户禁令 `NOT_RUN`。不修改或跳过失败来制造 PASS，不声称全项目通过。

最终运行源码提交 `63f2f8c35d4f84d9dfd11d2bab8e4b4d2c6d876c` 上选定集合 **140 passed，0 failed/error/skipped，219.79s**，见 [JUnit](docs/train_corpus_20261018/evidence/release_code_63f2f8c.xml) 和 [140 项明细及未执行的 8 项](docs/train_corpus_20261018/evidence/VERIFICATION_SUMMARY.json)。最终生产命令随后按用户要求改为 20 worker、affinity 0–19；用户明确要求无需另验 20 核，因此该并发/利用率验证为 `NOT_RUN_PER_USER_REQUEST`，没有重跑。

原限制的代码根因是 Producer 的固定末两日切片、输出目录检查和未暴露的进程预算。最小实现保留原 Env/Driver/Replay/教师/质量公式与 fold 信任链，只扩展选择、预算、路径及明确的新生产者身份。新窗口截断使用原有 `end_cursor`，合成原生回归确认观察结束早于完整教师后缀时，账户仍可精确重播而不伪造终止。

正式新 Corpus 的真实 G3/Reader/质量覆盖、持续 20 核利用率/峰值进程树内存/工期、跨 Corpus 训练调度及恢复、全 18 合约、GPU、正式训练：**NOT_RUN**。下一步由用户运行 [生产方案中的 B01 命令](TRAIN_CORPUS_PRODUCTION_PLAN.md)，依据实际覆盖与资源收据再选择后续批次。

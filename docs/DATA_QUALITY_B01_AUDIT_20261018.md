# P0 DATA-QUALITY：B01 现场成果与覆盖审计

本轮只处理 [并行工作流任务书](</home/v/Downloads/FG_S3_E_Quality_Parallel_Workstreams_20261018_V1.md>) 的第一个P0。**首批 B01 的生产→G3→Reader→质量/成本报告闭环已完成；扩大跨合约教材仍待后续批次。** 本轮没有启动任何新生产或训练，没有推进DATA-MULTI/EVAL-BASE/TRAIN-PLAN。

## 成果清单

| 任务成果 | 实际证据与状态 |
| --- | --- |
| 合法生产与逐批保存 | 用户已生成B01；现场 `BATCH_MANIFEST.json` 状态 `NATIVE_G3_AND_INDEPENDENT_READER_COMPLETED` |
| Native G3 / 独立Reader | GO / PASS_READER_ONLY；六账户逐行重放各1035行，Reader覆盖LIMITED |
| 质量分布及分母 | 本文及 [B01_AUDIT.json](data_quality_b01_20261018/B01_AUDIT.json)，复用已有审计器并只读统计原Sidecar/SQLite |
| 缺题型表 | 见下表，零支持、少量支持与尚未运行分开 |
| 速度成本 | 五阶段现场GNU time与原性能收据，见下表 |
| 下一批TRAIN日选择 | B02_FG009，2020-08-07/10/11，SELECTED_NOT_RUN；命令见下文 |

原件：[BATCH_MANIFEST](</data/s3_corpus/early_train_20261018_cpu20_v2/receipts/B01_FG809/BATCH_MANIFEST.json>)、[G3](</data/s3_corpus/early_train_20261018_cpu20_v2/receipts/B01_FG809/native/native_verification.json>)、[Reader](</data/s3_corpus/early_train_20261018_cpu20_v2/receipts/B01_FG809/reader/WP03_COURSE_READER_COVERAGE.json>)、[质量](</data/s3_corpus/early_train_20261018_cpu20_v2/receipts/B01_FG809/quality_coverage.json>)。只读核对了最终清单全部125个文件SHA，并重新统计得到与原质量报告完全相同的结果；没有重跑G3或改写现场文件。

生产者固定 Gym `b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46` / Dreamer `fd144113ce706e2d8f1ce8e7828ece0799c8bf7a`，两仓生产时clean。Corpus身份 `3fcbd448e4048afbf29b8c61dbda11cae5af999ee73440471704b7b87414d3b5`，BATCH内容身份 `17a2cab3deffc560bca36e2c6a0451517af51408b887cebf616b1615ab52801a`；当前开发HEAD不替换这两个生产者SHA。

## 质量分布和分母

一个合约FG809、官方日2018-08-06/07/08；6210物理Replay行（每账户1035），1034个独立市场cursor，1033有效推进分钟，6198个账户决策。六账户共享行情，不是6210个独立市场位置。

原始L1候选为1968次账户×状态、有候选的独立市场cursor为1033。下表左列是候选次数，右列按合约×cursor去重；指标可重叠，不能相加当实际交易。正负方向分别由TP/SL和strict ADD/WAIT呈现；OTHER_EXIT及score无效不补成失败或WAIT。

| 候选题型 | 账户状态候选次数 | 独立cursor支持 |
| --- | ---: | ---: |
| TP5候选 | 973 | 490 |
| SL30候选 | 11 | 5 |
| 其他退出候选 | 984 | 538 |
| 快速TP（≤5市场分钟） | 8 | 5 |
| 较慢TP（>5市场分钟） | 965 | 485 |
| 5分钟内逆势≥10 | 31 | 11 |
| 15分钟内逆势≥10 | 92 | 36 |
| 5分钟内逆势10后SL30 | 6 | 2 |
| 15分钟内逆势10后SL30 | 9 | 3 |
| TP且完整MAE15≥10 | 38 | 14 |
| 同官方日TP | 394 | 206 |
| 跨官方日TP | 579 | 284 |

以下分布来自原始L1候选，单位：MAE为报价点，TP耗时/水下时长为有效市场分钟，score为0..100。分母n逐项列出，未把各账户分位数平均。

| 分布 | n | P50 | P90 | 最小..最大 |
| --- | ---: | ---: | ---: | --- |
| mae_15 | 1968 | 4 | 8 | 2.0 .. 25.0 |
| mae_full | 1968 | 9 | 26 | 2.0 .. 31.0 |
| underwater_minutes | 1968 | 137 | 874 | 0 .. 1025 |
| tp_market_minutes | 973 | 109 | 262.8 | 3 .. 372 |
| score | 984 | 45 | 97.81 | 0.0 .. 100.0 |

实际账户退出为 **14 TP、0 SL、4 MAIN_ROLL**；18笔均已结束，真实右删失0。P4的9笔TP来自后验教师行为，不是学生表现。候选SL30有11次/5个独立cursor，与实际账户没有走到SL并不矛盾。

## 有效位、无效原因和Reader支持

物理行分母6210：投影 outcome_valid=1961（31.5781%），score_valid=983，strict有效1961，其中 **ADD=23、WAIT=1938**。strict其余4249行保持无效，不能填WAIT；七事件窗口均1961/6210，touch八元素、adverse四元素也各为1961/6210。Q_v2/Soft有效数为0，符合其关闭合同。

原始无效原因如下；score行属于outcome已知但score未定义，不能与L1/L2原因相加。投影阶段还按reset/执行边界过滤，原始候选和投影数量不同不是缺失补标。

| 原始原因 | 行数 |
| --- | ---: |
| score/OTHER_OR_CENSORED | 984 |
| L2/NO_ACTIVE_DECISION | 12 |
| L1/FORCED_WAIT_ACTION_MASK | 7 |
| L2/NO_LEARNABLE_DECISION | 4230 |
| L1/FORCED_WAIT_POSITION | 4235 |

投影strict无效行的重叠标志：ADD非法4242、未执行12、first=12、last=12；标志可重叠。Reader没有把强制WAIT持仓状态转成可学习ADD/WAIT题。

64步的10个阶段/采样视图均SUPPORTED；P1的单账户HEADS_VALID合格窗口为0，故总体LIMITED。其余账户窗口数P0=907、P2_low=2、P2_mid=37、P3=216、P4=887；这只是该长度及任务条件下的窗口支持，不是独立样本数。

## 缺题型和下一批

| 当前缺口/薄弱项 | 本批观察 | 处理 |
| --- | --- | --- |
| 实际行为SL30退出 | 0；候选SL独立cursor=5 | 本批没有该真实行为题型；不补造SL |
| 真实/候选右删失 | 均0 | 标零支持，不声称已覆盖 |
| 快速TP与SL候选 | 各仅5个独立cursor | 有支持但规模有限；没有设未经批准的“达标”阈值 |
| 快速逆势后SL | 5分钟型2、15分钟型3个独立cursor | 保留稀少计数，扩大后复核 |
| P1 HEADS_VALID 64步窗口 | 0 | Reader显式UNSUPPORTED；原轨迹完整保留 |
| 跨合约/跨年份 | 当前只有1合约3日 | 下一批选已锁定B02；B02–B04全为NOT_RUN |

下一批保持原先锁定的 **B02_FG009：2020-08-07、10、11**，1035计划市场位置，20-worker预算，仍由用户正式生成。选择依据是预先列明的跨年份/合约范围，没有读取下一批未来结果挑日；不能保证该批必然填满上述题型，须生成后再核对。

```bash
cd /home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env
bash docs/data_macd_fix_20261018/RUN_TRAIN_CORPUS.sh produce B02_FG009
```

B01已经存在；不再执行prepare或重跑B01，也不自动跑全18合约。该命令本轮NOT_RUN。DATA-MULTI真实最终验收需要至少两份已认证Corpus，当前这里只完成一份；其工程工作本轮未开展。

## 速度与成本

下表是生产现场既有日志，预算20个账户+标签worker（6+14），不是本轮另做占核验证。

| 阶段 | 墙钟秒 | 平均CPU | 退出码 |
| --- | ---: | ---: | ---: |
| validate | 12.85 | 99% | 0 |
| generate | 3373.27 | 1270% | 0 |
| native_g3 | 1607.82 | 1163% | 0 |
| reader | 18.70 | 99% | 0 |
| coverage | 0.69 | 99% | 0 |

五阶段合计 **5013.33秒，约83分33秒**；generate约56分13秒，独立G3约26分48秒。平均CPU分别约12.7/11.63个逻辑核，不能把20-worker预算写成全程占满20核。GNU time最大RSS约631/637MiB是其所报进程峰值，**不是全进程树PSS**；进程树内存峰值未额外测量。

发布文件计176107342字节（Reader口径，不含顶层manifest自身），Reader索引14794752字节。Reader建索引10.194秒、抽样7.636秒，3840有效采样行约502.88行/秒；重复抽样不增加独立市场规模。

## 审计重现与限制

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B docs/data_quality_b01_20261018/audit.py \
  --out /tmp/B01_AUDIT_new.json
```

[只读审计脚本](data_quality_b01_20261018/audit.py) 复用既有 `audit_coverage.py`，拒绝覆盖已有结果或写入生产树。本轮只改报告与审计文档，不更改Producer/Reader/模型/金融规则或来源锁。优化器更新0；未打开TEST、未用GPU、未改WP04 worktree。正式训练仍未准入；后续批次及其他工作流不能用本批PASS代替。

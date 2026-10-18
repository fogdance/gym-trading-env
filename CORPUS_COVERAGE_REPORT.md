# Corpus 覆盖报告：2026-10-18，MACD V3

**正式V3 B01已由用户生成，Native G3=GO、独立Reader=PASS_READER_ONLY，覆盖LIMITED；B02–B04为NOT_RUN。** 现场收据和逐项审计见 [第一个P0 DATA-QUALITY成果](docs/DATA_QUALITY_B01_AUDIT_20261018.md)。旧cpu20_v1失败尝试、下述DEV小样、旧FG201 V2均不计入B01的正式规模。

## 正式B01：已核验的现场结果

6210物理行、1034独立cursor、1033推进分钟、3官方日、6198账户决策；125个最终清单文件SHA全部匹配。六账户各1035行Native G3逐行重放PASS。

| 本批候选题型 | 账户状态候选次数 | 独立cursor支持 |
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

上述候选总分母1968次/1033有候选cursor，指标可重叠；实际退出14 TP、0 SL、4 MAIN_ROLL。Reader投影strict ADD/WAIT=23/1938，有效1961/6210，其余无效不补WAIT；score有效983/6210。P1无HEADS_VALID 64步窗口，总体LIMITED；其他10个阶段/抽样视图均SUPPORTED。

缺口：真实SL退出和右删失为0，快速TP及SL候选独立位置各仅5个，单合约3日仍不足以证明跨合约覆盖。下一批选择已锁定B02 FG009的2020-08-07/10/11，状态SELECTED_NOT_RUN。

现场成本：生成3373.27秒、G3 1607.82秒、Reader 18.70秒；五阶段合计5013.33秒。报告含完整分布、无效原因、缺题型表及分母；产物和收据保持原位且未重签。

## 历史DEV小样（不计入正式B01）

### 开发小样记录

合约 CZCE.FG809，官方日2018-08-08；来源只含主力资格期间行情。观察窗口为5个cursor、4次有效市场推进，教师保留到真实主力退出点的完整TRAIN后缀。精确范围及身份见 [原生source lock](docs/data_macd_fix_20261018/evidence/DEV_source_lock.json)。

| 观察量 | 实测 |
| --- | ---: |
| 原生账户 P0/P1/P2_low/P2_mid/P3/P4 | 6 |
| StudentReplay物理行 | 30（每账户5） |
| 去重市场cursor / 推进分钟 / 官方日 | 5 / 4 / 1 |
| 已执行账户决策 | 24 |
| 六账户原生复核 | 每账户5行及5行Teacher标签，全部PASS |
| MACD数组 | 30/30有限float32，末维4；valid=1，来自完整因果预热 |
| 实际ADD成交 | 3，诊断停止时持仓未完成退出 |
| 实际TP / SL / 其他退出 | 0 / 0 / 0 |
| 原始L1反事实候选 | 21次账户状态，去重5个市场cursor |
| L1后验TP5 / SL30 / OTHER_EXIT | 0 / 0 / 21 |
| 投影strict有效 / ADD / WAIT | 12 / 0 / 12；其余18行不补标签 |
| 七事件窗口有效数 | 每个12/30=40% |
| 优化器更新 / 新独立Native G3 | 0 / NOT_RUN |

OTHER_EXIT是完整教师后缀上的候选结果，不是小样窗口内实际退出。诊断停止处3笔持仓的删失也不等于教师后缀缺失。候选次数、物理行和去重市场位置不能相加为独立金融样本。

Corpus内容身份 `636ceada845a4cf5c73be364584147acf99dff951ca52d40b3bd3191aafc7304`；原生manifest文件SHA256 `5d253bc87e269026965434bb281203f5f5a3a1840797f3dedf9c4274f19544a5`。真实产物目录及完整复核见 [DEV结果](docs/data_macd_fix_20261018/evidence/DEV_SMOKE_RESULT.json)、[不可变原生manifest副本](docs/data_macd_fix_20261018/evidence/DEV_corpus_manifest.json)。原始Replay/Sidecar保留在结果中的开发artifacts目录。

### DEV Reader和题型支持（历史）

Reader独立建立三方Replay/Trace/Sidecar Join，30行身份核对通过，`PASS_READER_ONLY / LIMITED`，优化器曝光0。索引SHA256 `9cbcd6a92608dac8e09b9273c9deb7ddd2a3047b4a9f3ae852580332959ab52e`。

WM_NATURAL、HEADS_VALID:NATURAL、ACTOR_STRICT:NATURAL的4步窗口可读取并精确恢复下一批。Quality Pair、缺少类别的分层/混合窗口，以及P1单账户的4步HEADS_VALID窗口标为 `INSUFFICIENT_CURRICULUM_SUPPORT`，未采样替代类别。

当前小样对快速TP、SL30、快速逆势后SL、TP后高MAE、strict ADD、Quality Pair均无支持；不能作为这些题型验收。Reader最初因无Quality Pair中断的日志保留，修复诊断报告后才返回明确LIMITED；没有放宽实际Reader.sample或模型检查。

原始 [Reader覆盖](docs/data_macd_fix_20261018/evidence/WP03_COURSE_READER_COVERAGE.json)、[Join](docs/data_macd_fix_20261018/evidence/WP03_TEACHER_JOIN_AUDIT.json)、[逐行质量统计](docs/data_macd_fix_20261018/evidence/dev_quality_coverage.json) 均可核对。StudentReplay中无Teacher/Quality/Oracle/Outcome字段，Teacher仍在独立Sidecar。

## 正式计划覆盖

| 批次 | 计划合约/官方日数/市场位置 | 正式生成 | 新Native G3 | 整批Reader及质量覆盖 |
| --- | --- | --- | --- | --- |
| B01_FG809 | FG809 / 3 / 1034 | COMPLETED | GO | PASS_READER_ONLY / LIMITED；已审计 |
| B02_FG009 | FG009 / 3 / 1035 | NOT_RUN | NOT_RUN | NOT_RUN |
| B03_FG209 | FG209 / 3 / 1035 | NOT_RUN | NOT_RUN | NOT_RUN |
| B04_FG405 | FG405 / 3 / 915 | NOT_RUN | NOT_RUN | NOT_RUN |

已完成1合约/3官方日/1034位置；4合约、12合约官方日、4019位置仍是总计划，其他三批NOT_RUN。详见 [生产计划](TRAIN_CORPUS_PRODUCTION_PLAN.md)。各批按实际TP/SL、MAE、L2 valid/invalid、strict ADD/WAIT/Pair、自然/分层采样支持填写结果，不能预填0或继承旧PASS。

## 保留的历史证据

旧 `/data/s3_corpus/early_train_20261018_cpu20_v1` B01：generate exit2、`MACD incomplete previous bucket`，42.50s，失败于预热，没有完整Corpus；本轮未改写。旧FG201 V2：690个cursor、4146物理行及历史G3/Reader收据保留在 [旧计划证据](docs/train_corpus_20261018/evidence)，本轮不重签、不迁移，也不将其旧数学合同当作V3认证。

现有18个TRAIN来源清点及SHA见 [历史来源清单](docs/train_corpus_20261018/evidence/SOURCE_INVENTORY.json)。四批来源在计划发布时已validate；本轮仅审计B01现场收据和标签，未启动新生产、未读取TEST、未使用GPU或真实优化器。

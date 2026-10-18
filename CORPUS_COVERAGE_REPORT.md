# Corpus 覆盖报告：2026-10-18，MACD V3

**正式 V3 B01–B04 全部 NOT_RUN。** 本轮仅生成单worker有限真实TRAIN DEV小样，标记 `DEV_ONLY / NOT_CERTIFIED`。旧B01正式尝试失败，旧FG201 V2历史证据不计入本轮V3规模。

## 已实际运行的小样

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

## 独立Reader和题型支持

Reader独立建立三方Replay/Trace/Sidecar Join，30行身份核对通过，`PASS_READER_ONLY / LIMITED`，优化器曝光0。索引SHA256 `9cbcd6a92608dac8e09b9273c9deb7ddd2a3047b4a9f3ae852580332959ab52e`。

WM_NATURAL、HEADS_VALID:NATURAL、ACTOR_STRICT:NATURAL的4步窗口可读取并精确恢复下一批。Quality Pair、缺少类别的分层/混合窗口，以及P1单账户的4步HEADS_VALID窗口标为 `INSUFFICIENT_CURRICULUM_SUPPORT`，未采样替代类别。

当前小样对快速TP、SL30、快速逆势后SL、TP后高MAE、strict ADD、Quality Pair均无支持；不能作为这些题型验收。Reader最初因无Quality Pair中断的日志保留，修复诊断报告后才返回明确LIMITED；没有放宽实际Reader.sample或模型检查。

原始 [Reader覆盖](docs/data_macd_fix_20261018/evidence/WP03_COURSE_READER_COVERAGE.json)、[Join](docs/data_macd_fix_20261018/evidence/WP03_TEACHER_JOIN_AUDIT.json)、[逐行质量统计](docs/data_macd_fix_20261018/evidence/dev_quality_coverage.json) 均可核对。StudentReplay中无Teacher/Quality/Oracle/Outcome字段，Teacher仍在独立Sidecar。

## 正式计划覆盖

| 批次 | 计划合约/官方日数/市场位置 | 正式生成 | 新Native G3 | 整批Reader及质量覆盖 |
| --- | --- | --- | --- | --- |
| B01_FG809 | FG809 / 3 / 1034 | NOT_RUN | NOT_RUN | NOT_RUN |
| B02_FG009 | FG009 / 3 / 1035 | NOT_RUN | NOT_RUN | NOT_RUN |
| B03_FG209 | FG209 / 3 / 1035 | NOT_RUN | NOT_RUN | NOT_RUN |
| B04_FG405 | FG405 / 3 / 915 | NOT_RUN | NOT_RUN | NOT_RUN |

4合约、12合约官方日、4019位置只是计划，详见 [生产计划](TRAIN_CORPUS_PRODUCTION_PLAN.md)。各批按实际TP/SL、MAE、L2 valid/invalid、strict ADD/WAIT/Pair、自然/分层采样支持填写结果，不能预填0或继承旧PASS。

## 保留的历史证据

旧 `/data/s3_corpus/early_train_20261018_cpu20_v1` B01：generate exit2、`MACD incomplete previous bucket`，42.50s，失败于预热，没有完整Corpus；本轮未改写。旧FG201 V2：690个cursor、4146物理行及历史G3/Reader收据保留在 [旧计划证据](docs/train_corpus_20261018/evidence)，本轮不重签、不迁移，也不将其旧数学合同当作V3认证。

现有18个TRAIN来源清点及SHA见 [历史来源清单](docs/train_corpus_20261018/evidence/SOURCE_INVENTORY.json)。本轮只对锁定4批重新validate，未自动生产全18合约；无TEST价格、GPU或真实优化器更新。

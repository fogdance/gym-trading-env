# FG ABC 开发记录 V1

日期：2026-10-10。开发分支：两个原仓库均为 `feat/fg-abcd-comparison-v1`。

## P0：配置与覆盖

入口：`tools/build_fg_abcd.py`；唯一实验配置：`configs/fg_entry_v1/abcd_comparison.yaml`，引用统一 `fg.yaml`，旧研究配置未改。

已生成清单：`artifacts/fg_abcd_v1/v1/experiment_manifest.json`，哈希 `7ba117ed5be9fa489d218ac6a5165b29e48d3c01d44ef2e933d45da19e3d753a`。生成文件不提交。

| 验证合约 | A 合约 / 根数 | B24 合约 / 根数 |
|---|---|---|
| FG205 | 11 / 296432 | 6 / 157980 |
| FG305 | 14 / 379305 | 6 / 164698 |
| FG405 | 17 / 461730 | 6 / 165298 |
| FG409 | 18 / 488160 | 5 / 136558 |

共 12 个 fold/方法组合，实例化为 3 个先导和 36 个正式方法实例、52 个阶段。52 个阶段通过现有 Dreamer FG 准入校验。Test 只有封存日期元数据，没有复制盲测价格。FG405 的 3795 根 NO_TRADE 保留。

## P1：账户轮转与迁移

`entry_dataset_env.py` 增加显式 `least_progress` 模式。旧清单不需要 `rows`，旧循环规则保留。新模式按已推进根数/合约总根数轮转；schema2 保存账户 bank、游标、累计推进、重复数及 seed 顺序。

新 C 迁移采用 `run.fg_stage_transfer` 收据。普通 `from_checkpoint` 仍拒绝模型单独恢复，普通同目录恢复仍完整恢复账户。收据校验源预算、seed、fold、模型结构、近期子集、文件哈希及独立日志目录。完整 Agent 包含原优化器状态；迁移保留 warm-up gate，近期 Replay/账户重新建立，预训练完整账户留在源 checkpoint。

## P2：统一评估

`tools/evaluate_fg_abcd.py` 校验开发验证清单；`fg_abcd_evaluation.py` 使用真实 Env 账户回放。包含三种基线、确定性模型回放、30 seed 闭环配额随机诊断、逐笔和账户 CSV、Reward 对账及风险指标。

随机配额仅使用已知观察时钟、人工方向、自身当前 mask，并仅在真实开仓成交后扣减。无法匹配明确 `partial_match`。日历来自用户确认的观察数据，并非独立交易所日历。

## 已完成的验证

```bash
env MPLCONFIGDIR=/tmp/fg-matplotlib PYTHONPATH=src \
  /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q \
  tests/entry_v1/test_fg_abcd.py tests/entry_v1/test_entry_contract.py \
  tests/entry_v1/test_entry_data.py
```

结果：39 项通过。覆盖轮转持仓浮亏、JSON 恢复后下一步一致、完整窗口选择、重复评估一致、Reward 与期末浮亏、固定分母以及原入口交易回归。

Dreamer `dreamerv3/tests/test_fg_stage_transfer.py`：1 项通过，覆盖结构、完成预算和 fold 不匹配拒绝。沙箱运行曾出现 pytest 缓存只读警告，测试本身通过。

`git diff --check` 两仓库通过。后续修改仍需重跑其相关测试和自查。

## 待完成，禁止误记完成

- P3 正式 batch16 × length64 的 CUDA smoke（2040 交互，包括同目录恢复、C 迁移）：正在验证。
- P4 A/B/C 100k：NOT_RUN。只有 P3 和评估验收通过后才启动。
- P5 正式 1M：36 个方法实例准备为 NOT_RUN，不自动启动。
- D：后续滚动设计，本轮不运行。

smoke 与先导的实际结果、失败和修复证据将补充到本记录。观察到基线数字不等于 Dreamer 具有择时能力。

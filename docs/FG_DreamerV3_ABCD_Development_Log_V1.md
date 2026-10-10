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

结果：40 项通过。覆盖轮转持仓浮亏、JSON 恢复后下一步一致、完整窗口选择、重复评估一致、Reward 与期末浮亏、固定分母、强退未成交及原入口交易回归。

Dreamer `dreamerv3/tests/test_fg_stage_transfer.py`：1 项通过，覆盖结构、完成预算和 fold 不匹配拒绝。沙箱运行曾出现 pytest 缓存只读警告，测试本身通过。

`git diff --check` 两仓库通过。后续修改仍需重跑其相关测试和自查。

## 待完成，禁止误记完成

- P3 正式 batch16 × length64 的 CUDA smoke：PASS；独立 v2 共 2040 交互、3 次真实 GPU 更新，同目录恢复与 C 迁移均通过。
- P4 A/B/C 100k：已启动顺序执行，当前 A RUNNING；B、C 尚未运行。通过学习审计的模型随后自动评估，未通过 gate 则明确不作策略学习结论。
- P5 正式 1M：36 个方法实例准备为 NOT_RUN，不自动启动。
- D：后续滚动设计，本轮不运行。

smoke 与先导的实际结果、失败和修复证据将补充到本记录。观察到基线数字不等于 Dreamer 具有择时能力。

## smoke 发现与修复

第一轮 `smoke/`：预训练 1040→1050 的完整恢复成功，但梯度更新为 0。Replay 长度 65，1040 交互只有 976 条可采样序列，不足 batch16×64 的 1024 门槛。近期迁移随后被配置结构校验拒绝：PyYAML 将 Dreamer 保存的 `4e-05`/`1e-20` 读成字符串。旧日志完整保留，不算通过。

修复：Dreamer 模型配置使用相同的 ruamel YAML 读取器；科学计数法回归测试通过。新独立 `smoke_v2/` 预算为 1120+920=2040，前段 1110→1120 验证恢复，已观察到 3 次真实 GPU 更新。Actor/Critic 参数有限但 gate 未开启、变化为零，符合 warm-up，不宣称策略学习。

评估自查修复：按开仓 bob 定位执行 bar，排除同时间戳的上一根信号 bar；新增信号 bar 极端低价回归。强退未成交保留账户亏损并单独计终局奖励失败，不冒充已平仓或右删失。旧 `validation_baselines_FG409/` 的 MAE 路径存在此边界问题，后续采用新 `validation_baselines_FG409_v2/`；旧产物保留。

`tools/audit_fg_abcd_smoke.py` 实际检查点审计 PASS：完整 Agent 所有参数与 182 个优化器变量在近期阶段精确相等；warm-up gate 保留；非活动账户完全相同，活动账户恢复后仅推进 10 根并保留历史账簿/成交前缀；近期局部 920 交互、实际 919 根，账户重新建立。该短阶段不发生梯度更新，因此能直接核对完整迁移；模型择时能力仍未验证。

## P4 启动证据

启动入口：`tools/run_fg_abcd.py --budget pilot --execute`；已确认 `[CudaDevice(id=0)]`。A 日志：`/data/logdir/fg_abcd_v1/v1/FG409/pilot/A/0/train/`。启动代码 SHA：Env `d2c105b`、Dreamer `fce03e4`，保存于 `launch_metadata.json`。

首份训练日志到 2900 交互，`train/opt/updates` 的窗口均值 57、`train/avail/ready=0`。该均值不是精确累计更新数，尚未证明 Actor/Critic 有效学习。最终以 `learning_audit.json` 的真实参数变化和 gate 为准。

修正后的三个 FG409 基线在 `validation_baselines_FG409_v2/` 全部完成：固定分母 91 日，Always/Random/Rule 的 Reward/day 分别约 0.249892 / 0.298110 / 0.250915，TP 批次 73 / 80 / 73，Reward 和动作对账错误均为零。属于基线事实，不是 Dreamer 结果。

`tools/report_fg_abcd.py` 从冻结清单、运行元数据及日志更新生成的 `assumptions.md`、`train_config_matrix.md`、`pilot_results.md`，生成文件不提交。正式 36 个方法实例仍全部 NOT_RUN。

## 评估与状态工具补充验收

相关 Env 回归共 41 项通过（`test_fg_abcd.py`、`test_entry_contract.py`、`test_entry_data.py`），包括真实 Env 的两进程随机回放：独立 seed 和独立账户均保留未完成仓位与实际亏损。无成交时 CSV 仍有可读取的字段表头。配额随机报告增加配对 Reward/day、各时间范围成功率与成功样本等待时间差；随机分位数不解释为市场置信区间。

`watch_fg_abcd.py` 汇总真实运行状态与已完成评估，不触发训练。smoke 工具自动保存恢复前账户快照，审计工具默认从同一 smoke 目录读取；既有 v2 运行按原快照路径再次审计 PASS。

早期 gate 未通过的实测证据：10310 步评估的 posterior accuracy 为 0.994099，门槛 0.995；posterior exact accuracy 为 0.988199，门槛 0.990。其余四项当次达标，但要求六项同时连续五次通过。此记录说明当次未通过原因，不预测后续收敛，也不调整门槛。

# FG ABC 开发记录 V1

最新状态：v3_shared_prefix 已完成代码与非模型准备验收，所有模型训练保持停止；下方 v1/v2 内容为历史记录。详见 FG_DreamerV3_ABCD_Checkpoint_Reuse_Plan_V1.md。

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

## 用户评审后的顺序训练版本

以上 v1 为历史开发/运行记录。用户要求停止训练后，实际训练进程及自动启动器均退出；v1 A 标记 STOPPED_BY_USER，已有最后 checkpoint 为 45060 步，B/C 没有启动。停止不会删除日志、Replay、账户状态或模型。

数量审计确认四个 fold 的正式 A100万/B100万/C70万+30万预算，在正常完整生命周期推进条件下均覆盖至少一轮；不再要求10万先导均衡覆盖历史。FG409 一轮全历史需 488629 交互步，近期需 136690 交互步，包含原同合约 chunk 的 reset 观察。

真实末根执行探针：18 个训练合约 × LONG/SHORT，共36场景全部强退6手、零剩余仓位、零速度Reward。终局无成交量的回归仍明确阻止 reset，不免费清除亏损；保证金不足提前终止不计作完整生命周期覆盖。数量容量不是模型学习能力证明。

按照用户要求，统一 `abcd_comparison.yaml` 默认 `scheduler.mode=sequential`，生成独立 `artifacts/fg_abcd_v1/v2_sequential/` 和 `/data/logdir/fg_abcd_v1/v2_sequential/`。先导与正式都不生成轮转 selector；旧 v1 文件与可选轮转代码保留。业务撮合、Reward、Obs、Action、模型与 Gate 未改。

覆盖报告兼容顺序 schema1：结束账户记录实际推进量；按每次真实访问前缀计算各合约、年份和方向暴露，不能把多次提前风险退出拼成完整遍历。只保存当前正常账户及既有完成指标，不增加独立账户轮转调度器。顺序恢复明确拒绝 schema2 checkpoint。

针对性回归4项、相关回归43项通过；新增恢复 schema2 拒绝断言及早退覆盖断言再次通过。新版本 GPU smoke、100k先导与正式1M均未启动，不以旧轮转 smoke 冒充新版训练验收。

新版清单哈希 `47731150954d74ec65f08448d777ac87a34f072b48c8f88ae504e54765971adf`，39个方法实例、52个阶段，全部52阶段通过FG准入检查，8个训练selector均无轮转调度配置。新版命令已生成；正式 `--execute` 仍在GPU初始化前拒绝。生成数据、清单、结果和命令文件不纳入Git。

## 固定步数与共享前缀阶段（2026-10-10）

已实现 FG_FULL_V1：普通恢复保留完整学习/账户状态；C 共享 A 精确 70K/700K 源节点、重建近期账户/Replay/统计。保存真实预取批次和 Agent 异步输出；非模型运行复现取样竞争后，新增 FG 专用确定性预取屏障，端到端回归通过。

Gate 检查固定 5000 步，保存固定 10000 步，关键节点精确停止；常规节点仅保留两个，旧实验不清理。生成 v3 清单、命令与报告，52 阶段准入及共享身份检查 PASS；37 条唯一线路承载 39 个逻辑方法实例。所有先导、正式、新 GPU smoke 均 NOT_RUN，最终盲测未读取。

## 2026-10-11：对齐整份计划并继续验收

用户明确正式训练自行启动，其他阶段允许继续。P0/P1/P2 的工具准备与非神经网络测试不代替 P3/P4 运行验收；实施计划第 11 节列出完整剩余清单。C 源节点身份原来未区分 70K/700K，现绑定显式源步数与迁移凭据哈希，错误分支拒绝。正式 runner 解锁用户显式 --execute，本轮不调用正式训练。

相关 62 项测试通过，两个仓库 diff --check 通过；准备审计 52 阶段、37 唯一线路通过。GPU smoke 目录 `/data/logdir/fg_abcd_v1/v3_shared_prefix/smoke_full_v1`，实际观察 Policy/Train 均 cuda:0；当前仍在运行，不能作为 PASS。生成产物不提交。

后台验收已于 00:07 启动，PID 334485，日志 `/data/logdir/fg_abcd_v1/v3_shared_prefix/acceptance.log`。`tools/continue_fg_abcd_acceptance.py` 仅包含先导启动路径：等 smoke 完成→检查完整快照→A/B/C 先导→按真实 Gate 决定模型评估→刷新状态；失败保留产物并停止，无正式自动启动和自动重训。监督器独立会话运行，关闭对话不会依赖当前终端存活。

实时结果见生成的 `artifacts/fg_abcd_v1/v3_shared_prefix/acceptance_status.json`、`acceptance_progress.md`、`pilot_results.md` 与 `comparison.md`。新增启动边界回归与相关测试 11 项通过。补齐运行开始阶段记录及实时指标进度读取，避免新版运行时显示 NOT_RUN。普通真实 GPU 恢复已经观察到 1110→1120，C 920 步仍在运行，P3 尚不能标记完成。

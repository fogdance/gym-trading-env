# FG ABC 顺序训练覆盖与终局审计 V1

日期：2026-10-10。先完成数量与终局审计；随后用户停止 v1 轮转先导，并要求先导/正式都恢复顺序模式。新版本为 v2_sequential，准备但未启动；正式 1M 仍 NOT_RUN。

## 结论与范围

四个 fold 的 A 100 万步、B 100 万步、C 全历史 70 万步、C 近期 30 万步，在顺序训练且正常推进到各主力退出边界时，全部足够完整遍历至少一次。没有发现为正式预算覆盖而必须启用跨合约 1024 根轮转的数量依据。

覆盖代表真实 K 线进入环境交互，不代表模型完整学会历史、每根行情都被 Replay 均衡采样、Gate 必然通过或模型优于规则。10 万步先导不要求完整覆盖历史。

审计输入是已冻结的 `artifacts/fg_abcd_v1/v1/experiment_manifest.json` 与对应训练文件；脚本校验清单、行情和 Env 配置哈希及实际行数，不读取封存盲测行情。不同 seed 的数据量相同，因此表格只计算 seed0。

## 环境交互步的实际口径

`embodied/core/driver.py` 每次收到一条 Env Obs 都累计一步，reset 返回的首条 Obs 也算一步，但不推进行情。

固定合约 Env 从第 0 行观察开始，第 1 行到最后一行共推进 `N-1` 根。原有 `entry.chunk_steps=1024` 仍会划分 episode；顺序模式遇到 CHUNK_END 在同一合约、同一账户继续。这个原有 chunk 不是本次新增的跨合约轮转。

保留原有同合约 chunk 时，正常完整生命周期的交互数为：

```text
行情推进量 = N - 1
首条观察与 chunk 恢复观察数 = ceil((N - 1) / 1024)
完整遍历环境步 = sum((N - 1) + ceil((N - 1) / 1024))
```

最后一根同时到达主力退出时，MAIN_ROLL 优先于 CHUNK_END，因此不会额外再产生同合约空 chunk。下一合约的 reset 计入下一合约首条观察。

## 最新 FG409 fold 的四阶段结果

| 阶段 | 合约数 | 原始 K 线数 | 实际可推进根数 | reset 交互数/轮 | 一轮交互步 | 预算 | 完整轮数 | 后续剩余交互步 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A 全历史 | 18 | 488160 | 488142 | 487 | 488629 | 1000000 | 2 | 22742 |
| B 最近24月 | 5 | 136558 | 136553 | 137 | 136690 | 1000000 | 7 | 43170 |
| C 全历史预训练 | 18 | 488160 | 488142 | 487 | 488629 | 700000 | 1 | 211371 |
| C 近期强化 | 5 | 136558 | 136553 | 137 | 136690 | 300000 | 2 | 26620 |

对应轮数比值分别约 2.0465、7.3158、1.4326、2.1947。剩余交互步包含 reset 观察，不能直接称为剩余行情根数。C 的两个阶段独立计算，近期账户与 Replay 按原 C 迁移协议建立，不继承预训练交易账户。

## 所有开发 fold 的结果

| 验证 fold | A 合约数 | A 一轮交互步 | A100万轮数 | B 合约数 | B 一轮交互步 | B100万轮数 | C70万全历史轮数 | C30万近期轮数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FG205 | 11 | 296716 | 3.3702 | 6 | 158132 | 6.3238 | 2.3592 | 1.8971 |
| FG305 | 14 | 379669 | 2.6339 | 6 | 164857 | 6.0659 | 1.8437 | 1.8198 |
| FG405 | 17 | 462174 | 2.1637 | 6 | 165458 | 6.0438 | 1.5146 | 1.8131 |
| FG409 | 18 | 488629 | 2.0465 | 5 | 136690 | 7.3158 | 1.4326 | 2.1947 |

此表证明预算容量充足，不是已执行 100 万步的训练覆盖记录。正式运行仍需保存真实访问合约、推进根数、终局原因与更新次数。

## 合约终局与亏损保留

代码证据：`entry_env.py::step()` 主力边界设置 MAIN_ROLL，优先强制退出而不发速度 Reward。真实模拟成交通过现有账务提交，费用和损益进入账户。只有 `entries` 为空才能 terminated。

`entry_dataset_env.py::reset()` 顺序模式只在 terminated 后切到下一合约。CHUNK_END 继续同一账户，不清空仓位、浮亏或成交记录。

真实训练数据的 18 个唯一合约末根均与配置的 main_exit_at 一致；bob/eob 时长均为 1 分钟；末根 Volume 最低 128，最高 10503，均大于最大持仓 6 手。即使末根新开 3 手，剩余 Volume 也大于最大 6 手。当前模拟成交规则下，没有在这些真实末根发现导致退出不足的时钟或成交量条件。

**边界例外不能隐藏：** 最后一根无法成交时，Env 保留未平仓头寸；随后 DATA_END 截断，reset 抛出 `Reset would discard an unfinished account`。这会明确中止训练，而不是免费清除亏损。不能笼统承诺“任何输入都不会卡住”。回归专门验证该失败保护。

保证金不足也可能在生命周期中途触发 MARGIN_CALL，实际结算后提前终止并切下一个合约。这会使部分尾部行情没有被该次账户访问，不能将预算容量推导当作无条件全历史覆盖保证。当前审计不是所有可能动作轨迹的证明，也没有为了覆盖而修改风险退出协议。

顺序模式结束账户目前只把 `_get_info()` 指标放入 `completed`，当前账户保存在 schema1 checkpoint；没有保存每个历史结束账户的完整账簿快照。换合约新建账户发生在真实退出后，属于既有训练行为，不是丢弃未平仓浮亏。但若要求逐笔追溯所有已结束训练账户，仍需要独立的归档能力，不能把汇总指标称为完整历史账户归档。

## 验证与复现

```bash
conda activate dreamerv3
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/fg-matplotlib
python tools/audit_fg_abcd_sequential.py
python -m pytest -q \
  tests/entry_v1/test_fg_abcd.py::test_sequential_dataset_refuses_to_discard_unfilled_terminal_loss \
  tests/entry_v1/test_entry_contract.py::test_dataset_switches_only_after_real_exit_and_restores_chunk \
  tests/entry_v1/test_entry_contract.py::test_main_roll_closes_old_contract_loss_without_reward \
  tests/entry_v1/test_entry_contract.py::test_expiry_and_margin_exit_zero_reward_including_insolvency
```

上述四项回归已通过，覆盖同合约 chunk 恢复、真实平仓后才切换、强退亏损零 Reward、到期与保证金不足退出、无法成交时拒绝 reset 且账户状态不变。`--terminal-probe` 使用真实倒数第二根价格构造 6 手已有持仓并执行最后一根，分别检查 LONG/SHORT；它只验证终局撮合，不冒充完整历史策略回放，不训练任何模型。

最初 v1 的真实末根探针已执行：18 个合约 × LONG/SHORT 共 36 个场景全部通过，每个均真实模拟强退 6 手、零速度 Reward、零剩余手数。该证据只针对终局路径，不是整段策略回放。原结果 `artifacts/fg_abcd_v1/v1/sequential_budget_audit.json` 保留；新版默认输出 `artifacts/fg_abcd_v1/v2_sequential/sequential_budget_audit.json`，生成文件不提交。

## 后续实施边界

数量审计支持恢复顺序模式作为默认，不要求删除已开发的可选轮转功能。不要原地修改正在运行先导的冻结清单或将轮转 schema2 checkpoint 恢复为顺序 schema1。

切换使用独立版本的顺序配置与日志，保留旧先导结果及 checkpoint。切换前 `run_fg_abcd.py::training_coverage()` 只接受 schema2，本次已适配 schema1 和实际已完成访问前缀，避免顺序训练预算结束时报告失败；这是工具适配，不是更改 Env 交易规则。

按用户后续指令，v1 A 已停止，最后已有 checkpoint 为 45060 步，B/C 未启动；独立 v2 顺序模式使用 schema1，覆盖工具已适配真实已完成/当前访问前缀。完整历史账簿归档不是本次新增功能，当前仍保留既有正常账户延续与指标记录。新先导和正式训练本轮均不启动。

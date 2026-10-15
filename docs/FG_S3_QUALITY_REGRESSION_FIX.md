# 当前合同回归清单与已确认根因

## Summary

本轮现有合同回归包含下面 6 组、34 项 CPU 测试。过时测试夹具已按当前账户合同更新，并补充奖励不一致的拒绝断言。没有为旧用例增加生产兼容分支；生产评价器继续执行完整账户与收据对账。历史收据文件名中的 compatibility 保留执行时名称。

| 文件（相对项目根） | 项数 | 保护的合同 |
| --- | ---: | --- |
| `tests/unit/test_fg_reward_candidates.py` | 13 | 原 Reward 曲线、早期逆行、元数据不计时、容量、原子校验、chunk 恢复及折扣对账 |
| `tests/entry_v1/test_fg_reward_offline.py` | 7 | 原 Env／离线 Reward 一致、市场时钟、cutoff／拒单区分、冻结来源与 Replay 复用 |
| `tests/entry_v1/test_fg_s3_v2_pool.py` | 5 | 多合约账户与 Checkpoint 归属、预算授权、B2 Outcome 调度／恢复、beta 按真实 Env 推进 |
| `tests/entry_v1/test_fg_s3_v2.py` | 4 | 已提交 Replay 归属、源码数学／CUDA 参数审计、因果决策与过期状态拒绝、恢复日志归档 |
| `tests/entry_v1/test_fg_s3_v2_observation.py` | 3 | Actor 提议与教师接管分离、部分日志、右删失分母、奖励对账、冻结工具入口 |
| `tests/entry_v1/test_fg_s3_tutor_e2.py` | 2 | E2 动态类归一化、stop-gradient、权重范围、缺类／空监督与旧版本等价 |

GPU 专项未运行；该清单不代表 GPU 数值或性能认证。

## Evidence observed

修复前在开发树运行 34 项：33 通过、1 失败，见 [原始失败 XML](receipts/quality_first_delivery_shared_compatibility.xml)。同一失败已在干净 `integration/s3-next@c3a808437ef277ee87f25b0ae8803e31fa5dca74` 复现，见 [基线 XML](receipts/preexisting_actor_metrics_c3a8084.xml)。失败原文为 `ValueError: Evaluation account reward differs from receipt`。

真实 Env 的 `entry_env.py:export_state()` 始终导出 `total_reward`；共享 `fg_s3_v2_evaluation.py` 同时从该账户写 `account_state.json`、从 `env.total_reward` 写收据。实际 DEV 和 FG401 留出账户的两个字段一致，见 [原生账户核对](receipts/quality_first_compatibility_real_account_check.json)。

## Root cause

`git blame` 确认：提交 `34023e7f`（2026-10-14）在 `fg_s3_v2_observation.py:97` 接入 `evaluation_metrics(state, receipt)`；同一提交的 `fg_s3_eval_metrics.py:57–58` 要求 `state.total_reward == receipt.reward`。测试 `test_actor_only_rates_include_censored_entries_and_reject_guidance` 的夹具仍来自 `19e72f88`（2026-10-13），只写 `attributions`，同时在收据声明 `reward=0`。因此实际比较为 `None != 0`，在进入原本要检查的 TP／SL 分母统计前就失败。根因是测试夹具未随真实账户合同更新。

## Changes made

仅修改该测试：账户夹具增加 `total_reward=0.`，并确认将收据 reward 改为 1 后仍抛出奖励对账错误。原有右删失分母、末尾未执行提议、教师接管拒绝的断言保留。

## Verification performed

目标测试已通过，见 [目标测试 XML](receipts/quality_first_compatibility_fix_targeted.xml)。全量 34 项重跑结果见 [修复后 XML](receipts/quality_first_compatibility_fixed.xml)；测试前后的收据分别保存，没有改写失败历史。

## Risks / not verified

此次修复只涉及测试夹具，运行源码哈希保持一致。没有改动原 Reward、模型、训练工件或放宽生产对账。GPU、10K、1M 均未启动。

## Next recommended step

将修复后回归收据纳入 Quality-first 最终交付索引，继续以真实留出预测结果决定后续预算与 Actor 准入。

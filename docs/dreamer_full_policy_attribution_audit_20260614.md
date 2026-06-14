# Dreamer Full-Policy Attribution Audit

日期：2026-06-14

状态：Phase 3.5 审计结果。不开启 Entry-only Dreamer，不新增 ATR/MACD/日线等 observation。

## 目标

本阶段只回答一个问题：

```text
988576 step 的无 MACD Dreamer checkpoint 到底靠什么赚钱：
1. 可泛化入场选择；
2. 退出 / 持仓 / 路径管理；
3. env execution timing 与 fixed evaluator 口径差异；
4. 或者只是 sampled episode 偶然性。
```

使用 checkpoint：

```text
/data/logdir/action-mask-formal-202606122100/ckpt/20260613T054804F270056
step = 988576
market_seq = 60 x 18，无 MACD
```

使用 entry evaluator 数据：

```text
configs/entry_eval_jm_dreamer6m_2024_v1.yaml
artifacts/entry_eval/entry_eval_jm_dreamer6m_2024_v1/
```

## 已实现

新增审计工具：

```text
tools/dreamer_checkpoint_audit.py
```

新增能力：

```text
1. sampled episode audit
2. per-day deterministic replay
3. policy action 对应的 decision metadata 采集
4. actual trade 与 fixed evaluator 的 entry-time / decision-time join
5. per-trade attribution:
   actual_net_pnl
   fixed_net_pnl
   exit_holding_delta_pnl = actual_net_pnl - fixed_net_pnl
   fixed_mfe_gross
   fixed_mae_gross
6. matched-random fixed-exit baseline，按同 split / 同交易数 / 非重叠约束统计
7. train / validation / test split 报告
```

## per-day 审计命令

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/dreamer_checkpoint_audit.py \
  --run-logdir /data/logdir/action-mask-formal-202606122100 \
  --checkpoint /data/logdir/action-mask-formal-202606122100/ckpt/latest \
  --entry-eval-dir artifacts/entry_eval/entry_eval_jm_dreamer6m_2024_v1 \
  --collect-mode per_day \
  --roles train,validation,test \
  --episodes 0 \
  --max-steps 120000 \
  --random-runs 200 \
  --output-dir artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday \
  --collect \
  --jax-platform cuda
```

输出：

```text
artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday/report.md
artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday/summary.json
artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday/dreamer_actual_trades.csv
artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday/dreamer_fixed_exit_entry_join.csv
artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday/dreamer_fixed_exit_decision_join.csv
artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday/dreamer_fixed_exit_strategy.csv
```

## 关键结果

per-day deterministic replay 覆盖：

```text
train: 78 days
validation: 20 days
test: 19 days
```

Phase 3.5A 后，主 attribution 口径已经从 loose timestamp join 改为：

```text
Dreamer decision_row + direction -> entry_eval candidate_id
```

entry-time / decision-time timestamp join 只保留为 timing diagnostics。

总体结果：

| 口径 | Trades | Net PnL | Expectancy | Profit Factor |
|---|---:|---:|---:|---:|
| Dreamer actual entry + Dreamer actual exit | 343 | 91512 | 266.80 | 7.02 |
| Candidate exact fixed-exit | 342 | 24858 | 72.68 | 1.53 |
| Entry-time fixed-exit diagnostic | 227 | 5838 | 25.72 | 1.17 |
| Executable non-overlap from candidate join | 224 | 14856 | 66.32 | 1.46 |
| matched-random same-count fixed exit | 224 | -8857.8 mean | NA | NA |
| constrained oracle fixed exit | 432 | 191658 | 443.65 | inf |

Split 结果：

| Split | Actual Net | Candidate Fixed Net | Executable Fixed Net | Matched Random Mean |
|---|---:|---:|---:|---:|
| train | 62022 | 20598 | 11160 | -6375.3 |
| validation | 21672 | 3252 | 4368 | -1088.85 |
| test | 7818 | 1008 | -672 | -1393.65 |

Matched-random by split：

| Split | Trade Count Mean | Net PnL Mean | P05 | P50 | P95 |
|---|---:|---:|---:|---:|---:|
| train | 145 | -6375.3 | -12852 | -6480 | 1171.5 |
| validation | 47 | -1088.85 | -5112 | -882 | 2668.5 |
| test | 32 | -1393.65 | -4275 | -1287 | 1107 |

## 判断

结论：

```text
Full-policy Dreamer: PROMISING
Candidate-level entry signal: PLAUSIBLE_BUT_WEAK
Executable entry gate: NOT PASS
Entry-only Dreamer admission: NOT PASS
Primary driver: EXIT/HOLDING + quota-consuming behavior
```

理由：

```text
1. per-day deterministic replay 下，actual Dreamer policy 在 train / validation / test 都为正。
2. candidate exact fixed-exit 在 train / validation / test 都为正。
3. candidate exact fixed-exit overall 明显优于 matched-random。
4. 但非重叠 executable fixed-exit 在 test 为 -672，仍未通过样本外可执行入场门槛。
5. test 里 95% 交易日开满 3 笔，Dreamer 仍像 quota-consuming policy，而不是高选择性入场器。
6. actual test +7818 中，candidate fixed-exit 只解释 +1008，剩余 +6810 主要来自 exit / holding / path。
```

因此当前不能放开 Entry-only Dreamer。这个 checkpoint 证明完整 RL policy 学到了有价值行为，但还没有证明“可执行固定退出入场能力”样本外稳定成立。

## Timing 风险

当前 join 命中：

```text
candidate exact join: 342 / 343
entry-time join: 227 / 343
decision-time join: 342 / 343
```

这说明 fixed evaluator 的语义是：

```text
decision timestamp t -> next valid 1m open entry
```

而当前 Dreamer env trade record 体现为：

```text
policy action 在当前 env step 直接成交
```

所以 entry-time join 会漏掉大量当前 bar 成交，尤其 session open 附近和 flip open。candidate exact join 回到“同一个 observation decision point + same direction”，是本轮 attribution 的主口径；entry-time join 只用于暴露旧 env execution timing 与 canonical next-open evaluator 的差异。

## 下一步

优先级：

```text
P0: 设计 env execution timing 契约迁移：legacy_current_step 只用于解释旧 checkpoint，canonical_next_open 用于未来训练和正式验收。
P1: 增加 checkpoint retention，未来训练保留 500k / 700k / 800k / 900k / latest。
P1: 对 margin termination 单独统计，确认 actual profit 是否伴随不可接受的风险路径。
P2: 若 timing 对齐后 test fixed-exit 仍不稳，则不要进入 Entry-only Dreamer，转向 exit/holding policy 研究或 Phase 4 单特征 ablation。
```

禁止事项仍然不变：

```text
不要进入 Entry-only Dreamer。
不要新增 ATR / MACD / daily trend observation。
不要实现智能退出、反手增强、加仓、仓位管理。
```

## Phase 3.5A Attribution Hardening

状态：已完成第一轮，不修改 env timing 主契约。

本轮修正：

```text
1. 主 attribution 从 loose timestamp join 改为 decision_row + direction exact join。
2. 每笔交易输出 candidate_id 级别审计文件：
   artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday/dreamer_candidate_attribution.csv
3. entry-time / decision-time join 降级为 timing diagnostics。
4. 修正 attribution 表头：
   actual_all_net
   actual_matched_net
   fixed_net
   unmatched_actual_net
   delta_on_matched_trades
5. trades/day 分布按 split 真实天数做分母。
6. 新增风险路径摘要：
   negative_return_episodes
   daily_quota_episode_ratio
   largest winning/losing trade
   fixed MFE/MAE
```

更新后的主口径：

```text
candidate exact join = Dreamer decision_row + direction -> entry_eval candidate_id
```

这个口径回答的是：

```text
在同一个 observation decision point 上，
如果按照 canonical fixed evaluator 的 next-open + fixed-exit 规则处理，
Dreamer 选择的方向是否有价值。
```

它不要求旧 env 的 actual_entry_timestamp 等于 evaluator 的 next-open timestamp。

### Phase 3.5A 结果

总体：

| 口径 | Trades | Net PnL | Expectancy | Profit Factor |
|---|---:|---:|---:|---:|
| Dreamer actual entry + actual exit | 343 | 91512 | 266.80 | 7.02 |
| Candidate exact fixed-exit | 342 | 24858 | 72.68 | 1.53 |
| Entry-time fixed-exit diagnostic | 227 | 5838 | 25.72 | 1.17 |
| Executable non-overlap from candidate join | 224 | 14856 | 66.32 | 1.46 |
| Matched random same-count | 224 | -8857.8 mean | NA | NA |

Split：

| Split | Actual Net | Candidate Fixed Net | Executable Fixed Net | Matched Random Mean |
|---|---:|---:|---:|---:|
| train | 62022 | 20598 | 11160 | -6375.3 |
| validation | 21672 | 3252 | 4368 | -1088.85 |
| test | 7818 | 1008 | -672 | -1393.65 |

Attribution：

| Split | Actual All Net | Candidate Fixed Net | Delta On Matched |
|---|---:|---:|---:|
| train | 62022 | 20598 | 41640 |
| validation | 21672 | 3252 | 18420 |
| test | 7818 | 1008 | 6810 |

Trades/day：

| Split | Mean Trades/Day | 0/day | 1/day | 2/day | 3/day |
|---|---:|---:|---:|---:|---:|
| train | 2.71 | 7.14% | 3.57% | 0.00% | 89.29% |
| validation | 2.90 | 0.00% | 5.00% | 0.00% | 95.00% |
| test | 2.85 | 5.00% | 0.00% | 0.00% | 95.00% |

风险路径：

| Split | Negative Episodes | Daily Quota Episode Ratio | Largest Losing Trade |
|---|---:|---:|---:|
| train | 9 / 78 | 96.15% | -1596 |
| validation | 2 / 20 | 95.00% | -1386 |
| test | 3 / 19 | 100.00% | -1026 |

### Phase 3.5A 判断

结论更新为：

```text
Full-policy Dreamer: PROMISING
Candidate-level entry signal: PLAUSIBLE_BUT_WEAK
Executable entry gate: NOT PASS
Entry-only Dreamer admission: NOT PASS
Primary driver: EXIT/HOLDING + quota-consuming behavior
```

理由：

```text
1. candidate exact fixed-exit 在 train / validation / test 都为正。
2. candidate exact fixed-exit overall 明显优于 matched random。
3. 但非重叠 executable fixed-exit 在 test 为 -672，仍未通过样本外可执行入场门槛。
4. test 里 95% 交易日开满 3 笔，Dreamer 仍像 quota-consuming policy，而不是高选择性入场器。
5. actual test +7818 中，candidate fixed-exit 只解释 +1008，剩余 +6810 来自 exit / holding / path。
```

因此，当前不能进入 Entry-only Dreamer。

下一步仍然是 timing 契约迁移设计：

```text
legacy_current_step:
  只用于解释旧 checkpoint。

canonical_next_open:
  observation at completed t
  action decided at t
  entry executed at next valid 1m open
  未来训练和正式验收唯一口径。
```

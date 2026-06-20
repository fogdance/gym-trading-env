# JM8Y Regime-Aware Validation Score Contract

Date: 2026-06-20

Status:

```text
VALIDATION_SCORE_FORMULA_FROZEN_V1
TRAINING_NOT_STARTED
TEST_NOT_USED
```

This document freezes the checkpoint-selection score for the next JM8Y Dreamer pilot. It is a validation-only contract. It must be fixed before training starts and must not be changed after seeing test results.

## Scope

This contract applies to:

```text
product: JM
split: jm8y_wf_v1
train: 2018-01-02 .. 2022-12-30
validation: 2023-01-03 .. 2024-12-31
test: 2025-01-02 .. 2025-12-02
execution: signal_on_close_plus_spread
obs: obs_v2
reward: unchanged for the experiment
action space: SHORT / FLAT / LONG
```

The source file is user-confirmed as JM main-continuous contract data. This score does not prove single-contract performance and does not remove the need to report roll-boundary candidate windows.

## Inputs

For each retained checkpoint, run validation-only deterministic per-day audit first. The score script consumes:

```text
validation audit directory:
  summary.json
  dreamer_actual_trades.csv
  dreamer_fixed_exit_strategy.csv
  oracle_fixed_exit_trades.csv

regime day table:
  /data/logdir/trading_contracts/jm8y_wf_v1/artifacts/dataset_audit/jm8y_wf_v1/regime_day_table.csv
```

The scoring script is:

```text
tools/score_regime_aware_validation.py
```

The script must be run only with:

```text
--split-role validation
```

It must not be run on test artifacts for checkpoint selection.

## Regime Groups

The score uses the already frozen dataset-side regime labels:

```text
major_regime
structure_regime
```

Groups with fewer than 5 validation days are reported but excluded from score aggregation. `WARMUP` groups are excluded from score aggregation.

## Formula

For each eligible regime group `g`:

```text
actual_ratio_g =
  clip(sum(actual_net_pnl_g) / max(abs(sum(oracle_fixed_exit_net_pnl_g)), 1), -1, 1)

fixed_ratio_g =
  clip(sum(dreamer_fixed_exit_executable_net_pnl_g) / max(abs(sum(oracle_fixed_exit_net_pnl_g)), 1), -1, 1)
```

Aggregate components:

```text
actual_regime_balance =
  mean(actual_ratio_g over eligible major_regime and structure_regime groups)

fixed_entry_regime_balance =
  mean(fixed_ratio_g over eligible major_regime and structure_regime groups)

matched_random_margin =
  clip(
    (actual_net_pnl - matched_random_net_pnl_mean)
    / max(abs(matched_random_net_pnl_mean), abs(oracle_net_pnl) * 0.10, 1),
    -1,
    1
  )

drawdown_penalty =
  clip(abs(validation_daily_actual_max_drawdown) / max(abs(oracle_net_pnl) * 0.10, 10000), 0, 1)

regime_instability_penalty =
  std(actual_ratio_g over eligible major_regime and structure_regime groups)

extreme_no_trade_penalty =
  clip((no_trade_day_ratio - 0.80) / 0.20, 0, 1)
```

Final score:

```text
validation_score =
  1000 * (
    0.50 * actual_regime_balance
  + 0.25 * fixed_entry_regime_balance
  + 0.15 * matched_random_margin
  - 0.05 * drawdown_penalty
  - 0.03 * regime_instability_penalty
  - 0.02 * extreme_no_trade_penalty
  )
```

Higher is better.

## Why These Components

`actual_regime_balance` is the primary full-policy objective. It prevents a checkpoint that only works in one validation regime from being selected only because total validation PnL is high.

`fixed_entry_regime_balance` is a secondary attribution check. It does not require Entry-only admission, but it penalizes checkpoints whose validation profit is entirely disconnected from their entry points.

`matched_random_margin` requires the full policy to beat a same-count random reference on validation.

`drawdown_penalty` prevents selecting a checkpoint whose validation profit is only achieved through large underwater periods.

`regime_instability_penalty` penalizes large dispersion across validation regimes.

`extreme_no_trade_penalty` prevents degenerate no-trade checkpoint selection.

## Selection Rule

For each retained checkpoint:

```text
1. run validation-only deterministic per-day audit;
2. run tools/score_regime_aware_validation.py;
3. select the checkpoint with the highest validation_score;
4. run test exactly once only for the selected checkpoint.
```

Tie-breakers, in order:

```text
1. higher actual_regime_balance;
2. higher matched_random_margin;
3. lower drawdown_penalty;
4. fewer missing / excluded regime groups;
5. earlier checkpoint step.
```

## Required Output

For every checkpoint:

```text
validation_score_components.csv
validation_score_by_regime.csv
validation_score_daily_pnl.csv
summary.json
report.md
```

For the whole experiment, create a selection table:

```text
checkpoint
validation_score
actual_regime_balance
fixed_entry_regime_balance
matched_random_margin
drawdown_penalty
regime_instability_penalty
extreme_no_trade_penalty
actual_net_pnl
fixed_executable_net_pnl
matched_random_net_pnl_mean
oracle_net_pnl
selected true/false
```

## Prohibitions

Do not:

```text
use test data to choose weights;
change the formula after seeing test;
select by total validation PnL only;
select by train score;
select by TensorBoard score alone;
ignore fixed-exit attribution;
ignore matched-random reference;
ignore drawdown;
silently include WARMUP or tiny regime groups in the score;
rename obs_v2 into extra sub-versions in this contract.
```

## Current Status

Dataset-side gates completed:

```text
continuous-contract audit: user-confirmed JM main-continuous
contract root: ready
entry_eval dataset: ready
oracle by regime: ready
supervised fixed-exit baseline: Gate 2 FAIL
```

Remaining before Dreamer pilot:

```text
obs_v2 train-only refit/audit path
validation audit commands for retained checkpoints
selection table generated by this frozen score
```

Final classification:

```text
VALIDATION_SCORE_FORMULA_FROZEN_V1
JM8Y_DREAMER_TRAINING_STILL_BLOCKED_BY_OBS_AUDIT
```

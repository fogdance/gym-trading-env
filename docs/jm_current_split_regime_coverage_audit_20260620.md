# JM Current Split Regime Coverage Audit

Date: 2026-06-20

Scope: audit the current JM walk-forward dataset split used by `jm_walk_forward_20240603_20251202`. This is a data split / regime coverage audit only. It does not train, evaluate, tune, or change any trading logic.

## Inputs

```text
split_manifest:
/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202/artifacts/walk_forward_splits/jm_walk_forward_20240603_20251202/split_manifest.json

raw 1m:
/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202/data/raw/jm2601_18m_1m.csv

prior daily trend report:
docs/jm_train_validation_test_daily_trend_20260619.md

latest current obs_v2 audit:
/data/logdir/audits/action-mask-jm-v2-seed-c-20260619-retry4/aggregate_report.md
```

## Final Answer

The current split is chronological, but it is not a good final test of cross-regime Dreamer generalization. The train period is dominated by a long downtrend with a late sharp rebound, validation is a short high-volatility rebound / pullback month, and test is a mixed range regime with a later down leg. This creates a material regime mismatch between what the model mostly trains on and what it must validate/test on.

The current failed v2 seed-c run should be kept as a valid failed sample for this narrow fold, but it should not be used to conclude that JM 8Y data cannot train a generalized policy. A more appropriate next experiment is a JM 8Y chronological regime-aware split whose train pool contains UP, DOWN, and RANGE regimes together.

## Current Split

| role | range | trading days | valid 1m rows |
| --- | --- | ---: | ---: |
| train | 2024-06-03..2025-07-31 | 284 | 96,540 |
| validation | 2025-08-01..2025-08-29 | 21 | 7,245 |
| test | 2025-09-01..2025-12-02 | 61 | 20,685 |

## Regime Summary

Regime labels below use the existing MA20 descriptive method from `docs/jm_train_validation_test_daily_trend_20260619.md`:

```text
UP:    close > MA20 * 1.005
DOWN:  close < MA20 * 0.995
RANGE: otherwise
```

These labels are for dataset audit only. They are not trading rules and were not used for training or checkpoint selection.

| role | close return | linear slope | UP days | DOWN days | RANGE days | insufficient |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | -38.37% | -2.82 points/day | 58 | 186 | 21 | 19 |
| validation | +5.35% | -1.53 points/day | 14 | 7 | 0 | 0 |
| test | -1.97% | -0.04 points/day | 25 | 34 | 2 | 0 |

Interpretation:

- Train is mostly a long downtrend. It does include a strong 2025-06/2025-07 rebound, but the dominant sample mass is DOWN.
- Validation is only 21 trading days and is high-volatility rebound / pullback behavior. It is too short and too regime-specific for robust checkpoint selection.
- Test is not a simple continuation of validation. It mixes 2025-09/10 strength with a clear 2025-11 selloff, so policy behavior selected on validation can fail for regime reasons.

## Monthly Structure

| role | month | return | high-level regime note |
| --- | --- | ---: | --- |
| train | 2024-06 | -7.25% | down |
| train | 2024-07 | -8.86% | down |
| train | 2024-08 | -4.23% | down |
| train | 2024-09 | +16.46% | rebound inside broader down regime |
| train | 2024-10 | -8.07% | volatile down |
| train | 2024-11 | -6.84% | down |
| train | 2024-12 | -5.92% | down |
| train | 2025-01 | -3.32% | weak/down |
| train | 2025-02 | -0.50% | range/weak |
| train | 2025-03 | -10.46% | down |
| train | 2025-04 | -7.69% | down |
| train | 2025-05 | -20.35% | strong down |
| train | 2025-06 | +14.74% | sharp rebound |
| train | 2025-07 | +28.36% | sharp rebound / squeeze |
| validation | 2025-08 | +5.35% | high-volatility rebound then pullback |
| test | 2025-09 | +0.67% | range |
| test | 2025-10 | +10.48% | rebound |
| test | 2025-11 | -16.93% | down |
| test | 2025-12 | +0.32% | too short to classify |

## Why This Matters For Dreamer

Dreamer does not just learn isolated entries. It learns a world model plus actor behavior involving entry, direction, holding, flipping, and exit timing. If train is dominated by one regime, the learned policy can become regime-specific.

Likely train-biased behaviors in this split:

```text
1. Learn short-side or trend-continuation behavior from the long downtrend.
2. Learn rebound-specific behavior from the late 2025-06/07 squeeze.
3. Overfit exit/holding/flip behavior to the train path shape.
4. Fail when validation/test alternate between rebound, range, and later downtrend.
```

This is consistent with the latest current obs_v2 audit:

| checkpoint | role | actual PnL | fixed-exit PnL | exit/holding delta | comment |
| --- | --- | ---: | ---: | ---: | --- |
| 500k | validation | -1296 | -1176 | -120 | fail |
| 700k | validation | -1776 | -966 | -810 | fail |
| 900k | validation | -1476 | +534 | -2010 | entry looked usable, exit/holding damaged it |
| latest | validation | -966 | -1176 | +210 | least bad validation actual |
| latest | test | -2568 | -6108 | +3540 | exit/holding reduced loss but still failed |

The important point is not only that test PnL is negative. It is that validation itself did not produce a positive full-policy checkpoint, and the split provides a weak basis for selecting a robust cross-regime policy.

## Assessment

### Chronology

PASS. The split is chronological:

```text
train < validation < test
```

There is no evidence in this audit that validation/test dates were included in training.

### Regime Coverage

PARTIAL. The train period technically contains UP, DOWN, and RANGE labels, but its distribution and sequence are not balanced:

```text
train DOWN days: 186 / 284
train UP days:    58 / 284
train RANGE days: 21 / 284
```

The model sees a long down regime first and a late explosive rebound. It does not see enough long, stable range behavior as a training base.

### Validation Robustness

FAIL. Validation is only 21 trading days and mostly one high-volatility transition month. It is not sufficient as the only checkpoint-selection filter for a full-policy Dreamer.

### Test Representativeness

MIXED. Test has range/rebound/down components, but because validation is short and different, a validation-selected checkpoint is not strongly protected against test regime shift.

## Recommendation For JM 8Y

JM has enough historical data to build a better experiment. The next formal split should use JM 8Y rather than the current narrow 2024-2025 window.

Recommended protocol:

```text
1. Aggregate JM 8Y to completed daily bars.
2. Label each trading day with descriptive regimes:
   UP / DOWN / RANGE / HIGH_VOL_TRANSITION.
3. Build chronological folds, not random splits.
4. Require train to contain meaningful counts of UP, DOWN, and RANGE.
5. Require validation to contain more than one regime, or use multiple validation sub-periods.
6. Keep test as future holdout and run it once.
7. Report every result by regime.
```

Training should mix regimes inside the same training process:

```text
Correct:
  train pool contains UP + DOWN + RANGE;
  reset sampling draws from the full train pool, optionally regime-balanced.

Avoid:
  train UP first, then RANGE, then DOWN.
```

Sequential regime training risks catastrophic forgetting and makes the final policy biased toward the last regime.

## Suggested JM 8Y Fold Design

If the 8Y data starts around 2018 and extends into 2025, use expanding chronological folds:

| fold | train | validation | test |
| --- | --- | --- | --- |
| fold_1 | 2018-2021 | 2022 | 2023 |
| fold_2 | 2018-2022 | 2023 | 2024 |
| fold_3 | 2018-2023 | 2024 | 2025 |

If training cost is high, start with one pilot:

```text
train:      2018-01-01..2023-12-31
validation: 2024-01-01..2024-12-31
test:       2025-01-01..2025-12-02
```

Before training, generate and review:

```text
regime_day_table.csv
regime_distribution_by_split.csv
split_manifest.json
split_visibility_report.md
```

## Final Classification

```text
CURRENT_SPLIT_CHRONOLOGICAL_PASS
CURRENT_SPLIT_REGIME_COVERAGE_PARTIAL
VALIDATION_SELECTION_WEAK
FORMAL_GENERALIZATION_CONCLUSION_NOT_FINAL
JM_8Y_REGIME_AWARE_SPLIT_REQUIRED
```

The current obs_v2 seed-c failure remains valid for this fold. However, because the fold is regime-imbalanced and validation is short, the next decision experiment should be built on JM 8Y with chronological regime-aware split construction before making a broader claim about Dreamer generalization on JM.

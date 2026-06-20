# JM 8Y Regime-Aware Split Design

Date: 2026-06-20

Scope: analyze `/home/v/Documents/work/dreamerv3/data/8Y_DCE_JM2601_1m.csv` and design a better train / validation / test protocol for Dreamer. This is a dataset and experiment-design report only. It does not train, evaluate checkpoints, change env behavior, change reward, or add indicators.

## Final Answer

The current narrow `2024-06-03..2025-12-02` split is chronological but regime-imbalanced. It trains mostly on a long downtrend, selects on a short high-volatility transition month, and tests on a mixed 2025 regime. That is a weak setup for testing Dreamer generalization.

The 8Y JM dataset is sufficient to fix the split-design problem. The recommended next split is:

```text
JM8Y-WF-V1

train:      2018-01-02 .. 2022-12-30
validation: 2023-01-03 .. 2024-12-31
test:       2025-01-02 .. 2025-12-02
```

This keeps strict chronology, gives the training pool five years of mixed UP / DOWN / RANGE regimes, gives validation two years with both balanced and severe down-regime behavior, and reserves 2025 as a future holdout.

Status:

```text
ACCEPT_AS_SPLIT_DESIGN_V1
JM8Y_MAIN_CONTINUOUS_DATA_ACCEPTED
DREAMER_TRAINING_PENDING_OBS_GATE
```

Training should mix all train regimes in one run through random episode starts from the train pool. Do not train sequentially as UP then RANGE then DOWN.

Before any Dreamer training, the main-continuous experiment scope, new split artifacts, obs fit parameters, observation audits, and validation-score formula must be frozen and reviewed.

## Data

```text
source:
/home/v/Documents/work/dreamerv3/data/8Y_DCE_JM2601_1m.csv

rows:
655,099 raw 1m rows

range:
2018-01-02 09:01:00 .. 2025-12-02 15:00:00

daily aggregation:
current futures strict 345-minute trading-day alignment

trading days:
1,921
```

The aggregation uses the repository futures session convention: night bars, if present, are assigned to the next trading day; daily bars are built from valid 1m rows only.

Important data caveat:

```text
The source file is named 8Y_DCE_JM2601_1m.csv, but it spans 2018..2025.
That cannot be treated as an ordinary single listed futures contract without audit.
```

Before training, this file must be classified as one of:

```text
1. actual single contract data;
2. main-contract continuous series;
3. back-adjusted continuous series;
4. manually stitched contract series;
5. unknown.
```

The split design assumes only that the file has enough historical JM-like price paths. It does not yet certify that the 8Y contract/roll semantics are safe for formal Dreamer training.

## Regime Labeling For Audit

These labels are descriptive and are used only for split design. They are not trading rules and must not be used to tune test decisions.

### MA20 Regime

```text
UP:    close > MA20 * 1.005
DOWN:  close < MA20 * 0.995
RANGE: otherwise
```

### Major 60-Day Regime

```text
UP:
  60-day return > +8%
  and 60-day log-close slope > 0

DOWN:
  60-day return < -8%
  and 60-day log-close slope < 0

HIGH_VOL_RANGE:
  not UP/DOWN
  and 20-day realized volatility is in the top quartile

RANGE:
  not UP/DOWN/HIGH_VOL_RANGE
```

### Structural Regime

Approximate causal daily swing structure:

```text
STRUCT_DOWN:
  last two confirmed swing highs are lower-high
  and last two confirmed swing lows are lower-low

STRUCT_UP:
  last two confirmed swing highs are higher-high
  and last two confirmed swing lows are higher-low

STRUCT_RANGE:
  otherwise
```

The swing structure in this report uses a small confirmation window and is for audit only. It is not currently part of observation.

## Yearly Regime Coverage

| year | days | return | MA20 UP | MA20 DOWN | MA20 RANGE | major UP | major DOWN | major RANGE | high-vol range | structure UP | structure DOWN | structure RANGE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 243 | -11.85% | 115 | 97 | 12 | 47 | 29 | 92 | 16 | 93 | 39 | 91 |
| 2019 | 244 | +1.66% | 102 | 96 | 46 | 42 | 71 | 131 | 0 | 82 | 73 | 89 |
| 2020 | 243 | +38.46% | 157 | 61 | 25 | 84 | 26 | 133 | 0 | 79 | 27 | 137 |
| 2021 | 243 | +32.02% | 145 | 82 | 16 | 154 | 39 | 20 | 30 | 110 | 62 | 71 |
| 2022 | 242 | -20.22% | 125 | 111 | 6 | 70 | 98 | 43 | 31 | 47 | 88 | 107 |
| 2023 | 242 | +3.34% | 112 | 115 | 15 | 85 | 86 | 63 | 8 | 80 | 66 | 96 |
| 2024 | 242 | -39.87% | 48 | 182 | 12 | 2 | 125 | 88 | 27 | 23 | 144 | 75 |
| 2025 | 222 | -6.60% | 79 | 127 | 16 | 68 | 120 | 33 | 1 | 58 | 85 | 79 |

Read this as regime availability, not as strategy performance.

Important observations:

- 2018-2022 already contains strong UP, DOWN, and RANGE exposure.
- 2023 is balanced and therefore useful for validation.
- 2024 is a severe down-regime stress year and therefore also useful for validation.
- 2025 is mixed and should be reserved as unseen holdout for the next decision experiment.

## Current Narrow Split Problem

Current split:

```text
train:      2024-06-03 .. 2025-07-31
validation: 2025-08-01 .. 2025-08-29
test:       2025-09-01 .. 2025-12-02
```

| role | days | return | major UP | major DOWN | major RANGE | high-vol range | structure UP | structure DOWN | structure RANGE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_train | 284 | -38.37% | 8 | 202 | 63 | 11 | 28 | 167 | 89 |
| current_validation | 21 | +5.35% | 21 | 0 | 0 | 0 | 19 | 0 | 2 |
| current_test | 61 | -1.97% | 39 | 0 | 22 | 0 | 21 | 9 | 31 |

This explains why the current fold is weak:

```text
train is dominated by down-regime behavior;
validation is only one short UP / transition month;
test is mostly UP/RANGE by 60-day regime but includes structural range and later weakness.
```

The model can learn a regime-specific policy and still fail validation/test. The failed seed-c v2 run remains a valid failure for this narrow fold, but it is not a final statement about 8Y JM trainability.

## Recommended Single Pilot: JM8Y-WF-V1

Use:

```text
train:      2018-01-02 .. 2022-12-30
validation: 2023-01-03 .. 2024-12-31
test:       2025-01-02 .. 2025-12-02
```

### Why This Split

The train pool is five years and contains all regimes:

| role | days | return | major UP | major DOWN | major RANGE | high-vol range | structure UP | structure DOWN | structure RANGE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train 2018-2022 | 1215 | +39.35% | 397 | 263 | 419 | 77 | 411 | 289 | 495 |

Validation combines a balanced year and a severe down year:

| role | days | return | major UP | major DOWN | major RANGE | high-vol range | structure UP | structure DOWN | structure RANGE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation 2023 | 242 | +3.34% | 85 | 86 | 63 | 8 | 80 | 66 | 96 |
| validation 2024 | 242 | -39.87% | 2 | 125 | 88 | 27 | 23 | 144 | 75 |
| validation 2023-2024 | 484 | -36.46% | 87 | 211 | 151 | 35 | 103 | 210 | 171 |

Test is future unseen 2025:

| role | days | return | major UP | major DOWN | major RANGE | high-vol range | structure UP | structure DOWN | structure RANGE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2025 | 222 | -6.60% | 68 | 120 | 33 | 1 | 58 | 85 | 79 |

This addresses the current data issue directly:

```text
1. Train no longer sees mostly one downtrend.
2. Validation is no longer a single short month.
3. Test remains a future holdout.
4. Evaluation can be reported by regime.
```

This split is a design recommendation, not a training authorization.

## Training Readiness Blockers

Do not start formal Dreamer training until all of these pass:

```text
1. JM8Y continuous-contract / roll audit
2. jm8y_wf_v1 contract root construction
3. split_manifest + regime_day_table + regime_distribution_by_split
4. obs_v2 fit params recomputed on train only
5. obs_v2 causality / formula / distribution audits rerun under jm8y_wf_v1
6. entry_eval oracle by regime
7. supervised fixed-exit baseline by regime
8. validation score formula frozen before training
```

Current dataset status:

```text
1. Continuous-contract audit: USER_CONFIRMED_MAIN_CONTINUOUS_CONTRACT.
2. Contract root construction: DONE.
3. Split/regime dataset audit: DONE.
4. Entry_eval dataset and oracle-by-regime audit: DONE.
5. Supervised fixed-exit baseline by regime: DONE, Gate 2 FAIL.
6. Regime-aware validation score formula: FROZEN_V1.
```

Current dataset reports:

```text
docs/jm8y_continuous_contract_roll_audit_20260620.md
docs/jm8y_wf_v1_contract_root_report_20260620.md
docs/jm8y_wf_v1_dataset_audit_20260620.md
docs/jm8y_wf_v1_entry_eval_dataset_audit_20260620.md
docs/jm8y_wf_v1_supervised_baseline_by_regime_20260620.md
```

Current split manifest hash:

```text
c84954dcc601ca7acdbb5216ea736ab3a88edbd328de8bc5e86e7293c58bdfd1
```

Current entry_eval opportunity result:

```text
candidates: 418147
outcomes: 836294

constrained oracle:
  train:      +998316 / 2379 trades
  validation: +362154 / 816 trades
  test:       +294264 / 666 trades

matched-random same-count mean:
  train:      -124288.2
  validation: -34839.6
  test:       -33846.0
```

Interpretation: fixed-exit opportunities exist across train / validation / test. This is an opportunity existence result only; it is not an obs_v2 or Dreamer admission result.

Current supervised fixed-exit baseline result:

```text
classification: SUPERVISED_BASELINE_GATE2_FAIL
primary model: XGBoost flattened_60x18_market_seq
validation chosen threshold: no-trade threshold
test: +396 / 9 trades / expectancy +44
top 5% selective entry: -12456
top 10% selective entry: -17664
top 20% selective entry: -20700
Gate 3: blocked by Gate 2
```

This means the broader JM8Y split is useful, and fixed-exit opportunities exist, but the current supervised entry selector still does not prove stable out-of-sample entry-selection ability.

Current validation-selection contract:

```text
docs/jm8y_regime_aware_validation_score_contract_20260620.md
tools/score_regime_aware_validation.py
classification: VALIDATION_SCORE_FORMULA_FROZEN_V1
```

### Continuous-Contract / Roll Audit

The source filename contains `JM2601`, while the rows span eight years. This must be resolved before using it as a formal training dataset.

Current result: `USER_CONFIRMED_MAIN_CONTINUOUS_CONTRACT`.

The audit at `docs/jm8y_continuous_contract_roll_audit_20260620.md` found no symbol / contract metadata column in the CSV and 41 roll-boundary risk / data-jump candidate boundaries.

Decision on 2026-06-20: the data owner confirmed this file is JM main-continuous contract data. This allows `jm8y_wf_v1` construction and later diagnostic Dreamer training after the obs_v2 gate. All results must be interpreted as JM main-continuous contract results. Do not claim verified single listed contract performance.

The audit must answer:

```text
What real contract symbol does each row/day represent?
Is this a main-contract continuous series?
What is the roll rule?
Does the roll rule use future volume/open-interest information?
Are prices back-adjusted or raw stitched?
Are there roll-date price jumps?
How do volume and open interest behave across roll boundaries?
Do OI-relative features spike or reset at roll boundaries?
Does entry_eval behave correctly around roll boundaries?
```

Minimum outputs:

```text
contract_symbol_by_day.csv
roll_event_table.csv
roll_gap_distribution.csv
oi_volume_roll_boundary_audit.csv
continuous_contract_audit_report.md
```

## Training Protocol

Train all regimes together.

Correct:

```text
train_env date pool = 2018-01-02 .. 2022-12-30
reset sampling = random episode starts across the train pool
optional later enhancement = regime-balanced reset sampling
```

Avoid:

```text
train UP first
then train RANGE
then train DOWN
```

Sequential regime training is not recommended because Dreamer can forget earlier regimes and become biased toward the final stage.

For the first pilot, use ordinary randomize-start sampling across the full train pool. Add regime-balanced reset sampling only if training logs show the model overfits to the most frequent regime.

## Checkpoint Selection

Validation must not be scored only by total PnL. The score formula must be frozen before training and must not be changed after seeing test results.

Minimum v1 validation score:

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

Minimum report dimensions:

```text
overall validation actual PnL
validation actual PnL by major_regime
validation actual PnL by structure_regime
validation fixed-exit attribution by regime
validation exit/holding/path delta by regime
monthly validation PnL
floating drawdown
matched-random same-count baseline
oracle fixed-exit reference
2023 validation separately
2024 validation separately
LONG / SHORT
trade ordinal
```

If a checkpoint makes money only in 2024 DOWN but loses badly in 2023 balanced/range conditions, do not select it as robust.

## Test Protocol

Run test once for the validation-selected checkpoint:

```text
test = 2025-01-02 .. 2025-12-02
policy = deterministic pred()
execution = signal_on_close_plus_spread
observation = obs_v2
reward = unchanged
max_entries_per_day = unchanged
```

Note: do not introduce additional obs_v2 version names. There is only one current obs_v2 contract, and its exact field set must be audited before training.

Report:

```text
actual full-policy PnL
fixed-exit attribution
exit/holding/path delta
matched-random same-count baseline
oracle fixed-exit
closed equity curve
mark-to-market equity curve
drawdown
daily/monthly PnL
direction / ordinal / sequence breakdown
PnL by major_regime and structure_regime
```

## Optional Expanding Walk-Forward Folds

For a stronger conclusion, run expanding chronological folds after the single pilot:

| fold | train | validation | test | purpose |
| --- | --- | --- | --- | --- |
| fold_1 | 2018-2021 | 2022 | 2023 | tests recovery from 2022 down/volatile validation to 2023 balanced test |
| fold_2 | 2018-2022 | 2023 | 2024 | tests balanced validation to severe down test |
| fold_3 | 2018-2022 | 2023-2024 | 2025 | recommended pilot, tests mixed validation to future 2025 |

Do not mix test periods back into train. Each fold must have its own split manifest and audit report.

## Required Artifacts For Implementation

Create a new trading contract root:

```text
/data/logdir/trading_contracts/jm8y_wf_v1
```

Expected contents:

```text
data/raw/8Y_DCE_JM2601_1m.csv
data/generated/jm8y_wf_v1_train_1m.csv
data/generated/jm8y_wf_v1_full_1m.csv
configs/env/jm8y_wf_v1_train_futures_intraday_reward.yaml
configs/env/jm8y_wf_v1_full_futures_intraday_reward.yaml
configs/entry_eval/jm8y_wf_v1_signal_close.yaml
artifacts/walk_forward_splits/jm8y_wf_v1/split_manifest.json
artifacts/walk_forward_splits/jm8y_wf_v1/split_manifest.sha256
artifacts/walk_forward_splits/jm8y_wf_v1/regime_day_table.csv
artifacts/walk_forward_splits/jm8y_wf_v1/regime_distribution_by_split.csv
artifacts/continuous_contract_audit/jm8y_wf_v1/continuous_contract_audit_report.md
artifacts/obs_v2_audit/jm8y_wf_v1/summary.json
artifacts/entry_eval/jm8y_wf_v1/report.md
```

The split guard should confirm:

```text
train max date <= 2022-12-30
validation dates are excluded from train
test dates are excluded from train and validation selection
entry_eval split hash matches training metadata
```

## Recommended Next Experiment

After the readiness blockers pass, run exactly one seed-c pilot first:

```text
product: JM
data: 8Y_DCE_JM2601_1m.csv
split: JM8Y-WF-V1
train: 2018-01-02 .. 2022-12-30
validation: 2023-01-03 .. 2024-12-31
test: 2025-01-02 .. 2025-12-02
obs: obs_v2
reward: futures_intraday_reward_function
execution: signal_on_close_plus_spread
checkpoint retention: 500k / 700k / 900k / latest
selection: validation only, regime-aware report
test: selected checkpoint only, once
```

If seed-c is promising, repeat with multi-seed. If it fails, the failure is much more meaningful than the current narrow-fold failure because the train pool has adequate regime coverage.

## Final Classification

```text
JM_8Y_DATA_SUFFICIENT
CURRENT_NARROW_SPLIT_REGIME_IMBALANCED
JM8Y_WF_V1_RECOMMENDED
JM8Y_SPLIT_DESIGN_ACCEPTED
JM8Y_MAIN_CONTINUOUS_DATA_ACCEPTED
ROLL_BOUNDARY_RISK_MUST_BE_REPORTED
JM8Y_CONTRACT_ROOT_READY
JM8Y_ENTRY_EVAL_DATASET_READY
JM8Y_SUPERVISED_BASELINE_GATE2_FAIL
JM8Y_VALIDATION_SCORE_FORMULA_FROZEN_V1
OBS_V2_REFIT_AND_REAUDIT_REQUIRED
TRAIN_REGIMES_SHOULD_BE_MIXED_NOT_SEQUENTIAL
REGIME_AWARE_VALIDATION_REQUIRED
DREAMER_TRAINING_PENDING_OBS_GATE
```

The immediate engineering step is not to tune reward or add indicators. The dataset-side construction, entry_eval baselines, and validation-selection formula are now complete under the JM main-continuous assumption. Before any pilot training, complete the separate obs_v2 train-only refit/audit path. Any pilot report must include roll-boundary candidate +/-1 / +/-3 / +/-5 trading-day attribution.

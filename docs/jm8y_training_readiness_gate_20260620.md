# JM8Y Training Readiness Gate

Date: 2026-06-20

Status:

```text
JM8Y_SPLIT_DESIGN_ACCEPTED
JM8Y_MAIN_CONTINUOUS_DATA_ACCEPTED
DATASET_AND_ENTRY_EVAL_BASELINES_READY
SUPERVISED_BASELINE_GATE2_FAIL
VALIDATION_SCORE_FORMULA_FROZEN_V1
OBS_V2_REFIT_AND_REAUDIT_REQUIRED
DREAMER_TRAINING_PENDING_OBS_GATE
```

This gate applies before any Dreamer training on `/home/v/Documents/work/dreamerv3/data/8Y_DCE_JM2601_1m.csv`.

## Accepted Split Design

Use `JM8Y-WF-V1` after the blockers below pass:

```text
train:      2018-01-02 .. 2022-12-30
validation: 2023-01-03 .. 2024-12-31
test:       2025-01-02 .. 2025-12-02
```

Rationale:

```text
train contains UP / DOWN / RANGE exposure;
validation contains balanced 2023 plus severe-down 2024;
test preserves future 2025 as holdout.
```

## Gate 1: Main-Continuous Contract Audit

The file name `8Y_DCE_JM2601_1m.csv` spans 2018..2025. It must not be assumed to be a clean single-contract dataset.

Current audit result: `USER_CONFIRMED_MAIN_CONTINUOUS_CONTRACT`.

Produced artifacts:

```text
docs/jm8y_continuous_contract_roll_audit_20260620.md
artifacts/jm8y_continuous_contract_audit/20260620/summary.json
artifacts/jm8y_continuous_contract_audit/20260620/daily_contract_audit.csv
artifacts/jm8y_continuous_contract_audit/20260620/roll_gap_distribution.csv
artifacts/jm8y_continuous_contract_audit/20260620/roll_event_table.csv
```

The 8Y CSV has no symbol / contract metadata column, so the audit cannot reconstruct the underlying listed contract for each trading day from the file itself. The data owner has confirmed this file is JM main-continuous contract data. The 41 detected roll-risk / data-jump candidate boundaries are therefore treated as continuous-contract roll-boundary risk points, not as a reason to reject the dataset.

Decision on 2026-06-20: continue under the explicit assumption that this file is JM main-continuous contract data. It is valid to train and evaluate a main-continuous experiment on this dataset.

Conclusion boundaries:

```text
Allowed: Dreamer result on JM main-continuous contract data.
Not allowed: Dreamer result on one verified single listed JM contract.
```

Optional metadata questions if a separate roll/symbol source becomes available later:

```text
Is this single-contract, main-continuous, back-adjusted, or stitched data?
What real symbol belongs to each trading day?
What roll dates exist?
What roll rule was used?
Does the roll rule use future information?
Are prices adjusted or raw?
Are there close / volume / OI jumps at roll boundaries?
Do obs_v2 OI-relative features behave safely around rolls?
Does entry_eval avoid invalid roll-boundary artifacts?
```

Required reporting around roll-boundary candidates:

```text
normal days PnL
roll-boundary candidate +/-1 trading day PnL
roll-boundary candidate +/-3 trading day PnL
roll-boundary candidate +/-5 trading day PnL
OI / volume / volatility behavior around roll-boundary candidates
oracle and Dreamer trade concentration around roll-boundary candidates
```

Gate:

```text
PASS for JM main-continuous contract experiments.
ROLL_BOUNDARY_RISK_MUST_BE_REPORTED.
```

## Gate 2: Trading Contract Root

Status: `DATASET_SPLIT_READY_MAIN_CONTINUOUS`.

Created:

```text
/data/logdir/trading_contracts/jm8y_wf_v1
```

Produced artifacts:

```text
data/raw/8Y_DCE_JM2601_1m.csv
data/generated/jm8y_wf_v1_train_20180102_20221230_1m.csv
configs/env/jm8y_wf_v1_train.yaml
configs/env/jm8y_wf_v1_full.yaml
configs/entry_eval/jm8y_wf_v1_signal_close.yaml
artifacts/walk_forward_splits/jm8y_wf_v1/split_manifest.json
artifacts/walk_forward_splits/jm8y_wf_v1/split_manifest.sha256
artifacts/walk_forward_splits/jm8y_wf_v1/summary.json
artifacts/dataset_audit/jm8y_wf_v1/split_table.csv
artifacts/dataset_audit/jm8y_wf_v1/regime_day_table.csv
artifacts/dataset_audit/jm8y_wf_v1/regime_distribution_by_split.csv
artifacts/dataset_audit/jm8y_wf_v1/roll_risk_by_split.csv
artifacts/dataset_audit/jm8y_wf_v1/summary.json
```

Report:

```text
docs/jm8y_wf_v1_contract_root_report_20260620.md
docs/jm8y_wf_v1_dataset_audit_20260620.md
```

Split manifest hash:

```text
c84954dcc601ca7acdbb5216ea736ab3a88edbd328de8bc5e86e7293c58bdfd1
```

Gate:

```text
PASS for dataset split construction under main-continuous assumption.
Training is pending the obs_v2 refit/audit gate.
```

## Gate 3: Obs v2 Refit And Audit

All train-fitted obs parameters must be recomputed using only:

```text
train: 2018-01-02 .. 2022-12-30
```

Do not reuse parameters from the 2024-2025 split.

Affected families include:

```text
vol_train_percentile
atr_1m_30_train_percentile
atr_1m_60_train_percentile
daily_range_percentile_train
h1_range_percentile_train
obs_cumVWAP_t_train_percentile
obs_session_high_t_train_percentile
obs_range_t_train_percentile
obs_pct_chg_from_ref_t_train_percentile
```

## Gate 4: Entry Eval / Oracle By Regime

Status: `ENTRY_EVAL_DATASET_READY`.

Produced artifacts:

```text
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval/jm8y_wf_v1
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/candidate_distribution_by_split.csv
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/candidate_distribution_by_regime.csv
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/oracle_by_split.csv
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/oracle_by_regime.csv
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/matched_random_by_split.csv
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/summary.json
```

Report:

```text
docs/jm8y_wf_v1_entry_eval_dataset_audit_20260620.md
```

Key result:

```text
candidates: 418147
outcomes: 836294
execution_timing: signal_on_close_plus_spread
unknown_split_candidates: 0
```

Constrained oracle remains strongly positive in all splits:

```text
train:      +998316 / 2379 trades / expectancy 419.64
validation: +362154 / 816 trades / expectancy 443.82
test:       +294264 / 666 trades / expectancy 441.84
```

Matched random same-count remains negative in all splits:

```text
train mean:      -124288.2
validation mean: -34839.6
test mean:       -33846.0
```

This confirms that fixed-exit opportunities exist across the JM8Y split. It does not prove that obs_v2 or Dreamer can identify them.

Required audits:

```text
obs_v2 schema audit
full mutation causality audit
formula reference audit
distribution drift audit
fit-param train-only audit
```

Gate:

```text
PASS only if obs_v2 outputs are finite, causal, formula-correct, and fit params use train only.
```

## Blocker 4: Entry-Eval And Baselines By Regime

Status: `SUPERVISED_BASELINE_GATE2_FAIL`.

Completed:

```text
entry_eval oracle by regime
matched-random fixed-exit by regime
supervised fixed-exit baseline by regime
```

Report by:

```text
train / validation / test
2023 validation separately
2024 validation separately
major UP / DOWN / RANGE / HIGH_VOL_RANGE
structure UP / DOWN / RANGE
LONG / SHORT
session phase
trade ordinal if applicable
```

Produced artifacts:

```text
docs/jm8y_wf_v1_supervised_baseline_by_regime_20260620.md
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/model_summary.csv
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/strategy_by_split.csv
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/strategy_by_regime.csv
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/prediction_deciles_by_regime.csv
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/summary.json
```

Key supervised result:

```text
Ridge:              Gate 2 FAIL, test -504 / 9 trades
XGBoost window:     Gate 2 FAIL, test +276 / 4 trades
XGBoost flattened:  Gate 2 FAIL, test +396 / 9 trades
Primary:            Gate 2 FAIL, flattened_60x18_market_seq
```

The small positive primary executable test result is not sufficient:

```text
validation selected a no-trade threshold;
primary test executed only 9 trades;
top 5% / 10% / 20% selective-entry diagnostics are negative;
cost stress degrades quickly;
Gate 3 remains blocked by Gate 2.
```

Primary test trades by regime:

```text
DOWN/DOWN: +870 / 5 trades, all SHORT
UP/UP:     -474 / 4 trades
```

Gate:

```text
DATASET BASELINE COMPLETE.
SUPERVISED FIXED-EXIT ENTRY SIGNAL NOT PROVEN.
Do not use this result as Dreamer training admission.
```

## Blocker 5: Validation Score Freeze

Status: `VALIDATION_SCORE_FORMULA_FROZEN_V1`.

Contract:

```text
docs/jm8y_regime_aware_validation_score_contract_20260620.md
```

Scoring script:

```text
tools/score_regime_aware_validation.py
```

Frozen v1 formula:

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

Selection report must show:

```text
overall validation actual PnL
validation actual PnL by major_regime
validation actual PnL by structure_regime
validation fixed-exit attribution by regime
exit/holding/path delta by regime
2023 validation separately
2024 validation separately
monthly validation PnL
floating drawdown
matched-random same-count baseline
oracle fixed-exit reference
```

Gate:

```text
PASS for formula freeze.
Do not change this formula after seeing test.
```

## Training Contract After Gates Pass

Only after all blockers pass:

```text
product: JM
split: JM8Y-WF-V1
obs: obs_v2
execution: signal_on_close_plus_spread
reward: unchanged futures_intraday_reward_function
action space: SHORT / FLAT / LONG
max_entries_per_day: unchanged
seed: seed-c first
checkpoint retention: 500k / 700k / 900k / latest
```

There is only one current obs_v2 contract. Its exact field set is still under active development and must be audited before training.

## Final Classification

```text
ACCEPT_AS_SPLIT_DESIGN_V1
JM8Y_MAIN_CONTINUOUS_DATA_ACCEPTED
DATASET_SPLIT_READY_MAIN_CONTINUOUS
ENTRY_EVAL_DATASET_READY
SUPERVISED_BASELINE_GATE2_FAIL
VALIDATION_SCORE_FORMULA_FROZEN_V1
OBS_V2_REFIT_AND_REAUDIT_REQUIRED
DREAMER_TRAINING_PENDING_OBS_GATE
NEXT_REQUIRED:
  obs_v2 train-only refit and audit
```

# JM8Y WF V1 Supervised Fixed-Exit Baseline By Regime

Final answer: the JM8Y dataset and entry-eval dataset are usable under the user-confirmed JM main-continuous contract assumption, but the supervised fixed-exit baseline does not pass Gate 2. It remains a dataset/baseline diagnostic result, not fixed-exit entry admission.

## Classification

- `SUPERVISED_BASELINE_GATE2_FAIL`
- split manifest hash: `c84954dcc601ca7acdbb5216ea736ab3a88edbd328de8bc5e86e7293c58bdfd1`
- entry_eval: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval/jm8y_wf_v1`

## Model Summary

| model | encoding | gate2 | val trades | val net | test trades | test net | test exp | decile spearman |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ridge | ridge_window_summary | FAIL | 0 | 0.00 | 9 | -504.00 | -56.00 | NA |
| xgboost_window | window_summary_latest_mean_std_delta | FAIL | 0 | 0.00 | 4 | 276.00 | 69.00 | 0.82 |
| xgboost_flattened | flattened_60x18_market_seq | FAIL | 0 | 0.00 | 9 | 396.00 | 44.00 | 0.30 |
| primary | flattened_60x18_market_seq | FAIL | 0 | 0.00 | 9 | 396.00 | 44.00 | 0.30 |

Important: primary flattened XGBoost selected a validation no-trade threshold, then produced only 9 test trades. Top 5% / 10% / 20% selective-entry diagnostics are all negative, so the small positive 9-trade executable result is not sufficient for Gate 2.

## Baseline References

| split | oracle trades | oracle net | oracle exp | matched-random mean | matched-random p50 |
|---|---:|---:|---:|---:|---:|
| train | 2379 | 998316.00 | 419.64 | -124288.20 | -125529.00 |
| validation | 816 | 362154.00 | 443.82 | -34839.60 | -36336.00 |
| test | 666 | 294264.00 | 441.84 | -33846.00 | -34296.00 |

## Primary Test Trades By Regime

| split | major | structure | trades | net | exp | long | short |
|---|---|---|---:|---:|---:|---:|---:|
| test | DOWN | DOWN | 5 | 870.00 | 174.00 | 0 | 5 |
| test | UP | UP | 4 | -474.00 | -118.50 | 1 | 3 |

## Files

- `model_summary`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/model_summary.csv`
- `strategy_by_split`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/strategy_by_split.csv`
- `strategy_by_regime`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/strategy_by_regime.csv`
- `strategy_by_direction`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/strategy_by_direction.csv`
- `strategy_by_session`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/strategy_by_session.csv`
- `strategy_by_month`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/strategy_by_month.csv`
- `prediction_deciles_by_regime`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/prediction_deciles_by_regime.csv`
- `oracle_by_split`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/oracle_by_split.csv`
- `oracle_by_regime`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/oracle_by_regime.csv`
- `matched_random_by_split`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/matched_random_by_split.csv`
- `summary`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_model_audit/jm8y_wf_v1/summary.json`
- `report`: `docs/jm8y_wf_v1_supervised_baseline_by_regime_20260620.md`

## Decision

- Do not treat this supervised baseline as fixed-exit entry admission.
- The JM8Y split remains useful as a broader dataset, but current fixed-exit supervised entry selection is still weak.
- A diagnostic Dreamer pilot may proceed only after the separate obs_v2 train-only refit/audit gate passes.

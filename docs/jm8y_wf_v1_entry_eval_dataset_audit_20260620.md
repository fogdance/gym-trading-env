# JM8Y WF V1 Entry Eval Dataset Audit

## Final Answer

Classification: `ENTRY_EVAL_DATASET_READY`.

This report audits fixed-exit entry opportunities on the generated entry_eval dataset. It does not train Dreamer and does not train supervised models.

## Dataset

- Candidates: 418147
- Outcomes: 836294
- Split manifest hash: `c84954dcc601ca7acdbb5216ea736ab3a88edbd328de8bc5e86e7293c58bdfd1`
- Execution timing: `signal_on_close_plus_spread`

## Constrained Oracle By Split

| split | trades | net_pnl | expectancy | PF | long | short |
|---|---:|---:|---:|---|---:|---:|
| test | 666 | 294264.00 | 441.8378378378378 | inf | 307 | 359 |
| train | 2379 | 998316.00 | 419.6368221941992 | inf | 1200 | 1179 |
| validation | 816 | 362154.00 | 443.81617647058823 | inf | 397 | 419 |

## Matched Random By Split

| split | runs | target_trades | pnl_mean | pnl_p05 | pnl_p50 | pnl_p95 |
|---|---:|---:|---:|---:|---:|---:|
| test | 50 | 666 | -33846.00 | -44800.50 | -34296.00 | -21645.00 |
| train | 50 | 2379 | -124288.20 | -141069.00 | -125529.00 | -104487.00 |
| validation | 50 | 816 | -34839.60 | -49048.50 | -36336.00 | -16096.50 |

## Regime Metrics

See `oracle_by_regime.csv` for split/major/structure regime detail.

## Outputs

- `candidate_distribution_by_split`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/candidate_distribution_by_split.csv`
- `candidate_distribution_by_regime`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/candidate_distribution_by_regime.csv`
- `oracle_by_split`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/oracle_by_split.csv`
- `oracle_by_regime`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/oracle_by_regime.csv`
- `matched_random_by_split`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/matched_random_by_split.csv`
- `summary`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/entry_eval_audit/jm8y_wf_v1/summary.json`
- `report`: `docs/jm8y_wf_v1_entry_eval_dataset_audit_20260620.md`

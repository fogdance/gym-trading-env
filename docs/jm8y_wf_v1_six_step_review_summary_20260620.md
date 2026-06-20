# JM8Y WF V1 Six-Step Review Summary

Date: 2026-06-20

Purpose: summarize the six dataset-side preparation steps for `jm8y_wf_v1`, identify which reports should be reviewed, and freeze the current decision state before any Dreamer training.

## Final Status

```text
JM8Y_SPLIT_DESIGN_ACCEPTED
JM8Y_MAIN_CONTINUOUS_DATA_ACCEPTED
DATASET_SPLIT_READY_MAIN_CONTINUOUS
ENTRY_EVAL_DATASET_READY
SUPERVISED_BASELINE_GATE2_FAIL
VALIDATION_SCORE_FORMULA_FROZEN_V1
OBS_V2_REFIT_AND_REAUDIT_REQUIRED
DREAMER_TRAINING_PENDING_OBS_GATE
```

The JM8Y dataset path is user-confirmed as JM main-continuous contract data. It is suitable for JM main-continuous experiments. It is not a verified single listed contract dataset, and results must not be described as single-contract performance.

The data itself is no longer a training blocker. The remaining blocker is the separate `obs_v2` train-only refit/audit path.

## Six Steps

| Step | Report | Status | Review Priority | Why Review |
|---:|---|---|---|---|
| 1 | `jm8y_continuous_contract_roll_audit_20260620.md` | `USER_CONFIRMED_MAIN_CONTINUOUS_CONTRACT` | P0 | Defines experiment object. No per-row symbol metadata, 41 roll-boundary risk/data-jump candidate boundaries to report. |
| 2 | `jm8y_wf_v1_contract_root_report_20260620.md` | `CONTRACT_ROOT_READY` | P1 | Verifies `/data/logdir/trading_contracts/jm8y_wf_v1` layout, configs, split hash, env smoke. |
| 3 | `jm8y_wf_v1_dataset_audit_20260620.md` | `DATASET_SPLIT_READY_MAIN_CONTINUOUS` | P0 | Confirms chronological split, train-only CSV, regime distribution, roll-boundary risk distribution by split. |
| 4 | `jm8y_wf_v1_entry_eval_dataset_audit_20260620.md` | `ENTRY_EVAL_DATASET_READY` | P0 | Confirms candidates/outcomes, oracle opportunities, matched-random negative baselines by split/regime. |
| 5 | `jm8y_wf_v1_supervised_baseline_by_regime_20260620.md` | `SUPERVISED_BASELINE_GATE2_FAIL` | P0 | Prevents overclaiming. Supervised fixed-exit baseline does not prove sample-out entry selection. |
| 6 | `jm8y_regime_aware_validation_score_contract_20260620.md` | `VALIDATION_SCORE_FORMULA_FROZEN_V1` | P0 | Freezes future checkpoint selection before training/test, avoiding post-test selection bias. |

Supporting overview documents:

| Report | Review Priority | Purpose |
|---|---|---|
| `jm8y_training_readiness_gate_20260620.md` | P0 | Single readiness gate page; current authoritative status. |
| `jm_8y_regime_aware_split_design_20260620.md` | P1 | Original split-design reasoning and regime coverage analysis. |

## Key Facts

Split:

```text
train:      2018-01-02 .. 2022-12-30
validation: 2023-01-03 .. 2024-12-31
test:       2025-01-02 .. 2025-12-02
```

Split manifest hash:

```text
c84954dcc601ca7acdbb5216ea736ab3a88edbd328de8bc5e86e7293c58bdfd1
```

Contract root:

```text
/data/logdir/trading_contracts/jm8y_wf_v1
```

Entry-eval dataset:

```text
candidates: 418147
outcomes: 836294
execution_timing: signal_on_close_plus_spread
unknown_split_candidates: 0
```

Oracle fixed-exit opportunities:

```text
train:      +998316 / 2379 trades / expectancy 419.64
validation: +362154 / 816 trades / expectancy 443.82
test:       +294264 / 666 trades / expectancy 441.84
```

Matched-random same-count baselines:

```text
train mean:      -124288.2
validation mean: -34839.6
test mean:       -33846.0
```

Supervised fixed-exit baseline:

```text
Ridge:              Gate 2 FAIL, test -504 / 9 trades
XGBoost window:     Gate 2 FAIL, test +276 / 4 trades
XGBoost flattened:  Gate 2 FAIL, test +396 / 9 trades
Primary:            Gate 2 FAIL, flattened_60x18_market_seq
```

Why primary does not pass:

```text
validation selected a no-trade threshold;
primary test executed only 9 trades;
top 5% / 10% / 20% selective-entry diagnostics are negative;
cost stress degrades quickly;
Gate 3 remains blocked by Gate 2.
```

## Review Recommendations

### Must Review Before Any Training

Review these first:

```text
1. jm8y_wf_v1_six_step_review_summary_20260620.md
2. jm8y_training_readiness_gate_20260620.md
3. jm8y_continuous_contract_roll_audit_20260620.md
4. jm8y_wf_v1_dataset_audit_20260620.md
5. jm8y_wf_v1_entry_eval_dataset_audit_20260620.md
6. jm8y_wf_v1_supervised_baseline_by_regime_20260620.md
7. jm8y_regime_aware_validation_score_contract_20260620.md
```

Reason: these define the main-continuous experiment object, roll-boundary reporting risk, split validity, opportunity existence, failed supervised entry gate, and future checkpoint selection protocol.

### Useful Background

Review if someone needs the reasoning behind the split:

```text
jm_8y_regime_aware_split_design_20260620.md
jm8y_wf_v1_contract_root_report_20260620.md
```

## Current Decision

Do not use these dataset-side results as fixed-exit entry admission.

The dataset-side path is ready for the next engineering gate:

```text
obs_v2 train-only refit/audit under jm8y_wf_v1
```

Only after that passes should a single seed-c Dreamer pilot be launched with:

```text
product: JM
split: jm8y_wf_v1
obs: obs_v2
execution: signal_on_close_plus_spread
reward: unchanged for the experiment
action space: SHORT / FLAT / LONG
checkpoint retention: 500k / 700k / 900k / latest
selection: validation only using VALIDATION_SCORE_FORMULA_FROZEN_V1
test: selected checkpoint only, once
extra reporting: roll-boundary candidate +/-1 / +/-3 / +/-5 trading day PnL
```

## Review Bundle Location

The review copy is prepared under:

```text
/data/logdir/need_review/jm8y_wf_v1_20260620
```

The bundle should contain the six step reports plus the two overview reports.

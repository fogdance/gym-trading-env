# JM8Y WF V1 Contract Root Report

## Final Answer

`jm8y_wf_v1` contract root has been generated under the user-confirmed JM main-continuous contract assumption.

This is not a verified single listed contract dataset. The prior roll audit found no symbol metadata and 41 roll-boundary risk / data-jump candidate boundaries. All downstream results must be described as JM main-continuous contract results, and roll-boundary contribution must be reported in future checkpoint audits.

## Contract Root

```text
/data/logdir/trading_contracts/jm8y_wf_v1
```

Generated files:

```text
/data/logdir/trading_contracts/jm8y_wf_v1/data/raw/8Y_DCE_JM2601_1m.csv
/data/logdir/trading_contracts/jm8y_wf_v1/data/generated/jm8y_wf_v1_train_20180102_20221230_1m.csv
/data/logdir/trading_contracts/jm8y_wf_v1/configs/env/jm8y_wf_v1_train.yaml
/data/logdir/trading_contracts/jm8y_wf_v1/configs/env/jm8y_wf_v1_full.yaml
/data/logdir/trading_contracts/jm8y_wf_v1/configs/entry_eval/jm8y_wf_v1_signal_close.yaml
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/walk_forward_splits/jm8y_wf_v1/split_manifest.json
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/walk_forward_splits/jm8y_wf_v1/split_manifest.sha256
/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/walk_forward_splits/jm8y_wf_v1/summary.json
```

## Split

| role | date range | trading days | raw rows |
|---|---:|---:|---:|
| train | 2018-01-02 .. 2022-12-30 | 1215 | 414531 |
| validation | 2023-01-03 .. 2024-12-31 | 484 | 164938 |
| test | 2025-01-02 .. 2025-12-02 | 222 | 75630 |

Split manifest hash:

```text
c84954dcc601ca7acdbb5216ea736ab3a88edbd328de8bc5e86e7293c58bdfd1
```

Raw CSV hash:

```text
d6887df95e0a8f0b165e87b190d3af438ceb23445702c35f191e5d546c03a688
```

Train CSV hash:

```text
96866616c5994c9b672fccf7268c4960dc9a28680db3da67b8797a0430d952e8
```

## Validation

Builder checks:

```text
chronological: true
no_overlap: true
train_csv_matches_split_train: true
validation_hidden_from_training: true
test_hidden_from_training: true
```

Additional strict futures trading-day check on generated train CSV:

```text
train strict trading_day min/max/count: 20180102 / 20221230 / 1215
train outside train split: 0
train intersects validation: 0
train intersects test: 0
```

Env construction smoke on `jm8y_wf_v1_train.yaml`:

```text
obs keys: action_mask, agent_state, market_seq
market_seq shape: (60, 22)
agent_state shape: (14,)
action_mask shape: (3,)
store rows: 419175
store trading_day min/max: 20180102 / 20221230
```

## Config Notes

Training env:

```text
/data/logdir/trading_contracts/jm8y_wf_v1/configs/env/jm8y_wf_v1_train.yaml
```

The training env points to train-only CSV:

```text
/data/logdir/trading_contracts/jm8y_wf_v1/data/generated/jm8y_wf_v1_train_20180102_20221230_1m.csv
```

Entry evaluator:

```text
/data/logdir/trading_contracts/jm8y_wf_v1/configs/entry_eval/jm8y_wf_v1_signal_close.yaml
```

The entry evaluator points to full raw CSV and the split manifest. Its `data.contract` is `jm8y_wf_v1`.

## Remaining Gates Before Training

Next required gates:

1. Refit obs_v2 parameters using train split only.
2. Re-run obs_v2 formula / causality / distribution audits under `jm8y_wf_v1`.
3. Run one diagnostic Dreamer pilot only after obs_v2 passes, selecting checkpoints by the frozen validation score.
4. Report future checkpoint results by roll-boundary candidate windows.

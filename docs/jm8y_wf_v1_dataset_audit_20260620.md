# JM8Y WF V1 Dataset Audit

## Final Answer

Classification: `DATASET_SPLIT_READY_MAIN_CONTINUOUS`.

The `jm8y_wf_v1` dataset split is chronological and train-only data is separated from validation/test. The raw CSV is user-confirmed as JM main-continuous contract data. The file itself has no contract symbol metadata, so the experiment conclusion must remain scoped to main-continuous JM and not to one verified single listed contract.

## Split Table

| role | start | end | days | valid_rows | valid_ratio | low_valid_days | total_log_return |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 20180102 | 20221230 | 1215 | 405820 | 0.9681 | 111 | 0.3499 |
| validation | 20230103 | 20241231 | 484 | 164938 | 0.9878 | 17 | -0.4739 |
| test | 20250102 | 20251202 | 222 | 75630 | 0.9875 | 8 | -0.0584 |

## Regime Distribution

| role | family | regime | days | ratio |
|---|---|---|---:|---:|
| train | major_regime | DOWN | 286 | 0.2354 |
| train | major_regime | HIGH_VOL_RANGE | 55 | 0.0453 |
| train | major_regime | RANGE | 424 | 0.3490 |
| train | major_regime | UP | 390 | 0.3210 |
| train | major_regime | WARMUP | 60 | 0.0494 |
| train | structure_regime | DOWN | 356 | 0.2930 |
| train | structure_regime | RANGE | 249 | 0.2049 |
| train | structure_regime | UP | 571 | 0.4700 |
| train | structure_regime | WARMUP | 39 | 0.0321 |
| validation | major_regime | DOWN | 219 | 0.4525 |
| validation | major_regime | HIGH_VOL_RANGE | 18 | 0.0372 |
| validation | major_regime | RANGE | 159 | 0.3285 |
| validation | major_regime | UP | 88 | 0.1818 |
| validation | structure_regime | DOWN | 252 | 0.5207 |
| validation | structure_regime | RANGE | 88 | 0.1818 |
| validation | structure_regime | UP | 144 | 0.2975 |
| test | major_regime | DOWN | 123 | 0.5541 |
| test | major_regime | RANGE | 30 | 0.1351 |
| test | major_regime | UP | 69 | 0.3108 |
| test | structure_regime | DOWN | 123 | 0.5541 |
| test | structure_regime | RANGE | 30 | 0.1351 |
| test | structure_regime | UP | 69 | 0.3108 |

## Roll-Boundary Risk Distribution

- train: 29 candidate roll-risk boundaries
- validation: 8 candidate roll-risk boundaries
- test: 4 candidate roll-risk boundaries

## Data Limitation

The source CSV has no symbol/contract column. The data owner has confirmed it is JM main-continuous contract data. Do not claim verified single-contract semantics. Future checkpoint reports must split PnL around roll-boundary candidate windows so we can see whether results depend on continuous-contract construction artifacts.

## Outputs

- `split_table`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/dataset_audit/jm8y_wf_v1/split_table.csv`
- `regime_day_table`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/dataset_audit/jm8y_wf_v1/regime_day_table.csv`
- `regime_distribution_by_split`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/dataset_audit/jm8y_wf_v1/regime_distribution_by_split.csv`
- `roll_risk_by_split`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/dataset_audit/jm8y_wf_v1/roll_risk_by_split.csv`
- `summary`: `/data/logdir/trading_contracts/jm8y_wf_v1/artifacts/dataset_audit/jm8y_wf_v1/summary.json`
- `report`: `docs/jm8y_wf_v1_dataset_audit_20260620.md`

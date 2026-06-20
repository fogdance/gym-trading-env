# JM 8Y Continuous Contract / Roll Audit

## Final Answer

USER_CONFIRMED_MAIN_CONTINUOUS_CONTRACT: the data is accepted as JM main-continuous contract data for the JM8Y experiment.

No symbol / contract metadata column is present in the raw CSV, so this audit cannot reconstruct the exact underlying listed contract for each trading day from the file itself. That is no longer a data blocker because the data owner has confirmed the file is JM main-continuous contract data.

The 41 detected roll-risk / data-jump candidates should be treated as normal continuous-contract roll-boundary risk points, not as proof that the data is dirty or unusable.

This audit does not train Dreamer, does not change env behavior, and does not build a walk-forward split.

## Dataset

| item | value |
|---|---:|
| raw_csv | `/home/v/Documents/work/dreamerv3/data/8Y_DCE_JM2601_1m.csv` |
| trading_days | 1921 |
| date_start | 20180102 |
| date_end | 20251202 |
| candidate_roll_risk_days | 41 |
| low_valid_ratio_days | 136 |
| max_abs_open_gap_pct | 0.179037 |
| max_abs_oi_gap_pct | 1.175643 |
| max_volume_ratio | 4.162 |
| min_volume_ratio | 0.307 |

## Method

Trading days are inferred with the same `strict_reindex_futures_345` session logic used by the env.
The audit aggregates valid 1m rows by trading day, then compares consecutive trading days for:

- next-session open gap versus previous close;
- next first open interest versus previous last open interest;
- next daily volume versus previous daily volume;
- long calendar gaps and low valid-row coverage.

Because the CSV lacks a contract/symbol column, detected events are roll-boundary risk or data-jump candidates inferred from price / OI / volume behavior. They are useful for reporting and post-training attribution, but they are not confirmed roll dates from metadata.

## Top Roll-Risk Candidates

| prev_day -> next_day | open_gap_pct | oi_gap_pct | volume_ratio | risk_score | flags |
|---|---:|---:|---:|---:|---|
| 20210420 -> 20210421 | 0.0535 | 1.1756 | 1.78 | 274.46 | large_price_gap, large_oi_gap |
| 20200821 -> 20200824 | -0.0156 | 1.0028 | 1.32 | 221.93 | large_oi_gap |
| 20190820 -> 20190821 | -0.0484 | 0.8776 | 3.79 | 213.81 | large_price_gap, large_oi_gap |
| 20230811 -> 20230814 | -0.0502 | 0.7458 | 1.80 | 181.30 | large_price_gap, large_oi_gap |
| 20250415 -> 20250416 | 0.0837 | 0.6801 | 1.73 | 178.85 | large_price_gap, large_oi_gap |
| 20180808 -> 20180809 | 0.0563 | 0.6998 | 0.98 | 171.94 | large_price_gap, large_oi_gap |
| 20240816 -> 20240819 | -0.0048 | 0.7166 | 1.62 | 158.19 | large_oi_gap |
| 20231207 -> 20231208 | -0.0179 | 0.6605 | 2.00 | 152.23 | large_oi_gap |
| 20240410 -> 20240411 | 0.0614 | 0.5795 | 1.79 | 149.59 | large_price_gap, large_oi_gap |
| 20230413 -> 20230414 | -0.1130 | 0.4630 | 1.99 | 143.71 | large_price_gap, large_oi_gap |
| 20191219 -> 20191220 | -0.0400 | 0.5940 | 0.86 | 142.59 | large_price_gap, large_oi_gap |
| 20181212 -> 20181213 | -0.1580 | 0.3507 | 2.43 | 137.02 | large_price_gap, large_oi_gap |
| 20250731 -> 20250801 | 0.0713 | 0.4953 | 2.25 | 136.61 | large_price_gap, large_oi_gap |
| 20220818 -> 20220819 | -0.1063 | 0.4360 | 1.73 | 134.60 | large_price_gap, large_oi_gap |
| 20241217 -> 20241218 | 0.0541 | 0.4223 | 1.72 | 113.05 | large_price_gap, large_oi_gap |
| 20191231 -> 20200102 | 0.0026 | -0.5029 | 0.46 | 111.66 | large_oi_gap |
| 20200403 -> 20200407 | -0.1074 | 0.2916 | 1.62 | 103.67 | large_price_gap, large_oi_gap |
| 20221205 -> 20221206 | -0.1790 | 0.1417 | 1.44 | 96.24 | large_price_gap, large_oi_gap |
| 20190409 -> 20190410 | 0.0830 | 0.2163 | 1.57 | 78.61 | large_price_gap, large_oi_gap |
| 20211202 -> 20211203 | -0.0859 | 0.1781 | 3.79 | 77.34 | large_price_gap, large_oi_gap |

## Outputs

- `artifacts/jm8y_continuous_contract_audit/20260620/daily_contract_audit.csv`
- `artifacts/jm8y_continuous_contract_audit/20260620/roll_gap_distribution.csv`
- `artifacts/jm8y_continuous_contract_audit/20260620/roll_event_table.csv`
- `artifacts/jm8y_continuous_contract_audit/20260620/summary.json`

## Training Readiness Impact

The data itself is allowed for JM main-continuous contract experiments.

Allowed statement:

```text
Dreamer is evaluated on JM main-continuous contract data under the configured execution/reward/action contract.
```

Disallowed statement:

```text
Dreamer is evaluated on one verified single listed JM contract from 2018 to 2025.
```

Required reporting for future training/evaluation:

```text
normal days PnL
roll-boundary candidate +/-1 trading day PnL
roll-boundary candidate +/-3 trading day PnL
roll-boundary candidate +/-5 trading day PnL
OI / volume / volatility feature behavior around roll-boundary candidates
oracle and Dreamer trade concentration around roll-boundary candidates
```

Final classification:

```text
USER_CONFIRMED_MAIN_CONTINUOUS_CONTRACT
TRAINING_ALLOWED_AS_MAIN_CONTINUOUS_EXPERIMENT
ROLL_BOUNDARY_RISK_MUST_BE_REPORTED
```

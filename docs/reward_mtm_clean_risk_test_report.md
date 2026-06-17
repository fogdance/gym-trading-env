# MTM Clean/Risk Reward Test Report

Date: 2026-06-17

Final classification: `REWARD_TEST_PASS`

## Scope

This report covers `futures_intraday_mtm_clean_reward_function` (v1-clean) and `futures_intraday_mtm_risk_reward_function` (v2-risk). No formal Dreamer training was run; only unit tests, scripted rollouts, and 760-step metrics smoke runs were used.

## Results

### futures_intraday_mtm_clean_reward_function

- Scripted steps: `2`
- Attributed trades: `1`
- Missing scripted step keys: `[]`
- Dreamer smoke metrics rows: `77`
- Missing metrics keys: `[]`
- Disabled component sums zero: `{'close': True, 'r_atr_close': True, 'dd': True, 'eod': True, 'invalid_total': True}`
- Dreamer smoke logdir: `/tmp/reward-mtm-smoke-futures-intraday-mtm-clean-reward-function-20260617-163137`

### futures_intraday_mtm_risk_reward_function

- Scripted steps: `2`
- Attributed trades: `1`
- Missing scripted step keys: `[]`
- Dreamer smoke metrics rows: `77`
- Missing metrics keys: `[]`
- Disabled component sums zero: `{'close': True, 'r_atr_close': True, 'eod': True, 'invalid_total': True}`
- Dreamer smoke logdir: `/tmp/reward-mtm-smoke-futures-intraday-mtm-risk-reward-function-20260617-163224`

## Notes

- v1-clean disables fee, dd, EOD, close, stop-loss, market-closed, invalid-action, and ATR-close shaping. Fees still affect reward through ledger cash/equity deltas.
- v2-risk disables close/ATR/EOD/invalid/market-closed/stop-loss shaping but keeps `dd` active for explicit risk penalties.
- v2-risk exposes extra numeric audit fields: `risk_dd`, `risk_adverse`, `risk_loss_time`, `drawdown_cash`, `drawdown_inc_cash`, `adverse_cash`, `adverse_inc_cash`, and `loss_steps`.
- Trade-level attribution is a smoke attribution from env trade records and reward debug sums; it is sufficient for visibility, not a full accounting report.

## Outputs

- `artifacts/reward_mtm_clean_risk_tests/summary.json`
- `artifacts/reward_mtm_clean_risk_tests/v1_smoke_steps.csv`
- `artifacts/reward_mtm_clean_risk_tests/v2_smoke_steps.csv`
- `artifacts/reward_mtm_clean_risk_tests/v1_trade_attribution.csv`
- `artifacts/reward_mtm_clean_risk_tests/v2_trade_attribution.csv`
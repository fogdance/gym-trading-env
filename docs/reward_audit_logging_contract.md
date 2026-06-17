# Reward Audit Logging Contract

Date: 2026-06-17

## Contract

Every formal reward function must implement `RewardAuditMixin` with schema `reward_audit_v1`. The env validates the reward audit snapshot after every reward call and fails fast if keys are missing, non-numeric, non-finite, or `total` does not match the returned reward.

Every formal trading env step must expose scalar `log/env/reward/*` keys from reset onward. Dreamer must preserve them into `metrics.jsonl` through `epstats/log/env/reward/*/{avg,max,sum}`.

Mandatory per-step keys:

- `log/env/reward/pnl`
- `log/env/reward/total`
- `log/env/reward/raw_total`
- `log/env/reward/mtm_equity`
- `log/env/reward/prev_mtm_equity`
- `log/env/reward/delta_equity`
- `log/env/reward/scale_cash`
- `log/env/reward/fee_cash_debug`
- `log/env/reward/close`
- `log/env/reward/r_atr_close`
- `log/env/reward/dd`
- `log/env/reward/eod`
- `log/env/reward/invalid_total`
- `log/env/reward/invalid_action_debug`

Episode-level aggregation is emitted through `log/env/reward_episode/*`, non-zero only on terminal/truncated step so Dreamer `sum` equals the episode value.

Mandatory episode keys:

- `log/env/reward_episode/reward_total`
- `log/env/reward_episode/reward_pnl_sum`
- `log/env/reward_episode/reward_close_sum`
- `log/env/reward_episode/reward_atr_close_sum`
- `log/env/reward_episode/reward_dd_sum`
- `log/env/reward_episode/reward_eod_sum`
- `log/env/reward_episode/reward_invalid_sum`
- `log/env/reward_episode/actual_net_pnl`
- `log/env/reward_episode/final_equity`
- `log/env/reward_episode/max_floating_drawdown`
- `log/env/reward_episode/trade_count`

## Smoke Result

- Env reset mandatory keys present: `True`
- Scripted env trades attributed: `3`
- Dreamer metrics mandatory reward keys present: `True`
- Dreamer metrics episode reward keys present: `True`
- Dreamer smoke logdir: `/tmp/reward-logging-dreamer-smoke-20260617-155822`

## V1 Clean Note

`futures_intraday_mtm_clean_reward_function` is not implemented in the current repository. This pass hardens the interface it must implement. A new reward must inherit `RewardAuditMixin`, update its audit snapshot on every `__call__`, and declare any disabled components. Disabled shaping components (`close`, `r_atr_close`, `dd`, `eod`, `invalid_total`) are present from reset and default to zero until a reward function writes non-zero values.

## Outputs

- `artifacts/reward_logging_smoke/current/summary.json`
- `artifacts/reward_logging_smoke/current/reward_component_smoke_steps.csv`
- `artifacts/reward_logging_smoke/current/reward_component_by_trade.csv`
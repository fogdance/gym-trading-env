# JM3 Profit-Lock B-Group Diagnostic Training - 20260703

Final classification: `JM3_PROFIT_LOCK_B_RUN_FAIL_ENTRY_DAMAGE_PRELIM`

## Scope

- Ran JM3 B-group diagnostic training only; no official test and no checkpoint promotion.
- Reward: `futures_intraday_reward_profit_lock_function`.
- Training config: `configs/jm3_profit_lock_diag_seed_c_20260703.yaml`.
- Run logdir: `/data/logdir/future_jm3_profit_lock_diag_seed_c_20260703`.
- Validation env config: `artifacts/jm3_profit_lock_b_run/profit_lock_full_eval_env_20260703.yaml`.
- Validation protocol: same fallback full-validation protocol used by Step1 old baseline re-audit.

## Preflight and Smoke Evidence

- Step0 audit scripts: `PROFIT_LOCK_AB_AUDIT_SCRIPTS_READY`.
- Step1 old baseline: `JM3_OLD_BASELINE_REAUDIT_READY_WITH_CAVEATS`.
- Tiny smoke passed; see `artifacts/jm3_profit_lock_b_run/training_smoke_summary.json`.
- Training used CUDA; validation replay printed `JAX devices (1): [cuda:0]`.
- Full eval config has `live_mode: false` and `live_replay_on_reset: false`.

## Checkpoint Summary

| checkpoint | trades | actual_net_pnl | fixed_candidate_pnl | executable_fixed_pnl | exit_holding_delta | MFE>=1R | big_MFE_to_loss | overlay_abs_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step_000000300000 | 59 | -1704 | 1116 | 888 | -2820 | 25 | 4 | 0.001788 |
| step_000000500000 | 61 | -3486 | -1506 | 558 | -1980 | 27 | 7 | 0.001253 |
| step_000000700000 | 63 | -5298 | -1812 | -192 | -3060 | 24 | 8 | 0.000847 |

## Observed Result

- The B policy did not collapse to no-trade: retained checkpoints produced 59, 61, and 63 validation trades.
- Best B checkpoint by actual PnL was 300k, but it was still negative: actual_net_pnl = -1704.
- At 300k, fixed-exit entry attribution was positive (+1116) while actual PnL was negative, so exit/holding damaged otherwise positive entries.
- At 500k and 700k, fixed-exit attribution turned negative (-1506 and -1812), and 700k executable fixed PnL was also negative (-192).
- Overlay dominance was not the observed failure mode: overlay_abs_share stayed below 0.0015 and reward_sign_changed_trade_count was 0 at all retained checkpoints.

## Comparison Anchor

- Step1 old baseline best checkpoint: `step_000000700000`, actual_net_pnl=3360.0, executable_fixed_pnl=840.0, trades=20.
- Profit-lock 700k checkpoint: actual_net_pnl=-5298.0, executable_fixed_pnl=-192.0, trades=63.

## Audit Bug Fixed During Step2

- During validation review, `profit_lock_trade_rollup.csv` had non-physical capture ratios from step debug carryover.
- Code fix: `tools/dreamer_checkpoint_audit.py` now computes trade-level `capture_ratio = realized_pnl / max(mfe_cash, eps)` directly from trade-level MFE and realized PnL.
- Targeted test: `tests/unit/test_profit_lock_ab_audit_outputs.py` asserts the corrected calculation.

## Outputs

- checkpoint_summary: `artifacts/jm3_profit_lock_b_run/checkpoint_summary.csv`
- profit_lock_trade_rollup: `artifacts/jm3_profit_lock_b_run/profit_lock_trade_rollup.csv`
- profit_lock_component_summary: `artifacts/jm3_profit_lock_b_run/profit_lock_component_summary.csv`
- trade_mfe_capture_summary: `artifacts/jm3_profit_lock_b_run/trade_mfe_capture_summary.csv`
- fixed_exit_attribution_by_checkpoint: `artifacts/jm3_profit_lock_b_run/fixed_exit_attribution_by_checkpoint.csv`
- direction_breakdown: `artifacts/jm3_profit_lock_b_run/direction_breakdown.csv`
- monthly_breakdown: `artifacts/jm3_profit_lock_b_run/monthly_breakdown.csv`
- mfe_bucket_breakdown: `artifacts/jm3_profit_lock_b_run/mfe_bucket_breakdown.csv`
- close_reason_breakdown: `artifacts/jm3_profit_lock_b_run/close_reason_breakdown.csv`
- overlay_dominance_summary: `artifacts/jm3_profit_lock_b_run/overlay_dominance_summary.csv`
- forced_close_intervention_summary: `artifacts/jm3_profit_lock_b_run/forced_close_intervention_summary.csv`

## Caveats

- This is validation replay/audit only, not official test.
- The Step1 old baseline used the same fallback full-validation protocol because the original old training env data does not align with the current validation role map.
- Final A/B judgment should compare this B-group output against `artifacts/jm3_old_reward_baseline_reaudit` under the caveat above.


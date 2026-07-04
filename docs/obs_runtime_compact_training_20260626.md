# Runtime Compact OBS for Training

Date: 2026-06-26
Updated: 2026-07-04

## Purpose

This note documents the temporary Python runtime OBS crop used for faster
training. It is not the formal full OBS contract.

The full compute contract remains available for feature construction and formula
audits. The runtime contract is defined by the `*_OUTPUT` lists in
`src/gym_trading_env/utils/market_features.py`.

## Current Runtime Output

When `trading.obs_feature_mode: "obs"` is enabled, the current runtime OBS emits:

```text
market_seq: float32[window_size, 18]
agent_state
action_mask
```

It does not emit:

```text
risk_context
htf_context
```

The full feature computation remains available. Current code adds the old JM3
market fields back into the full compute contract, so the current full compute
counts are:

```text
FEATURES_MARKET_OBS: 48 fields
FEATURES_RISK_CONTEXT: 7 fields
FEATURES_HTF_CONTEXT: 20 fields
```

Only the `*_OUTPUT` lists define what the agent sees.

## Runtime market_seq Fields

```text
obs_V_t
obs_I_t
obs_cumVWAP_t
obs_dC_minus_cumVWAP_t
obs_cmp_C_vs_cumVWAP_t
obs_session_high_t
obs_session_low_t
obs_range_t
obs_open_drift_t
obs_bar_dir_t
obs_minute_index_t
obs_session_phase_t
obs_volatility_t
obs_dI_from_yclose_t
obs_pct_chg_from_ref_t
obs_mask_t
obs_weekday_sin_t
obs_weekday_cos_t
```

This 18-field runtime list intentionally restores the old JM3 compact market
fields while keeping the current full v2 compute fields available internally.

## Restore Full OBS

To restore full OBS, change `src/gym_trading_env/utils/market_features.py` back
to:

```python
FEATURES_MARKET_OBS_OUTPUT = list(FEATURES_MARKET_OBS)
FEATURES_RISK_CONTEXT_OUTPUT = list(FEATURES_RISK_CONTEXT)
FEATURES_HTF_CONTEXT_OUTPUT = list(FEATURES_HTF_CONTEXT)
```

After restoring, `obs_feature_mode: "obs"` should emit:

```text
market_seq: float32[window_size, 48]
risk_context: float32[7]
htf_context: float32[20]
agent_state
action_mask
```

## Test Implication

Contract tests and oracle code should distinguish full compute lists from
runtime output lists. While this runtime compact crop is active, `market_seq`
checks should use `FEATURES_MARKET_OBS_OUTPUT`; `risk_context` and `htf_context`
are absent because their output lists are empty.

Use a runtime smoke check for training runs:

```bash
/home/v/miniconda3/envs/forex/bin/python - <<'PY'
from gym_trading_env.utils.market_features import (
    FEATURES_MARKET_OBS,
    FEATURES_MARKET_OBS_OUTPUT,
    FEATURES_RISK_CONTEXT,
    FEATURES_RISK_CONTEXT_OUTPUT,
    FEATURES_HTF_CONTEXT,
    FEATURES_HTF_CONTEXT_OUTPUT,
)

print("compute market/risk/htf:", len(FEATURES_MARKET_OBS), len(FEATURES_RISK_CONTEXT), len(FEATURES_HTF_CONTEXT))
print("output market/risk/htf:", len(FEATURES_MARKET_OBS_OUTPUT), len(FEATURES_RISK_CONTEXT_OUTPUT), len(FEATURES_HTF_CONTEXT_OUTPUT))
PY
```

Expected compact output:

```text
compute market/risk/htf: 48 7 20
output market/risk/htf: 18 0 0
```

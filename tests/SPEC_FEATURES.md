# Market + Agent Features Spec

本文件是测试契约。测试 oracle 必须按本文公式独立计算，不能直接引用生产实现。

当前阶段修改正式 OBS 的 `market_seq` 与 `risk_context`；`FEATURES_MARKET`、`agent_state`、
`action_mask` 契约保持不变。

## Common Rules

通用常量：

```text
eps = 1e-12
valid_t = mask_t > 0
ref_safe_t = ref_close_t if ref_close_t > eps else (C_t if C_t > eps else eps)
```

所有 OBS 字段必须是 finite numeric。缺失、NaN、Inf、warmup 不足时输出 `0`。

除以下字段外，所有 `FEATURES_MARKET_OBS` 字段在最终输出阶段都乘以 `mask_t`：

```text
volume_impulse_ready_flag
oi_impulse_ready_flag
obs_mask_t
obs_minute_index_t
obs_session_phase_t
dyn5m_macd_ready_flag
```

### Helper: Causal Rolling Percentile

用于 `*_rolling_percentile` 字段：

```text
hist_t = finite values from rows [max(0, t-window+1), t]
if len(hist_t) < min_periods or value_t is not finite:
  out_t = 0
else:
  out_t = clip(2 * count(hist_t <= value_t) / len(hist_t) - 1, -1, 1)
```

默认参数：

```text
window = 240
min_periods = 30
```

这类字段只使用当前行和历史行，不使用未来行，也不使用 train/test 全样本分位网格。

### Helper: Rolling Slope

`rolling_slope(value, N)`：

```text
if t + 1 < N:
  0
else:
  x = [0, 1, ..., N-1] - mean([0, 1, ..., N-1])
  y = value[t-N+1 : t+1]
  slope = sum(x * (y - mean(y))) / sum(x * x)
```

若窗口内存在非 finite 值，输出 `0`。

## Market RAW (`FEATURES_MARKET`)

RAW 字段顺序不变：

```text
C_t
V_t
I_t
cumVWAP_t
dC_minus_cumVWAP_t
cmp_C_vs_cumVWAP_t
ref_close_t
session_open_t
session_high_t
session_low_t
range_frac_t
bar_dir_t
turnover_t
minute_index_t
limit_up_price_t
limit_down_price_t
dI_from_yclose_t
dP_from_ref_t
pct_chg_from_ref_t
open_drift_t
mask_t
weekday_sin_t
weekday_cos_t
```

本轮不修改 RAW 公式。RAW 字段继续由 strict futures 345 对齐、session 累计 VWAP、
session high/low、ref close、session open、dI_from_yclose 等既有契约生成。

## Market OBS (`FEATURES_MARKET_OBS`)

正式 `market_seq` 字段顺序如下：

```text
volume_surprise_logratio_floor
volume_surprise_rolling_percentile
volume_impulse_recent
volume_impulse_slope_3
volume_impulse_slope_5
volume_impulse_ready_flag
oi_rel_yclose_log
oi_rel_session_open_log
oi_delta_rolling_z
oi_rolling_percentile
oi_delta_slope_3
oi_delta_slope_5
oi_price_confirm_short_continuous
oi_price_confirm_long_continuous
oi_impulse_ready_flag
obs_cumVWAP_t
obs_cumVWAP_t_rolling_percentile
obs_dC_minus_cumVWAP_t
obs_cmp_C_vs_cumVWAP_t
obs_session_high_t
obs_session_high_t_rolling_percentile
obs_session_low_t
obs_range_t
obs_range_t_rolling_percentile
obs_open_drift_t
obs_bar_dir_t
obs_minute_index_t
obs_session_phase_t
vol_rolling_percentile
obs_dI_from_yclose_t
obs_pct_chg_from_ref_t
obs_pct_chg_from_ref_t_rolling_percentile
obs_mask_t
dyn5m_macd_line_norm
dyn5m_macd_signal_norm
dyn5m_macd_hist_norm
dyn5m_macd_hist_delta
dyn5m_macd_distance_norm
dyn5m_macd_hist_slope_3
dyn5m_macd_hist_slope_5
dyn5m_macd_cross_age_frac
dyn5m_macd_cross_dir
dyn5m_macd_ready_flag
```

旧字段不再进入正式 `market_seq`：

```text
obs_V_t
obs_I_t
obs_volatility_t
obs_weekday_sin_t
obs_weekday_cos_t
```

### Price / Local Market Features

```text
obs_cumVWAP_t = clip(log(max(cumVWAP_t, eps) / ref_safe_t), -1, 1) if valid_t else 0

obs_session_high_t = clip(log(max(session_high_t, eps) / ref_safe_t), -1, 1) if valid_t else 0

obs_session_low_t = clip(log(max(session_low_t, eps) / ref_safe_t), -1, 1) if valid_t else 0

obs_pct_chg_from_ref_t = clip(log(max(C_t, eps) / ref_safe_t), -1, 1) if valid_t else 0

obs_dC_minus_cumVWAP_t =
  clip(sign(dC_minus_cumVWAP_t) * log1p(abs(dC_minus_cumVWAP_t) / ref_safe_t), -1, 1)
  if valid_t else 0

obs_range_t = clip(log1p(max(range_frac_t, 0)), 0, 1) if valid_t else 0

obs_open_drift_t = clip(open_drift_t, -1, 1)

obs_cmp_C_vs_cumVWAP_t = cmp_C_vs_cumVWAP_t

obs_bar_dir_t = bar_dir_t
```

Rolling percentile supplements:

```text
obs_cumVWAP_t_rolling_percentile =
  causal_rolling_percentile(obs_cumVWAP_t, window=240, min_periods=30)

obs_session_high_t_rolling_percentile =
  causal_rolling_percentile(obs_session_high_t, window=240, min_periods=30)

obs_range_t_rolling_percentile =
  causal_rolling_percentile(obs_range_t, window=240, min_periods=30)

obs_pct_chg_from_ref_t_rolling_percentile =
  causal_rolling_percentile(obs_pct_chg_from_ref_t, window=240, min_periods=30)
```

这些字段不是 train-fitted percentile；字段名不得包含 `train_percentile`。

### Time / Mask Features

```text
obs_mask_t = mask_t

obs_minute_index_t =
  clip(minute_index_t / max(minute_index_t within same day_id, lower bounded by 1), 0, 1)

obs_session_phase_t =
  1 if max(minute_index_t) <= 400 and minute_index_t >= 120 else 0
```

### Volume Features

有效成交量：

```text
V_valid_t = V_t if valid_t else NaN
```

同 session 内过去成交量基线：

```text
ema_t = ewm_mean(V_valid, span=30, min_periods=5, adjust=False, ignore_na=True) within session
ema_prev_t = ema_t shifted by 1 within session
exp_mean_t = expanding_mean(V_valid, min_periods=1) within session
base_t = ema_prev_t if finite else exp_mean_t shifted by 1 within session
base_t = 0 if still missing
```

字段：

```text
volume_surprise_logratio_floor =
  clip(log(max((V_t + 1) / (max(base_t, 1) + 1), eps)), -5, 5) * mask_t

volume_surprise_rolling_percentile =
  causal_rolling_percentile(volume_surprise_logratio_floor, 240, 30)

volume_impulse_recent =
  rolling_mean(volume_surprise_logratio_floor, window=5, min_periods=1)

volume_impulse_slope_3 =
  rolling_slope(volume_surprise_logratio_floor, 3)

volume_impulse_slope_5 =
  rolling_slope(volume_surprise_logratio_floor, 5)

volume_impulse_ready_flag =
  1 if rolling_sum(valid_t, window=30, min_periods=1) >= 30 else 0
```

### Open Interest Features

```text
I_yclose_t = I_t - dI_from_yclose_t

obs_dI_from_yclose_t =
  clip(sign(dI_from_yclose_t) * log1p(abs(dI_from_yclose_t) / max(abs(I_yclose_t), eps)), -1, 1)
  if valid_t else 0

oi_rel_yclose_log =
  clip(sign(I_t - I_yclose_t) * log1p(abs(I_t - I_yclose_t) / max(abs(I_yclose_t), eps)), -5, 5)
  if valid_t else 0

first_oi_session_t = first valid I_t in current session

oi_rel_session_open_log =
  clip(sign(I_t - first_oi_session_t) * log1p(abs(I_t - first_oi_session_t) / max(abs(first_oi_session_t), eps)), -5, 5)
  if valid_t else 0
```

OI delta z-score：

```text
oi_delta_t = diff(I_t), fill first as 0, and set to 0 when invalid
oi_mean_t = rolling_mean(oi_delta, window=240, min_periods=30).shift(1)
oi_std_t = rolling_std(oi_delta, window=240, min_periods=30, ddof=0).shift(1)
oi_delta_rolling_z = clip((oi_delta_t - oi_mean_t) / oi_std_t, -5, 5) * mask_t
```

若 `oi_std_t` 为 0 或缺失，`oi_delta_rolling_z = 0`。

```text
oi_rolling_percentile =
  causal_rolling_percentile(oi_delta_rolling_z, 240, 30)

oi_delta_slope_3 =
  rolling_slope(oi_delta_rolling_z, 3)

oi_delta_slope_5 =
  rolling_slope(oi_delta_rolling_z, 5)
```

价格确认字段：

```text
logret5_t = log(C_t / C_{t-5}) if valid_t and C_{t-5} > eps and C_t > eps else 0

oi_price_confirm_short_continuous =
  clip(max(-logret5_t, 0) * max(oi_delta_rolling_z, 0), 0, 5)

oi_price_confirm_long_continuous =
  clip(max(logret5_t, 0) * max(oi_delta_rolling_z, 0), 0, 5)

oi_impulse_ready_flag =
  1 if rolling_sum(valid_t, window=30, min_periods=1) >= 30 else 0
```

### Volatility Feature

```text
C_valid_t = C_t if valid_t else NaN
logret_t = log(C_valid_t / C_valid_{t-1}), with NaN/Inf -> 0
raw_vol_t = rolling_std(logret, window=30, min_periods=5), missing -> 0, lower clipped at 0

vol_rolling_percentile =
  causal_rolling_percentile(raw_vol_t, 240, 30)
```

### Dynamic 5m MACD Features

5m MACD 构造口径：

```text
只使用有效 1m bar。
每 5 根有效 1m bar 完成一根 5m bar。
当前未完成 5m bar 每来一根有效 1m bar 动态更新。
夜盘、日盘、休市、跨交易日不重置 EMA 状态。
invalid row 沿用上一行 MACD 状态。
```

参数：

```text
fast = 12
slow = 26
signal = 9
hist = macd - signal
```

动态 EMA 更新：

```text
ema_next(prev, value, span):
  if prev is missing: value
  else: prev + 2/(span+1) * (value - prev)
```

当进入新的 5 根有效 1m 分组时，先用上一根有效 close 完成上一根 5m bar，
更新 completed fast/slow/signal EMA；再用当前 close 计算动态未完成 5m bar。

归一化分母：

```text
TR_t = max(H_t - L_t, abs(H_t - prev_close_t), abs(L_t - prev_close_t))
ATR30_t = rolling_mean(TR_t over valid rows, window=30, min_periods=30), missing -> 0
denom_macd_t = max(ATR30_t, ref_safe_t * 1e-6, eps)
```

字段：

```text
dyn5m_macd_line_norm = clip(macd / denom_macd, -5, 5)
dyn5m_macd_signal_norm = clip(signal / denom_macd, -5, 5)
dyn5m_macd_hist_norm = clip(hist / denom_macd, -5, 5)
dyn5m_macd_hist_delta = clip((hist_t - hist_{t-1}) / denom_macd, -5, 5)
dyn5m_macd_distance_norm = clip((macd - signal) / denom_macd, -5, 5)
dyn5m_macd_hist_slope_3 = clip(rolling_slope(hist, 3) / denom_macd, -5, 5)
dyn5m_macd_hist_slope_5 = clip(rolling_slope(hist, 5) / denom_macd, -5, 5)
dyn5m_macd_cross_dir = sign(macd - signal)
```

Cross age：

```text
cross occurs when sign(macd - signal) changes from non-zero to the opposite non-zero sign.
dyn5m_macd_cross_age_frac =
  clip(valid_1m_bars_since_last_cross / 120, 0, 1)
```

Ready flag：

```text
dyn5m_macd_ready_flag = 1 if completed_5m_bar_count >= 35 else 0
```

## Risk Context (`FEATURES_RISK_CONTEXT`)

`risk_context` 是 obs 模式下的当前行风险尺度向量，不是时间窗口。字段顺序固定：

```text
atr_1m_30_price_frac
atr_1m_60_price_frac
atr_1m_30_rolling_percentile
atr_1m_60_rolling_percentile
current_bar_range_atr_30
intraday_volatility_percentile
atr_ready_flag
```

该向量只使用当前行 `t` 和历史行 `<= t`。它不使用 train-fitted percentile。
字段名不得包含 `train_percentile`，除非未来引入显式 split-aware fit artifact。

True range：

```text
prev_close_t = C_{t-1}, first row uses C_t
TR_t = max(H_t - L_t, abs(H_t - prev_close_t), abs(L_t - prev_close_t))
TR_t is NaN for invalid rows before rolling calculation
```

ATR：

```text
atr_1m_30_t = rolling_mean(TR, window=30, min_periods=30), missing -> 0
atr_1m_60_t = rolling_mean(TR, window=60, min_periods=60), missing -> 0
atr_1m_30_ready_t = rolling_sum(valid_t, window=30, min_periods=1) >= 30
atr_1m_60_ready_t = rolling_sum(valid_t, window=60, min_periods=1) >= 60

atr_1m_30_price_frac =
  clip(atr_1m_30_t / C_t, 0, 1) if valid_t and C_t > eps and atr_1m_30_ready_t else 0

atr_1m_60_price_frac =
  clip(atr_1m_60_t / C_t, 0, 1) if valid_t and C_t > eps and atr_1m_60_ready_t else 0
```

Percentile and range fields：

```text
atr_1m_30_rolling_percentile =
  causal_rolling_percentile(atr_1m_30_price_frac, 240, 30) * mask_t

atr_1m_60_rolling_percentile =
  causal_rolling_percentile(atr_1m_60_price_frac, 240, 30) * mask_t

current_bar_range_atr_30 =
  clip((H_t - L_t) / max(atr_1m_30_t, eps), 0, 20)
  if valid_t and atr_1m_30_ready_t else 0

intraday_volatility_percentile =
  vol_rolling_percentile * mask_t
```

Ready flag：

```text
atr_ready_flag =
  1 if valid_t and atr_1m_30_ready_t and atr_1m_60_ready_t else 0
```

所有 `risk_context` 字段必须 finite。invalid row 或 warmup 不足时输出 `0`。

## Agent RAW (`FEATURES_AGENT`)

`agent_state` 不在本轮修改，但其契约必须保持完整。

RAW 字段顺序：

```text
pos_t
have_long_t
have_short_t
entry_price_t
holding_minutes_t
upnl_t
realized_pnl_step_t
realized_pnl_cum_t
fee_step_t
fee_cum_t
equity_t
max_equity_t
drawdown_t
sigma_entry_t
sl_ticks_t
tp_ticks_t
sl_price_t
tp_price_t
minutes_to_timeout_t
```

关键公式：

```text
pos_t = long_lots - short_lots
have_long_t = 1 if long_lots > 0 else 0
have_short_t = 1 if short_lots > 0 else 0
entry_price_t = active side weighted average entry, or 0 when flat
holding_minutes_t = active side weighted/selected holding age, or 0 when flat
upnl_t = sum(long unrealized pnl) + sum(short unrealized pnl)
equity_t = cash_balance + used_margin + upnl_t
max_equity_t = max(prev_max_equity, equity_t)
drawdown_t = max(max_equity_t - equity_t, 0)
```

## Agent OBS (`FEATURES_AGENT_OBS`)

OBS 字段顺序：

```text
obs_market_open_t
obs_can_long_open_t
obs_can_short_open_t
obs_can_long_close_t
obs_can_short_close_t
obs_pos_side_t
obs_entries_left_frac_t
obs_minutes_to_eod_frac_t
obs_holding_frac_t
obs_upnl_R_t
obs_realized_today_R_t
obs_equity_frac_t
obs_drawdown_frac_t
obs_action_result_t
```

关键公式：

```text
obs_market_open_t = 1 if market_open else 0
obs_can_long_open_t = 1 if can_long_open else 0
obs_can_short_open_t = 1 if can_short_open else 0
obs_can_long_close_t = 1 if can_long_close else 0
obs_can_short_close_t = 1 if can_short_close else 0

obs_pos_side_t =
  +1 for long only
  -1 for short only
   0 for flat or mixed/undefined

obs_entries_left_frac_t =
  max(0, max_entries_per_day - entries_used_today) / max(max_entries_per_day, 1)

obs_minutes_to_eod_frac_t =
  clip(minutes_to_eod, 0, day_len) / max(day_len, 1)

obs_holding_frac_t =
  clip(holding_minutes_t / max(day_len, 1), 0, 1)

obs_upnl_R_t =
  clip(upnl_t / R_cash, -5, 5) if R_cash > 0 else 0

obs_realized_today_R_t =
  clip(realized_today_cash / R_cash, -10, 10) if R_cash > 0 else 0

obs_equity_frac_t =
  clip((equity_t - initial_balance) / initial_balance, -1, 1) if initial_balance > 0 else 0

obs_drawdown_frac_t =
  clip(drawdown_t / initial_balance, 0, 1) if initial_balance > 0 else 0

obs_action_result_t =
  clamp(action_result_code, 0, action_result_max_code) / action_result_max_code
  if action_result_max_code > 0 else 0
```

## Action Mask

顺序固定：

```text
SHORT
FLAT
LONG
```

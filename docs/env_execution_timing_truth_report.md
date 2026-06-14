# Env Execution Timing Truth Report

日期：2026-06-14

状态：Phase 3.5 timing truth audit。只审计，不修改 env 语义，不训练，不改变 entry evaluator。

## Final Answer

结论：当前 `/data/logdir/action-mask-formal-202606122100` 使用的 Dreamer env 是 **Mode C: NON_CAUSAL_CURRENT_BAR**。`market_seq` 包含 `current_step = t` 的当前 bar 特征，且 action 在同一个 `current_step = t` 上成交；开仓 fill price 由 `row_C[t] +/- spread` 生成，不是 `open[t]`，也不是 `open[t+1]`。因此当前 checkpoint 的收益只能作为 legacy diagnostic，不能作为正式 canonical next-open 验收证据。entry evaluator 与当前 env 不对齐：evaluator 是 `decision t -> next valid 1m open[t+1]`，当前 env 是 `observe t -> execute current close[t] +/- spread`。

## Scope

本报告审计的 run：

```text
/data/logdir/action-mask-formal-202606122100
task: gymnasium_CustomTradingEnv-v0
env.gymnasium.config_path: data/trading_stage1.yaml
```

审计脚本：

```text
tools/audit_env_timing_truth.py
```

运行命令：

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/audit_env_timing_truth.py \
  --output artifacts/env_timing_truth/env_timing_truth.csv
```

输出：

```text
artifacts/env_timing_truth/env_timing_truth.csv
```

## Evidence Table

| Case | Obs End Row | Obs End Timestamp | Exec Row | Exec Timestamp | Price Source | Includes Exec Bar | Next Valid Row | Classification |
|---|---:|---|---:|---|---|---|---:|---|
| normal_continuous | 1040 | 2024-06-05T21:06:00+08:00 | 1040 | 2024-06-05T21:06:00+08:00 | current_close_plus_spread | true | 1041 | Mode_C |
| session_open | 1035 | 2024-06-05T21:01:00+08:00 | 1035 | 2024-06-05T21:01:00+08:00 | current_close_plus_spread | true | 1036 | Mode_C |
| before_break | 944 | 2024-06-05T11:30:00+08:00 | 944 | 2024-06-05T11:30:00+08:00 | current_close_plus_spread | true | 945 | Mode_C |
| after_break | 945 | 2024-06-05T13:31:00+08:00 | 945 | 2024-06-05T13:31:00+08:00 | current_close_plus_spread | true | 946 | Mode_C |
| near_eod | 1377 | 2024-06-06T14:58:00+08:00 | 1377 | 2024-06-06T14:58:00+08:00 | current_close_plus_spread | true | 1378 | Mode_C |
| flip_long_to_short_continuation | 1041 | 2024-06-05T21:07:00+08:00 | 1041 | 2024-06-05T21:07:00+08:00 | current_close_derived_other | true | 1042 | Mode_C |

说明：

```text
current_close_derived_other 出现在 FLIP_CLOSE_LONG case。
该 trade record 是 flip 的 close leg，实际 quote_action_price = row_C[t]，close exec = row_C[t] - spread。
同一个 quote 还会在同一 timestamp 写入 flip open leg。
```

## Raw Examples

### Normal Continuous Row

```text
decision/current row: 1040, 2024-06-05T21:06:00+08:00
raw bar:
  open=1661.0
  high=1661.5
  low=1656.5
  close=1657.0
  volume=732

market_seq[-1]:
  row=1040
  timestamp=2024-06-05T21:06:00+08:00

LONG execution:
  execution_row=1040
  execution_timestamp=2024-06-05T21:06:00+08:00
  quote_action_price=1657.0
  quote_exec_price=1657.5
  source=current_close_plus_spread

next valid row:
  row=1041
  timestamp=2024-06-05T21:07:00+08:00
  open=1657.5
  close=1657.5
```

这里 `market_seq` 已经包含 row 1040 的 close/high/low/volume，action 仍在 row 1040 成交。

### Break Boundary Row

```text
decision/current row: 944, 2024-06-05T11:30:00+08:00
raw bar:
  open=1661.0
  high=1664.0
  low=1661.0
  close=1662.0
  volume=208

market_seq[-1]:
  row=944
  timestamp=2024-06-05T11:30:00+08:00

LONG execution:
  execution_row=944
  execution_timestamp=2024-06-05T11:30:00+08:00
  quote_action_price=1662.0
  quote_exec_price=1662.5
  source=current_close_plus_spread

next valid row:
  row=945
  timestamp=2024-06-05T13:31:00+08:00
  open=1660.0
  close=1662.5
```

即使下一根有效 bar 已经跨午休到 13:31，env 仍在 11:30 当前 bar 的 close 上成交。

### Session Open Row

```text
decision/current row: 1035, 2024-06-05T21:01:00+08:00
raw bar:
  open=1652.5
  high=1657.5
  low=1652.5
  close=1657.0
  volume=2753

LONG execution:
  execution_row=1035
  quote_action_price=1657.0
  quote_exec_price=1657.5
```

session open 也不是 open fill，而是当前 bar close + spread fill。

## Code Paths

| Behavior | File | Lines | Evidence |
|---|---|---:|---|
| Run uses Gymnasium trading env config | `/data/logdir/action-mask-formal-202606122100/config.yaml` | 120, 193 | `gymnasium: {config_path: data/trading_stage1.yaml}` and `task: gymnasium_CustomTradingEnv-v0` |
| Dreamer constructs env | `/home/v/Documents/work/dreamerv3/dreamerv3/main.py` | 238-273 | `make_env()` selects `embodied.envs.from_gymnasium:FromGymnasium` and calls ctor |
| Dreamer wrappers do not alter timing | `/home/v/Documents/work/dreamerv3/dreamerv3/main.py` | 276-285 | wraps with `UnifyDtypes`, `CheckSpaces`, optional normalize/clip for continuous actions |
| Gymnasium wrapper forwards action directly | `/home/v/Documents/work/dreamerv3/embodied/envs/from_gymnasium.py` | 57-77 | wrapper calls `self._env.step(action)` after reset handling |
| Env stores current row | `src/gym_trading_env/envs/trading_env.py` | 469-471 | `self.current_step = int(start_row)` and timestamp from `store.index[current_step]` |
| Observation slices through current row | `src/gym_trading_env/envs/trading_env.py` | 3062-3088 | `end_i = self.current_step`; window is `X_all[start_i:end_i + 1]` |
| Observation publishes action table for same row | `src/gym_trading_env/envs/trading_env.py` | 3115-3120 | `_publish_target_transition_table()` is called inside `_get_obs()` |
| Action price is current close | `src/gym_trading_env/envs/trading_env.py` | 603-607 | `_action_price_and_market_open()` returns `row_C[current_step]` |
| Transition quote is built from current close | `src/gym_trading_env/envs/trading_env.py` | 1109-1117 | transition table calls `_build_target_transition_table(price, market_open, ...)` |
| Step consumes obs table before time advances | `src/gym_trading_env/envs/trading_env.py` | 1166-1207 | `step_before=current_step`, consume table, commit quote |
| Time advances only after execution | `src/gym_trading_env/envs/trading_env.py` | 1236-1279 | comment and code move to next step after action execution |
| Open quote timestamp and price | `src/gym_trading_env/envs/trading_env.py` | 741-803 | LONG uses `price + spread`, SHORT uses `price - spread`, timestamp is `store.index[current_step]` |
| Flip quote timestamp and price | `src/gym_trading_env/envs/trading_env.py` | 886-990 | flip uses current `price +/- spread`, timestamp is current row |
| Trade records written with quote timestamp/price | `src/gym_trading_env/envs/trading_env.py` | 2227-2238, 2317-2348 | open/flip records use `timestamp=quote.timestamp`, `open_price=quote.exec_price` |
| Entry evaluator uses next valid open | `src/gym_trading_env/research/entry_evaluator.py` | 347-370 | `entry_row = context.next_valid[decision_row]`, `entry_ref = opens[entry_row]`, LONG `entry_exec = entry_ref + spread` |
| Entry evaluator disallows cross-break entry | `src/gym_trading_env/research/entry_evaluator.py` | 350-355 | rejects `entry_row != decision_row + 1` if `allow_entry_across_break=false` |
| Canonical config requires delay 1 | `configs/entry_eval_jm_dreamer6m_2024_v1.yaml` | 22-31 | `entry_delay_bars: 1`, `allow_entry_across_break: false` |

## Classification

The audited env satisfies Mode C:

```text
observation_window_end_row == current_step
market_seq_last_row == current_step
execution_row == current_step
execution_price_source == current_close +/- spread
```

It does not satisfy Mode A:

```text
Mode A requires observation_window_end_row < execution_row
and fill at the first unseen bar open.
Observed: observation_window_end_row == execution_row.
```

It does not satisfy Mode B:

```text
Mode B requires execution_row == next_valid_bar_row after decision_row
and fill at next valid 1m open.
Observed: execution_row == decision_row, not next_valid_bar_row.
```

## Answers A-F

A. Does Dreamer env observe bar t and execute on bar t?

```text
Yes. market_seq includes bar t, and action executes on row t.
```

B. Does Dreamer env observe bar t-1 and execute on bar t open?

```text
No. market_seq[-1] is row t, not t-1, and fill is not open[t].
```

C. Does Dreamer env observe bar t and execute on t+1 open?

```text
No. next_valid_bar_row is logged, but execution_row remains t.
```

D. What exact bar index and price field are used for entry fill?

```text
Open/close/flip quote uses row_C[current_step] as action_price.
LONG open fill = row_C[current_step] + spread.
SHORT open fill = row_C[current_step] - spread.
Flip close/open legs use the same current row timestamp and current close-derived exec_price.
```

E. Is the current env causal or non-causal?

```text
Under the definitions in this audit, it is non-causal current-bar mode.
The policy sees current bar close/high/low/volume in market_seq and can execute on that same bar's close-derived price.
```

F. Is the entry evaluator aligned with env timing, or offset by one bar?

```text
It is not aligned. The entry evaluator is one valid bar later and uses next valid bar open.
The env is same-row and uses current close. At break boundaries this is not just one row:
for 2024-06-05 11:30, evaluator would reject cross-break entry or use 13:31 open if allowed,
while env fills at 11:30 close + spread.
```

## Implication For Current Checkpoint

The 988576-step checkpoint from `/data/logdir/action-mask-formal-202606122100` was trained and evaluated under this legacy same-row close-fill semantics. Its actual PnL can still be useful as a diagnostic that the full policy learned something under the old environment, but it is not formally valid for canonical next-open validation.

This also explains the Phase 3.5A attribution mismatch:

```text
candidate exact join works because decision_row matches.
entry-time join misses many trades because evaluator entry timestamp is next valid bar,
while env trade timestamp is current decision row.
```

## Minimal Change If Alignment Is Required

Do not implement in this pass. The minimal future change is a contract-level env timing migration:

```text
execution_timing:
  legacy_current_close:
    observe row t
    execute at row_C[t] +/- spread
    only for old checkpoint diagnostics

  canonical_next_open:
    observe completed row t
    action decided after row t
    execute at next valid row open[t+1] +/- spread
    official future training/evaluation mode
```

Required implementation details for a future pass:

```text
1. Add an execution_timing config flag.
2. Keep legacy_current_close default only for old checkpoint replay if needed.
3. In canonical_next_open, transition quote must carry execution_row = next_valid(current_step).
4. Quote timestamp must be store.index[execution_row].
5. Quote action/reference price must be raw exec_open[execution_row], not row_C[current_step].
6. market_seq remains through completed decision row t.
7. Add parity tests against entry evaluator candidate_id / entry_row / entry_timestamp / entry_execution_price.
```

Until this migration is done and retrained/evaluated, current Dreamer actual PnL should be reported as:

```text
legacy current-close diagnostic PnL
```

not:

```text
canonical next-open causal PnL
```

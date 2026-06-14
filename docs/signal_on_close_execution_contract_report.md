# Signal On Close Execution Contract Report

日期：2026-06-14

状态：Phase 3.5A follow-up。只形式化并评估现有成交合约，不修改 Dreamer env 行为，不训练，不新增指标，不进入 Entry-only Dreamer。

## Final Answer

当前 `/data/logdir/action-mask-formal-202606122100` 的 Dreamer env 与新定义的 **`signal_on_close_plus_spread`** 合约精确对齐：observation 包含完成 bar `t`，action 在 `close[t]` 之后生成，LONG 以 `close[t] + spread` 成交，SHORT 以 `close[t] - spread` 成交。spread 在这里被定义为 adverse slippage / bid-ask / one-tick penalty，因此本报告不再把该口径称为 cheating；它是一个明确的信号收盘成交合约，但它不同于更保守的 `canonical_next_open`。

该 checkpoint 在匹配合约下通过 **candidate-level entry attribution**：343/343 trade 全部 join，fixed-exit candidate PnL 为 `+24072`，validation `+3252`，test `+768`。但它没有通过 **executable non-overlap fixed-exit admission**：整体 `+13188` 且优于 matched-random `-9177.45`，validation `+4062`，但 test 为 `-444`。因此 Entry-only Dreamer 仍然 **NOT PASS**。完整 Dreamer actual PnL `+91512` 中，`+67440` 来自 fixed-exit 之外的 exit / holding / path contribution。

## Contract

`signal_on_close_plus_spread` 语义：

```text
observation includes completed bar t
action is generated after close[t]
entry_row = decision_row = t
entry_reference_price = close[t]
LONG entry_execution_price = close[t] + spread
SHORT entry_execution_price = close[t] - spread
spread = adverse slippage / bid-ask / one-tick penalty
first managed bar = next valid 1m bar after t
```

与 `canonical_next_open` 的差异：

```text
canonical_next_open:
  observation includes completed bar t
  action generated after close[t]
  entry executes at next valid 1m open[t+1] +/- spread

signal_on_close_plus_spread:
  observation includes completed bar t
  action generated after close[t]
  entry executes at close[t] +/- spread
```

如果未来官方训练继续使用当前 env 行为，则官方 evaluator 必须使用 `signal_on_close_plus_spread`。如果未来目标是更贴近实盘 next-open 成交，则需要单独修改 env timing，并在新合约下重新训练和评估；不能把旧 checkpoint 的结果直接迁移过去。

## Implementation

新增配置：

```text
configs/entry_eval_jm_dreamer6m_2024_signal_close_v1.yaml
execution_timing: signal_on_close_plus_spread
entry_delay_bars: 0
signal_on_close_safety_filters: true
near_eod_bars: 2
```

新增 evaluator 行为：

| Behavior | File | Lines |
|---|---|---:|
| `execution_timing` supports `canonical_next_open` and `signal_on_close_plus_spread` | `src/gym_trading_env/research/entry_evaluator.py` | 51-72 |
| Signal-on-close sets `entry_row = decision_row`, `entry_ref = close[t]`, management starts at next valid row | `src/gym_trading_env/research/entry_evaluator.py` | 340-366 |
| LONG/SHORT fixed-exit entry execution uses `entry_ref +/- spread` | `src/gym_trading_env/research/entry_evaluator.py` | 504-511 |
| Signal-on-close outcomes are tagged `fixed_barrier_signal_on_close_v1` | `src/gym_trading_env/research/entry_evaluator.py` | 576-596 |
| Manifest stores `candidate_filter_audit` | `src/gym_trading_env/research/entry_dataset.py` | 184-216 |
| Dreamer audit report reads entry-eval timing from manifest | `tools/dreamer_checkpoint_audit.py` | 462-489 |
| Env action price is current row close | `src/gym_trading_env/envs/trading_env.py` | 603-607 |
| Env LONG/SHORT quote uses `price +/- spread` and current row timestamp | `src/gym_trading_env/envs/trading_env.py` | 741-803 |
| Env observation slices through `current_step` | `src/gym_trading_env/envs/trading_env.py` | 3062-3120 |
| Env step consumes the obs-published transition table before time advances | `src/gym_trading_env/envs/trading_env.py` | 1166-1207 |

## Safety Filters

Formal signal-on-close candidates use safety filters so the contract does not accept unmanageable boundary entries:

```text
reject if no following valid management bar
reject if next valid bar is across break / no immediate tradable continuation
reject if candidate is within near_eod_bars of day end
reject if flat canonical account cannot open one lot
```

Generated dataset:

```text
artifacts/entry_eval/entry_eval_jm_dreamer6m_2024_signal_close_v1/
```

Candidate filter audit:

| Item | Count |
|---|---:|
| valid_market_rows | 48840 |
| accepted | 47530 |
| rejected_total | 1310 |
| no_next_valid_same_day | 144 |
| cross_break_or_no_immediate_continuation | 425 |
| near_eod | 144 |
| insufficient_margin | 597 |
| nonfinite_entry_reference | 0 |

Diagnostic fill parity disables these filters only to compare raw env fill behavior on break/EOD rows.

## Parity Validation

Added parity coverage:

```text
tests/unit/test_entry_evaluator.py
tests/integration/test_signal_on_close_env_parity.py
```

Direct parity smoke was run with `/home/v/miniconda3/envs/dreamerv3/bin/python` because that env has no `pytest` installed. It checked LONG and SHORT opens on deterministic rows:

```text
normal_continuous: row 1040
session_open: row 1035
before_break: row 944
after_break: row 945
near_eod: row 1377
flip open short after long: rows 1040 -> 1041
```

Assertions passed:

```text
evaluator entry_timestamp == env trade timestamp
evaluator entry_execution_price == env open_price
evaluator entry_row == decision_row
LONG/SHORT direction mapping matches env
flip open leg matches signal-on-close evaluator
```

`dreamerv3` Python dependency state:

```text
pytest: unavailable
pyarrow: unavailable
fastparquet: unavailable
```

Because no parquet engine is available and base Python is forbidden, generated table artifacts use CSV sidecars.

## Attribution Run

No Dreamer recollection or retraining was done. The signal-close attribution reused:

```text
artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday/dreamer_episodes.csv
artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday/dreamer_actual_trades.csv
```

Command:

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/dreamer_checkpoint_audit.py \
  --run-logdir /data/logdir/action-mask-formal-202606122100 \
  --checkpoint /data/logdir/action-mask-formal-202606122100/ckpt/latest \
  --entry-eval-dir artifacts/entry_eval/entry_eval_jm_dreamer6m_2024_signal_close_v1 \
  --collect-mode per_day \
  --roles train,validation,test \
  --episodes 0 \
  --max-steps 120000 \
  --random-runs 200 \
  --output-dir artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday_signal_close
```

Output:

```text
artifacts/dreamer_checkpoint_audit/action-mask-formal-202606122100_latest_perday_signal_close/
```

## Overall Results

| Strategy | Trades | Net PnL | Expectancy | Profit Factor | Long | Short |
|---|---:|---:|---:|---:|---:|---:|
| Dreamer actual entry + Dreamer actual exit | 343 | 91512 | 266.80 | 7.02 | 229 | 114 |
| Dreamer entries + signal-close fixed exit, candidate exact | 343 | 24072 | 70.18 | 1.51 | 229 | 114 |
| Dreamer entries + signal-close fixed exit, entry-time join | 343 | 24072 | 70.18 | 1.51 | 229 | 114 |
| Dreamer entries + signal-close fixed exit, executable non-overlap | 227 | 13188 | 58.10 | 1.39 | 159 | 68 |
| Matched random same-count executable fixed exit, 200 runs | 227 avg | -9177.45 mean | NA | NA | NA | NA |
| Constrained oracle signal-close fixed exit | 432 | 191658 | 443.65 | inf | 208 | 224 |

Join diagnostics:

```text
candidate exact join: 343 matched, 0 missed
entry-time join: 343 matched, 0 missed
decision-time join: 343 matched, 0 missed
```

This answers the prior timing mismatch: under the matching signal-close evaluator, entry-time join is no longer missing trades.

## Split Results

| Split | Actual Trades | Actual Net | Candidate Fixed Trades | Candidate Fixed Net | Executable Trades | Executable Net | Matched Random Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 228 | 62022 | 228 | 20052 | 145 | 9570 | -7040.25 |
| validation | 58 | 21672 | 58 | 3252 | 48 | 4062 | -903.00 |
| test | 57 | 7818 | 57 | 768 | 34 | -444 | -1234.20 |

Candidate-level attribution is positive on train, validation, and test. Executable non-overlap fixed-exit attribution is positive overall and on validation, but test remains negative.

## Exit / Holding Contribution

Candidate exact attribution:

| Split | Actual Net | Signal-Close Fixed Net | Exit/Holding/Path Delta |
|---|---:|---:|---:|
| train | 62022 | 20052 | 41970 |
| validation | 21672 | 3252 | 18420 |
| test | 7818 | 768 | 7050 |
| overall | 91512 | 24072 | 67440 |

Interpretation:

```text
entry signal exists at candidate level, but most realized PnL still comes from Dreamer actual exit / holding / path behavior.
test actual PnL is +7818, while same entries under fixed exit are only +768 at candidate level and -444 after executable non-overlap.
```

## Canonical Next-Open Comparison

Previous canonical-next-open attribution used the same Dreamer actual replay but a different evaluator contract:

| Metric | canonical_next_open | signal_on_close_plus_spread |
|---|---:|---:|
| candidate exact join matched | 342/343 | 343/343 |
| entry-time join matched | 227/343 | 343/343 |
| candidate fixed net | 24858 | 24072 |
| candidate fixed validation | 3252 | 3252 |
| candidate fixed test | 1008 | 768 |
| executable fixed net | 14856 | 13188 |
| executable validation | 4368 | 4062 |
| executable test | -672 | -444 |
| matched-random executable mean | -8857.80 | -9177.45 |

Main difference:

```text
signal_on_close_plus_spread fixes timing alignment and entry-time join.
It does not materially change the admission conclusion: executable fixed-exit test remains negative.
```

## Answers A-F

A. Is `signal_on_close_plus_spread` exactly aligned with current Dreamer env fills?

```text
Yes. Env fills at row_C[current_step] +/- spread, timestamp=current_step.
The evaluator now uses decision_row=t, entry_row=t, close[t] +/- spread.
Parity checks passed for LONG, SHORT, break/session/EOD diagnostic rows, and flip open leg.
```

B. Does the checkpoint pass candidate-level entry attribution under this matching contract?

```text
Yes at candidate level.
343/343 trades matched.
overall candidate fixed-exit net = +24072
validation = +3252
test = +768
```

C. Does it pass executable non-overlap fixed-exit attribution on validation/test?

```text
No.
validation executable fixed-exit net = +4062
test executable fixed-exit net = -444
Formal admission still fails because test is not positive.
```

D. How much actual PnL is still exit/holding/path contribution?

```text
overall actual net = +91512
candidate signal-close fixed net = +24072
delta = +67440

test actual net = +7818
test candidate fixed net = +768
test delta = +7050
```

Most realized profit is still exit / holding / path contribution, not fixed-exit entry selection alone.

E. How different are results from canonical_next_open evaluator?

```text
Timing join is materially different: 227/343 entry-time matches under canonical vs 343/343 under signal-close.
Attribution PnL is directionally similar: both show candidate-level positive signal, both fail executable fixed-exit test.
```

F. Which contract should be used for future official training/evaluation?

```text
For the current Dreamer env behavior and existing checkpoint lineage:
  use signal_on_close_plus_spread as the matching official evaluator contract.

For a more conservative exchange-style live execution experiment:
  use canonical_next_open, but only after explicitly changing env timing and retraining/evaluating under that new contract.

Do not mix contracts in one admission decision.
```

## Final Classification

```text
Full-policy Dreamer under current signal-close contract: PROMISING
Candidate-level entry attribution: PASS / PLAUSIBLE
Executable fixed-exit validation/test admission: NOT PASS
Entry-only Dreamer admission: NOT PASS
Primary realized PnL driver: EXIT / HOLDING / PATH, with weak candidate-level entry signal
Timing artifact concern vs current env: resolved by matching contract
Timing difference vs canonical_next_open: still a separate contract decision
```

The next technical question is no longer “is entry-time join broken?” It is:

```text
Does the full Dreamer policy rely on a quota-consuming daily action template and exit/holding behavior,
or does it make market-dependent timing decisions that survive template/quota audits?
```

That belongs to Phase 3.5B. It should still avoid new indicators, retraining, Entry-only Dreamer, and env behavior changes until this attribution question is closed.

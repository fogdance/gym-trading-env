# Contract Audit

日期：2026-06-14

状态：review + RPC 修复记录。RPC payload/receiver 契约已补测试并修正运行时接收逻辑。

## Scope

本次审计把契约分成三类：

1. Env runtime contract：训练和回放直接依赖的 `reset()`、`step()`、`observation_space`、`action_mask`、ledger 和 live replay。
2. Data/feature contract：OHLCVI 标准化、期货 345 分钟时钟、market/agent feature spec、oracle 和 env obs 对齐。
3. Research/evaluation contract：Phase 0-3 entry capability evaluator、outcome、split、baseline、Gate 0-3。

不把 DreamerV3 另一个仓库的 masked actor 实现纳入本次代码审计；这里只记录本仓库对 Dreamer 暴露的 env 契约。

## Contract Inventory

| ID | Contract | Source of Truth | Implementation | Companion Tests | Status |
|---|---|---|---|---|---|
| C-01 | Env action API | `docs/action_mask_implementation_review.md`, `docs/action_mask_full_chain_implementation_plan.md` | `CustomTradingEnv.valid_actions = [SHORT, FLAT, LONG]`, `spaces.Discrete(3)` | `tests/contract/test_target_action_mask.py`, migrated contract tests | PASS |
| C-02 | `action_mask` order/type/nonempty | Action-mask docs | `_action_mask_from_table()` from `TargetTransitionTable` | `test_target_action_mask.py`, `test_observation_space.py` | PASS |
| C-03 | `obs_t.action_mask` and `step(action_t)` consume same decision | Action-mask implementation review | `_get_obs()` publishes table; `step()` consumes pending table and fail-fast on stale state | `test_step_fails_fast_when_observation_transition_table_is_stale` | PASS |
| C-04 | Allowed non-HOLD decision carries execution-ready quote | Action-mask plan section 5.3/5.5 | `TargetTransitionDecision.__post_init__`, `_quote_target_execution()`, `_commit_target_execution_quote()` | `test_transition_quotes_are_execution_ready_and_side_effect_free`, no-requote tests | PASS |
| C-05 | Rejected action is strict no-op | Action-mask plan section 5.7 | `step()` sets `executed_action=HOLD`; no ledger/position mutation | `test_invalid_actions_strict.py`, `test_masked_request_is_strict_noop_and_reports_decision` | PASS |
| C-06 | Execution failure is not action rejection | Action-mask review P1 | `action_rejected` and `execution_failed` separated in `step()` and info metrics | `test_execution_failure_is_separate_from_action_rejection` | PASS |
| C-07 | Transaction rollback covers ledger, positions, PnL, trade records, entry count | Action-mask plan section 5.6 | `_snapshot_execution_state()` / `_restore_execution_state()` | open/close/flip legacy and target quote failure tests | PASS |
| C-08 | Observation schema | Env code and feature spec | `Dict(market_seq, agent_state, action_mask, optional daily_context/daily_seq_7)` | `test_observation_space.py`, `test_obs_features_hold_only.py` | PASS |
| C-09 | Market feature field order/formulas | `tests/SPEC_FEATURES.md` and executable oracle | `market_features.py` | `test_market_features_match_spec.py`, `test_env_obs_match_spec.py` | PASS after doc drift fix |
| C-10 | `obs_V_t` formula | `docs/action_mask_implementation_review.md`, `tests/SPEC_FEATURES.md` | `clip(log1p(V/past_volume_baseline),0,3)*mask` | executable market oracle | PASS after doc drift fix |
| C-11 | Agent feature field order/formulas | `tests/oracles/agent_spec_oracle.py` | `agent_features.py`, `_refresh_agent_state()` | `test_agent_features_match_spec.py`, `test_env_obs_match_spec.py`, `test_agent_state_contract.py` | PASS |
| C-12 | Futures strict 345 clock/session mapping | `session_futures_strict.py` and feature spec | `strict_reindex_futures_345()` | `test_futures_strict_345.py`, `test_futures_sessions.py` | PASS |
| C-13 | OHLCVI/time ingestion | `ohlcvi_contract.py`, `time_contract.py` | normalize/sort/dedupe/tz conversion helpers | covered indirectly by market feature and replay tests | PASS, with limited direct tests |
| C-14 | Ledger/accounting conservation | ledger and env accounting code | `_post_atomic()`, ledger postings, position manager close/open | `test_bookkeeping_conservation.py`, `test_ledger_basic.py`, accounting integration | PASS |
| C-15 | Stop loss/take profit/external transitions are not agent action-mask legality | Action-mask plan exclusions | stop/take-profit applied after time advance | `test_stop_loss.py`, reward/invalid-action tests | PASS for existing scope |
| C-16 | Live action log replay determinism | replay integration contract | JSONL action log stores target action plus execution action; reset replay restores state | `test_env_replay_determinism.py` | PASS when external MySQL is reachable; skipped otherwise |
| C-17 | External RPC signal protocol | `rpc_protocol.py` / `_rpc_send_after_execute()` / Juejin `lan_trade` receiver | RPC uses execution `Action.value`, separate from env target action index; non-success actions are not live-executable | `test_rpc_payload_contract.py`, Juejin `test_lan_trade_rpc_contract.py` | PASS |
| C-18 | Entry evaluator config contract | `configs/entry_eval_jm_v1.yaml`, entry validation plan | `EntryEvalConfig.validate()` | `test_stop_distance_tick_mismatch_fails_fast` | PASS |
| C-19 | Entry outcome semantics | entry validation plan | `evaluate_entry()` | `test_entry_evaluator.py`, cost/collision/causality aliases | PASS |
| C-20 | Candidate exclusion: next valid bar, no break/session crossing, no cross-day | entry validation plan | `candidate_rows()`, `_next_valid_same_day()` | `test_candidates_exclude_segment_break_missing_slot_and_insufficient_margin`, EOD/max-hold tests | PASS |
| C-21 | Split purge/embargo | entry validation plan | `purged_fold_masks()` | `test_entry_split_purge.py` | PASS |
| C-22 | Non-overlapping oracle/random baseline | entry validation plan | `constrained_oracle()`, `matched_random_strategies()` | `test_entry_split_purge.py` | PASS |
| C-23 | Gate 2 report and no-RL gate | entry validation plan | `run_entry_capability.py`, `entry_analysis.py` | report artifacts and unit coverage around split/baselines | PARTIAL: report logic has less direct unit coverage |

## Detailed Review

### C-01 to C-07: Action Mask Runtime

Formal env action contract is target-position based:

```text
0 = SHORT
1 = FLAT
2 = LONG
```

`Action.HOLD`, `Action.LONG_OPEN0`, `Action.LONG_CLOSE0`, `Action.SHORT_OPEN0`, `Action.SHORT_CLOSE0` and `Action.FLIP_*` are internal execution/scenario labels. Tests may still use them through `action_to_index()` or `step_wrapper()`, but those names are not external `env.action_space` values.

Review result:

- Runtime code matches the target-position contract.
- Contract tests now exercise normal mask matrices, strict no-op rejection, stale table fail-fast, execution-ready quotes, no requote during `step()`, transaction rollback, and mask-respecting rollout.
- No P0 runtime drift found in this group.

### C-08 to C-13: Observation and Feature Contracts

Observation keys are:

```text
market_seq
agent_state
action_mask
optional daily_context
optional daily_seq_7
```

When `use_daily_context=false` and `use_daily_seq_7=false`, only the first three keys are emitted.

Review result:

- Executable feature oracle and env integration tests match current implementation.
- One human-readable spec drift was found and fixed: `tests/SPEC_FEATURES.md` still described old `obs_V_t = clip(V/I*100, 0, 2)`. The current contract is the past-volume-baseline log spike formula.
- `test_day_key_large_date_like_float_should_not_be_scientific` remains xfailed. This is acceptable only if large date-like floats are not used as `day_id`. If upstream can emit `20240520.0`, this becomes a real contract gap.

### C-14 to C-17: Accounting, Replay, RPC

Ledger/accounting tests cover balanced entries, fee/margin flow, open/close roundtrip, and rollback after commit failures.

Replay determinism is contract-level but depends on external MySQL data. In sandbox it may skip; with network/database access it should be run as an integration gate.

External RPC has a separate semantic contract from env action-space:

```text
env step action index: target-position index 0..2
RPC action_index: execution Action.value
RPC action_result: ForexCode.value; only SUCCESS(0) is live-executable
```

The RPC protocol comment previously implied env action-space indexing and `0..4`; that was corrected. `_rpc_send_after_execute()` now emits top-level `action_result` in addition to `meta.result`.

The Juejin receiver under `/home/v/Documents/work/juejin/ed27929a-30a5-11f0-a3a6-366f24a5ee2d/lan_trade` now enforces the same contract:

- `TradeSignal.from_dict()` accepts `exec_action` and falls back to `action_name`.
- `action_result != 0` is rejected before enqueue and also skipped defensively during drain.
- `FLIP_LONG_TO_SHORT` and `FLIP_SHORT_TO_LONG` map to explicit close+open execution sequences.
- Sequential live orders stop on an immediate `order_volume()` failure, so a failed close does not continue to send the follow-up open order.

### C-18 to C-23: Entry Capability Evaluator

Canonical evaluator contract is fixed:

```text
entry_delay_bars = 1
stop_distance_price = stop_distance_ticks * tick_size
take_profit_rr = 1.5
max_hold_bars = valid 1m bars
allow_entry_across_break = false
force_flatten_eod = true
intrabar_collision = stop_first
net_pnl after spread and fees is acceptance PnL
chronological walk-forward split with purge/embargo
```

Review result:

- Core evaluator behavior is covered: tick-size fail-fast, next-open execution, stop-first collision, cost parity with position math, valid-bar max hold, same-day EOD, no future data in features, deterministic dataset, purge/embargo, constrained oracle, and matched random count matching.
- Report/gate logic is less directly unit-tested than evaluator mechanics. The generated reports already encode Gate 2 FAIL and no-RL conclusion, but if this becomes a long-lived research pipeline, add dedicated tests for `classify_gate2()`, profit-factor status, alignment audit, and report markdown contents.

## Priority Findings

### P0

None found in current committed runtime contracts.

### P1

1. Add direct tests for Gate/report classification.
   The evaluator mechanics are well-covered; the report gating logic is mostly protected by artifact generation and indirect checks.

### P2

1. Decide whether large float `day_id` values are impossible. If not impossible, un-xfail and fix `_day_key_default()`.

2. Keep old `Action.*` scenario comments explicit in tests.
   They are acceptable only through `action_to_index()` or `step_wrapper()`.

## Current Verdict

Runtime env action-mask contracts are aligned with current code and tests.

The largest confirmed drift was documentation and RPC receiver semantics:

- `tests/SPEC_FEATURES.md` had the old `obs_V_t` formula.
- `rpc_protocol.py` comment still described obsolete action indexing.
- Juejin `lan_trade` receiver did not reject non-success signals and did not map `FLIP_*` execution actions.

These were corrected in this review pass.

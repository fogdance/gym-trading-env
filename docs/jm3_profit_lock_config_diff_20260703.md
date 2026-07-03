# JM3 Profit-Lock B Config Diff - 2026-07-03

Old env config:
`/data/logdir/action-mask-wf1-ms-seed-c-20260615/env_env_trading_stage1_jm_walk_forward_train_20240603_20250731.yaml`

New env config:
`configs/jm3_profit_lock_diag_seed_c_20260703.yaml`

Behavior-changing differences intentionally introduced:

| field | old | new | reason |
| --- | --- | --- | --- |
| `training.reward_function` | `futures_intraday_reward_function` | `futures_intraday_reward_profit_lock_function` | B-group reward under test |
| `trading.live_mode` | absent/default false | `false` | explicit live safety |
| `trading.live_replay_on_reset` | absent/default false | `false` | explicit live replay safety |

No obs, action space, action mask, execution model, fee/spread/session/EOD, stop/take-profit, data symbol, data interval, or max-entries fields were changed in this env config.

Smoke evidence:
- Instantiating from `/home/v/Documents/work/dreamerv3` loads `data/JM_WF1_TRAIN_20240603_20250731_1m.csv` with the unchanged symbolic data path.
- Reward resolved to `futures_intraday_reward_profit_lock_function`.
- `live_mode=false`, `live_replay_on_reset=false`, and `_order_client is None`.
- Runtime obs keys: `market_seq`, `agent_state`, `action_mask`.

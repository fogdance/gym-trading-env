# Dreamer Walk-Forward Seed Protocol

日期：2026-06-15

状态：开发期强约束。正式 walk-forward multi-seed 实验必须显式设置 `experiment_seed` 和 `dreamer.seed`；旧的 `--seed` 只作为兼容入口，且必须和 `dreamer.seed` 一致。

## 目标

walk-forward repeatability 实验必须把随机源拆开记录，避免以后只看到一个 checkpoint，却无法判断它来自哪个 Dreamer seed、env reset seed、replay seed 或 matched-random seed。

正式 seed 体系：

```yaml
experiment_seed: 101

dreamer:
  seed: 101

env:
  train_seed: 101101
  eval_seed: 0

replay:
  seed: 101202

audit:
  matched_random_seed: 20260615
```

如果 `env.train_seed` 为 `0` 或 `null`，启动时按 `experiment_seed * 1000 + 101` 派生。
如果 `replay.seed` 为 `0` 或 `null`，启动时按 `experiment_seed * 1000 + 202` 派生。

## Seed 含义

`experiment_seed`

实验级 seed id。用于派生 train env seed 和 replay seed，也用于报告聚合。

`dreamer.seed`

Dreamer/JAX seed。控制：

- JAX/Ninjax 参数初始化；
- policy/train/report PRNG；
- train-mode action sampling / exploration；
- imagination action sampling。

启动后会同步到 Dreamer 内部 `config.seed`，因为 Agent 仍读取 `config.seed`。硬约束：

- 如果只传旧 `--seed`，且 `dreamer.seed` 仍为默认 `0`，启动会用 `--seed` 填充 `dreamer.seed` 并打印 warning；
- 如果同时传 `--seed` 和 `--dreamer.seed`，两者必须一致；
- 如果两者不一致，启动 fail-fast；
- 正式 multi-seed 命令应显式传 `--dreamer.seed`，不要只依赖旧 `--seed`。

`env.train_seed`

训练环境 reset RNG seed。Dreamer Gymnasium wrapper 只在每个 env 实例第一次 reset 时调用：

```python
env.reset(seed=train_seed + env_index)
```

之后 episode reset 不再重复传 seed，让 `CustomTradingEnv.np_random` 自然推进。因此它只控制训练期间 `randomize_start` 的起点采样序列，不改变成交、ledger、action mask、reward 或执行合约。

`env.eval_seed`

固定记录为 `0`。validation/test 的 per-day checkpoint audit 会强制每个交易日的 start row，且 policy 用 eval `pred()`，不依赖随机 start sampling。该字段只用于审计记录。

`replay.seed`

Replay selector seed。控制 replay/batch sampling 顺序。未来 multi-seed 中它随 `experiment_seed` 改变，避免多个 Dreamer seed 共享完全相同 replay sampler 随机源。

`audit.matched_random_seed`

matched-random fixed-exit baseline seed。必须跨 Dreamer seed 固定，推荐统一：

```yaml
audit.matched_random_seed: 20260615
```

这样 seed_a / seed_b / seed_c 的随机基线可比。

## 不确定性边界

即使 seed 完整记录，GPU/JAX 训练仍可能存在非严格 bitwise determinism，特别是并行 actor、GPU kernel、prefetch/thread 调度和浮点归约顺序。repeatability 判断应看 3 个独立 seed 的统计稳定性，而不是要求逐步完全一致。

## Seed A/B/C 启动方式

正式启动推荐使用 Dreamer 仓库脚本：

```bash
/home/v/Documents/work/dreamerv3/tools/run_walk_forward_seed.sh seed_a 101
/home/v/Documents/work/dreamerv3/tools/run_walk_forward_seed.sh seed_b 202
/home/v/Documents/work/dreamerv3/tools/run_walk_forward_seed.sh seed_c 303
```

脚本统一派生并传入：

```text
train_env_seed = experiment_seed * 1000 + 101
replay_seed    = experiment_seed * 1000 + 202
matched_random_seed = 20260615
```

需要自定义 logdir 时：

```bash
/home/v/Documents/work/dreamerv3/tools/run_walk_forward_seed.sh seed_a 101 /data/logdir/custom-seed-a
```

等价完整命令如下，主要用于审计脚本行为或手工排错。

seed_a:

```bash
cd /home/v/Documents/work/dreamerv3
CONTRACT_ROOT=/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202

/home/v/miniconda3/envs/dreamerv3/bin/python dreamerv3/main.py \
  --configs action_mask_formal \
  --logdir /data/logdir/action-mask-wf1-ms-seed-a-20260615 \
  --env.gymnasium.config_path "$CONTRACT_ROOT/configs/env/jm_walk_forward_20240603_20251202_train.yaml" \
  --experiment_seed 101 \
  --dreamer.seed 101 \
  --env.train_seed 101101 \
  --env.eval_seed 0 \
  --replay.seed 101202 \
  --audit.matched_random_seed 20260615 \
  --audit.entry_eval_version jm_walk_forward_20240603_20251202 \
  --audit.entry_eval_config "$CONTRACT_ROOT/configs/entry_eval/jm_walk_forward_20240603_20251202_signal_close.yaml" \
  --audit.split_manifest_hash 9de809fa1aae612a4aeef453110dee10d7d4fb8f8e1c49c2c9474ddffec388ee \
  --audit.execution_timing signal_on_close_plus_spread
```

seed_b:

```bash
  --experiment_seed 202 \
  --dreamer.seed 202 \
  --env.train_seed 202101 \
  --replay.seed 202202
```

seed_c:

```bash
  --experiment_seed 303 \
  --dreamer.seed 303 \
  --env.train_seed 303101 \
  --replay.seed 303202
```

`env.eval_seed` 和 `audit.matched_random_seed` 保持不变。

## Audit 命令

checkpoint audit 只接受 matched-random 专用 seed：

```bash
CONTRACT_ROOT=/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202

/home/v/miniconda3/envs/dreamerv3/bin/python tools/dreamer_checkpoint_audit.py \
  --dreamer-root /home/v/Documents/work/dreamerv3 \
  --run-logdir /data/logdir/action-mask-wf1-ms-seed-a-20260615 \
  --checkpoint /data/logdir/action-mask-wf1-ms-seed-a-20260615/ckpt_retained/step_000000900000 \
  --entry-eval-dir "$CONTRACT_ROOT/artifacts/entry_eval/jm_walk_forward_20240603_20251202" \
  --env-config-path "$CONTRACT_ROOT/configs/env/jm_walk_forward_20240603_20251202_full.yaml" \
  --output-dir /data/logdir/audits/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/seed_a_900k_validation_signal_close \
  --collect \
  --collect-mode per_day \
  --roles validation \
  --random-runs 200 \
  --matched-random-seed 20260615 \
  --jax-platform cuda
```

`--seed` 不应再用于 checkpoint audit，避免误解为 Dreamer checkpoint seed。

## Multi-Seed Aggregate

多 seed 汇总使用：

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/report_dreamer_multiseed_repeatability.py \
  --experiment-name action-mask-wf1-multiseed-20260615 \
  --output-dir /data/logdir/audits/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615 \
  --seed-report seed_a=/data/logdir/audits/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/seed_a_selected_test_signal_close/summary.json \
  --seed-report seed_b=/data/logdir/audits/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/seed_b_selected_test_signal_close/summary.json \
  --seed-report seed_c=/data/logdir/audits/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/seed_c_selected_test_signal_close/summary.json
```

输出：

```text
summary.json
report.md
```

汇总报告必须逐 seed 保留 `seed_protocol`，并在表格中显示 Dreamer seed、train env seed、replay seed、eval env seed、matched-random seed 和 split manifest hash。

## Manifest 要求

以下文件必须记录 seed metadata：

- training logdir `config.yaml`
- `ckpt_retained/retention_manifest.json`
- checkpoint audit `summary.json`
- walk-forward multi-seed aggregate `summary.json` / `report.md`

必须包含：

```text
experiment_seed
dreamer_seed
train_env_seed
replay_seed
eval_env_seed
matched_random_seed
split_manifest_hash
entry_eval_version
execution_timing
seed usage fields:
  train_env_seed_usage
  replay_seed_usage
  eval_env_seed_usage
  eval_policy_mode
  matched_random_seed_usage
  gymnasium_constructor_seed_usage
```

## 验收

最小 smoke 不跑完整训练，只验证：

- seed resolver 将 `experiment_seed=101` 派生为 `train_env_seed=101101`、`replay_seed=101202`；
- seed resolver 将 `dreamer.seed` 同步到内部 `config.seed`；
- 同时设置不一致的 `--seed` 和 `--dreamer.seed` 会 fail-fast；
- Gymnasium wrapper 只在第一次 reset 传 `seed`；
- replay 构造收到 `replay.seed`；
- checkpoint retention manifest 写入 seed metadata；
- checkpoint audit summary 写入 seed protocol 和 split manifest hash；
- checkpoint audit 可从 `config.yaml` 或 `ckpt_retained/retention_manifest.json` 读取 seed protocol，且两者不一致时 fail-fast；
- per-day validation/test audit 仍使用强制 start row + eval `pred()`，不依赖随机起点。

训练起点随机性的 smoke 验收：

- 同一个 `experiment_seed` 连跑两次 tiny smoke，`randomize_start` 序列一致；
- 不同 `experiment_seed` 的 `randomize_start` 序列不同。

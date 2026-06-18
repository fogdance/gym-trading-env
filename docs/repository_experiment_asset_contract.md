# Repository Experiment Asset Contract

Date: 2026-06-18

Status: active refactor contract. Compatibility with the old mixed layout is not
required.

## Final Contract

`gym-trading-env` owns trading-domain source code: environment semantics, reward
code, data loading, entry-eval logic, split builders, audit tools, tests,
templates, and documentation.

`dreamerv3` owns model/training source code: Dreamer agent code, embodied
runtime, model configs, retention logic, and launch scripts.

Neither repository owns formal training artifacts. Formal experiment assets live
under an external experiment root that both repositories access by absolute
path. The default external root is:

```text
/data/logdir
```

In short:

```text
gym-trading-env code produces the experiment data contract.
the generated contract is stored under /data/logdir/trading_contracts/<split_name>.
dreamerv3 consumes the external contract to train or run a checkpoint.
```

## Repository Ownership

### gym-trading-env Repository Owns

```text
src/gym_trading_env/
  Env, reward, data loading, entry-eval, audit, and split-builder code.

tools/
  build_walk_forward_split.py
  run_entry_capability.py
  dreamer_checkpoint_audit.py
  dreamer_equity_curve_audit.py
  report_dreamer_multiseed_repeatability.py

configs/
  Human-authored templates and small examples only.

tests/
  Unit and integration tests.

docs/
  Contracts and implementation reports.
```

### dreamerv3 Repository Owns

```text
dreamerv3/configs.yaml
  Agent/model/training hyperparameters, replay, checkpoint retention, seed
  protocol, and guard settings.

dreamerv3/
  Agent/model code.

embodied/
  Runtime, replay, driver, checkpoint, and logging code.

tools/
  Training launchers only. They pass already-generated env config paths,
  entry_eval versions, split hashes, and checkpoint retention settings.

docs/
  Dreamer-specific training protocol notes.
```

## External Experiment Root Ownership

Default root:

```text
/data/logdir
```

Formal data contracts:

```text
/data/logdir/trading_contracts/<split_name>/
  data/raw/
    Raw CSV snapshot for this contract.

  data/generated/
    Deterministic train-only CSV and other derived files.

  configs/env/
    Generated CustomTradingEnv train/full/live/replay YAML.

  configs/entry_eval/
    Generated entry evaluator YAML.

  artifacts/walk_forward_splits/<split_name>/
    split_manifest.json
    split_manifest.sha256
    summary.json
    generated_files.json

  artifacts/entry_eval/<split_name>/
    candidates/outcomes/features/model reports, if generated.
```

Dreamer run directories:

```text
/data/logdir/<run_name>/
  metrics.jsonl
  checkpoint files
  replay state
  checkpoint retention copies
  run manifests
```

## Must Not Own

### dreamerv3 Must Not Own

```text
trading raw CSV data
generated train-only CSV data
env YAML configs
entry-eval YAML configs
split manifests
split hashes authored by hand
entry capability datasets
checkpoint attribution/equity audit logic
```

### gym-trading-env Repository Must Not Own

```text
formal raw CSV snapshots
formal generated train-only CSVs
formal split manifests
formal entry-eval datasets
checkpoint audit outputs
Dreamer training logs or checkpoints
```

Small unit-test fixtures and source-controlled template/example configs are
allowed.

## Standard Layout

Target layout:

```text
/home/v/Documents/work/gym-trading-env/
  src/
  tools/
  tests/
  docs/
  configs/                  # templates/examples only

/home/v/Documents/work/dreamerv3/
  dreamerv3/
  embodied/
  tools/
  docs/
  configs.yaml

/data/logdir/trading_contracts/rb8y_fixed_holdout_v1/
  data/raw/rb8y_1m.csv
  data/generated/rb8y_fixed_holdout_v1_train_1m.csv
  configs/env/rb8y_fixed_holdout_v1_train.yaml
  configs/env/rb8y_fixed_holdout_v1_full.yaml
  configs/entry_eval/rb8y_fixed_holdout_v1_signal_close.yaml
  artifacts/walk_forward_splits/rb8y_fixed_holdout_v1/

/data/logdir/action-mask-rb8y-pilot1m-seed-a-20260617/
  metrics.jsonl
  ckpt/
  ckpt_retained/
```

## Walk-Forward Split Contract

A formal walk-forward experiment must have exactly one source of truth:

```text
/data/logdir/trading_contracts/<split_name>/artifacts/walk_forward_splits/<split_name>/summary.json
```

The split builder must generate:

```text
split_manifest.json
split_manifest.sha256
train-only CSV
train env config
full eval env config
entry-eval config
summary.json
generated_files.json
```

Manual editing of these generated outputs invalidates the experiment unless the
builder is rerun and the hashes are updated.

## Required Split Properties

Formal split must satisfy:

```text
train days are strictly before validation days
validation days are strictly before test days
train / validation / test have no overlap
train-only CSV contains only split train days
full eval config points to the full raw CSV
entry-eval config points to the full raw CSV
split_manifest_hash is computed from split_manifest.json
Dreamer launch script records the same split_manifest_hash
```

If any of these fail, the run is not a formal holdout experiment.

## Current RB Target

First generated RB fixed holdout:

```text
raw data:   SHFE.RB 1m, 2017-12-25 .. 2025-12-02
train:      2017-12-25 .. 2024-12-31
validation: 2025-01-01 .. 2025-06-30
test:       2025-07-01 .. 2025-12-02
contract:   signal_on_close_plus_spread
```

The builder must derive actual trading-day membership from the raw CSV. If a
calendar date has no rows, it is absent from that split.

## Current Generated Contracts

As of 2026-06-18:

```text
/data/logdir/trading_contracts/jm_walk_forward_20240603_20251202
  product: DCE.JM
  train: 2024-06-03 .. 2025-07-31
  validation: 2025-08-01 .. 2025-08-29
  test: 2025-09-01 .. 2025-12-02
  split_manifest_hash:
    9de809fa1aae612a4aeef453110dee10d7d4fb8f8e1c49c2c9474ddffec388ee

/data/logdir/trading_contracts/rb8y_fixed_holdout_v1
  product: SHFE.RB
  train: 2017-12-25 .. 2024-12-31
  validation: 2025-01-02 .. 2025-06-30
  test: 2025-07-01 .. 2025-12-02
  split_manifest_hash:
    d92ea861774fe5ca3acf34fc90ff75605883a67b0e440a8129300dfe18c2ce49
```

## DreamerV3 Consumption Contract

Dreamer launch scripts may only consume generated outputs by absolute path:

```bash
CONTRACT_ROOT=/data/logdir/trading_contracts/<split>
ENV_CONFIG=$CONTRACT_ROOT/configs/env/<split>_train.yaml
ENTRY_EVAL_CONFIG=$CONTRACT_ROOT/configs/entry_eval/<split>_signal_close.yaml
SPLIT_MANIFEST_HASH=$(cat $CONTRACT_ROOT/artifacts/walk_forward_splits/<split>/split_manifest.sha256)
```

DreamerV3 must not hard-code train/validation/test date ranges.

DreamerV3 `walk_forward_guard` remains the runtime defense:

```text
builder creates consistent external artifacts
guard verifies the artifacts before training starts
```

## Phase Plan

### Phase 0: Contract

Create this document and review the ownership model.

Review gate:

```text
The contract states which repo owns each asset type.
The external experiment root is explicit.
The phase plan is explicit.
```

### Phase 1: Builder And CLI

Implement:

```text
src/gym_trading_env/research/walk_forward_split_builder.py
tools/build_walk_forward_split.py
tests/unit/test_walk_forward_split_builder.py
```

The builder must generate all split artifacts from one raw CSV and existing
template configs. The CLI default output root is external:

```text
/data/logdir/trading_contracts/<split_name>
```

Review gate:

```text
Unit tests prove chronological split, no overlap, reproducible hash, generated
train CSV, generated configs, and configured split manifest consumption.
```

### Phase 2: Generate RB Fixed Holdout v1

Run builder for RB and generate under:

```text
/data/logdir/trading_contracts/rb8y_fixed_holdout_v1/
```

Review gate:

```text
summary.json shows train/validation/test boundaries and no overlap.
Dreamer guard passes with the generated train env config and split hash.
No generated RB data/config/split artifacts are left in either source repo.
```

### Phase 3: Update DreamerV3 Launch Scripts

Modify Dreamer launchers to consume generated external configs and hashes.

Review gate:

```text
Dreamer scripts contain no manually-authored split dates or split hashes.
Dreamer scripts read CONTRACT_ROOT and generated absolute paths.
```

### Phase 4: Remove Old Mixed Ownership

Stop using repo-local formal data/config/artifact paths for new experiments.

Review gate:

```text
Formal run commands only reference /data/logdir/trading_contracts/<split_name>
and /data/logdir/<run_name>.
```

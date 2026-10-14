# S3-V2 shared entry and experiment configurations

## Tutor-CD Full E2 formal 1M (integration/s3-next)

`TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M` uses the accepted integration baseline,
the F0 composition `legacy_1m + tutor_cd`, existing Natural/Balanced/Pair loss
and E2 difficulty (`alpha=.8`, pair lambda `.1`, tau `1`, kappa `1`, cap `2`).
Beta stays `1`; RL/Critic stay frozen. It uses rebuilt market-minute v2 TRAIN
Env/Oracle/TeacherCatalog and parameter warm-start from the audited Online180K.
The 18-contract budget is 1M actual Env advances, 4 joint updates per 100 advances
(40000 updates), Replay capacity32768 and a full checkpoint every100K.

Frozen sibling worktrees: `research/fg-s3-tutor-full-train/`. Its new Run is
`/data/logdir/fg_single3/s3_v2/TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014`.
Prepare on the final clean commits before start. Use the same command with
`resume` to recover an interrupted Run; keep its frozen source/config unchanged.
This preparation does not start training. The user starts the formal Run:

```bash
cd /home/v/Documents/work/research/fg-s3-tutor-full-train/gym-trading-env
env PYTHONDONTWRITEBYTECODE=1 FG_S3_VISIBLE_GPUS=0 \
  WANDB_ENTITY=xkm104501-none WANDB_PROJECT=fg-single3-research WANDB_MODE=online \
  /home/v/Documents/work/research/fg-s3-wandb-env/bin/python tools/fg_s3_wandb.py launch \
  --no-evals \
  --run-dir /data/logdir/fg_single3/s3_v2/TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014 \
  --prepared "$PWD/artifacts/fg_s3_v2/prepared/TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014" \
  --trainer-cwd "$PWD" -- \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start \
  --config TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M --authorize-training
```

W&B tracks formal training only; `--no-evals` prevents routine evaluation imports.
The existing tracker supplies its internal `http://127.0.0.1:10808` proxy and
offline fallback. Preparation status/identities are in this worktree pair's
`training-lock.json`; see `docs/S3_TUTOR_FULL_1M_PREPARATION.md`.

A2, Tutor, B2 and C2 share `tools/fg_s3_v2_experiment.py` and this configuration
directory. Use `--config` to select an experiment; reuse `prepare/start/resume`
instead of creating another trainer or copying launch logic into artifacts.
Preparation resolves the audited source model and experiment settings using
`dreamerv3.configuration.read_config_yaml`; resume uses the same frozen snapshot.

## Tutor-CD 800K live DB simulation (FG701 SHORT)

Use the isolated `research/fg-s3-tutor-online/{gym-trading-env,dreamerv3}` pair,
created from both repositories' `integration/s3-next`. Select
`TUTOR_CD_800K_FG701_SHORT` through the same shared entry:

```bash
cd /home/v/Documents/work/research/fg-s3-tutor-online/gym-trading-env
/home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py simulate-online --config TUTOR_CD_800K_FG701_SHORT --check-only --run-dir artifacts/fg_s3_online/preflight_NEW_ID
/home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py simulate-online --config TUTOR_CD_800K_FG701_SHORT
```

The pinned checkpoint is the archived `step_000000800000`, verified against its
manifest and frozen model configuration. Existing `latest` points to 1M and is
not used. Source parameters and synchronized policy parameters are checked
byte-for-byte after exact loading. The TRAIN C1 rule hash pins the market clock,
reward and simulation economics: 300,000 CNY, three lots per ADD, max three lots,
TP5/SL30, lot multiplier20, margin10%, fee3 per lot per side and spread1.

Completed `CZCE.FG701` bars come from the existing `fut_bar_1m_v2` DB settings.
Historical bars supply the causal feature window without executing historical
trades. The online polling adapter reuses `FixedDirectionEntryEnv` accounting,
the original Driver and ActorPolicy (`eval` argmax), and `continue_data`.
No training, Oracle queries, teacher guidance or Replay collection occurs.
Signals older than90 seconds become WAIT, including the Actor's previous-action
carry. Missing session minutes stop the account with `DATA_GAP`.

The user confirmed that the existing enabled RPC endpoint connects to a simulated
account. Committed minute simulation fills map to existing `SHORT_OPEN0` /
`SHORT_CLOSE0` execution enums, including TP/SL exits. RPCs are emitted when the
next completed bar makes the simulated fill known; the receiver executes using
its current quote. Thus simulation prices and broker prices can differ. `/health`
and `enqueued` acknowledge the service/queue; broker fills are not reconciled by
this adapter. The preflight emits zero orders. The fixed FG701 direction lasts
until the user stops this run or the DB delisting date begins; no automatic
main-contract switch is inferred.

Outputs live under `/data/logdir/fg_single3/online_sim/` and include `runner.log`,
`status.json`, `actions.jsonl`, `rpc.jsonl`, `account_state.json` and `state.pkl`.
Resume the same configuration with `--resume-online`; it restores the Driver,
account, market history, policy carry and counter. Concurrent launch is refused
by a file lock. RPC errors stop with a retained outbox and
`RPC_RECONCILIATION_REQUIRED`; pending RPCs require reconciliation before resume.
Existing output directories and source training/archive artifacts are retained.

### Live console in tmux

Run the repository monitor from a development/integration worktree; keep the
running model's frozen source untouched:

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py monitor-online \
  --run-dir /data/logdir/fg_single3/online_sim/TUTOR_CD_800K_FG701_SHORT_20261014_v1
```

Healthy repeated samples are silent. A new decision prints one compact line;
each new open/close, RPC acknowledgement/failure, alert and recovery is shown
separately. Alerts repeat only after five minutes if still unresolved. Output is
flushed to stdout for tmux and appended to the run's `console.log`. An RPC
acknowledgement is labelled queued, not broker-filled. Trades between polls are
read from the complete action/RPC logs, with a saved observation boundary for
monitor restarts.

The monitor uses no model, GPU, training or order submission. Sampling defaults
to 30 seconds and continues until stopped; `--once`, `--interval`, `--until`
and JSON events via `--json` are optional. `--runner-pid PID` adds a process
health check; heartbeat freshness remains checked without it. Bounded audit
samples are written under `<run-dir>/monitor/`; `--output-dir` can select an
existing monitor directory during handover. A file lock prevents two monitors
using that directory. Raw `runner.log` remains the separate full audit stream.

## B2 Outcome-Aware diagnostics (market-minute V1)

Use the independent `research/fg-s3-b2-dev/` pair. New E/O and live DAgger labels,
H60 market bars, two lightweight outcome heads, and a bounded signed adapter on
the original Actor. Natural Oracle BC supervises the final masked distribution;
no Tutor ranking/difficulty or RL/Critic updates. Old Online180k is authenticated
parameter initialization only; optimizer, Gate, replay and progress are fresh.

The verified `B2_OUTCOME_GPU_TRAIN_10K_V3` Run completed500→1000 restore and10K,
then fixed FG409 Actor-only5K. Its bounded budget was frozen before startup.
Original/v2 preparations retain rejected optimizer-schema startup history;
v3 derives the exact Ninjax-encoded new optimizer leaves. Existing outputs are
not overwritten; create a new configuration/run name for another experiment.

```bash
cd /home/v/Documents/work/research/fg-s3-b2-dev/gym-trading-env
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config B2_OUTCOME_GPU_TRAIN_10K_V3
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config B2_OUTCOME_GPU_TRAIN_10K_V3 --stop-at 500
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config B2_OUTCOME_GPU_TRAIN_10K_V3 --stop-at 1000
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config B2_OUTCOME_GPU_TRAIN_10K_V3
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py evaluate --config B2_OUTCOME_GPU_TRAIN_10K_V3 --eval-steps 5000
```

For a fresh standalone1K run use `B2_OUTCOME_GPU_SMOKE_1K` with the same entry.
The original B2 V0 worktree rejects formal scope; the Full version and its
separate 1M preparation are described below. Actual V0 results and limitations are in
`docs/B2_IMPLEMENTATION_REPORT.md`: engineering checks passed, DEV remained All-WAIT.
Tutor's freeze and `/tmp/done_cd` release remain a separate procedure requiring
its own regenerated data and real training/restore verification.

## Tutor E2 bounded dynamic curriculum (10K authorized)

Use the isolated sibling pair `research/fg-s3-tutor-e2-dev/`; the earlier static
Tutor and frozen A2 pairs remain unchanged. `TUTOR_CD_E2_GPU_TRAIN_10K` starts
independently from the admitted original Online180k checkpoint. It keeps beta=1,
alpha=.8 and uniform pair lambda=.1, and uses kappa=1, weight cap=2. Difficulty is
computed from current Actor probabilities, detached, and normalized inside each
ADD/WAIT class of balanced BC. Natural BC remains unchanged; missing classes
fall back to natural CE. No extra replay, curriculum cache, RNG or optimizer.

`tutor/balanced_ce` continues to describe unweighted class CE;
`tutor/hard_balanced_ce` records the actual weighted balanced objective.
The existing model settings/run identity/checkpoint freeze kappa and the cap.
Resume requires the same prepared snapshot and code; hyperparameter changes
require a new run. This 10K diagnostic does not authorize Tutor formal 1M.

```bash
cd /home/v/Documents/work/research/fg-s3-tutor-e2-dev/gym-trading-env
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config TUTOR_CD_E2_GPU_TRAIN_10K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config TUTOR_CD_E2_GPU_TRAIN_10K --stop-at 5000
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config TUTOR_CD_E2_GPU_TRAIN_10K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py evaluate --config TUTOR_CD_E2_GPU_TRAIN_10K --eval-steps 5000
```

Teacher-guided TRAIN TP5 counts remain separate from independent Actor-only DEV.
Actual results: `docs/FG_S3_Tutor_E2_10K_20261013.md`.

## Tutor-CD V0 (independent worktree)

Use `research/fg-s3-tutor-dev/` with its own sibling Dreamer. Tutor reuses the
original A2 WM/Replay/Sidecar/FullCheckpoint; one Actor forward supplies natural
BC, balanced BC (falls back when a class is absent) and same-episode ADD/WAIT
ranking. Critic and RL stay frozen even when the shared Gate becomes READY.
Defaults: alpha=0.8, pair lambda=0.1, tau=1, hard_kappa=0. The frozen settings
are loaded identically for training, resume and Actor-only DEV evaluation.

The 1K smoke uses the original admitted Online180k source and beta=1 for certified
TRAIN coverage. Teacher-controlled trades do not demonstrate autonomous trading.
Only diagnose on GPU within the user's resource permission; leave the frozen A2
worktree and run untouched. Tutor formal scope is rejected even with authorization.

```bash
cd /home/v/Documents/work/research/fg-s3-tutor-dev/gym-trading-env
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config TUTOR_CD_V0_GPU_SMOKE_1K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config TUTOR_CD_V0_GPU_SMOKE_1K --stop-at 500
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config TUTOR_CD_V0_GPU_SMOKE_1K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py evaluate --config TUTOR_CD_V0_GPU_SMOKE_1K --eval-steps 1000
```

Preparation/start refuse existing outputs. Repeat experiments require new names;
resume validates the original frozen code, hyperparameters and complete checkpoint.
The `TUTOR_CD_V0_GPU_TRAIN_10K` configuration is a template, requiring separate
user permission before execution. The original static V0 worktree rejects
nonzero hard_kappa; the independently authorized E2 worktree supports it above.
Tests and measured results are in `docs/FG_S3_Tutor_CD_V0_Implementation_20261013.md`.

## A2 beta schedule and 18-contract path (2026-10-13)

`A2_GPU_FORMAL_1M_BETA` uses the original ALL_TRAIN pool and Replay Online180k
source. Beta is 1 through 10K real Env advances, then decreases linearly to
0.98 at 100K and 0.90 at 1M. Its progress, current value, points, RNG and pending
execution are saved together by the existing FullCheckpoint. Resets and model
updates do not advance beta. Invalid Oracle answers remain excluded from BC.

Use the frozen pair in `research/fg-s3-v2-train-beta/`; preparation is saved there.
This is a new run; the old fixed-beta preparations/runs are retained. Formal
training still requires user authorization and has a 1M **environment-step** budget:

```bash
cd /home/v/Documents/work/research/fg-s3-v2-train-beta/gym-trading-env
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config A2_GPU_FORMAL_1M_BETA --authorize-training
# Same-run recovery, with its saved schedule and full training state:
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config A2_GPU_FORMAL_1M_BETA --authorize-training
```

Checkpoints are saved every 100K steps. Review Actor-controlled ADD, certified
label coverage and entry outcomes at 100K. `--stop-at 100000` can make this a
planned review boundary; do not mechanically wait for 1M if ALL_WAIT, lost
supervision or poor entries persist. Changing a frozen schedule/source is a new
experiment, not editing old snapshots to bypass recovery checks.

`A2_GPU_ALL_TRAIN_BETA_8K` is the bounded GPU rotation/recovery diagnostic:
ALL_TRAIN, the same original scheduler with 200-bar slices, and an accelerated
beta time axis (1 through 3600, 0.98 at 5200, 0.90 at 8000). It exercises all 18
members and saves/exits at 5000 before a separate `resume` to 8000. Its results
do not certify the formal 10K/100K learning milestones or Actor-only performance.

## Previous P0 A2 training preparation (2026-10-13)

The current measured version uses `_FAST` configurations and frozen pair
Gym `b71250c` / Dreamer `e64004e` in `research/fg-s3-v2-train/`.
GPU recovery and the independent 10000-step/400-update run passed;
Actor-only DEV remains ALL_WAIT. Results, throughput and the prepared formal
command are in `docs/FG_S3_V2_P0_GPU_10K_Training_20261013.md`.
Earlier configurations and interrupted runs remain historical artifacts.

Use `A2_P0_GPU_SMOKE_1K` for GPU save/interruption/resume, then the independent
`A2_P0_GPU_TRAIN_10K` diagnostic. Both start from the original admitted Online180k
checkpoint. They use the original TRAIN account scheduler, WM + equal Oracle BC,
and beta=1 to retain the existing funding certificate; executed scores are
teacher-guided. Actor proposals and BC ADD recall are reported separately.

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config A2_P0_GPU_SMOKE_1K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config A2_P0_GPU_SMOKE_1K --stop-at 500
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config A2_P0_GPU_SMOKE_1K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config A2_P0_GPU_TRAIN_10K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config A2_P0_GPU_TRAIN_10K
```

Replay capacity bounds native sampling. Collector snapshots retain teachers for
every physical Replay row, plus pending execution; old decisions remain in
`rollout/teacher_journal.jsonl`. Resume checks native owners and archives any
uncheckpointed journal tail before continuing. This is practical consistent
recovery, not a claim of byte-identical uninterrupted GPU execution.

`A2_P0_GPU_FORMAL_1M` is a preparation template over the original TRAIN contracts.
It uses `/data/logdir/fg_single3/s3_v2/` and requires `--authorize-training` for
start/resume. **Do not start it without the user's separate formal authorization.**
A2-to-C2 source migration, R1 and B2 remain deferred.

## Online A2 / C2 GPU diagnostics

Use `tools/fg_s3_v2_experiment.py` for `prepare / start / resume / evaluate / report`.
The configuration selects A2 (WM + equal Oracle BC) or C2 (WM + equal Oracle BC,
original RL only after the shared Single3 Gate opens). No network layers are added.
Model work requires GPU; the entry restarts only its own process without an
inherited `LD_LIBRARY_PATH` to prevent the independently reproduced system/Conda
cuBLAS library mix. It does not change CUDA packages or system settings.

`A2_GPU_FG201_1K.yaml` uses the admitted Online180k model, real FG201 TRAIN,
beta=0.5, lambda=1 and 1000 actual Env advances. Every 250 advances it performs
two joint WM/Actor updates. Global model update counts inherit the source;
this new run's Env and local update counts start at zero.

Run from this Gym worktree with its sibling Dreamer worktree:

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config A2_GPU_FG201_1K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config A2_GPU_FG201_1K --stop-at 500
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config A2_GPU_FG201_1K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py evaluate --config A2_GPU_FG201_1K --eval-steps 1000
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py report --config A2_GPU_FG201_1K
```

Preparation records the exact experiment YAML, resolved model configuration,
source admission, Oracle identity, both Git versions and code hashes. Start
refuses an existing run; resume validates its own full checkpoint and restores
the model, optimizer, Gate, Env/Driver/carry, Replay, committed teacher sidecar,
pending action, stream and RNG. It does not reload the source as a fresh transfer.
Changed code/configuration needs a new preparation/run name; old outputs remain.

Actor-only evaluation uses the existing FG409 DEV fold without querying Oracle
or reading teacher labels. TRAIN collection diagnostics include mixed actions
and are not an independent Actor-only result. Diagnostic configurations limit
training to 65–10000 actual Env advances and at most 1000 updates. Formal training
uses a separate scope, path and authorization flag as described above.
TEST and R1 comparisons are outside the current implementation scope.

## Tutor Full E2 / B2 Full (2026-10-14)

Both new diagnostic configurations use the audited Online180k parameter
warm-start, the existing market-minute v2 TRAIN inputs, FG201, beta=1, and a
10,000 **environment advance** budget with 400 joint updates. They have separate
preparation/Run names ending in `v3`; development-only v1/v2 preparations are kept.

- `TUTOR_CD_FULL_E2_MM_10K`: Natural/Balanced/Pair, alpha=.8, Pair=.1,
  hard_kappa=1, hard_wmax=2; Actor term gradients sampled every100 updates.
- `B2_FULL_MM_10K`: event/early loss=.1 each; preheat100, ramp100,
  minimum100 valid event updates; signed adapter cap2. A1K engineering run
  remains at mix0 by design. Values are research recommendations, not optima.

The current task authorizes CPU preparation/testing, **not these GPU commands**:

```bash
cd /home/v/Documents/work/research/fg-s3-full-integration-dev/gym-trading-env
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
"$PY" tools/fg_s3_v2_experiment.py prepare --config B2_FULL_MM_10K
"$PY" tools/fg_s3_v2_experiment.py start --config B2_FULL_MM_10K --stop-at 1000
"$PY" tools/fg_s3_v2_experiment.py resume --config B2_FULL_MM_10K
"$PY" tools/fg_s3_v2_experiment.py evaluate --config B2_FULL_MM_10K --eval-steps 5000
```

`prepare` uses CPU and refuses to rewrite an existing snapshot; the delivered
integration worktree already has final preparations, so start with `start` after
Review/authorization. Replace the configuration name with Tutor's for that route.
Every config/byte change requires a new preparation identity, not a rewritten
old receipt. Source transfer resets optimizer/local clocks for the changed
market-minute economics and preserves original model provenance.

CPU post-processing comes **after the causal Actor-only predictions are sealed**:

```bash
"$PY" tools/fg_s3_v2_experiment.py score-predictions \
  --eval-dir artifacts/fg_s3_v2/runs/B2_FULL_MM_10K_v3/evaluations/actor_only_REPLACE_WITH_ID \
  --output-dir artifacts/fg_s3_v2/runs/B2_FULL_MM_10K_v3/prediction_evaluations/dev_REPLACE_WITH_NEW_ID
```

Scoring validates the actual Driver/account replay, original candidate/funding
proof, and hashes. DEV labels are evaluation-only and never enter the policy,
training Replay or teacher sidecar. Historical evaluations require external
output directories. Old missing logits stay null. New adverse-time fields use
`S3_EVAL_METRIC_V2`; `early_mae10` remains legacy whole-hold.

Reuse `tools/fg_s3_wandb.py launch` with the original trainer command for
automatic tracking; `import-evals` also discovers completed prediction receipts
under `prediction_evaluations/*/receipt.json` independently of Actor-only imports.
External receipts can be selected using `--prediction-report`. W&B's internal
default proxy is `http://127.0.0.1:10808`; no extra shell proxy settings are needed.
The original trainer environment/network behavior is preserved. Online failure
falls back to offline, and the existing `sync` workflow resumes publication.
The project remains `fg-single3-research`. Old800K milestone work is excluded.

Exact commands, evidence and the T01–T12/B01–B13 matrix are in
`docs/FG_S3_Full_Implementation_Report_20261014.md`.

## B2 Full formal 1M preparation (2026-10-14)

`B2_FULL_MARKET_MINUTE_FORMAL_1M` uses the same shared entry, ALL_TRAIN
18-contract account scheduler, rebuilt market-minute v2 Oracle and shared
TeacherCatalog SourceAdmission. Each contract delegates to the existing
OutcomeCatalog; the pool adds routing and identity checks only.

The budget is 1,000,000 real Env advances, four joint WM/BC/Outcome updates per
100 advances (40,000 local updates), and a complete checkpoint every100K.
Beta stays1, warmup/ramp/minimum event updates stay100/100/100; beta decay is a
separate research experiment. RL/Critic remain frozen. Teacher-controlled trade
results are not Actor-only performance.

The delivered CPU-prepared frozen pair is `research/fg-s3-b2-full-train/`.
The Run directory is new and starts only when the user executes this command:

```bash
cd /home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env
env PYTHONDONTWRITEBYTECODE=1 WANDB_ENTITY=xkm104501-none WANDB_PROJECT=fg-single3-research WANDB_MODE=online \
  /home/v/Documents/work/research/fg-s3-wandb-env/bin/python tools/fg_s3_wandb.py launch \
  --run-dir /data/logdir/fg_single3/s3_v2/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014 \
  --prepared "$PWD/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014" \
  --trainer-cwd "$PWD" -- \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start \
  --config B2_FULL_MARKET_MINUTE_FORMAL_1M --authorize-training
```

For interrupted same-run recovery use the identical launch command with
`resume` instead of `start`. Independent Actor-only DEV uses the same entry's
`evaluate --config B2_FULL_MARKET_MINUTE_FORMAL_1M --eval-steps 5000`; follow it
with sealed-prediction CPU scoring and the existing W&B `import-evals` command.
No new Full GPU training or formal1M has been executed by the developer.
See `docs/FG_S3_B2_Full_1M_Preparation_20261014.md` for verified source/data
identities, CPU pool/restore evidence and exact preparation hashes.

`gpu_check` runs a small original-network fixture on CUDA for gradient ownership,
frozen Critic/momentum, empty BC, C2 CLOSED/controlled READY and lambda=0. It is
an automated contract test, not production-market or twenty-window Gate acceptance.

Additional bounded configurations use the same entrypoint:

- `C2_GPU_FG201_500`: 500 real TRAIN advances and four updates with the shared
  production Gate CLOSED; READY is not forced for a real run.
- `A2_GPU_FG201_1K_TEACHER`: 1000 real TRAIN advances, beta=1, four updates every
  100 advances (40 total). This keeps early collection inside the certified
  funding domain; all executed performance is teacher-guided. Its completed
  Actor-only DEV evaluation is All-WAIT, not a successful trading model.

Actual metrics, limitations and commands are in
`docs/FG_S3_V2_Live_Learning_Implementation_20261013.md`.

## Historical D0 CPU prototype

Single entry: `tools/fg_s3_dagger_d0.py`, selected with `--config`. D0 is limited
to 100–1000 actual cursor advances, uses only TRAIN, and never starts GPU or 1M
training. Current configuration is `D0_FG201.yaml`, C1 reward and gamma 332/333.

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fg_s3_d0_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_dagger_d0.py collect \
  --config configs/experiments/fg_s3_v2/D0_FG201.yaml --mode mixed --env-steps 1000
```

Modes: `wait`, `scripted`, `mixed`. Scripted proposals are a seeded causal
simulator, not a trained Actor. Mixed execution uses beta=0.5; unproved states
continue with Actor proposals and have invalid teacher supervision.

Use `--save-at 400` to preserve a collection checkpoint and later
`--resume-collection <run>/collection_checkpoints/env_000400` in a **new** output
directory. This restores the original Driver, native Replay, RNG and sidecar
pending state. It is a D0 collection checkpoint without model parameters and
cannot initialize a Dreamer model. Model resume belongs to FullCheckpoint.

`verify_oracle` independently replays original-Env TRAIN suffix intervals.
`model_smoke` admits the immutable original Online180k source, runs 100 CPU
mixed steps with Oracle beta=1 and one original World Model update. It does not
implement A2 training, and does not update the source checkpoint.

Outputs are new directories under `artifacts/fg_s3_v2_d0/`; existing outputs
are refused. No changes are made to V1 configurations or frozen train worktrees.
# F0 composition configuration

`FG_S3_V2_COMPOSITION_CONFIG_V1` uses only `composition.encoder/teaching`; never
write a top-level `mode` alongside it. Historical mode configs keep their original
schema and frozen Run identity. The derived runtime mode is an adapter for the
existing algorithms. No additional BC, Tutor or Outcome loss is introduced.

`composition_fragments/` contains all eight minimal syntax examples. These are
description fixtures, **not complete training configs**. E and Tutor+B2 remain
NOT_IMPLEMENTED and cannot prepare/start/resume/evaluate.

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py describe-composition \
  --config configs/experiments/fg_s3_v2/composition_fragments/e_multiscale_1m15m__b2.yaml
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config F0_B2_LEGACY_1K
```

`F0_A2_LEGACY_1K`, `F0_TUTOR_LEGACY_1K` and `F0_B2_LEGACY_1K` are independent
diagnostic configs with the original learning parameters and 1000 Env advances.
Their common start/resume/evaluate entry remains `tools/fg_s3_v2_experiment.py`;
this F0 task runs CPU admission and tests only, no GPU training. Implementation,
identity contracts and C01-C12 evidence: `docs/S3_F0_COMPOSITION_IMPLEMENTATION.md`.

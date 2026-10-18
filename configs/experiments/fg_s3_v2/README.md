# S3-V2 shared entry and experiment configurations

## E + Tutor-CD Full formal 1M (2026-10-16)

`E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_20261016` preserves the existing
Full E2 recipe and uses the byte-equivalent E collection built for the throughput
optimization. Fresh seed0 size50m, 18 TRAIN contracts, beta1, BC1, alpha.8,
pair.1/tau1/kappa1/cap2, 1M Env advances, 40000 updates, Replay32768 and
checkpoints every100K. Training uses the separate frozen sibling worktrees
`research/fg-s3-e-tutor-train-20261016/`. Prepare with the shared entry and this
config; start/resume from its immutable prepared snapshot with
`--authorize-training`. The exact prepared path and W&B launch command are
saved in that pair's `training-lock.json`. CPU preparation is not GPU acceptance.

## E-only formal 1M (2026-10-16)

`E_ONLY_MARKET_MINUTE_FORMAL_1M` uses `e_multiscale_1m15m + teaching: none`,
fresh seed0 size50m initialization, 18 TRAIN contracts, 1M actual Env advances,
40000 joint updates, Replay32768 and FullCheckpoint every100K. Beta and BC are
zero; Oracle label queries, takeover, A2/Tutor/B2/Quality objectives and heads
are disabled. Existing catalog/Oracle receipts authenticate the original data
and reward; their labels are not queried or used for learning. The original
Single3 availability Gate remains closed at initialization; WM learns first,
then the original imagined Actor/Critic RL starts only after the Gate passes.
The original C1 reward and market-minute clock remain in use.

Use the independent frozen pair `fg-s3-e-only-train-20261016` and shared
`prepare/start/resume/evaluate/report` commands. Exact configuration, evidence,
preparation status and W&B launch/resume commands are recorded in
`docs/S3_E_ONLY_FORMAL_1M_20261016.md`. No A2/Tutor/B2 experiment is launched by
this configuration. CPU acceptance does not certify a production Gate opening,
GPU arithmetic, or strategy performance.

## E + A2 / E + B2 Full formal entry (2026-10-16)

`E_A2_MARKET_MINUTE_FORMAL_1M` and `E_B2_FULL_MARKET_MINUTE_FORMAL_1M` use
fresh seed0 size50m E initialization, the authenticated current TRAIN collection,
all 18 TRAIN contracts and the existing 1M/40000-update schedule. B2 retains the
accepted Full head/adapter recipe. Both modes completed real CPU acceptance,
GPU 1000 updates and exact independent cold resume before readiness release.
Tutor-CD+B2 remains NOT_IMPLEMENTED. Historical `*_WAITING` placeholders and old
acceptance snapshots are retained; use the new formal configurations for new runs.

Prepare/start/resume use `tools/fg_s3_v2_experiment.py`. Formal start/resume still
require explicit user authorization and `--authorize-training`. The checked
preparations belong to the separate frozen `fg-s3-e-ab-release-20261016` pair;
run them from that pair, preserving its code and configs. No formal 1M was started.
Receipts and source/checkpoint evidence: `docs/e_a2_b2_admission_20261015/REPORT.md`.

## E + Tutor-CD Full formal 1M

`E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M` selects the native E encoder and the
accepted Tutor Full recipe (alpha=.8, pair=.1, tau1, kappa1, cap2), fresh size50m
initialization, all TRAIN contracts, beta1, 1M market minutes, 40000 joint updates,
Replay32768 and full checkpoints every100K. The user explicitly released this
entry on2026-10-15; incomplete GPU smoke is not recorded as passed. Source
authentication, the original trading Gate and `--authorize-training` remain.
See `docs/e_native_env_20261015/FORMAL_ENTRY_RELEASE.md` for exact commands.

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

## E-Model V1：早期占位配置（历史记录）

`E_MODEL_V1_A2_WAITING`、`E_MODEL_V1_TUTOR_CD_WAITING`、`E_MODEL_V1_B2_WAITING`
使用新的 encoder 轴、原 teaching 轴及 `direct_init / model_init(seed0,size50m)`。
三份配置的 `e_data_manifest` 是待真实联调的占位路径，不能用于训练。
这些占位来源不能用于 prepare/start/resume/evaluate；describe 可只读查看组合。
当前正式能力与真实来源配置见本文件顶部，不能将占位路径当作已认证数据。

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py \
  describe-composition --config E_MODEL_V1_B2_WAITING
```

初始 CPU 模型验收在共享内部核心与确定性合成 provider 上执行，当时未创建正式 Prepared/Run。
接口、EM01–EM18 及后置 Review 要求见 `docs/S3_E_MODEL_V1_INTERFACE.md` 和
`docs/S3_E_MODEL_V1_IMPLEMENTATION_REPORT.md`。不把 CPU Checkpoint 当正式 E Run。

## Native E real TRAIN acceptance (2026-10-15)

`E_NATIVE_{A2,TUTOR,B2}_{CPU,GPU}_20261015_v3` uses the shared entry and
native Env E observation profile. Admission verifies the original immutable
18-contract TRAIN collection; these bounded runs select FG201 and FG301 to
exercise account rotation and cold restore. The profile authenticates source
metadata; it does not load a second market DataFrame or wrap the Env.
Beta1 supplies certified TRAIN supervision for the engineering check; executed
trades are teacher-controlled. Both devices retain the canonical Single3 Gate.

CPU uses an explicitly small network for six joint updates over 2200 real Env
advances. GPU uses the canonical size50m preset for 1000 joint updates over
1000 advances. These protocols retain `waiting_e` and cannot authorize formal
1M. Every retry requires a fresh configuration/preparation/run name. The GPU
configuration requires matching completed CPU evidence before preparation.
These are historical bounded configurations; current production readiness is
documented in the formal-entry sections above.

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py admit-e-prepare --config E_NATIVE_A2_CPU_20261015_v3
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py admit-e-start --prepared artifacts/fg_s3_v2/prepared/E_NATIVE_A2_CPU_20261015_v3 --stop-at 1100
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py admit-e-resume --prepared artifacts/fg_s3_v2/prepared/E_NATIVE_A2_CPU_20261015_v3
```

Resume first verifies the next sample/update in an isolated child process,
then restores the same immutable checkpoint in a fresh training process. It
does not rewind a running Prefetch worker. Versions v1/v2 retain failed trials.
For GPU use the matching GPU configuration and stop at500 before a separate
resume to1000. Raw command logs, failures, source identities and actual PASS
receipts are archived in `docs/e_native_env_20261015/`; configuration existence
alone is not an acceptance result. Formal1M remains a separate Review gate.


## Quality-first V1 (2026-10-15)

2026-10-16: merged into `integration/s3-next`; see
`docs/S3_NEXT_QUALITY_FIRST_INTEGRATION_REVIEW_20261016.md` for both-parent
review and actual CPU receipts. Quality uses the admitted legacy encoder;
E+Quality remains unimplemented. Reprepare in a new frozen workspace before
any separately authorized GPU/research run.

2026-10-16 review fixes: see `docs/FG_S3_QUALITY_FIRST_REVIEW_FIXES.md`.
Current isolated verification uses `QUALITY_FIRST_REVIEW_CPU_FG201_v2`
(100 CPU steps), `QUALITY_FIRST_REVIEW_POOL_CPU_v2` (two-contract CPU
preparation/profile), and `QUALITY_FIRST_REVIEW_HEADS_GPU_1K_v2` (CPU
preparation only; GPU has not started). The corresponding `v1` files and
artifacts retain development evidence; source changes invalidate their
preparation. Every invocation must use its own experiment/path or resume
exactly the same frozen snapshot.

The accepted Gym patch and real FG201 loss-account evidence are in
`docs/quality_first_patch_review_20261015/ACCEPTANCE.md`. The implementation
record is `docs/FG_S3_QUALITY_FIRST_IMPLEMENTATION.md`.

`QUALITY_V2_CPU_SMOKE` and `QUALITY_V2_GPU_1K` now resolve to the user-confirmed
Quality-first stage: `quality_stage: HEADS_ONLY`, original C1 Reward, `beta: 0`,
`bc_lambda: 0`. Only Outcome/Quality auxiliary heads train. The inherited
Encoder/RSSM/Actor/Critic and retained Full Actor/Q modules are frozen. The
quality target is B(t,d); MAE is a separate target, with no score deduction.
`QUALITY_FIRST_HEADS_PILOT_10K` is the corresponding proposed small research
budget. GPU start/resume requires separate authorization and the existing
`--authorize-training` flag, including diagnostic budgets.
`QUALITY_FIRST_EVENT_ONLY_PILOT_10K` and
`QUALITY_FIRST_SCORE_ONLY_PILOT_10K` isolate the auxiliary tasks. All preserve
the original C1 Reward; the optional Quality TP Reward is a separate later axis.

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py prepare --config QUALITY_V2_CPU_SMOKE
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py start --prepared artifacts/fg_s3_v2/prepared/QUALITY_FIRST_HEADS_CPU_SMOKE_v1
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py resume --prepared artifacts/fg_s3_v2/prepared/QUALITY_FIRST_HEADS_CPU_SMOKE_v1
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py evaluate --prepared artifacts/fg_s3_v2/prepared/QUALITY_FIRST_HEADS_CPU_SMOKE_v1 --eval-steps 100 --output-dir <new-eval-dir>
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py score-predictions --eval-dir <sealed-eval-dir> --output-dir <new-posthoc-dir>
```

Use a new experiment/prepared/run name if a directory already exists. Source
changes invalidate old preparation; preserve those receipts and prepare anew.
The CPU-only `tools/fg_s3_quality_v2.py inspect --prepared ... --output-dir ...
--steps 250 --beta 0` builds an independently signed TRAIN quality Catalog
from a scripted native trajectory, without neural training. Its counterfactual
profits overlap and must not be summed as executable PnL.

The retained `QUALITY_V2_{B1,B2,B3,FULL}_PILOT_10K` files are later controlled
experiment templates. They intentionally fail preparation while prediction
admission is missing. Enabling `CONTROLLED_FULL` requires a checkpoint-bound,
disjoint TRAIN holdout evaluation with measured improvements over the TRAIN
prior and reference, plus independent DEV evidence. Numeric Q fusion has a
separate proof gate. None of these templates authorizes training.

The new V2 admission policy has no production numerical defaults. Supply
`quality_admission_policy: <signed-json>` before preparation for any research
intended to admit predictions later. The initial FullCheckpoint owns its
immutable policy hash. The plan must freeze every heldout/DEV official-day
window, replay seed, support/calibration/risk threshold and intervention
recipe. Tests use synthetic thresholds that are not research approvals.
Admission requires the final predeclared Env budget, the actual initial
reference, all seven horizons and the complete fold/contract/seed grid.
Existing HEADS_ONLY configs without this policy remain technical experiments.

Forecast admission alone cannot enable prediction-to-Actor corrections.
`quality_decision_admitted` requires independently replayed relative WAIT/ADD
and native closed-loop safety evidence for the exact component set.
Numerical Q additionally requires full native proofs and the evaluated Q
weights, revalidated at start and copied with fresh optimizer state.

```bash
python -B tools/fg_s3_v2_experiment.py evaluate --prepared <prepared> --diagnostic-components outcome,score --eval-steps 100 --output-dir <new-private-diagnostic>
python -B tools/fg_s3_quality_v2.py validate-admission --kind prediction --receipt <admission.json>
python -B tools/fg_s3_quality_v2.py validate-admission --kind decision --receipt <admission.json>
python -B tools/fg_s3_quality_v2.py validate-admission --kind q --receipt <admission.json>
```

Diagnostic fusion uses a private HEADS_ONLY evaluation copy, preserves the
Checkpoint, and does not claim admitted autonomous performance. For Q, both
baseline and candidate must also supply `--diagnostic-q-admission` with the
same independently verified Q source. Future labels and strict audits run
only after causal inference is sealed.

Quality evaluation also supports a TRAIN contract excluded from the new
quality-head supervision (the historical source Backbone can have seen it):

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py evaluate --prepared <prepared> --eval-steps 100 --holdout-contract CZCE.FG401 --output-dir <new-holdout-inference>
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py evaluate-reference --eval-dir <sealed-holdout-inference> --source-checkpoint <initial-checkpoint> --eval-device cpu --output-dir <new-reference-inference>
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py score-predictions --eval-dir <sealed-holdout-inference> --output-dir <new-holdout-posthoc>
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py score-predictions --eval-dir <sealed-reference-inference> --output-dir <new-reference-posthoc>
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py compare-predictions --eval-dir <holdout-posthoc> --reference-dir <reference-posthoc> --output-dir <new-comparison>
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_quality_v2.py validate --catalog <checkpoint-catalog.json>
```

Reference predictions reconstruct their own model carry on the sealed actual
actions; their account return is not an autonomous reference-Actor result.
All future labels are generated after causal predictions are sealed. Catalog
validation authenticates the committed journal prefix, TRAIN sources,
teacher/Replay ownership, preparation, code and optional FullCheckpoint.

`quality_max_market_steps` defaults to null (native replay to exit/data end).
A positive integer bounds counterfactual exposure in valid market minutes;
an unfinished trade becomes RIGHT_CENSORED with missing quality score and
independently masked outcomes. This setting is part of the teacher identity.
Controlled Full preparation additionally binds the actual evaluated predictor
Checkpoint; transfer copies its four predictor parameter groups and requires
an identical fixed base NN. Admitted numerical Q uses its own verified source
and copies both evaluated Q parameter groups. Optimizer slots remain fresh.
Native strict preparation records randomized/risk-stratified candidate parity
and every trade of the selected path; it does not claim exhaustive native
replay of all candidates. The default financial Q domain remains 12 bars.
`quality_build_performance.json` records real member RSS/time/holding lengths;
parallel builders use a measured largest-member pilot and single-thread BLAS.

`E_ONLY_PERF_BASE_GPU_4K.yaml` and `E_ONLY_PERF_OPT_GPU_4K.yaml` are isolated
4K speed comparisons with the same size50m recipe, batch1, 100:4 collection
and update schedule, Gate and seed. BASE runs on Gym `bed3a2cefd3923f4a7d45709d979892264c6298d`
with Dreamer `51ed1d094c92fc52a910acda3e2a3421bc71c9e5`; OPT runs on the performance
worktree with its newly verified E collection. Use the shared `prepare --config
<absolute YAML>` then `start --prepared <that snapshot>` commands. Both keep
their own prepared and Run directories under their corresponding worktree's
`artifacts/fg_s3_v2/`; compare warmed cycles after 2K environment advances.


### E+Quality F1 foundation (WP04)

The F1 contract uses `FOUNDATION_WM` execution and `WM_NATURAL` sampling.
The example `e_quality_f1_cpu.example.yaml` creates an engineering preparation
only; it does not authorize real TRAIN parameter updates or GPU training.
All relative example paths are used from the Gym worktree directory.

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py f1-prepare --config configs/experiments/fg_s3_v2/e_quality_f1_cpu.example.yaml
```

`--source-purpose native-verification --producer-gym <frozen-WP02-Gym>`
explicitly replays the native G3 before an F1_DATA_READY data receipt.
F1_ENGINEERING_READY and F1_DATA_READY never authorize training. Ordinary
E+Quality prepare/start/resume/evaluate keep their existing public Guard.

# S3-V2 entries and diagnostic configurations

## C2 joint learning and isolated CPU diagnostics (2026-10-13)

C2 development uses the sibling worktrees in `research/fg-s3-v2-c2-dev/`.
The running A2 worktrees, snapshots, observer and outputs are retained unchanged.
Use the existing entrypoint; no private DAgger, teacher index or migration API is added.

```bash
cd /home/v/Documents/work/research/fg-s3-v2-c2-dev/gym-trading-env
# Controlled small original-network test, with no CUDA initialization:
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 20-23 nice -n 10 /home/v/miniconda3/envs/dreamerv3/bin/python \
  tools/fg_s3_v2_experiment.py c2_check --device cpu
# Original Replay model and real FG201 TRAIN, with authentic production Gate:
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config C2_CPU_FG201_300
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 20-23 nice -n 10 /home/v/miniconda3/envs/dreamerv3/bin/python \
  tools/fg_s3_v2_experiment.py start --config C2_CPU_FG201_300 --stop-at 200
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 20-23 nice -n 10 /home/v/miniconda3/envs/dreamerv3/bin/python \
  tools/fg_s3_v2_experiment.py resume --config C2_CPU_FG201_300
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 20-23 nice -n 10 /home/v/miniconda3/envs/dreamerv3/bin/python \
  tools/fg_s3_v2_experiment.py evaluate --config C2_CPU_FG201_300 --eval-steps 100
```

CPU is allowed only for C2 diagnostic scope. A2 and formal runs retain GPU.
`c2_check` tests CLOSED → controlled READY → saved/restored READY → CLOSED,
Actor BC/RL gradient additivity, one joint optimizer update, Critic momentum,
SlowCritic and stateful Normalizer boundaries. The fixture enables stateful
value/advantage normalizers; the production `none` configuration is unchanged.
Its controlled Gate reports and prices do not certify production Gate readiness
or market performance. Production regression still stops after five bad reports.

`C2_GPU_FG201_500_V2` is a new, independent original-model GPU diagnostic, to be
run after A2 releases the GPU. Use the same prepare/start/resume/evaluate commands.
The old `C2_GPU_FG201_500` configuration and outputs remain historical records.
No formal C2 training is started by either diagnostic.

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
Formal and A2 model work requires GPU; isolated C2 diagnostics also support CPU.
The entry restarts only its own process without an
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
TEST, R1 comparisons and B2 are outside the current implementation scope.

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

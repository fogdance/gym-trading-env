# S3-V2 entries and diagnostic configurations

## Current P0 A2 training preparation (2026-10-13)

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

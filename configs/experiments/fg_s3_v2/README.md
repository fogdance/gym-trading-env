# S3-V2 D0 CPU prototype

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

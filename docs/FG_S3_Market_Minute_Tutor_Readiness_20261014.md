# Market-minute data and Tutor release — 2026-10-14

Public code frozen pair: Gym `3a6f889adf65583a1ab636f244ca05218f19196f`, Dreamer `c18a8b5917f73110f9b6ce840495f4da1d7ca1d4`. Independent detached training pair `/home/v/Documents/work/research/fg-s3-tutor-mm-train/{gym-trading-env,dreamerv3}`; clean source, no B2 modules. Old A2/Tutor artifacts remain untouched.

## Completed required data

All18TRAIN members rebuilt under `/data/logdir/fg_s3_market_minute/20261013_v2/`: E/causal input audit/R/exact O/full native Replay/Oracle labels/TeacherCatalog PASS. Raw input prices/calendar/terms only shared read-only through the authorized data alias. Previous derived truths not reused.

Manifest SHA256:

| Artifact | SHA256 |
|---|---|
| E | cf8fbf93953142b743f718effe2f377a80eba8a0cddf196a901d5344cf3d19ad |
| O | 376f2a0c0296f857e29957ba6620dc584d9caba0798dea60f6dc4a6ec0e5403a |
| Replay | 4b004e3a9346a814ae9fbc646fbdf87181c3c8e449e01f7be8a69c4e45140822 |
| Labels | 20b64ef783c8cd76c33b2b1c55661401217fba862630088ab363897a856a15d5 |
| Catalog | 5185bd8fa8079a2f800d3fd4dfe3bdda4af7648e42a13af776f4ad8a0096f726 |

Catalog:488629 rows,395718valid,16496ADD,379222WAIT. Reward hash6a44ade43ea7168318c479ac2b17f1058590259f33ef463035d3ece2f20cca8c; observation identitya4a205df98c611f282b782b0355729bf35a347afb325c75d9b008d0a2fa413d6. Full old S3-A feature caches/S3-B weighted value caches have NOT_RUN regeneration; they are not Tutor online training inputs and are not claimed accepted.

## Actual Tutor GPU and restore

Existing `tools/fg_s3_v2_experiment.py`: prepare `TUTOR_MARKET_MINUTE_GPU_1K`, start `--stop-at500`, separate-process resume to1000. Original full model42584132parameters, authenticated Online180k NN inherited exactly, fresh optimization/progress. Actor and WM changed in both halves; Critic/RL and their moments frozen.40/40 updates pairable,11447pairs,40jointupdates,20afterrestore. Native executed actions/Replay/Sidecar/pending and counters verified from actual0/500/1000 checkpoints. Receipt:

`artifacts/fg_s3_v2/runs/TUTOR_MARKET_MINUTE_GPU_1K/gpu_restore_verification.json`

Final checkpoint manifest SHA4f46318840c1423552fcd98b1e1dfc2d09859628470e71426f728a565f3161e0. This is actual short GPU acceptance, not an autonomous trading-success claim. Independent B2 10K/DEV completed on another pair; B2's results did not substitute for this verification.

## Formal launch preparation and marker

Frozen worktree `prepare --config TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M` completed: all18TRAIN pool,1M actual environment advances, save every100K, beta1, static V0 alpha.8/pairlambda.1/kappa0. Source progress21183 recorded only, local updates0. Readiness receipt in prepared directory: `user_launch_readiness.json`. Formal run path:

`/data/logdir/fg_single3/s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1`

User authorized `/tmp/done_cd` only after Tutor required data and own training/restore checks. Those conditions now passed; the marker releases the user's already-staged tmux command. No extra formal launcher is created and no B2 formal1M is started.

```bash
cd /home/v/Documents/work/research/fg-s3-tutor-mm-train/gym-trading-env
env PYTHONDONTWRITEBYTECODE=1 FG_S3_VISIBLE_GPUS=0 \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start \
  --config TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M --authorize-training
```

Same-run interruption uses `resume` with the same config and authorization flag. Preserve old checkpoints/data and this frozen source pair. Actor-only evaluation uses the existing `evaluate` entry with its frozen Tutor extension; periodic formal evaluation/long-term effectiveness is NOT_RUN by this readiness task.

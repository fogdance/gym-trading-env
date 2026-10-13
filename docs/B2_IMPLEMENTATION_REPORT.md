# S3-B2 Outcome-Aware Decision Learning — actual execution

Worktree pair: `/home/v/Documents/work/research/fg-s3-b2-dev/{gym-trading-env,dreamerv3}`; branch `feat/fg-s3-b2-outcome-aware`. Task authority: Reviewed market-minute B2 plan dated20261013. Executed20261013–14. **D1–D4 engineering GO; autonomous trading result FAIL / All-WAIT. No B2 formal1M.**

## Implemented

Shared `S3_MARKET_MINUTE` (valid1m bar=1; entry bar inclusive), new exact E/O only. Independent flat-account/funding/mask outcome certification, H60 event classes and separately masked early10 targets. Native Replay contains no supervision; existing atomically committed Sidecar supplies aligned labels. A2/Tutor remain separate.

Two128-unit heads on original causal RSSM repfeat; bounded signed six-weight adapter, no free intercept. Final ADDlogit is corrected before the original mask/unimix. Real policy, natural BC and Actor-only use the same function. BC detaches repfeat/outcome probabilities; outcome losses train heads and encoder/RSSM. One original joint optimizer; Critic/RL-specific state frozen. Mix4-update warmup/16-update ramp with4 valid-event updates minimum, saved and synchronized to policy.

Public SourceAdmission supports authenticated live TRAIN Oracle binding for diagnostic parameter warm-start. Original Online180k NN copied exactly; new head/adapter, optimizer and counters fresh. No historical teacher truth imported. New original-network source progress21183 model updates/180000 env steps is provenance; B2 starts at0.

## Actual evidence so far

- Controlled original-Env outcome contracts:10PASS. Small CPU neural contracts:4PASS; additional lambda0 teacher-independence:1PASS. Ninjax schema regression:1PASS; updated joint/momentum regression:1PASS.
- Real new FG201 catalog:31259 rows; FAST8048, MID7481, LATE7115, SL4676, OTHER18, OPEN3921. These are certified counterfactual candidates, not actual Actor trades.
- Actual CUDA original model initialization43896273 trainable parameters. Transfer receipt `PASS_REAL_PARAMETER_WARM_START`:89 original NN keys byte-equal;13 new model/state keys fresh.
- Actual500→1000 separate-process FullCheckpoint restore:40 joint updates,39 head parameter-changing updates,35 adapter updates; Actor/WM/head parameters changed, Critic/RL updates0. Mix1, valid event/early coverage870/1000=87%;704 real legal scoring corrections,332 argmax changes.
- Read-only verification of real0/500/1000 checkpoints:1PASS (`test_fg_s3_b2_gpu_receipts.py`), including NN/optimizer/Replay/Sidecar/pending/counters/scheduler and inference synchronization.
- TRAIN104 entries/TP at1K were Oracle-controlled (beta1); independent Actor trading success is not established.

1K node: `artifacts/fg_s3_v2/runs/B2_OUTCOME_GPU_TRAIN_10K_v3/ckpt/step_000000001000`, manifestSHA `de57830fdf4e1c49ddf40f2eb696df3b700e8665bad6840ad4e333c229080322`. `gpu_1k_verification.json` contains actual verification receipt.

## Failures resolved without bypass

Initial small GPU probe failed because its config omitted the new Replay clock field; public small-config initialization corrected on both development branches. Initial B2 startup rejected only newly added optimizer keys: Ninjax Tree encodes `b2_event` as `b2event`. Original source keys/shapes were identical. Admission now derives exact allowed optimizer suffixes via `ninjax.flatten`; foreign keys/original shape changes remain rejected. Failedv1/v2 preparations/run directories retained; v3 uses new independent paths.

## Frozen diagnostic and commands

Prepared runtime pair Gym `65477b0c36948043b9b1e794223ce3addc0c5800`, Dreamer `a194a220719fc7fe719c1354b35211d40f71c903`. Subsequent Dreamer `d76a423` changes only the serialized regression fixture; runtime Python bytes are unchanged.

From B2 Gym worktree, using `/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py`:

```bash
python tools/fg_s3_v2_experiment.py prepare --config B2_OUTCOME_GPU_TRAIN_10K_V3
python tools/fg_s3_v2_experiment.py start --config B2_OUTCOME_GPU_TRAIN_10K_V3 --stop-at 500
python tools/fg_s3_v2_experiment.py resume --config B2_OUTCOME_GPU_TRAIN_10K_V3 --stop-at 1000
python tools/fg_s3_v2_experiment.py resume --config B2_OUTCOME_GPU_TRAIN_10K_V3
python tools/fg_s3_v2_experiment.py evaluate --config B2_OUTCOME_GPU_TRAIN_10K_V3 --eval-steps 5000
```

All five commands executed successfully for v3. Existing snapshots are immutable; use a new config/run name for a repeat. Diagnostic10K budget was frozen at preparation, no budget/config rewrite at1K. Standalone1K config also supplied. The base10K/v2 configs and preparations retain failed startup history; use the v3 configuration for this verified result.

## D4 actual results

10K real FG201 advances,400 joint updates; WM400, outcome parameter updates399, adapter395, RL Actor/Critic0. Valid event/early labels8305/10000=83.05%, mix-active legal decisions8139/8305=98.00%, argmax changes414. Actor proposal ADD2550 is mixed TRAIN behavior, not an Actor-only trade count. Teacher-executed entries/TP906, SL0, reward864.3908 are Oracle-controlled and excluded from autonomous performance claims.

Unique collected TRAIN outcome diagnostics (`prediction_diagnostics_10k.json`): eventNLL1.74029, Brier.83107; earlyNLL.52054, Brier.16953; early predicted mean.25424 vs observed.17965. FAST recall.26186/precision.22126; MID.33511/.24370; LATE.30453/.23379; SL.02878/.08997; OPEN.07091/.19020. OTHER observed0, recall undefined. This is TRAIN diagnostic calibration, not held-out forecasting performance.

Sampled first20→last20 updates: BC.63362→.32853, eventloss1.92113→1.69035, earlyloss.81184→.45625. Loss decreases do not establish trading skill. Warm last20-cycle throughput27.12 actual advances/s; overall core15.77/s includes repeated process compilation. Model process peakRSS4.14GiB; observed GPU footprint about3.1GiB including desktop, sampled utilization2% during collection (not a peak or whole-run average).

Final FullCheckpoint `step_000000010000`, manifestSHA `28401fdbe97aabebeb518156c2be1d747649e006bc911b6520e9e3b4794fe4fd`.

FG409 DEV Actor-only5000 actual advances: **ADD0, TP0, SL0, reward0, All-WAIT**. Base meanpADD.11601, final.10796, delta−.10771, mix enabled throughout, argmax changes0. No Oracle/teacher takeover; no future labels loaded by the policy. TP market-minute statistics undefined because there were no trades. `evaluations/actor_only_f648bbb1f4/receipt.json`, hash `4d3c6b1dcaa33c1adcc35b06c943eaddcd04713c8cf349f505b1270ef70158e1`.

## Review and verification

Reviewed public clock reuse, independent label masks and actual-account certification, policy/BC/evaluation distribution, stop-gradients, original optimizer update count, encoded optimizer schema, policy synchronization and full restore. Existing live collector/recovery regression:4PASS,1real-GPU case deselected (B2's actual GPU case was executed separately). Runtime code stayed frozen for the verified Run; later changes are tests/docs only.

Known limits: conservative funding floor retained; no losing-account teaching expansion; six-class head learned but SL/OPEN recall remains weak, Actor-only stayed All-WAIT. B2 V0 deliberately accepts one TRAIN contract and diagnostics≤10K. Formal budget, multiple seeds, C2/R1/Tutor losses, A2→B2 optimizer migration and TEST: NOT_RUN/not implemented by this task. Existing generic evaluator's `early_mae10` counts all first-adverse events; B2's teacher target is first5. This DEV had zero trades/events, so early5/15 counts are both0; future result reports should use canonical `effective_bar` to distinguish those windows. No extra experiment was run to force ADD.

PASS: controlled original Env10cases; small neural6unique cases (separate targeted commands); actual CUDA1K/restore; actual CUDA10K; actual FG409 Actor-only execution; shared collector4cases. FAIL: improved autonomous trading / All-WAIT. NOT_RUN: formalB21M, TEST, extra ablations. Initial failed startup and incomplete test-fixture failures were corrected and retained in logs; they are not counted as passes.

## Pending and protected work

D4 actual10K/Actor-only DEV: COMPLETE. Formal B2 1M: NOT_RUN and not authorized. Final source SHAs are recorded at handoff; runtime pair above remains recoverable in the unchanged B2 development paths.

Tutor's training worktree stays independent, without B2 networks. New18-contract Oracle/full Replay/labels/catalog are now complete; catalog488629 total rows,395718valid,16496ADD,379222WAIT. Tutor new-market-minute GPU preparation has started in the separate clock development pair; its own training/restore validation still required. `/tmp/done_cd` has not been created and will not be created until Tutor's data and required actual training/restore validation pass. B2 PASS cannot substitute for Tutor PASS.

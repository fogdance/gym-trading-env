# S3-B2 implementation and isolation

Authority: `FG_S3_B2_Codex_Implementation_Plan_gy9_d39_MarketMinutes_Reviewed_20261013.md` in the original research Gym docs. Worktree pair: `research/fg-s3-b2-dev`, branch `feat/fg-s3-b2-outcome-aware`. Public clock baseline: Gym `37f97c4`, Dreamer `c18a8b5`; these commits are a code baseline, not completed GPU acceptance.

## D1 → D4

1. Reuse `S3_MARKET_MINUTE`, new signed E/O data in `/data/logdir/fg_s3_market_minute/20261013_v2`; create independently authenticated H=60 outcome labels, event/early masks, atomic native-row sidecar joins. Old Oracle truth is forbidden.
2. Add two lightweight 128-unit heads and a bounded signed decision adapter. A single final masked distribution serves real policy, natural Oracle BC and Actor-only eval. Outcome auxiliary gradients reach RSSM/encoder; BC detaches features/probabilities. Joint optimizer updates once; all Critic/RL-specific state remains frozen.
3. Shared SourceAdmission admits signed live-Oracle TRAIN bindings for parameter warm-start and explicit B2 architecture extension. Authenticate source Online180k, copy original NN weights exactly, keep new modules/optimizer/counters fresh. Extend policy synchronization to heads, adapter and mix state.
4. Minimal CPU contract tests, then real FG201 GPU collect/train/collect at 500→1000 with separate-process restore, then extend the same frozen 10K diagnostic run. Independent FG409 Actor-only 5K evaluation; no B2 formal1M.

Defaults: event/early loss weights .1/.1; BC1; heads one layer of128; event order FAST/MID/LATE/SL/OTHER/OPEN; adapter TP-positive, SL/OPEN/early-negative, OTHER0, coefficients bounded2, total correction bounded±2. Mix starts after4 model updates and4 valid event updates, ramps over16 model updates; checkpointed. Diagnostic beta1, collect100/update4, batch1/64+1; no Tutor losses.

The 10K run is prepared once, stopped at500 and1000 using the existing CLI, then resumed to10K. The standalone1K config is also provided; frozen configurations are never edited to extend budgets.

## Tutor release constraint (user reconfirmed)

Tutor's frozen training worktree remains independent of B2. `/tmp/done_cd` may only be created after Tutor's required regenerated data and actual training/restore verification pass. B2 verification cannot substitute for Tutor verification. Neither marker nor Tutor formal training is triggered by this B2 implementation.

## Evidence status

B2 outcome tests, neural tests, GPU closed loop/restore and DEV: NOT_RUN at plan creation. Runtime receipts and final two-repo SHAs will be recorded in `B2_IMPLEMENTATION_REPORT.md`; fixture, GPU diagnostic and production claims remain separate.

Progress: controlled original-Env outcome tests10PASS; small CPU neural contracts4PASS, including detached BC, live policy mix0 equivalence, joint update/restore and exact Critic/momentum freeze. Real new FG201 source loaded:31259 candidates; FAST8048/MID7481/LATE7115/SL4676/OTHER18/OPEN3921. These catalog counts are hypothetical certified flat-account candidates, not collected Actor trades or trading success. Actual GPU B2 acceptance remains NOT_RUN at this stage.

Tutor data generation is active through the original CPU CLI; labels/catalog automatically follow successful new Replay publication via `/tmp/fg_s3_tutor_mm_data_ready.sh`. Required commands use existing tools and new `/data/logdir` outputs; the helper never creates `done_cd`. Persistent permission for `/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py` was verified before user sleep.

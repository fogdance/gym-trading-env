# FG S3 W&B implementation and acceptance

Development uses the independent `feat/fg-s3-wandb-tracking` paired worktree,
based on Gym `61a3e88` and DreamerV3 `d76a423`. The running Tutor tree stays
at Gym `3a6f889` / DreamerV3 `c18a8b5`. Tracker code is outside the scanned
training packages and does not modify `tools/fg_s3_v2_experiment.py`.

1. P0: isolated Python environment; signed provenance, committed-prefix
   parsing, redaction, durable local events, Mock and real offline SDK.
2. P0 real input: Tutor 800K DEV must produce ADD 7, TP 5, SL 2, Reward
   -3.805870048525721, closed trade net PnL -2226, adverse windows 0/1/3/4.
3. P1: after account/private-project/cloud authorization, import Tutor only.
   Entity is the user's personal slug, supplied at runtime, never hardcoded.
   Do not import or process historical A2 Runs. Future new Runs use the shared launcher.
   Validate real upload, repeat import, offline sync and online continuation.
4. P2: only after P1 passes, an external launcher may run an explicit original
   trainer command; test it with a fake trainer, never a new GPU/1M run.

All tracker writes belong to its independent output root, never a source Run.
Only complete Checkpoint metadata and matching cycle/update prefixes certify
canonical history. Full bundle integrity is the trainer's responsibility;
the tracker does not read model parameters or Replay. Uncommitted history
belongs exclusively to a provisional monitor, with explicit rollback segments.

SDK 0.19.1 ignores `resume` offline. Offline export therefore uses closed,
non-overlapping event segments and a separate monotonic publishing sequence.
Cloud sync/resume is not accepted until an actual authorized cloud test passes.

This document records the implementation plan, not successful acceptance.
Execution receipts and final PASS/FAIL/NOT_RUN are published separately.

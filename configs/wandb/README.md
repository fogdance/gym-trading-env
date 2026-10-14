# FG Single3 research tracking

The shared entry is `tools/fg_s3_wandb.py`. Run it with the isolated
`/home/v/Documents/work/research/fg-s3-wandb-env/bin/python`; the trainer's
`dreamerv3` Conda environment remains unchanged. Project is fixed to
`fg-single3-research`. Account, credentials and proxy are runtime settings.

```bash
export WANDB_ENTITY=xkm104501-none
export WANDB_PROJECT=fg-single3-research
export WANDB_MODE=online
```

Login in your own terminal with `wandb login`. Never put an API key into this
YAML, Git, a receipt or chat. `auth-check` verifies the existing private project.

```bash
cd /home/v/Documents/work/research/fg-s3-wandb-dev/gym-trading-env
TRACKER_PY=/home/v/Documents/work/research/fg-s3-wandb-env/bin/python
TUTOR_RUN=/data/logdir/fg_single3/s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1

"$TRACKER_PY" tools/fg_s3_wandb.py auth-check --proxy http://127.0.0.1:10808
"$TRACKER_PY" tools/fg_s3_wandb.py inspect --run-dir "$TUTOR_RUN"
"$TRACKER_PY" tools/fg_s3_wandb.py import --run-dir "$TUTOR_RUN" \
  --publication tutor-cd-reupload-v1 --mode offline
"$TRACKER_PY" tools/fg_s3_wandb.py sync --run-dir "$TUTOR_RUN" \
  --publication tutor-cd-reupload-v1 --proxy http://127.0.0.1:10808
"$TRACKER_PY" tools/fg_s3_wandb.py follow --run-dir "$TUTOR_RUN" \
  --publication tutor-cd-reupload-v1 --proxy http://127.0.0.1:10808
"$TRACKER_PY" tools/fg_s3_wandb.py import-evals --run-dir "$TUTOR_RUN" \
  --publication tutor-cd-reupload-v1 --proxy http://127.0.0.1:10808
```

`--publication tutor-cd-reupload-v1` is this Tutor's explicit replacement mapping
after the user requested deletion of the incomplete cloud Run. Keep it on later
commands for this same Tutor. New research Runs normally omit `--publication`.
Re-publication leaves the local Run identity and all old tracker files intact.

`--mode disabled` produces no tracker output or uploads. Offline export clears
the child API key and keeps closed SDK records plus immutable redacted event
batches. Use the tracker `sync` command to upload those batches and verify every
event sequence/hash. The SDK's legacy bulk `wandb sync` returned HTTP 500 on this
real history; the tracker uses its shared online SDK worker with a 256 KiB
file-stream limit. `synced=true` requires cloud readback, never a successful
local export alone.

Tracker output is `/data/logdir/fg_single3/experiment_tracking/<stable-train-id>`.
No Checkpoint, Replay, Oracle, raw account state, market data or source diff is
uploaded. Frozen configuration, checkpoint metadata and derived evaluation
summaries are small JSON artifacts. Missing legacy metadata stays unknown.
Check `last_operation.json` and `last_transport_error.json` in this independent
directory when an operation fails; retain the queued files and retry.

Canonical curves contain only complete Checkpoint-confirmed cycle/update
prefixes. Live monitor curves are explicitly `provisional/*`; source rollback
starts a new monitor segment. TRAIN trades can include Oracle control; only the
separate Actor-only DEV Runs measure autonomous trading. This first import
includes Tutor-CD only. Historical A2 is excluded.

## Automatic tracking for future research Runs

Use `launch` around the original, explicitly selected command. It starts that
command immediately in its frozen worktree with the original environment, then
waits independently for the signed Run identity and a completed Checkpoint.
It follows both committed and provisional records, imports completed Actor-only
receipts, and drains once after the original command exits. A tracker failure
prints an error and cannot stop the original command. Cloud failures fall back
to local offline records; the next successful poll syncs them before publishing
new online records. The wrapper preserves the original command's exit status.

Template for a **new, separately authorized** experiment:

```bash
# Replace all three placeholders with the actual prepared experiment.
FROZEN_GYM=/absolute/path/to/frozen/gym-trading-env
PREPARED=/absolute/path/to/new/prepared
NEW_RUN=/data/logdir/fg_single3/s3_v2/NEW_RUN

"$TRACKER_PY" tools/fg_s3_wandb.py launch --run-dir "$NEW_RUN" \
  --prepared "$PREPARED" --mode online --proxy http://127.0.0.1:10808 \
  --trainer-cwd "$FROZEN_GYM" -- \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start \
    --prepared "$PREPARED" --run-dir "$NEW_RUN"
```

For an authorized formal Run, retain the trainer's existing
`--authorize-training` option. Change the original command to `resume` for
same-Run recovery, or `evaluate --eval-steps 5000` for Actor-only evaluation;
keep the same Run path and publication mapping. The trainer still performs its
own identity/admission checks. This documentation does not authorize starting
a new formal experiment. No real training command was executed for this integration.

Existing unwrapped Runs use `follow` in a separate tmux window. This preserves
old entry points; only future commands run through `launch` automatically start
tracking. After the producer exits, the wrapper may wait for a bounded SDK
transport timeout to drain pending records. Network is never a prerequisite for
the producer to start or continue.

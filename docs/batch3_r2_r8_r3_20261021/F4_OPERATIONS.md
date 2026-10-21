# F4 prepare, pause, review and resume

Status: selected CPU/synthetic verification passed; final integration receipts are
indexed in BATCH3_IMPLEMENTATION_REVIEW.md. Commands are instructions for the
operator; no formal training, GPU or TEST authorization is inferred.

Use the same frozen sibling Gym/Dreamer code pair for prepare/start/resume. Old
online Run V1/V2 checkpoints stay with their original code; new Run V3 is not a
migration. Existing six-behavior V3 Corpora retain ABI 3.1.0 and their bytes.

## Source settings

Retain the existing online execution settings and historical MultiCorpus config.
For controlled RL add explicit `rl_admission`:

```yaml
rl_admission:
  warmup_env_steps: 1024
  calibration_batches: 32
  calibration_horizon: 3
  calibration_seed: 202
```

Controlled RL also requires explicit `wm_policy.max_reward_mae` and
`wm_policy.max_continue_mae` in the existing online settings; there are no
automatic permissive thresholds.

These are example values for a new plan, not a claim that the sample count or
model is sufficient. Warmup must align with collect_every and be strictly below
env_steps. Horizon must be below the F3 target length. The existing 25,000 unique
owner-step support, 20 passing reports, ADD support and strict mask policy remain
mandatory. Insufficient data or calibration failure pauses RL_BLOCKED; changing
the frozen plan/model/source requires a new Run, not editing Prepared.

Common CLI Producer flags apply to every source. If verifiers differ, configure
`source_verifiers` keyed by `F3_ANCHOR` and each history corpus_id, each containing
`producer_gym`, `producer_lock`, `producer_lock_sha256`; supply no common override.
Each pin's external digest must be reviewed independently.

## Operator commands

From the frozen Gym directory, with the existing Dreamer Python environment:

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py online-prepare --config PLAN.yaml \
  --producer-gym FROZEN_PRODUCER_GYM --producer-lock REVIEWED_PIN.json \
  --producer-lock-sha256 EXTERNALLY_REVIEWED_PIN_SHA256

/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py online-start --prepared PREPARED_DIR \
  --authorize-training --producer-gym FROZEN_PRODUCER_GYM \
  --producer-lock REVIEWED_PIN.json --producer-lock-sha256 EXTERNALLY_REVIEWED_PIN_SHA256

/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py online-rl-review-packet \
  --run-dir RUN_DIR --output-dir NEW_EXTERNAL_PACKET_DIR
```

The controlled RL start collects/BC-trains only through the explicit prefix,
executes read-only WM calibration and at most two isolated candidate optimizer
invocations, requiring actual Actor and critic changes without altering their
existing update schedule,
then pauses AWAIT_OPERATOR_REVIEW. Technical failure instead reports RL_BLOCKED.
The packet exporter writes an AWAIT packet, never an APPROVED file.

An external human reviews the packet and authors a separate review JSON with
protocol `FG_S3_E_QUALITY_FORMAL_RL_REVIEW_V2`, verdict
`APPROVED_FOR_CONTROLLED_RL`, reviewer_role `RESEARCH_OPERATOR`, their review_id,
the reviewed packet_path/packet_sha256, and its exact review_contract. The human
must supply the approved review file's SHA256 independently. The trainer must not
create, edit or approve this document.

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py online-resume --prepared PREPARED_DIR \
  --authorize-training --authorize-rl --rl-review EXTERNAL_HUMAN_REVIEW.json \
  --rl-review-sha256 EXTERNALLY_APPROVED_REVIEW_SHA256 \
  --producer-gym FROZEN_PRODUCER_GYM --producer-lock REVIEWED_PIN.json \
  --producer-lock-sha256 EXTERNALLY_REVIEWED_PIN_SHA256
```

Every new process validates the review and source evidence again. Saved Gate
booleans grant no process authority. Authorized native collection can extend the
reviewed prefix within the frozen budget; its current full-account flag remains
pending until actual reconciliation. Existing Reader/action-mask/account guards
continue to check consumed windows. The final full trace must independently
reconcile before any verified completion or protected final checkpoint.

Reports distinguish BC/RL update counts, technical blocking, current process
permission, historical reviewed identity and full-account coverage. Zero actual
RL updates never produce RL_COMPLETE_VERIFIED. Engineering/synthetic receipts
and mathematical calibration fixtures never establish formal source admission or
research performance.

`bc_updates` and `rl_updates` count active loss components and can overlap when
the existing RL objective includes BC. `optimizer_updates` counts actual live
optimizer invocations; isolated candidate updates never increment these counts.

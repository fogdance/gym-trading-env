# F1/F2/F3 Corpus and verifier interface

All three stages retain their existing configs, original WP03 Reader and
training algorithms. Their `preflight`, `prepare`, `start_prepared` and CPU
`read_check` APIs accept `producer_lock=None, producer_lock_sha256=None` alongside
`producer_gym`. Both pin arguments are required for native verification. DEV
integrity checks need no pin and grant no training permission.

The pin is `FG_S3_E_QUALITY_PRODUCER_PIN_V1`: exact `repositories.gym.sha` and
`repositories.dreamer.sha`, with its SHA256 independently approved by the
operator. The frozen Gym's sibling must be named `dreamerv3`. A valid Git HEAD
with modified/untracked runtime Python cannot certify a source. Pin files are
not cryptographic third-party signatures; the existing external approval
boundary remains required.

Prepared receipts bind the original source/content identity, exact compatibility
result and stable `verifier_binding` (pair SHA and operator pin SHA). Run identity
retains those fields plus execution code/config/index and recovery policy.
Actual per-invocation argv/cwd/exit/log SHA/timing remain separate execution
evidence. Every start/resume runs G3 again; editable JSON PASS is not authority.
The original producer inventory remains historical provenance. It is not
rewritten to the consumer or verifier HEAD.

## Command templates

From the new consumer Gym worktree, using its sibling Dreamer. Replace all
quoted placeholders with reviewed paths/hashes. CPU affinity caps native G3 to
20 logical CPUs; the existing frozen Producer is not edited.

```bash
env PYTHONPATH=.:src:../dreamerv3 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  taskset -c 0-19 /home/v/miniconda3/envs/dreamerv3/bin/python -B \
  tools/fg_s3_v2_experiment.py f1-prepare \
  --config '<F1_CONFIG>' --source-purpose native-verification \
  --producer-gym '<FROZEN_VERIFIER_GYM>' \
  --producer-lock '<REVIEWED_VERIFIER_PIN_JSON>' \
  --producer-lock-sha256 '<EXTERNALLY_APPROVED_PIN_SHA256>'
```

For `f1-read-check`, use the same config/purpose/pin arguments plus a new isolated
`--output-dir` allowed by the existing read-check rules. This runs no optimizer.
F2/F3 read-check use their stage configs without consuming source parameters;
their prepare/start routes additionally require a valid completed upstream model.

After **separate user authorization for formal training**, start with:

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py \
  f1-start --prepared '<F1_PREPARED>' --authorize-training \
  --producer-gym '<FROZEN_VERIFIER_GYM>' \
  --producer-lock '<SAME_REVIEWED_VERIFIER_PIN_JSON>' \
  --producer-lock-sha256 '<SAME_EXTERNALLY_APPROVED_PIN_SHA256>'
```

Exact recovery changes only `f1-start` to `f1-resume`, with the same Prepared,
pin and frozen consumer code. Use the same CPU environment/affinity as prepare
when needed. Substitute the `f2` or `f3` prefix for the corresponding existing
stage entry; no MultiCorpus, Loss or training-target rewrite is required.
F3's required Natural stream, auxiliary role permissions and Pair semantics are
unchanged. Refer to RECOVERY_SOP.md for damaged checkpoints; do not repair one by
running start or overwriting a Run directory.

These templates are not evidence of executed real training. Real G3/formal
prepare/start/resume remain NOT_RUN unless separately recorded and authorized.

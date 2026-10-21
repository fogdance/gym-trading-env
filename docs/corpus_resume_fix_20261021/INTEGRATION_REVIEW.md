# Integration of Corpus resume repair

Integration target before merge:

- Gym `61220dc515d0a6a75ddc5684dd80e16c29924264`
- Dreamer `990f61f8b7ac6bad25ff9e41b65624ae13ccf497`

Source commit: Gym `bac20cd1652d00871378c34afa6e02c0fe469f64`.
The only incoming commit was the Corpus repair; merge was conflict-free.
All four incoming runtime modules are byte-identical to that tested source
commit. Existing F1/F3 default resume changes and Batch3 work on the target
are preserved. No Dreamer source change or merge is needed.

CPU interface checks on this integration pair use:

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 \
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider \
  tests/entry_v1/test_corpus_resume_fix.py \
  tests/entry_v1/test_corpus_provenance.py \
  ../dreamerv3/embodied/tests/test_fg_s3_gate.py \
  ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py \
  ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py \
  --junitxml=docs/corpus_resume_fix_20261021/integration_checks.xml -q
```

Result: exit 0, **27 passed, 0 failed, 0 skipped** (157.799 seconds).
The actual result and collected cases are in `integration_checks.xml` and
`INTEGRATION_EVIDENCE.json`. This is targeted CPU verification, not a full
project run or real-contract G3 acceptance.

The frozen production pair remains Gym `bac20cd1652d00871378c34afa6e02c0fe469f64`
and Dreamer `a6a6ad9e18196f799c4517f370dd26a4ca68aaf5`, clean. Its original
release and repaired configuration are independent of this integration merge.
Production preparation completed with exit 0 / RESUME_PREPARED: six successful
stages retained, original TRAIN01/02 inventories verified, and TRAIN02 selected
for adoption without regenerating data. Original failed attempts remain failed.

`PRODUCTION_FINAL_EVIDENCE.json` and `PRODUCTION_RESUME_READY.json` archive that
specific production pair's evidence; they are not re-labelled as tests on this
new integration pair. The release preparation helper is archived as
`prepare_resume_release.py`; its original lives beside the frozen RUN_CORPUS.sh.

The operator continuation command remains:

```bash
bash /home/v/Documents/work/research/fg-s3-formal-corpus-scheduler-20261021/release/RUN_CORPUS.sh resume
```

No production or training process was launched by this merge. Real production
G3/Reader completion after resume remains NOT_RUN by the agent. No TEST market,
GPU, financial Env/Reward/Teacher algorithm or training Loss was changed.

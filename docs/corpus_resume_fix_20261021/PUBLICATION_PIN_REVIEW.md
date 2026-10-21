# FG809 publication failure: completed verifier identity

## Confirmed failure

The resumed pipeline completed TRAIN01/CZCE.FG809 Reader attempt 0002 and
coverage attempt 0001, then publish attempt 0001 exited 2:
`Batch native/Reader/producer receipt mismatch`.
TRAIN02/CZCE.FG901 Native G3 attempt 0001 was subsequently interrupted with -15.

Every native/Reader predicate was true except equality between the saved G3
audit and the current publisher's audit. The successful G3 used Gym
`8619f04b6f996f4c4d7f169486e4d10c25a3b2b9`, Dreamer
`b7a693b3c7b0b45eeecdec954818be42d40e67b6`, external Pin SHA256
`d87082946aaa9db7408d68b2c2d147bf6cf1d28f27c1afb36d07c32b59239425`.
The publisher was Gym `bac20cd1652d00871378c34afa6e02c0fe469f64`,
Dreamer `a6a6ad9e18196f799c4517f370dd26a4ca68aaf5`.
This last equality was an incomplete part of the earlier provenance repair.

## Repair

Publication still authenticates the current publisher through its current
frozen Pin and validates all selected stage completions and file hashes.
The completed G3 is authenticated against its own sealed request/execution
binding, exact original external Pin SHA256, repository heads, zero runtime
diff and empty untracked-runtime inventory. An unrelated path, changed head,
dirty audit or substituted Pin is rejected.

Historical certificate bytes may be read from an authenticated relocated
copy; the old verifier checkout is not opened. New G3 stages archive their
Pin alongside their result. Publication archives the selected exact Pin
bytes while retaining the original G3 receipt, corpus identity and all six
complete native replay/teacher-label checks.

No Corpus, Reader index, financial rules, Teacher, Loss or F1 frozen worktree
is modified. The successful FG809 G3/Reader/coverage stages are reused.
The interrupted FG901 G3 must be rerun in a new attempt, not marked complete.

## Necessary verification

From this Gym worktree:

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 \
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider \
  tests/entry_v1/test_corpus_publication_pin.py \
  --junitxml=docs/corpus_resume_fix_20261021/publication_pin_checks.xml -q
```

Exit 0: six passed. Two positives cover legacy and explicit executor
bindings with nonexistent historical checkout paths. Four negatives cover
head substitution, dirty audit, external Pin substitution and wrong cwd.
The real FG809 receipt also passed the new binding function using the
release's existing authenticated Pin copy; no historical checkout was read.

The complete real publication-path review and final frozen executor are
recorded in `PUBLICATION_FIX_EVIDENCE.json`. Official production resume and
FG901 G3 completion remain NOT_RUN by the agent; no optimizer/GPU or TEST
work is performed by this fix.

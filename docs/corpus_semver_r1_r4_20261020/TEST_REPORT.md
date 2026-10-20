# Targeted validation report

Final tested pair: Gym `83bc3018de9598969d9f06df39f48da57f8dbb5c`, Dreamer `4c4a36636bfa1a3f6a02c6df7f7d164ca8ec7036`.

268 distinct collected cases; 298 executions across six suites. All passed; 0 failures/errors/skips. Counts do not claim full-project coverage.

| Suite | Cases / passed | Exit | Seconds |
|---|---:|---:|---:|
| f1 | 22 | 0 | 139.924 |
| f2 | 8 | 0 | 143.158 |
| f3 | 11 | 0 | 184.512 |
| shared | 20 | 0 | 133.020 |
| contracts | 67 | 0 | 42.512 |
| reader_multi | 170 | 0 | 205.998 |

F1/F2/F3 use real CPU JAX small updates on synthetic Corpus only. Real TRAIN optimizer updates: 0. `contracts` includes the full external strict fault matrix and all-stage early rejection. `shared` is the three mandatory AGENTS.md files. `reader_multi` preserves native Reader, role/RNG/Pair and causal-MACD contracts.

Exact argv, cwd, environment, source pair, wall time and exit code are preserved in FINAL_COMMANDS.json and each Z1_v2_*.json. FINAL_TEST_INDEX.tsv/json enumerates every collected case and source-file hash.

## Actual final commands

### f1

```bash
cd /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/artifacts/semver_validation/f1
env PYTHONPATH=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/src:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/dreamerv3:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-5 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_wp04_checkpoint.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_wp04_resume_identity.py --junitxml=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/gym-trading-env/docs/receipts/corpus_semver_r1_r4_20261020/Z1_v2_f1.xml
```

### f2

```bash
cd /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/artifacts/semver_validation/f2
env PYTHONPATH=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/src:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/dreamerv3:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 6-11 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_wp05_checkpoint.py --junitxml=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/gym-trading-env/docs/receipts/corpus_semver_r1_r4_20261020/Z1_v2_f2.xml
```

### f3

```bash
cd /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/artifacts/semver_validation/f3
env PYTHONPATH=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/src:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/dreamerv3:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 12-19 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_wp06_checkpoint.py --junitxml=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/gym-trading-env/docs/receipts/corpus_semver_r1_r4_20261020/Z1_v2_f3.xml
```

### shared

```bash
cd /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env
env PYTHONPATH=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/src:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/dreamerv3:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-19 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/../dreamerv3/embodied/tests/test_fg_s3_gate.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/../dreamerv3/embodied/tests/test_action_mask_formal_guards.py --junitxml=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/gym-trading-env/docs/receipts/corpus_semver_r1_r4_20261020/Z1_v2_shared.xml
```

### contracts

```bash
cd /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env
env PYTHONPATH=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/src:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/dreamerv3:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-19 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_corpus_producer_pin.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_corpus_offline_recovery.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/../dreamerv3/embodied/tests/test_fg_course_strict_checkpoint.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/../dreamerv3/embodied/tests/test_fg_full_checkpoint.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_wp07_checkpoint_core.py --junitxml=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/gym-trading-env/docs/receipts/corpus_semver_r1_r4_20261020/Z1_v2_contracts.xml
```

### reader_multi

```bash
cd /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env
env PYTHONPATH=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/src:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/dreamerv3:/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-19 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_corpus_compatibility.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_corpus_producer_pin.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_corpus_verifier_binding.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_wp03_admission.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_wp03_join.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_wp03_reader.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_data_multi.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_data_multi_f3.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_wp02_macd.py /home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/rounds/cold-v2/gym-trading-env/tests/entry_v1/test_macd_dated_tail.py --junitxml=/home/v/Documents/work/research/fg-s3-corpus-semver-r1-r4-20261020/gym-trading-env/docs/receipts/corpus_semver_r1_r4_20261020/Z1_v2_reader_multi.xml
```

## Retained failures and fixes

- B1: 58 passed / 1 failed. The same failure reproduced on unchanged integration (exit 1): the fixed historical Corpus is V2, rejected before its obsolete MACD. The assertion now checks the fixed asset identity and V2 rejection; recheck exit 0.
- B2: 6 passed / 2 failed because a cold 30-minute fixture had no F2/F3 target support. Replaced it with the existing warmed multiscale synthetic fixture; did not relax support checks. Recheck: 8 passed, exit 0.
- C1: the new mock Agent omitted base save/load methods. Fixed the mock; all three real Run assembly paths passed, exit 0.
- C2: before fixes, four fault cases failed (wrong backend accepted by probe, node-step mismatch, explicit old load/save bypass, orphan diagnostic). All are covered by the final matrix.
- Self-review: interrupted final protection staged an extra file inside a published node; a targeted regression failed with file-set mismatch, then passed after outside-node staging. External strict durability barriers and their failure path were also added. The entire final six-suite validation above was rerun after that runtime change.

## Real assets and unexecuted acceptance

Two original, byte-unchanged TRAIN V3 assets passed the existing independent Reader command with --without-model: 10 views / 10 exact next-batch restores each, requested-window support PASS, optimizer exposures 0. Commands/logs and full reports are under Z1_v2_real_reader_commands.json and real_reader_v2_phase2/phase3/.

Actual production Native G3, formal real-model prepare/start/resume, GPU, host power-loss testing and TEST market reads: NOT_RUN. Synthetic G3 routing stubs prove argument/identity wiring only and do not count as real certification.

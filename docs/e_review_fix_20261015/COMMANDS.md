# 原始定向验证命令

所有命令工作目录：`/home/v/Documents/work/research/fg-s3-e-review-fix-20261015/gym-trading-env`。未执行GPU或正式训练命令。

修复前负例（5 FAIL）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_s3_e_market_identity.py --basetemp=/home/v/Documents/work/research/fg-s3-e-review-fix-20261015/pytest_before --junitxml=docs/e_review_fix_20261015/evidence/market_identity_before.xml > docs/e_review_fix_20261015/evidence/market_identity_before.log 2>&1
```

修复后首5项（5 PASS）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_s3_e_market_identity.py --basetemp=/home/v/Documents/work/research/fg-s3-e-review-fix-20261015/pytest_identity_after --junitxml=docs/e_review_fix_20261015/evidence/market_identity_after.xml > docs/e_review_fix_20261015/evidence/market_identity_after.log 2>&1
```

受影响合同首轮（19 PASS、2 FAIL；测试目录违反Collector隔离规则，没有放宽守卫）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_s3_e_market_identity.py tests/entry_v1/test_s3_e_native_env.py tests/entry_v1/test_entry_market_cache.py tests/entry_v1/test_s3_composition.py::test_reviewed_reader_warm_start_still_rejects_unknown_changes --basetemp=/home/v/Documents/work/research/fg-s3-e-review-fix-20261015/pytest_targeted --junitxml=docs/e_review_fix_20261015/evidence/targeted.xml > docs/e_review_fix_20261015/evidence/targeted.log 2>&1
```

修正输出目录后（21 PASS）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_s3_e_market_identity.py tests/entry_v1/test_s3_e_native_env.py tests/entry_v1/test_entry_market_cache.py tests/entry_v1/test_s3_composition.py::test_reviewed_reader_warm_start_still_rejects_unknown_changes --basetemp=artifacts/fg_s3_v2/checks/e_market_identity_review_20261015 --junitxml=docs/e_review_fix_20261015/evidence/targeted_final.xml > docs/e_review_fix_20261015/evidence/targeted_final.log 2>&1
```

自审补充显式reset/恢复，仅复验共享哈希1项（1 FAIL；夹具在未完成账户上普通reset，被原保护正确拒绝）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_s3_e_market_identity.py::test_shared_e_market_content_is_stable_across_accounts_and_restore --basetemp=artifacts/fg_s3_v2/checks/e_market_identity_hash_self_review_20261015 --junitxml=docs/e_review_fix_20261015/evidence/shared_hash_final.xml > docs/e_review_fix_20261015/evidence/shared_hash_final.log 2>&1
```

把普通reset移至合法初始边界后，只复验同一项（PASS）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_s3_e_market_identity.py::test_shared_e_market_content_is_stable_across_accounts_and_restore --basetemp=artifacts/fg_s3_v2/checks/e_market_identity_hash_reset_final_20261015 --junitxml=docs/e_review_fix_20261015/evidence/shared_hash_reset_final.xml > docs/e_review_fix_20261015/evidence/shared_hash_reset_final.log 2>&1
```

这些组存在重复用例，不相加成独立覆盖总数。完整测试前先创建证据目录；初次重定向因目录尚未存在而失败，测试没有执行。只读Checkpoint诊断初次误用`counters.pkl`，检查实际manifest后改读`v2_counters.pkl`；未改写任何Checkpoint。

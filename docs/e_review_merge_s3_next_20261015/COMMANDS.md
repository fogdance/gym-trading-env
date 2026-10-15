# 合并及最小验证命令

两仓目标目录分别为 `research/fg-s3-next-integration/{gym-trading-env,dreamerv3}`。

Dreamer（成功，双父 merge）：

```bash
git merge --no-ff --no-edit fix/s3-e-market-identity-20261015 -m "Merge reviewed E model into integration/s3-next"
```

Gym（命令列表冲突，保留两侧命令后暂存）：

```bash
git merge --no-ff --no-commit fix/s3-e-market-identity-20261015
git add tools/fg_s3_v2_experiment.py
```

以下命令工作目录为目标 Gym。首轮（2 PASS / 7 ERROR；临时目录被既有隔离守卫拒绝）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_s3_e_market_identity.py tests/entry_v1/test_s3_composition.py::test_reviewed_reader_warm_start_still_rejects_unknown_changes tests/entry_v1/test_fg_s3_online_monitor.py::test_shared_entry_monitors_existing_run_without_rpc_or_model --basetemp=artifacts/fg_s3_v2/checks/e_review_merge_s3_next_20261015 --junitxml=docs/e_review_merge_s3_next_20261015/evidence/targeted.xml > docs/e_review_merge_s3_next_20261015/evidence/targeted.log 2>&1
```

确认新的绝对临时目录不存在后，只重跑前述 7 项（7 PASS）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_s3_e_market_identity.py --basetemp=/home/v/Documents/work/research/diagnostics/e_review_merge_s3_next_20261015_identity_v2 --junitxml=docs/e_review_merge_s3_next_20261015/evidence/identity_final.xml > docs/e_review_merge_s3_next_20261015/evidence/identity_final.log 2>&1
```

CLI 采用相同的 CPU / 空 CUDA 环境；逐条真实 argv、退出码和输出保存在 `evidence/interface_checks.json`：

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py --help
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py describe-composition --config E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py describe-composition --config E_MODEL_V1_A2_WAITING
/home/v/miniconda3/envs/dreamerv3/bin/python -B tools/fg_s3_v2_experiment.py describe-composition --config E_MODEL_V1_B2_WAITING
```

身份对照与 diff 检查的真实输出保存在 `evidence/identity_review.json`。默认检查只提示原任务书三行 Markdown 双空格换行；原任务书不改写：

```bash
git diff --cached --check
git diff --cached --check -- . ':!docs/**'
git ls-files --unmerged
git add docs/e_review_merge_s3_next_20261015
git add -f docs/e_review_merge_s3_next_20261015
git commit -m "Merge reviewed E Env integration and Actor-only evidence into s3-next"
```

普通 `git add docs/...` 被现有忽略规则拒绝（退出码 1）；随后只对本次交付目录使用 `-f`，没有更改忽略规则。

没有执行训练、GPU、模型评价、push 或 main 合并命令。

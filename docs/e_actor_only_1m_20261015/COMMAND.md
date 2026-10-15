# 实际 GPU 评估命令

工作目录：`/home/v/Documents/work/research/fg-s3-e-review-fix-20261015/gym-trading-env`。以下命令已使用`require_escalated`启动；未先运行GPU存在性探测。

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/home/v/Documents/work/research/fg-s3-e-review-fix-20261015/actor_only_1m_latest_20261015_v1_mpl /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_evaluate_frozen.py --prepared /home/v/Documents/work/research/fg-s3-e-model-dev/gym-trading-env/artifacts/fg_s3_v2/prepared/E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261015 --run-dir /data/logdir/fg_single3/s3_v2/E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261015 --output-dir /home/v/Documents/work/research/fg-s3-e-review-fix-20261015/actor_only_1m_latest_20261015_v1 --eval-steps 5000 > /home/v/Documents/work/research/fg-s3-e-review-fix-20261015/actor_only_1m_latest_20261015_v1.log 2>&1
```

输出是独立目录，重复执行应换新的输出目录/日志名；不覆盖本次结果。原评估器仍严格验证完整Checkpoint、冻结配置和训练源码；新适配器只提供DEV原生E输入和输出隔离，不替换模型或放宽身份守卫。

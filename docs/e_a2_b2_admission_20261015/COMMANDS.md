# 实际执行命令与结果

以下均为已执行命令；保留原 cwd、参数与重定向。原始工具收据见 [commands.json](evidence/commands.json)。正式 prepare 两个工具句柄在 daemon 重启后丢失；其退出码保持 unknown，以完整 stdout、签名收据、文件哈希和当前源码独立复核完成。未运行正式 1M。

## A2_CPU_prepare

exit 1

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-prepare --config E_READY_A2_CPU_20261015_v1 > docs/e_a2_b2_admission_20261015/evidence/A2_CPU_prepare.log 2>&1
```

## B2_CPU_prepare

exit 1

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-prepare --config E_READY_B2_CPU_20261015_v1 > docs/e_a2_b2_admission_20261015/evidence/B2_CPU_prepare.log 2>&1
```

## data_build

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -u -m e_data_v1 build --source /home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/train_pool.json --out /home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/e_data_current_v1 --stride 15 > docs/e_a2_b2_admission_20261015/evidence/data_build.log 2>&1
```

## data_verify

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -u -m e_data_v1 verify --bundle /home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/e_data_current_v1 > docs/e_a2_b2_admission_20261015/evidence/data_verify.json 2>&1
```

## A2_CPU_prepare_v2

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 JAX_COMPILATION_CACHE_DIR=/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-prepare --config E_READY_A2_CPU_20261015_v2 > docs/e_a2_b2_admission_20261015/evidence/A2_CPU_prepare_v2.log 2>&1
```

## B2_CPU_prepare_v2

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 JAX_COMPILATION_CACHE_DIR=/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-prepare --config E_READY_B2_CPU_20261015_v2 > docs/e_a2_b2_admission_20261015/evidence/B2_CPU_prepare_v2.log 2>&1
```

## A2_CPU_start

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
taskset -c 0-11 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_COMPILATION_CACHE_DIR=/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-start --prepared artifacts/fg_s3_v2/prepared/E_READY_A2_CPU_20261015_v2 --stop-at 1100 > docs/e_a2_b2_admission_20261015/evidence/A2_CPU_start.log 2>&1
```

## B2_CPU_start

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
taskset -c 12-23 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_COMPILATION_CACHE_DIR=/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-start --prepared artifacts/fg_s3_v2/prepared/E_READY_B2_CPU_20261015_v2 --stop-at 1100 > docs/e_a2_b2_admission_20261015/evidence/B2_CPU_start.log 2>&1
```

## A2_CPU_resume

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
taskset -c 0-11 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_COMPILATION_CACHE_DIR=/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-resume --prepared artifacts/fg_s3_v2/prepared/E_READY_A2_CPU_20261015_v2 > docs/e_a2_b2_admission_20261015/evidence/A2_CPU_resume.log 2>&1
```

## B2_CPU_resume

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
taskset -c 12-23 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_COMPILATION_CACHE_DIR=/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-resume --prepared artifacts/fg_s3_v2/prepared/E_READY_B2_CPU_20261015_v2 > docs/e_a2_b2_admission_20261015/evidence/B2_CPU_resume.log 2>&1
```

## A2_GPU_prepare

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 JAX_COMPILATION_CACHE_DIR=/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-prepare --config E_READY_A2_GPU_20261015_v2 > docs/e_a2_b2_admission_20261015/evidence/A2_GPU_prepare.log 2>&1
```

## B2_GPU_prepare

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 JAX_COMPILATION_CACHE_DIR=/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-prepare --config E_READY_B2_GPU_20261015_v2 > docs/e_a2_b2_admission_20261015/evidence/B2_GPU_prepare.log 2>&1
```

## A2_GPU_start

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -u docs/e_a2_b2_admission_20261015/evidence/GPU_RESOURCE_AUDIT.py.txt A2_GPU_start -- taskset -c 0-11 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-start --prepared artifacts/fg_s3_v2/prepared/E_READY_A2_GPU_20261015_v2 --stop-at 500
```

## B2_GPU_start

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -u docs/e_a2_b2_admission_20261015/evidence/GPU_RESOURCE_AUDIT.py.txt B2_GPU_start -- taskset -c 12-23 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-start --prepared artifacts/fg_s3_v2/prepared/E_READY_B2_GPU_20261015_v2 --stop-at 500
```

## A2_GPU_resume

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -u docs/e_a2_b2_admission_20261015/evidence/GPU_RESOURCE_AUDIT.py.txt A2_GPU_resume -- taskset -c 0-11 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-resume --prepared artifacts/fg_s3_v2/prepared/E_READY_A2_GPU_20261015_v2
```

## B2_GPU_resume

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env`

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python -u docs/e_a2_b2_admission_20261015/evidence/GPU_RESOURCE_AUDIT.py.txt B2_GPU_resume -- taskset -c 12-23 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py admit-e-resume --prepared artifacts/fg_s3_v2/prepared/E_READY_B2_GPU_20261015_v2
```

## release_contracts

exit 0

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-release-20261016/gym-trading-env`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_s3_composition.py tests/entry_v1/test_fg_s3_e_source.py --basetemp=artifacts/fg_s3_v2/checks/e_ab_release_20261016 --junitxml=docs/e_a2_b2_admission_20261015/evidence/release_contracts.xml > docs/e_a2_b2_admission_20261015/evidence/release_contracts.log 2>&1
```

## A2_formal_prepare

PASS_FORMAL_PREPARE_AND_SOURCE_AUDIT；raw exit unknown

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-release-20261016/gym-trading-env`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py prepare --config E_A2_MARKET_MINUTE_FORMAL_1M > docs/e_a2_b2_admission_20261015/evidence/A2_formal_prepare.log 2>&1
```

## B2_formal_prepare

PASS_FORMAL_PREPARE_AND_SOURCE_AUDIT；raw exit unknown

cwd: `/home/v/Documents/work/research/fg-s3-e-ab-release-20261016/gym-trading-env`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:.:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py prepare --config E_B2_FULL_MARKET_MINUTE_FORMAL_1M > docs/e_a2_b2_admission_20261015/evidence/B2_formal_prepare.log 2>&1
```

## A2_GPU_start：资源包装内实际训练命令

```json
{
  "cwd": "/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env",
  "argv": [
    "taskset",
    "-c",
    "0-11",
    "/home/v/miniconda3/envs/dreamerv3/bin/python",
    "-u",
    "tools/fg_s3_v2_experiment.py",
    "admit-e-start",
    "--prepared",
    "artifacts/fg_s3_v2/prepared/E_READY_A2_GPU_20261015_v2",
    "--stop-at",
    "500"
  ],
  "environment": {
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "12",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "FG_S3_VISIBLE_GPUS": "0",
    "PYTHONPATH": "src:.:../dreamerv3",
    "JAX_COMPILATION_CACHE_DIR": "/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache"
  },
  "returncode": 0
}
```

## A2_GPU_resume：资源包装内实际训练命令

```json
{
  "cwd": "/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env",
  "argv": [
    "taskset",
    "-c",
    "0-11",
    "/home/v/miniconda3/envs/dreamerv3/bin/python",
    "-u",
    "tools/fg_s3_v2_experiment.py",
    "admit-e-resume",
    "--prepared",
    "artifacts/fg_s3_v2/prepared/E_READY_A2_GPU_20261015_v2"
  ],
  "environment": {
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "12",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "FG_S3_VISIBLE_GPUS": "0",
    "PYTHONPATH": "src:.:../dreamerv3",
    "JAX_COMPILATION_CACHE_DIR": "/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache"
  },
  "returncode": 0
}
```

## B2_GPU_start：资源包装内实际训练命令

```json
{
  "cwd": "/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env",
  "argv": [
    "taskset",
    "-c",
    "12-23",
    "/home/v/miniconda3/envs/dreamerv3/bin/python",
    "-u",
    "tools/fg_s3_v2_experiment.py",
    "admit-e-start",
    "--prepared",
    "artifacts/fg_s3_v2/prepared/E_READY_B2_GPU_20261015_v2",
    "--stop-at",
    "500"
  ],
  "environment": {
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "12",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "FG_S3_VISIBLE_GPUS": "0",
    "PYTHONPATH": "src:.:../dreamerv3",
    "JAX_COMPILATION_CACHE_DIR": "/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache"
  },
  "returncode": 0
}
```

## B2_GPU_resume：资源包装内实际训练命令

```json
{
  "cwd": "/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/gym-trading-env",
  "argv": [
    "taskset",
    "-c",
    "12-23",
    "/home/v/miniconda3/envs/dreamerv3/bin/python",
    "-u",
    "tools/fg_s3_v2_experiment.py",
    "admit-e-resume",
    "--prepared",
    "artifacts/fg_s3_v2/prepared/E_READY_B2_GPU_20261015_v2"
  ],
  "environment": {
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "12",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "FG_S3_VISIBLE_GPUS": "0",
    "PYTHONPATH": "src:.:../dreamerv3",
    "JAX_COMPILATION_CACHE_DIR": "/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/jax_cache"
  },
  "returncode": 0
}
```

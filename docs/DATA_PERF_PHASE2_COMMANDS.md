# DATA-PERF 第二阶段：后续 CPU 教材生产命令

正式生产由用户执行。下面只选择一个合约和明确官方交易日范围；不是 18 合约循环。
完整主力资格生产需要把 START_DAY/END_DAY 改为该合约已认证的完整官方首末交易日。
示例 FG201 2021-12-08 至 2021-12-09 是本轮验收窗口，不冒充完整主力区间。

冻结 Producer 目录：
`/home/v/Documents/work/research/fg-s3-data-perf-phase2-20261020/rounds/final-candidate/gym-trading-env`。
Gym SHA `2dd45283567398e782cbcd0a958e9be873386921`，Dreamer SHA
`b7a693b3c7b0b45eeecdec954818be42d40e67b6`，均为 clean 冻结工作树。
外部 pin SHA256：`20523134f4241a0bdf3fb4a8f89bd7dc1ed4f9237a5b67c64eddf69e89786eed`。
不使用旧 TRAIN01 Producer 或覆盖旧目录；生成新的批次身份。

```bash
set -euo pipefail
set -o noclobber

PRODUCER=/home/v/Documents/work/research/fg-s3-data-perf-phase2-20261020/rounds/final-candidate/gym-trading-env
PIN=/home/v/Documents/work/research/fg-s3-data-perf-phase2-20261020/gym-trading-env/docs/locks/DATA_PERF_PHASE2_PRODUCER_PIN_20261020.json
PIN_SHA=20523134f4241a0bdf3fb4a8f89bd7dc1ed4f9237a5b67c64eddf69e89786eed
PYTHON=/home/v/miniconda3/envs/dreamerv3/bin/python
SOURCE_POOL=/home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/train_pool.json
ENV_RECEIPT=/data/logdir/fg_s3_market_minute/20261013_v2/env/v1

CONTRACT=CZCE.FG201
START_DAY=2021-12-08
END_DAY=2021-12-09
BATCH_ROOT=/data/s3_corpus/phase2_new_batch_FG201_20261020

mkdir -p "$BATCH_ROOT/receipts"
cd "$PRODUCER"
export PYTHONPATH=.:src:../dreamerv3
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export PYTHONINTMAXSTRDIGITS=0 MPLCONFIGDIR=/tmp/fg_data_perf_mpl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

taskset -c 0-19 "$PYTHON" -B tools/fg_s3_e_quality_corpus.py validate \
  --source-pool "$SOURCE_POOL" --env-receipt "$ENV_RECEIPT" --contract "$CONTRACT" \
  --start-day "$START_DAY" --end-day "$END_DAY" > "$BATCH_ROOT/receipts/validate.log" 2>&1

taskset -c 0-19 "$PYTHON" -B tools/fg_s3_e_quality_corpus.py generate \
  --source-pool "$SOURCE_POOL" --env-receipt "$ENV_RECEIPT" --contract "$CONTRACT" \
  --start-day "$START_DAY" --end-day "$END_DAY" --mode data --seed 202 --workers 20 \
  --out "$BATCH_ROOT/corpus" --debug > "$BATCH_ROOT/receipts/generate.log" 2>&1

taskset -c 0-19 "$PYTHON" -B tools/fg_s3_e_quality_reader.py source \
  --corpus "$BATCH_ROOT/corpus" --out "$BATCH_ROOT/native" --purpose native-verification \
  --producer-gym "$PRODUCER" --producer-lock "$PIN" --producer-lock-sha256 "$PIN_SHA" \
  --workers 20 > "$BATCH_ROOT/receipts/native_g3.log" 2>&1

taskset -c 0-19 "$PYTHON" -B tools/fg_s3_e_quality_reader.py read \
  --corpus "$BATCH_ROOT/corpus" --out "$BATCH_ROOT/reader" --length 64 --seed 202 \
  --without-model > "$BATCH_ROOT/receipts/reader.log" 2>&1

taskset -c 0-19 "$PYTHON" -B tools/fg_s3_e_quality_corpus.py coverage \
  --corpus "$BATCH_ROOT/corpus" > "$BATCH_ROOT/receipts/coverage.log" 2>&1
```

Corpus、source lock、Teacher、课程和代码身份写入独立 manifest；保留 Native G3、Reader
及 Coverage 收据。`COLLECTED_UNVERIFIED` 原 manifest 不改写成 PASS，用新增验收收据
关联不可变内容。Reader 支持程度和真实质量覆盖看本批分母及缺题型，不能由命令成功推断。
本命令不运行训练、不打开正式优化器，不为后续 F1/F2/F3 授权。

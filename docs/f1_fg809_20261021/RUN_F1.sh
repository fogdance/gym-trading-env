#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-status}
if [[ $# -gt 1 ]]; then
  echo 'Usage: RUN_F1.sh [status|prepare|start|resume]' >&2
  exit 2
fi
case "$ACTION" in status|prepare|start|resume) ;; *) echo 'Unknown F1 action' >&2; exit 2 ;; esac

GYM=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=$(dirname -- "$GYM")
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
CONFIG="$GYM/configs/experiments/fg_s3_v2/e_quality_f1_fg809_size12m_u10000_20261021.yaml"
CONTROL=/data/logdir/fg_single3/s3_v2/F1_FG809_12M_U10000_20261021
PREPARED="$CONTROL/prepared"
cd "$GYM"
export PYTHONPATH="$GYM:$GYM/src:$WORKSPACE/dreamerv3"
export PYTHONDONTWRITEBYTECODE=1 PYTHONINTMAXSTRDIGITS=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$CONTROL/tmp" TMP="$CONTROL/tmp" TEMP="$CONTROL/tmp"
export XDG_CACHE_HOME="$CONTROL/cache" MPLCONFIGDIR="$CONTROL/cache/matplotlib"
export JAX_COMPILATION_CACHE_DIR="$CONTROL/cache/jax" CUDA_CACHE_PATH="$CONTROL/cache/cuda"

if [[ "$ACTION" == status ]]; then
  export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
  exec "$PY" -B docs/f1_fg809_20261021/check_launch.py
fi

# Check the frozen training selection. Production certification has its own
# external digest in the YAML; it never requires a live Producer or G3 rerun.
"$PY" -B docs/f1_fg809_20261021/check_launch.py --config-only
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH"

prepare_f1() {
  "$PY" -B tools/fg_s3_v2_experiment.py f1-prepare --config "$CONFIG" \
    --source-purpose certified-corpus
}

if [[ "$ACTION" == prepare ]]; then
  prepare_f1
  exit
fi
if [[ "$ACTION" == start && ! -e "$PREPARED" ]]; then
  prepare_f1
fi
if [[ ! -f "$PREPARED/settings.json" || ! -f "$PREPARED/f1_preparation.json" ||
      ! -f "$PREPARED/reader_binding.json" || ! -f "$PREPARED/join.sqlite" ]]; then
  echo "F1 preparation is absent or incomplete: $PREPARED" >&2
  echo 'Existing partial results are retained; inspect the native preparation error before retrying.' >&2
  exit 2
fi
exec "$PY" -B tools/fg_s3_v2_experiment.py "f1-$ACTION" --prepared "$PREPARED" \
  --authorize-training

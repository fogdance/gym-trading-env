#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-status}
if [[ $# -gt 1 ]]; then
  echo 'Usage: RUN_F1.sh [status|prepare|start|resume]' >&2
  exit 2
fi
case "$ACTION" in status|prepare|start|resume) ;; *) echo 'Unknown F1 action' >&2; exit 2 ;; esac

GYM=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
CONFIG="$GYM/configs/experiments/fg_s3_v2/e_quality_f1_fg809_size12m_u10000_20261021.yaml"
PREPARED="$GYM/artifacts/fg_s3_v2/e_quality_wp04/prepared/F1_FG809_12M_U10000_20261021"
PIN="$GYM/../release/VERIFIER_PIN.json"
LOCK="$GYM/../release/LAUNCH_LOCK.json"
cd "$GYM"
export PYTHONPATH="$GYM:$GYM/src:$GYM/../dreamerv3"
export PYTHONDONTWRITEBYTECODE=1 PYTHONINTMAXSTRDIGITS=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/fg_s3_f1_fg809_mpl

if [[ "$ACTION" == status ]]; then
  export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
  exec "$PY" -B docs/f1_fg809_20261021/check_launch.py
fi

# Use the independently recorded external digest; never regenerate a Pin or
# infer trust from whatever HEAD happens to be checked out when launching.
PIN_SHA=$("$PY" -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["verifier_pin"]["sha256"])' "$LOCK")
PIN_ARGS=(--producer-gym "$GYM" --producer-lock "$PIN" --producer-lock-sha256 "$PIN_SHA")

prepare_f1() {
  "$PY" -B tools/fg_s3_v2_experiment.py f1-prepare --config "$CONFIG" \
    --source-purpose native-verification "${PIN_ARGS[@]}"
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
  "${PIN_ARGS[@]}" --authorize-training

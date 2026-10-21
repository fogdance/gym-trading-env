#!/usr/bin/env bash
set -euo pipefail
ACTION=${1:-status}
if [[ $# -gt 1 ]]; then echo 'Usage: RUN_F3.sh [status|prepare|start|resume]' >&2; exit 2; fi
case "$ACTION" in status|prepare|start|resume) ;; *) echo 'Unknown F3 action' >&2; exit 2 ;; esac
GYM=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=$(dirname -- "$GYM")
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
CONFIG="$GYM/configs/experiments/fg_s3_v2/e_quality_f3_fg809_size12m_u10000_20261022.yaml"
CONTROL=/data/logdir/fg_single3/s3_v2/F3_FG809_12M_U10000_20261022
PREPARED="$CONTROL/prepared"
cd "$GYM"
export PYTHONPATH="$GYM:$GYM/src:$WORKSPACE/dreamerv3" PYTHONDONTWRITEBYTECODE=1 PYTHONINTMAXSTRDIGITS=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$CONTROL/tmp" TMP="$CONTROL/tmp" TEMP="$CONTROL/tmp"
export XDG_CACHE_HOME="$CONTROL/cache" MPLCONFIGDIR="$CONTROL/cache/matplotlib"
export JAX_COMPILATION_CACHE_DIR="$CONTROL/cache/jax" CUDA_CACHE_PATH="$CONTROL/cache/cuda"
# Metadata-only frozen selection check; shared admission reuses production G3.
"$PY" -B - "$CONFIG" "$CONTROL/LAUNCH_LOCK.json" <<'PY'
import json, subprocess, sys
from pathlib import Path
from gym_trading_env.research.fg_replay import digest,read_manifest
config,lock_path=map(Path,sys.argv[1:])
lock=read_manifest(lock_path);repo=Path.cwd()
if digest(config)!=lock['config_sha256']:raise ValueError('F3_FROZEN_CONFIGURATION_CHANGED')
for role,root in [('gym',repo),('dreamer',repo.parent/'dreamerv3')]:
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if head!=lock['repositories'][role]['sha']:raise ValueError('F3_FROZEN_HEAD_CHANGED: '+role)
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
        raise ValueError('F3_FROZEN_WORKTREE_CHANGED: '+role)
if digest(Path(lock['source_checkpoint'])/'manifest.json')!=lock['source_manifest_sha256']:
    raise ValueError('F3_SELECTED_F1_CHECKPOINT_CHANGED')
print(json.dumps(dict(status='F3_FROZEN_CONFIGURATION_VERIFIED',source_stage='F1',
    source_checkpoint=lock['source_checkpoint'],optimizer_updates=10000)))
PY
if [[ "$ACTION" == status ]]; then exit; fi
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH"
echo "F3 launch log: $CONTROL/launch.log"
echo 'F3 update log: /data/logdir/fg_single3/s3_v2/E_QUALITY_F3_FG809_12M_U10000_20261022/updates.jsonl'
prepare_f3() {
  "$PY" -B tools/fg_s3_v2_experiment.py f3-prepare --config "$CONFIG" \
    --source-purpose certified-corpus 2>&1 | tee -a "$CONTROL/launch.log"
}
if [[ "$ACTION" == prepare ]]; then prepare_f3; exit; fi
if [[ "$ACTION" == start && ! -e "$PREPARED" ]]; then prepare_f3; fi
if [[ ! -f "$PREPARED/settings.json" || ! -f "$PREPARED/f3_preparation.json" ||
      ! -f "$PREPARED/reader_binding.json" || ! -f "$PREPARED/join.sqlite" ]]; then
  echo "F3 preparation is absent or incomplete: $PREPARED" >&2
  exit 2
fi
"$PY" -B tools/fg_s3_v2_experiment.py "f3-$ACTION" --prepared "$PREPARED" \
  --authorize-training 2>&1 | tee -a "$CONTROL/launch.log"

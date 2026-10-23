#!/usr/bin/env bash
# Sequential Thinking Student V1 base-corpus production, one contract at a time.
# Run: bash tools/run_ts_v1_train18_corpus.sh
# Optional overrides: TS_V1_GYM_DIR, TS_V1_PYTHON, TS_V1_TRAIN_POOL,
# TS_V1_CORPUS_ROOT, TS_V1_WORKERS.
set -euo pipefail

TS_V1_GYM_DIR=${TS_V1_GYM_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
TS_V1_PYTHON=${TS_V1_PYTHON:-/home/v/miniconda3/envs/dreamerv3/bin/python}
TS_V1_TRAIN_POOL=${TS_V1_TRAIN_POOL:-/data/logdir/fg_single3/shared/pools/fg_train18_quality_tp_v1/train_pool.json}
TS_V1_CORPUS_ROOT=${TS_V1_CORPUS_ROOT:-/data/logdir/fg_single3/thinking_student/corpus}
TS_V1_WORKERS=${TS_V1_WORKERS:-24}

cd "$TS_V1_GYM_DIR"
export PYTHONPATH="$PWD/src:$PWD:$PWD/../dreamerv3"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""

# Keep jq errors in the main shell so a failed pool read stops before production.
contract_list=$(jq -er '.files[]
  | select(.role == "train" and .eligible_for_training == true)
  | .contract' "$TS_V1_TRAIN_POOL")
mapfile -t contracts <<< "$contract_list"

if [[ ${#contracts[@]} -ne 18 ]]; then
  echo "TRAIN 合约数量不是 18，停止。" >&2
  exit 1
fi
if [[ ! $TS_V1_WORKERS =~ ^[1-9][0-9]*$ ]]; then
  echo "TS_V1_WORKERS 必须是正整数，停止。" >&2
  exit 1
fi

export MPLCONFIGDIR="$TS_V1_CORPUS_ROOT/.mpl-cache"
mkdir -p "$TS_V1_CORPUS_ROOT/logs" "$MPLCONFIGDIR"

for contract in "${contracts[@]}"; do
  if [[ -f "$TS_V1_CORPUS_ROOT/$contract/source.json" ]]; then
    echo "SKIP $contract（已完成）"
    continue
  fi

  echo "START $contract"
  "$TS_V1_PYTHON" -B -u -m gym_trading_env.research.ts_v1_corpus \
    --contract "$contract" \
    --pool "$TS_V1_TRAIN_POOL" \
    --out "$TS_V1_CORPUS_ROOT" \
    --workers "$TS_V1_WORKERS" \
    2>&1 | tee -a "$TS_V1_CORPUS_ROOT/logs/$contract.log"

  echo "DONE $contract"
done

# Load every completed contract through the actual training reader.
"$TS_V1_PYTHON" -B - "$TS_V1_CORPUS_ROOT" "${contracts[@]}" <<'PY'
import sys
from pathlib import Path

from dreamerv3.thinking_student.corpus import TSCorpus

root = Path(sys.argv[1])
total = 0
for contract in sys.argv[2:]:
    corpus = TSCorpus(root / contract)
    if corpus.source['contract'] != contract or corpus.source['fold'] != 'TRAIN':
        raise ValueError(f'{contract}: corpus contract or TRAIN fold differs')
    print(f'OK {contract}: {corpus.rows} rows')
    total += corpus.rows
    del corpus

print(f'完成：{len(sys.argv) - 2} 个合约，{total} 行。')
PY

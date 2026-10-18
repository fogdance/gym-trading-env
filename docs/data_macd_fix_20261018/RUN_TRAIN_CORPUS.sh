#!/usr/bin/env bash
# Explicit user-operated CPU production. No default batch and no all-contract mode.
set -euo pipefail
PLAN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
EDIT_ROOT=/home/v/Documents/work/research/fg-s3-data-integration-20261018
PRODUCTION=/data/s3_corpus/early_train_20261018_cpu20_v2
PRODUCER="$PRODUCTION/code/gym-trading-env"
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
GYM_SHA=b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46
DREAMER_SHA=fd144113ce706e2d8f1ce8e7828ece0799c8bf7a
PIN_SHA=7310b2a2cdcfba7fdc8697bc46fff0d6ca39e4f7932b90710eff7de1a7179515
PIN="$PRODUCTION/locks/PRODUCER_PIN.json"
POOL=/home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/train_pool.json
ENV_RECEIPT=/data/logdir/fg_s3_market_minute/20261013_v2/env/v1

case "${1:-help}" in
  help|--help|-h)
    printf '%s\n' 'Usage: bash RUN_TRAIN_CORPUS.sh prepare' \
      '       bash RUN_TRAIN_CORPUS.sh produce B01_FG809' \
      'Explicit alternatives: B02_FG009, B03_FG209, B04_FG405.' \
      'Each invocation authorizes only its locked TRAIN range and up to 20 CPU workers.'
    exit 0 ;;
  prepare|produce) ;;
  *) printf '%s\n' 'Unknown command; use --help.' >&2; exit 2 ;;
esac

(cd "$PLAN_DIR" && sha256sum --quiet -c PLAN_SHA256SUMS)
[[ "$(findmnt -n -o SOURCE -T /data)" == /dev/sda1 ]] || { printf '%s\n' 'Unexpected /data device.' >&2; exit 2; }
[[ "$(readlink -m -- "$PRODUCTION")" == "$PRODUCTION" ]] || { printf '%s\n' 'Symlink production ancestor refused.' >&2; exit 2; }
if [[ -e /data/s3_corpus ]]; then
  [[ "$(findmnt -n -o SOURCE -T /data/s3_corpus)" == /dev/sda1 ]] || { printf '%s\n' 'Unexpected Corpus device.' >&2; exit 2; }
fi
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=.:src:../dreamerv3
export MPLCONFIGDIR="$PRODUCTION/cache/matplotlib" TMPDIR="$PRODUCTION/cache/tmp"

if [[ "$1" == prepare ]]; then
  [[ $# == 1 && ! -e "$PRODUCTION" ]] || { printf '%s\n' 'Preparation requires a new version directory.' >&2; exit 2; }
  mkdir -p "$PRODUCTION/code" "$PRODUCTION/locks" "$MPLCONFIGDIR" "$TMPDIR"
  git -C "$EDIT_ROOT/gym-trading-env" worktree add --detach "$PRODUCER" "$GYM_SHA"
  git -C "$EDIT_ROOT/dreamerv3" worktree add --detach "$PRODUCTION/code/dreamerv3" "$DREAMER_SHA"
  cp "$PLAN_DIR/locks/PRODUCER_PIN.json" "$PIN"
  printf '%s  %s\n' "$PIN_SHA" "$PIN" | sha256sum --quiet -c -
  printf '%s\n' 'Frozen code prepared; Corpus production NOT_RUN.'
  exit 0
fi

[[ $# == 2 ]] || { printf '%s\n' 'Select exactly one batch.' >&2; exit 2; }
BATCH=$2
case "$BATCH" in
  B01_FG809) CONTRACT=CZCE.FG809; START=2018-08-06; END=2018-08-08 ;;
  B02_FG009) CONTRACT=CZCE.FG009; START=2020-08-07; END=2020-08-11 ;;
  B03_FG209) CONTRACT=CZCE.FG209; START=2022-08-08; END=2022-08-10 ;;
  B04_FG405) CONTRACT=CZCE.FG405; START=2024-04-02; END=2024-04-08 ;;
  *) printf '%s\n' 'Batch absent from the locked four-batch plan.' >&2; exit 2 ;;
esac
CORPUS="$PRODUCTION/batches/$BATCH"
RECEIPTS="$PRODUCTION/receipts/$BATCH"
[[ -d "$PRODUCER" && ! -e "$CORPUS" && ! -e "$RECEIPTS" ]] || { printf '%s\n' 'Run prepare first; never overwrite or retry in an existing batch directory.' >&2; exit 2; }
[[ "$(git -C "$PRODUCER" rev-parse HEAD)" == "$GYM_SHA" ]]
[[ "$(git -C "$PRODUCTION/code/dreamerv3" rev-parse HEAD)" == "$DREAMER_SHA" ]]
mkdir -p "$RECEIPTS" "$MPLCONFIGDIR" "$TMPDIR"
cp "$PLAN_DIR/batches/$BATCH.manifest.json" "$RECEIPTS/PLANNED_BATCH_MANIFEST.json"
cd "$PRODUCER"
"$PY" -B - "$PRODUCER" "$PIN" "$PIN_SHA" > "$RECEIPTS/frozen_producer_audit.json" <<'PY'
import json, sys
from gym_trading_env.research.fg_s3_e_quality_supervision import frozen_producer_audit
print(json.dumps(frozen_producer_audit(sys.argv[1], producer_lock=sys.argv[2], producer_lock_sha256=sys.argv[3])))
PY
free -b > "$RECEIPTS/memory_before.txt"
df -B1 /data > "$RECEIPTS/disk_before.txt"

phase() {
  local name=$1
  shift
  printf '%s\n' "$BATCH $name: $RECEIPTS/$name.log"
  printf '%q ' "$@" > "$RECEIPTS/$name.command.txt"
  printf '\n' >> "$RECEIPTS/$name.command.txt"
  /usr/bin/time -v -o "$RECEIPTS/$name.time.txt" taskset -c 0-19 "$@" > "$RECEIPTS/$name.log" 2>&1
}

phase validate "$PY" -B tools/fg_s3_e_quality_corpus.py validate \
  --source-pool "$POOL" --env-receipt "$ENV_RECEIPT" --contract "$CONTRACT" --start-day "$START" --end-day "$END"
"$PY" -B - "$RECEIPTS/validate.log" "$PLAN_DIR/batches/$BATCH.manifest.json" <<'PY'
import json, sys
actual=json.loads(open(sys.argv[1]).read().splitlines()[-1])
plan=json.load(open(sys.argv[2]))
if actual['source']['content_hash'] != plan['source_lock_content_hash']:
    raise SystemExit('Source lock changed; prepare a new reviewed plan')
PY
phase generate "$PY" -B tools/fg_s3_e_quality_corpus.py generate \
  --source-pool "$POOL" --env-receipt "$ENV_RECEIPT" --contract "$CONTRACT" \
  --start-day "$START" --end-day "$END" --workers 20 --seed 202 --mode data --debug --out "$CORPUS"
phase native_g3 "$PY" -B tools/fg_s3_e_quality_reader.py source \
  --corpus "$CORPUS" --out "$RECEIPTS/native" --purpose native-verification \
  --producer-gym "$PRODUCER" --producer-lock "$PIN" --producer-lock-sha256 "$PIN_SHA" --workers 20
phase reader "$PY" -B tools/fg_s3_e_quality_reader.py read \
  --corpus "$CORPUS" --out "$RECEIPTS/reader" --length 64 --seed 202 --without-model
phase coverage "$PY" -B "$PLAN_DIR/../train_corpus_20261018/audit_coverage.py" \
  --corpus "$CORPUS" --reader-dir "$RECEIPTS/reader" --out "$RECEIPTS/quality_coverage.json"
"$PY" -B - "$CORPUS" "$RECEIPTS" "$PIN_SHA" <<'PY'
import hashlib, json, sys
from pathlib import Path
from gym_trading_env.research.fg_replay import read_manifest
corpus, receipts=map(Path, sys.argv[1:3])
def read(path): return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
m=read_manifest(corpus/'corpus_manifest.json')
native=read(receipts/'native/native_verification.json')
reader=read(receipts/'reader/WP03_COURSE_READER_COVERAGE.json')
plan=read(receipts/'PLANNED_BATCH_MANIFEST.json')
if (native['exit_code'] != 0 or native['result']['G3'] != 'GO'
        or native['result']['corpus_identity'] != m['content_hash']
        or reader['corpus_identity'] != m['content_hash']
        or m['source_lock_hash'] != plan['source_lock_content_hash']
        or reader['actual_optimizer_exposures'] != 0 or m['training_admitted'] is not False):
    raise SystemExit('Batch receipt identity mismatch')
files={label+'/'+str(path.relative_to(root)):sha(path)
    for label,root in [('corpus',corpus),('receipts',receipts)]
    for path in sorted(root.rglob('*')) if path.is_file()}
if (any(files['corpus/'+name] != expected for name,expected in m['files'].items())
        or sha(receipts/'reader/join.sqlite') != reader['index_sha256']):
    raise SystemExit('Corpus or Reader index changed after verification')
value=dict(protocol='FG_S3_E_TRAIN_CORPUS_BATCH_RESULT_V1',batch_id=plan['batch_id'],
    status='NATIVE_G3_AND_INDEPENDENT_READER_COMPLETED',corpus_identity=m['content_hash'],
    source_lock=m['source_lock_hash'],course_identity=m['course_identity_hash'],
    teacher_identity=m['teacher_identity'],producer_repositories=m['code']['repositories'],
    producer_pin_sha256=sys.argv[3],index_sha256=reader['index_sha256'],
    curriculum_support=reader['curriculum_support'],training_admitted=False,
    optimizer_updates=0,files=files)
value['content_hash']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
with (receipts/'BATCH_MANIFEST.json').open('x') as stream:
    json.dump(value,stream,indent=2); stream.write('\n')
print(json.dumps({k:v for k,v in value.items() if k!='files'}))
PY
printf '%s\n' "$BATCH complete. Review coverage before selecting another batch. Formal training remains unadmitted."

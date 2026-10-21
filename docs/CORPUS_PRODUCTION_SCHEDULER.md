# CPU Corpus production and stage recovery

The shared entry is `tools/fg_s3_e_quality_corpus.py`. This is CPU DATA
production, with no optimizer or training authorization. The template is
`configs/experiments/fg_s3_v2/corpus_production.yaml`; the frozen, ready
configuration and producer pin live outside the two repositories in
`../release/`, so writing production state never changes a producer inventory.

```bash
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
$PY -B tools/fg_s3_e_quality_corpus.py status --config ../release/corpus_production.yaml
$PY -B tools/fg_s3_e_quality_corpus.py resume --config ../release/corpus_production.yaml
# For a new root without adopted data, produce-all uses the same configuration.
```

The shipped template intentionally has an unresolved **new** producer pin.
After committing the final runtime, write a signed
`FG_S3_E_QUALITY_PRODUCER_PIN_V1` containing both exact repository SHAs,
hash the pin bytes, and put that digest into an external copy of the YAML.
The implementation requires the existing frozen-producer audit; no code
inventory or producer identity is bypassed. The old first batch retains its
original absolute path, inventory and producer pin. Its missing ABI field is
handled by the existing reviewed 3.1.0 mapping, without rewriting the Corpus.

| Policy | Total processes at capacity 24 | Account | Teachers |
| --- | ---: | ---: | ---: |
| P0 WAIT | 8 | 1 | 7 |
| P1 always ADD | 2 | 1 | 1 |
| P2_low | 2 | 1 | 1 |
| P2_mid | 2 | 1 | 1 |
| P3 delayed entry | 2 | 1 | 1 |
| P4 Quality Oracle schedule | 8 | 1 | 7 |

The budgets include the account process. Native threads are one per worker;
affinity is all 24 logical CPUs. The minimum is two processes per policy.
If physical memory reduces capacity, extra processes are apportioned by largest
remainder over configured demand above that minimum. At capacity 17 the
allocation is `[5,2,2,2,2,4]`; at 20 it is `[6,2,2,2,2,6]`. Capacity below 12
stops production with measured memory and reservation details. Swap is never
part of the memory budget. Change memory estimates only with measured evidence.

One controller owns a root lock and a CPU/memory ledger. At most two batches
are in flight, with one generation phase and one G3 phase. The next batch
may prepare and generate after the previous generation completes. Preparation
and publication need one CPU token. Every strategy requests its own allocation;
older-batch G3 has priority, P0/P4 lead their phase, and fitting smaller tasks
can use free capacity. Teacher counts stay fixed within a strategy. Updated
account processes exit before their complete lease is returned. Idle phase
coordinators keep memory reservations. Admission subtracts nonresident reserved
memory from `MemAvailable`, accounts for managed group PSS, and grows the
worker estimate to 125% of the largest observed process PSS. Headroom is for
growth/system operation, not a CPU reservation for other work.

The first adopted batch runs G3 under its recorded old producer, using only
the old `--workers` interface. Structured completion events release closed
teacher pools once; old executor account reservations stay held until process
exit. These progress events never certify G3. Each batch must finish:

`runtime_limits → validate → generate → native_g3 → reader → coverage → publish`

State is atomic JSON in `<production_root>/control/state.json`. Each attempt
has a fresh numbered directory under `attempts/<batch>/<phase>/`. A generation
retry replays the whole batch; a G3 retry reuses the generated Corpus and
rechecks all six policies. Later phases retry only the incomplete phase.
Successful phases are reused only after completion receipts, artifact identities
and file digests pass validation. Exit zero, a directory, or a `.time.txt`
summary alone cannot establish success. A phase that finished without committing
state is rerun safely. Previous attempts are preserved; Corpus directories
are never renamed because their bindings contain absolute paths.

`resources` and `max_inflight_batches` can change on a stopped resume; each
attempt records the effective settings. Dates, seeds, plans, source paths and
producer pins belong to immutable recipe identity. A changed recipe needs a
new production root. An artifact mutation fails closed for manual audit.

Ctrl+C or a failed phase stops admission and terminates owned process groups.
Resume checks PID, process-group and `/proc` start ticks before cleaning stale
processes. Recorded live descendants can prove ownership after a leader exits;
an unprovable group is reported for audit rather than signaled by PID alone.
The controller must run under a supervisor that preserves its state directory.

Publication preserves the existing `BATCH_MANIFEST.json`/Reader contract,
including canonical `native/` and `reader/` paths. Only selected successful
attempts are included. Mutable controller logs, scheduler state and failed
attempts stay outside publication inventories. `status` reports controller
state without claiming a new native verification. A completed production
resume rechecks stored artifacts and launches no new work.

Existing `generate`, `verify` and Reader `source` commands also accept
`--policy-cpus P0=8,P1=2,P2_low=2,P2_mid=2,P3=2,P4=8` with `--workers 24`.
Omitting that flag retains their previous worker allocation contract. CPU
utilization alone is not acceptance: compare verified market-minute throughput,
wall time, and native arrays/labels using `tools/benchmark_s3_corpus.py compare`.

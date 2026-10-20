# Source baseline — A0

Gym `8619f04b6f996f4c4d7f169486e4d10c25a3b2b9`; Dreamer `b7a693b3c7b0b45eeecdec954818be42d40e67b6`. Both actual integration HEADs were clean before creating sibling worktrees on `feat/corpus-semver-r1-r4-20261020`.

## Observed paths and defects

- Corpus → `SupervisionSource` → `inspect_source` → `QualitySupervisionAdmission` → WP03 Reader/Join → F1/F2/F3 stage adapters → original Agent and FullCheckpoint.
- Low-level verifier accepts an externally reviewed producer pin, but all three stage APIs and the experiment CLI omit it. Historical default pair is e57189b/44a6cc6; current frozen production is 8619f04/b7a693b.
- F1/F2/F3 probes/runtime Checkpoints omit strict_latest. A temporary CPU reproduction returned step0 after removing step1/done; strict mode rejected the same fault.
- Existing external+strict API also accepted explicit load(step0) while latest=step1, then save(step0) regressed the pointer. Strict latest() did not check backend until load(). These reproductions used temporary mock state, not real Runs.
- Offline stages reconcile update chronology after restoring mutable state; unlike F4, their Checkpoints do not bind committed journal bytes.

## Identity and compatibility

1. Corpus ABI: exact schemas, observation geometry/order/dtype, causal visibility/MACD, financial/reward and L1/L2/Join semantics.
2. Immutable instance: original manifest/content hash, all member hashes, source/course/teacher identities. Never replace these with consumer Git SHA.
3. Producer and verifier provenance: original producer inventory remains unchanged; actual frozen verifier pair and external pin are separately recorded.
4. Run identity: execution code/config, model/optimizer, source/index, sampler and recovery policy remain strict for same-Run resume. New consumer code may need a new Run, not a new Corpus.

Both completed performance V3 assets use Teacher V3, FG_S3_EXACT_RATIONAL_HEX_V1, gamma 332/333, full six behaviors and source_guard_interval=1. Historical 20261018 v2 production has Teacher V2 without hexadecimal witness declaration and is not mapped to ABI 3.1.0. Internal Course/Model/trace V2 names are independent version axes, not evidence of legacy teacher compatibility.

## Protection

`A0_baseline.json` records every declared immutable member hash for the completed phase2/phase3 assets and metadata-only production observations. Running production and frozen sibling source trees are read-only. Live production may legitimately add files; no claim of whole-tree immutability is made for a job still producing. No TEST market bytes, production G3, GPU or formal training were executed for A0.

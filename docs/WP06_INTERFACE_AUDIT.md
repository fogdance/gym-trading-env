# WP06 interface audit

Observed source baseline: Gym c1679a7 + explicit V3-only Join diagnostics (7ea9556); Dreamer 0e39dfd. Public branches not advanced. This is DEV integration, not FORMAL training admission.

| Existing interface | Reuse / required F3 extension |
| --- | --- |
| Reader INDEX_SCHEMA | Already V3; builder writes database and receipt as V3. Explicitly refuse old/missing receipt and forged V3 outer/V2 inner protocol. No compatibility path. |
| EQualityCourseReader | Existing ACTOR_STRICT plus NATURAL/STRATIFIED/MIXED and RNG restore. All F3 streams use the same admitted corpus; no DATA-MULTI implementation. |
| audit_windows | Authenticates native observation/action/reward/stepid/episode bytes but deliberately excludes teacher fields. F3 must additionally authenticate selected supervision fields against the same index at each post-step Stepid; t=0 remains unsupervised prefix. |
| project_label | Existing L2 strict label and two-action legality projection remains canonical. No inference from behavior action or L1 score. |
| causal_context / wm_only | Reuse original prefix + K causal history and T target ids, WM training=False and stop_gradient. |
| quality_actor / masked policy | Reuse original head and single native mask/unimix; F3 route bypasses prediction computation/fusion, not admission checks. |
| teaching_objective | Existing L1 score-only pair is unsuitable for F3. New F3 loss shares mean/pair lineage primitives, adds explicit L2-consistent filtering; legacy behavior remains separate. |
| masked_joint_optimizer | Extend explicit allowed stages to F3 actor prefix; preserve optimizer schema, per-parameter slots and signed zero. |
| transfer_parameters / inspect_f1 | Reuse non-opt copier with truthful target-stage receipt; add F2 source inspection including skipped-batch ledger and upstream F1 proof. No F2-to-F3 same-run resume. |
| CourseStream / FullCheckpoint / RuntimeState | Thin F3 composition of original readers/prefetch; store every stream RNG, queued input, consumed-batch and optimizer counters. |

Real-source observation during planning: B01 MACD V3 corpus/read-only feasibility was present and passed a fresh V3 Join + 2 windows at K32/T64. On resumed execution, the original corpus_manifest path no longer exists; G0_WP06_real exited 2. Historical B01 audit is not current execution evidence. Current source location requested from user. Formal or real-data GO cannot be inferred from the old report.

Stage sequence: A protocol/factory; B source handoff; C native supervised update; D loss components/auxiliary sampling; E student-only evaluation/learning; F original checkpoint/CLI/acceptance. All parameter updates in this task are synthetic; TEST and production runs remain excluded.

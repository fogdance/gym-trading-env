# WP07 implementation plan

Paired baseline: Gym 2129afa2d21e23f0bf915c15e41f649919f4931e; Dreamer 208a499a7a3afa1dfd240e51de5c6a1e27dd1c47. Independent paired feature branch feat/s3-e-quality-wp07-20261019.

A: versioned execution/source handoff. B: native teacher-free SHADOW. C: delayed TRAIN sidecar and native online/history streams. D: original Actor BC and exact freeze. E: original imagination/optimizer using the same quality_actor plus Value. F: independent technical/formal gates and exploration. G: original FullCheckpoint atomic state and exact cold resume. H: isolated integration, source review, scoped regression and Roadmap. Each stage has paired commits and actual test receipts.

Reuse MultiCorpusF3Stream, original Env/Driver/Replay, F3 objective, original imagination and FullCheckpoint. No TEST reads, real TRAIN updates or long GPU training. Missing formal evidence remains NOT_RUN/NO_GO. DEV does not require per-edit code freezing or full G3. Financial rules, causal MACD, teacher isolation and exact checkpoint boundaries are always enforced.

Runtime artifacts: artifacts/fg_s3_v2/e_quality_wp07. Text receipts: docs/receipts/wp07. Baseline truth: WP07_BASELINE.json. Final report: WP07_IMPLEMENTATION_REVIEW.md.

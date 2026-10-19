# WP07 progress

Baseline inspected; independent paired worktrees created. A–H pending.

A: versioned modes, F3 source chain and complete parameter-only handoff implemented. A_contract: 10 passed, exit 0; baseline mandatory shared: 20 passed, exit 0. Both CPU, no skips. Formal sources remain NOT_RUN. Self-review: explicit fields/path ownership, completed ledger, source freeze and fresh optimizer checked.

B: original Collector/ActorPolicy/Driver/Replay, teacher-free account snapshots and shared native trace construction implemented. B_shadow_archive: 21 passed, exit 0, no skips (2 native SHADOW + 19 WP03 Reader). 256 synthetic market minutes, SHORT TP/SL and official day boundary; model/slots unchanged and original reconciler PASS. Earlier fixture/reset failures retained. Native immutable chunks archived before Replay eviction.

C: original native L1/L2 publication, post-step proof projection, episode-safe live Replay windows and shared multi-Corpus/history mixing implemented. C_data_retry: 3 passed, exit 0. Both strict ADD and WAIT published while Actor executed only WAIT; pending stayed invalid, no optimizer exposure claimed. Source/owner/label attacks and RNG/group restore covered. Initial combined run had fixture Dreamer path error; broad cold cases will be rerun on stable final sources and are not counted as PASS here.

D: explicit SHADOW rejects optimizer calls; ONLINE_SUPERVISED delegates the original F3 train/features/objective and native optimizer. D_supervised_candidate: 3 passed, exit 0. Four actual canonical online L2 updates changed only quality_actor and its slots/clocks; both ADD/WAIT exposed, all other owners byte-identical. No-label inherited slots freeze and overflowing finite-input candidate rollback passed. Initial invalid-input test hit the existing Encoder rejection (XLA callback), then was replaced with the intended candidate-overflow regression; Guard unchanged.

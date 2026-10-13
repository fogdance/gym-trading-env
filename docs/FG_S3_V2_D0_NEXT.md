# S3-V2 after D0: minimum closed loop and decisions still required

Date: 2026-10-13. Stop after D0. This is the requested next-stage outline,
not an authorized training launch or the final V2 implementation taskbook.

## Minimum MVP: A2 first

1. Promote the D0 Oracle query/sidecar contract into a shared source resolver.
   Preserve historical P4 provenance and DAgger executed-action independence.
   Resolve publication of the pending row before enabling interleaved sampling.
2. Reuse the native Driver collector and online Replay, with explicit real
   environment advances and independent model-update/BC-update counters. Add a
   checkpointed beta scheduler; do not choose its schedule from D0 trade reward.
3. Implement WM+equal Oracle BC through the current Agent's extension points.
   Encoder/RSSM/heads continue learning from real executed trajectories; BC
   trains the original Actor from current detached repfeat and real mask. Prior
   Gate does not suppress BC. Disable RL Actor/Critic and RL state updates for A2.
4. Restore the complete run through the original FullCheckpoint with online
   source ownership, sidecar transactions, Oracle/scheduler identity, RNG and
   Prefetch. Verify interrupted vs continuous training before any long run.
5. Only a separately authorized independent DEV Actor-only evaluation can
   establish learned trading behavior. Oracle/scripted TRAIN entry counts,
   reward and loss do not establish Actor success.

## Incremental B2 and C2

- B2 reuses A2 data/interaction/admission. D0 Q values are null. Reliable
  state-conditioned action-value differences need additional exact/interval
  certification before weighted BC is implemented; no midpoint-based weights.
- C2 reuses the same pipeline. WM+BC continue while Gate is closed. Once the
  unchanged Single3 0.30/20-report requirements pass, add original imagination
  RL with one shared optimizer update. Keep BC's true mask separate from the
  predicted imagination mask.
- Fair comparisons require the same environment-step budget, source/data
  identity and declared sampling/mixing policy. Official 1M means new actual
  advances, excluding resets/no-advance events, with valid market minutes also
  reported. Existing source update counters remain model progress, not the new
  environment budget.

## D0 limits to resolve before the full taskbook

- The original certificate covers flat accounts at cash>=300000. Real loss
  histories outside this domain are skipped even when the real mask permits
  ADD. Decide how to prove a broader funding domain or add a bounded exact
  fallback after inspecting measured coverage. Do not lower the floor without
  proof or paste old P4 labels onto losing accounts.
- FG201 is a single all-SHORT TRAIN contract. Multi-contract/direction handling,
  real Gap accounts, contract switching and source-owner lifecycle need further
  validation. Original-Env fixtures cover edge semantics, not all market histories.
- Full state/prefix hashing and equality checks in D0 are intentionally direct.
  Per-query cost grows with account history. Design a verified incremental
  state identity and benchmark late-contract segments before extrapolating 1M
  time or declaring the existing prototype suitable for production throughput.
- GPU Actor+WM+BC/RL smoke, peak memory, Replay retention/prefetch throughput,
  data disk budget and actual Gate calibration are NOT_RUN in D0. Use the
  selected immutable pair and new run directories for later admission.
- Freeze final loss-phase controls, beta schedule, replay demonstration mix,
  training ratio, retention and acceptance thresholds only in the new complete
  V2 taskbook. Do not expand this prototype into an unreviewed 1M trainer.

No official A2/B2/C2, GPU, DEV or TEST execution is authorized by this document.

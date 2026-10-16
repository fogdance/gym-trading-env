# WP02 V2 implementation contract

Five ordered stages A–E; each repository commits independently. Reuse WP01 sibling worktrees on new WP02 branches. Preserve WP01 branches and all historical runs.

A: COURSE_V2 + native E Observation V2 and causal partial MACD. B: actual Encoder and native Replay + same-version checkpoint. C: P0–P3 and isolated L1. D: native Quality TP strict suffix, L2, P4. E: bounded Corpus CLI, authenticated real sample, coverage and review.

MACD contract is exposed by observation_contract and the pure course, with exact equality checked in preflight. EMA initializes from the first completed close, signal from zero DIF. Readiness requires 35 completed buckets before the current bucket; completing bucket 35 remains unready, bucket 36 is ready. Current partial never commits. Session boundaries reset buckets, not EMA history; contracts reset everything. Missing/duplicate/out-of-order/late bars fail closed. Raw diagnostics remain HOST only.

Policy fields: market_seq_e_shared [60,18], market_context_15m [24,7], market_macd_5m [4], position_state [9], action_mask [2], all float32. Audit market_seq is not an Encoder input. No teacher/posthoc fields enter these arrays. Decoder targets remain shared minute features and position state.

L1 Quality Score is the existing 0–100 native outcome score, independent of MAE. Underwater counts executable closing quotes only while still holding at minute end, excluding the settled exit bar. L2 compares full native Quality TP discounted feedback at gamma 332/333, WAIT on exact ties; unsupported/censored states stay null and invalid. Q/Prediction/Decision research permissions are not granted by labels.

Real sample: FG201 last two official days of MAIN QUALIFICATION, not delivery expiry. Earlier trusted TRAIN history is allowed for causal warmup. End must be the authentic main-roll boundary; no step budget becomes a terminal. All tests use CPU, no GPU/model production training, TEST, W&B or all-contract production.

Every stage records argv, cwd, environment, raw log, JUnit and exit under artifacts/fg_s3_v2/e_quality_wp02/evidence. The final review distinguishes G1/G2/G3 and NOT_RUN.

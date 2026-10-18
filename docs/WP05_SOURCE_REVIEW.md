# WP05 source / reverse review

Self-review against task R01–R16; this is a source/behavior review, not an external human certification.

| Risk | Disposition and evidence |
|---|---|
| R01 course hash | Original course_contract.py byte-identical; independent execution_contract subset. A01, F_protected_source_audit.json. |
| R02 F1 not trained | Actual source is controlled WP04 synthetic, always FOUNDATION_NOT_TRAINED; real read consumes no F1. B04, F02/F05. |
| R03 stage transfer | Complete non-opt strict keys/shape/dtype/bytes, fresh original optimizer/counters, original sharding and mirror sync; no historical load API. B01–03. |
| R04 explicit routing | quality_f2_protocol and QUALITY_HEADS separate from F1 and legacy HEADS_ONLY. A02/A04, F1/legacy regression. |
| R05 allowlist | Original masked optimizer, stage-specific positive four-head set intersected with dynamic activation. C09 injected WM rule rejected. |
| R06 exact freezing | All inactive parameters/slots restored from original values including -0; all-zero valid restores clocks and n_updates. C01/02/04, E03. |
| R07 Q NaN | F2 calls prediction_terms with four predictions only, no Q/rank/Soft/Actor math. Invalid targets sanitized before CE/BCE/Huber/mono; type-invalid categories rejected at adapter. C07, D03/D05. |
| R08 score math | Original sigmoid score scale, Huber delta .1 and >=80 auxiliary BCE; no MAE target penalty. Shared original function and old tests. |
| R09 mono | Only comparable first five market-minute windows, both endpoints valid. Official-day endpoints never silently joined into that order. C07/08. |
| R10 denominators | Full source vs sampled exposure vs unique cursor/account rows separated. Natural evaluation guard; insufficient support/null classes explicit. D01–03, real read report. |
| R11 prediction gate | Prediction/Q/Decision flags and fusion OFF, Actor/Critic/Q/WM state frozen. A02, C/E byte evidence, original guards. |
| R12 F3 role | Reader actor views retained, no F3/online code or admission changed. WP06 remains independent of prediction admission. |
| R13 resources | No routine CPU/memory/GPU polling; independent CPU groups concurrent. Only GPU initialization failure triggered device inspection. Host CUDA still failed; no system changes. |
| R14 DATA | No new production batch or DATA branch integrated/claimed; old finite FG201 only. |
| R15 resume | Original FullCheckpoint/RuntimeState/Prefetch; batch snapshot axis advances on skip. Cold process exactness and no-repeat-handoff tested. |
| R16 honesty | CPU CODE_GO, LIMITED READ_GO, TRAIN/RESEARCH NO_GO; GPU ATTEMPTED_FAILED_ENV; full G3 and research positives NOT_RUN. |

Final provenance review additionally tightened FORMAL F1 acceptance: actual complete update ledger, initial complete node, full state schema/finite and WM parameter change required. Engineering fake weights remain explicitly non-admitted. F_formal_records_complete tests this validation without calling it a real trained F1 positive.

Reviewed changed boundaries: Gym source/config/window adapter and read-only CLI; Dreamer explicit train route, shared prediction_terms, detached original causal context, original masked optimizer; original Agent host counter commit; thin HeadsRun and original checkpoint/log reconciliation. Default legacy log axis stays unchanged. FullCheckpoint implementation, native Env/Reward/Score/MACD/Encoder/RSSM/WM/course bytes remain unchanged.

Remaining limits: no formal trained source, independent holdout, native FORMAL G3 in this WP or successful GPU model update. Short-window 15m realized prediction equality was not declared a broken encoder; long native window and intermediate diagnostics prove the connection without changing model mathematics.

Completion audit addendum: actual P1/K90/T8 source had two HEADS_VALID windows but all labels only in burn-in. Before fix, both sampled target segments had zero valid labels; run budget could never advance. F2 now checks existence of enabled target loss in every explicitly requested sampling component using the original immutable index/conditions, without changing the sampler or RNG. Unsupported prepare emits no ready receipt; unsupported Run constructs no model. F_target_support_before/fix/branches/admission and unchanged native read reports provide evidence. Individual zero-valid batches still skip and checkpoint normally.

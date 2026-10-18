# WP06 progress

- Independent paired integration worktrees created; Gym synchronized to the newer DATA B01 audit commit c1679a7. Dreamer remains 0e39dfd. Existing WP04/WP05 merge preserved.
- G0 V3-only source/test commit: 7ea9556. Explicit receipt/database version errors; no legacy fallback. Directed 4 PASS, exit 0.
- G0 contract/Reader/MACD/market-clock: 69 PASS, exit 0. Raw records: receipts/wp05/G0_WP06_contract_reader.*.
- G0 current real TRAIN read: exit 2, corpus_manifest missing at the earlier B01 path. Source relocation requested; not counted as PASS. Earlier planning-time read remains historical evidence only.
- G0 F1/F2 cold restore/handoff and required shared Guards: running. WP06 A–F not started; public integration not advanced.

G0 complete: 126 distinct CPU cases PASS, exit 0; real TRAIN DEV 4-minute read PASS/LIMITED, zero updates. Full earlier B01 corpus missing: read exit 2 / NOT_RUN, not formal admission. Baseline published Gym 7ea95563166bf7fa03c6ce836e7484053cac0b96 / Dreamer 0e39dfd498f9146e10f966cfe118a24b20e5143e. A implementation and directed tests underway.

06.A: explicit F3 protocol/config/complete native model and immutable course execution subset implemented. Stage mixing, legacy CONTROLLED_FULL, Q/Soft/prediction/fusion rejected. A_contract: 32 PASS, exit 0 (includes existing F2 contract regression). F3 update route deliberately fail-closed until 06.C; no real TRAIN update.

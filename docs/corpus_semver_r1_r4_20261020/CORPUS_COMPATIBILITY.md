# Corpus ABI 3.1.0

The ABI describes an asset's mathematical and Reader contract. It is not an
approval, signature, model version, producer Git version or same-Run recovery
identity. The only supported version is exactly `3.1.0`; `3.1.1` is not implied.

`CorpusCompatibility` freezes descriptor fingerprints for the complete native
observation (including shape/order/dtype and causal partial5m MACD V3), all 20
loss-only label fields, Course observation/trading/teacher/reward, E-data feature
math and the post-step/account-contiguous Join contract. It retains the native
Short ADD=3 lots / WAIT, TP5 / SL30, Quality TP Reward, exact gamma=332/333,
tie=WAIT, TRAIN-only complete suffix and future Teacher/Observation separation.
Internal Course V2, Model V2 and trace V2 names are separate version axes.

Old V3 assets have no version field. After the original inventory, source,
Teacher, Course and Observation checks pass, an exact reviewed contract maps to
`3.1.0` with `resolution=REVIEWED_V3_MAPPING`. No file is changed or re-signed.
New Producer manifests declare `course_abi_version=3.1.0` and remain subject to
all original checks. An explicit unknown or null version is rejected.

The registry admits two exact E-profile schema hashes recorded in A0: phase2
`7d82865e…` and phase3/current `365c5dd6…`. These historical hashes include
implementation inventories. Their common semantic descriptor is frozen as
`b0faae74…`; the historical full schema hash and all asset member hashes remain
instance provenance. Only an explicit registry review can add another profile.
The current implementation's descriptor must also match the frozen math. WP01's
standalone preflight retains its original exact profile check; this historical
profile mapping is used only after complete Corpus compatibility resolution.

The source `content_identity` payload is unchanged. Compatibility results live
in consumer receipts, not old assets. A consumer commit or model/Loss change
therefore never changes an old Corpus identity. New execution code still changes
the Run identity and cannot resume an old frozen Run in place. Model/Teacher
changes that violate this ABI require a separately reviewed ABI or asset.

Authentication remains independent: complete six behaviors, immutable member
SHA256, source_lock, official TRAIN eligibility, Teacher identity, actual native
G3, frozen Verifier pin and explicit training authorization are all retained.
DATA-MULTI continues using the WP03 Reader; its additional exact model-contract
mixing guard is preserved.

Evidence: A1 targeted tests: 28 passed, exit 0. Both completed real V3 assets
passed new-consumer development Admission without Generate. Native G3 and
formal training were NOT_RUN in this check.

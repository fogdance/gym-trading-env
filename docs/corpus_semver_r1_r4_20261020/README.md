# Corpus SemVer + R1/R4 delivery

- [Implementation review](IMPLEMENTATION_REVIEW.md): outcome, exact tested pair,
  original-asset proof, limits and follow-up.
- [Source baseline](SOURCE_BASELINE.md): actual integration baseline, protection
  inventory, identity layers and observed defects.
- [Corpus compatibility](CORPUS_COMPATIBILITY.md): exact 3.1.0 contract and
  read-only V3 mapping.
- [API / command templates](INTERFACE.md): F1/F2/F3 pins and prepare/start/resume.
- [Recovery SOP](RECOVERY_SOP.md): pending/orphan/damaged-node handling.
- [Test report](TEST_REPORT.md): exact commands, exit codes and case counts.
- [Requirements matrix](REQUIREMENTS_MATRIX.md): A0–Z1 and fault coverage.

Raw evidence is under `docs/receipts/corpus_semver_r1_r4_20261020/`, including
the final command/index JSON, both repository code patches, before/after Corpus
hashes, historical profile comparison, cold-worker receipts and independent
Reader reports. The release checksum lock covers the published evidence files.
Derived SQLite Join indexes remain in the validation worktrees' artifacts;
their hashes and exact rebuild commands are recorded without adding database
copies to the code delivery.

This is a code delivery on an independent paired branch. Formal real-data G3
and real-model training acceptance remain separate from CODE_GO. Existing
production and frozen Runs are not upgraded or re-signed by this patch.

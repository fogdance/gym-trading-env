# R2 native admission review

Status: R2_CODE_GO, selected CPU checks passed; production Native G3 NOT_RUN.

The unified CLI now accepts and forwards the explicit verifier path, reviewed
Producer Pin and external SHA256 for online prepare/start/resume. Preparation
authenticates the verifier without executing G3, freezes ordered source content
identities and stable verifier bindings, and grants no training permission.

Every formal start/resume validates the saved identities before creating Run or
Agent. It executes the original F3 native preflight and each historical Corpus's
original WP03 Admission with all native arguments. This also authenticates formal
SHADOW sources, without permitting gradients. Each invocation has a unique,
exclusive log outside Prepared/Run/Corpus. Failure preserves the attempt and
prevents Run creation. Same-pin multi-source and complete per-source pin maps
are both supported; ambiguous overrides are rejected.

Fresh source evidence has an internal execution registration. Loading a JSON
PASS or reconstructing the public dataclass does not grant constructor admission.
The stable bindings enter Run identity; changing a verifier requires a new
Prepared/Run. The current resource-aware native verifier is reused unchanged.

## Public interface

`prepare` and `start_prepared` accept `producer_gym`, `producer_lock`,
`producer_lock_sha256`. Optional `source_verifiers` in online configuration maps
`F3_ANCHOR` and every historical `corpus_id` to those exact three fields. Supply
either the complete map or common CLI fields. Prepared preserves source order
in a list; dictionary JSON key sorting does not alter that contract.

## Verification scope

`R2_VERIFIED_20261021T034255536124Z`: 40 selected tests PASS, exit 0 (2.96 s).
These cover real CLI dispatch, original pin validation using two temporary Git
repositories, dirty/digest/missing/second-source rejection, anchor/source identity
and order, unique logs, source argument forwarding, and legacy F1/F2/F3 routes.
Native subprocess results in routing tests are controlled stubs and are explicitly
not G3 evidence. Real trained F3/native G3 integration remains NOT_RUN.

The first run had XML case evidence but empty stdout after the CLI's existing
LD_LIBRARY_PATH process restart. The runner now removes that variable before
pytest, and the verified rerun retains both XML and raw stdout. All receipts
remain preserved; no formal source was modified.

`R2_SHARED_20261021T034307544219Z`: 23 selected shared Gate, SourceTransfer,
Formal Guards and native checkpoint core tests PASS, exit 0 (173.00 s).
Together these are 63 selected tests, not a complete project or formal F4 run.
Self-review checked exclusive logs, source order after sorted JSON, complete
source maps and the pre-Agent admission boundary. No financial implementation,
Reader sampler, Loss or shared FullCheckpoint code was changed.

# R8 collection prefix and final account evidence

Status: code under targeted verification; this is not formal F4 or G3 admission.

OnlineRun V2 / OnlineGates V2 retain the existing native Replay, financial Env,
OnlineHistoryStream and RuntimeState. New `account_verification` checkpoint state
records bounded audit attempts and finalization. Gate evidence is created only
by the original native account reconciliation, not a caller-supplied PASS dict.
Every restored external trace/receipt is rehashed before Agent restore.

Evidence binds Run/source/verifier, market-step endpoint, logical account/row
prefix, native array/event/trace hashes and the actual reconciliation receipt.
Appending collection invalidates current readiness but retains the old prefix.
Only the first needed prefix and final full trace are reconciled by default.

The native Collector fills the previous last decision's `decision_executed` and
`decision_valid_minute` flags when a prefix is extended. A completed development
checkpoint showed exactly those two changes at row 16 between 17- and 33-row
traces. Prefix hashing therefore seals that last decision with both pending flags
false; action, mask, reward, account fields, and all preceding rows remain hashed.
This changes no native financial/decision semantics. Final full reconciliation
still checks every actual row and the true final account.

Audit gzip headers use fixed mtime and an empty header filename so continuous and
cold-restored copies of the same native trace have identical bytes. This applies
only to new online audit exports; no Corpus file or format is rewritten.

Final publication order: complete native collection/publication -> full account
reconciliation -> immutable evidence -> NEW transaction with final evidence and
protected FullCheckpoint -> completion receipt binding its manifest digest.
Failure saves a non-final NO_GO state. A commit-before-receipt interruption can
publish the same receipt from the validated committed node. Invocation results
are separate immutable files, never an overwritten online_result.json.

The first four-test run failed because the new tests used the Corpus signed-JSON
reader on FullCheckpoint's own manifest format. Assertions now use the checkpoint
format. Original failure stdout/XML are retained. Frozen-code validation results
and exact SHAs will be recorded before claiming R8_CODE_GO.

SHADOW remains zero updates. RL with zero actual RL updates cannot publish an RL
completion or protected final node. R3 adds the earlier explicit pause boundary.
Online account evidence is never six-behavior Native G3 evidence.

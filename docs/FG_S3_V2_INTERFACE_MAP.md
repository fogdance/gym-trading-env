# S3-V2 D0 actual interface map

Date: 2026-10-13. This maps the CPU D0 prototype and the minimum changes still
needed for a V2 training implementation. It does not declare A2/B2/C2 complete.

## Reused interfaces and D0 extensions

| Module | Actual interface | D0 use / remaining boundary |
| --- | --- | --- |
| Gym `fg_single3_oracle` | `CandidateIndex(env)`, `funds_certificate(env)`, `optimize(candidates,end,candidate,gamma,*,max_states,return_suffix=False)` | Same candidates and exact comparisons. Opt-in suffix has `first[t]`, `value_intervals[t]`; original default return is unchanged. `first[t]==t` means ADD now. Original schedule membership does not answer arbitrary suffix queries. |
| Gym `fg_s3_oracle_state_query` | `load_train_query(env_receipt,oracle_receipt,contract)`; `OracleStateQuery.query(env,action_mask,*,snapshot=None)` | Signed E/O/candidate provenance, local Env byte checks, TRAIN whitelist, market equality and snapshot equality. Returns action/valid/certified/reason, proof/state/Oracle hashes; Q fields null. Original funding floor is 300000, not the smaller envelope bound. |
| Gym `FixedDirectionEntryEnv` | `reset`, `step`, `export_state`, `import_state`; existing `load_env` and `execute_observed` | Original three-lot financial settlement, next-bar rechecks, reward and pauses. No new trading engine. |
| Dreamer `MixedExecutionPolicy` | `(carry,obs)->(carry,executed_actions,outs)`; `replace_prevact(carry,actions)` | Original Driver policy signature. Actor sees only original seven observation fields. Teacher results remain in a host mailbox. Real public Agent carry is a four-tuple of trees with per-leaf lists produced by `_split`, not a list of per-env four-tuples. |
| Original `Driver` | `Driver([factory],parallel=False)`, `on_step`, `save`, `load` | Driver saves the mixed executed action. Recorder finalizes the preceding decision using the subsequent actual Env result. Rejected ADD remains action=1 in the previous-action input. |
| Original `Replay` | `add`, `sample`, `save`; `save_replay/load_replay` from `fg_checkpoint` | Native uint8 stepid = 16 UUID bytes + 4 big-endian offset bytes. Online insertion, actual fields, exact collection snapshot; no teacher fields in physical native chunks. |
| Gym `D0Collector` | `advance(cumulative_target<=1000)`, `finish`, `save_collection/load_collection` | TRAIN-only bounded collector. Collection state restores Driver, Replay, RNG, record list and pending decision. Its protocol explicitly says `model_present=False`; it is not a transferable Dreamer checkpoint. |
| Gym `join_d0_teacher` | `join_d0_teacher(batch,records,*,fold='TRAIN',catalog=None)` | New rows authenticate observation hash/action/owner. Historical rows delegate to the unchanged original `attach_teacher`. Sampling reset annotation is handled explicitly. Teacher labels may differ from executed actions only for verified DAgger records. |
| Original streams | `Stateless`, `Consec(length=64,consec=1,prefix=1)`, `Prefetch(amount=1)` | D0 model interface probe samples actual new Replay and connects a batch containing historical P4 and new DAgger sequences. Each sequence keeps its own account chronology; accounts are not spliced in time. |
| SourceAdmission / transfer | `inspect_source('direct_full',...)`, `make_agent`, `apply_and_audit_transfer` | Optional actual Online180k CPU transfer. Whole existing model/optimizer/counters are admitted. Same original networks; explicit threshold 0.30 transfer. No independent BC optimizer is imported. |

## Time and record contract

| Representation | Observation | Action / target | Feedback |
| --- | --- | --- | --- |
| Native Driver/Replay row t | `s_t`, reward from the preceding advance, current real mask | outgoing `executed_action_t` | Native row t+1 carries `reward_(t+1)` |
| D0 sidecar row t | pre-action observation/state hash, cursor/time and flags | Oracle `a*_t` plus Actor proposal and executed action | After execution: reward, fills/rejections/exits and next-state hash |

Sidecar protocol: `FG_S3_V2_DAGGER_D0_V1`. All taskbook fields are present;
additional fields include native stepid, `source_type=dagger`, certified/proof
hashes, mixing draw, observation reward, equity and teacher sentinels.
`teacher_valid` requires both a certified learnable query and an actual advance.
The budget-end outgoing row is retained but has invalid supervision and no next
state. Forced WAIT is never a voluntary-entry BC target.

The D0 joint input uses int32 `teacher_action` and bool `teacher_valid`, both
B×T; label/Q fields are joined only for TRAIN. With 64 effective steps and one
recomputed prefix, native batch length is 65 and learning targets start at index
1. Replay actions remain the actually executed controls. Teacher actions must
never substitute for `prevact`. Future market/Oracle hashes are sidecar metadata,
not observation or Actor carry fields.

## Confirmed V1 wiring obstacles and minimum V2 changes

1. V1 Catalog owners are frozen and unknown new UUIDs are rejected. V1 P4
   invariants also require label==replay action. V2 needs a shared resolver with
   separate authenticated P4 and DAgger source contracts; preserve V1 files and
   invariants. The D0 composite join proves the two representations can coexist.
2. `S3CAgent.s3c_loss` multiplies BC by `avail_actor_gate`. V2 BC must be active
   on certified real-mask decisions regardless of Prior Gate. Add a V2 loss
   adapter at `training_loss_fn`; do not alter V1 algorithm behavior.
3. Original `Agent.loss` computes imagination and updates RL normalizers even
   when Gate multiplies policy/value losses by zero; `Agent.train` also updates
   slow value. A2/B2 need a WM+BC route that disables RL computation/state
   updates and protects Critic parameters/optimizer state, rather than assuming
   zero loss alone freezes an inherited optimizer. C2 uses WM+BC continuously
   and adds original RL only after the common Gate. One joint optimizer call
   per update; BC uses stop-gradient RSSM repfeat and changes the original Actor.
4. D0 teacher records are finalized one advance after Replay insertion. Live
   training must register provisional invalid rows before sampling, then commit
   executed transitions/teacher state atomically, or delay publication. The D0
   bridge runs after collection; it does not prove a concurrent training race
   is solved.
5. The collection checkpoint deliberately lacks Agent and Prefetch state.
   V2 FullCheckpoint must register Agent, original Driver/Env, online Replay,
   protected sidecar owner/version/pending records, Oracle identity, beta
   scheduler, RNG, streams/Prefetch/carry and distinct environment/update counters.
6. Raw `Replay.sample()` cannot be passed directly to the public `Agent.train()`:
   it needs Consec annotations and the device/seed preparation performed by
   `Agent.stream()`. The prototype initially observed `KeyError: seed`; the
   corrected CPU probe uses the original stream, without fabricating RNG fields.

V2 Gate retains the shared Single3 20-report protocol, threshold 0.30 and
existing recall/support/regression rules. Gate governs imagination RL only;
BC/WM and RL updates need separate counters. D0 does not report that Gate has
passed 20 real production windows.

# Quality Teacher V2 / Quality Student Full — implementation evidence

**2026-10-15 user amendment:** the Quality-first ZIP supersedes the earlier
score/activation semantics below. Current target is B(t,d), MAE is separate,
and auxiliary-only training precedes Actor influence. This file retains the
initial D0 audit; current evidence and limitations are recorded in
[FG_S3_QUALITY_FIRST_IMPLEMENTATION.md](FG_S3_QUALITY_FIRST_IMPLEMENTATION.md).

This is the archived initial D0 audit. Reward, score and activation defaults
below describe that initial review; the amendment above and the current
Quality-first implementation record define the final behavior.

## D0 source audit (2026-10-15)

| Repository | Audited integration HEAD | Status |
| --- | --- | --- |
| Gym | `c3a808437ef277ee87f25b0ae8803e31fa5dca74` | clean |
| Dreamer | `eb63de710fd693fad7ac76f797f3d9f39e331b98` | clean |

Development uses the paired `fg-s3-quality-v2-dev` worktrees. Historical runs
and frozen training worktrees are read-only inputs.

Inputs:

- Reviewed V1 SHA256: `8021fdb87d958d0d159cbef4d5449c7d55ae3104ab887bb225ba8cb2e33f041a`.
- Reward patch SHA256: `aedca2301f0733c9a1f94f5c0c7725b1d03fbd3227ae034f0c05bfb2a43c8525`.
- User amendment: strict teaching and Q_C1 in the new experiment optimize
  `FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1`. They do not reuse old C1
  labels. Q_V2 retains the Reviewed V1 dimensionless trade utility.
- Reward beta is 0.5; collection beta is 1.0. They are separate parameters.

Observed code/runtime:

- `OracleStateQuery.query` compares actual equity with the certificate's
  `retained_paths_cash_lower_bound`; the observed floor is 300000.
- Native three-lot SL with fees changed equity to 298182, with action mask
  `[1,1]`. The old query returned `FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE` and
  null Q values. Funding certification is not the Env's legal ADD threshold.
- `MarketClock.held` includes the fill bar. `reward_exponent` gives exponent
  zero to the first real feedback. Metadata rows consume no market minute.
- Real FG201 data maps Friday 2021-07-23 night bars to official trading date
  2021-07-26. The source calendar is user-confirmed observed market history;
  fees are user-configured constants, not historical exchange fee schedules.
- Existing candidate generation constructs legacy scorers explicitly. The
  Reward patch alone cannot certify a new strict teacher.
- Baseline targeted CPU checks: 24 passed (funding, market clock, Tutor
  Natural/Balanced/Pair, E2). These are not new-feature acceptance.

Frozen initialization candidate:

- Tutor Full `FORMAL_COMPLETE`, 1,000,000 real advances.
- Node: `/data/logdir/fg_single3/s3_v2/TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/ckpt/step_000001000000`.
- Manifest SHA256: `d6b03a8f2e3604885a7550c9bd217471e290c7840f91dd1ecc93bf93658e9098`;
  agrees with the training receipt; `done` exists. Full bundle admission is
  separate from this manifest check.
- Frozen defaults: BC=1, alpha=.8, pair=.1, tau=1, E2 kappa=1, wmax=2,
  collection beta=1, length=64, prefix=1; policy 3x512, unimix=.05.

## Acceptance status

At the end of this archived D0 audit, D1–D7 were IN_PROGRESS; see the current
Quality-first receipt index for completed implementation and CPU verification.
GPU smoke / pilots / formal training: NOT_RUN and require
separate authorization. Calibration: NOT_CALIBRATED until a legitimate fixed
TRAIN-only calibration split is supplied. E encoder is not part of this
legacy_1m implementation acceptance.

"""TRAIN-only numeric evidence. Clipping is reported, not tuned on DEV/TEST."""
from __future__ import annotations

import numpy as np

from .schema import LEGACY_FIELDS, RAW_FIELDS


def distribution(values, clip=None):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite diagnostic values")
    if not len(a):
        return dict(n=0)
    result = dict(n=int(len(a)), min=float(a.min()), p01=float(np.quantile(a, .01)),
                  p50=float(np.quantile(a, .5)), p99=float(np.quantile(a, .99)), max=float(a.max()))
    if clip is not None:
        result.update(clip=list(clip), clipped_count=int(((a < clip[0]) | (a > clip[1])).sum()),
                      clip_saturation_rate=float(((a < clip[0]) | (a > clip[1])).mean()))
    return result


def bucket_diagnostics(data):
    complete = [x for x in data.buckets if x["complete"]]
    ready = [x for x in complete if x["feature_ready"]]
    # Rejected future empty slots after the last source timestamp are not
    # historical gaps and must not inflate the real-data rejection count.
    end = data.frame.index[-1].isoformat()
    rejected = [x for x in data.buckets if not x["complete"] and x["bucket_end"] <= end]
    vol = [x["vol_relative_raw"] for x in complete if x["vol_relative_raw"] is not None]
    oi = [x["oi_change_raw"] for x in complete if x["oi_change_raw"] is not None]
    jumps = [dict(bucket_id=x["bucket_id"], previous_OI=x["previous_OI"],
                  OpenInterest=x["OpenInterest"], oi_change_raw=x["oi_change_raw"])
             for x in complete if x["oi_change_raw"] is not None and abs(x["oi_change_raw"]) > .1]
    f = data.frame.loc[data.frame._legal]
    summary = dict(source_contract=data.contract, rows=len(data.frame),
        valid_minute_rows=int(data.frame._legal.sum()), opening_marks=int(data.frame._duration.eq(0).sum()),
        complete_buckets=len(complete), feature_ready_buckets=len(ready),
        warmup_buckets=len(complete)-len(ready), warmup_rate=(len(complete)-len(ready))/len(complete) if complete else 1.0,
        rejected_buckets=len(rejected), rejected_examples=rejected[:5],
        volume_zero_minutes=int(f.Volume.eq(0).sum()), OI_nonpositive_minutes=int(f.OpenInterest.le(0).sum()),
        minute_price=distribution(f[list(RAW_FIELDS[:4])].to_numpy()),
        minute_volume=distribution(f.Volume), minute_OI=distribution(f.OpenInterest),
        volume_logratio=distribution(vol, data.config.volume_clip),
        oi_logratio=distribution(oi, data.config.oi_clip), OI_jumps=jumps,
        minute_rejections=[x for x in data.events if x["type"] == "MINUTE_REJECTED"])
    return summary, dict(volume=vol, oi=oi)


def shadow_diagnostics(bundle):
    delta = bundle.market_seq_e_shared.astype(np.float64) - bundle.market_seq.astype(np.float64)
    valid = bundle.market_seq[:, :, LEGACY_FIELDS.index("obs_mask_t")] == 1
    result = []
    for i, name in enumerate(LEGACY_FIELDS):
        values = delta[:, :, i][valid]
        result.append(dict(channel=name, changed_values=int(np.count_nonzero(values)),
                           absolute_delta=distribution(np.abs(values))))
    context_valid = bundle.context_15m[:, :, 6] == 1
    prices = bundle.context_15m[:, :, :4][context_valid]
    return dict(decisions=len(bundle.decision_meta), valid_minute_elements=int(valid.sum()),
                channels=result, context_price_logratio=distribution(prices),
                context_feature_ready_rows=int(context_valid.sum()))

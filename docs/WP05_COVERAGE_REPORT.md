# WP05 coverage report

Actual command F_real_read, exit 0; raw: receipts/wp05/F_real_read_02/read_check.json. Fresh controlled diagnostic weights, no F1 weights consumed, zero real updates and all 349 state hashes unchanged. Native TRAIN content/Join/Reader checks executed; native FORMAL G3 in WP05 NOT_RUN.

| Denominator | Actual |
|---|---:|
| Immutable physical rows / executed account decisions | 4146 / 4134 |
| Unique market cursors / accounts | 690 / 6 |
| HEADS_VALID NATURAL windows | 12 |
| Target exposures / burn-in / prefixes | 768 / 384 / 12 |
| Unique target market cursors | 165 |
| Unique target account physical rows | 764 |
| Actual real optimizer exposure | 0 |

Target physical rows are not all legal ADD decisions. Six accounts do not create six independent markets. Full source vs selected-window denominators remain separate.

Full projected source valid/invalid horizon elements:

```json
{
  "event": {
    "valid": 10801,
    "invalid": 18221
  },
  "touch": {
    "valid": 12344,
    "invalid": 20824
  },
  "adverse": {
    "valid": 6172,
    "invalid": 10412
  },
  "score": {
    "valid": 1430,
    "invalid": 2716
  }
}
```

Selected-window valid horizon exposures (zero optimization):

```json
{
  "event": 2296,
  "touch": 2624,
  "adverse": 1312,
  "score": 328
}
```

Event TP/SL/OTHER/OPEN by 5/15/60/120/240 market minutes, official-day end, next official-day end:

```json
[
  [
    288,
    0,
    10,
    1245
  ],
  [
    646,
    5,
    24,
    868
  ],
  [
    1033,
    49,
    36,
    425
  ],
  [
    1250,
    100,
    39,
    154
  ],
  [
    1299,
    131,
    107,
    6
  ],
  [
    1143,
    113,
    113,
    174
  ],
  [
    1299,
    131,
    113,
    0
  ]
]
```

Touch/adverse positive-negative and score>=80 counts, per-policy losses/calibration/support and class-null AUC/PR are in the raw report. Source Score valid 1430. Missing class/independent split support is UNSUPPORTED, never a perfect metric. Natural probabilities here are conditional on HEADS_VALID eligible windows, not unconditional market-entry calibration.

Corpus `640865b4239fb483e5d98daf18d3104e670ff9b3c1486cb63ec9924cf6e089a9`; course `14cadcdaaa7d96bc0f33f491ef38741494957ee8e7423a1ba5bf598fe230baba`; diagnostic model `3762233aa0902b089a2570d2df7114bfab3fe215eba4a791c19738f42bac2e5b`. The course is unchanged. No new DATA batches or independent research result claimed.

"""Review figures of V2 positive events (ts_v2_positive): 1m candles with the
known prior high H, its entry zone, the SL, the activation, every positive
ADD with its TP and exit, and the volume and RV panels on the real clock.

Colors: three categorical slots that pass the all-pairs CVD check on the
light surface (blue = the level H, orange = the positive entries, aqua =
the activation); candles, volume and the SL are ink. Every colored mark has
a direct text label, so identity never rests on color alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
LEVEL, ENTRY, ACTIVATION = '#2a78d6', '#eb6834', '#1baf7a'
CANDLE_UP, CANDLE_DOWN, VOLUME = '#898781', '#52514e', '#c3c2b7'
# Entry rows printed under a figure; the rest are in positive_entries.csv.
TABLE_ROWS = 30
# Inches: the plot panels, the title band, one table row.
PANELS, TITLE, ROW = 7.6, 0.67, 0.128


def plot_events(folder, event_ids, tables, markets, cfg):
    """One PNG per event id; the list of (event_id, file)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    events = tables['positive_events'].set_index('event_id')
    out = []
    for k, event_id in enumerate(event_ids, 1):
        x = events.loc[event_id].copy()
        x['event_id'] = event_id
        entries = tables['positive_entries'][tables['positive_entries'].event_id == event_id]
        others = tables['candidates'][(tables['candidates'].event_id == event_id)
                                      & (tables['candidates'].reason != 'POSITIVE')]
        timeline = tables['positive_timeline'][tables['positive_timeline'].event_id == event_id]
        stamp = pd.Timestamp(x['activation_time']).strftime('%Y%m%d_%H%M')
        name = f'{k:02d}_{x["contract"].split(".")[-1]}_{stamp}.png'
        plot_event(folder / name, x, entries, others, timeline, markets[x['contract']], cfg)
        out.append(dict(event_id=event_id, file=name))
    return out


def plot_event(path, x, entries, others, timeline, market, cfg):
    e = cfg['entry']
    H, act = float(x['H']), int(x['activation'])
    sl = H + e['stop_above_points']
    first = int(timeline.cursor.min()) if len(timeline) else act
    last = int(np.nanmax(entries.label_exit_cursor)) if len(entries) else act
    lo, hi = max(0, min(first, act - 30) - 5), min(len(market) - 1, max(last, act + 30) + 10)
    lo, hi = max(lo, act - 150), min(hi, act + 420)
    m = market.iloc[lo:hi + 1]
    xs = np.arange(lo, hi + 1)
    rows = min(len(entries), TABLE_ROWS) + (len(entries) > TABLE_ROWS)
    height = PANELS + 2 * TITLE + ROW * rows
    fig, (ax, axv, axr) = plt.subplots(3, 1, figsize=(15, height), sharex=True, facecolor=SURFACE,
                                       gridspec_kw=dict(height_ratios=[4.2, 1.1, 1.1], hspace=0.06))
    for a in (ax, axv, axr):
        a.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            a.spines[side].set_color(VOLUME)
        a.tick_params(colors=INK2, labelsize=8)
        a.grid(axis='y', color=GRID, lw=0.6)
        a.set_axisbelow(True)

    # 15m bucket edges (bars whose open starts a quarter hour).
    bobs = pd.to_datetime(m.bob)
    for b in xs[(bobs.dt.minute % 15 == 0).to_numpy()]:
        for a in (ax, axv, axr):
            a.axvline(b - 0.5, color=GRID, lw=0.6, zorder=0)

    # The pre-activation WAIT span of the teaching timeline.
    waits = timeline[timeline.teach.str.startswith('WAIT')]
    if len(waits):
        ax.axvspan(int(waits.cursor.min()) - 0.5, act - 0.5, color='#f0efec', zorder=0)
        ax.text(act - 1, 0.975, 'WAIT (taught)', color=INK2, fontsize=8, va='top', ha='right',
                transform=ax.get_xaxis_transform())

    # Candles: hollow up, filled down; wick from low to high.
    o, h, l, c = (m[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    ax.vlines(xs, l, h, color=CANDLE_UP, lw=0.7, zorder=2)
    up = c >= o
    ax.bar(xs[up], np.maximum(c - o, 0.2)[up], 0.62, o[up], facecolor=SURFACE,
           edgecolor=CANDLE_UP, lw=0.7, zorder=3)
    ax.bar(xs[~up], np.maximum(o - c, 0.2)[~up], 0.62, c[~up], color=CANDLE_DOWN, zorder=3)

    # The level, its zone and the structural stop.
    ax.axhspan(H - e['zone_points'], H, color=LEVEL, alpha=0.10, lw=0, zorder=1)
    ax.axhline(H, color=LEVEL, lw=1.6, zorder=4)
    ax.axhline(sl, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=4)
    # Level labels sit in the right margin, outside the plot.
    margin = ax.get_yaxis_transform()
    ax.text(1.005, H, f'H {H:g}', color=INK, fontsize=9, va='center', transform=margin)
    ax.text(1.005, H - e['zone_points'] / 2, f'zone H-{e["zone_points"]}..H', color=INK2, fontsize=8,
            va='center', transform=margin)
    ax.text(1.005, sl, f'SL H+{e["stop_above_points"]} = {sl:g}', color=INK, fontsize=9, va='center',
            transform=margin)
    confirm = int(x['level_confirm'])
    if lo <= confirm <= hi:
        ax.plot([confirm], [H], marker='|', ms=14, mew=2, color=LEVEL, zorder=5)
        ax.annotate('H known', (confirm, H), xytext=(0, 10), textcoords='offset points',
                    color=INK2, fontsize=8, ha='center')

    # Activation: the first RV >= 2 near H.
    for a in (ax, axv, axr):
        a.axvline(act, color=ACTIVATION, lw=1.4, zorder=4)
    ax.text(act + 1, 0.975, f'activation RV {x["activation_rv"]:.1f}', color=INK, fontsize=8.5, va='top',
            transform=ax.get_xaxis_transform())

    # Candidates not in the library (audit only), then the positive ADDs.
    if len(others):
        f = others[others.label_fill_in_zone]
        ax.scatter(f.label_fill_cursor, f.label_fill_expected, s=30, facecolor='none',
                   edgecolor=MUTED, lw=1, zorder=5, label='other candidates (audit only)')
    lines = []
    letters = entries.entry_id.str.rsplit(':', n=1).str[1].tolist()
    for letter, (_, r) in zip(letters, entries.iterrows()):
        fx, fy = int(r.label_fill_cursor), float(r.label_fill_expected)
        ex = int(r.label_exit_cursor)
        ax.plot([fx, ex], [r.label_tp, r.label_tp], color=ENTRY, lw=1.1, ls=(0, (2, 2)), zorder=5)
        ax.scatter([fx], [fy], marker='v', s=70, color=ENTRY, edgecolor=SURFACE, lw=1.2, zorder=6)
        ax.scatter([ex], [r.label_exit_price], marker='o', s=40, facecolor=SURFACE, edgecolor=ENTRY,
                   lw=1.6, zorder=6)
        gap = ' (segment-open exit)' if r.get('label_exit_at_segment_open', False) else ''
        lines.append(
            f'{letter:<2}  {r.label_grade:<9}  decide {pd.Timestamp(r.time).strftime("%m-%d %H:%M")}  '
            f'fill {fy:g} @ {pd.Timestamp(r.label_fill_time).strftime("%H:%M")}  RV {r.obs_rv:4.1f}  '
            f'R {r.label_R:g}  TP {r.label_tp:g}  exit {pd.Timestamp(r.label_exit_time).strftime("%m-%d %H:%M")}  '
            f'held {int(r.label_held_minutes)} min  net {r.label_net_pnl:+.0f} ({r.label_net_R:+.2f}R)  '
            f'MAE {r.label_mae_points:g} ({r.label_mae_R:.2f}R)  MFE {r.label_mfe_points:g}{gap}')
    # Letters of fills within a few bars share one label ("B-D"); TP labels
    # share one per exit bar and TP price.
    fills = entries.label_fill_cursor.astype(int).tolist()
    groups, start = [], 0
    for i in range(1, len(fills) + 1):
        if i == len(fills) or fills[i] - fills[i - 1] > 3:
            groups.append(range(start, i))
            start = i
    for g in groups:
        name = letters[g[0]] if len(g) == 1 else f'{letters[g[0]]}-{letters[g[-1]]}'
        top = max(float(entries.label_fill_expected.iloc[i]) for i in g)
        ax.annotate(name, ((fills[g[0]] + fills[g[-1]]) / 2, top), xytext=(0, 9),
                    textcoords='offset points', ha='center', color=INK, fontsize=9, fontweight='bold')
    exits = {}
    for letter, (_, r) in zip(letters, entries.iterrows()):
        exits.setdefault((int(r.label_exit_cursor), float(r.label_tp)), []).append(letter)
    for (ex, tp), names in exits.items():
        names = names if len(names) <= 6 else names[:5] + [f'+{len(names) - 5}']
        ax.annotate(f'TP {",".join(names)}', (ex, tp), xytext=(7, 0), textcoords='offset points',
                    color=INK2, fontsize=7.5, va='center',
                    bbox=dict(boxstyle='square,pad=0.15', fc=SURFACE, ec='none', alpha=0.9))
    ax.scatter([], [], marker='v', s=70, color=ENTRY, label='positive ADD fill (A, B, ...)')
    ax.scatter([], [], marker='o', s=40, facecolor=SURFACE, edgecolor=ENTRY, lw=1.6, label='its TP exit')
    ax.legend(loc='lower left', fontsize=8, frameon=False, labelcolor=INK2)
    ax.set_ylabel('price', color=INK2, fontsize=9)
    ymin = min(np.nanmin(l), np.nanmin(entries.label_tp) if len(entries) else H) - 3
    ax.set_ylim(ymin, max(np.nanmax(h), sl) + 6)

    # Volume and RV (separate panels, one axis each).
    vol = m.volume.to_numpy()
    colors = np.where(xs == act, ACTIVATION, VOLUME)
    axv.bar(xs, vol, 0.7, color=colors)
    axv.set_ylabel('volume', color=INK2, fontsize=9)
    rv = m.rv.to_numpy()
    axr.plot(xs, rv, color=INK2, lw=1.2)
    axr.axhline(cfg['volume']['ratio'], color=MUTED, lw=1, ls=(0, (4, 3)))
    axr.text(1.005, cfg['volume']['ratio'], f'RV {cfg["volume"]["ratio"]:g}', color=INK2, fontsize=8,
             va='center', transform=axr.get_yaxis_transform())
    if lo <= act <= hi and np.isfinite(rv[act - lo]):
        axr.scatter([act], [rv[act - lo]], s=36, color=ACTIVATION, edgecolor=SURFACE, lw=1.2, zorder=5)
    axr.set_ylabel('RV', color=INK2, fontsize=9)
    axr.set_ylim(0, max(np.nanmax(rv) if np.isfinite(rv).any() else 1, cfg['volume']['ratio']) * 1.1)

    # Real clock on the bar axis: a date wherever the trading date changes.
    times = pd.to_datetime(m.time)
    ticks = np.linspace(0, len(xs) - 1, min(12, len(xs))).round().astype(int)
    labels, previous = [], None
    for i in ticks:
        t = times.iloc[i]
        labels.append(t.strftime('%H:%M') if t.date() == previous else t.strftime('%m-%d\n%H:%M'))
        previous = t.date()
    axr.set_xticks(xs[ticks])
    axr.set_xticklabels(labels)
    axr.set_xlim(lo - 1, hi + 8)

    fig.suptitle(f'{x["contract"]}  event {x["event_id"]}  ·  H {H:g}  ·  {int(x["positives"])} positive ADD '
                 f'({int(x["excellent"])} EXCELLENT)  ·  direction SHORT, {x["direction_source"]}',
                 x=0.06, ha='left', y=1 - 0.14 / height, color=INK, fontsize=11.5)
    fig.text(0.06, 1 - 0.53 / height, f'activation {x["activation_time"][:16]}  ·  event end {x["end_time"][:16]} '
             f'({x["end_reason"]})  ·  1m bars, fill = next open - spread; entries are alternatives, '
             'never summed', color=INK2, fontsize=8.5)
    if len(lines) > TABLE_ROWS:
        lines = lines[:TABLE_ROWS] + [f'... {len(lines) - TABLE_ROWS} more in teacher/positive_entries.csv']
    fig.text(0.06, 0.05 / height, '\n'.join(lines), color=INK, fontsize=7.6, family='monospace', va='bottom')
    fig.subplots_adjust(left=0.06, right=0.9, top=1 - TITLE / height, bottom=(TITLE + ROW * rows) / height)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)

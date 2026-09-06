"""Look-ahead-free screener and the stitched single-tape representation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import Market


def rank_daily(market: Market) -> np.ndarray:
    """Daily market-cap competition rank of every stock (1 = largest)."""
    cap = market.cap
    n, T = cap.shape
    rank = np.empty((n, T), int)
    order = np.argsort(-cap, axis=0)
    rank[order, np.arange(T)[None, :]] = np.arange(1, n + 1)[:, None]
    return rank


def find_windows(market: Market, buffer: int = 2, pre_cutoff: int = 10,
                 fail_rank: int = 100, post_effective: int = 1,
                 max_window: int = 25) -> list[dict]:
    """Candidate windows ``[{'stock', 'start', 'end', 'kind'}, ...]``.

    At most one window is active at any time and entries are decided on
    Fridays only. Two setups, in priority order: *announced* additions
    (between announcement and effective day) and *predicted* additions (a
    non-member ranked <= add_rank - buffer within ``pre_cutoff`` sessions of a
    review cutoff). Only information available on the decision day is used.
    """
    p = market.params
    T, n = market.n_bars, market.n_stocks
    cap = market.cap
    ranks = rank_daily(market)
    reviews = market.reviews
    windows: list[dict] = []
    busy_until = -1
    for t in np.flatnonzero(market.calendar.weekday == 4):
        if t <= busy_until:
            continue
        mem, stock = market.members_at(t), None
        rv = next((r for r in reviews if r['announce'] <= t < r['effective']), None)
        if rv:
            stock, kind = max(rv['adds'], key=lambda i: cap[i, t], default=None), 'announced'
        else:
            rv = next((r for r in reviews if r['cutoff'] >= t), None)
            if rv and rv['cutoff'] - t <= pre_cutoff:
                cands = [i for i in range(n)
                         if i not in mem and ranks[i, t] <= p.add_rank - buffer]
                stock, kind = min(cands, key=lambda i: ranks[i, t], default=None), 'predicted'
        if stock is None:
            continue
        end = min(rv['effective'] + post_effective, t + max_window, T - 3)
        if kind == 'predicted':
            if stock not in rv['adds']:
                end = min(end, rv['announce'] + 1)
            failed = next((u for u in range(t + 1, min(rv['cutoff'], end))
                           if ranks[stock, u] > fail_rank), None)
            if failed:
                end = min(end, failed)
        windows.append({'stock': stock, 'start': int(t), 'end': int(end), 'kind': kind})
        busy_until = end + 2
    return windows


def build_tape(market: Market, windows: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    """Splice the candidates' OHLCV bars into one continuous tape.

    Each segment is rescaled by a constant (returns unchanged) and every window
    ends at least two sessions before its segment does, so no trade ever spans
    a splice. Returns the tape and a per-bar segment id.
    """
    T = market.n_bars
    parts, seg_id, prev_end = [], np.zeros(T, int), -1
    if not windows:
        raise ValueError('no candidate windows')
    for k, w in enumerate(windows):
        lo = prev_end + 1
        hi = T - 1 if k == len(windows) - 1 else min(w['end'] + 2, T - 1)
        s = w['stock']
        scale = 100 / market.close[s, w['start']]
        part = pd.DataFrame({'Open': market.open[s, lo:hi + 1] * scale,
                             'High': market.high[s, lo:hi + 1] * scale,
                             'Low': market.low[s, lo:hi + 1] * scale,
                             'Close': market.close[s, lo:hi + 1] * scale,
                             'Volume': market.volume[s, lo:hi + 1] / scale,
                             'Signal': 0, 'Window': -1},
                            index=market.calendar[lo:hi + 1])
        part.iloc[w['start'] - lo:w['end'] - lo, part.columns.get_loc('Signal')] = 1
        part.iloc[w['start'] - lo:w['end'] - lo, part.columns.get_loc('Window')] = k
        seg_id[lo:hi + 1] = k
        parts.append(part)
        prev_end = hi
    tape = pd.concat(parts)
    assert tape.index.equals(market.calendar)
    return tape, seg_id

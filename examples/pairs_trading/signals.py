"""Z-score signal generation for a spread series.

The z-score measures how stretched the spread is relative to its own trailing
history; the trading rule is the literature-standard hysteresis loop
(enter beyond ``entry``, exit on reversion through ``exit``, hard stop beyond
``stop``, optional time stop). Signals are *targets decided on the close of
bar t*; the engine applies the execution lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SignalConfig

#: Exit reasons recorded in the trades ledger ('flip' and 'bust' are added by
#: the engine: direct reversals and equity wipeouts respectively).
REASONS = ('converged', 'stopped', 'time', 'end', 'flip', 'bust')


def zscore(spread: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with a strictly trailing window (no look-ahead)."""
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std(ddof=1)
    return (spread - mean) / std


def zscore_frozen(spread: pd.Series, mean: float, std: float) -> pd.Series:
    """Z-score against a frozen (formation-period) mean/std, GGR-style."""
    return (spread - mean) / std


def generate_signals(z: pd.Series, cfg: SignalConfig) -> pd.DataFrame:
    """Run the hysteresis state machine over a z-score series.

    Returns a DataFrame indexed like ``z`` with columns:

    - ``side``: target spread position decided at that bar's close
      (+1 long spread, -1 short spread, 0 flat)
    - ``reason``: exit reason on bars where a close decision is made
      ('converged' | 'stopped' | 'time'), else ''

    Rules (for a long-spread position; short is the mirror image):

    - enter when ``z <= -entry``
    - exit when ``z >= -exit``   (spread reverted through the exit band)
    - stop when ``z <= -stop``   (divergence blowout — likely structural break)
    - time-stop after ``max_holding`` bars in position

    NaN z-scores never trigger entries; an existing position is held through
    isolated NaNs (no information, no action). After any exit the machine is
    *disarmed*: it may not re-enter until |z| has first reverted inside the
    entry band. Without this hysteresis a stop-loss would be immediately
    followed by a fresh entry in the same stretched spread, defeating its
    purpose as a structural-break guard.
    """
    zv = z.to_numpy(dtype=float)
    n = len(zv)
    side = np.zeros(n, dtype=np.int8)
    reason = np.full(n, '', dtype=object)
    state = 0
    held = 0
    armed = True
    for t in range(n):
        zt = zv[t]
        if np.isnan(zt):
            side[t] = state
            held += state != 0
            continue
        if state == 0:
            if abs(zt) < cfg.entry:
                armed = True
            # Enter only inside the [entry, stop) band: a z-score already
            # beyond the stop is treated as a structural break, not a signal.
            if armed and -cfg.stop < zt <= -cfg.entry:
                state, held = 1, 0
            elif armed and cfg.stop > zt >= cfg.entry:
                state, held = -1, 0
        elif state == 1:
            held += 1
            if zt <= -cfg.stop:
                state, reason[t], armed = 0, 'stopped', False
            elif zt >= -cfg.exit:
                state, reason[t], armed = 0, 'converged', abs(zt) < cfg.entry
            elif cfg.max_holding is not None and held >= cfg.max_holding:
                state, reason[t], armed = 0, 'time', abs(zt) < cfg.entry
        else:  # state == -1
            held += 1
            if zt >= cfg.stop:
                state, reason[t], armed = 0, 'stopped', False
            elif zt <= cfg.exit:
                state, reason[t], armed = 0, 'converged', abs(zt) < cfg.entry
            elif cfg.max_holding is not None and held >= cfg.max_holding:
                state, reason[t], armed = 0, 'time', abs(zt) < cfg.entry
        side[t] = state
    return pd.DataFrame({'side': side, 'reason': reason}, index=z.index)

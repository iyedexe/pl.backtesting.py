"""Universe-wide cointegration screening.

Given a wide close-price panel, enumerate candidate pairs, prefilter by return
correlation (cheap), then run the Engle-Granger machinery (expensive) on the
survivors. Screening a universe of N assets tests N(N-1)/2 hypotheses, so at a
5% level dozens of "cointegrated" pairs appear by chance alone (Clegg 2014);
the report treats in-sample screens as *descriptive* and relies on the
walk-forward harness for any performance claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import engle_granger

SCREEN_COLUMNS = ['y', 'x', 'corr', 'eg_pvalue', 'eg_tstat', 'beta',
                  'half_life', 'hurst', 'n_obs']


def screen_panel(panel: pd.DataFrame,
                 corr_min: float = 0.7,
                 min_obs: int = 500,
                 max_pairs_tested: int | None = None) -> pd.DataFrame:
    """Screen all pairs in a wide close panel; returns rows sorted by EG p-value.

    ``y``/``x`` record the orientation chosen by the two-sided Engle-Granger
    test (regressand/regressor with the lower MacKinnon p-value).
    """
    panel = panel.where(panel > 0)  # log-prices need positivity (negative WTI day)
    logp = np.log(panel.astype(float))
    rets = logp.diff()
    corr = rets.corr(min_periods=max(60, min_obs // 4))
    cols = list(panel.columns)
    counts = panel.notna().sum()

    candidates: list[tuple[str, str, float]] = []
    for i, ci in enumerate(cols):
        if counts[ci] < min_obs:
            continue
        for cj in cols[i + 1:]:
            if counts[cj] < min_obs:
                continue
            c = corr.at[ci, cj]
            if pd.notna(c) and c >= corr_min:
                candidates.append((ci, cj, float(c)))
    candidates.sort(key=lambda t: -t[2])
    if max_pairs_tested is not None:
        candidates = candidates[:max_pairs_tested]

    rows = []
    for ci, cj, c in candidates:
        both = panel[[ci, cj]].dropna()
        if len(both) < min_obs:
            continue
        try:
            eg = engle_granger(both[ci], both[cj], names=(ci, cj))
        except Exception:
            continue
        rows.append({'y': eg.y, 'x': eg.x, 'corr': c, 'eg_pvalue': eg.pvalue,
                     'eg_tstat': eg.tstat, 'beta': eg.beta,
                     'half_life': eg.half_life, 'hurst': eg.hurst,
                     'n_obs': eg.n_obs})
    df = pd.DataFrame(rows, columns=SCREEN_COLUMNS)
    df = df.astype({c: float for c in SCREEN_COLUMNS
                    if c not in ('y', 'x', 'n_obs')} | {'n_obs': int})
    return df.sort_values('eg_pvalue').reset_index(drop=True)


def expected_false_positives(n_assets: int, level: float = 0.05) -> float:
    """Pairs expected to pass an EG test at ``level`` by pure chance."""
    n_pairs = n_assets * (n_assets - 1) / 2
    return n_pairs * level

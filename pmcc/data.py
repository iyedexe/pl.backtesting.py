"""
Data access for the PMCC backtest.

The bundled dataset (see ``pmcc/tools/build_dataset_from_qrmdata.py`` for full
provenance) contains **dividend- and split-adjusted** daily closes for 17
French large caps and the CAC 40 index, 2000-01-03 .. 2015-12-31, anchored so
that the *last* observation equals the raw close of that day.  An adjusted
(total-return-style) series is what we want for return dynamics, but option
strikes and moneyness live in *raw* price space, so
:func:`reconstruct_raw_prices` converts back to an approximate raw series
using a constant per-stock dividend yield:

    P_raw(t) = P_adj(t) * exp(q * (T_anchor - t))

which by construction matches the true raw close at the anchor date and makes
the raw series under-perform the adjusted one by exactly the dividend yield
``q`` -- the continuous-dividend approximation used consistently across the
whole backtest (option pricing uses the same ``q``).

Also provides an EUR short-rate curve (annual EURIBOR-3M averages,
interpolated) and an empirical implied/realised volatility premium factor
estimated from VIX vs S&P 500 realised volatility.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@dataclass(frozen=True)
class StockMeta:
    ticker: str
    name: str
    sector: str
    div_yield: float        # long-run average dividend yield (approximation)


#: Universe metadata. ``div_yield`` values are long-run (2000-2015) average
#: dividend yields, encoded as documented approximations (the bundled price
#: data contains no dividend records).  The backtest exposes a sensitivity
#: sweep on this parameter.
UNIVERSE: dict[str, StockMeta] = {m.ticker: m for m in [
    StockMeta('AI.PA',  'Air Liquide',       'Industrial gases',  0.026),
    StockMeta('AIR.PA', 'Airbus',            'Aerospace',         0.013),
    StockMeta('BN.PA',  'Danone',            'Food & beverage',   0.032),
    StockMeta('BNP.PA', 'BNP Paribas',       'Banking',           0.039),
    StockMeta('CA.PA',  'Carrefour',         'Retail',            0.032),
    StockMeta('CS.PA',  'AXA',               'Insurance',         0.046),
    StockMeta('DG.PA',  'Vinci',             'Construction',      0.039),
    StockMeta('FP.PA',  'TotalEnergies',     'Energy',            0.053),
    StockMeta('GLE.PA', 'Societe Generale',  'Banking',           0.034),
    StockMeta('MC.PA',  'LVMH',              'Luxury goods',      0.019),
    StockMeta('OR.PA',  "L'Oreal",           'Cosmetics',         0.016),
    StockMeta('ORA.PA', 'Orange',            'Telecom',           0.058),
    StockMeta('SAF.PA', 'Safran',            'Aerospace',         0.017),
    StockMeta('SAN.PA', 'Sanofi',            'Pharmaceuticals',   0.035),
    StockMeta('SGO.PA', 'Saint-Gobain',      'Building materials', 0.032),
    StockMeta('SU.PA',  'Schneider Electric', 'Electrical equipment', 0.029),
    StockMeta('VIV.PA', 'Vivendi',           'Media',             0.042),
]}

#: The ten names with the historically deepest Euronext (MONEP) option markets;
#: headline results focus on these.
LIQUID_OPTIONS = ['FP.PA', 'MC.PA', 'SAN.PA', 'BNP.PA', 'OR.PA',
                  'AI.PA', 'SU.PA', 'AIR.PA', 'CS.PA', 'DG.PA']

# EURIBOR 3M annual averages (percent), used as the risk-free/cash rate.
_EUR_RATES = {
    2000: 4.40, 2001: 4.26, 2002: 3.32, 2003: 2.33, 2004: 2.11, 2005: 2.19,
    2006: 3.08, 2007: 4.28, 2008: 4.64, 2009: 1.23, 2010: 0.81, 2011: 1.39,
    2012: 0.58, 2013: 0.22, 2014: 0.21, 2015: -0.02, 2016: -0.26,
}


_CACHE: dict = {}


def load_adjusted_prices() -> pd.DataFrame:
    """Daily adjusted closes for the universe + '^FCHI', 2000..2015.

    Cached; treat the result as read-only.
    """
    if 'px' not in _CACHE:
        path = os.path.join(DATA_DIR, 'prices_adj_fr_2000_2015.csv.gz')
        px = pd.read_csv(path, index_col=0, parse_dates=True)
        px.index.name = 'Date'
        _CACHE['px'] = px
    return _CACHE['px']


def reconstruct_raw_prices(adj: pd.Series, div_yield: float) -> pd.Series:
    """Approximate raw price path from an end-anchored adjusted series."""
    s = adj.dropna()
    t_anchor = s.index[-1]
    years_back = (t_anchor - s.index).days / 365.25
    return s * np.exp(div_yield * years_back)


def eur_short_rate(dates: pd.DatetimeIndex) -> pd.Series:
    """EUR short rate (decimal p.a.) on ``dates``, interpolated across years."""
    knots_x = [pd.Timestamp(f'{y}-07-01').value for y in sorted(_EUR_RATES)]
    knots_y = [_EUR_RATES[y] / 100 for y in sorted(_EUR_RATES)]
    vals = np.interp(dates.asi8.astype('float64'),
                     np.array(knots_x, dtype='float64'), knots_y)
    return pd.Series(vals, index=dates, name='r')


def load_vix_sp500() -> pd.DataFrame:
    if 'vix' not in _CACHE:
        path = os.path.join(DATA_DIR, 'vix_sp500_1990_2015.csv.gz')
        _CACHE['vix'] = pd.read_csv(path, index_col=0, parse_dates=True)
    return _CACHE['vix']


def implied_premium_factor(dates: pd.DatetimeIndex,
                           lam: float = 0.94,
                           smooth: int = 21,
                           clip: tuple[float, float] = (0.85, 1.60)) -> pd.Series:
    """
    Empirical implied/realised volatility ratio, estimated as
    ``VIX / realised_vol(S&P 500)`` (EWMA, lambda=0.94, annualised), smoothed
    and clipped.  This borrows the *shape* of the US volatility risk premium
    through time and applies it to French realised vols, for lack of a freely
    available long implied-vol history for the CAC universe.
    """
    vs = load_vix_sp500().dropna()
    ret = np.log(vs['SP500']).diff()
    ewvar = ret.pow(2).ewm(alpha=1 - lam).mean()
    rv = np.sqrt(ewvar * 252)
    ratio = (vs['VIX'] / 100.0) / rv.replace(0, np.nan)
    ratio = ratio.rolling(smooth, min_periods=5).mean().clip(*clip)
    return ratio.reindex(dates).ffill().bfill().rename('iv_premium')

"""Loaders for the vendored daily panels in ``research/data/vendored/``.

Every loader returns a wide ``pd.DataFrame`` indexed by a naive ``DatetimeIndex``
with one column per instrument, prices quoted in USD. See
``examples/research/data/SOURCES.md`` for provenance, licenses and caveats.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd

_EXAMPLES_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = _EXAMPLES_ROOT / 'research' / 'data' / 'vendored'

_PANEL_FILES = {
    'crypto': 'crypto_usd_daily.csv.gz',
    'forex': 'fx_usd_daily.csv.gz',
    'commodities': 'commodities_daily.csv.gz',
    'stocks': 'sp500_close_daily.csv.gz',
    'stocks_long': 'stocks_1990_2018_daily.csv.gz',
}


def _read(name: str) -> pd.DataFrame:
    path = VENDOR_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Re-create the vendored data with '
            f'scripts/fetch_sources.sh + scripts/build_vendored_data.py '
            f'(see examples/research/data/SOURCES.md).')
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df.index.name = 'date'
    return df


@cache
def load_panel(asset_class: str) -> pd.DataFrame:
    """Load the daily close panel for one asset class (see ``_PANEL_FILES``)."""
    try:
        fname = _PANEL_FILES[asset_class]
    except KeyError:
        raise KeyError(f'Unknown asset class {asset_class!r}; '
                       f'one of {sorted(_PANEL_FILES)}') from None
    return _read(fname)


@cache
def load_sp500_ohlcv() -> pd.DataFrame:
    """Long-format OHLCV bars for the 32 selected S&P 500 tickers."""
    df = pd.read_csv(VENDOR_DIR / 'sp500_ohlcv_selected.csv.gz', parse_dates=['date'])
    return df.set_index('date').sort_index()


def ohlc_for(ticker: str) -> pd.DataFrame:
    """OHLCV frame (backtesting.py column convention) for one selected ticker."""
    df = load_sp500_ohlcv()
    sub = df[df['ticker'] == ticker].drop(columns='ticker')
    if sub.empty:
        raise KeyError(f'{ticker!r} not in the selected OHLCV set')
    sub.columns = [c.capitalize() for c in sub.columns]
    return sub


def get_series(asset_class: str, symbol: str) -> pd.Series:
    """One instrument's close series. ``asset_class='cross'`` searches all panels."""
    if asset_class == 'cross':
        for cls in _PANEL_FILES:
            panel = load_panel(cls)
            if symbol in panel.columns:
                return panel[symbol].dropna()
        raise KeyError(f'{symbol!r} not found in any vendored panel')
    s = load_panel(asset_class)[symbol].dropna()
    return s


def aligned_pair(asset_class: str, a: str, b: str,
                 start=None, end=None) -> pd.DataFrame:
    """Inner-joined close prices for a pair, columns ['a', 'b'].

    Uses an inner join (no forward-filling): a bar exists only where *both*
    instruments printed a price, so daily returns never straddle fake flat days.
    """
    sa, sb = get_series(asset_class, a), get_series(asset_class, b)
    df = pd.concat({'a': sa, 'b': sb}, axis=1, join='inner').dropna()
    # Log-price methods require positive prices. This drops exactly one real
    # observation in the vendored data: WTI's negative settlement on 2020-04-20.
    df = df[(df > 0).all(axis=1)]
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    if len(df) < 100:
        raise ValueError(f'Pair {a}/{b}: only {len(df)} overlapping bars')
    return df

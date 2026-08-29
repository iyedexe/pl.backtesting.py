"""
Refresh / extend the bundled price data with Yahoo! Finance (run this on your
own machine -- most sandboxes block finance sites).

    pip install yfinance
    python -m pmcc.fetch_data                  # extend to today
    python -m pmcc.fetch_data --start 2000-01-01

Writes ``pmcc/data/prices_adj_fr_extended.csv.gz`` with dividend+split
adjusted closes ("Adj Close") for the whole universe plus ^FCHI, and
``dividend_yields_estimated.csv`` with realised trailing yields you can use to
override the constants in ``pmcc/data.py``.

After fetching, point the backtest at the new file by replacing
``prices_adj_fr_2000_2015.csv.gz`` or editing ``pmcc.data.load_adjusted_prices``.
The engine anchors raw-price reconstruction on each series' last date, so an
extended file "just works".
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from .data import DATA_DIR, UNIVERSE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2000-01-01')
    args = ap.parse_args()
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit('pip install yfinance first')

    tickers = list(UNIVERSE) + ['^FCHI']
    print(f'downloading {len(tickers)} tickers from {args.start}...')
    raw = yf.download(tickers, start=args.start, auto_adjust=False,
                      actions=True, group_by='column', threads=True)
    adj = raw['Adj Close'][tickers].dropna(how='all')
    out = os.path.join(DATA_DIR, 'prices_adj_fr_extended.csv.gz')
    adj.round(6).to_csv(out, compression='gzip')
    print('wrote', out, adj.shape)

    # realised trailing-12m dividend yields (to sanity-check pmcc.data constants)
    if 'Dividends' in raw:
        divs = raw['Dividends'].fillna(0.0)
        close = raw['Close']
        y = (divs.rolling(252, min_periods=200).sum() / close).median()
        y.rename('median_trailing_yield').round(4).to_csv(
            os.path.join(DATA_DIR, 'dividend_yields_estimated.csv'))
        print('median trailing dividend yields:')
        print(y.round(3).to_string())


if __name__ == '__main__':
    main()

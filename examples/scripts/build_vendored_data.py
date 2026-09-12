"""Build the compact vendored datasets in ``research/data/vendored/`` from raw sources.

The raw sources are public GitHub repositories (see ``examples/scripts/fetch_sources.sh`` and
``research/data/SOURCES.md``).  This script normalizes them into small, analysis-ready
``.csv.gz`` panels that are committed to the repository so that every experiment in the
project is exactly reproducible offline.

Usage::

    uv run python examples/scripts/build_vendored_data.py --src /path/to/raw/clones

The layout expected under ``--src`` is the one produced by ``examples/scripts/fetch_sources.sh``:

    exchange-rates/   github.com/datasets/exchange-rates   (US Fed H.10 via FRED)
    oil-prices/       github.com/datasets/oil-prices       (US EIA)
    natural-gas/      github.com/datasets/natural-gas      (US EIA, Henry Hub)
    coinmetrics/      github.com/coinmetrics-io/data       (Coin Metrics community data)
    kaggle-code/      github.com/CNuge/kaggle-code         (S&P 500 5-year daily OHLCV)
    ppo/              github.com/robertmartin8/PyPortfolioOpt (20 large caps, 1990-2018)
"""

import argparse
import io
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = EXAMPLES_ROOT / 'research' / 'data' / 'vendored'

#: FRED H.10 country -> ISO currency code. All rates in the datahub mirror are quoted
#: as *national currency units per one U.S. dollar* (verified in tests below).
FX_COUNTRY_TO_CCY = {
    'Australia': 'AUD',
    'Brazil': 'BRL',
    'Canada': 'CAD',
    'China': 'CNY',
    'Denmark': 'DKK',
    'Euro': 'EUR',
    'Hong Kong': 'HKD',
    'India': 'INR',
    'Japan': 'JPY',
    'Malaysia': 'MYR',
    'Mexico': 'MXN',
    'New Zealand': 'NZD',
    'Norway': 'NOK',
    'Singapore': 'SGD',
    'South Africa': 'ZAR',
    'South Korea': 'KRW',
    'Sweden': 'SEK',
    'Switzerland': 'CHF',
    'Taiwan': 'TWD',
    'Thailand': 'THB',
    'United Kingdom': 'GBP',
}

CRYPTO_ASSETS = ['btc', 'eth', 'ltc', 'bch', 'xrp', 'ada', 'doge', 'sol', 'dot',
                 'xmr', 'link', 'etc', 'bnb', 'avax', 'paxg']

#: Tickers whose full OHLCV bars we keep (famous-pair candidates + liquid megacaps),
#: used by the backtesting.py cross-validation adapter.
OHLCV_TICKERS = [
    'KO', 'PEP', 'XOM', 'CVX', 'V', 'MA', 'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C',
    'HD', 'LOW', 'UPS', 'FDX', 'MCD', 'PG', 'CL', 'MMM', 'HON', 'CAT', 'DE',
    'DAL', 'LUV', 'T', 'VZ', 'GOOG', 'GOOGL', 'AAPL', 'MSFT', 'SPGI',
]


def _git_sha(path: Path) -> str:
    try:
        return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def _write(df: pd.DataFrame, name: str, manifest: dict, source: str, src_dir: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, compression='gzip')
    manifest[name] = {
        'source': source,
        'source_commit': _git_sha(src_dir),
        'rows': len(df),
        'columns': list(df.columns),
        'start': str(df.index.min().date()),
        'end': str(df.index.max().date()),
        'bytes': path.stat().st_size,
    }
    print(f'  wrote {name}: {len(df)} rows x {len(df.columns)} cols, '
          f'{df.index.min().date()} -> {df.index.max().date()}, '
          f'{path.stat().st_size / 1024:.0f} KiB')


def build_crypto(src: Path, manifest: dict):
    print('crypto (Coin Metrics community data, PriceUSD):')
    frames = {}
    for asset in CRYPTO_ASSETS:
        f = src / 'coinmetrics' / 'csv' / f'{asset}.csv'
        if not f.exists():
            print(f'  ! missing {f}, skipping')
            continue
        header = pd.read_csv(f, nrows=0).columns
        if 'PriceUSD' not in header:
            print(f'  ! {asset}: no PriceUSD column, skipping')
            continue
        df = pd.read_csv(f, usecols=['time', 'PriceUSD'], parse_dates=['time'])
        s = df.set_index('time')['PriceUSD'].dropna()
        frames[asset.upper()] = s
    panel = pd.DataFrame(frames).sort_index()
    panel.index.name = 'date'
    _write(panel, 'crypto_usd_daily.csv.gz', manifest,
           'github.com/coinmetrics-io/data (CC BY-NC 4.0)', src / 'coinmetrics')


def build_fx(src: Path, manifest: dict):
    print('forex (Fed H.10 via FRED, datahub mirror):')
    df = pd.read_csv(src / 'exchange-rates' / 'data' / 'daily.csv',
                     parse_dates=['Date'])
    df['ccy'] = df['Country'].map(FX_COUNTRY_TO_CCY)
    df = df.dropna(subset=['ccy'])
    wide = df.pivot_table(index='Date', columns='ccy', values='Exchange rate')
    # Sanity-check quoting convention: units of national currency per 1 USD.
    eur = wide['EUR'].dropna()
    jpy = wide['JPY'].dropna()
    assert 0.5 < eur.iloc[-1] < 1.5, 'EUR should be ~0.8-1.1 per USD'
    assert jpy.iloc[-1] > 50, 'JPY should be >50 per USD'
    # Normalize to the *USD price of one unit of foreign currency* (so every series
    # is a tradable "asset price" in USD, like every other panel in this project).
    usd_price = (1.0 / wide).sort_index()
    usd_price.index.name = 'date'
    usd_price.columns.name = None
    _write(usd_price, 'fx_usd_daily.csv.gz', manifest,
           'github.com/datasets/exchange-rates (PDDL-1.0; underlying: FRED H.10)',
           src / 'exchange-rates')


def build_commodities(src: Path, manifest: dict):
    print('commodities (EIA spot prices, datahub mirrors):')

    def read_one(path, name):
        df = pd.read_csv(path, parse_dates=['Date'])
        return df.set_index('Date').iloc[:, 0].rename(name)
    wti = read_one(src / 'oil-prices' / 'data' / 'wti-daily.csv', 'WTI')
    brent = read_one(src / 'oil-prices' / 'data' / 'brent-daily.csv', 'BRENT')
    gas = read_one(src / 'natural-gas' / 'data' / 'daily.csv', 'NATGAS')
    panel = pd.concat([wti, brent, gas], axis=1).sort_index()
    panel.index.name = 'date'
    _write(panel, 'commodities_daily.csv.gz', manifest,
           'github.com/datasets/{oil-prices,natural-gas} (PDDL-1.0; underlying: US EIA)',
           src / 'oil-prices')


def build_sp500(src: Path, manifest: dict):
    print('stocks (S&P 500 constituents, 5y daily OHLCV, Kaggle camnugent/CNuge):')
    zpath = src / 'kaggle-code' / 'stock_data' / 'individual_stocks_5yr.zip'
    closes = {}
    ohlcv_rows = []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith('_data.csv'):
                continue
            ticker = Path(name).stem.replace('_data', '')
            raw = z.read(name)
            try:
                text = raw.decode('utf-8')
            except UnicodeDecodeError:
                text = raw.decode('latin-1')
            try:
                df = pd.read_csv(io.StringIO(text), parse_dates=['date'])
            except Exception as e:
                print(f'  ! {ticker}: unreadable ({e}), skipping')
                continue
            df = df.dropna(subset=['close'])
            if df.empty:
                continue
            closes[ticker] = df.set_index('date')['close']
            if ticker in OHLCV_TICKERS:
                sub = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                sub.insert(0, 'ticker', ticker)
                ohlcv_rows.append(sub)
    close_panel = pd.DataFrame(closes).sort_index()
    close_panel.index.name = 'date'
    _write(close_panel, 'sp500_close_daily.csv.gz', manifest,
           'github.com/CNuge/kaggle-code (Kaggle "S&P 500 stock data", CC0)',
           src / 'kaggle-code')
    ohlcv = pd.concat(ohlcv_rows).set_index('date').sort_index()
    _write(ohlcv, 'sp500_ohlcv_selected.csv.gz', manifest,
           'github.com/CNuge/kaggle-code (Kaggle "S&P 500 stock data", CC0)',
           src / 'kaggle-code')


def build_stocks_long(src: Path, manifest: dict):
    print('stocks long history (PyPortfolioOpt test fixture, 20 large caps):')
    df = pd.read_csv(src / 'ppo' / 'tests' / 'resources' / 'stock_prices.csv',
                     parse_dates=['date'])
    panel = df.set_index('date').sort_index()
    panel.index.name = 'date'
    _write(panel, 'stocks_1990_2018_daily.csv.gz', manifest,
           'github.com/robertmartin8/PyPortfolioOpt tests/resources (MIT)', src / 'ppo')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', required=True, type=Path,
                    help='Directory containing the raw source clones')
    args = ap.parse_args()
    manifest: dict = {}
    build_crypto(args.src, manifest)
    build_fx(args.src, manifest)
    build_commodities(args.src, manifest)
    build_sp500(args.src, manifest)
    build_stocks_long(args.src, manifest)
    meta = {
        'built_utc': datetime.now(UTC).isoformat(timespec='seconds'),
        'files': manifest,
    }
    (OUT_DIR / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'manifest written to {OUT_DIR / "manifest.json"}')


if __name__ == '__main__':
    main()

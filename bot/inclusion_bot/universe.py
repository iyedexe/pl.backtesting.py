"""Data layer: index constituents from Wikipedia, market data from Yahoo.

Design notes
------------
* Wikipedia constituent tables are parsed *tolerantly* (any column whose name
  contains ticker/symbol/epic; any containing sector/industry) so routine page
  edits don't break the bot.
* Market data comes from yfinance ``fast_info`` (price, full market cap,
  exchange) fetched in a small thread pool. Free-float shares (DAX ranks by
  float cap) and quarterly earnings (S&P 500 profitability gate) require the
  heavier ``get_info``/statements endpoints and are fetched only for the few
  symbols that need them.
* Everything network-touching funnels through two functions
  (``fetch_constituents`` and ``fetch_market_data``) that cache to disk per
  day, and the screener consumes plain DataFrames - so tests (and the
  ``selftest`` command) can run fully offline on fixtures.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

WIKI_URL = 'https://en.wikipedia.org/wiki/{}'
USER_AGENT = ('inclusion-bot/0.1 (index-inclusion educational signal bot; '
              'github.com/iyedexe/pl.backtesting.py)')

TICKER_COL = re.compile(r'ticker|symbol|epic', re.I)
SECTOR_COL = re.compile(r'sector|industry', re.I)
NAME_COL = re.compile(r'company|security|name|constituent', re.I)


class DataError(RuntimeError):
    pass


def _cache_path(cache_dir: str, kind: str, key: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', key)
    return os.path.join(cache_dir, f'{kind}-{safe}-{dt.date.today():%Y%m%d}.json')


def _cached(cache_dir: str, kind: str, key: str, producer):
    path = _cache_path(cache_dir, kind, key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    value = producer()
    with open(path, 'w') as f:
        json.dump(value, f)
    return value


def to_yahoo(ticker: str, suffix: str) -> str:
    """Normalize a Wikipedia ticker to a Yahoo symbol.

    US class shares use '-' on Yahoo (BRK.B -> BRK-B); LSE/Xetra symbols get
    their exchange suffix unless the page already includes one.
    """
    t = re.sub(r'\[.*?\]', '', str(ticker)).strip().upper()  # drop footnotes
    if not t or t == 'NAN':
        return ''
    if suffix:
        return t if '.' in t else t + suffix
    return t.replace('.', '-')


def _parse_constituents(html: str) -> list[dict]:
    """Largest table that has a ticker-ish column wins."""
    from io import StringIO
    tables = pd.read_html(StringIO(html))
    best = None
    for tbl in tables:
        cols = [str(c) for c in tbl.columns]
        if any(TICKER_COL.search(c) for c in cols) and (
                best is None or len(tbl) > len(best)):
            best = tbl
    if best is None or len(best) < 10:
        raise DataError('no constituents table found')
    cols = {str(c): c for c in best.columns}
    tick = next(cols[c] for c in cols if TICKER_COL.search(c))
    name = next((cols[c] for c in cols if NAME_COL.search(c)), None)
    sector = next((cols[c] for c in cols if SECTOR_COL.search(c)), None)
    return [{'ticker': str(row[tick]),
             'name': str(row[name]) if name is not None else str(row[tick]),
             'sector': str(row[sector]) if sector is not None else ''}
            for _, row in best.iterrows()]


def fetch_constituents(page: str, cache_dir: str) -> list[dict]:
    """[{'ticker', 'name', 'sector'}, ...] from a Wikipedia list page."""
    def produce():
        import requests
        r = requests.get(WIKI_URL.format(page),
                         headers={'User-Agent': USER_AGENT}, timeout=30)
        r.raise_for_status()
        return _parse_constituents(r.text)

    rows = _cached(cache_dir, 'constituents', page, produce)
    if len(rows) < 10:
        raise DataError(f'suspiciously small constituent list for {page}')
    return rows


def _fast_info_row(symbol: str) -> dict:
    import yfinance as yf
    fi = yf.Ticker(symbol).fast_info

    def get(*keys):
        for k in keys:
            try:
                v = fi[k]
            except (KeyError, TypeError, AttributeError):
                continue
            if v is not None:
                return v
        return None
    return {'symbol': symbol,
            'price': get('lastPrice', 'last_price', 'regularMarketPreviousClose'),
            'cap': get('marketCap', 'market_cap'),
            'exchange': get('exchange') or '',
            'currency': get('currency') or ''}


def fetch_market_data(symbols: list[str], cache_dir: str,
                      workers: int = 8) -> pd.DataFrame:
    """DataFrame indexed by symbol with price, cap, exchange, currency.

    Symbols that fail twice are dropped; the caller decides whether the
    remaining coverage is good enough (`assert_coverage`).
    """
    def produce():
        rows = []

        def task(sym):
            for _ in range(2):
                try:
                    return _fast_info_row(sym)
                except Exception:
                    continue
            return None
        with ThreadPoolExecutor(workers) as pool:
            for row in pool.map(task, symbols):
                if row and row['price'] and row['cap']:
                    rows.append(row)
        return rows

    rows = _cached(cache_dir, 'marketdata', ','.join(sorted(symbols))[:80]
                   + f'-{len(symbols)}', produce)
    df = pd.DataFrame(rows)
    return df.set_index('symbol') if len(df) else df


def assert_coverage(df: pd.DataFrame, symbols: list[str], floor: float = .8):
    if len(df) < floor * len(symbols):
        raise DataError(f'market data coverage too low: {len(df)}/{len(symbols)} '
                        '- refusing to rank on partial data')


def fetch_float_caps(symbols: list[str], cache_dir: str) -> dict[str, float]:
    """floatShares x price for the given symbols (used for DAX ranking)."""
    def produce():
        import yfinance as yf
        out = {}
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                shares = t.get_info().get('floatShares')
                price = _fast_info_row(sym)['price']
                if shares and price:
                    out[sym] = float(shares) * float(price)
            except Exception:
                continue
        return out

    return _cached(cache_dir, 'floatcaps', ','.join(sorted(symbols))[:80]
                   + f'-{len(symbols)}', produce)


def check_profitability(symbol: str) -> bool | None:
    """S&P 500 gate: positive GAAP net income, last quarter AND trailing 4.

    Returns None when statements are unavailable (reported as 'unverified').
    """
    try:
        import yfinance as yf
        inc = yf.Ticker(symbol).quarterly_income_stmt
        ni = inc.loc['Net Income'].dropna().astype(float)
        if len(ni) < 4:
            return None
        latest4 = ni.iloc[:4]
        return bool(latest4.iloc[0] > 0 and latest4.sum() > 0)
    except Exception:
        return None


def load_index_universe(idx, all_indices, cache_dir: str):
    """(members, candidates) as lists of dicts with Yahoo symbols.

    The Nasdaq-100 candidate pool is derived: S&P 500 + S&P 400 constituents
    (their Wikipedia tables carry GICS sectors) minus current NDX members,
    excluding Financials; the Nasdaq-listing requirement is applied later
    from market data (exchange codes), since Wikipedia doesn't list venues.
    """
    members = [dict(r, symbol=to_yahoo(r['ticker'], idx.yahoo_suffix))
               for r in fetch_constituents(idx.members_page, cache_dir)]
    if idx.candidates_page:
        cand_rows = fetch_constituents(idx.candidates_page, cache_dir)
    else:  # derived pool (Nasdaq-100)
        spx = next(i for i in all_indices if i.key == 'SPX500')
        cand_rows = (fetch_constituents(spx.members_page, cache_dir)
                     + fetch_constituents(spx.candidates_page, cache_dir))
    member_syms = {m['symbol'] for m in members}
    candidates, seen = [], set()
    for r in cand_rows:
        sym = to_yahoo(r['ticker'], idx.yahoo_suffix)
        if not sym or sym in member_syms or sym in seen:
            continue
        if idx.exclude_financials and 'financial' in r.get('sector', '').lower():
            continue
        seen.add(sym)
        candidates.append(dict(r, symbol=sym))
    return members, candidates

"""
Price feeds: the bot only needs "what does TICKER trade at right now?".

- `StaticPriceFeed`  : dict you can mutate (tests).
- `CSVPriceFeed`     : OHLC CSV files per ticker, clock-aware (replay).
- `YFinancePriceFeed`: last price via `yfinance` (optional dependency).
- `AlpacaPriceFeed`  : see `newsbot.alpaca`.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Union

import pandas as pd

from .clock import Clock, SystemClock
from .models import to_utc

log = logging.getLogger(__name__)


class PriceFeed(ABC):
    @abstractmethod
    def price(self, ticker: str) -> Optional[float]:
        """Latest tradeable price, or None if unavailable."""


class StaticPriceFeed(PriceFeed):
    def __init__(self, prices: Optional[Mapping[str, float]] = None):
        self.prices: Dict[str, float] = dict(prices or {})

    def price(self, ticker: str) -> Optional[float]:
        return self.prices.get(ticker.upper())

    def set(self, ticker: str, price: float) -> None:
        self.prices[ticker.upper()] = float(price)


def load_ohlc_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load an OHLC(V) CSV (Date index; Open/High/Low/Close[/Volume] columns) in the format backtesting.py expects."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip().capitalize() for c in df.columns]
    df.index.name = 'Date'
    return df.sort_index()


class CSVPriceFeed(PriceFeed):
    """
    Serves the Close of the most recent bar at or before `clock.now()`.
    `frames` maps ticker -> OHLC DataFrame (datetime index, naive = UTC).
    """

    def __init__(self, frames: Mapping[str, pd.DataFrame], clock: Optional[Clock] = None,
                 field: str = 'Close'):
        self.frames = {k.upper(): v for k, v in frames.items()}
        self.clock = clock or SystemClock()
        self.field = field
        self._utc_index = {k: pd.DatetimeIndex([to_utc(t) for t in v.index]) for k, v in self.frames.items()}

    @classmethod
    def from_files(cls, files: Mapping[str, Union[str, Path]], clock: Optional[Clock] = None) -> 'CSVPriceFeed':
        return cls({t: load_ohlc_csv(p) for t, p in files.items()}, clock)

    def bar_at(self, ticker: str, when=None) -> Optional[pd.Series]:
        ticker = ticker.upper()
        df = self.frames.get(ticker)
        if df is None or df.empty:
            return None
        when = to_utc(when or self.clock.now())
        pos = self._utc_index[ticker].searchsorted(when, side='right') - 1
        if pos < 0:
            return None
        return df.iloc[pos]

    def price(self, ticker: str) -> Optional[float]:
        bar = self.bar_at(ticker)
        return None if bar is None else float(bar[self.field])


class YFinancePriceFeed(PriceFeed):
    def __init__(self, cache_seconds: float = 15.0, clock: Optional[Clock] = None):
        import yfinance  # noqa: PLC0415  (optional dependency)
        self._yf = yfinance
        self._cache: Dict[str, tuple] = {}
        self._cache_seconds = cache_seconds
        self._clock = clock or SystemClock()

    def price(self, ticker: str) -> Optional[float]:
        now = self._clock.now().timestamp()
        cached = self._cache.get(ticker)
        if cached and now - cached[0] < self._cache_seconds:
            return cached[1]
        try:
            info = self._yf.Ticker(ticker).fast_info
            px = float(info['last_price'])
        except Exception as e:  # noqa: BLE001
            log.warning('yfinance price for %s failed: %s', ticker, e)
            return None
        self._cache[ticker] = (now, px)
        return px


class CallablePriceFeed(PriceFeed):
    def __init__(self, fn: Callable[[str], Optional[float]]):
        self._fn = fn

    def price(self, ticker: str) -> Optional[float]:
        return self._fn(ticker)

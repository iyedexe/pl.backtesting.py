"""
Broker abstraction plus an in-memory paper broker.

The paper broker fills market orders instantly at the price feed's latest price
(plus optional slippage and commission) and tracks cash/positions so that the
bot's own exit logic can be tested end to end without any external account.
"""
from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from .clock import Clock, SystemClock, is_regular_session
from .models import Fill
from .prices import PriceFeed

log = logging.getLogger(__name__)


class BrokerError(RuntimeError):
    pass


@dataclass
class BrokerPosition:
    ticker: str
    qty: float
    avg_price: float


class Broker(ABC):
    #: True if `buy()` attaches take-profit/stop-loss legs server-side (bracket order).
    manages_exits: bool = False

    @abstractmethod
    def cash(self) -> float:
        ...

    @abstractmethod
    def equity(self) -> float:
        ...

    @abstractmethod
    def buy(self, ticker: str, qty: float, *, take_profit: Optional[float] = None,
            stop_loss: Optional[float] = None, reference_price: Optional[float] = None) -> Fill:
        ...

    @abstractmethod
    def sell(self, ticker: str, qty: float) -> Fill:
        ...

    @abstractmethod
    def positions(self) -> Dict[str, BrokerPosition]:
        ...

    @abstractmethod
    def is_market_open(self, now: Optional[datetime] = None) -> bool:
        ...


class PaperBroker(Broker):
    def __init__(self, price_feed: PriceFeed, cash: float = 100_000.0, *, clock: Optional[Clock] = None,
                 commission: float = 0.0, slippage_bps: float = 0.0, always_open: bool = False):
        self.feed = price_feed
        self.clock = clock or SystemClock()
        self._cash = float(cash)
        self.commission = commission          # per-order fixed cost
        self.slippage = slippage_bps / 10_000
        self.always_open = always_open
        self._positions: Dict[str, BrokerPosition] = {}
        self.fills: list = []

    def cash(self) -> float:
        return self._cash

    def equity(self) -> float:
        value = self._cash
        for p in self._positions.values():
            px = self.feed.price(p.ticker)
            value += p.qty * (px if px is not None else p.avg_price)
        return value

    def positions(self) -> Dict[str, BrokerPosition]:
        return dict(self._positions)

    def is_market_open(self, now: Optional[datetime] = None) -> bool:
        return self.always_open or is_regular_session(now or self.clock.now())

    def _price(self, ticker: str, reference_price: Optional[float]) -> float:
        px = self.feed.price(ticker)
        if px is None:
            px = reference_price
        if px is None or px <= 0:
            raise BrokerError(f'no price available for {ticker}')
        return float(px)

    def buy(self, ticker: str, qty: float, *, take_profit=None, stop_loss=None, reference_price=None) -> Fill:
        ticker = ticker.upper()
        if qty <= 0:
            raise BrokerError('qty must be positive')
        px = self._price(ticker, reference_price) * (1 + self.slippage)
        cost = qty * px + self.commission
        if cost > self._cash + 1e-9:
            raise BrokerError(f'insufficient cash for {qty} {ticker} @ {px:.2f} '
                              f'(need {cost:.2f}, have {self._cash:.2f})')
        self._cash -= cost
        pos = self._positions.get(ticker)
        if pos:
            total = pos.qty + qty
            pos.avg_price = (pos.avg_price * pos.qty + px * qty) / total
            pos.qty = total
        else:
            self._positions[ticker] = BrokerPosition(ticker, qty, px)
        fill = Fill(ticker, qty, px, self.clock.now(), order_id=uuid.uuid4().hex[:10])
        self.fills.append(('buy', fill))
        log.info('PAPER BUY  %s x%g @ %.4f', ticker, qty, px)
        return fill

    def sell(self, ticker: str, qty: float) -> Fill:
        ticker = ticker.upper()
        pos = self._positions.get(ticker)
        if not pos or pos.qty < qty - 1e-9:
            raise BrokerError(f'no/insufficient position in {ticker} to sell {qty}')
        px = self._price(ticker, pos.avg_price) * (1 - self.slippage)
        self._cash += qty * px - self.commission
        pos.qty -= qty
        if pos.qty <= 1e-9:
            del self._positions[ticker]
        fill = Fill(ticker, qty, px, self.clock.now(), order_id=uuid.uuid4().hex[:10])
        self.fills.append(('sell', fill))
        log.info('PAPER SELL %s x%g @ %.4f', ticker, qty, px)
        return fill

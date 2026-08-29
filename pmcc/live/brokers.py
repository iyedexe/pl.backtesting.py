"""
Broker abstraction for live/paper PMCC execution.

Only two implementations ship here:

* :class:`PaperBroker` -- a fully offline simulator that fills orders at the
  same synthetic model prices as the backtest (plus spread).  Use it to dry-run
  the daily executor loop end to end.
* ``pmcc.live.ib_adapter.IBBroker`` -- an **untested skeleton** for
  Interactive Brokers (Euronext options via ib_insync); see that module's
  docstring before even thinking about pointing it at a real account.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from typing import Protocol

from ..pricing import OptionSpec


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass
class OrderResult:
    ok: bool
    filled_qty: int = 0
    avg_price: float = 0.0
    message: str = ''


class Broker(Protocol):
    """What the executor needs from a broker. Option quantities are contracts,
    positive = long. ``qty`` on orders is signed (negative = sell/write)."""

    def spot(self, ticker: str) -> float: ...
    def option_quote(self, ticker: str, spec: OptionSpec) -> Quote: ...
    def option_positions(self, ticker: str) -> dict[OptionSpec, int]: ...
    def cash(self) -> float: ...
    def place_option_order(self, ticker: str, spec: OptionSpec, qty: int,
                           limit: float) -> OrderResult: ...


@dataclass
class PaperBroker:
    """Offline broker: quotes = model mid +/- half-spread, orders always fill
    at the limit if it is inside the quoted spread.  Pass ``state_file`` to
    persist cash/positions across daily invocations."""
    prices: dict[str, float]                     # ticker -> spot
    model_mid: 'callable'                        # (ticker, spec) -> model mid
    half_spread_pct: float = 0.02
    half_spread_min: float = 0.03
    _cash: float = 100_000.0
    _positions: dict = field(default_factory=dict)   # (ticker, spec) -> qty
    mult: int = 100
    state_file: str | None = None

    def __post_init__(self):
        if self.state_file and os.path.exists(self.state_file):
            with open(self.state_file) as f:
                raw = json.load(f)
            self._cash = raw['cash']
            self._positions = {
                (t, OptionSpec(s['strike'],
                               dt.date.fromisoformat(s['expiry']),
                               s['is_call'])): q
                for t, s, q in raw['positions']}

    def _persist(self):
        if not self.state_file:
            return
        raw = {'cash': self._cash,
               'positions': [[t, {'strike': sp.strike,
                                  'expiry': sp.expiry.isoformat(),
                                  'is_call': sp.is_call}, q]
                             for (t, sp), q in self._positions.items() if q]}
        with open(self.state_file, 'w') as f:
            json.dump(raw, f, indent=2)

    def spot(self, ticker: str) -> float:
        return self.prices[ticker]

    def option_quote(self, ticker: str, spec: OptionSpec) -> Quote:
        mid = self.model_mid(ticker, spec)
        half = max(self.half_spread_pct * mid, self.half_spread_min)
        return Quote(bid=max(mid - half, 0.01), ask=mid + half)

    def option_positions(self, ticker: str) -> dict[OptionSpec, int]:
        return {spec: q for (t, spec), q in self._positions.items()
                if t == ticker and q != 0}

    def cash(self) -> float:
        return self._cash

    def place_option_order(self, ticker: str, spec: OptionSpec, qty: int,
                           limit: float) -> OrderResult:
        if qty == 0:
            return OrderResult(ok=False, message='zero qty')
        q = self.option_quote(ticker, spec)
        # marketable limit? buys need limit >= ask, sells limit <= bid
        px = None
        if qty > 0 and limit >= q.ask:
            px = q.ask
        elif qty < 0 and limit <= q.bid:
            px = q.bid
        if px is None:
            return OrderResult(ok=False, message=f'limit {limit} not marketable '
                                                 f'({q.bid}/{q.ask})')
        self._cash -= qty * self.mult * px
        key = (ticker, spec)
        self._positions[key] = self._positions.get(key, 0) + qty
        self._persist()
        return OrderResult(ok=True, filled_qty=qty, avg_price=px)

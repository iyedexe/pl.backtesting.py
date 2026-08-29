"""
Interactive Brokers adapter -- UNTESTED SKELETON.

This module sketches how the :class:`pmcc.live.brokers.Broker` protocol maps
onto Interactive Brokers for Euronext Paris single-stock options.  It has
**never been run against a real gateway** (it was written in an offline
environment) and MUST be validated on an IB *paper* account first.  Expect to
fix contract details (trading class, multiplier, exchange routing) per
underlying before anything fills.

Requirements: ``pip install ib_insync`` and a running TWS/IB Gateway.
Paper TWS listens on port 7497, live on 7496 -- this adapter refuses to
connect to a live port unless ``allow_live=True`` is passed explicitly.

Euronext option specifics to verify per name (use ``reqContractDetails``):
  * ``exchange``: usually 'FTA' legacy names or 'MONEP' for Paris;
  * ``tradingClass``: e.g. 'MC1'/'MC' style per underlying;
  * ``multiplier``: 100 for most Paris stock options, 10 for some;
  * expiries: monthly third Friday; long-dated series exist only for the
    most liquid names -- confirm an 18-24 month expiry actually trades
    before running a PMCC on it.
"""
from __future__ import annotations

import datetime as dt

from ..pricing import OptionSpec
from .brokers import OrderResult, Quote

PAPER_PORTS = {7497, 4002}


class IBBroker:
    """Broker-protocol implementation on top of ib_insync (skeleton)."""

    def __init__(self, host: str = '127.0.0.1', port: int = 7497,
                 client_id: int = 17, allow_live: bool = False,
                 exchange: str = 'MONEP', currency: str = 'EUR'):
        if port not in PAPER_PORTS and not allow_live:
            raise RuntimeError(
                f'port {port} looks like a LIVE gateway; pass allow_live=True '
                'only after the strategy has run for weeks on paper.')
        try:
            from ib_insync import IB  # noqa: WPS433 (optional dependency)
        except ImportError as e:
            raise ImportError('pip install ib_insync to use IBBroker') from e
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id, timeout=10)
        self.exchange = exchange
        self.currency = currency

    # -- contract helpers -------------------------------------------------
    def _stock(self, ticker: str):
        from ib_insync import Stock
        symbol = ticker.replace('.PA', '')
        return Stock(symbol, 'SBF', self.currency)

    def _option(self, ticker: str, spec: OptionSpec):
        from ib_insync import Option
        symbol = ticker.replace('.PA', '')
        expiry = spec.expiry.strftime('%Y%m%d')
        right = 'C' if spec.is_call else 'P'
        opt = Option(symbol, expiry, spec.strike, right, self.exchange,
                     currency=self.currency)
        details = self.ib.reqContractDetails(opt)
        if not details:
            raise LookupError(f'no IB contract for {ticker} {spec}')
        return details[0].contract

    # -- Broker protocol --------------------------------------------------
    def spot(self, ticker: str) -> float:
        [t] = self.ib.reqTickers(self._stock(ticker))
        px = t.marketPrice()
        if not px or px != px:
            raise RuntimeError(f'no market price for {ticker}')
        return float(px)

    def option_quote(self, ticker: str, spec: OptionSpec) -> Quote:
        [t] = self.ib.reqTickers(self._option(ticker, spec))
        if not t.bid or not t.ask or t.bid <= 0 or t.ask <= 0:
            raise RuntimeError(f'no two-sided quote for {spec}')
        return Quote(bid=float(t.bid), ask=float(t.ask))

    def option_positions(self, ticker: str) -> dict[OptionSpec, int]:
        symbol = ticker.replace('.PA', '')
        out: dict[OptionSpec, int] = {}
        for pos in self.ib.positions():
            c = pos.contract
            if c.secType == 'OPT' and c.symbol == symbol:
                spec = OptionSpec(float(c.strike),
                                  dt.datetime.strptime(
                                      c.lastTradeDateOrContractMonth, '%Y%m%d').date(),
                                  c.right == 'C')
                out[spec] = out.get(spec, 0) + int(pos.position)
        return out

    def cash(self) -> float:
        for row in self.ib.accountSummary():
            if row.tag == 'TotalCashValue' and row.currency == self.currency:
                return float(row.value)
        raise RuntimeError('TotalCashValue not found in account summary')

    def place_option_order(self, ticker: str, spec: OptionSpec, qty: int,
                           limit: float) -> OrderResult:
        from ib_insync import LimitOrder
        contract = self._option(ticker, spec)
        action = 'BUY' if qty > 0 else 'SELL'
        order = LimitOrder(action, abs(qty), round(limit, 2), tif='DAY')
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(5)  # give it a moment; real code should poll properly
        status = trade.orderStatus
        return OrderResult(ok=status.status in ('Filled', 'Submitted', 'PreSubmitted'),
                           filled_qty=int(status.filled) * (1 if qty > 0 else -1),
                           avg_price=float(status.avgFillPrice or 0),
                           message=status.status)

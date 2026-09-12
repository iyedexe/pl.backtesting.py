"""
Daily live/paper executor: one invocation = one decision cycle.

Reuses the exact backtest decision cores (:mod:`pmcc.strategy`).  The flow:

1. pull spot + positions from the broker and reconcile with the state file;
2. build a :class:`~pmcc.strategy.MarketView` (volatility surface from recent
   realised vol, or -- with a real broker -- implied from quotes);
3. call ``strategy.decide()``;
4. turn each action into a *marketable limit* order with safety rails.

Safety rails (all enforced here, not in the strategy):
  * ``dry_run`` (default!) prints intended orders without sending anything;
  * at most ``max_orders`` per cycle;
  * every order is sanity-banded: reject if the limit deviates more than
    ``price_band`` from the model mid;
  * a kill-switch file (``STOP`` next to the state file) aborts the cycle.
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd

from .. import data as datamod
from ..strategy import (MarketView, OptionPosition, OptionTrade,
                        Portfolio, StockTrade)
from ..vol import VolSurface, longrun_vol, realized_vol
from .brokers import Broker
from .state import LiveState


def surface_from_history(closes: pd.Series, premium: float = 1.15,
                         skew_slope: float = 0.10) -> VolSurface:
    """Fallback surface from recent daily closes (>= 6 months of history)."""
    rv = realized_vol(closes)
    rvl = longrun_vol(rv, window=min(len(closes) - 1, 756))
    return VolSurface(atm_short=float(rv.iloc[-1]) * premium,
                      atm_long=float(rvl.dropna().iloc[-1]) * premium,
                      skew_slope=skew_slope)


class LiveExecutor:
    def __init__(self, broker: Broker, strategy, ticker: str, state_path: str,
                 *, closes: pd.Series, r: float = 0.02,
                 div_yield: float | None = None, dry_run: bool = True,
                 max_orders: int = 8, price_band: float = 0.25):
        self.broker = broker
        self.strategy = strategy
        self.ticker = ticker
        self.state_path = state_path
        self.closes = closes
        self.r = r
        meta = datamod.UNIVERSE.get(ticker)
        self.q = div_yield if div_yield is not None else (meta.div_yield if meta else 0.03)
        self.dry_run = dry_run
        self.max_orders = max_orders
        self.price_band = price_band

    # ------------------------------------------------------------------
    def _portfolio_from_broker(self, state: LiveState) -> Portfolio:
        """Reconcile broker option positions with our role bookkeeping."""
        pos = self.broker.option_positions(self.ticker)
        pf = Portfolio(cash=self.broker.cash())
        for spec, qty in pos.items():
            role = 'leaps' if qty > 0 else 'short'
            # trust the state file for role tagging when it matches
            for slot, name in ((state.leaps, 'leaps'), (state.short, 'short')):
                if slot:
                    sspec, _ = LiveState.dict_to_spec(slot)
                    if sspec == spec:
                        role = name
            pf.options.append(OptionPosition(spec, qty, role, 0.0))
        return pf

    def run_once(self) -> list[dict]:
        stop_file = os.path.join(os.path.dirname(os.path.abspath(self.state_path)), 'STOP')
        if os.path.exists(stop_file):
            print(f'kill switch present ({stop_file}); doing nothing')
            return []

        state = LiveState.load(self.state_path, self.ticker)
        spot = self.broker.spot(self.ticker)
        view = MarketView(date=dt.date.today(), spot=spot, r=self.r, q=self.q,
                          surface=surface_from_history(self.closes))
        pf = self._portfolio_from_broker(state)

        actions = self.strategy.decide(view, pf)
        sent: list[dict] = []
        for act in actions[:self.max_orders]:
            if isinstance(act, StockTrade):
                print(f'[skip] stock trade {act} (PMCC live executor is options-only)')
                continue
            assert isinstance(act, OptionTrade)
            quote = self.broker.option_quote(self.ticker, act.spec)
            model_mid = view.price(act.spec)
            limit = quote.ask if act.qty > 0 else quote.bid
            if model_mid > 0.05 and abs(limit - model_mid) > self.price_band * max(model_mid, 0.10):
                print(f'[reject] {act}: limit {limit:.2f} outside band of model '
                      f'mid {model_mid:.2f}')
                continue
            order = {'spec': act.spec, 'qty': act.qty, 'limit': round(limit, 2),
                     'reason': act.reason, 'role': act.role}
            if self.dry_run:
                print(f'[dry-run] would send: {order}')
            else:
                res = self.broker.place_option_order(self.ticker, act.spec,
                                                     act.qty, limit)
                order['result'] = res.message or f'filled {res.filled_qty} @ {res.avg_price}'
                print(f'[sent] {order}')
                if res.ok:
                    self._update_state_after_fill(state, act, res)
            state.record('order', dry_run=self.dry_run,
                         spec=str(act.spec), qty=act.qty, limit=limit,
                         reason=act.reason)
            sent.append(order)

        state.last_run = dt.datetime.now(dt.timezone.utc).isoformat()
        state.save(self.state_path)
        return sent

    def _update_state_after_fill(self, state: LiveState, act: OptionTrade, res) -> None:
        slot = state.leaps if act.role == 'leaps' else state.short
        qty = res.filled_qty
        if slot:
            spec, have = LiveState.dict_to_spec(slot)
            if spec == act.spec:
                qty += have
        d = LiveState.spec_to_dict(act.spec, qty) if qty else None
        if act.role == 'leaps':
            state.leaps = d
        else:
            state.short = d

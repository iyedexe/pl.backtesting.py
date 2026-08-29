"""
Daily mark-to-market simulation engine.

The engine owns everything the strategy must not: trade execution with
transaction costs, option expiry settlement, interest and dividend accrual,
and daily equity marking.  Option trades fill at ``model mid +/- half-spread``
plus a per-contract commission; stock trades pay a proportional fee (plus the
French financial transaction tax on buys after Aug 2012).

Settlement is *cash settlement at intrinsic value* on the first trading day
on/after expiry -- economically equivalent to physical assignment followed by
an immediate repurchase, minus weekend gap risk (documented approximation).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as datamod
from .pricing import OptionSpec
from .strategy import (MULT, Action, MarketView, OptionPosition, OptionTrade,
                       Portfolio, StockTrade)
from .vol import VolSurface, longrun_vol, realized_vol


@dataclass
class CostModel:
    opt_commission: float = 1.75        # EUR per contract per leg
    opt_half_spread_pct: float = 0.02   # of model mid
    opt_half_spread_min: float = 0.03   # EUR per share
    stock_fee_pct: float = 0.001
    ftt_pct: float = 0.003              # French FTT, buy side
    ftt_start: dt.date = dt.date(2012, 8, 1)


@dataclass
class Result:
    ticker: str
    strategy: str
    equity: pd.Series
    net_delta: pd.Series
    spot: pd.Series
    trades: pd.DataFrame
    counters: dict
    params: dict = field(default_factory=dict)


class Engine:
    def __init__(self, ticker: str, *, initial_cash: float = 100_000.0,
                 prem: str | float = 'vix', skew_slope: float = 0.10,
                 div_yield: float | None = None, costs: CostModel | None = None,
                 start: str | None = None, end: str | None = None,
                 vol_floor: float = 0.10):
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.costs = costs or CostModel()
        meta = datamod.UNIVERSE.get(ticker)
        self.q = div_yield if div_yield is not None else (meta.div_yield if meta else 0.03)

        adj = datamod.load_adjusted_prices()[ticker].dropna()
        raw = datamod.reconstruct_raw_prices(adj, self.q)
        rv = realized_vol(raw, floor=vol_floor)
        rvl = longrun_vol(rv)
        if isinstance(prem, str) and prem == 'vix':
            pf = datamod.implied_premium_factor(raw.index)
        else:
            pf = pd.Series(float(prem), index=raw.index)
        r = datamod.eur_short_rate(raw.index)

        df = pd.DataFrame({'raw': raw, 'rv': rv, 'rvl': rvl, 'prem': pf, 'r': r}).dropna()
        if start:
            df = df.loc[start:]
        if end:
            df = df.loc[:end]
        if len(df) < 252:
            raise ValueError(f'not enough data for {ticker}')
        self.df = df
        self.skew_slope = skew_slope

    # ------------------------------------------------------------------ views
    def view_at(self, i: int) -> MarketView:
        row = self.df.iloc[i]
        surface = VolSurface(atm_short=row.rv * row.prem,
                             atm_long=row.rvl * row.prem,
                             skew_slope=self.skew_slope)
        return MarketView(date=self.df.index[i].date(), spot=float(row.raw),
                          r=float(row.r), q=self.q, surface=surface)

    # ----------------------------------------------------------------- pricing
    def _fill_price(self, view: MarketView, spec: OptionSpec, buying: bool) -> float:
        mid = view.price(spec)
        half = max(self.costs.opt_half_spread_pct * mid, self.costs.opt_half_spread_min)
        px = mid + half if buying else mid - half
        return max(px, 0.01)

    # ---------------------------------------------------------------- running
    def run(self, strategy) -> Result:
        c = self.costs
        pf = Portfolio(cash=self.initial_cash)
        counters = {'commissions': 0.0, 'spread_cost': 0.0, 'stock_fees': 0.0,
                    'ftt': 0.0, 'interest': 0.0, 'dividends': 0.0,
                    'short_premium_received': 0.0, 'short_buyback_paid': 0.0,
                    'short_settle_paid': 0.0, 'n_assignments': 0,
                    'n_short_cycles': 0, 'n_leaps_rolls': 0, 'n_trades': 0,
                    'min_cash': self.initial_cash}
        trade_rows: list[dict] = []
        eq_rows, delta_rows = [], []

        def log_trade(view, kind, detail, qty, px, cashflow):
            trade_rows.append({'date': view.date, 'kind': kind, 'detail': detail,
                               'qty': qty, 'price': round(px, 4),
                               'cashflow': round(cashflow, 2),
                               'spot': round(view.spot, 3)})

        def exec_option(view: MarketView, tr: OptionTrade):
            nonlocal pf
            if tr.qty == 0:
                return
            buying = tr.qty > 0
            px = self._fill_price(view, tr.spec, buying)
            mid = view.price(tr.spec)
            qty = tr.qty
            gross = -qty * MULT * px          # cash out when buying
            comm = c.opt_commission * abs(qty)
            if buying and pf.cash + gross - comm < -1e-6:
                afford = int((pf.cash - comm) // (MULT * px))
                if afford <= 0:
                    return
                qty = min(qty, afford)
                gross = -qty * MULT * px
                comm = c.opt_commission * abs(qty)
            pf.cash += gross - comm
            counters['commissions'] += comm
            counters['spread_cost'] += abs(qty) * MULT * abs(px - mid)
            counters['n_trades'] += 1
            if tr.role == 'short':
                if qty < 0:
                    counters['short_premium_received'] += gross
                    counters['n_short_cycles'] += 1
                else:
                    counters['short_buyback_paid'] += -gross
            # merge into portfolio
            for p in pf.options:
                if p.spec == tr.spec and p.role == tr.role:
                    p.qty += qty
                    break
            else:
                pf.options.append(OptionPosition(tr.spec, qty, tr.role, px))
            pf.options = [p for p in pf.options if p.qty != 0]
            log_trade(view, f'opt-{tr.role}', tr.reason, qty, px, gross - comm)

        def exec_stock(view: MarketView, tr: StockTrade):
            nonlocal pf
            n = tr.shares
            if n == 0:
                return
            px = view.spot
            fee_rate = c.stock_fee_pct + (c.ftt_pct if n > 0 and view.date >= c.ftt_start else 0.0)
            if n > 0:
                afford = int(pf.cash // (px * (1 + fee_rate)))
                n = min(n, afford)
                if n <= 0:
                    return
            gross = -n * px
            fee = abs(n) * px * c.stock_fee_pct
            ftt = n * px * c.ftt_pct if (n > 0 and view.date >= c.ftt_start) else 0.0
            pf.cash += gross - fee - ftt
            pf.shares += n
            counters['stock_fees'] += fee
            counters['ftt'] += ftt
            counters['n_trades'] += 1
            log_trade(view, 'stock', tr.reason, n, px, gross - fee - ftt)

        prev_date: dt.date | None = None
        for i in range(len(self.df)):
            view = self.view_at(i)

            # --- accrue interest on cash and dividends on shares
            if prev_date is not None:
                days = (view.date - prev_date).days
                if days > 0:
                    interest = pf.cash * (np.exp(view.r * days / 365.0) - 1.0)
                    pf.cash += interest
                    counters['interest'] += interest
                    if pf.shares:
                        div = pf.shares * view.spot * (np.exp(self.q * days / 365.0) - 1.0)
                        pf.cash += div
                        counters['dividends'] += div
            prev_date = view.date

            # --- settle expiries (first trading day on/after expiry)
            for p in list(pf.options):
                if p.spec.expiry <= view.date:
                    intrinsic = max(view.spot - p.spec.strike, 0.0) if p.spec.is_call \
                        else max(p.spec.strike - view.spot, 0.0)
                    assigned_shares = 0
                    if (p.role == 'short' and p.qty < 0 and intrinsic > 0
                            and p.spec.is_call and pf.shares >= -p.qty * MULT):
                        # physical assignment: shares called away at the strike
                        assigned_shares = -p.qty * MULT
                        proceeds = assigned_shares * p.spec.strike
                        fee = proceeds * c.stock_fee_pct
                        pf.shares -= assigned_shares
                        pf.cash += proceeds - fee
                        counters['stock_fees'] += fee
                        cashflow = proceeds - fee
                        detail = f'assigned: {assigned_shares} sh @ {p.spec.strike}'
                    else:
                        cashflow = p.qty * MULT * intrinsic
                        pf.cash += cashflow
                        detail = 'settled ITM' if intrinsic > 0 else 'expired worthless'
                    if p.role == 'short' and intrinsic > 0:
                        counters['n_assignments'] += 1
                        counters['short_settle_paid'] += abs(p.qty) * MULT * intrinsic
                    log_trade(view, f'expire-{p.role}', detail, -p.qty, intrinsic, cashflow)
                    pf.options.remove(p)

            # --- strategy decisions
            for act in strategy.decide(view, pf):
                if isinstance(act, OptionTrade):
                    if act.role == 'leaps' and act.qty < 0:
                        counters['n_leaps_rolls'] += 1
                    exec_option(view, act)
                else:
                    exec_stock(view, act)

            # --- mark to market
            counters['min_cash'] = min(counters['min_cash'], pf.cash)
            eq = pf.cash + pf.shares * view.spot
            ndelta = float(pf.shares)
            for p in pf.options:
                eq += p.qty * MULT * view.price(p.spec)
                ndelta += p.qty * MULT * view.delta(p.spec)
            eq_rows.append(eq)
            delta_rows.append(ndelta * view.spot)  # delta exposure in EUR

        # --- final liquidation for comparability
        view = self.view_at(len(self.df) - 1)
        for p in list(pf.options):
            exec_option(view, OptionTrade(p.spec, -p.qty, p.role, 'final liquidation'))
        if pf.shares:
            exec_stock(view, StockTrade(-pf.shares, 'final liquidation'))
        eq_rows[-1] = pf.cash

        idx = self.df.index
        return Result(
            ticker=self.ticker, strategy=strategy.name,
            equity=pd.Series(eq_rows, index=idx, name='equity'),
            net_delta=pd.Series(delta_rows, index=idx, name='delta_eur'),
            spot=self.df['raw'].copy(),
            trades=pd.DataFrame(trade_rows),
            counters=counters,
            params={'q': self.q, 'skew': self.skew_slope,
                    'initial': self.initial_cash},
        )

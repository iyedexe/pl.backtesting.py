"""
Strategy decision cores.

Strategies are *pure decision functions*: given a :class:`MarketView` (today's
market as the strategy is allowed to see it) and the current
:class:`Portfolio`, ``decide()`` returns a list of desired trades.  All
execution mechanics (costs, settlement, accrual) live in the engine, and the
same ``decide()`` cores are reused by the live-trading scaffold in
``pmcc/live/`` -- backtest and production share one brain.

Implemented:

* :class:`PMCC` -- the poor man's covered call: long deep-ITM LEAPS call
  (~0.80 delta, ~18 months), short OTM call (~0.25 delta, ~1 month), with
  mechanical rolling of both legs.
* :class:`CoveredCall` -- classic benchmark: 100-share lots + short OTM call.
* :class:`BuyHold` -- shares, dividends reinvested.
* :class:`LeapsOnly` -- the LEAPS ladder without short calls.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace

from .pricing import (OptionSpec, bs_delta, bs_price, next_expiry_with_dte,
                      snap_strike, strike_for_delta)
from .vol import VolSurface

MULT = 100  # Euronext single-stock option contract multiplier


# --------------------------------------------------------------------- state
@dataclass
class OptionPosition:
    spec: OptionSpec
    qty: int                    # contracts, >0 long / <0 short
    role: str                   # 'leaps' | 'short'
    entry_price: float          # model trade price per share at entry


@dataclass
class Portfolio:
    cash: float
    shares: int = 0
    options: list[OptionPosition] = field(default_factory=list)

    def option(self, role: str) -> OptionPosition | None:
        for p in self.options:
            if p.role == role and p.qty != 0:
                return p
        return None


# ---------------------------------------------------------------- market view
@dataclass(frozen=True)
class MarketView:
    date: dt.date
    spot: float                 # reconstructed *raw* price
    r: float                    # short rate (cont. comp., p.a.)
    q: float                    # dividend yield (cont., p.a.)
    surface: VolSurface

    def T(self, expiry: dt.date) -> float:
        return max((expiry - self.date).days, 0) / 365.0

    def iv(self, spec: OptionSpec) -> float:
        T = self.T(spec.expiry)
        F = self.spot * math.exp((self.r - self.q) * max(T, 1 / 365))
        return self.surface.iv(spec.strike, F, T)

    def price(self, spec: OptionSpec) -> float:
        return bs_price(self.spot, spec.strike, self.T(spec.expiry), self.r,
                        self.q, self.iv(spec), spec.is_call)

    def delta(self, spec: OptionSpec) -> float:
        return bs_delta(self.spot, spec.strike, self.T(spec.expiry), self.r,
                        self.q, self.iv(spec), spec.is_call)

    def strike_at_delta(self, target_delta: float, expiry: dt.date) -> float:
        """Grid-snapped strike whose model delta is closest to target."""
        T = self.T(expiry)
        # fixed-point on the skewed surface: two iterations are plenty
        F = self.spot * math.exp((self.r - self.q) * max(T, 1 / 365))
        sigma = self.surface.atm(T)
        for _ in range(3):
            k = strike_for_delta(self.spot, max(T, 1 / 365), self.r, self.q,
                                 sigma, target_delta)
            sigma = self.surface.iv(k, F, T)
        return snap_strike(k)


# --------------------------------------------------------------------- trades
@dataclass(frozen=True)
class OptionTrade:
    spec: OptionSpec
    qty: int                    # signed contracts to trade
    role: str
    reason: str = ''


@dataclass(frozen=True)
class StockTrade:
    shares: int                 # signed
    reason: str = ''


Action = OptionTrade | StockTrade


# ------------------------------------------------------------------- helpers
def _equity(view: MarketView, pf: Portfolio) -> float:
    eq = pf.cash + pf.shares * view.spot
    for p in pf.options:
        eq += p.qty * MULT * view.price(p.spec)
    return eq


# ----------------------------------------------------------------- strategies
@dataclass
class PMCCParams:
    leaps_target_delta: float = 0.80
    leaps_open_min_dte: int = 540      # open ~18 months out
    leaps_roll_dte: int = 90           # roll when fewer days remain
    short_target_delta: float = 0.25
    short_min_dte: int = 20
    short_max_dte: int = 50
    short_min_strike_over_spot: float = 1.01   # never sell at/below spot
    short_min_strike_over_leaps: float = 1.03  # keep spread width positive
    defend_delta: float | None = None  # e.g. 0.60: roll short when breached
    sizing: str = 'match_stock'        # 'match_stock' | 'budget'
    leaps_budget: float = 0.75         # for sizing='budget'


class PMCC:
    """Poor man's covered call decision core."""

    name = 'PMCC'

    def __init__(self, params: PMCCParams | None = None):
        self.p = params or PMCCParams()

    # -- sizing -------------------------------------------------------------
    def _target_units(self, view: MarketView, pf: Portfolio,
                      leaps_spec: OptionSpec) -> int:
        eq = _equity(view, pf)
        if self.p.sizing == 'match_stock':
            # one PMCC unit per 100 shares the same equity could buy outright
            return max(int(eq // (MULT * view.spot)), 0)
        cost = max(view.price(leaps_spec), 1e-6) * MULT
        return max(int(eq * self.p.leaps_budget // cost), 0)

    def _new_leaps_spec(self, view: MarketView) -> OptionSpec:
        expiry = next_expiry_with_dte(view.date, self.p.leaps_open_min_dte)
        strike = view.strike_at_delta(self.p.leaps_target_delta, expiry)
        return OptionSpec(strike, expiry)

    # -- decision -----------------------------------------------------------
    def decide(self, view: MarketView, pf: Portfolio) -> list[Action]:
        acts: list[Action] = []
        p = self.p
        leaps = pf.option('leaps')
        short = pf.option('short')

        # 0) margin call: expiry settlements may have driven cash negative
        #    (possible under sizing='budget'); forcibly deleverage.
        if pf.cash < -1e-6 and leaps is not None and leaps.qty > 0:
            leaps_px = max(view.price(leaps.spec), 0.01)
            trim_est = 0.0
            if short is not None:
                trim_est = abs(short.qty) * MULT * view.price(short.spec)
            sell = math.ceil((-pf.cash + trim_est) / (leaps_px * MULT))
            sell = min(sell, leaps.qty)
            acts.append(OptionTrade(leaps.spec, -sell, 'leaps', 'margin call'))
            remaining = leaps.qty - sell
            if short is not None and abs(short.qty) > remaining:
                acts.append(OptionTrade(short.spec, abs(short.qty) - remaining,
                                        'short', 'margin call trim'))
            return acts

        # 1) LEAPS entry / roll
        roll_leaps = leaps is not None and view.T(leaps.spec.expiry) * 365 <= p.leaps_roll_dte
        if leaps is None or roll_leaps:
            if short is not None:
                acts.append(OptionTrade(short.spec, -short.qty, 'short', 'unwind with LEAPS roll'))
                short = None
            if leaps is not None:
                acts.append(OptionTrade(leaps.spec, -leaps.qty, 'leaps', 'roll LEAPS'))
            spec = self._new_leaps_spec(view)
            units = self._target_units(view, pf, spec)
            if units > 0:
                acts.append(OptionTrade(spec, units, 'leaps',
                                        'open LEAPS' if leaps is None else 'roll LEAPS'))
            leaps_qty, leaps_spec = units, spec
        else:
            leaps_qty, leaps_spec = leaps.qty, leaps.spec

        if leaps_qty <= 0:
            return acts

        # 2) short call management
        if short is not None and p.defend_delta is not None \
                and view.delta(short.spec) >= p.defend_delta:
            acts.append(OptionTrade(short.spec, -short.qty, 'short', 'defensive roll'))
            short = None

        if short is None:
            expiry = next_expiry_with_dte(view.date, p.short_min_dte, p.short_max_dte)
            if expiry is None:
                expiry = next_expiry_with_dte(view.date, p.short_min_dte)
            strike = view.strike_at_delta(p.short_target_delta, expiry)
            floor_k = max(view.spot * p.short_min_strike_over_spot,
                          leaps_spec.strike * p.short_min_strike_over_leaps)
            strike = max(strike, snap_strike(floor_k))
            spec = OptionSpec(strike, expiry)
            if view.price(spec) >= 0.05:      # don't sell worthless teenies
                acts.append(OptionTrade(spec, -leaps_qty, 'short', 'sell monthly call'))
        elif abs(short.qty) > leaps_qty:
            acts.append(OptionTrade(short.spec, abs(short.qty) - leaps_qty,
                                    'short', 'reduce shorts to LEAPS units'))
        return acts


@dataclass
class CoveredCallParams:
    short_target_delta: float = 0.25
    short_min_dte: int = 20
    short_max_dte: int = 50
    short_min_strike_over_spot: float = 1.01
    defend_delta: float | None = None


class CoveredCall:
    """Classic covered call on 100-share lots; premiums/dividends reinvested."""

    name = 'CoveredCall'

    def __init__(self, params: CoveredCallParams | None = None):
        self.p = params or CoveredCallParams()

    def decide(self, view: MarketView, pf: Portfolio) -> list[Action]:
        acts: list[Action] = []
        p = self.p
        if pf.cash < -1e-6 and pf.shares > 0:
            n = min(pf.shares, int(math.ceil(-pf.cash / view.spot)))
            return [StockTrade(-n, 'margin call')]
        lot_cost = MULT * view.spot
        # deploy spare cash into additional lots
        add_lots = int(pf.cash // (lot_cost * 1.005))
        if add_lots > 0:
            acts.append(StockTrade(add_lots * MULT, 'deploy cash into lot(s)'))
        lots = (pf.shares + add_lots * MULT) // MULT
        short = pf.option('short')

        if short is not None and p.defend_delta is not None \
                and view.delta(short.spec) >= p.defend_delta:
            acts.append(OptionTrade(short.spec, -short.qty, 'short', 'defensive roll'))
            short = None

        if short is None and lots > 0:
            expiry = next_expiry_with_dte(view.date, p.short_min_dte, p.short_max_dte)
            if expiry is None:
                expiry = next_expiry_with_dte(view.date, p.short_min_dte)
            strike = view.strike_at_delta(p.short_target_delta, expiry)
            strike = max(strike, snap_strike(view.spot * p.short_min_strike_over_spot))
            spec = OptionSpec(strike, expiry)
            if view.price(spec) >= 0.05:
                acts.append(OptionTrade(spec, -int(lots), 'short', 'sell monthly call'))
        elif short is not None and abs(short.qty) > lots:
            acts.append(OptionTrade(short.spec, abs(short.qty) - int(lots),
                                    'short', 'reduce shorts to lots'))
        return acts


class BuyHold:
    """Buy & hold with dividends (continuous accrual) reinvested."""

    name = 'BuyHold'

    def decide(self, view: MarketView, pf: Portfolio) -> list[Action]:
        n = int(pf.cash // (view.spot * 1.005))
        if n > 0:
            return [StockTrade(n, 'reinvest')]
        return []


class LeapsOnly(PMCC):
    """PMCC without the short-call leg (isolates the LEAPS effect)."""

    name = 'LeapsOnly'

    def decide(self, view: MarketView, pf: Portfolio) -> list[Action]:
        acts = super().decide(view, pf)
        return [a for a in acts
                if not (isinstance(a, OptionTrade) and a.role == 'short')]

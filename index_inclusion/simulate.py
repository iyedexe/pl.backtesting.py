"""Synthetic point-in-time market with a mechanical index rulebook.

Reproduces the market of the tutorial notebook exactly (same random stream for
the same seed): 220 stocks over 12 years, a fictional "SIX 100" index governed
by the FTSE 100 rulebook (quarterly reviews, data cutoff on the Tuesday before
the first Friday, effective after the third Friday, add at rank <= 90, drop at
rank >= 111), and an *injected* index effect — additions drift up ~5% between
announcement and effective day, partially reverting afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketParams:
    n_stocks: int = 220
    index_size: int = 100
    add_rank: int = 90         # automatic inclusion at this rank or better ...
    drop_rank: int = 111       # ... automatic deletion at this rank or worse
    start: str = '2014-01-01'
    end: str = '2025-12-31'
    impact: float = 0.05       # extra return of an addition, announcement -> effective
    reversion: float = 0.4     # fraction of the impact that reverts ...
    rev_bars: int = 10         # ... over this many sessions after the effective day
    seed: int = 11


@dataclass
class Market:
    params: MarketParams
    calendar: pd.DatetimeIndex
    tickers: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    shares: np.ndarray
    delta: np.ndarray                       # daily log-price perturbations from index flows
    reviews: list[dict]                     # cutoff/announce/effective bar indices + adds/drops
    initial_members: frozenset
    membership: list[tuple[int, frozenset]] = field(default_factory=list)

    @property
    def n_bars(self) -> int:
        return len(self.calendar)

    @property
    def n_stocks(self) -> int:
        return len(self.tickers)

    @property
    def cap(self) -> np.ndarray:
        return self.close * self.shares

    def members_at(self, t: int) -> frozenset:
        """Index membership in force on bar ``t`` (by effective date only)."""
        mem = self.initial_members
        for live_from, m in self.membership:
            if live_from > t:
                break
            mem = m
        return mem


def review_schedule(calendar: pd.DatetimeIndex) -> list[dict]:
    """Quarterly reviews as bar indices: data cutoff at the close of the Tuesday
    before the first Friday of Mar/Jun/Sep/Dec, announced the next trading day,
    effective after the close of the third Friday."""
    reviews = []
    n = len(calendar)
    for year in range(calendar[0].year, calendar[-1].year + 1):
        for month in (3, 6, 9, 12):
            days = pd.date_range(f'{year}-{month}-01', periods=31, freq='D')
            fridays = days[(days.month == month) & (days.weekday == 4)]
            cutoff = calendar.searchsorted(fridays[0] - pd.Timedelta(days=3), side='right') - 1
            effective = calendar.searchsorted(fridays[2], side='right') - 1
            if cutoff < 10 or effective >= n - 3:
                continue
            reviews.append({'cutoff': cutoff, 'announce': cutoff + 1, 'effective': effective})
    return reviews


def rank_of(cap: np.ndarray) -> np.ndarray:
    """Competition ranks, 1 = largest market cap."""
    rank = np.empty(len(cap), int)
    rank[np.argsort(-cap)] = np.arange(1, len(cap) + 1)
    return rank


def make_market(params: MarketParams | None = None, **overrides) -> Market:
    """Simulate the market and replay the index rulebook, injecting the index effect."""
    p = params or MarketParams()
    if overrides:
        p = MarketParams(**{**p.__dict__, **overrides})
    rng = np.random.default_rng(p.seed)
    n, index_size = p.n_stocks, p.index_size
    calendar = pd.bdate_range(p.start, p.end)
    T = len(calendar)

    tickers: list[str] = []
    while len(tickers) < n:
        t = ''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=rng.integers(3, 5)))
        if t not in tickers:
            tickers.append(t)

    # Log-prices: single market factor + per-stock drift and idiosyncratic noise.
    beta = rng.uniform(.7, 1.4, n)
    idio = rng.uniform(.15, .45, n) / np.sqrt(252)
    drift = rng.normal(.04, .10, n)[:, None] / 252
    market = rng.normal(.06 / 252, .16 / np.sqrt(252), T)
    log_close = (np.cumsum(drift + beta[:, None] * market
                           + idio[:, None] * rng.standard_normal((n, T)), axis=1)
                 + np.log(rng.uniform(20, 300, n))[:, None])

    # Shares outstanding: piecewise-constant, tweaked a little every quarter.
    quarter = pd.factorize(calendar.quarter + 10 * calendar.year)[0]
    q_factor = np.cumprod(1 + rng.normal(0, .015, (n, quarter[-1] + 1)), axis=1)
    caps0 = 10 ** rng.normal(9.8, .75, n)
    shares = (caps0 / np.exp(log_close[:, 0]))[:, None] * (q_factor / q_factor[:, :1])[:, quarter]

    reviews = review_schedule(calendar)
    delta = np.zeros((n, T))
    members = set(np.argsort(-np.exp(log_close[:, 0]) * shares[:, 0])[:index_size])
    initial_members = frozenset(members)
    membership: list[tuple[int, frozenset]] = []

    for rv in reviews:
        c = rv['cutoff']
        cap = np.exp(log_close[:, c] + delta[:, :c + 1].sum(1)) * shares[:, c]
        rank = rank_of(cap)
        adds = {i for i in range(n) if i not in members and rank[i] <= p.add_rank}
        drops = {i for i in members if rank[i] >= p.drop_rank}
        excess = len(members) + len(adds) - len(drops) - index_size
        if excess > 0:
            drops |= set(sorted(members - drops, key=lambda i: rank[i])[-excess:])
        elif excess < 0:
            pool = sorted(set(range(n)) - members - adds, key=lambda i: rank[i])
            adds |= set(pool[:-excess])
        members = members | adds
        members -= drops
        membership.append((rv['effective'] + 1, frozenset(members)))
        rv['adds'], rv['drops'] = sorted(adds), sorted(drops)

        ramp = np.log(1 + p.impact) / (rv['effective'] - rv['announce'] + 1)
        rev = np.log(1 + p.impact) * p.reversion / p.rev_bars
        for sign, stocks in ((1, adds), (-1, drops)):
            for i in stocks:
                delta[i, rv['announce']:rv['effective'] + 1] += sign * ramp
                delta[i, rv['effective'] + 1:rv['effective'] + 1 + p.rev_bars] -= sign * rev

    close = np.exp(log_close + np.cumsum(delta, axis=1))
    open_ = np.roll(close, 1, axis=1) * np.exp(rng.normal(0, .003, (n, T)))
    open_[:, 0] = close[:, 0]
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, .006, (n, T))))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, .006, (n, T))))
    volume = shares * .004 * np.exp(rng.normal(0, .5, (n, T)))
    for rv in reviews:
        volume[rv['adds'] + rv['drops'], rv['effective']] *= 6

    return Market(params=p, calendar=calendar, tickers=tickers, open=open_, high=high,
                  low=low, close=close, volume=volume, shares=shares, delta=delta,
                  reviews=reviews, initial_members=initial_members, membership=membership)

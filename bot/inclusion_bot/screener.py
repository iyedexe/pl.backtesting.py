"""Signal generation - the pure logic layer.

Mirrors the strategy in ``doc/examples/Index Inclusion Strategy.py``:

* **Rank-based indices** (FTSE 100, DAX 40, Nasdaq-100): in the run-up to a
  review cutoff, a non-member ranked comfortably inside the entry band is a
  *predicted* addition; from the cutoff until the effective day it is a
  *projected* addition (ranks are locked); either way the position is closed
  right after the effective day, when the trackers have traded.
* **S&P 500**: no calendar - a *watchlist* signal fires when a non-member
  crosses the published eligibility gates (market cap + GAAP profitability),
  with the committee's discretion clearly flagged.
* One theoretical position per index at a time, and a ~1-month maximum hold,
  as in the backtested example.

Everything here is a pure function of (config, market DataFrame, membership,
today, state) so it is fully testable offline.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from . import state as st
from .calendars import active_review, bdays_between


@dataclass
class Signal:
    action: str                    # 'BUY' | 'SELL' | 'INFO'
    index_key: str
    index_name: str
    symbol: str = ''
    name: str = ''
    currency: str = ''
    price: float | None = None
    rank: int | None = None
    band: int | None = None
    kind: str = ''                 # 'predicted' | 'projected' | 'watchlist'
    confidence: str = ''
    review_label: str = ''
    cutoff: dt.date | None = None
    effective: dt.date | None = None
    expected_exit_date: dt.date | None = None
    expected_exit_price: float | None = None
    effect: float = 0.0
    theoretical_pnl: float | None = None   # on the configured notional
    notional: float = 0.0
    reason: str = ''               # SELL: why we're closing
    entry_price: float | None = None
    entry_date: str = ''
    realized_pnl: float | None = None
    notes: list = field(default_factory=list)


def rank_universe(market: pd.DataFrame) -> pd.Series:
    """Competition rank by ranking cap, 1 = largest."""
    return market['rank_cap'].rank(ascending=False, method='min').astype(int)


def _position_id(index_key, symbol, review_cutoff):
    return f'{index_key}:{symbol}:{review_cutoff or "watch"}'


def scan_sells(idx, cfg, market: pd.DataFrame, today: dt.date,
               state: dict) -> list[Signal]:
    signals = []
    ranks = rank_universe(market) if len(market) else pd.Series(dtype=int)
    for pid, pos in list(st.open_positions(state, idx.key).items()):
        sym = pos['symbol']
        price = float(market.loc[sym, 'price']) if sym in market.index else None
        rank = int(ranks[sym]) if sym in ranks.index else None
        cutoff = (dt.date.fromisoformat(pos['review_cutoff'])
                  if pos.get('review_cutoff') else None)
        exit_date = (dt.date.fromisoformat(pos['expected_exit_date'])
                     if pos.get('expected_exit_date') else None)
        held = bdays_between(dt.date.fromisoformat(pos['entry_date']), today)

        reason = ''
        if exit_date and today >= exit_date:
            reason = ('Effective day has passed - the index trackers have '
                      'traded, which is the move this position was for.')
        elif (pos['kind'] == 'predicted' and cutoff and today < cutoff
                and rank is not None and rank > idx.fail_rank):
            reason = (f'Candidacy failed - rank decayed to {rank} '
                      f'(abandon beyond {idx.fail_rank}).')
        elif (pos['kind'] == 'watchlist' and sym in market.index
                and float(market.loc[sym, 'cap']) < idx.min_cap):
            reason = 'Fell back below the market-cap eligibility threshold.'
        elif held >= cfg.max_hold_sessions:
            reason = (f'Maximum holding period reached '
                      f'({cfg.max_hold_sessions} sessions ≈ 1 month).')
        if not reason:
            continue

        realized = (cfg.notional * (price / pos['entry_price'] - 1)
                    if price else None)
        signals.append(Signal(
            action='SELL', index_key=idx.key, index_name=idx.name,
            symbol=sym, name=pos.get('name', sym), currency=pos['currency'],
            price=price, kind=pos['kind'], reason=reason,
            entry_price=pos['entry_price'], entry_date=pos['entry_date'],
            realized_pnl=realized, notional=cfg.notional))
        del state['open'][pid]
        state['closed'].append({**pos, 'exit_date': str(today),
                                'exit_price': price, 'reason': reason})
        if pos['kind'] == 'watchlist':
            # allow a future re-crossing of the threshold to signal again
            st.unmark_sent(state, f'watch:{idx.key}:{sym}')
    return signals


def scan_rank_index(idx, cfg, market: pd.DataFrame, members: set,
                    today: dt.date, state: dict) -> list[Signal]:
    """Predicted/projected additions for FTSE 100, DAX 40, Nasdaq-100."""
    active = active_review(idx.key, today, idx.pre_cutoff_days)
    if not active or not len(market):
        return []
    review, phase = active
    if phase == 'effective':
        return []
    band = review.threshold - (idx.buffer if phase == 'pre_cutoff' else 0)
    ranks = rank_universe(market)
    eligible = sorted(((int(ranks[s]), s) for s in market.index
                       if s not in members and ranks[s] <= band))
    signals = []

    info_key = f'info:{idx.key}:{review.cutoff}'
    if phase == 'post_cutoff' and eligible and not st.was_sent(state, info_key):
        head = ', '.join(f'{s} (#{r})' for r, s in eligible[:3])
        signals.append(Signal(
            action='INFO', index_key=idx.key, index_name=idx.name,
            review_label=review.label, cutoff=review.cutoff,
            effective=review.effective, confidence=review.confidence,
            notes=[f'Review cutoff reached - ranks are locked. Projected '
                   f'addition(s) inside the {review.threshold}-band: {head}. '
                   f'Verify against the official announcement before acting.']))
        st.mark_sent(state, info_key)

    if len(st.open_positions(state, idx.key)) >= cfg.max_open_per_index:
        return signals
    for rank, sym in eligible:
        buy_key = f'buy:{idx.key}:{sym}:{review.cutoff}'
        if st.was_sent(state, buy_key):
            continue
        row = market.loc[sym]
        price = float(row['price'])
        signal = Signal(
            action='BUY', index_key=idx.key, index_name=idx.name,
            symbol=sym, name=str(row.get('name', sym)),
            currency=str(row.get('currency', idx.currency)) or idx.currency,
            price=price, rank=rank, band=band,
            kind='predicted' if phase == 'pre_cutoff' else 'projected',
            confidence=review.confidence, review_label=review.label,
            cutoff=review.cutoff, effective=review.effective,
            expected_exit_date=review.exit_date,
            expected_exit_price=price * (1 + idx.expected_effect),
            effect=idx.expected_effect,
            theoretical_pnl=cfg.notional * idx.expected_effect,
            notional=cfg.notional)
        st.mark_sent(state, buy_key)
        state['open'][_position_id(idx.key, sym, review.cutoff)] = {
            'index': idx.key, 'symbol': sym, 'name': signal.name,
            'currency': signal.currency, 'entry_price': price,
            'entry_date': str(today), 'kind': signal.kind,
            'review_cutoff': str(review.cutoff),
            'expected_exit_date': str(review.exit_date),
            'expected_exit_price': signal.expected_exit_price,
        }
        signals.append(signal)
        break                                   # one stock at a time
    return signals


def scan_watchlist(idx, cfg, market: pd.DataFrame, members: set,
                   today: dt.date, state: dict, profitability) -> list[Signal]:
    """S&P 500: eligibility-gate watchlist (committee decides the timing)."""
    if not len(market):
        return []
    signals = []
    eligible = market[(~market.index.isin(members))
                      & (market['cap'] >= idx.min_cap)]
    if len(st.open_positions(state, idx.key)) >= cfg.max_open_per_index:
        return signals
    for sym in eligible.sort_values('cap', ascending=False).index:
        key = f'watch:{idx.key}:{sym}'
        if st.was_sent(state, key):
            continue
        profit = profitability(sym) if idx.require_profitability else True
        if profit is False:
            continue
        row = market.loc[sym]
        price = float(row['price'])
        notes = ['Committee-governed index: eligibility is necessary, not '
                 'sufficient - inclusion timing is at the committee\'s '
                 'discretion.']
        if profit is None:
            notes.append('Earnings gate could not be verified from public '
                         'statements - check GAAP profitability yourself.')
        signal = Signal(
            action='BUY', index_key=idx.key, index_name=idx.name,
            symbol=sym, name=str(row.get('name', sym)),
            currency=str(row.get('currency', idx.currency)) or idx.currency,
            price=price, kind='watchlist', confidence='speculative',
            expected_exit_price=price * (1 + idx.expected_effect),
            effect=idx.expected_effect,
            theoretical_pnl=cfg.notional * idx.expected_effect,
            notional=cfg.notional, notes=notes)
        st.mark_sent(state, key)
        state['open'][_position_id(idx.key, sym, None)] = {
            'index': idx.key, 'symbol': sym, 'name': signal.name,
            'currency': signal.currency, 'entry_price': price,
            'entry_date': str(today), 'kind': 'watchlist',
            'review_cutoff': None, 'expected_exit_date': None,
            'expected_exit_price': signal.expected_exit_price,
        }
        signals.append(signal)
        break                                   # one stock at a time
    return signals


def scan_index(idx, cfg, market: pd.DataFrame, members: set, today: dt.date,
               state: dict, profitability=lambda s: None) -> list[Signal]:
    """Sell checks first (they free the slot), then new entries."""
    signals = scan_sells(idx, cfg, market, today, state)
    if idx.min_cap:
        signals += scan_watchlist(idx, cfg, market, members, today, state,
                                  profitability)
    else:
        signals += scan_rank_index(idx, cfg, market, members, today, state)
    return signals

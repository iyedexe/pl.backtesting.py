"""Two-leg, dollar-neutral pair backtest engine.

Unlike the single-instrument approximation (backtesting a synthetic ratio
series), this engine models both legs explicitly: share quantities, per-leg
traded notionals and per-leg transaction costs, so the P&L of "long the spread"
is exactly the P&L of the underlying long/short portfolio.

Position convention: a *long spread* position (+1) with hedge ratio ``beta``
allocates ``N = equity * leverage / (1 + beta)`` dollars long leg A and
``beta * N`` dollars short leg B, so that the position's return per bar is
``N * (r_A - beta * r_B)`` and gross exposure equals ``equity * leverage``
(beta-weighted hedging: net exposure is ``N * (1 - beta)``, zero at beta = 1).

Timing convention: the ``side`` input is the target decided on the close of
bar *t* (from information up to and including that close); execution happens on
the close of bar ``t + execution_lag``. The default lag of one bar removes the
same-bar look-ahead and the bid-ask bounce subsidy documented by Gatev,
Goetzmann & Rouwenhorst (2006).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import EngineConfig

TRADE_COLUMNS = ['entry_date', 'exit_date', 'side', 'beta', 'gross_entry',
                 'pnl', 'pnl_bp', 'costs', 'holding', 'reason']


@dataclass
class PairBacktestResult:
    """Output of :func:`backtest_pair`."""
    name: str
    equity: pd.Series                 # equity curve, starts at 1.0
    returns: pd.Series                # per-bar simple returns of the equity curve
    positions: pd.DataFrame           # columns: side, qa, qb (share quantities)
    trades: pd.DataFrame              # one row per round trip (TRADE_COLUMNS)
    config: EngineConfig
    skipped_entries: int = 0          # entries refused (bad/degenerate hedge ratio)
    meta: dict = field(default_factory=dict)


def backtest_pair(prices: pd.DataFrame,
                  side: pd.Series,
                  beta: pd.Series | float,
                  cfg: EngineConfig,
                  reasons: pd.Series | None = None,
                  name: str = 'pair') -> PairBacktestResult:
    """Backtest one pair.

    Parameters
    ----------
    prices : DataFrame with columns ['a', 'b'] (close prices, inner-joined).
    side : target spread position (+1/0/-1) decided on each bar's close.
    beta : hedge ratio series (causally estimated) or a constant; the value
        in effect on the *execution* bar is locked in for the whole trade.
    cfg : execution assumptions (costs, lag, leverage).
    reasons : optional exit-reason series aligned with `side` (signal dates).
    """
    idx = prices.index
    a = prices['a'].to_numpy(float)
    b = prices['b'].to_numpy(float)
    n = len(idx)
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError('prices contain NaN — inner-join the legs first '
                         '(see pairs_trading.data.aligned_pair)')
    if isinstance(beta, (int, float)):
        beta = pd.Series(float(beta), index=idx)
    # The hedge ratio is lagged like the signal: a trade executed on close
    # t + lag is sized with the beta known at the *signal* close t.
    beta_exec = (beta.reindex(idx).ffill()
                 .shift(cfg.execution_lag).to_numpy(float))
    target = (side.reindex(idx).fillna(0).astype(int)
              .shift(cfg.execution_lag).fillna(0).to_numpy())
    if reasons is None:
        reasons = pd.Series('', index=idx)
    reason_exec = reasons.reindex(idx).fillna('').shift(cfg.execution_lag).fillna('')
    reason_exec = reason_exec.to_numpy(dtype=object)

    rate = cfg.cost.rate()
    equity = np.empty(n)
    qa_arr = np.zeros(n)
    qb_arr = np.zeros(n)
    side_arr = np.zeros(n, dtype=np.int8)

    e = 1.0
    qa = qb = 0.0
    cur = 0
    skipped = 0
    trades: list[dict] = []
    open_trade: dict | None = None

    for t in range(n):
        if t > 0:
            e += qa * (a[t] - a[t - 1]) + qb * (b[t] - b[t - 1])
        tgt = int(target[t])
        if tgt != cur:
            # Close any open position at this close.
            if cur != 0 and open_trade is not None:
                gross_now = abs(qa) * a[t] + abs(qb) * b[t]
                cost = rate * gross_now
                e -= cost
                pnl = (qa * (a[t] - open_trade['_pa']) + qb * (b[t] - open_trade['_pb'])
                       - open_trade['costs'] - cost)
                open_trade.update(
                    exit_date=idx[t], pnl=pnl,
                    pnl_bp=1e4 * pnl / open_trade['gross_entry'],
                    costs=open_trade['costs'] + cost,
                    holding=int(t - open_trade['_t0']),
                    reason=reason_exec[t] or ('flip' if tgt != 0 else ''))
                trades.append({k: v for k, v in open_trade.items()
                               if not k.startswith('_')})
                open_trade, qa, qb, cur = None, 0.0, 0.0, 0
            # Open the new position, unless the hedge ratio is degenerate or
            # this is the final bar (a same-bar open+force-close round trip
            # could never exist and would only pollute the trade statistics).
            if tgt != 0 and t < n - 1:
                bt_ = beta_exec[t]
                if not np.isfinite(bt_) or not (cfg.min_beta <= bt_ <= cfg.max_beta):
                    skipped += 1
                else:
                    gross = e * cfg.leverage
                    nn = gross / (1.0 + bt_)
                    qa = tgt * nn / a[t]
                    qb = -tgt * bt_ * nn / b[t]
                    cost = rate * gross
                    e -= cost
                    cur = tgt
                    open_trade = {
                        'entry_date': idx[t], 'exit_date': idx[t], 'side': tgt,
                        'beta': bt_, 'gross_entry': gross, 'pnl': 0.0,
                        'pnl_bp': 0.0, 'costs': cost, 'holding': 0, 'reason': '',
                        '_pa': a[t], '_pb': b[t], '_t0': t,
                    }
        if e <= 0:
            # Bust: the short leg gapped through the account (possible for a
            # crypto pair even at leverage 1). Model limited liability of the
            # capital slice: equity floors at exactly zero, the position is
            # liquidated at this bar's close, and the curve stays dead. The
            # trade ledger keeps the *uncapped* economics (reason 'bust').
            if open_trade is not None:
                gross_now = abs(qa) * a[t] + abs(qb) * b[t]
                cost = rate * gross_now
                pnl = (qa * (a[t] - open_trade['_pa'])
                       + qb * (b[t] - open_trade['_pb'])
                       - open_trade['costs'] - cost)
                open_trade.update(
                    exit_date=idx[t], pnl=pnl,
                    pnl_bp=1e4 * pnl / open_trade['gross_entry'],
                    costs=open_trade['costs'] + cost,
                    holding=int(t - open_trade['_t0']), reason='bust')
                trades.append({k: v for k, v in open_trade.items()
                               if not k.startswith('_')})
                open_trade, qa, qb, cur = None, 0.0, 0.0, 0
            equity[t:] = 0.0
            qa_arr[t:], qb_arr[t:], side_arr[t:] = 0, 0, 0
            break
        equity[t] = e
        qa_arr[t], qb_arr[t], side_arr[t] = qa, qb, cur

    # Force-close anything still open on the final bar.
    if open_trade is not None:
        t = n - 1
        gross_now = abs(qa) * a[t] + abs(qb) * b[t]
        cost = rate * gross_now
        equity[t] -= cost
        pnl = (qa * (a[t] - open_trade['_pa']) + qb * (b[t] - open_trade['_pb'])
               - open_trade['costs'] - cost)
        open_trade.update(
            exit_date=idx[t], pnl=pnl,
            pnl_bp=1e4 * pnl / open_trade['gross_entry'],
            costs=open_trade['costs'] + cost,
            holding=int(t - open_trade['_t0']), reason='end')
        trades.append({k: v for k, v in open_trade.items() if not k.startswith('_')})

    equity_s = pd.Series(equity, index=idx, name='equity')
    returns = equity_s.pct_change().fillna(0.0)
    positions = pd.DataFrame({'side': side_arr, 'qa': qa_arr, 'qb': qb_arr}, index=idx)
    trades_df = (pd.DataFrame(trades, columns=TRADE_COLUMNS) if trades
                 else pd.DataFrame(columns=TRADE_COLUMNS))
    return PairBacktestResult(name=name, equity=equity_s, returns=returns,
                              positions=positions, trades=trades_df, config=cfg,
                              skipped_entries=skipped)

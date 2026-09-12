"""Performance statistics for backtest equity curves."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_stats(equity: pd.Series, r: pd.Series | None = None) -> dict:
    eq = equity.dropna()
    ret = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(TRADING_DAYS)

    if r is not None:
        rf_daily = r.reindex(ret.index).ffill() / TRADING_DAYS
        excess = ret - rf_daily
    else:
        excess = ret
    sharpe = excess.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = excess.mean() * TRADING_DAYS / downside if downside > 0 else np.nan

    peak = eq.cummax()
    dd = eq / peak - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    yearly = eq.resample('YE').last().pct_change().dropna()
    first_year = eq.resample('YE').last().iloc[0] / eq.iloc[0] - 1
    yearly = pd.concat([pd.Series([first_year], index=[eq.resample('YE').last().index[0]]), yearly])

    return {
        'Final [EUR]': round(eq.iloc[-1], 0),
        'Total Return [%]': round(100 * total, 1),
        'CAGR [%]': round(100 * cagr, 2),
        'Vol (ann.) [%]': round(100 * vol, 1),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'Max Drawdown [%]': round(100 * max_dd, 1),
        'Calmar': round(calmar, 2),
        'Worst Year [%]': round(100 * yearly.min(), 1),
        'Best Year [%]': round(100 * yearly.max(), 1),
        'Years': round(years, 1),
    }


def drawdown_series(equity: pd.Series) -> pd.Series:
    eq = equity.dropna()
    return eq / eq.cummax() - 1

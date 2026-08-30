"""Performance and statistical-significance metrics for pair backtests.

Conventions follow the stat-arb literature (see research/reports/report.md):
Sharpe annualized with the class-appropriate period count (252 trading days,
365 calendar days for crypto), Newey-West HAC t-statistics on mean daily
returns, and the Bailey & López de Prado (2014) deflated Sharpe ratio for
results selected from a parameter grid.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps


def annualized_sharpe(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.to_numpy(float)
    sd = r.std(ddof=1)
    if sd == 0 or len(r) < 3:
        return float('nan')
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def annualized_sortino(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.to_numpy(float)
    downside = r[r < 0]
    if len(downside) < 2:
        return float('nan')
    dd = math.sqrt((downside ** 2).mean())
    if dd == 0:
        return float('nan')
    return float(r.mean() / dd * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, as a positive fraction."""
    e = equity.to_numpy(float)
    peak = np.maximum.accumulate(e)
    dd = 1.0 - e / peak
    return float(dd.max())


def cagr(equity: pd.Series, periods_per_year: int) -> float:
    n = len(equity)
    if n < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float('nan')
    years = (n - 1) / periods_per_year
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def newey_west_tstat(returns: pd.Series, lags: int | None = None) -> float:
    """HAC t-statistic of the mean per-bar return (Newey-West 1987).

    Default lag choice is the standard automatic rule
    ``floor(4 * (T/100)^(2/9))``.
    """
    r = returns.to_numpy(float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 30:
        return float('nan')
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    mu = r.mean()
    e = r - mu
    gamma0 = float((e @ e) / n)
    var = gamma0
    for lag in range(1, lags + 1):
        gamma = float((e[lag:] @ e[:-lag]) / n)
        var += 2.0 * (1 - lag / (lags + 1)) * gamma
    if var <= 0:
        return float('nan')
    return float(mu / math.sqrt(var / n))


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes: np.ndarray) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Probability that the observed (per-bar) Sharpe of ``returns`` exceeds the
    expected maximum Sharpe obtainable by chance across ``len(trial_sharpes)``
    tried configurations (whose per-bar Sharpes are supplied), accounting for
    track length, skewness and kurtosis. Values near 1 indicate a Sharpe
    unlikely to be a selection artifact; values near 0.5 or below indicate it
    is entirely consistent with multiple-testing noise.
    """
    r = returns.to_numpy(float)
    n = len(r)
    sd = r.std(ddof=1)
    if n < 30 or sd == 0:
        return float('nan')
    sr = r.mean() / sd  # per-bar Sharpe
    trials = np.asarray(trial_sharpes, float)
    trials = trials[np.isfinite(trials)]
    n_trials = max(len(trials), 2)
    var_trials = trials.var(ddof=1) if len(trials) > 1 else 0.0
    emc = 0.5772156649015329  # Euler-Mascheroni
    max_z = ((1 - emc) * sps.norm.ppf(1 - 1 / n_trials)
             + emc * sps.norm.ppf(1 - 1 / (n_trials * math.e)))
    sr0 = math.sqrt(max(var_trials, 1e-18)) * max_z
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr0) * math.sqrt(n - 1) / denom
    return float(sps.norm.cdf(z))


def summarize(result, periods_per_year: int | None = None) -> pd.Series:
    """One-row summary of a :class:`~pairs_trading.engine.PairBacktestResult`."""
    ppy = periods_per_year or result.config.periods_per_year
    active = result.positions['side'] != 0
    return summarize_run(result.name, result.returns, result.equity,
                         result.trades, float(active.mean()), ppy)


def summarize_run(name: str, ret: pd.Series, eq: pd.Series, tr: pd.DataFrame,
                  exposure: float, ppy: int) -> pd.Series:
    """Summary from raw run components (works for stitched walk-forward output)."""
    n_tr = len(tr)
    closed = tr[tr['reason'] != ''] if n_tr else tr
    out = {
        'total_return_pct': 100 * (eq.iloc[-1] / eq.iloc[0] - 1),
        'cagr_pct': 100 * cagr(eq, ppy),
        'ann_vol_pct': 100 * ret.std(ddof=1) * math.sqrt(ppy),
        'sharpe': annualized_sharpe(ret, ppy),
        'sortino': annualized_sortino(ret, ppy),
        'max_dd_pct': 100 * max_drawdown(eq),
        'nw_tstat': newey_west_tstat(ret),
        'exposure_pct': 100 * exposure,
        'n_trades': n_tr,
        'win_rate_pct': 100 * float((tr['pnl'] > 0).mean()) if n_tr else float('nan'),
        'avg_trade_bp': float(tr['pnl_bp'].mean()) if n_tr else float('nan'),
        'med_holding_bars': float(tr['holding'].median()) if n_tr else float('nan'),
        'converged_pct': (100 * float((closed['reason'] == 'converged').mean())
                          if len(closed) else float('nan')),
        'stopped_pct': (100 * float((closed['reason'] == 'stopped').mean())
                        if len(closed) else float('nan')),
        'skew': float(sps.skew(ret.to_numpy())) if len(ret) > 30 else float('nan'),
        'years': (len(ret) - 1) / ppy,
    }
    return pd.Series(out, name=name)


def summarize_walkforward(wf, ppy: int) -> pd.Series:
    """Summary of a :class:`~pairs_trading.walkforward.WalkForwardResult`.

    Exposure is approximated as total bars-in-position over total bars.
    """
    n = max(len(wf.returns), 1)
    exposure = float(wf.trades['holding'].sum()) / n if len(wf.trades) else 0.0
    s = summarize_run(wf.name, wf.returns, wf.equity, wf.trades, exposure, ppy)
    s['gate_pass_pct'] = 100 * wf.gate_pass_rate
    return s

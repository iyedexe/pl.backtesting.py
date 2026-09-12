"""
Volatility estimation and the synthetic implied-volatility surface.

Because no free historical option-chain data exists for Euronext Paris single
stocks, implied vols are *modelled* from realised volatility:

    ATM short-dated IV(t) = blended_realised_vol(t) * premium(t)

where ``premium(t)`` is either a constant (sensitivity parameter) or the
empirical VIX / S&P-realised ratio (see :func:`pmcc.data.implied_premium_factor`).

The surface then applies:
  * **term structure** -- longer expiries blend toward the stock's long-run
    volatility (mean reversion), and
  * **skew** -- IV falls with log-moneyness at rate ``skew_slope / sqrt(T)``,
    which makes OTM calls *cheaper* to sell and deep-ITM calls (our LEAPS)
    *dearer* to buy.  Both effects work **against** the PMCC, so the default
    surface is conservative for the strategy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realized_vol(prices: pd.Series, lam: float = 0.94, window: int = 63,
                 floor: float = 0.10, cap: float = 1.20) -> pd.Series:
    """Blend of EWMA (RiskMetrics lambda) and rolling-window realised vol."""
    ret = np.log(prices).diff()
    ew = np.sqrt(ret.pow(2).ewm(alpha=1 - lam, min_periods=21).mean() * TRADING_DAYS)
    ro = ret.rolling(window, min_periods=21).std() * math.sqrt(TRADING_DAYS)
    rv = 0.5 * ew + 0.5 * ro
    return rv.clip(lower=floor, upper=cap).rename('rv')


def longrun_vol(rv: pd.Series, window: int = 756) -> pd.Series:
    """Slow-moving anchor for the term structure (3-year mean realised vol)."""
    return rv.rolling(window, min_periods=126).mean().rename('rv_long')


@dataclass(frozen=True)
class VolSurface:
    """Synthetic IV surface for one date."""
    atm_short: float          # short-dated ATM IV (premium already applied)
    atm_long: float           # long-run anchor IV (premium already applied)
    skew_slope: float = 0.10  # vol-point drop per unit log-moneyness per sqrt(year)
    term_tau: float = 1.5     # years; speed of blending toward atm_long

    def atm(self, T: float) -> float:
        w = math.exp(-max(T, 0.0) / self.term_tau)
        return self.atm_short * w + self.atm_long * (1 - w)

    def iv(self, K: float, F: float, T: float) -> float:
        """IV for strike K given forward F and maturity T (years)."""
        T = max(T, 1 / 365)
        base = self.atm(T)
        iv = base - self.skew_slope * math.log(K / F) / math.sqrt(T)
        # Keep the wings sane: no less than half the ATM level, floor at 5%.
        return float(max(iv, 0.5 * base, 0.05))

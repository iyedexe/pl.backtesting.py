"""Cointegration, hedge-ratio and mean-reversion diagnostics.

All price inputs are *levels*; every function converts to log prices internally
where appropriate. The cointegrating regression convention throughout the
project is::

    ln(A_t) = alpha + beta * ln(B_t) + spread_t

so a stationary ``spread_t`` (Engle-Granger) makes the pair tradable, and a unit
of "long spread" means long A / short beta-dollars of B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class EGResult:
    """Engle-Granger two-step test result for one orientation of a pair."""
    y: str
    x: str
    alpha: float
    beta: float
    tstat: float
    pvalue: float
    half_life: float
    hurst: float
    n_obs: int

    def spread(self, log_a: pd.Series, log_b: pd.Series) -> pd.Series:
        return log_a - self.alpha - self.beta * log_b


def ols_hedge(log_a: pd.Series, log_b: pd.Series) -> tuple[float, float]:
    """OLS ``log_a = alpha + beta*log_b``; returns (alpha, beta)."""
    x = np.asarray(log_b, float)
    y = np.asarray(log_a, float)
    beta, alpha = np.polyfit(x, y, 1)
    return float(alpha), float(beta)


def engle_granger(price_a: pd.Series, price_b: pd.Series,
                  names: tuple[str, str] = ('a', 'b'),
                  try_both_orientations: bool = True) -> EGResult:
    """Engle-Granger cointegration test on log prices.

    The EG test is not symmetric in the choice of regressand; when
    ``try_both_orientations`` we run both directions and keep the one with the
    lower MacKinnon p-value (recording which leg ended up as ``y``).
    """
    la, lb = np.log(price_a.astype(float)), np.log(price_b.astype(float))

    def one(y: pd.Series, x: pd.Series, ny: str, nx: str) -> EGResult:
        tstat, pvalue, _ = coint(y, x, trend='c')
        alpha, beta = ols_hedge(y, x)
        resid = y - alpha - beta * x
        return EGResult(y=ny, x=nx, alpha=alpha, beta=beta,
                        tstat=float(tstat), pvalue=float(pvalue),
                        half_life=half_life(resid), hurst=hurst_exponent(resid),
                        n_obs=len(y))

    r1 = one(la, lb, names[0], names[1])
    if not try_both_orientations:
        return r1
    r2 = one(lb, la, names[1], names[0])
    return r1 if r1.pvalue <= r2.pvalue else r2


def adf_pvalue(series: pd.Series) -> float:
    """Plain ADF p-value (constant, AIC lag selection) for a residual series."""
    return float(adfuller(np.asarray(series, float), regression='c', autolag='AIC')[1])


def half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion in bars, from the discrete OU/AR(1) fit.

    Regress ``Δs_t = a + φ·s_{t-1} + ε``; the OU mean-reversion speed is
    ``κ = -φ`` per bar and the half-life is ``ln(2)/κ``. Returns ``inf`` when
    the fit shows no mean reversion (φ >= 0).
    """
    s = np.asarray(spread, float)
    s = s[~np.isnan(s)]
    if len(s) < 20:
        return float('inf')
    ds = np.diff(s)
    lag = s[:-1]
    phi, _ = np.polyfit(lag, ds, 1)
    if phi >= 0:
        return float('inf')
    return float(np.log(2) / -phi)


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """Hurst exponent via the variance-of-differences (aggregated Var) method.

    ``Var[x_{t+τ} - x_t] ~ τ^{2H}``: H < 0.5 mean-reverting, ≈ 0.5 random walk,
    > 0.5 trending. Estimated on the raw series (use the *spread*, not returns).
    """
    x = np.asarray(series, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 60:
        return float('nan')
    max_lag = int(min(max_lag, n // 4))
    lags = np.unique(np.logspace(np.log10(2), np.log10(max_lag), 20).astype(int))
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    tau = np.asarray(tau)
    ok = tau > 0
    if ok.sum() < 4:
        return float('nan')
    slope, _ = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)
    return float(slope)


def rolling_beta(log_a: pd.Series, log_b: pd.Series, window: int) -> pd.Series:
    """Rolling OLS hedge ratio (slope of log_a on log_b), NaN until `window` bars."""
    cov = log_a.rolling(window).cov(log_b)
    var = log_b.rolling(window).var()
    return cov / var


class KalmanHedge:
    """Dynamic hedge ratio via a random-walk state-space model (Kalman filter).

    Observation:  ``ln A_t = alpha_t + beta_t · ln B_t + v_t``,  v ~ N(0, Ve)
    State:        ``[alpha, beta]_t = [alpha, beta]_{t-1} + w_t``,
                  w ~ N(0, (δ/(1-δ))·I)

    This is the classic formulation popularized by E. Chan (2013), where ``δ``
    controls how fast the hedge ratio may drift. The filter is strictly causal:
    the estimate at *t* uses observations up to and including *t*.
    """

    def __init__(self, delta: float = 1e-4, ve: float = 1e-3,
                 beta0: float = 1.0, alpha0: float = 0.0, p0: float = 1.0):
        self.delta, self.ve = delta, ve
        self.theta0 = np.array([alpha0, beta0], float)
        self.p0 = p0

    def filter(self, log_a: pd.Series, log_b: pd.Series) -> pd.DataFrame:
        """Run the filter; returns DataFrame [alpha, beta, resid] indexed like input."""
        y = np.asarray(log_a, float)
        x = np.asarray(log_b, float)
        n = len(y)
        wt = self.delta / (1 - self.delta) * np.eye(2)
        theta = self.theta0.copy()
        p = np.eye(2) * self.p0
        out = np.empty((n, 3))
        for t in range(n):
            h = np.array([1.0, x[t]])
            # Predict
            p = p + wt
            # Innovation
            yhat = h @ theta
            e = y[t] - yhat
            qt = h @ p @ h + self.ve
            # Update
            k = p @ h / qt
            theta = theta + k * e
            p = p - np.outer(k, h) @ p
            out[t] = theta[0], theta[1], e
        return pd.DataFrame(out, index=log_a.index, columns=['alpha', 'beta', 'resid'])


def log_return_correlation(price_a: pd.Series, price_b: pd.Series) -> float:
    ra = np.log(price_a).diff()
    rb = np.log(price_b).diff()
    return float(ra.corr(rb))

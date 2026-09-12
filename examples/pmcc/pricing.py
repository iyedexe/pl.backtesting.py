"""
Option pricing utilities: Black-Scholes with continuous dividend yield,
delta-targeted strike solving, Euronext-style strike grids and the monthly
(third-Friday) expiry calendar.

All prices are *model mids*; the backtest engine applies half-spreads and
commissions on top.  European exercise is assumed (French single-stock options
are American style; with dividends modelled as a continuous yield the early
exercise premium is small and ignored -- see the methodology notes in
``pmcc/README.md``).
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

SQRT_2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f'p must be in (0,1), got {p}')
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    # One Halley refinement step
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
             is_call: bool = True) -> float:
    """Black-Scholes price with continuous dividend yield ``q``."""
    if T <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if sigma <= 0:
        f = S * math.exp(-q * T) - K * math.exp(-r * T)
        return max(f, 0.0) if is_call else max(-f, 0.0)
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float,
             is_call: bool = True) -> float:
    if T <= 0:
        intrinsic = (S > K) if is_call else (S < K)
        return (1.0 if intrinsic else 0.0) * (1 if is_call else -1)
    d1 = _d1(S, K, T, r, q, sigma)
    if is_call:
        return math.exp(-q * T) * norm_cdf(d1)
    return -math.exp(-q * T) * norm_cdf(-d1)


def strike_for_delta(S: float, T: float, r: float, q: float, sigma: float,
                     target_delta: float, is_call: bool = True) -> float:
    """Strike whose Black-Scholes delta equals ``target_delta`` (call: 0..1)."""
    if not is_call:
        raise NotImplementedError('only call strikes needed for PMCC')
    x = target_delta * math.exp(q * T)
    x = min(max(x, 1e-6), 1 - 1e-6)
    d1 = norm_ppf(x)
    return S * math.exp(-(d1 * sigma * math.sqrt(T)) + (r - q + 0.5 * sigma * sigma) * T)


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float,
                is_call: bool = True, lo: float = 0.01, hi: float = 3.0,
                tol: float = 1e-6) -> float | None:
    """Implied volatility by bisection; None if outside no-arbitrage bounds."""
    if T <= 0:
        return None
    p_lo, p_hi = (bs_price(S, K, T, r, q, s, is_call) for s in (lo, hi))
    if not (p_lo <= price <= p_hi):
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, q, mid, is_call) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def snap_strike(K: float) -> float:
    """Snap to a Euronext-style strike grid (interval widens with price)."""
    if K <= 0:
        raise ValueError('strike must be positive')
    if K < 10:
        step = 0.5
    elif K < 25:
        step = 1.0
    elif K < 50:
        step = 2.0
    elif K < 100:
        step = 2.5
    elif K < 250:
        step = 5.0
    else:
        step = 10.0
    return max(step, round(K / step) * step)


@dataclass(frozen=True)
class OptionSpec:
    strike: float
    expiry: dt.date
    is_call: bool = True


# ----------------------------------------------------------------- calendar
def third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def monthly_expiries(start: dt.date, end: dt.date) -> list[dt.date]:
    """All third-Friday expiries in [start, end]."""
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        e = third_friday(y, m)
        if start <= e <= end:
            out.append(e)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def next_expiry_with_dte(today: dt.date, min_dte: int, max_dte: int | None = None,
                         horizon_days: int = 1200) -> dt.date | None:
    """Earliest monthly expiry at least ``min_dte`` days out (and at most
    ``max_dte`` if given)."""
    for e in monthly_expiries(today, today + dt.timedelta(days=horizon_days)):
        dte = (e - today).days
        if dte >= min_dte and (max_dte is None or dte <= max_dte):
            return e
    return None

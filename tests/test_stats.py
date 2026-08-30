import numpy as np
import pandas as pd
import pytest
from conftest import simulate_ou

from pairs_trading import stats


def test_half_life_recovers_known_kappa():
    # kappa = 0.05/bar -> half-life = ln2/0.05 ~ 13.9 bars
    s = pd.Series(simulate_ou(20_000, kappa=0.05, seed=7))
    hl = stats.half_life(s)
    assert 11 < hl < 17


def test_half_life_infinite_for_random_walk():
    rng = np.random.default_rng(3)
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 5000)))
    hl = stats.half_life(walk)
    assert hl > 200 or np.isinf(hl)


def test_engle_granger_detects_simulated_cointegration(coint_pair):
    eg = stats.engle_granger(coint_pair['a'], coint_pair['b'], names=('a', 'b'))
    assert eg.pvalue < 0.01
    assert abs(eg.beta - 1.5) < 0.1
    assert eg.half_life < 30


def test_engle_granger_rejects_independent_walks(independent_pair):
    eg = stats.engle_granger(independent_pair['a'], independent_pair['b'])
    assert eg.pvalue > 0.05


def test_hurst_classifies_series_types():
    rng = np.random.default_rng(5)
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 5000)))
    ou = pd.Series(simulate_ou(5000, kappa=0.1, seed=6))
    # AR(1)-persistent increments -> trending
    inc = np.empty(5000)
    inc[0] = 0
    shocks = rng.normal(0, 1, 5000)
    for t in range(1, 5000):
        inc[t] = 0.8 * inc[t - 1] + shocks[t]
    trend = pd.Series(np.cumsum(inc))
    h_walk = stats.hurst_exponent(walk)
    h_ou = stats.hurst_exponent(ou)
    h_trend = stats.hurst_exponent(trend)
    assert 0.4 < h_walk < 0.6
    assert h_ou < 0.4
    assert h_trend > 0.6


def test_kalman_converges_to_static_beta(coint_pair):
    la, lb = np.log(coint_pair['a']), np.log(coint_pair['b'])
    kf = stats.KalmanHedge()
    out = kf.filter(la, lb)
    # After burn-in the filtered beta should hover near the true 1.5
    assert abs(out['beta'].iloc[-500:].mean() - 1.5) < 0.15


def test_kalman_tracks_drifting_beta():
    rng = np.random.default_rng(11)
    n = 3000
    lb = np.log(50) + np.cumsum(rng.normal(0, 0.01, n))
    beta_path = np.linspace(1.0, 2.0, n)
    la = 0.1 + beta_path * lb + rng.normal(0, 0.005, n)
    idx = pd.bdate_range('2010-01-01', periods=n)
    out = stats.KalmanHedge().filter(pd.Series(la, idx), pd.Series(lb, idx))
    est_late = out['beta'].iloc[-100:].mean()
    est_early = out['beta'].iloc[200:300].mean()
    assert est_late > est_early + 0.5  # must have followed the drift upward
    assert abs(est_late - 2.0) < 0.25


def test_rolling_beta_matches_global_on_static_relationship(coint_pair):
    la, lb = np.log(coint_pair['a']), np.log(coint_pair['b'])
    rb = stats.rolling_beta(la, lb, 252).dropna()
    assert abs(rb.mean() - 1.5) < 0.15


@pytest.mark.parametrize('kappa,lo,hi', [(0.2, 2, 5), (0.02, 25, 46)])
def test_half_life_scales_with_kappa(kappa, lo, hi):
    s = pd.Series(simulate_ou(30_000, kappa=kappa, seed=8))
    assert lo < stats.half_life(s) < hi


def test_kalman_alpha_can_be_pinned():
    rng = np.random.default_rng(21)
    n = 1000
    lb = pd.Series(np.log(50) + np.cumsum(rng.normal(0, 0.01, n)))
    la = 0.3 + 1.5 * lb + rng.normal(0, 0.005, n)
    out = stats.KalmanHedge(alpha0=0.3, beta0=1.5, alpha_drift=False,
                            delta=1e-5).filter(pd.Series(la), lb)
    assert (out['alpha'] == 0.3).all()
    assert abs(out['beta'].iloc[-200:].mean() - 1.5) < 0.1


def test_engle_granger_nan_and_short_input_yield_nan_pvalue():
    rng = np.random.default_rng(4)
    a = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, 100))) * 100)
    b = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, 100))) * 50)
    a.iloc[:75] = np.nan  # only 25 joint observations remain
    eg = stats.engle_granger(a, b)
    assert np.isnan(eg.pvalue)  # gate treats NaN as reject, never as pass
    # NaNs inside an otherwise long series are dropped pairwise and it runs
    a2, b2 = a.copy(), b.copy()
    a2.iloc[:] = np.exp(np.cumsum(rng.normal(0, 0.01, 100))) * 100
    a2.iloc[[5, 50]] = np.nan
    eg2 = stats.engle_granger(a2, b2)
    assert eg2.n_obs == 98 and np.isfinite(eg2.pvalue) and eg2.pvalue > 0


def test_half_life_uses_exact_ar1_formula():
    # phi = -0.5 -> AR coefficient 0.5 -> exact half-life exactly 1 bar
    s = pd.Series(simulate_ou(30_000, kappa=0.5, sigma=1.0, seed=2))
    hl = stats.half_life(s)
    assert 0.9 < hl < 1.1

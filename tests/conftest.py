import numpy as np
import pandas as pd
import pytest


def simulate_ou(n: int, kappa: float, sigma: float = 0.02, mu: float = 0.0,
                seed: int = 0) -> np.ndarray:
    """Discrete Ornstein-Uhlenbeck path: s_t = s_{t-1} + kappa*(mu - s_{t-1}) + eps."""
    rng = np.random.default_rng(seed)
    s = np.empty(n)
    s[0] = mu
    eps = rng.normal(0, sigma, n)
    for t in range(1, n):
        s[t] = s[t - 1] + kappa * (mu - s[t - 1]) + eps[t]
    return s


def simulate_cointegrated_pair(n: int = 1500, beta: float = 1.5, alpha: float = 0.3,
                               kappa: float = 0.05, spread_sigma: float = 0.01,
                               seed: int = 1) -> pd.DataFrame:
    """Pair where ln A = alpha + beta * ln B + OU spread (cointegrated by design)."""
    rng = np.random.default_rng(seed)
    log_b = np.log(50.0) + np.cumsum(rng.normal(0.0002, 0.015, n))
    spread = simulate_ou(n, kappa=kappa, sigma=spread_sigma, seed=seed + 100)
    log_a = alpha + beta * log_b + spread
    idx = pd.bdate_range('2015-01-01', periods=n)
    return pd.DataFrame({'a': np.exp(log_a), 'b': np.exp(log_b)}, index=idx)


def simulate_independent_walks(n: int = 1500, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    la = np.log(100.0) + np.cumsum(rng.normal(0, 0.015, n))
    lb = np.log(50.0) + np.cumsum(rng.normal(0, 0.015, n))
    idx = pd.bdate_range('2015-01-01', periods=n)
    return pd.DataFrame({'a': np.exp(la), 'b': np.exp(lb)}, index=idx)


@pytest.fixture
def coint_pair() -> pd.DataFrame:
    return simulate_cointegrated_pair()


@pytest.fixture
def independent_pair() -> pd.DataFrame:
    return simulate_independent_walks()


def simulate_drifting_beta_pair(n: int = 2000, seed: int = 3, b0: float = 1.0,
                                b1: float = 2.0) -> pd.DataFrame:
    """Cointegrated pair whose hedge ratio drifts linearly from b0 to b1."""
    rng = np.random.default_rng(seed)
    log_b = np.log(50.0) + np.cumsum(rng.normal(0.0002, 0.012, n))
    beta_path = np.linspace(b0, b1, n)
    spread = simulate_ou(n, kappa=0.05, sigma=0.01, seed=seed + 7)
    log_a = 0.2 + beta_path * log_b + spread
    idx = pd.bdate_range('2015-01-01', periods=n)
    return pd.DataFrame({'a': np.exp(log_a), 'b': np.exp(log_b)}, index=idx)

import numpy as np
import pandas as pd
import pytest

from pairs_trading import metrics


def test_sortino_uses_full_sample_downside_deviation():
    r = pd.Series([0.0] * 90 + [-0.01] * 5 + [0.02] * 5)
    dd = np.sqrt((np.minimum(r.to_numpy(), 0) ** 2).mean())
    expected = r.mean() / dd * np.sqrt(252)
    assert metrics.annualized_sortino(r, 252) == pytest.approx(expected)


def test_deflated_sharpe_penalizes_many_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1500))
    few = metrics.deflated_sharpe_ratio(r, np.array([0.01, 0.02]))
    sr_grid = rng.normal(0.02, 0.02, 200)
    many = metrics.deflated_sharpe_ratio(r, sr_grid)
    assert many < few

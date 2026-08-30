import numpy as np
import pandas as pd

from pairs_trading.config import SignalConfig
from pairs_trading.signals import generate_signals, zscore


def _z(values):
    return pd.Series(values, index=pd.bdate_range('2020-01-01', periods=len(values)),
                     dtype=float)


CFG = SignalConfig(entry=2.0, exit=0.0, stop=4.0, z_window=5)


def test_long_entry_and_convergence_exit():
    sigs = generate_signals(_z([0, -2.5, -1.0, 0.1, 0.0]), CFG)
    assert list(sigs['side']) == [0, 1, 1, 0, 0]
    assert sigs['reason'].iloc[3] == 'converged'


def test_short_entry_and_convergence_exit():
    sigs = generate_signals(_z([0, 2.5, 1.0, -0.1]), CFG)
    assert list(sigs['side']) == [0, -1, -1, 0]
    assert sigs['reason'].iloc[3] == 'converged'


def test_stop_loss_on_blowout():
    sigs = generate_signals(_z([0, -2.5, -4.5, -5.0]), CFG)
    assert list(sigs['side']) == [0, 1, 0, 0]
    assert sigs['reason'].iloc[2] == 'stopped'


def test_rearm_after_stop_requires_reversion_inside_band():
    # After the stop at -4.5, no entry while z stays beyond the entry band;
    # once z reverts inside (|z| < entry) the machine re-arms and may enter again.
    sigs = generate_signals(_z([0, -2.5, -4.5, -3.0, -1.0, -2.5]), CFG)
    assert list(sigs['side']) == [0, 1, 0, 0, 0, 1]


def test_nan_never_enters_and_holds_position():
    sigs = generate_signals(_z([np.nan, -2.5, np.nan, -1.0, 0.5]), CFG)
    assert list(sigs['side']) == [0, 1, 1, 1, 0]


def test_time_stop():
    cfg = SignalConfig(entry=2.0, exit=0.0, stop=10.0, z_window=5, max_holding=2)
    sigs = generate_signals(_z([0, -2.5, -2.4, -2.3, -2.2]), cfg)
    assert list(sigs['side']) == [0, 1, 1, 0, 0]
    assert sigs['reason'].iloc[3] == 'time'


def test_partial_exit_band():
    cfg = SignalConfig(entry=2.0, exit=0.5, stop=4.0, z_window=5)
    # long exits once z >= -0.5 (partial reversion), not only at 0
    sigs = generate_signals(_z([0, -2.5, -0.6, -0.4]), cfg)
    assert list(sigs['side']) == [0, 1, 1, 0]


def test_zscore_is_strictly_trailing():
    s = pd.Series(np.arange(10, dtype=float),
                  index=pd.bdate_range('2020-01-01', periods=10))
    z = zscore(s, 5)
    assert z.iloc[:4].isna().all()          # needs a full window
    # changing the future must not change the past z-values
    s2 = s.copy()
    s2.iloc[-1] = 1000
    z2 = zscore(s2, 5)
    pd.testing.assert_series_equal(z.iloc[:-1], z2.iloc[:-1])


def test_no_entry_beyond_stop_band():
    # A z-score already past the stop is a structural break, not an entry.
    sigs = generate_signals(_z([0.5, -5.0, -4.9, -1.0, -2.5]), CFG)
    assert list(sigs['side']) == [0, 0, 0, 0, 1]

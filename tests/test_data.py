import pandas as pd
import pytest

from pairs_trading import data
from pairs_trading.screening import screen_panel

pytestmark = pytest.mark.skipif(not data.VENDOR_DIR.exists(),
                                reason='vendored data not built')


def test_all_panels_load_and_are_daily():
    for cls in ['crypto', 'forex', 'commodities', 'stocks', 'stocks_long']:
        panel = data.load_panel(cls)
        assert isinstance(panel.index, pd.DatetimeIndex)
        assert panel.index.is_monotonic_increasing
        assert len(panel) > 1000
        nonpos = panel.stack().dropna() <= 0
        # The single legitimate non-positive price in the vendored data is the
        # famous negative WTI settlement of 2020-04-20 (-$36.98).
        if nonpos.any():
            assert cls == 'commodities'
            idx = nonpos[nonpos].index
            assert len(idx) == 1 and idx[0][0] == pd.Timestamp('2020-04-20')


def test_aligned_pair_excludes_negative_wti_day():
    df = data.aligned_pair('commodities', 'WTI', 'BRENT')
    assert pd.Timestamp('2020-04-20') not in df.index
    assert (df > 0).all().all()


def test_aligned_pair_inner_join():
    df = data.aligned_pair('crypto', 'BTC', 'ETH', start='2017-01-01')
    assert list(df.columns) == ['a', 'b']
    assert df.notna().all().all()
    assert df.index[0].year == 2017


def test_cross_asset_lookup():
    wti = data.get_series('cross', 'WTI')
    cad = data.get_series('cross', 'CAD')
    assert wti.index.min().year <= 1990
    assert 0.5 < cad.iloc[-1] < 1.5  # USD price of 1 CAD


def test_ohlc_for_backtesting_convention():
    ohlc = data.ohlc_for('KO')
    assert list(ohlc.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
    assert (ohlc['High'] >= ohlc['Low']).all()


def test_screen_small_universe_smoke():
    panel = data.load_panel('commodities')['2005':'2015']
    scr = screen_panel(panel, corr_min=0.3, min_obs=1000)
    assert {'y', 'x', 'eg_pvalue', 'half_life'} <= set(scr.columns)
    # WTI/Brent must appear and correlate strongly
    row = scr[(scr[['y', 'x']].isin(['WTI', 'BRENT']).all(axis=1))]
    assert len(row) == 1
    assert row.iloc[0]['corr'] > 0.4  # spot-assessment daily corr is modest

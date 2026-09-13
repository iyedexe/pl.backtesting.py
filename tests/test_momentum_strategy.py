import numpy as np
import pandas as pd
from momentum_strategy import (
    CASH,
    DAYS_PER_YEAR,
    MultiHorizonMomentum,
    book_equity,
    coin_tape,
    held_notional,
    index_stats,
    momentum_score,
    realized_vol,
    score_diagnostic,
    sleeve_backtest,
)

LOOKBACKS = (5, 10, 21, 42)


def gbm(n: int = 1500, mu: float = 0.0, sigma: float = 0.8, seed: int = 0) -> pd.Series:
    """Geometric random walk on calendar days (crypto never closes)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(mu / DAYS_PER_YEAR, sigma / np.sqrt(DAYS_PER_YEAR), n)
    idx = pd.date_range('2018-01-01', periods=n, freq='D')
    return pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx)


def test_score_is_the_sum_of_four_signs_and_causal():
    up = np.linspace(1.0, 2.0, 100)
    s = momentum_score(up, LOOKBACKS)
    assert np.isnan(s[:42]).all() and (s[42:] == 4).all()
    assert (momentum_score(up[::-1], LOOKBACKS)[42:] == -4).all()
    px = gbm(300).to_numpy()
    a = momentum_score(px, LOOKBACKS)
    assert set(np.unique(a[42:])) <= {-4, -2, 0, 2, 4}
    later = px.copy()
    later[200:] *= 3                      # the future must not change the past
    np.testing.assert_array_equal(a[:200], momentum_score(later, LOOKBACKS)[:200])


def test_realized_vol_recovers_the_simulated_vol():
    v = realized_vol(gbm(3000, sigma=0.8, seed=1).to_numpy(), 42)
    assert np.isnan(v[:42]).all()
    assert abs(np.nanmedian(v) - 0.8) < 0.1


def test_tape_opens_at_the_previous_close_and_orders_fill_there():
    tape = coin_tape(gbm(400, seed=2))
    assert (tape['Open'].to_numpy()[1:] == tape['Close'].to_numpy()[:-1]).all()
    assert (tape['High'] >= tape[['Open', 'Close']].max(axis=1)).all()
    assert (tape['Low'] <= tape[['Open', 'Close']].min(axis=1)).all()
    tr = sleeve_backtest(tape).run()['_trades']
    assert len(tr) > 5
    # decided on a close, filled at the next open — which is that same close
    np.testing.assert_allclose(tr['EntryPrice'],
                               tape['Close'].to_numpy()[tr['EntryBar'] - 1], rtol=1e-6)


def test_positions_are_sized_to_the_target_risk():
    tape = coin_tape(gbm(3000, sigma=0.8, seed=3))
    st = sleeve_backtest(tape, cost=0.0).run(long_only=True, hold_band=0.1, target_vol=0.10)
    eq = st['_equity_curve']['Equity']
    realized = np.log(eq).diff().std() * np.sqrt(DAYS_PER_YEAR)
    assert 0.07 < realized < 0.13                     # within 30% of the 10% target
    exposure = (held_notional(st['_trades'], tape) / eq).iloc[42:]
    assert 0.09 < exposure.mean() < 0.16              # about target / sigma = 0.125
    assert exposure.max() < 0.10 / MultiHorizonMomentum.vol_floor * 1.1


def test_hold_band_trades_less_and_long_only_never_shorts():
    tape = coin_tape(gbm(1500, seed=4))
    n = [sleeve_backtest(tape).run(hold_band=b)['# Trades'] for b in (0.0, 0.25, 1.0)]
    assert n[0] > n[1] > n[2] > 0
    assert (sleeve_backtest(tape).run(long_only=True)['_trades']['Size'] > 0).all()


def test_a_trend_is_ridden_long_then_short():
    n = 400
    up = 100 * np.exp(np.linspace(0, 1.0, n))
    down = up[-1] * np.exp(-np.linspace(0, 1.0, n))[1:]
    px = pd.Series(np.r_[up, down], index=pd.date_range('2018-01-01', periods=2 * n - 1, freq='D'))
    st = sleeve_backtest(coin_tape(px), cost=0.0).run(hold_band=1.0)
    tr = st['_trades']
    assert (tr['Size'].iloc[0] > 0) and (tr['Size'].iloc[-1] < 0)
    assert (tr['PnL'] > 0).all()
    assert st['Equity Final [$]'] > CASH


def test_book_sums_sleeve_pnl_on_one_capital_base():
    a = {'pnl': pd.Series([0.0, 10.0, 20.0, 30.0], index=pd.date_range('2020-01-01', periods=4))}
    b = {'pnl': pd.Series([0.0, -5.0, 5.0], index=pd.date_range('2020-01-02', periods=3))}
    assert book_equity([a, b]).tolist() == [CASH, CASH + 10, CASH + 15, CASH + 35]


def test_index_stats_reproduce_the_curve():
    idx = pd.date_range('2020-01-01', periods=366, freq='D')
    st = index_stats(pd.Series(CASH * np.exp(np.linspace(0, 0.5, 366)), index=idx))
    assert abs(st['Return [%]'] - (np.exp(0.5) - 1) * 100) < 0.01
    assert abs(st['Max. Drawdown [%]']) < 1e-6
    assert abs(st['CAGR [%]'] - (np.exp(0.5) - 1) * 100) < 0.5


def test_score_diagnostic_pools_every_coin_day():
    panel = pd.DataFrame({'BTC': gbm(800, seed=5), 'ETH': gbm(800, seed=6),
                          'XRP': gbm(800, seed=7)})
    by, ts = score_diagnostic(panel, LOOKBACKS, 42, start='2018-06-01')
    assert set(by.index) <= {-4, -2, 0, 2, 4}
    assert by['count'].sum() == ts['n'] < 3 * 800
    assert np.isfinite(ts['t']) and np.isfinite(ts['t_ex_btc'])

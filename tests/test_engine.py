import numpy as np
import pandas as pd
import pytest

from pairs_trading.config import CostModel, EngineConfig, SignalConfig, WalkForwardConfig
from pairs_trading.engine import backtest_pair
from pairs_trading.signals import generate_signals, zscore
from pairs_trading.walkforward import walk_forward_pair


def _prices(a, b):
    idx = pd.bdate_range('2021-01-01', periods=len(a))
    return pd.DataFrame({'a': a, 'b': b}, index=idx, dtype=float)


def _side(values, idx):
    return pd.Series(values, index=idx, dtype=int)


NOCOST = EngineConfig(cost=CostModel(0.0), execution_lag=1)


def test_hand_computed_pnl_long_spread():
    # Signal at t0, executed at t1 close (lag=1). Long spread, beta=1:
    # 0.5 dollars long A at 100, 0.5 dollars short B at 50.
    px = _prices([100, 100, 110, 110], [50, 50, 50, 50])
    res = backtest_pair(px, _side([1, 1, 1, 1], px.index), 1.0, NOCOST)
    # qa = 0.5/100 = 0.005 shares -> +0.005 * 10 = +0.05 on the A leg; B leg flat
    assert res.equity.iloc[2] == pytest.approx(1.05)
    assert res.trades.iloc[0]['pnl'] == pytest.approx(0.05)
    assert res.trades.iloc[0]['reason'] == 'end'


def test_short_leg_gains_when_b_falls():
    px = _prices([100, 100, 100, 100], [50, 50, 40, 40])
    res = backtest_pair(px, _side([1, 1, 1, 1], px.index), 1.0, NOCOST)
    # short 0.5$ of B at 50 -> qb = -0.01 shares -> pnl = -0.01 * (40-50) = +0.10
    assert res.equity.iloc[-1] == pytest.approx(1.10)


def test_dollar_neutral_at_entry():
    px = _prices([123, 123, 130], [47, 47, 45])
    res = backtest_pair(px, _side([1, 1, 1], px.index), 1.0, NOCOST)
    qa, qb = res.positions[['qa', 'qb']].iloc[1]
    assert qa * 123 == pytest.approx(0.5)
    assert qb * 47 == pytest.approx(-0.5)


def test_beta_scales_short_leg():
    px = _prices([100, 100, 100], [50, 50, 50])
    res = backtest_pair(px, _side([1, 1, 1], px.index), 3.0, NOCOST)
    qa, qb = res.positions[['qa', 'qb']].iloc[1]
    # N = 1/(1+3) = 0.25 long A, 0.75 short B; gross = 1.0 = leverage * equity
    assert qa * 100 == pytest.approx(0.25)
    assert qb * 50 == pytest.approx(-0.75)


def test_execution_lag_prices_the_trade_at_next_close():
    px = _prices([100, 120, 120, 120], [50, 50, 50, 50])
    res0 = backtest_pair(px, _side([1, 1, 1, 1], px.index), 1.0,
                         EngineConfig(cost=CostModel(0.0), execution_lag=0))
    res1 = backtest_pair(px, _side([1, 1, 1, 1], px.index), 1.0, NOCOST)
    # lag=0 buys A at 100 and catches the +20% move; lag=1 buys at 120 and misses it
    assert res0.equity.iloc[-1] > 1.09
    assert res1.equity.iloc[-1] == pytest.approx(1.0)


def test_signal_on_last_bar_never_executes_with_lag():
    px = _prices([100, 100, 100], [50, 50, 50])
    res = backtest_pair(px, _side([0, 0, 1], px.index), 1.0, NOCOST)
    assert len(res.trades) == 0
    assert (res.positions['side'] == 0).all()


def test_costs_charged_on_both_legs_at_entry_and_exit():
    bp = 10.0
    cfg = EngineConfig(cost=CostModel(bp), execution_lag=1)
    px = _prices([100, 100, 100, 100], [50, 50, 50, 50])
    res = backtest_pair(px, _side([1, 1, 0, 0], px.index), 1.0, cfg)
    rate = bp / 1e4
    # entry: rate * gross(=1.0); exit: rate * position notional (prices flat -> 1.0)
    expected = 1.0 - 2 * rate
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert res.trades.iloc[0]['pnl'] == pytest.approx(-2 * rate, rel=1e-6)


def test_flat_signal_flat_equity():
    px = _prices([100, 101, 99, 102], [50, 51, 49, 50])
    res = backtest_pair(px, _side([0, 0, 0, 0], px.index), 1.0, NOCOST)
    assert (res.equity == 1.0).all()


def test_degenerate_beta_skips_entry():
    px = _prices([100, 100, 100], [50, 50, 50])
    res = backtest_pair(px, _side([1, 1, 1], px.index), -2.0, NOCOST)
    assert res.skipped_entries >= 1
    assert len(res.trades) == 0


def test_engine_is_causal_wrt_future_prices():
    # Everything up to bar k identical => positions up to bar k identical,
    # regardless of what happens after k.
    a1 = [100, 100, 105, 110, 120]
    a2 = [100, 100, 105, 55, 44]
    px1 = _prices(a1, [50] * 5)
    px2 = _prices(a2, [50] * 5)
    side = _side([1, 1, 1, 1, 1], px1.index)
    r1 = backtest_pair(px1, side, 1.0, NOCOST)
    r2 = backtest_pair(px2, side, 1.0, NOCOST)
    pd.testing.assert_frame_equal(r1.positions.iloc[:3], r2.positions.iloc[:3])
    pd.testing.assert_series_equal(r1.equity.iloc[:3], r2.equity.iloc[:3])


def test_walk_forward_on_simulated_cointegrated_pair(coint_pair):
    wf = walk_forward_pair(coint_pair, SignalConfig(entry=1.5, z_window=30),
                           EngineConfig(cost=CostModel(2.0)),
                           WalkForwardConfig(formation=252, trading=63),
                           name='sim')
    assert len(wf.windows) > 10
    assert wf.gate_pass_rate > 0.5           # truly cointegrated: gate mostly open
    assert len(wf.trades) > 10
    # every trade lives inside one trading window (force-flat at boundaries)
    assert (wf.trades['holding'] <= 63).all()
    assert wf.equity.iloc[-1] > 1.0          # OU spread + tiny costs: profitable
    summary = summarize_result(wf)
    assert summary['sharpe'] > 1.0


def summarize_result(wf):
    from pairs_trading import metrics

    class _R:
        pass

    r = _R()
    r.name = wf.name
    r.returns = wf.returns
    r.equity = wf.equity
    r.trades = wf.trades
    r.positions = pd.DataFrame({'side': (wf.returns != 0).astype(int)})
    r.config = EngineConfig()
    return metrics.summarize(r, periods_per_year=252)


def test_walk_forward_gate_blocks_independent_walks(independent_pair):
    wf = walk_forward_pair(independent_pair, SignalConfig(),
                           EngineConfig(cost=CostModel(2.0)),
                           WalkForwardConfig(formation=252, trading=63),
                           name='indep')
    # Random walks should mostly fail the cointegration gate
    assert wf.gate_pass_rate < 0.4


def test_metrics_sane_on_synthetic_returns():
    from pairs_trading.metrics import annualized_sharpe, max_drawdown, newey_west_tstat
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 2520))
    assert 1.0 < annualized_sharpe(r, 252) < 2.2
    assert newey_west_tstat(r) > 2
    eq = (1 + r).cumprod()
    assert 0 < max_drawdown(eq) < 0.5


def test_zscore_signal_engine_roundtrip_no_nan(coint_pair):
    la = np.log(coint_pair['a'])
    lb = np.log(coint_pair['b'])
    spread = la - 1.5 * lb
    z = zscore(spread, 30)
    sigs = generate_signals(z, SignalConfig())
    res = backtest_pair(coint_pair, sigs['side'], 1.5,
                        EngineConfig(cost=CostModel(5.0)),
                        reasons=sigs['reason'])
    assert res.equity.notna().all()
    assert res.returns.notna().all()
    assert len(res.trades) > 0


def test_walk_forward_kalman_variant_trades(coint_pair):
    wf = walk_forward_pair(coint_pair, SignalConfig(entry=1.5, z_window=30),
                           EngineConfig(cost=CostModel(2.0)),
                           WalkForwardConfig(formation=252, trading=63,
                                             use_kalman=True),
                           name='sim-kalman')
    assert wf.gate_pass_rate > 0.4
    assert len(wf.trades) > 10
    assert wf.equity.iloc[-1] > 0.9   # comparable to OLS on a static-beta pair


def test_kalman_keeps_trading_through_beta_drift():
    # Under hedge-ratio drift the static-EG gate shuts; the continuous Kalman
    # filter keeps tracking and its gate stays open substantially more often.
    from conftest import simulate_drifting_beta_pair
    pair = simulate_drifting_beta_pair(seed=3)
    sig = SignalConfig(entry=1.5, z_window=30)
    eng = EngineConfig(cost=CostModel(2.0))
    wf_ols = walk_forward_pair(pair, sig, eng,
                               WalkForwardConfig(formation=252, trading=63))
    wf_kal = walk_forward_pair(pair, sig, eng,
                               WalkForwardConfig(formation=252, trading=63,
                                                 use_kalman=True))
    assert wf_kal.gate_pass_rate > wf_ols.gate_pass_rate + 0.2
    assert len(wf_kal.trades) > len(wf_ols.trades)

"""Cross-validation of the pairs strategy through the bundled backtesting.py.

`backtesting.py` is a single-instrument engine, so the pair is mapped onto one
synthetic instrument: the *price ratio* A/B. Trading the ratio with 100% of
equity approximates a dollar-neutral pair with hedge ratio beta = 1 in return
space (r_ratio = r_A - r_B + O(r^2)); it is not exactly the two-leg P&L (no
per-leg costs or beta hedging), which is why the two-leg engine in
`pairs_trading.engine` is the source of truth for headline results. The value
of this adapter is (a) an independent implementation of the same z-score rule,
executed by a widely used engine, to cross-check signal timing and the sign
and rough magnitude of returns, and (b) backtesting.py's interactive Bokeh
tearsheet for a featured pair.

Costs: `commission` on the synthetic instrument is set to 2x the per-side rate
because each synthetic transaction corresponds to trading two legs.
"""

from __future__ import annotations

import pandas as pd

from backtesting import Backtest, Strategy

from .config import SignalConfig


def ratio_ohlc(a: pd.DataFrame | pd.Series, b: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """OHLC frame of the ratio A/B for backtesting.py.

    With full OHLC legs, the ratio's open/close are exact; intrabar extremes of
    a ratio are not observable from daily bars, so High/Low are set to the
    envelope of open/close (conservative flat bars). Close-only inputs produce
    O=H=L=C bars.
    """
    if isinstance(a, pd.Series):
        df = pd.concat({'a': a, 'b': b}, axis=1).dropna()
        close = df['a'] / df['b']
        out = pd.DataFrame({'Open': close, 'High': close, 'Low': close,
                            'Close': close})
    else:
        df = a[['Open', 'Close']].join(b[['Open', 'Close']],
                                       lsuffix='_a', rsuffix='_b').dropna()
        o = df['Open_a'] / df['Open_b']
        c = df['Close_a'] / df['Close_b']
        out = pd.DataFrame({'Open': o, 'High': pd.concat([o, c], axis=1).max(axis=1),
                            'Low': pd.concat([o, c], axis=1).min(axis=1), 'Close': c})
    out.index.name = None
    return out


def make_strategy(sig_cfg: SignalConfig) -> type[Strategy]:
    """Build a backtesting.py Strategy class implementing the z-score rule."""

    class ZScorePairs(Strategy):
        entry = sig_cfg.entry
        exit_ = sig_cfg.exit
        stop = sig_cfg.stop
        z_window = sig_cfg.z_window

        def init(self):
            close = pd.Series(self.data.Close, index=self.data.index)

            def zscore(series, window):
                s = pd.Series(series)
                m = s.rolling(window, min_periods=window).mean()
                sd = s.rolling(window, min_periods=window).std(ddof=1)
                return ((s - m) / sd).to_numpy()

            self.z = self.I(zscore, close, self.z_window, name='z', overlay=False)
            self._armed = True

        def next(self):
            z = self.z[-1]
            if z != z:  # NaN
                return
            if not self.position:
                if abs(z) < self.entry:
                    self._armed = True
                if self._armed and z <= -self.entry:
                    self.buy()
                elif self._armed and z >= self.entry:
                    self.sell()
            elif self.position.is_long:
                if z <= -self.stop:
                    self.position.close()
                    self._armed = False
                elif z >= -self.exit_:
                    self.position.close()
                    self._armed = abs(z) < self.entry
            else:
                if z >= self.stop:
                    self.position.close()
                    self._armed = False
                elif z <= self.exit_:
                    self.position.close()
                    self._armed = abs(z) < self.entry

    return ZScorePairs


def run_backtestingpy(ohlc: pd.DataFrame, sig_cfg: SignalConfig,
                      per_side_bp: float, plot_path: str | None = None,
                      cash: float = 100_000):
    """Run the ratio strategy through backtesting.py; returns its stats Series.

    Orders are filled by backtesting.py at the *next bar's open* (its default),
    matching the two-leg engine's one-bar execution lag.
    """
    bt = Backtest(ohlc, make_strategy(sig_cfg), cash=cash,
                  commission=2 * per_side_bp / 1e4,
                  exclusive_orders=True, finalize_trades=True)
    stats = bt.run()
    if plot_path:
        bt.plot(filename=str(plot_path), open_browser=False, resample=False)
    return stats

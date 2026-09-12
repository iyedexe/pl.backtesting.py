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


# ---------------------------------------------------------------------------
# Walk-forward hedged-spread tape: the pair as ONE backtesting.py instrument
# ---------------------------------------------------------------------------

def build_walkforward_tape(prices: pd.DataFrame,
                           wf_cfg,
                           eng_cfg,
                           z_windows: tuple[int, ...] = (20, 30, 60, 90, 120),
                           ) -> pd.DataFrame:
    """Turn a pair into a single synthetic instrument for `backtesting.py`.

    For every walk-forward window the hedge ratio ``beta`` is estimated by OLS
    on log prices of the preceding *formation* window only. ``beta`` is
    therefore a *dollar* hedge ratio, so one synthetic unit is defined in
    window-normalized dollars: ``1/A_0`` shares of A against ``beta/B_0``
    shares of B (``A_0``, ``B_0`` = prices on the window's first bar), priced
    as ``P_t = A_t/A_0 - beta * B_t/B_0`` (plus a constant shift to keep
    prices positive). Holding ``N`` units is then exactly that two-leg
    position — its P&L is reproduced by construction — with gross notional
    ``N * Gross``, ``Gross = A_t/A_0 + beta * B_t/B_0`` (≈ ``1 + beta`` at the
    window start, the two-leg engine's own ``equity/(1+beta)`` sizing). The
    cost of a fill on ``N`` units is ``rate * N * Gross`` (see
    :class:`PerLegCommission`).

    Because ``beta`` changes at window boundaries the synthetic price jumps
    there; the strategy is therefore forced flat one bar before each boundary
    (``Exit`` column), so no trade ever spans two windows — the same splice
    discipline as the index-inclusion tape.

    Extra columns (all computed causally, formation data only, exactly as in
    :func:`pairs_trading.walkforward.walk_forward_pair`):

    - ``Z<w>``     rolling z-score of the spread for each window length ``w``
    - ``Gate``     1 if the window passed the formation-period Engle-Granger
                   p-value, half-life and hedge-ratio gates, else 0
    - ``Window``   walk-forward window id; ``Exit`` 1 on its last two bars
    - ``Gross``    gross notional of one unit, ``A + beta * B``; ``Beta``
    """
    import numpy as np

    from .signals import zscore
    from .stats import engle_granger, half_life, ols_hedge

    la = np.log(prices['a'].astype(float))
    lb = np.log(prices['b'].astype(float))
    a = prices['a'].to_numpy(float)
    b = prices['b'].to_numpy(float)
    n = len(prices)
    parts = []
    for k, start in enumerate(range(wf_cfg.formation, n, wf_cfg.trading)):
        end = min(start + wf_cfg.trading, n)
        if end - start < 5:
            break
        form, trade = slice(start - wf_cfg.formation, start), slice(start, end)
        alpha, beta = ols_hedge(la.iloc[form], lb.iloc[form])
        spread_all = pd.concat([la.iloc[form], la.iloc[trade]]) - alpha - beta * \
            pd.concat([lb.iloc[form], lb.iloc[trade]])
        n_form = start - (start - wf_cfg.formation)
        spread_f = spread_all.iloc[:n_form]
        eg = engle_granger(prices['a'].iloc[form], prices['b'].iloc[form],
                           try_both_orientations=False)
        hl = half_life(spread_f)
        gate = bool((eg.pvalue <= wf_cfg.coint_pvalue_gate)
                    and (wf_cfg.min_half_life <= hl <= wf_cfg.max_half_life)
                    and (eng_cfg.min_beta <= beta <= eng_cfg.max_beta)) if wf_cfg.gate else \
            bool(eng_cfg.min_beta <= beta <= eng_cfg.max_beta)
        part = pd.DataFrame(index=prices.index[trade])
        a_rel, b_rel = a[trade] / a[start], b[trade] / b[start]
        p_raw = a_rel - beta * b_rel
        part['Open'] = part['High'] = part['Low'] = part['Close'] = p_raw
        part['Gross'] = a_rel + beta * b_rel
        part['Beta'] = beta
        part['Gate'] = int(gate)
        part['Window'] = k
        exit_flag = np.zeros(end - start, dtype=int)
        exit_flag[-2:] = 1
        part['Exit'] = exit_flag
        for w in z_windows:
            part[f'Z{w}'] = zscore(spread_all, w).iloc[n_form:].to_numpy()
        parts.append(part)
    tape = pd.concat(parts)
    shift = max(0.0, -float(tape['Close'].min())) + 0.05 * float(tape['Gross'].median())
    for col in ('Open', 'High', 'Low', 'Close'):
        tape[col] = tape[col] + shift
    tape.attrs['price_shift'] = shift
    tape.index.name = None
    return tape


class PerLegCommission:
    """Commission callback for `backtesting.py` charging per-leg costs.

    `Backtest(commission=callable)` only receives ``(order_size, price)``; the
    gross notional of one synthetic unit lives in the tape's ``Gross`` column,
    so the strategy tells this object which bar the next fill lands on
    (``self.bar``) before placing an order.
    """

    def __init__(self, per_side_rate: float, gross: pd.Series | None = None):
        self.rate = per_side_rate
        self.gross = None if gross is None else gross.to_numpy(float)
        self.bar = 0

    def __call__(self, order_size: float, price: float) -> float:
        if self.gross is None:
            return self.rate * abs(order_size) * price
        i = min(self.bar, len(self.gross) - 1)
        return self.rate * abs(order_size) * float(self.gross[i])

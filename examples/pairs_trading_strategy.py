"""Pairs trading on backtesting.py — walk-forward, five asset classes.

The pair is fed to `backtesting.py` as ONE synthetic instrument: per
walk-forward window the hedged spread ``A - β·B`` (β fitted on the preceding
formation window only), so a position of N units is exactly long N·A / short
N·β·B and the two-leg P&L is reproduced by construction (see
``pairs_trading.btpy_adapter.build_walkforward_tape``). Per-leg costs come
through a commission callback, the z-score rule lives in ``Strategy.next()``,
and the framework's own optimizer sweeps the thresholds.

What it produces (``examples/figures``, ``examples/tables``):

- ``pairs_ranking.png``     out-of-sample Sharpe of every featured pair
- ``pairs_best_grid.png``   ``bt.optimize`` heatmap: entry threshold × z-window
- ``pairs_best_equity.png`` best configuration vs the literature default
- ``pairs_tearsheet.html``  the framework's interactive tearsheet of the best run

Usage::

    uv run python examples/pairs_trading_strategy.py [--quick]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from _common import SERIES, out_dirs, print_table, save_table

from backtesting import Backtest, Strategy
from pairs_trading import data, plotting
from pairs_trading.btpy_adapter import PerLegCommission, build_walkforward_tape
from pairs_trading.config import (
    DEFAULT_COSTS_BP,
    SignalConfig,
    WalkForwardConfig,
    engine_config_for,
)
from pairs_trading.experiments import FEATURED, STUDY_START
from pairs_trading.metrics import newey_west_tstat, summarize_walkforward
from pairs_trading.walkforward import walk_forward_pair

WF = WalkForwardConfig(formation=252, trading=63)
Z_WINDOWS = (20, 30, 60, 90, 120)
CASH = 100_000
COSTS = PerLegCommission(0.0)   # configured per pair before each Backtest


class PairsTrading(Strategy):
    """Z-score mean reversion on a walk-forward hedged-spread tape.

    Enter long the spread at z ≤ −entry (short at z ≥ +entry) while the
    window's cointegration gate is open and |z| is still inside the stop band;
    exit on reversion through ``exit_``; hard stop at |z| ≥ ``stop``; forced
    flat before every window boundary. After any exit the rule is disarmed
    until |z| re-enters the entry band, so a stop is never re-entered at once.
    Position size: ``leverage × equity`` of gross notional (A + β·B).
    """
    entry = 2.0
    exit_ = 0.0
    stop = 4.0
    z_window = 60
    leverage = 1.0

    def init(self):
        zcol = self.data.df[f'Z{self.z_window}'].to_numpy()
        self.z = self.I(lambda arr: arr, zcol, name=f'z-score ({self.z_window}d)',
                        overlay=False)
        self._armed = True
        self._window = -1
        self._cur_window = -1

    def next(self):
        COSTS.bar = len(self.data)              # the next fill lands on the next bar
        z = self.z[-1]
        window = int(self.data.Window[-1])
        if window != self._cur_window:          # fresh window: new β, new z history
            self._cur_window, self._armed = window, True
        if self.position:
            if self.data.Exit[-1] or window != self._window:
                self.position.close()           # force-flat before the β switch
                self._armed = abs(z) < self.entry if z == z else False
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
            return
        if z != z or not self.data.Gate[-1] or self.data.Exit[-1]:
            return
        if abs(z) < self.entry:
            self._armed = True
        if not self._armed:
            return
        units = int(self.equity * self.leverage / self.data.Gross[-1])
        if units < 1:
            return
        if -self.stop < z <= -self.entry:
            self.buy(size=units)
            self._window = window
        elif self.stop > z >= self.entry:
            self.sell(size=units)
            self._window = window


def backtest_pair(study: str, a: str, b: str, panel: str, **params):
    """Build the tape for one pair and run it through backtesting.py."""
    prices = data.aligned_pair(panel, a, b, start=STUDY_START.get(study))
    tape = build_walkforward_tape(prices, WF, engine_config_for(study), Z_WINDOWS)
    COSTS.rate = DEFAULT_COSTS_BP[study] / 1e4
    COSTS.gross = tape['Gross'].to_numpy(float)
    bt = Backtest(tape, PairsTrading, cash=CASH, commission=COSTS,
                  finalize_trades=True)
    return bt, bt.run(**params), prices


def main(quick: bool = False) -> None:
    fig_dir, _ = out_dirs()
    warnings.filterwarnings('ignore')
    featured = [(s, a, b, p) for s, lst in FEATURED.items() for a, b, p in lst]
    if quick:
        featured = [f for f in featured if f[1] in ('WTI', 'KO', 'BTC')]

    # 1) every featured pair under the default rule, ranked
    rows, runs = [], {}
    for study, a, b, panel in featured:
        try:
            bt, st, prices = backtest_pair(study, a, b, panel)
        except (KeyError, ValueError) as e:
            print(f'skip {a}/{b}: {e}')
            continue
        eq = st['_equity_curve']['Equity']
        rets = eq.pct_change().fillna(0.0)
        name = f'{a}/{b}'
        runs[name] = (study, a, b, panel, bt, st, prices)
        rows.append({'pair': name, 'class': study,
                     'Sharpe': float(st['Sharpe Ratio']),
                     'Return [%]': float(st['Return [%]']),
                     'CAGR [%]': float(st['CAGR [%]']),
                     'Max DD [%]': float(st['Max. Drawdown [%]']),
                     '# Trades': int(st['# Trades']),
                     'Win Rate [%]': float(st['Win Rate [%]']),
                     'Exposure [%]': float(st['Exposure Time [%]']),
                     'NW t-stat': newey_west_tstat(rets)})
    ranking = pd.DataFrame(rows).set_index('pair').sort_values('Sharpe', ascending=False)
    print_table(ranking, 'Featured pairs on backtesting.py — walk-forward, out of sample')
    save_table(ranking, 'pairs_ranking')

    classes = list(dict.fromkeys(ranking['class']))
    color_of = {c: SERIES[i % len(SERIES)] for i, c in enumerate(classes)}
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8))
    vals = ranking['Sharpe'].fillna(0.0)
    ax.barh(ranking.index[::-1], vals[::-1],
            color=[color_of[c] for c in ranking['class'][::-1]])
    for y, (name, row) in enumerate(ranking[::-1].iterrows()):
        label = (f' t={row["NW t-stat"]:.1f} / {row["# Trades"]} trades'
                 if row['# Trades'] else ' no trades (gate never opened)')
        ax.annotate(label, (max(float(vals[name]), 0), y), fontsize=7.5, va='center',
                    color=plotting.INK2)
    ax.axvline(0, color=plotting.AXIS, linewidth=0.8)
    ax.set_xlabel('out-of-sample Sharpe (backtesting.py, default 2σ rule, base costs); '
                  'labels: Newey-West t-stat / trades')
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_of[c]) for c in classes]
    ax.legend(handles, classes, loc='lower right', ncols=3, title='asset class')
    ax.set_title('Pairs trading: every featured pair, ranked', loc='left', fontweight='bold')
    fig.savefig(fig_dir / 'pairs_ranking.png', bbox_inches='tight')
    plt.close(fig)

    # 2) the most statistically robust pair (Newey-West t among pairs with ≥ 20 trades)
    pool = ranking[ranking['# Trades'] >= (5 if quick else 20)]
    pool = pool if len(pool) else ranking
    best_name = pool.sort_values('NW t-stat', ascending=False).index[0]
    study, a, b, panel, bt, st_default, prices = runs[best_name]
    print(f'\nBest pair by Newey-West t-stat: {best_name} ({study}); '
          f'cross-checking against the two-leg engine ...')

    # cross-check: same signals through pairs_trading's own two-leg engine
    wf = walk_forward_pair(prices, SignalConfig(), engine_config_for(study), WF, name=best_name)
    two_leg = summarize_walkforward(wf, 252)
    check = pd.DataFrame({
        'backtesting.py': {'Return [%]': float(st_default['Return [%]']),
                           'Sharpe': float(st_default['Sharpe Ratio']),
                           '# Trades': int(st_default['# Trades']),
                           'Win Rate [%]': float(st_default['Win Rate [%]'])},
        'two-leg engine': {'Return [%]': two_leg['total_return_pct'],
                           'Sharpe': two_leg['sharpe'], '# Trades': two_leg['n_trades'],
                           'Win Rate [%]': two_leg['win_rate_pct']}}).T
    print_table(check, f'{best_name}: framework vs two-leg engine, default rule')
    save_table(check, 'pairs_crosscheck')

    # 3) framework optimizer over the thresholds
    grid = {'entry': [1.5, 2.0], 'exit_': [0.0], 'z_window': [30, 60]} if quick else \
        {'entry': [1.0, 1.5, 2.0, 2.5, 3.0], 'exit_': [0.0, 0.5], 'z_window': list(Z_WINDOWS)}
    COSTS.rate = DEFAULT_COSTS_BP[study] / 1e4
    COSTS.gross = bt._data['Gross'].to_numpy(float)
    best_st, heat = bt.optimize(**grid, maximize='Sharpe Ratio', return_heatmap=True,
                                constraint=lambda p: p.exit_ < p.entry)
    heat = heat.dropna()
    top = heat.sort_values(ascending=False).head(10).rename('Sharpe').to_frame()
    print_table(top, f'{best_name}: top configurations (bt.optimize, Sharpe)')
    save_table(heat.rename('Sharpe').to_frame(), 'pairs_best_grid')
    h2 = heat.groupby(level=['entry', 'z_window']).max().unstack('z_window')
    h2.index.name, h2.columns.name = 'entry z', 'z-score window (bars)'
    plotting.sensitivity_heatmap(h2, f'{best_name}: OOS Sharpe by entry × z-window '
                                 '(bt.optimize, best exit)', fig_dir / 'pairs_best_grid.png')

    bp = best_st['_strategy']
    label = f'best: enter {bp.entry:.1f}σ / exit {bp.exit_:.1f} / window {bp.z_window}'
    plotting.equity_curves(
        {label: best_st['_equity_curve']['Equity'] / CASH,
         'default (2σ / 0 / 60)': st_default['_equity_curve']['Equity'] / CASH},
        f'{best_name} on backtesting.py: walk-forward equity (start = 1)',
        fig_dir / 'pairs_best_equity.png')
    bt.plot(filename=str(fig_dir / 'pairs_tearsheet.html'), open_browser=False,
            resample=False)
    print(f'\nBest configuration for {best_name}: {label} -> Sharpe '
          f'{float(best_st["Sharpe Ratio"]):.2f}, return {float(best_st["Return [%]"]):.1f}%, '
          f'max DD {float(best_st["Max. Drawdown [%]"]):.1f}%, {int(best_st["# Trades"])} trades'
          f' (default: Sharpe {float(st_default["Sharpe Ratio"]):.2f})')
    print(f'figures -> {fig_dir}')
    assert np.isfinite(float(best_st['Sharpe Ratio']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true', help='3 pairs, tiny grid (CI)')
    main(ap.parse_args().quick)

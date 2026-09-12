"""Index inclusion (the index effect) on backtesting.py.

Stocks about to be added to a rules-based index can be identified before the
announcement; index trackers then have to buy them. On the tutorial's
synthetic point-in-time market (220 stocks, an FTSE-100-style rulebook, a
planted ~5% index effect — see ``index_inclusion``), the look-ahead-free
screener stitches the candidates into one tape and ``IndexInclusion`` trades
it one stock at a time. The framework's optimizer sweeps the holding period
for each screener setting, and the best configuration is re-run on fresh
market seeds to see whether it generalizes.

Outputs (``examples/figures``, ``examples/tables``):

- ``index_best_grid.png``   Sharpe by holding period × prediction buffer (bt.optimize)
- ``index_best_equity.png`` best configuration vs the tutorial default
- ``index_seeds.png``       best vs default across fresh market seeds
- ``index_tearsheet.html``  the framework's interactive tearsheet of the best run

Usage::

    uv run python examples/index_inclusion_strategy.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import AXIS, SERIES, out_dirs, print_table, save_table

from backtesting import Backtest, Strategy
from index_inclusion import Market, build_tape, find_windows, make_market
from pairs_trading import plotting

CASH = 100_000
COMMISSION = .002
DEFAULT = {'buffer': 2, 'pre_cutoff': 10, 'hold_limit': 21}


class IndexInclusion(Strategy):
    """Buy at Monday's open when a window switched ``Signal`` on at the Friday
    close; exit when the window ends or after ``hold_limit`` sessions."""

    hold_limit = 21                    # max sessions in a position (~1 calendar month)

    def init(self):
        self._last_window = -1

    def next(self):
        if self.position:
            held = len(self.data) - 1 - self.trades[-1].entry_bar
            if not self.data.Signal[-1] or held + 1 >= self.hold_limit:
                self.position.close()
        elif self.data.Signal[-1]:
            window = int(self.data.Window[-1])
            if window != self._last_window:
                self._last_window = window
                self.buy()


def make_backtest(market: Market, buffer: int, pre_cutoff: int) -> Backtest:
    windows = find_windows(market, buffer=buffer, pre_cutoff=pre_cutoff)
    tape, _ = build_tape(market, windows)
    return Backtest(tape, IndexInclusion, cash=CASH, commission=COMMISSION,
                    finalize_trades=True)


def run_backtest(market: Market, *, buffer: int = 2, pre_cutoff: int = 10,
                 hold_limit: int = 21) -> pd.Series:
    """Screen -> tape -> Backtest for one parameter set; returns the stats."""
    return make_backtest(market, buffer, pre_cutoff).run(hold_limit=hold_limit)


def main(quick: bool = False) -> None:
    fig_dir, _ = out_dirs()
    warnings.filterwarnings('ignore')
    market = make_market()
    holds = [11, 21] if quick else list(range(5, 27, 2))
    buffers = [0, 4] if quick else [0, 2, 4, 6, 8]
    pres = [10] if quick else [5, 10, 15]

    # screener settings outside, the framework optimizer over the holding period inside
    cells = []
    for b, p in itertools.product(buffers, pres):
        bt = make_backtest(market, b, p)
        _, heat = bt.optimize(hold_limit=holds, maximize='Sharpe Ratio', return_heatmap=True)
        for h, sharpe in heat.dropna().items():
            h = h[0] if isinstance(h, tuple) else h    # 1-param heatmaps key by 1-tuples
            cells.append({'hold_limit': int(h), 'buffer': b, 'pre_cutoff': p,
                          'Sharpe': float(sharpe)})
    grid = pd.DataFrame(cells).sort_values('Sharpe', ascending=False)
    print_table(grid.head(10).set_index(['hold_limit', 'buffer', 'pre_cutoff']),
                'Top configurations by Sharpe (bt.optimize over hold_limit per screen)')
    save_table(grid, 'index_best_grid')
    best = grid.iloc[0]
    cfg_best = {k: int(best[k]) for k in ('buffer', 'pre_cutoff', 'hold_limit')}

    h2 = grid.groupby(['hold_limit', 'buffer'])['Sharpe'].max().unstack('buffer')
    h2.index.name = 'max holding period (sessions)'
    h2.columns.name = 'prediction buffer (ranks)'
    plotting.sensitivity_heatmap(h2, 'Index inclusion: Sharpe by holding period × buffer '
                                 '(bt.optimize, best pre-cutoff)', fig_dir / 'index_best_grid.png',
                                 value_label='Sharpe ratio')

    bt_best = make_backtest(market, cfg_best['buffer'], cfg_best['pre_cutoff'])
    st_best = bt_best.run(hold_limit=cfg_best['hold_limit'])
    st_default = run_backtest(market, **DEFAULT)
    label = (f'best: hold ≤{cfg_best["hold_limit"]} / buffer {cfg_best["buffer"]} / '
             f'hunt {cfg_best["pre_cutoff"]}d before cutoff')
    plotting.equity_curves(
        {label: st_best['_equity_curve']['Equity'] / CASH,
         'tutorial default (21 / 2 / 10)': st_default['_equity_curve']['Equity'] / CASH},
        'Index inclusion on backtesting.py, synthetic SIX 100 market: equity (start = 1)',
        fig_dir / 'index_best_equity.png', logy=True)
    bt_best.plot(filename=str(fig_dir / 'index_tearsheet.html'), open_browser=False,
                 resample=False)

    # generalization: best vs default on fresh market seeds
    seeds = [11, 12] if quick else [11, 12, 13, 14, 15, 16]
    seed_rows = []
    for seed in seeds:
        m = make_market(seed=seed)
        for name, cfg in (('best', cfg_best), ('default', DEFAULT)):
            st = run_backtest(m, **cfg)
            seed_rows.append({'seed': seed, 'config': name,
                              'Sharpe Ratio': float(st['Sharpe Ratio']),
                              'Return (Ann.) [%]': float(st['Return (Ann.) [%]']),
                              '# Trades': float(st['# Trades'])})
    seeds_df = pd.DataFrame(seed_rows)
    piv = seeds_df.pivot(index='seed', columns='config', values='Sharpe Ratio')
    print_table(piv, 'Sharpe of best vs default configuration on fresh market seeds')
    save_table(seeds_df, 'index_seeds')
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(piv))
    ax.bar(x - 0.18, piv['best'], width=0.36, color=SERIES[0], label='best configuration')
    ax.bar(x + 0.18, piv['default'], width=0.36, color=SERIES[1], label='tutorial default')
    ax.set_xticks(x, [f'seed {s}' for s in piv.index])
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel('Sharpe ratio')
    ax.legend(loc='upper right', ncols=2)
    ax.set_title('Does the best configuration generalize to other market draws?',
                 loc='left', fontweight='bold')
    fig.savefig(fig_dir / 'index_seeds.png', bbox_inches='tight')
    plt.close(fig)

    print(f'\nBest configuration: {label} -> Sharpe {float(st_best["Sharpe Ratio"]):.2f}, '
          f'ann. return {float(st_best["Return (Ann.) [%]"]):.1f}%, '
          f'{int(st_best["# Trades"])} trades, win rate {float(st_best["Win Rate [%]"]):.0f}% '
          f'(default Sharpe {float(st_default["Sharpe Ratio"]):.2f}); mean over seeds: '
          f'best {piv["best"].mean():.2f} vs default {piv["default"].mean():.2f}')
    print(f'figures -> {fig_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true', help='small sweep for CI')
    main(ap.parse_args().quick)

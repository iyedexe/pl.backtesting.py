"""Index inclusion (trading the index effect) — best-performing configuration.

Sweeps the three knobs of the strategy on the synthetic point-in-time market
of the tutorial (`doc/examples/Index Inclusion Strategy.py`): how many
sessions before a review cutoff to start hunting (``pre_cutoff``), how deep
inside the entry band a candidate must rank (``buffer``), and the maximum
holding period (``hold_limit``). The best configuration is then re-run on
fresh market seeds to show whether it generalizes or was fitted to one
draw of the simulator.

Outputs (``examples/figures/index_*.png``, ``examples/tables/index_*.csv``):

- ``index_best_grid.png``   Sharpe heatmap over hold limit × prediction buffer
- ``index_best_equity.png`` equity: best configuration vs tutorial default
- ``index_seeds.png``       best vs default configuration across market seeds

Usage::

    uv run python examples/run_index_inclusion.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import AXIS, SERIES, out_dirs, print_table, save_table

from index_inclusion import make_market, run_backtest
from pairs_trading import plotting

DEFAULT = {'buffer': 2, 'pre_cutoff': 10, 'hold_limit': 21}
KEYS = ['Return [%]', 'Return (Ann.) [%]', 'Sharpe Ratio', 'Max. Drawdown [%]',
        '# Trades', 'Win Rate [%]', 'Avg. Trade [%]', 'Exposure Time [%]']


def main(quick: bool = False) -> None:
    fig_dir, _ = out_dirs()
    market = make_market()
    holds = [11, 21] if quick else list(range(5, 27, 2))
    buffers = [0, 4] if quick else [0, 2, 4, 6, 8]
    pres = [10] if quick else [5, 10, 15]

    rows = []
    for h, b, p in itertools.product(holds, buffers, pres):
        st = run_backtest(market, buffer=b, pre_cutoff=p, hold_limit=h)
        rows.append({'hold_limit': h, 'buffer': b, 'pre_cutoff': p,
                     **{k: float(st[k]) for k in KEYS}, '_equity': st['_equity_curve']['Equity']})
    grid = pd.DataFrame(rows)
    grid_view = grid.drop(columns='_equity').sort_values('Sharpe Ratio', ascending=False)
    print_table(grid_view.head(10).set_index(['hold_limit', 'buffer', 'pre_cutoff']),
                'Top configurations by Sharpe (synthetic SIX 100 market, seed 11)')
    save_table(grid.drop(columns='_equity'), 'index_best_grid')
    best = grid.sort_values('Sharpe Ratio', ascending=False).iloc[0]

    heat = grid.groupby(['hold_limit', 'buffer'])['Sharpe Ratio'].max().unstack('buffer')
    heat.index.name = 'max holding period (sessions)'
    heat.columns.name = 'prediction buffer (ranks)'
    plotting.sensitivity_heatmap(heat, 'Index inclusion: Sharpe by holding period × buffer '
                                 '(best pre-cutoff)', fig_dir / 'index_best_grid.png',
                                 value_label='Sharpe ratio')

    default_st = run_backtest(market, **DEFAULT)
    label = (f'best: hold ≤{int(best["hold_limit"])} / buffer {int(best["buffer"])} / '
             f'hunt {int(best["pre_cutoff"])}d before cutoff')
    plotting.equity_curves(
        {label: best['_equity'] / 100_000,
         'tutorial default (21 / 2 / 10)': default_st['_equity_curve']['Equity'] / 100_000},
        'Index inclusion on the synthetic SIX 100 market: equity (start = 1)',
        fig_dir / 'index_best_equity.png', logy=True)

    # generalization: re-run best and default on fresh market seeds
    seeds = [11, 12] if quick else [11, 12, 13, 14, 15, 16]
    seed_rows = []
    for seed in seeds:
        m = make_market(seed=seed)
        for name, cfg in (('best', {'buffer': int(best['buffer']),
                                    'pre_cutoff': int(best['pre_cutoff']),
                                    'hold_limit': int(best['hold_limit'])}),
                          ('default', DEFAULT)):
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

    print(f'\nBest configuration: {label} -> Sharpe {best["Sharpe Ratio"]:.2f}, '
          f'ann. return {best["Return (Ann.) [%]"]:.1f}%, {int(best["# Trades"])} trades, '
          f'win rate {best["Win Rate [%]"]:.0f}% (default Sharpe '
          f'{float(default_st["Sharpe Ratio"]):.2f}); mean over seeds: best '
          f'{piv["best"].mean():.2f} vs default {piv["default"].mean():.2f}')
    print(f'figures -> {fig_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true', help='small sweep for CI')
    main(ap.parse_args().quick)

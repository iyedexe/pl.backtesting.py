"""Pairs trading — find and plot the best-performing configuration.

Two questions, answered strictly out of sample (walk-forward, next-close
execution, per-leg costs):

1. Across the five asset classes, which featured pair performs best under the
   default rule (enter 2σ / exit 0 / stop 4σ, 252/63-day windows)?
2. For that pair, which entry/exit/z-window configuration is best — and does
   it beat the literature default?

Outputs (``examples/figures/pairs_*.png``, ``examples/tables/pairs_*.csv``):

- ``pairs_ranking.png``   OOS Sharpe of every featured pair, colored by class
- ``pairs_best_grid.png`` Sharpe heatmap over entry threshold × z-window
- ``pairs_best_equity.png`` OOS equity: best configuration vs default vs no gate

Usage::

    uv run python examples/run_pairs_trading.py [--quick]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import SERIES, out_dirs, print_table, save_table

from pairs_trading import data, plotting
from pairs_trading.config import (
    PERIODS_PER_YEAR,
    SignalConfig,
    WalkForwardConfig,
    engine_config_for,
)
from pairs_trading.experiments import FEATURED, STUDY_START
from pairs_trading.metrics import summarize_walkforward
from pairs_trading.walkforward import walk_forward_pair

DEFAULT_SIG = SignalConfig(entry=2.0, exit=0.0, stop=4.0, z_window=60)
WF = WalkForwardConfig(formation=252, trading=63)


def run_pair(study: str, a: str, b: str, panel: str, sig: SignalConfig,
             wf: WalkForwardConfig = WF):
    prices = data.aligned_pair(panel, a, b, start=STUDY_START.get(study))
    res = walk_forward_pair(prices, sig, engine_config_for(study), wf, name=f'{a}/{b}')
    return res, summarize_walkforward(res, PERIODS_PER_YEAR[study])


def main(quick: bool = False) -> None:
    fig_dir, _ = out_dirs()

    # 1) rank every featured pair under the default rule
    featured = [(s, a, b, p) for s, lst in FEATURED.items() for a, b, p in lst]
    if quick:
        featured = [f for f in featured if f[1] in ('WTI', 'KO', 'BTC')]
    rows, results = [], {}
    for study, a, b, panel in featured:
        try:
            res, summ = run_pair(study, a, b, panel, DEFAULT_SIG)
        except (KeyError, ValueError) as e:
            print(f'skip {a}/{b}: {e}')
            continue
        results[f'{a}/{b}'] = (study, a, b, panel, res)
        rows.append(summ.rename(f'{a}/{b}').to_frame().T.assign(**{'class': study}))
    ranking = pd.concat(rows).sort_values('sharpe', ascending=False)
    cols = ['class', 'sharpe', 'cagr_pct', 'max_dd_pct', 'n_trades', 'win_rate_pct',
            'avg_trade_bp', 'nw_tstat', 'gate_pass_pct', 'years']
    print_table(ranking[cols], 'Featured pairs, walk-forward out-of-sample (default rule)')
    save_table(ranking[cols], 'pairs_ranking')

    classes = list(dict.fromkeys(ranking['class']))
    color_of = {c: SERIES[i % len(SERIES)] for i, c in enumerate(classes)}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    vals = ranking['sharpe'].fillna(0.0)
    ax.barh(ranking.index[::-1], vals[::-1],
            color=[color_of[c] for c in ranking['class'][::-1]])
    for y, (name, row) in enumerate(ranking[::-1].iterrows()):
        ax.annotate(f' t={row["nw_tstat"]:.1f} / {int(row["n_trades"])}',
                    (max(float(vals[name]), 0), y), fontsize=7.5, va='center',
                    color=plotting.INK2)
    ax.axvline(0, color=plotting.AXIS, linewidth=0.8)
    ax.set_xlabel('out-of-sample annualized Sharpe (default 2σ rule, base costs); '
                  'bar labels: Newey-West t-stat / trades')
    for c in classes:
        ax.barh([], [], color=color_of[c], label=c)
    ax.legend(loc='lower right', ncols=3, title='asset class')
    ax.set_title('Pairs trading: every featured pair, ranked', loc='left', fontweight='bold')
    fig.savefig(fig_dir / 'pairs_ranking.png', bbox_inches='tight')
    plt.close(fig)

    # 2) sweep the best pair's configuration. "Best" = highest Newey-West
    # t-statistic of the out-of-sample returns (Sharpe scaled by track length)
    # among pairs with a meaningful trade count: raw Sharpe would pick a
    # 21-trade fluke over a 79-trade, t > 3 relationship.
    eligible = ranking[ranking['n_trades'] >= (5 if quick else 20)]
    pool = eligible if len(eligible) else ranking
    best_name = pool.sort_values('nw_tstat', ascending=False).index[0]
    study, a, b, panel, default_res = results[best_name]
    print(f'\nBest pair by Newey-West t-stat: {best_name} ({study}, '
          f't = {ranking.loc[best_name, "nw_tstat"]:.2f}, '
          f'{int(ranking.loc[best_name, "n_trades"])} trades) — sweeping thresholds')
    entries = [1.5, 2.0] if quick else [1.0, 1.5, 2.0, 2.5, 3.0]
    exits = [0.0] if quick else [0.0, 0.5]
    windows = [30, 60] if quick else [20, 30, 60, 90, 120]
    grid_rows = []
    for e_ in entries:
        for x_ in exits:
            for w_ in windows:
                sig = SignalConfig(entry=e_, exit=x_, stop=max(4.0, e_ + 1.5), z_window=w_)
                res, summ = run_pair(study, a, b, panel, sig)
                grid_rows.append({'entry': e_, 'exit': x_, 'z_window': w_,
                                  'sharpe': summ['sharpe'], 'cagr_pct': summ['cagr_pct'],
                                  'max_dd_pct': summ['max_dd_pct'],
                                  'n_trades': summ['n_trades'], '_res': res})
    grid = pd.DataFrame(grid_rows)
    best = grid.sort_values('sharpe', ascending=False).iloc[0]
    print_table(grid.drop(columns='_res').sort_values('sharpe', ascending=False).head(10)
                .set_index(['entry', 'exit', 'z_window']),
                f'{best_name}: top configurations by OOS Sharpe')
    save_table(grid.drop(columns='_res'), 'pairs_best_grid')

    heat = (grid.groupby(['entry', 'z_window'])['sharpe'].max().unstack('z_window'))
    heat.index.name, heat.columns.name = 'entry z', 'z-score window (bars)'
    plotting.sensitivity_heatmap(
        heat, f'{best_name}: OOS Sharpe by entry threshold × z-window (best exit)',
        fig_dir / 'pairs_best_grid.png')

    no_gate, _ = run_pair(study, a, b, panel, DEFAULT_SIG, replace(WF, gate=False))
    label = (f'best: enter {best["entry"]:.1f}σ / exit {best["exit"]:.1f} / '
             f'window {int(best["z_window"])}')
    plotting.equity_curves(
        {label: best['_res'].equity, 'default (2σ / 0 / 60)': default_res.equity,
         'default, no cointegration gate': no_gate.equity},
        f'{best_name}: walk-forward out-of-sample equity',
        fig_dir / 'pairs_best_equity.png')
    print(f'\nBest configuration for {best_name}: {label} -> Sharpe {best["sharpe"]:.2f}, '
          f'CAGR {best["cagr_pct"]:.2f}%, max DD {best["max_dd_pct"]:.1f}%, '
          f'{int(best["n_trades"])} trades')
    print(f'figures -> {fig_dir}')
    assert np.isfinite(best['sharpe'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true', help='small sweep for CI')
    main(ap.parse_args().quick)

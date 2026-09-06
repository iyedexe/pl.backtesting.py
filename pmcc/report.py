"""
Generate result charts (PNG) and ``RESULTS.md`` from the CSVs produced by
``python -m pmcc.run_backtest``.  Charts follow a validated light-mode
palette: PMCC blue, CoveredCall orange, BuyHold aqua, leveraged PMCC yellow.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import data as datamod  # noqa: E402
from .stats import drawdown_series  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

COLORS = {'PMCC': '#2a78d6', 'CoveredCall': '#eb6834', 'BuyHold': '#1baf7a',
          'PMCC_budget': '#eda100', 'LeapsOnly': '#e87ba4'}
LABELS = {'PMCC': 'PMCC', 'CoveredCall': 'Covered call', 'BuyHold': 'Buy & hold',
          'PMCC_budget': 'PMCC (leveraged)', 'LeapsOnly': 'LEAPS only'}
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, BASE, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'


def _style(ax, title=''):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc='left', pad=10)


def fig_portfolio_curves():
    port = pd.read_csv(os.path.join(RESULTS_DIR, 'portfolio_curves.csv'),
                       index_col=0, parse_dates=True)
    series = ['PMCC', 'CoveredCall', 'BuyHold', 'PMCC_budget']
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.4), dpi=150,
                                  height_ratios=[2.4, 1], sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for s in series:
        ax.plot(port.index, port[s], color=COLORS[s], lw=1.6, label=LABELS[s])
        ax.annotate(LABELS[s], (port.index[-1], port[s].iloc[-1]),
                    xytext=(6, 0), textcoords='offset points',
                    color=COLORS[s], fontsize=8, va='center', fontweight='bold')
        dd = drawdown_series(port[s])
        ax2.plot(dd.index, 100 * dd, color=COLORS[s], lw=1.2)
    ax.set_yscale('log')
    ax.set_yticks([100e3, 200e3, 400e3, 800e3, 1600e3])
    ax.set_yticklabels(['100k', '200k', '400k', '800k', '1.6M'])
    _style(ax, 'Equal-weight portfolio, 10 French large caps, 2000-2015 '
               '(EUR 100k start, log scale)')
    _style(ax2, 'Drawdown (%)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper left')
    ax.margins(x=0.10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'portfolio_curves.png'),
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def fig_per_ticker_cagr():
    t = pd.read_csv(os.path.join(RESULTS_DIR, 'headline_stats.csv'))
    piv = t.pivot_table(index='ticker', columns='strategy', values='CAGR [%]')
    piv = piv.sort_values('BuyHold')
    strategies = ['BuyHold', 'CoveredCall', 'PMCC']
    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(piv))
    for i, tk in enumerate(piv.index):
        vals = [piv.loc[tk, s] for s in strategies]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=1.4, zorder=1)
    for s in strategies:
        ax.scatter(piv[s], y, s=46, color=COLORS[s], zorder=3,
                   edgecolors=SURFACE, linewidths=1.6, label=LABELS[s])
    names = [f"{datamod.UNIVERSE[t_].name}" for t_ in piv.index]
    ax.set_yticks(y, names)
    ax.tick_params(axis='y', colors=INK2, labelsize=8.5)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_xlabel('CAGR 2000-2015 (%, net of costs)', color=INK2, fontsize=9)
    _style(ax, 'Per-stock outcome: PMCC vs covered call vs buy & hold')
    ax.grid(axis='y', visible=False)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'per_ticker_cagr.png'),
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def fig_risk_return():
    t = pd.read_csv(os.path.join(RESULTS_DIR, 'headline_stats.csv'))
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for s in ['BuyHold', 'CoveredCall', 'PMCC']:
        d = t[t.strategy == s]
        ax.scatter(-d['Max Drawdown [%]'], d['CAGR [%]'], s=42, color=COLORS[s],
                   edgecolors=SURFACE, linewidths=1.4, label=LABELS[s], zorder=3)
    for s in ['BuyHold', 'CoveredCall', 'PMCC']:
        d = t[t.strategy == s]
        ax.scatter(-d['Max Drawdown [%]'].median(), d['CAGR [%]'].median(),
                   s=230, marker='X', color=COLORS[s], edgecolors=INK,
                   linewidths=0.8, zorder=4)
    ax.annotate('X = median stock', (0.985, 0.03), xycoords='axes fraction',
                ha='right', color=MUTED, fontsize=8)
    ax.set_xlabel('Max drawdown (%, deeper ->)', color=INK2, fontsize=9)
    ax.set_ylabel('CAGR (%)', color=INK2, fontsize=9)
    _style(ax, 'Risk vs return, 17 French large caps, 2000-2015')
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'risk_return.png'),
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def fig_sensitivity_premium():
    t = pd.read_csv(os.path.join(RESULTS_DIR, 'sensitivity_stats.csv'))
    g = t[(t.sweep == 'grid') & (t['skew'] == 0.10) & (t['spread'] == 0.02)
          & (t['p_short_target_delta'] == 0.25)]
    order = ['1.0', '1.2', 'vix']
    xt = {m: i for i, m in enumerate(order)}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, strat in zip(axes, ['PMCC', 'CoveredCall']):
        d = g[g.strategy == strat]
        for tk in d.ticker.unique():
            dt_ = d[d.ticker == tk].set_index('prem')['CAGR [%]']
            xs = [xt[str(m)] for m in dt_.index]
            ys = [dt_[m] for m in dt_.index]
            xs, ys = zip(*sorted(zip(xs, ys)))
            ax.plot(xs, ys, color=GRID, lw=1.2, zorder=1)
            ax.scatter(xs, ys, s=26, color=COLORS[strat],
                       edgecolors=SURFACE, linewidths=1.2, zorder=3)
            ax.annotate(tk.replace('.PA', ''), (xs[-1], ys[-1]), xytext=(7, 0),
                        textcoords='offset points', color=MUTED, fontsize=7.5,
                        va='center')
        ax.set_xticks(range(3), ['none\n(prem=1.0)', 'moderate\n(prem=1.2)',
                                 'empirical\n(VIX-shaped)'])
        ax.margins(x=0.18)
        _style(ax, LABELS[strat])
        ax.axhline(0, color=BASE, lw=1)
    axes[0].set_ylabel('CAGR (%)', color=INK2, fontsize=9)
    fig.suptitle('Everything rides on the volatility risk premium '
                 '(CAGR vs premium assumption)', color=INK, fontsize=10, x=0.02,
                 ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(RESULTS_DIR, 'sensitivity_premium.png'),
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def build_results_md():
    h = pd.read_csv(os.path.join(RESULTS_DIR, 'headline_stats.csv'))
    p = pd.read_csv(os.path.join(RESULTS_DIR, 'portfolio_stats.csv'), index_col=0)
    lines = ['# PMCC backtest results (auto-generated)', '',
             'Run `python -m pmcc.run_backtest all` to regenerate.', '',
             '## Equal-weight portfolio (10 liquid names, 2000-2015)', '',
             p.to_markdown(), '',
             '## Per-ticker headline stats', '']
    cols = ['ticker', 'strategy', 'CAGR [%]', 'Vol (ann.) [%]', 'Sharpe',
            'Max Drawdown [%]', 'Worst Year [%]', 'dividends',
            'short_premium_received', 'n_assignments', 'min_cash']
    lines += [h[cols].to_markdown(index=False), '']
    with open(os.path.join(RESULTS_DIR, 'RESULTS.md'), 'w') as f:
        f.write('\n'.join(lines))


def build():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig_portfolio_curves()
    fig_per_ticker_cagr()
    fig_risk_return()
    fig_sensitivity_premium()
    build_results_md()
    print('report artifacts written to', RESULTS_DIR)


if __name__ == '__main__':
    build()

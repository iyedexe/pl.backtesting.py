"""Matplotlib figures for the research report.

Style follows a validated data-viz palette: categorical hues assigned in fixed
order, one axis per chart (never dual y-scales), diverging color only for
signed quantities (blue = positive, red = negative, gray midpoint), recessive
grid, direct labels where few series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

matplotlib.use('Agg')

# Validated palette (light mode)
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
          '#e87ba4', '#008300', '#4a3aa7', '#e34948']
SURFACE = '#fcfcfb'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
DIVERGING = LinearSegmentedColormap.from_list(
    'pl_div', ['#e34948', '#f0efec', '#2a78d6'])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'pl_seq', ['#cde2fb', '#3987e5', '#0d366b'])

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK2,
    'axes.titlecolor': INK,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.6,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.frameon': False,
    'legend.fontsize': 8.5,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.6,
    'figure.dpi': 120,
})


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def pair_overview(prices: pd.DataFrame, spread: pd.Series, z: pd.Series,
                  trades: pd.DataFrame, labels: tuple[str, str],
                  sig_entry: float, sig_stop: float,
                  title: str, path: Path) -> str:
    """Three-panel pair figure: normalized legs, spread, z-score with trades."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(9.5, 7.2), sharex=True,
        gridspec_kw={'height_ratios': [2, 1.2, 1.4], 'hspace': 0.12})
    norm = prices / prices.iloc[0]
    ax1.plot(norm.index, norm['a'], color=SERIES[0], label=labels[0])
    ax1.plot(norm.index, norm['b'], color=SERIES[1], label=labels[1])
    ax1.set_ylabel('price (rebased = 1)')
    ax1.legend(loc='upper left', ncols=2)
    ax1.set_title(title, loc='left', fontweight='bold')

    ax2.plot(spread.index, spread, color=SERIES[6], linewidth=1.2)
    ax2.axhline(float(spread.mean()), color=AXIS, linewidth=0.8, linestyle='--')
    ax2.set_ylabel('log spread')

    ax3.plot(z.index, z, color=INK2, linewidth=1.0)
    for level, style in [(0, '-'), (sig_entry, '--'), (-sig_entry, '--'),
                         (sig_stop, ':'), (-sig_stop, ':')]:
        ax3.axhline(level, color=AXIS, linewidth=0.8, linestyle=style)
    if len(trades):
        longs = trades[trades['side'] > 0]
        shorts = trades[trades['side'] < 0]
        z_at_l = z.reindex(pd.DatetimeIndex(longs['entry_date'])).to_numpy()
        z_at_s = z.reindex(pd.DatetimeIndex(shorts['entry_date'])).to_numpy()
        ax3.scatter(longs['entry_date'], z_at_l, marker='^', s=34,
                    color=SERIES[2], zorder=5, label='enter long spread')
        ax3.scatter(shorts['entry_date'], z_at_s, marker='v', s=34,
                    color=SERIES[7], zorder=5, label='enter short spread')
        exit_z = z.reindex(pd.DatetimeIndex(trades['exit_date'])).to_numpy()
        ax3.scatter(trades['exit_date'], exit_z, marker='x', s=26,
                    color=MUTED, zorder=5, label='exit')
        ax3.legend(loc='upper left', ncols=3)
    ax3.set_ylabel('z-score')
    return _save(fig, path)


def equity_curves(curves: dict[str, pd.Series], title: str, path: Path,
                  ylabel: str = 'equity (start = 1)', logy: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for i, (name, eq) in enumerate(curves.items()):
        color = SERIES[i % len(SERIES)]
        ax.plot(eq.index, eq, color=color, label=name)
        if len(curves) <= 4 and len(eq):
            ax.annotate(f' {name}', (eq.index[-1], float(eq.iloc[-1])),
                        color=color, fontsize=8.5, va='center')
    if logy:
        ax.set_yscale('log')
    ax.axhline(1.0, color=AXIS, linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontweight='bold')
    if len(curves) > 1:
        ax.legend(loc='upper left', ncols=min(len(curves), 3))
    return _save(fig, path)


def sensitivity_heatmap(grid: pd.DataFrame, title: str, path: Path,
                        value_label: str = 'annualized Sharpe') -> str:
    """Heatmap of a signed metric over (entry x exit) thresholds, centered at 0."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    vals = grid.to_numpy(float)
    vmax = np.nanmax(np.abs(vals)) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = ax.imshow(vals, cmap=DIVERGING, norm=norm, aspect='auto')
    ax.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [str(i) for i in grid.index])
    ax.set_xlabel(grid.columns.name or 'exit threshold')
    ax.set_ylabel(grid.index.name or 'entry threshold')
    ax.grid(False)
    for (i, j), v in np.ndenumerate(vals):
        if np.isfinite(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                    color=INK)
    fig.colorbar(im, ax=ax, label=value_label, shrink=0.85)
    ax.set_title(title, loc='left', fontweight='bold')
    return _save(fig, path)


def screening_scatter(scr: pd.DataFrame, title: str, path: Path,
                      pvalue_line: float = 0.05, hl_max: float = 60) -> str:
    """Half-life vs Engle-Granger p-value for all screened pairs."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    df = scr[np.isfinite(scr['half_life'])]
    sc = ax.scatter(df['half_life'].clip(upper=250), df['eg_pvalue'].clip(1e-5, 1),
                    c=df['corr'], cmap=SEQUENTIAL, s=22, alpha=0.85,
                    edgecolors=SURFACE, linewidths=0.5)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.axhline(pvalue_line, color=SERIES[7], linewidth=0.9, linestyle='--')
    ax.axvline(hl_max, color=AXIS, linewidth=0.9, linestyle=':')
    ax.set_xlabel('spread half-life (bars, log scale, clipped at 250)')
    ax.set_ylabel('Engle-Granger p-value (log)')
    fig.colorbar(sc, ax=ax, label='daily log-return correlation', shrink=0.85)
    ax.set_title(title, loc='left', fontweight='bold')
    # annotate the strongest few pairs
    for _, row in df.nsmallest(4, 'eg_pvalue').iterrows():
        ax.annotate(f"{row['y']}/{row['x']}",
                    (max(row['half_life'], 1.05), max(row['eg_pvalue'], 1.2e-5)),
                    fontsize=7.5, color=INK2)
    return _save(fig, path)


def cost_sensitivity(curves: pd.DataFrame, title: str, path: Path) -> str:
    """Sharpe (y) vs per-side cost in bp (x), one line per pair; single axis."""
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for i, col in enumerate(curves.columns):
        color = SERIES[i % len(SERIES)]
        ax.plot(curves.index, curves[col], color=color, marker='o',
                markersize=4, label=col)
    ax.axhline(0.0, color=AXIS, linewidth=0.8)
    ax.set_xlabel('per-side cost (bp of traded notional)')
    ax.set_ylabel('out-of-sample annualized Sharpe')
    ax.legend(loc='best', ncols=2)
    ax.set_title(title, loc='left', fontweight='bold')
    return _save(fig, path)


def gate_timeline(windows: pd.DataFrame, title: str, path: Path) -> str:
    """Which walk-forward windows passed the cointegration gate, over time."""
    fig, ax = plt.subplots(figsize=(9.5, 2.4))
    passed = windows[windows['traded']]
    failed = windows[~windows['traded']]
    ax.scatter(passed['start'], np.zeros(len(passed)) + 1, marker='s', s=42,
               color=SERIES[0], label='traded (gate passed)')
    ax.scatter(failed['start'], np.zeros(len(failed)), marker='s', s=42,
               color=MUTED, alpha=0.55, label='flat (gate failed)')
    ax.set_yticks([0, 1], ['flat', 'traded'])
    ax.set_ylim(-0.7, 1.7)
    ax.legend(loc='upper left', ncols=2)
    ax.set_title(title, loc='left', fontweight='bold')
    return _save(fig, path)

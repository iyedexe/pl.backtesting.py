"""Poor Man's Covered Call — find and plot the best-performing configuration.

Runs the PMCC engine (synthetic Black-Scholes options on French large caps,
2000-2015, net of costs) over a grid of the two structural knobs of the
strategy — the short call's target delta and how far out the LEAPS is
opened — on the 10 most option-liquid names, aggregated into an equal-weight
daily-rebalanced portfolio, against the covered-call and buy-and-hold
benchmarks priced with the same model.

Outputs (``examples/figures/pmcc_*.png``, ``examples/tables/pmcc_*.csv``):

- ``pmcc_best_grid.png``    portfolio Sharpe over short delta × LEAPS tenor
- ``pmcc_best_equity.png``  best PMCC vs default PMCC vs covered call vs buy & hold
- ``pmcc_per_ticker.png``   per-name CAGR of the best PMCC vs its benchmarks

Usage::

    uv run python examples/run_pmcc.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import AXIS, MUTED, SERIES, out_dirs, print_table, save_table

from pairs_trading import plotting
from pmcc import data as datamod
from pmcc.engine import Engine
from pmcc.stats import compute_stats, drawdown_series
from pmcc.strategy import PMCC, BuyHold, CoveredCall, CoveredCallParams, PMCCParams

INITIAL = 100_000.0
DEFAULT = {'short_target_delta': 0.25, 'leaps_open_min_dte': 540}


def run_one(job: dict) -> dict:
    """One (ticker, strategy, params) backtest -> record with the equity curve."""
    eng = Engine(job['ticker'], initial_cash=INITIAL)
    kind = job['strategy']
    if kind == 'PMCC':
        strat = PMCC(PMCCParams(**job['params']))
    elif kind == 'CoveredCall':
        strat = CoveredCall(CoveredCallParams(short_target_delta=0.25))
    else:
        strat = BuyHold()
    res = eng.run(strat)
    st = compute_stats(res.equity, eng.df['r'])
    return {**job, 'equity': res.equity, 'CAGR [%]': st['CAGR [%]'],
            'Sharpe': st['Sharpe'], 'Max Drawdown [%]': st['Max Drawdown [%]']}


def portfolio(curves: list[pd.Series]) -> pd.Series:
    """Equal-weight, daily-rebalanced portfolio of per-ticker equity curves."""
    rets = pd.concat([c.pct_change() for c in curves], axis=1)
    return INITIAL * (1 + rets.mean(axis=1).fillna(0.0)).cumprod()


def main(quick: bool = False) -> None:
    fig_dir, _ = out_dirs()
    tickers = datamod.LIQUID_OPTIONS[:3] if quick else list(datamod.LIQUID_OPTIONS)
    deltas = [0.25, 0.35] if quick else [0.20, 0.25, 0.30, 0.35]
    tenors = [540] if quick else [360, 540, 720]

    jobs = [{'ticker': t, 'strategy': 'PMCC',
             'params': {'short_target_delta': d, 'leaps_open_min_dte': n}}
            for t, d, n in itertools.product(tickers, deltas, tenors)]
    jobs += [{'ticker': t, 'strategy': s, 'params': {}}
             for t in tickers for s in ('CoveredCall', 'BuyHold')]
    if DEFAULT not in [j['params'] for j in jobs]:
        jobs += [{'ticker': t, 'strategy': 'PMCC', 'params': dict(DEFAULT)} for t in tickers]
    print(f'{len(jobs)} backtests on {len(tickers)} names ...')
    workers = max(1, (os.cpu_count() or 2) - 1)
    if workers == 1 or quick:
        recs = [run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(run_one, jobs, chunksize=2))

    # portfolio per configuration
    rates = None
    rows, curves = [], {}
    keys = {}
    for r in recs:
        key = (r['strategy'], r['params'].get('short_target_delta'),
               r['params'].get('leaps_open_min_dte'))
        keys.setdefault(key, []).append(r)
    for key, rs in keys.items():
        port = portfolio([r['equity'] for r in rs])
        if rates is None:
            rates = datamod.eur_short_rate(port.index)
        st = compute_stats(port, rates)
        label = (key[0] if key[0] != 'PMCC'
                 else f'PMCC δ={key[1]:.2f} / LEAPS ≥{key[2]}d')
        curves[label] = port
        rows.append({'config': label, 'strategy': key[0], 'short_delta': key[1],
                     'leaps_dte': key[2], **{k: st[k] for k in
                     ('CAGR [%]', 'Vol (ann.) [%]', 'Sharpe', 'Max Drawdown [%]',
                      'Worst Year [%]')}})
    table = pd.DataFrame(rows).set_index('config').sort_values('Sharpe', ascending=False)
    print_table(table, 'Equal-weight portfolio of liquid names, 2000-2015 (net of costs)')
    save_table(table, 'pmcc_configs')

    pm = table[table['strategy'] == 'PMCC']
    best_label = pm.index[0]
    best = pm.iloc[0]
    default_label = (f'PMCC δ={DEFAULT["short_target_delta"]:.2f} / '
                     f'LEAPS ≥{DEFAULT["leaps_open_min_dte"]}d')

    # heatmap: Sharpe over short delta x LEAPS tenor
    if len(deltas) > 1 or len(tenors) > 1:
        heat = pm.assign(leaps_dte=pm['leaps_dte'].astype(int)).pivot_table(
            index='short_delta', columns='leaps_dte', values='Sharpe')
        heat.index.name = 'short call target delta'
        heat.columns.name = 'LEAPS opened at ≥ days'
        plotting.sensitivity_heatmap(heat, 'PMCC portfolio Sharpe by short delta × LEAPS tenor',
                                     fig_dir / 'pmcc_best_grid.png',
                                     value_label='Sharpe (net of costs)')

    # equity curves + drawdowns
    series = [(best_label, SERIES[0]), (default_label, SERIES[6]),
              ('CoveredCall', SERIES[1]), ('BuyHold', SERIES[2])]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                                  gridspec_kw={'height_ratios': [2.4, 1], 'hspace': 0.1})
    for label, color in series:
        if label not in curves or (label == default_label and label == best_label):
            continue
        c = curves[label]
        ax.plot(c.index, c, color=color, label=label)
        ax2.plot(c.index, 100 * drawdown_series(c), color=color, linewidth=1.0)
    ax.set_yscale('log')
    ax.set_yticks([100e3, 200e3, 400e3, 800e3, 1600e3],
                  ['100k', '200k', '400k', '800k', '1.6M'])
    ax.set_ylabel('portfolio value (EUR, log)')
    ax.legend(loc='upper left')
    ax.set_title('Poor Man\'s Covered Call: best configuration vs benchmarks '
                 f'({len(tickers)} French large caps, equal weight)', loc='left',
                 fontweight='bold')
    ax2.set_ylabel('drawdown %')
    ax2.axhline(0, color=AXIS, linewidth=0.8)
    fig.savefig(fig_dir / 'pmcc_best_equity.png', bbox_inches='tight')
    plt.close(fig)

    # per-ticker CAGR: best PMCC vs covered call vs buy & hold
    bkey = ('PMCC', best['short_delta'], best['leaps_dte'])
    per = pd.DataFrame({
        best_label: {r['ticker']: r['CAGR [%]'] for r in keys[bkey]},
        'CoveredCall': {r['ticker']: r['CAGR [%]'] for r in keys[('CoveredCall', None, None)]},
        'BuyHold': {r['ticker']: r['CAGR [%]'] for r in keys[('BuyHold', None, None)]},
    }).sort_values(best_label, ascending=False)
    save_table(per, 'pmcc_per_ticker')
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(per))
    w = 0.27
    for i, (col, color) in enumerate(zip(per.columns, SERIES[:3], strict=True)):
        ax.bar(x + (i - 1) * w, per[col], width=w, color=color, label=col)
    ax.set_xticks(x, per.index, rotation=0, fontsize=8)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel('CAGR % 2000-2015, net of costs')
    ax.legend(loc='upper right', ncols=3)
    ax.set_title('Per-name CAGR: best PMCC vs covered call vs buy & hold', loc='left',
                 fontweight='bold')
    ax.tick_params(axis='x', colors=MUTED)
    fig.savefig(fig_dir / 'pmcc_per_ticker.png', bbox_inches='tight')
    plt.close(fig)

    print(f'\nBest PMCC configuration: {best_label} -> Sharpe {best["Sharpe"]:.2f}, '
          f'CAGR {best["CAGR [%]"]:.2f}%, max DD {best["Max Drawdown [%]"]:.1f}% '
          f'(covered call: Sharpe {table.loc["CoveredCall", "Sharpe"]:.2f}, '
          f'buy & hold: {table.loc["BuyHold", "Sharpe"]:.2f})')
    print(f'figures -> {fig_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true', help='3 names, tiny grid (CI)')
    main(ap.parse_args().quick)

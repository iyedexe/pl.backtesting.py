"""Poor Man's Covered Call on backtesting.py — a regime-filtered PMCC index.

`backtesting.py` trades one instrument and cannot hold option legs, so the
mechanical PMCC package (deep-ITM LEAPS call ~0.80Δ opened ≥ 540 days out,
short ~0.25Δ monthly call, both rolled by rule) is priced day by day into a
**total-return index** per stock by the ``pmcc`` engine — the same idea as
the CBOE BXM covered-call index — and the framework trades *that*. The
strategy decision left to the framework is the one the mechanical package
lacks: **when to be in it**. ``PMCCRegime`` steps aside when the underlying's
realized volatility spikes (the LEAPS leg is what crashes crush) and
re-enters when it calms, with hysteresis; ``bt.optimize`` finds the best
thresholds and ``MultiBacktest`` checks them name by name. Benchmarks are the
covered-call and buy-and-hold indices under the same model.

Outputs (``examples/figures``, ``examples/tables``):

- ``pmcc_best_grid.png``    portfolio Sharpe over exit × re-entry vol thresholds
- ``pmcc_best_equity.png``  best regime-filtered PMCC vs always-in vs benchmarks
- ``pmcc_per_ticker.png``   MultiBacktest: best filter vs always-in per name
- ``pmcc_tearsheet.html``   the framework's interactive tearsheet of the best run

Usage::

    uv run python examples/pmcc_strategy.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import AXIS, MUTED, SERIES, out_dirs, print_table, save_table

from backtesting import Backtest, Strategy
from backtesting.lib import MultiBacktest
from pairs_trading import plotting
from pmcc import data as datamod
from pmcc.engine import Engine
from pmcc.strategy import PMCC, BuyHold, CoveredCall, CoveredCallParams, PMCCParams

CASH = 100_000
SWITCH_COST = 0.01   # per side, on index notional: unwinding/re-opening the option package
ALWAYS_IN = {'vol_exit': 9.0, 'vol_reenter': 9.0}


def realized_vol(prices, lookback: int) -> np.ndarray:
    """Annualized trailing realized volatility of the underlying (causal)."""
    return (np.log(pd.Series(prices)).diff().rolling(lookback).std()
            * np.sqrt(252)).to_numpy()


class PMCCRegime(Strategy):
    """Long the PMCC index unless the underlying's realized vol is too high.

    Exit when trailing ``lookback``-day realized vol exceeds ``vol_exit``;
    re-enter once it drops below ``vol_reenter`` (hysteresis). Setting both
    thresholds very high gives the always-in mechanical PMCC.
    """
    vol_exit = 0.45
    vol_reenter = 0.30
    lookback = 21

    def init(self):
        self.rv = self.I(realized_vol, self.data.Underlying, self.lookback,
                         name='realized vol (underlying)', overlay=False)

    def next(self):
        rv = self.rv[-1]
        if rv != rv:
            return
        if self.position:
            if rv > self.vol_exit:
                self.position.close()
        elif rv < self.vol_reenter:
            self.buy()


def build_index(job: tuple) -> dict:
    """Price one (ticker, package) into a daily index tape via the pmcc engine."""
    ticker, kind = job
    eng = Engine(ticker, initial_cash=CASH)
    strat = {'PMCC': lambda: PMCC(PMCCParams()),
             'CoveredCall': lambda: CoveredCall(CoveredCallParams()),
             'BuyHold': BuyHold}[kind]()
    res = eng.run(strat)
    idx = res.equity / res.equity.iloc[0] * 100.0
    under = eng.df['raw'].reindex(idx.index).ffill()
    return {'ticker': ticker, 'kind': kind, 'index': idx, 'underlying': under}


def tape_from(index: pd.Series, underlying: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({'Open': index, 'High': index, 'Low': index, 'Close': index,
                       'Underlying': underlying})
    df.index.name = None
    return df.dropna()


def portfolio_tape(recs: list[dict], kind: str) -> pd.DataFrame:
    """Equal-weight daily-rebalanced index of all tickers' package indices; the
    'underlying' is the equal-weight underlying basket (drives the vol filter)."""
    idx = pd.concat([r['index'].pct_change() for r in recs if r['kind'] == kind], axis=1)
    und = pd.concat([r['underlying'].pct_change() for r in recs if r['kind'] == kind], axis=1)
    index = 100 * (1 + idx.mean(axis=1).fillna(0.0)).cumprod()
    under = 100 * (1 + und.mean(axis=1).fillna(0.0)).cumprod()
    return tape_from(index, under)


def main(quick: bool = False) -> None:
    fig_dir, _ = out_dirs()
    warnings.filterwarnings('ignore')
    tickers = datamod.LIQUID_OPTIONS[:3] if quick else list(datamod.LIQUID_OPTIONS)
    jobs = [(t, k) for t in tickers for k in ('PMCC', 'CoveredCall', 'BuyHold')]
    print(f'pricing {len(jobs)} package indices with the pmcc engine ...')
    workers = max(1, (os.cpu_count() or 2) - 1)
    if quick or workers == 1:
        recs = [build_index(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(build_index, jobs))

    tapes = {k: portfolio_tape(recs, k) for k in ('PMCC', 'CoveredCall', 'BuyHold')}
    bt = Backtest(tapes['PMCC'], PMCCRegime, cash=CASH, commission=SWITCH_COST,
                  finalize_trades=True)
    always_in = bt.run(**ALWAYS_IN)

    # framework optimizer over the regime thresholds
    # the (9.0, 8.0) cell is the always-in mechanical PMCC, so the optimizer may pick it
    exits = [0.35, 0.50, 9.0] if quick else [0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 9.0]
    reenters = [0.25, 8.0] if quick else [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 8.0]
    best, heat = bt.optimize(vol_exit=exits, vol_reenter=reenters, maximize='Sharpe Ratio',
                             constraint=lambda p: p.vol_reenter < p.vol_exit,
                             return_heatmap=True)
    heat = heat.dropna()
    h2 = heat.unstack('vol_reenter')
    h2.index.name, h2.columns.name = 'exit when realized vol >', 're-enter when vol <'
    plotting.sensitivity_heatmap(h2, 'PMCC regime filter: portfolio Sharpe (bt.optimize)',
                                 fig_dir / 'pmcc_best_grid.png', value_label='Sharpe')
    bp = best['_strategy']
    label = ('PMCC, always in (best)' if bp.vol_exit >= 9 else
             f'PMCC, step aside at vol > {bp.vol_exit:.2f}, back in < {bp.vol_reenter:.2f}')

    # benchmarks through the same framework (always in)
    bench = {k: Backtest(tapes[k], PMCCRegime, cash=CASH, commission=SWITCH_COST,
                         finalize_trades=True).run(**ALWAYS_IN)
             for k in ('CoveredCall', 'BuyHold')}
    rows = {label: best, 'PMCC, always in': always_in} if bp.vol_exit < 9 else {label: best}
    rows.update({'Covered call, always in': bench['CoveredCall'],
                 'Buy & hold': bench['BuyHold']})
    table = pd.DataFrame({k: {c: float(v[c]) for c in
                              ('Return [%]', 'CAGR [%]', 'Sharpe Ratio', 'Max. Drawdown [%]',
                               'Exposure Time [%]', '# Trades')}
                          for k, v in rows.items()}).T
    print_table(table, 'Equal-weight portfolio of the liquid names, 2000-2015, on backtesting.py')
    save_table(table, 'pmcc_configs')

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                                  gridspec_kw={'height_ratios': [2.4, 1], 'hspace': 0.1})
    palette = [SERIES[0], SERIES[6], SERIES[1], SERIES[2]]
    if len(rows) < 4:                       # always-in is the best: one PMCC curve
        palette = [SERIES[0], SERIES[1], SERIES[2]]
    for (name, st), color in zip(rows.items(), palette, strict=True):
        eq = st['_equity_curve']['Equity']
        ax.plot(eq.index, eq, color=color, label=name)
        ax2.plot(eq.index, -st['_equity_curve']['DrawdownPct'] * 100, color=color, linewidth=1)
    ax.set_yscale('log')
    ax.set_yticks([100e3, 200e3, 400e3, 800e3, 1600e3],
                  ['100k', '200k', '400k', '800k', '1.6M'])
    ax.set_ylabel('portfolio value (EUR, log)')
    ax.legend(loc='upper left')
    ax.set_title("Poor Man's Covered Call on backtesting.py: best regime filter vs benchmarks",
                 loc='left', fontweight='bold')
    ax2.set_ylabel('drawdown %')
    ax2.axhline(0, color=AXIS, linewidth=0.8)
    fig.savefig(fig_dir / 'pmcc_best_equity.png', bbox_inches='tight')
    plt.close(fig)
    bt.plot(filename=str(fig_dir / 'pmcc_tearsheet.html'), open_browser=False, resample=False)

    # per-name check with MultiBacktest: best filter vs always in
    per_tapes = [tape_from(r['index'], r['underlying']) for r in recs if r['kind'] == 'PMCC']
    names = [r['ticker'] for r in recs if r['kind'] == 'PMCC']
    mbt = MultiBacktest(per_tapes, PMCCRegime, cash=CASH, commission=SWITCH_COST,
                        finalize_trades=True)
    filt = mbt.run(vol_exit=bp.vol_exit, vol_reenter=bp.vol_reenter)
    base = mbt.run(**ALWAYS_IN)
    filt.columns, base.columns = names, names
    per = pd.DataFrame({'best filter': filt.loc['Sharpe Ratio'].astype(float),
                        'always in': base.loc['Sharpe Ratio'].astype(float)})
    per = per.sort_values('always in', ascending=False)
    print_table(per, 'Per-name Sharpe (MultiBacktest): best regime filter vs always in')
    save_table(per, 'pmcc_per_ticker')
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(per))
    ax.bar(x - 0.19, per['best filter'], width=0.38, color=SERIES[0], label='best regime filter')
    ax.bar(x + 0.19, per['always in'], width=0.38, color=SERIES[6], label='always in')
    ax.set_xticks(x, per.index, fontsize=8)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel('Sharpe, 2000-2015, net of costs')
    ax.legend(loc='upper right', ncols=2)
    ax.set_title('Per-name PMCC index: does the regime filter help?', loc='left',
                 fontweight='bold')
    ax.tick_params(axis='x', colors=MUTED)
    fig.savefig(fig_dir / 'pmcc_per_ticker.png', bbox_inches='tight')
    plt.close(fig)

    print(f'\nBest: {label} -> Sharpe {float(best["Sharpe Ratio"]):.2f}, CAGR '
          f'{float(best["CAGR [%]"]):.2f}%, max DD {float(best["Max. Drawdown [%]"]):.1f}% '
          f'(always-in PMCC {float(always_in["Sharpe Ratio"]):.2f}, covered call '
          f'{float(bench["CoveredCall"]["Sharpe Ratio"]):.2f}, buy & hold '
          f'{float(bench["BuyHold"]["Sharpe Ratio"]):.2f}); filter helps on '
          f'{int((per["best filter"] > per["always in"]).sum())}/{len(per)} names')
    print(f'figures -> {fig_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true', help='3 names, tiny grid (CI)')
    main(ap.parse_args().quick)

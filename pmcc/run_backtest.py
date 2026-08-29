"""
Run the PMCC backtest across the French large-cap universe.

Usage::

    python -m pmcc.run_backtest headline      # all tickers x all strategies
    python -m pmcc.run_backtest sensitivity   # parameter grid on 5 names
    python -m pmcc.run_backtest all           # both + plots + RESULTS.md

Outputs go to ``pmcc/results/``.
"""
from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import data as datamod
from .engine import CostModel, Engine
from .stats import compute_stats
from .strategy import (BuyHold, CoveredCall, CoveredCallParams, LeapsOnly,
                       PMCC, PMCCParams)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
TICKERS = list(datamod.UNIVERSE)
INITIAL = 100_000.0


def make_strategy(kind: str, **kw):
    if kind == 'PMCC':
        s = PMCC(PMCCParams(**kw))
    elif kind == 'PMCC_budget':
        s = PMCC(PMCCParams(sizing='budget', **kw))
        s.name = 'PMCC_budget'
    elif kind == 'CoveredCall':
        s = CoveredCall(CoveredCallParams(**kw))
    elif kind == 'BuyHold':
        s = BuyHold()
    elif kind == 'LeapsOnly':
        s = LeapsOnly(PMCCParams(**kw))
    else:
        raise ValueError(kind)
    return s


def run_one(job: dict) -> dict:
    """One (ticker, strategy, engine-params) backtest -> flat result record."""
    eng = Engine(job['ticker'],
                 prem=job.get('prem', 'vix'),
                 skew_slope=job.get('skew', 0.10),
                 div_yield=job.get('div_yield'),
                 costs=CostModel(opt_half_spread_pct=job.get('spread', 0.02)),
                 initial_cash=INITIAL)
    strat = make_strategy(job['strategy'], **job.get('strat_kw', {}))
    res = eng.run(strat)
    st = compute_stats(res.equity, eng.df['r'])
    rec = {'ticker': job['ticker'], 'strategy': strat.name, **st}
    for k in ('dividends', 'interest', 'short_premium_received',
              'short_buyback_paid', 'short_settle_paid', 'commissions',
              'spread_cost', 'stock_fees', 'ftt', 'n_assignments',
              'n_short_cycles', 'n_leaps_rolls', 'n_trades', 'min_cash'):
        rec[k] = round(res.counters[k], 0)
    rec.update({k: v for k, v in job.items()
                if k in ('prem', 'skew', 'spread', 'div_yield')})
    rec.update({f'p_{k}': v for k, v in job.get('strat_kw', {}).items()})
    if job.get('keep_curve'):
        rec['_equity'] = res.equity
    return rec


HEADLINE_STRATEGIES = ['PMCC', 'PMCC_budget', 'CoveredCall', 'BuyHold', 'LeapsOnly']


def headline_jobs() -> list[dict]:
    return [{'ticker': t, 'strategy': s, 'keep_curve': True}
            for t in TICKERS for s in HEADLINE_STRATEGIES]


SENS_TICKERS = ['FP.PA', 'MC.PA', 'BNP.PA', 'OR.PA', 'AIR.PA']


def sensitivity_jobs() -> list[dict]:
    jobs = []
    # main grid: premium model x skew x spread x short delta
    for t, prem, skew, spread, sd in itertools.product(
            SENS_TICKERS, ['vix', 1.0, 1.2], [0.0, 0.10], [0.01, 0.02, 0.03],
            [0.20, 0.25, 0.35]):
        base = {'ticker': t, 'prem': prem, 'skew': skew, 'spread': spread}
        jobs.append({**base, 'strategy': 'PMCC',
                     'strat_kw': {'short_target_delta': sd}})
        jobs.append({**base, 'strategy': 'CoveredCall',
                     'strat_kw': {'short_target_delta': sd}})
    # mini-sweeps off the default point
    for t in SENS_TICKERS:
        for dte in (360, 540, 720):
            jobs.append({'ticker': t, 'strategy': 'PMCC', 'sweep': 'leaps_dte',
                         'strat_kw': {'leaps_open_min_dte': dte}})
        for dd in (None, 0.60):
            jobs.append({'ticker': t, 'strategy': 'PMCC', 'sweep': 'defend',
                         'strat_kw': {'defend_delta': dd}})
        meta = datamod.UNIVERSE[t]
        for shift in (-0.01, 0.0, 0.01):
            jobs.append({'ticker': t, 'strategy': 'PMCC', 'sweep': 'divyield',
                         'div_yield': round(meta.div_yield + shift, 4)})
            jobs.append({'ticker': t, 'strategy': 'CoveredCall', 'sweep': 'divyield',
                         'div_yield': round(meta.div_yield + shift, 4)})
    return jobs


def run_jobs(jobs: list[dict], workers: int | None = None) -> list[dict]:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1 or len(jobs) < 8:
        return [run_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_one, jobs, chunksize=4))


def aggregate_portfolio(records: list[dict], tickers: list[str]) -> pd.DataFrame:
    """Daily-rebalanced equal-weight portfolio per strategy: each day's
    portfolio return is the mean of the daily returns of all live per-ticker
    backtests (late starters simply join the average when their data begins)."""
    curves = {}
    for strategy in HEADLINE_STRATEGIES:
        cs = [r['_equity'] for r in records
              if r['strategy'] == strategy and r['ticker'] in tickers and '_equity' in r]
        rets = pd.concat([c.pct_change() for c in cs], axis=1)
        port_ret = rets.mean(axis=1).fillna(0.0)
        curves[strategy] = INITIAL * (1 + port_ret).cumprod()
    return pd.DataFrame(curves)


def main(mode: str = 'all'):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if mode in ('headline', 'all'):
        print(f'headline: {len(headline_jobs())} runs...')
        recs = run_jobs(headline_jobs())
        tbl = pd.DataFrame([{k: v for k, v in r.items() if k != '_equity'}
                            for r in recs])
        tbl.to_csv(os.path.join(RESULTS_DIR, 'headline_stats.csv'), index=False)
        print('wrote headline_stats.csv', tbl.shape)

        port = aggregate_portfolio(recs, datamod.LIQUID_OPTIONS)
        port.to_csv(os.path.join(RESULTS_DIR, 'portfolio_curves.csv'))
        rates = datamod.eur_short_rate(port.index)
        pstats = pd.DataFrame({s: compute_stats(port[s], rates) for s in port})
        pstats.to_csv(os.path.join(RESULTS_DIR, 'portfolio_stats.csv'))
        print(pstats.to_string())

        # per-ticker curves for plotting
        eq = {}
        for r in recs:
            if '_equity' in r:
                eq[(r['ticker'], r['strategy'])] = r['_equity']
        curves = pd.DataFrame(eq)
        curves.columns = [f'{t}|{s}' for t, s in curves.columns]
        curves.to_csv(os.path.join(RESULTS_DIR, 'all_curves.csv.gz'),
                      compression='gzip')

    if mode in ('sensitivity', 'all'):
        jobs = sensitivity_jobs()
        print(f'sensitivity: {len(jobs)} runs...')
        recs = run_jobs(jobs)
        for j, r in zip(jobs, recs):
            r['sweep'] = j.get('sweep', 'grid')
        tbl = pd.DataFrame([{k: v for k, v in r.items() if k != '_equity'}
                            for r in recs])
        tbl.to_csv(os.path.join(RESULTS_DIR, 'sensitivity_stats.csv'), index=False)
        print('wrote sensitivity_stats.csv', tbl.shape)

    if mode == 'all':
        from . import report
        report.build()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')

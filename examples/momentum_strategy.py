"""Multi-horizon momentum on backtesting.py — Man AHL's trend rule on crypto.

The rule is the multi-horizon *time-series momentum* that Man AHL describes
(Moskowitz, Ooi & Pedersen, "Time Series Momentum", 2012; Hurst, Ooi &
Pedersen, "A Century of Evidence on Trend-Following Investing"): for each of
four look-backs — one week, two weeks, one month, two months (5, 10, 21 and
42 bars) — take the sign of today's close minus the close that many bars
ago. The **score** is the sum of the four signs, −4 (fully short) to +4
(fully long), ±2 a half position. Every coin then carries the same risk:
``position = score / 4 × target risk / realized volatility`` of the sleeve's
equity. The score is computed on the daily close and executed at the next
bar's open — in a market that never closes that is the same price, plus
slippage — and small drifts of the target position are held rather than
traded (``hold_band``), which is where most of the turnover would go.

`backtesting.py` trades one instrument, so every coin is one
``FractionalBacktest`` sleeve (same rule, same sizing, same costs) and the
**book** is the sum of the sleeves' P&L on one capital base — each sleeve
sized against its own equity rather than the book's, which drifts by a few
percent and nothing else. The book's statistics come from the framework too:
its equity curve goes through ``Backtest`` as an always-in index. The
parameter grid (horizon set × hold band) is judged at the book level — every
cell runs every sleeve — because one coin's Sharpe is noise; the optimum
each coin's own ``bt.optimize`` would pick is read off the same runs.

Data: the vendored Coin Metrics daily reference rates from 2018 (the video's
start). The panel has no volume, so the monthly re-selection of the ten most
traded coins becomes the fixed universe of the panel's twelve non-stablecoin
assets (PAXG, a gold token, is excluded); DOT enters once it has a history.

Outputs (``examples/figures``, ``examples/tables``):

- ``momentum_best_grid.png``        book Sharpe over horizon set × hold band
- ``momentum_best_equity.png``      best book vs default vs vol-scaled long-only vs BTC
- ``momentum_per_coin.png``         per-coin Sharpe: the book's best cell vs each coin's own
- ``momentum_score_diagnostic.png`` next-day vol-scaled return by score (the video's check)
- ``momentum_tearsheet.html``       the framework's interactive tearsheet of the BTC sleeve

Usage::

    TQDM_DISABLE=1 uv run python examples/momentum_strategy.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from _common import AXIS, MUTED, SERIES, out_dirs, print_table, save_table

from backtesting import Strategy
from backtesting.lib import FractionalBacktest
from pairs_trading import data, plotting

CASH = 100_000
START = '2018-01-01'         # evaluation window; sleeves warm up on the 200 days before it
WARMUP_DAYS = 200
DAYS_PER_YEAR = 365          # crypto trades every calendar day (the framework annualizes alike)
COST_PER_SIDE = 0.0015       # 10 bp taker fee + 5 bp slippage, charged on every fill
UNIT = 1e-6                  # sleeves trade millionths of a coin
# PAXG (a gold token) is the panel's one stablecoin-like asset and is left out
UNIVERSE = ('BTC', 'ETH', 'LTC', 'BCH', 'XRP', 'ADA', 'DOGE', 'DOT', 'XMR', 'LINK', 'ETC', 'BNB')
HORIZONS = {                  # look-backs in bars; 'video': 1 week, 2 weeks, 1 month, 2 months
    'fast': (3, 5, 10, 21),
    'video': (5, 10, 21, 42),
    'slow': (10, 21, 42, 84),
    'slowest': (21, 42, 84, 168),
}
BANDS = (0.0, 0.25, 0.5, 1.0)
DEFAULT = {'horizon': 'video', 'hold_band': 0.25}
COST_LEVELS = (0.0, 0.0010, 0.0015, 0.0025)


def momentum_score(close, lookbacks) -> np.ndarray:
    """Sum over look-backs of sign(close − close n bars ago); NaN while warming up."""
    c = pd.Series(np.asarray(close, float))
    return sum(np.sign(c - c.shift(n)) for n in lookbacks).to_numpy()


def realized_vol(close, lookback: int) -> np.ndarray:
    """Annualized trailing standard deviation of daily log returns (causal)."""
    r = np.log(pd.Series(np.asarray(close, float))).diff()
    return (r.rolling(lookback).std() * np.sqrt(DAYS_PER_YEAR)).to_numpy()


class MultiHorizonMomentum(Strategy):
    """Man AHL's multi-horizon momentum, one coin per sleeve.

    ``score`` = Σ sign(close − close n bars ago) over ``HORIZONS[horizon]``;
    target position = ``score / n × target_vol / realized vol`` of the sleeve's
    equity (``vol_floor`` caps the leverage in quiet markets). The order is
    placed on the close and fills at the next bar's open. A change of target
    smaller than ``hold_band`` × the full-score position is held, not traded.
    ``long_only=True`` pins the score at +n: the vol-scaled long benchmark.
    """
    horizon = 'video'
    hold_band = 0.25
    vol_lookback = 42
    target_vol = 0.03      # per coin, of the sleeve's capital: 12 coins ≈ the video's 7-8% book vol
    vol_floor = 0.15
    long_only = False

    def init(self):
        self.lookbacks = HORIZONS.get(self.horizon, self.horizon)
        self.score = self.I(momentum_score, self.data.Close, self.lookbacks,
                            name='score', overlay=False)
        self.vol = self.I(realized_vol, self.data.Close, self.vol_lookback,
                          name='realized vol', overlay=False)

    def next(self):
        score, vol = self.score[-1], self.vol[-1]
        if score != score or vol != vol:
            return
        n = len(self.lookbacks)
        if self.long_only:
            score = n
        price = self.data.Close[-1]    # the next open is this close: the market never shuts
        full = self.target_vol / max(vol, self.vol_floor) * self.equity / price
        delta = score / n * full - self.position.size
        if abs(delta) < self.hold_band * full:
            return
        units = round(abs(delta))
        if units:
            (self.buy if delta > 0 else self.sell)(size=units)


class AlwaysIn(Strategy):
    """Holds an index from its first bar: the framework's statistics for any curve."""
    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy()


def coin_tape(close: pd.Series) -> pd.DataFrame:
    """Daily tape whose open is the previous close (a 24/7 market): a market
    order placed on a bar's close therefore fills at that close, next bar."""
    close = close.dropna().astype(float)
    op = close.shift(1).fillna(close.iloc[0])
    df = pd.DataFrame({'Open': op, 'High': np.maximum(op, close), 'Low': np.minimum(op, close),
                       'Close': close, 'Volume': np.nan})
    df.index.name = None
    return df


def sleeve_backtest(tape: pd.DataFrame, cost: float = COST_PER_SIDE) -> FractionalBacktest:
    return FractionalBacktest(tape, MultiHorizonMomentum, fractional_unit=UNIT, cash=CASH,
                              commission=cost, finalize_trades=True)


def held_notional(trades: pd.DataFrame, tape: pd.DataFrame) -> pd.Series:
    """Absolute position value at every close, rebuilt from the trade list."""
    d = np.zeros(len(tape) + 1)
    np.add.at(d, trades['EntryBar'].to_numpy(int), trades['Size'].to_numpy(float))
    np.add.at(d, trades['ExitBar'].to_numpy(int), -trades['Size'].to_numpy(float))
    units = np.cumsum(d)[:-1]
    return pd.Series(np.abs(units) * tape['Close'].to_numpy(), index=tape.index)


def warm_start() -> pd.Timestamp:
    return pd.Timestamp(START) - pd.Timedelta(days=WARMUP_DAYS)


def run_job(job: tuple) -> dict:
    """One sleeve: (symbol, cost per side, strategy params) -> P&L curve and trade facts.

    The sleeve runs from ``WARMUP_DAYS`` before ``START`` so that every horizon
    set is live on the first evaluated bar; P&L and trade facts are those of
    the evaluation window (trades closed on or after ``START``).
    """
    symbol, cost, params = job
    tape = coin_tape(data.load_panel('crypto')[symbol].loc[warm_start():])
    st = sleeve_backtest(tape, cost).run(**params)
    tr = st['_trades']
    tr = tr[pd.to_datetime(tr['ExitTime']) >= pd.Timestamp(START)]   # (works when empty)
    eq = st['_equity_curve']['Equity'].loc[START:]
    return {'symbol': symbol, 'cost': cost, 'params': params,
            'pnl': eq - eq.iloc[0], 'sharpe': float(index_stats(eq)['Sharpe Ratio']),
            'n_trades': len(tr), 'wins': int((tr['PnL'] > 0).sum()),
            'commissions': float(tr['Commission'].sum()),
            'notional': float((tr['Size'].abs() * (tr['EntryPrice'] + tr['ExitPrice'])).sum()),
            'held': held_notional(tr, tape).loc[START:]}


def run_jobs(jobs: list[tuple], quick: bool) -> list[dict]:
    workers = max(1, (os.cpu_count() or 2) - 1)
    if quick or workers == 1:
        return [run_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_job, jobs, chunksize=4))


def book_equity(sleeves: list[dict]) -> pd.Series:
    """Book = CASH + Σ sleeve P&L; a sleeve contributes nothing before its coin has data."""
    idx = sleeves[0]['pnl'].index
    for s in sleeves[1:]:
        idx = idx.union(s['pnl'].index)
    pnl = sum(s['pnl'].reindex(idx).ffill().fillna(0.0) for s in sleeves)
    return (CASH + pnl).rename('Equity')


def index_stats(equity: pd.Series) -> pd.Series:
    """The framework's statistics for an equity curve, traded always-in at no cost.

    The strategy's first ``next()`` runs on the second bar, so the tape gets
    one flat bar in front and the first real return is not lost; the position
    stays open (marked to market) through the last bar for the same reason.
    """
    idx = equity / equity.iloc[0]
    pad = pd.Series([1.0], index=[idx.index[0] - (idx.index[1] - idx.index[0])])
    idx = pd.concat([pad, idx])
    tape = pd.DataFrame({'Open': idx, 'High': idx, 'Low': idx, 'Close': idx, 'Volume': np.nan})
    tape.index.name = None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Some trades remain open')
        return FractionalBacktest(tape, AlwaysIn, fractional_unit=UNIT, cash=CASH,
                                  trade_on_close=True).run()


def book_summary(sleeves: list[dict], btc: pd.Series) -> tuple[pd.Series, dict]:
    """Framework statistics of the book plus trade-level facts summed over sleeves."""
    eq = book_equity(sleeves)
    st = index_stats(eq)
    years = (eq.index[-1] - eq.index[0]).days / DAYS_PER_YEAR
    held = sum(s['held'].reindex(eq.index).fillna(0.0) for s in sleeves)
    gross = held / eq
    r = eq.pct_change().dropna()
    monthly = eq.resample('ME').last().pct_change().dropna()
    facts = {
        'Return [%]': float(st['Return [%]']), 'CAGR [%]': float(st['CAGR [%]']),
        'Volatility (Ann.) [%]': float(st['Volatility (Ann.) [%]']),
        'Sharpe Ratio': float(st['Sharpe Ratio']), 'Sortino Ratio': float(st['Sortino Ratio']),
        'Max. Drawdown [%]': float(st['Max. Drawdown [%]']),
        '# Trades': sum(s['n_trades'] for s in sleeves),
        'Win Rate [%]': 100 * sum(s['wins'] for s in sleeves)
        / max(1, sum(s['n_trades'] for s in sleeves)),
        'Turnover [x/yr]': sum(s['notional'] for s in sleeves) / eq.mean() / years,
        'Costs [% final eq.]': 100 * sum(s['commissions'] for s in sleeves) / eq.iloc[-1],
        'Gross exposure mean [%]': 100 * float(gross.mean()),
        'Gross exposure max [%]': 100 * float(gross.max()),
        'Corr. to BTC': float(r.corr(btc.pct_change().reindex(r.index))),
        'Monthly skew': float(monthly.skew()),
    }
    return eq, facts


def score_diagnostic(panel: pd.DataFrame, lookbacks, vol_lookback: int, start=START
                     ) -> tuple[pd.DataFrame, dict]:
    """The video's check: does a higher score precede a bigger next-day move?

    Pools every coin-day: next-day return in units of the coin's trailing
    daily vol, by score. t-statistics from OLS on the score with standard
    errors clustered by day (all coins move together), with and without
    BTC's own next-day move as a control (the latter on the alts only).
    """
    frames = []
    for sym in panel:
        c = panel[sym].dropna()
        vol = realized_vol(c.to_numpy(), vol_lookback) / np.sqrt(DAYS_PER_YEAR)
        frames.append(pd.DataFrame({
            'symbol': sym,
            'score': momentum_score(c.to_numpy(), lookbacks),
            'y': (c.pct_change().shift(-1) / vol).to_numpy(),
        }, index=c.index))
    pooled = pd.concat(frames).dropna()
    pooled = pooled[np.isfinite(pooled['y']) & (pooled.index >= pd.Timestamp(start))]
    pooled['btc_y'] = pooled.index.map(pooled.loc[pooled['symbol'] == 'BTC', 'y'])
    pooled['day'] = pd.factorize(pooled.index)[0]
    by = pooled.groupby('score')['y'].agg(['mean', 'sem', 'count'])
    by.index.name = 'score'

    def tstat(df: pd.DataFrame, cols: list[str]) -> float:
        fit = sm.OLS(df['y'], sm.add_constant(df[cols])).fit(
            cov_type='cluster', cov_kwds={'groups': df['day'].to_numpy()})
        return float(fit.tvalues['score'])

    alts = pooled[(pooled['symbol'] != 'BTC') & pooled['btc_y'].notna()]
    return by, {'t': tstat(pooled, ['score']), 't_ex_btc': tstat(alts, ['score', 'btc_y']),
                'n': len(pooled)}


def yearly_returns(curves: dict[str, pd.Series]) -> pd.DataFrame:
    out = {}
    for name, eq in curves.items():
        first = eq.iloc[[0]]
        first.index = [eq.index[0] - pd.Timedelta(days=1)]
        yearly = pd.concat([first, eq]).resample('YE').last().pct_change().dropna() * 100
        out[name] = yearly.rename(index=lambda d: d.year)
    return pd.DataFrame(out)


def cell_label(horizon: str, band: float) -> str:
    return f'{"/".join(map(str, HORIZONS[horizon]))}, band {band:g}'


def main(quick: bool = False) -> None:
    fig_dir, _ = out_dirs()
    warnings.filterwarnings('ignore')
    panel = data.load_panel('crypto').loc[warm_start():]
    symbols = [s for s in UNIVERSE if s in panel][:4 if quick else None]
    panel = panel[symbols]
    btc = panel['BTC'].loc[START:]
    horizons = ['video', 'slow'] if quick else list(HORIZONS)
    bands = [0.25, 1.0] if quick else list(BANDS)
    costs = [0.0, COST_PER_SIDE] if quick else list(COST_LEVELS)

    # 1. the grid, judged at the book level: every cell runs every sleeve
    cells = [(h, b) for h in horizons for b in bands]
    jobs = [(s, COST_PER_SIDE, {'horizon': h, 'hold_band': b}) for h, b in cells for s in symbols]
    print(f'{len(symbols)} coins x {len(cells)} cells = {len(jobs)} sleeve backtests ...')
    runs = run_jobs(jobs, quick)
    by_cell = {c: [r for r in runs if (r['params']['horizon'], r['params']['hold_band']) == c]
               for c in cells}
    books = {c: book_summary(by_cell[c], btc) for c in cells}
    grid = pd.Series({c: books[c][1]['Sharpe Ratio'] for c in cells})
    grid.index = pd.MultiIndex.from_tuples(grid.index, names=['horizon', 'hold_band'])
    h2 = grid.unstack('hold_band').reindex(horizons)
    h2.index = ['/'.join(map(str, HORIZONS[h])) for h in h2.index]
    h2.index.name, h2.columns.name = 'look-backs (bars)', 'hold band (× full position)'
    plotting.sensitivity_heatmap(h2, 'Multi-horizon momentum: book Sharpe (horizons × hold band)',
                                 fig_dir / 'momentum_best_grid.png', value_label='Sharpe')
    save_table(h2, 'momentum_best_grid')
    best_cell = grid.idxmax()
    default_cell = (DEFAULT['horizon'], DEFAULT['hold_band'])
    if default_cell not in books:
        default_cell = best_cell
    # per-coin Sharpe per cell: the argmax of each row is what bt.optimize would return per coin
    per_coin = pd.DataFrame({c: {r['symbol']: r['sharpe'] for r in by_cell[c]} for c in cells})
    per_coin.columns = pd.MultiIndex.from_tuples(per_coin.columns, names=['horizon', 'hold_band'])
    mean_sharpe_cell = per_coin.mean().idxmax()   # MultiBacktest.optimize's objective
    print(f'best book cell: {cell_label(*best_cell)}; mean-of-sleeves cell: '
          f'{cell_label(*mean_sharpe_cell)}; per-coin optima: '
          f'{per_coin.idxmax(axis=1).nunique()} distinct cells over {len(symbols)} coins')

    # 2. benchmarks: the same sizing with the score pinned long, and BTC itself
    best_params = {'horizon': best_cell[0], 'hold_band': best_cell[1]}
    long_runs = run_jobs([(s, COST_PER_SIDE, {**DEFAULT, 'long_only': True})
                          for s in symbols], quick)
    long_eq, long_facts = book_summary(long_runs, btc)
    btc_eq = (btc / btc.iloc[0] * CASH).rename('Equity')
    btc_st = index_stats(btc_eq)

    best_eq, best_facts = books[best_cell]
    default_eq, default_facts = books[default_cell]
    label_best = f'momentum, best: look-backs {cell_label(*best_cell)}'
    label_default = f'momentum, video default: {cell_label(*default_cell)}'
    rows = {label_best: best_facts}
    if default_cell != best_cell:
        rows[label_default] = default_facts
    rows['vol-scaled long only (same sizing, score pinned +4, band 0.25)'] = long_facts
    rows['BTC buy & hold'] = {k: float(btc_st[k]) for k in
                              ('Return [%]', 'CAGR [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio',
                               'Sortino Ratio', 'Max. Drawdown [%]')}
    table = pd.DataFrame(rows).T
    print_table(table, f'Book of {len(symbols)} sleeves, {START[:4]}-{panel.index[-1].year}, '
                       f'{COST_PER_SIDE * 1e4:.0f} bp per side, on backtesting.py')
    save_table(table, 'momentum_configs')

    curves = {label_best: best_eq}
    if default_cell != best_cell:
        curves[label_default] = default_eq
    curves['vol-scaled long only'] = long_eq
    curves['BTC buy & hold'] = btc_eq
    yearly = yearly_returns(curves)
    print_table(yearly, 'Calendar-year returns [%]', floatfmt='.1f')
    save_table(yearly, 'momentum_yearly')

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                                  gridspec_kw={'height_ratios': [2.4, 1], 'hspace': 0.1})
    palette = {label_best: SERIES[0], label_default: SERIES[6],
               'vol-scaled long only': SERIES[1], 'BTC buy & hold': MUTED}
    for name, eq in curves.items():
        ax.plot(eq.index, eq, color=palette[name], label=name,
                linewidth=1.2 if 'BTC' in name else 1.6)
        ax2.plot(eq.index, (eq / eq.cummax() - 1) * 100, color=palette[name], linewidth=1)
    ax.set_yscale('log')
    lo, hi = min(c.min() for c in curves.values()), max(c.max() for c in curves.values())
    ticks = [t for t in (12.5e3, 25e3, 50e3, 100e3, 200e3, 400e3, 800e3) if 0.7 * lo < t < 1.5 * hi]
    ax.set_yticks(ticks, [f'{t / 1e3:g}k' for t in ticks])
    ax.set_ylabel('book value (USD, log)')
    ax.legend(loc='upper left')
    ax.set_title('Multi-horizon momentum on backtesting.py: best book vs benchmarks',
                 loc='left', fontweight='bold')
    ax2.set_ylabel('drawdown %')
    ax2.axhline(0, color=AXIS, linewidth=0.8)
    fig.savefig(fig_dir / 'momentum_best_equity.png', bbox_inches='tight')
    plt.close(fig)

    # 3. per coin: the book's best cell vs the cell each coin would pick for itself
    own_best = per_coin.idxmax(axis=1)
    per = pd.DataFrame({'book best cell': per_coin[best_cell],
                        "coin's own best cell": per_coin.max(axis=1),
                        'own cell': [cell_label(*own_best[s]) for s in per_coin.index]})
    per = per.sort_values('book best cell', ascending=False)
    print_table(per, f'Per-coin sleeve Sharpe: book best ({cell_label(*best_cell)}) vs own optimum')
    save_table(per, 'momentum_per_coin')
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(per))
    ax.bar(x - 0.19, per['book best cell'], width=0.38, color=SERIES[0],
           label=f'book best cell ({cell_label(*best_cell)})')
    ax.bar(x + 0.19, per["coin's own best cell"], width=0.38, color=SERIES[6],
           label="coin's own best cell (in-sample)")
    for xi, (_, row) in zip(x, per.iterrows(), strict=True):
        ax.text(xi + 0.19, row["coin's own best cell"] + 0.02, row['own cell'].split(',')[0],
                ha='center', va='bottom', fontsize=6.5, color=MUTED, rotation=90)
    ax.set_xticks(x, per.index, fontsize=8)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylim(min(0.0, float(per.min(numeric_only=True).min())) - 0.05,
                1.5 * float(per.max(numeric_only=True).max()))
    ax.set_ylabel(f'Sharpe, {START[:4]}-{panel.index[-1].year}, net of costs')
    ax.legend(loc='upper right', ncols=2)
    ax.set_title('Per-coin sleeves: the book\'s cell vs each coin\'s own optimum',
                 loc='left', fontweight='bold')
    ax.tick_params(axis='x', colors=MUTED)
    fig.savefig(fig_dir / 'momentum_per_coin.png', bbox_inches='tight')
    plt.close(fig)

    # 4. the score diagnostic: next-day vol-scaled return by score (video default look-backs)
    by, ts = score_diagnostic(panel, HORIZONS[DEFAULT['horizon']],
                              MultiHorizonMomentum.vol_lookback)
    print_table(by, f'Next-day return (in daily σ) by score, {ts["n"]} coin-days; '
                    f't = {ts["t"]:.1f} (clustered by day), {ts["t_ex_btc"]:.1f} ex BTC beta',
                floatfmt='.3f')
    save_table(by.assign(t_stat=ts['t'], t_stat_ex_btc=ts['t_ex_btc']), 'momentum_score_diagnostic')
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = [SERIES[0] if m >= 0 else SERIES[7] for m in by['mean']]
    ax.bar(by.index, by['mean'], width=1.2, color=colors, yerr=1.96 * by['sem'],
           error_kw={'ecolor': MUTED, 'elinewidth': 1, 'capsize': 3})
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(by.index)
    ax.set_xlabel('score at the close (sum of four signs)')
    ax.set_ylabel('next-day return, in trailing daily σ (mean ± 95%)')
    ax.set_title(f'Does the score predict the next day? t = {ts["t"]:.1f} '
                 f'({ts["t_ex_btc"]:.1f} after BTC beta), {ts["n"] / 1e3:.0f}k coin-days',
                 loc='left', fontweight='bold')
    fig.savefig(fig_dir / 'momentum_score_diagnostic.png', bbox_inches='tight')
    plt.close(fig)

    # 5. costs: the best book under other fee assumptions
    cost_runs = run_jobs([(s, c, best_params) for c in costs if c != COST_PER_SIDE
                          for s in symbols], quick) + by_cell[best_cell]
    cost_rows = {}
    for c in costs:
        _, f = book_summary([r for r in cost_runs if r['cost'] == c], btc)
        cost_rows[f'{c * 1e4:.0f} bp per side'] = {k: f[k] for k in
                                                   ('Return [%]', 'CAGR [%]', 'Sharpe Ratio',
                                                    'Max. Drawdown [%]', 'Costs [% final eq.]')}
    cost_table = pd.DataFrame(cost_rows).T
    print_table(cost_table, 'Best book vs cost per side (fee + slippage)')
    save_table(cost_table, 'momentum_costs')

    # 6. the framework's tearsheet of the BTC sleeve under the best cell
    bt = sleeve_backtest(coin_tape(panel['BTC']))
    bt.run(**best_params)
    bt.plot(filename=str(fig_dir / 'momentum_tearsheet.html'), open_browser=False,
            resample=False)

    print(f'\nBest: {label_best} -> Sharpe {best_facts["Sharpe Ratio"]:.2f}, CAGR '
          f'{best_facts["CAGR [%]"]:.2f}%, vol {best_facts["Volatility (Ann.) [%]"]:.1f}%, max DD '
          f'{best_facts["Max. Drawdown [%]"]:.1f}%, {best_facts["# Trades"]} trades, win rate '
          f'{best_facts["Win Rate [%]"]:.0f}%, costs {best_facts["Costs [% final eq.]"]:.0f}% of '
          f'final equity, corr. to BTC {best_facts["Corr. to BTC"]:.2f} (video default '
          f'{default_facts["Sharpe Ratio"]:.2f}, vol-scaled long only '
          f'{long_facts["Sharpe Ratio"]:.2f}, BTC {float(btc_st["Sharpe Ratio"]):.2f})')
    print(f'figures -> {fig_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true', help='4 coins, 2x2 grid (CI)')
    main(ap.parse_args().quick)

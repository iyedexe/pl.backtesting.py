# Strategy examples — three strategies on the backtesting.py framework

Each script defines a `backtesting.Strategy`, runs it through `Backtest`
(execution, costs, equity, statistics), sweeps its parameters with the
framework's own `bt.optimize`, and saves the graphs of the best-performing
configuration to `figures/` (plus the framework's interactive tearsheet of the
best run as `*_tearsheet.html`) and the tables to `tables/`. The
strategy-specific machinery the framework cannot provide — cointegration
statistics, option pricing, the index-market simulator — lives in the support
packages next to the scripts (`pairs_trading/`, `pmcc/`, `index_inclusion/`).

```bash
uv run python examples/pairs_trading_strategy.py     # ~5 min
uv run python examples/pmcc_strategy.py              # ~3 min (multi-process)
uv run python examples/index_inclusion_strategy.py   # ~3 min
```

Add `--quick` for a small smoke run (what CI executes).

| | Pairs trading | Poor Man's Covered Call | Index inclusion |
|---|---|---|---|
| Script | `pairs_trading_strategy.py` | `pmcc_strategy.py` | `index_inclusion_strategy.py` |
| Strategy class | `PairsTrading` | `PMCCRegime` | `IndexInclusion` |
| Instrument fed to the framework | walk-forward hedged-spread tape (one pair) | the PMCC package's daily total-return index | stitched tape of screened candidates |
| What `next()` decides | z-score entries/exits/stops, force-flat per window | when to be in the package (vol regime) | buy announced/predicted additions, hold ≤ N sessions |
| `bt.optimize` over | entry × exit × z-window | exit × re-entry vol thresholds | holding period (per screener setting) |
| Best found | 2.5σ / 0 / 20-bar, Sharpe **0.52** (default 0.35) | vol > 0.35 out / < 0.30 in, Sharpe **0.77** (always-in 0.73) | hold ≤ 25 / buffer 2 / 15 d, Sharpe **1.39** (default 1.27) |
| …and does it hold up? | best keeps the default's 8% drawdown with fewer trades; NW t ≈ 3 on the default | **fails on 10/10 single names** — always-in is the honest answer | **0.86 vs 0.86 across fresh market seeds** — the peak is noise |

## 1. Pairs trading — `pairs_trading_strategy.py`

`backtesting.py` trades one instrument, so the pair becomes one: for each
walk-forward window the hedged spread `A/A₀ − β·B/B₀` (β from OLS on log
prices of the preceding 252-day *formation* window only; A₀, B₀ the window's
first prices), shifted to stay positive. A position of N units is then exactly
long N/A₀ shares of A and short N·β/B₀ shares of B — the two-leg P&L by
construction, with gross notional N·(A/A₀ + β·B/B₀) charged per leg through a
commission callback. The tape also carries the causal z-scores for several
windows, the formation-period cointegration gate (Engle-Granger p < 0.05,
half-life 1–60 bars), the window id and an exit flag two bars before every
β switch, so no trade ever spans two windows. `PairsTrading.next()` is the
usual rule: enter beyond ±entry while the gate is open, exit through the exit
level, stop beyond ±4σ, re-arm only once |z| is back inside the entry band.

Cross-check against the independent two-leg engine in `pairs_trading/engine.py`
(same signals, same lag): WTI/Brent, default rule — **79 vs 79 trades, return
105.3% vs 105.9%**, Sharpe 0.35 vs 0.39 (the framework annualizes with a zero
risk-free rate and its own day count).

| Ranking | `bt.optimize` sweep | Best vs default |
|---|---|---|
| ![](figures/pairs_ranking.png) | ![](figures/pairs_best_grid.png) | ![](figures/pairs_best_equity.png) |

Every featured pair of the five-asset-class study runs under the default rule
(the ranking chart labels each bar with its Newey-West t-statistic and trade
count); the most robust pair — WTI/Brent, t ≈ 3.2 — is then swept over entry
{1…3σ} × exit {0, 0.5} × z-window {20…120}. **Best: enter 2.5σ, exit 0,
20-bar window — Sharpe 0.52 vs 0.35 for the literature default**, with the same
8% drawdown and fewer, cleaner trades (41 vs 79); its total return is actually
lower (87% vs 105%) — the higher Sharpe comes from sitting out more noise, not
from earning more. Only two other pairs have enough trades to say anything
(CAD/WTI, t = 1.6; NATGAS/WTI, 4 trades); the famous stock, FX and crypto
pairs pass the walk-forward cointegration gate a handful of times in a decade.

## 2. Poor Man's Covered Call — `pmcc_strategy.py`

The framework cannot hold option legs, so the mechanical package (deep-ITM
LEAPS ≥ 540 days out at ~0.80Δ, short ~0.25Δ monthly call, both rolled by
rule) is priced day by day into a total-return index per stock by the `pmcc`
engine — the CBOE BXM idea — and `backtesting.py` trades the index. The
decision left to the framework is the one the package lacks: *when to be in
it*. `PMCCRegime` steps aside when the underlying's trailing 21-day realized
volatility exceeds `vol_exit` and re-enters below `vol_reenter`; switching
costs 1% per side of index notional (unwinding/re-opening the option package
at 2% half-spreads on option value). `bt.optimize` sweeps both thresholds
(including the always-in cell), `MultiBacktest` re-runs the best filter on
every name, and the covered-call and buy-and-hold indices run through the same
`Backtest` as benchmarks.

| `bt.optimize` sweep | Best vs benchmarks | `MultiBacktest` per name |
|---|---|---|
| ![](figures/pmcc_best_grid.png) | ![](figures/pmcc_best_equity.png) | ![](figures/pmcc_per_ticker.png) |

Equal-weight portfolio of the 10 most option-liquid French names, 2000–2015:
the optimizer's best filter (out above 0.35, back in below 0.30) posts Sharpe
**0.77 vs 0.73 for the always-in PMCC**, 0.66 for the covered call and 0.37 for
buy & hold — but **it loses to always-in on all 10 single names** (e.g. Air
Liquide 0.63 vs 0.80). The portfolio-level gain comes from ten well-timed
switches on a cross-sectional vol proxy and is the kind of edge one should not
believe; the honest reading is that no volatility timing improves the
mechanical PMCC, whose own structure (defined-risk LEAPS floor, cash buffer)
already halves the buy-and-hold drawdown. Note the framework's Sharpe uses a
zero risk-free rate; the PMCC study's own statistics (`pmcc/results/`) are
excess-over-EURIBOR, hence lower.

## 3. Index inclusion — `index_inclusion_strategy.py`

On the tutorial's synthetic point-in-time market (220 stocks, an FTSE-100-style
rulebook, a planted ~5% index effect), the look-ahead-free screener stitches
the candidates into one tape and `IndexInclusion` buys announced or predicted
additions one at a time, exiting after the effective day or `hold_limit`
sessions. For every screener setting (prediction buffer × how many sessions
before the cutoff to start hunting) the framework's `bt.optimize` sweeps the
holding period; the best configuration is then re-run on five fresh market
seeds next to the tutorial default.

| `bt.optimize` sweep | Best vs default | Fresh market seeds |
|---|---|---|
| ![](figures/index_best_grid.png) | ![](figures/index_best_equity.png) | ![](figures/index_seeds.png) |

**Best on the tutorial's seed: hold ≤ 25, buffer 2, hunt 15 days before the
cutoff — Sharpe 1.39** (43 trades, 79% win rate) vs the default's 1.27. The
heatmap shows the real structure: Sharpe plateaus once the holding period
reaches the effective day (~15 sessions) and a deeper buffer only forfeits
trades. **On fresh seeds the best and the default are indistinguishable (mean
Sharpe 0.86 vs 0.86)** — the peak was fitted to one draw of the simulator.
Absolute levels reflect a 1990s-sized planted effect; the real edge in today's
mega-cap indices is far smaller.

# pl.backtesting.py — three trading strategies, backtested honestly

A research fork of [backtesting.py](https://github.com/kernc/backtesting.py)
that bundles **three independent strategy studies**, each with its own
package, vendored data, tests and write-up, in one
[uv](https://docs.astral.sh/uv/)-managed project on **Python 3.13**:

| Strategy | What it trades | Where it's tested | Verdict (details below) |
|---|---|---|---|
| **Pairs trading** (`pairs_trading/`) | the spread between two cointegrated assets, walk-forward | crypto, US stocks, forex, commodities, cross-asset; 1971→2026 | only the WTI/Brent oil spread survives out of sample; famous stock/FX/crypto pairs don't |
| **Poor Man's Covered Call** (`pmcc/`) | deep-ITM LEAPS call + short monthly call | 17 French large caps, 2000→2015, synthetic option chains | covered-call-like returns with roughly half the drawdown, ~2 pp/yr less than the covered call |
| **Index inclusion** (`index_inclusion/`, `bot/`) | stocks about to be added to a rules-based index | synthetic point-in-time market with an FTSE-100 rulebook; live bot for S&P 500 / Nasdaq-100 / FTSE 100 / DAX 40 | the machinery captures the planted index effect; the real edge today is small in mega caps |

`examples/` contains one runnable script per strategy that sweeps its
parameters and plots the **best-performing configuration**
([examples/README.md](examples/README.md)):

| Pairs trading | Poor Man's Covered Call | Index inclusion |
|---|---|---|
| ![](examples/figures/pairs_best_equity.png) | ![](examples/figures/pmcc_best_equity.png) | ![](examples/figures/index_best_equity.png) |
| ![](examples/figures/pairs_best_grid.png) | ![](examples/figures/pmcc_best_grid.png) | ![](examples/figures/index_best_grid.png) |

## Quickstart

```bash
uv sync                                   # Python 3.13 env from uv.lock (uv fetches the interpreter)
uv run pytest                             # all strategy unit tests (pairs, pmcc, bot)
uv run python -m backtesting.test         # the upstream library's own suite

uv run python examples/run_pairs_trading.py     # best pairs configuration + figures
uv run python examples/run_pmcc.py              # best PMCC configuration + figures
uv run python examples/run_index_inclusion.py   # best index-inclusion configuration + figures

uv run pairs all                          # full five-asset-class pairs study -> research/reports/report.md
uv run pmcc all                           # full PMCC headline + 800-run sensitivity -> pmcc/results/
uv run inclusion-bot selftest             # offline demo cycle of the Telegram bot
```

Everything needed to reproduce every number is in the repository (vendored
daily panels, ~5 MB for pairs, ~300 KB for PMCC; the index study simulates its
market) — no API keys or network access required. Add `--quick` to any example
for a small smoke run.

## The three strategies

### 1. Pairs trading across asset classes — `pairs_trading/`

Buy one asset, short its statistical twin when the spread is stretched, bet on
convergence. The classic Engle-Granger / z-score version of the strategy,
rebuilt with the discipline the literature demands — **walk-forward
everything** (hedge ratio, cointegration gate and z-parameters estimated on a
252-day formation window, traded on the next 63 days), next-close execution,
per-leg costs, a proper two-leg engine, deflated-Sharpe grids — and run on real
daily data: 13 cryptos, 505 S&P 500 names, 11 currencies, WTI/Brent/gas, and
cross-asset pairs (petro-currency, BTC vs tokenized gold).

**Findings** ([full report](research/reports/report.md)): once the future is
kept out of the estimation, the formation-window cointegration gate opens on
only 4–12% of windows for famous stock/FX/crypto pairs — the test's own 5%
false-positive rate (Clegg 2014). The one robust survivor is **WTI/Brent**:
out-of-sample Sharpe 0.39, Newey-West t = 3.3, +94 bp per round trip over 35
years, deflated-Sharpe probability 0.97, mostly before the 2011 Cushing break.
Daily crypto pairs lose money even before costs (top-3 portfolio CAGR −24%,
short-leg blowups in alt manias); a re-selected top-10 S&P 500 pair portfolio
nets a Sharpe of 0.02 (the Do & Faff decay); same-close execution doubles
measured Sharpe (Gatev-Goetzmann-Rouwenhorst's "wait one day" artifact,
reproduced).

Package map: `data.py` (vendored panels), `stats.py` (Engle-Granger, ADF,
half-life, Hurst, Kalman hedge), `signals.py` (z-score state machine),
`engine.py` (two-leg dollar-neutral backtester), `walkforward.py`
(formation/trading harness, top-N portfolio re-selection), `screening.py`,
`metrics.py` (Newey-West, deflated Sharpe), `experiments.py`, `cli.py`
(`pairs`). Data provenance and licenses: [research/data/SOURCES.md](research/data/SOURCES.md)
(the crypto panel is Coin Metrics community data, CC BY-NC 4.0).

### 2. Poor Man's Covered Call — `pmcc/`

Replace the 100 shares of a covered call with a deep-in-the-money LEAPS call
(~0.80 delta, 18–24 months out) and sell the same ~0.25-delta monthly call
against it, rolling both legs mechanically. Backtested on 17 CAC 40 large caps
over 2000–2015 with **synthetic Black-Scholes option chains** (no free
historical Euronext chains exist), a volatility-risk-premium model driven by
VIX/realized vol, dividends, EUR rates, French FTT, spreads and commissions —
against covered-call, buy-and-hold and LEAPS-only benchmarks priced with the
same model, plus an 800-run sensitivity grid.

**Findings** ([pmcc/README.md](pmcc/README.md), [results](pmcc/results/RESULTS.md)):
equal-weight portfolio of the 10 most option-liquid names, net of costs —
PMCC CAGR 12.9% with a −23% max drawdown vs covered call 16.5% / −41% and buy
& hold 11.1% / −50%. The PMCC held up structurally through two −50% bear
markets, but French dividend yields hand the classic covered call ~2 pp/yr
more; everything rides on implied vol trading rich to realized; the leveraged
sizing variant is a ruin machine (−58% portfolio, −95% single-name).
`pmcc/live/` holds a paper-trading executor that reuses the same decision code.

### 3. Index inclusion — `index_inclusion/`, `doc/examples/`, `bot/`

When a stock is added to a major index, every tracker must buy it by the
effective date; rank-based rulebooks (FTSE 100, Nasdaq-100, Russell) make the
additions **computable before they are announced**. The tutorial notebook
([doc/examples/Index Inclusion Strategy.py](doc/examples/Index%20Inclusion%20Strategy.py))
builds a synthetic point-in-time market of 220 stocks with a fictional
"SIX 100" index under the FTSE 100 rulebook, injects a 1990s-sized index
effect, screens weekly without look-ahead, and trades one stock at a time
through `backtesting.py` on a stitched tape. `index_inclusion/` is that
machinery as an importable package; `bot/` (`inclusion-bot`, a uv workspace
member) applies the same screens to live data and pushes buy/sell signals to
Telegram ([bot/README.md](bot/README.md)).

**Findings**: the screener captures almost exactly the planted ~5% effect
(high win rate, ~25% exposure, every position closed within a month); the
holding-period knob plateaus once the effective day is reached, and demanding
a deeper prediction buffer only forfeits trades. On today's US mega caps the
real announcement-to-effective effect is mostly arbitraged away
(Greenwood & Sammon 2025); smaller and less-arbitraged indices retain more.

## Layout

```
backtesting/         the upstream backtesting.py library (kept working; AGPL-3.0)
pairs_trading/       pairs-trading research lab (+ research/data, research/reports)
pmcc/                Poor Man's Covered Call lab (+ data/, results/, live/)
index_inclusion/     index-inclusion strategy package (simulator, screener, strategy)
bot/                 inclusion-bot: Telegram index-inclusion signals (uv workspace member)
examples/            one best-configuration script per strategy + figures/
doc/examples/        upstream tutorials + the Index Inclusion Strategy notebook
tests/               pairs_trading test suite (pmcc/test_pmcc.py, bot/tests: theirs)
scripts/             data fetch/build scripts for the vendored pairs panels
```

## Development

```bash
uv sync --extra test --extra doc     # everything incl. the library's test/doc extras
uv run flake8 backtesting pairs_trading pmcc scripts tests examples
uv run ruff check pairs_trading scripts tests examples
uv run mypy --no-warn-unused-ignores backtesting
uv run doc/build.sh                  # API docs + rendered example notebooks
```

CI (`.github/workflows/ci.yml`) runs lint, the library suite on 3.13/3.14, the
three strategy test suites, the bot self-test, and a `--quick` smoke run of
each example script.

## Relationship to upstream

This is a fork of [kernc/backtesting.py](https://github.com/kernc/backtesting.py)
(AGPL-3.0). The original README is preserved at
[doc/README_upstream.md](doc/README_upstream.md); the library itself is
unchanged except for packaging (PEP 621 / uv, Python ≥ 3.13) and a few small
fixes listed in the [CHANGELOG](CHANGELOG.md). It still works exactly as
documented upstream:

```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


bt = Backtest(GOOG, SmaCross, commission=.002,
              exclusive_orders=True)
stats = bt.run()
bt.plot()
```

Results in:

```text
Start                     2004-08-19 00:00:00
End                       2013-03-01 00:00:00
Duration                   3116 days 00:00:00
Exposure Time [%]                       94.27
Equity Final [$]                     68935.12
Equity Peak [$]                      68991.22
Return [%]                             589.35
Buy & Hold Return [%]                  703.46
Return (Ann.) [%]                       25.42
Volatility (Ann.) [%]                   38.43
CAGR [%]                                16.80
Sharpe Ratio                             0.66
Sortino Ratio                            1.30
Calmar Ratio                             0.77
Alpha [%]                              450.62
Beta                                     0.02
Max. Drawdown [%]                      -33.08
Avg. Drawdown [%]                       -5.58
Max. Drawdown Duration      688 days 00:00:00
Avg. Drawdown Duration       41 days 00:00:00
# Trades                                   93
Win Rate [%]                            53.76
Best Trade [%]                          57.12
Worst Trade [%]                        -16.63
Avg. Trade [%]                           1.96
Max. Trade Duration         121 days 00:00:00
Avg. Trade Duration          32 days 00:00:00
Profit Factor                            2.13
Expectancy [%]                           6.91
SQN                                      1.78
Kelly Criterion                        0.6134
_strategy              SmaCross(n1=10, n2=20)
_equity_curve                          Equ...
_trades                       Size  EntryB...
dtype: object
```

None of this is investment advice; every study documents the assumptions
(synthetic options, indicative marks, planted effects, survivorship) that its
numbers rest on.

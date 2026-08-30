# pl.backtesting.py — a pairs-trading research lab

A research project on the classic **pairs trading** (statistical arbitrage)
strategy — buy one asset, short a related one when the spread between them is
stretched, bet on convergence — backtested with walk-forward discipline on
**five asset classes**:

| Class | Universe | Period | Example pairs |
|---|---|---|---|
| Crypto | 13 majors (Coin Metrics daily) | 2016 → 2026 | BTC/ETH, LTC/ETH |
| US stocks | 505 S&P 500 constituents | 2013 → 2018 (+ 20 large caps 1990 → 2018) | KO/PEP, JPM/BAC, GOOGL/GOOG |
| Forex | 11 USD rates (Fed H.10) | 1999 → 2026 | AUD/NZD, SEK/NOK, EUR/GBP |
| Commodities | WTI, Brent, Henry Hub (EIA spot) | 1990 → 2026 | WTI/Brent (+ gas/oil negative control) |
| Cross-asset | oil × petro-FX, BTC × tokenized gold | 1990/2020 → 2026 | CAD/WTI, BTC/PAXG |

**→ The full write-up with all tables and figures is in
[`research/reports/report.md`](research/reports/report.md).**

The repository is a fork of [kernc/backtesting.py](https://github.com/kernc/backtesting.py)
extended with a `pairs_trading` package; the bundled `backtesting` library is
kept fully working (its own test suite passes) and is used as an independent
cross-check engine for the pair signals.

## Quickstart

Everything is managed with [uv](https://docs.astral.sh/uv/); the data needed to
reproduce every number is vendored in the repo (~5 MB), so no network or API
keys are required:

```bash
uv sync                          # create the environment from uv.lock
uv run pytest tests              # unit tests (engine P&L identities, stats recovery)
uv run pairs screen --study crypto        # scan a universe for cointegrated pairs
uv run pairs run --study commodities      # one study end to end
uv run pairs all                 # all five studies + assemble the report (~10 min)
```

Refresh the vendored data (network required):

```bash
scripts/fetch_sources.sh /tmp/pairs-src
uv run python scripts/build_vendored_data.py --src /tmp/pairs-src
```

## What the lab does

1. **Screen** every universe mechanically: correlation prefilter → Engle-Granger
   cointegration test (both orientations) → spread half-life (OU fit) and Hurst
   exponent, with an explicit multiple-testing warning (Clegg 2014).
2. **Backtest out-of-sample** with the Gatev-Goetzmann-Rouwenhorst walk-forward
   template: hedge ratio, cointegration gate and z-parameters estimated on a
   252-bar *formation* window, traded on the next 63 bars, rolled forward;
   positions force-flat at window boundaries; execution one bar after the
   signal; per-leg proportional costs (12/5/2/5 bp per side for
   crypto/stocks/FX/commodities).
3. **Model the pair properly**: a two-leg dollar-neutral engine (share
   quantities, per-leg costs, trades ledger) rather than a synthetic-ratio
   approximation — plus a [`backtesting.py`](doc/README_upstream.md) adapter
   as an independent implementation cross-check.
4. **Stress everything**: entry/exit threshold grids with deflated-Sharpe
   correction, cost sweeps 0×→4×, same-close vs next-close execution (the GGR
   "wait one day" test), OLS vs Kalman hedge ratios, gate on/off, and
   Clegg-style gate-persistence tables.
5. **Portfolio mode**: for the big universes (S&P 500, crypto), pairs are
   re-selected every window from formation data only — a fully out-of-sample,
   Gatev-style top-N pair portfolio.

## Layout

```
backtesting/         the upstream backtesting.py library (kept working; AGPL-3.0)
pairs_trading/       the research lab
  data.py            loaders for the vendored panels
  stats.py           Engle-Granger, ADF, half-life, Hurst, Kalman hedge
  signals.py         z-score hysteresis state machine (entry/exit/stop/re-arm)
  engine.py          two-leg dollar-neutral backtest engine
  walkforward.py     formation/trading harness + top-N portfolio re-selection
  screening.py       universe-wide cointegration scans
  metrics.py         Sharpe/Sortino/MDD, Newey-West t-stats, deflated Sharpe
  experiments.py     the five studies
  btpy_adapter.py    cross-check through backtesting.py
  report.py, cli.py  report assembly and the `pairs` CLI
research/
  data/vendored/     committed input panels (+ SOURCES.md provenance & licenses)
  reports/           generated report.md, figures/, tables/
scripts/             data fetch/build scripts
tests/               pytest suite (simulated ground truth for every estimator)
```

## Method notes (why it's built this way)

- **Walk-forward everything.** Estimating β or z-parameters on the full sample
  plants the future in every signal; all performance tables here use
  formation-only estimation, and universe *selection* is also re-done per
  window in portfolio mode.
- **Costs and execution lag are on by default.** GGR showed ~200 bp/yr of
  distance-method "profit" disappears with one day of execution delay; the
  same experiment is reproduced here per pair.
- **The cointegration gate is honest.** A pair trades a window only if it
  passed the test *before* that window — and the report shows how often gates
  that pass keep passing (mostly: not often — consistent with Clegg 2014).
- **Two independent engines.** The two-leg engine is the source of truth; the
  bundled backtesting.py runs the same rule on the price ratio as a plumbing
  cross-check.

Full methodology, results, caveats and references:
[`research/reports/report.md`](research/reports/report.md).

## Relationship to upstream

Fork of [kernc/backtesting.py](https://github.com/kernc/backtesting.py)
(AGPL-3.0; original README preserved at
[`doc/README_upstream.md`](doc/README_upstream.md)). Changes to the library
itself are minimal: packaging modernized to PEP 621/uv, a pandas ≥ 3
copy-on-write fix in `FractionalBacktest`, and two typing annotations; the
library's full test suite passes. Everything else lives in the new
`pairs_trading` package.

Data licensing: see [`research/data/SOURCES.md`](research/data/SOURCES.md) —
note the crypto panel (Coin Metrics community data) is **CC BY-NC 4.0**
(non-commercial, attribution).

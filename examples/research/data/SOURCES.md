# Data provenance

All datasets in `vendored/` are derived from public sources with
`scripts/fetch_sources.sh` (raw download) + `scripts/build_vendored_data.py`
(normalization). Exact source commits are pinned in `vendored/manifest.json`.
They are committed to the repository so that the whole research pipeline is
reproducible offline; refresh them by re-running the two scripts.

| File | Asset class | Contents | Range | Source | Underlying data | License |
|---|---|---|---|---|---|---|
| `crypto_usd_daily.csv.gz` | Crypto | Daily reference `PriceUSD` for BTC, ETH, LTC, BCH, XRP, ADA, DOGE, DOT, XMR, LINK, ETC, BNB, PAXG | 2010‑07 → 2026‑05 | [coinmetrics-io/data](https://github.com/coinmetrics-io/data) | Coin Metrics Community Data | **CC BY‑NC 4.0** (attribution, non‑commercial) |
| `fx_usd_daily.csv.gz` | Forex | Daily USD price of 21 currencies (AUD, NZD, EUR, GBP, CHF, SEK, NOK, DKK, CAD, JPY, SGD, …) | 1971‑01 → 2026‑08 | [datasets/exchange-rates](https://github.com/datasets/exchange-rates) | US Fed H.10 noon rates via FRED | PDDL‑1.0 |
| `commodities_daily.csv.gz` | Commodities | Daily spot: WTI, Brent, Henry Hub natural gas | 1986‑01 → 2026‑08 | [datasets/oil-prices](https://github.com/datasets/oil-prices), [datasets/natural-gas](https://github.com/datasets/natural-gas) | US EIA | PDDL‑1.0 |
| `sp500_close_daily.csv.gz` | Stocks | Daily close for 505 S&P 500 constituents | 2013‑02 → 2018‑02 | [CNuge/kaggle-code](https://github.com/CNuge/kaggle-code) | Kaggle “S&P 500 stock data” (camnugent) | CC0 |
| `sp500_ohlcv_selected.csv.gz` | Stocks | Full daily OHLCV for 32 famous‑pair tickers | 2013‑02 → 2018‑02 | [CNuge/kaggle-code](https://github.com/CNuge/kaggle-code) | Kaggle “S&P 500 stock data” (camnugent) | CC0 |
| `stocks_1990_2018_daily.csv.gz` | Stocks | Daily adjusted close, 20 large caps (JPM, BAC, XOM, MA, …) | 1990 → 2018 | [robertmartin8/PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) test fixture | Yahoo Finance (adjusted) | MIT (repo) |

Notes and caveats:

- **Coin Metrics community data is CC BY‑NC 4.0**: this repository uses it for
  non‑commercial research with attribution ("Data provided by Coin Metrics
  Community Data"). Remove `crypto_usd_daily.csv.gz` if you intend commercial use.
- FX rates in the source are quoted as *units of national currency per USD*
  (verified programmatically at build time); the vendored panel is normalized to
  the **USD price of one unit of foreign currency**, so every panel in the project
  is a positive "asset price in USD" and log‑spreads are directly comparable.
- FX and commodity series are *indicative daily marks* (noon fixings / spot
  assessments), not tradable bid/ask quotes; oil series are spot prices rather
  than a rolled futures curve (no roll yield). Results on these classes measure
  statistical structure, not fully executable P&L — see the report's caveats.
- Equity closes from the Kaggle dataset are split‑adjusted but **not
  dividend‑adjusted**; the 1990‑2018 panel *is* dividend‑adjusted. Long/short
  pair P&L on the 5‑year panel therefore ignores dividend flows on both legs
  (partially offsetting for same‑sector pairs).
- Crypto "prices" are Coin Metrics reference rates (volume‑weighted across
  venues), a standard research proxy for executable mid prices.

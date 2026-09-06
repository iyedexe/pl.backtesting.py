# newsbot — news-driven long-only equity bot (max one-week hold)

`newsbot` listens to company news (earnings, guidance, FDA decisions, upgrades, buybacks,
takeovers, ...), scores each headline, and opens a **long** position when the score clears a
threshold. Every position carries three exits from the moment it is filled:

| exit   | rule                                                            |
|--------|-----------------------------------------------------------------|
| target | price >= fill × (1 + `target_pct`)  (5 % by default, wider for stronger catalysts) |
| stop   | price <= fill × (1 − `stop_pct`)    (3 % by default)            |
| time   | `max_hold_days` (7 calendar days = one week) after the fill      |

The same code path runs in three modes:

* **backtest** — the rules as a `backtesting.Strategy` (`NewsStrategy`), using intraday high/low
  for stops/targets and per-ticker statistics;
* **replay** — the live bot loop (sources → signals → paper broker → exits) driven by a simulated
  clock over CSV prices and a news file;
* **run** — live polling of RSS / Yahoo / Alpaca news, paper or real Alpaca account.

## Quick start

```bash
pip install -e .            # backtesting.py + numpy/pandas/bokeh
pip install pyyaml requests # config files and RSS/Alpaca (optional: yfinance, anthropic)

# 1. score a headline
python -m newsbot classify "Acme beats estimates, raises full-year guidance"

# 2. backtest on the bundled GOOG data with the synthetic sample headlines
python -m newsbot backtest --prices GOOG=backtesting/test/GOOG.csv --news newsbot/data/sample_news.json --plot

# 3. replay the live pipeline over the same data
python -m newsbot replay --prices GOOG=backtesting/test/GOOG.csv --news newsbot/data/sample_news.json

# 4. paper-trade live headlines
cp config.example.yaml config.yaml   # edit universe / sources / broker
python -m newsbot run --config config.yaml        # Ctrl-C to stop; state survives restarts
python -m newsbot status --state newsbot_state.json
```

`newsbot/data/sample_news.json` is **synthetic**: headlines were placed on real GOOG trading days
purely to exercise the pipeline. Its backtest result says nothing about the strategy's edge. Supply
your own history (same JSON/CSV schema: `published`, `tickers`, `headline`, `summary`) for a real test.

## Architecture

```
sources.py      NewsSource      -> FileNewsSource | RSSNewsSource | YahooFinanceRSS | AlpacaNewsSource
classifiers/    Classifier      -> RuleClassifier (regex, no deps) | ClaudeClassifier (anthropic SDK)
signals.py      SignalEngine    news + classification -> Signal(ticker, target_pct, stop_pct, max_hold_days, ttl)
prices.py       PriceFeed       -> StaticPriceFeed | CSVPriceFeed | YFinancePriceFeed | AlpacaPriceFeed
brokers.py      Broker          -> PaperBroker | AlpacaBroker (bracket orders, paper endpoint by default)
bot.py          NewsTradingBot  tick(): poll_news -> manage_positions -> execute_signals -> save state
state.py        BotState        JSON persistence (positions, pending signals, seen ids, trade log)
strategy.py     NewsStrategy    backtesting.py Strategy + run_news_backtest() for many tickers
replay.py       run_replay()    live loop over history with SimClock
config.py       build_bot()     wire everything from a YAML/JSON dict
```

### Signal rules (`SignalEngine`)

* `RuleClassifier` sums the weights of every matching pattern (`classifiers/rules.py`) and clamps to
  [−1, 1]; bearish patterns are negative so "beats estimates **but cuts guidance**" nets negative.
  Add rows to `DEFAULT_RULES` (regex, category, weight) to extend it.
* A signal is created when `score >= min_score` and the ticker is in `universe` (if set) and the
  category passes the allow/deny lists.
* `target_pct` scales with the score when `scale_target_by_score` is on: ×1.0 at 0.5, ×1.25 at 1.0.
* Signals live for `signal_ttl_hours` (18 h) so an after-close earnings release is bought at the
  next open; `max_chase_pct` skips the entry if price already gapped more than 5 % above the price
  seen when the signal was created.

### Live loop (`NewsTradingBot.tick`)

1. fetch headlines newer than `now − max_news_age` from every source, drop already-seen ids;
2. score and queue signals (one per ticker; no duplicates for a ticker already held/queued);
3. for each open position: if the broker manages exits (Alpaca brackets) detect the leg fill,
   otherwise sell on target/stop; sell on time when `now >= exit_by`;
4. fill queued signals, best score first, while `len(positions) < max_positions` and the market
   is open; size = `position_size_pct × equity`, optionally capped by `risk_per_trade_pct / stop_pct`;
5. persist state.

`market_hours_only` uses a simple Mon–Fri 09:30–16:00 New York session for the paper broker
(no holiday calendar) and the exchange clock for Alpaca.

### Backtest semantics (`NewsStrategy`)

* A headline published at T is scored on the last bar with timestamp ≤ T and bought at the
  **next bar's open** (daily data: an after-close release on day D fills at D+1 open) with a limit
  at close × (1 + `max_chase_pct`).
* Stop and target are attached on the entry bar from the actual fill price and evaluated on
  intraday high/low by the engine; a gap through the stop fills at the open (realistic).
* `hold_bars = 5` closes at the open of the 6th session (≈ one trading week);
  set `max_hold_days=7` for a strict calendar limit.

```python
from backtesting.test import GOOG
from newsbot import FileNewsSource, run_news_backtest

res = run_news_backtest({'GOOG': GOOG}, FileNewsSource('newsbot/data/sample_news.json').items,
                        min_score=0.6, target_pct=0.06, stop_pct=0.03, hold_bars=5)
print(res['summary']); res['backtests']['GOOG'].plot()
```

`NewsStrategy` is a normal `backtesting.Strategy`, so `Backtest.optimize(min_score=[...], target_pct=[...])`
works for parameter sweeps.

## Live trading with Alpaca

```bash
export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
```
```yaml
sources: [{type: alpaca}]
prices:  {type: alpaca}
broker:  {type: alpaca, paper: true}
```
Entries are bracket market orders (take-profit limit + stop). The bot cancels the bracket legs and
closes the position itself when the one-week deadline hits. Start with `paper: true`.

## LLM classification (optional)

`pip install anthropic`, set `ANTHROPIC_API_KEY`, and use `classifier: {type: claude}`. The model
returns strict JSON (`score`, `category`, `confidence`, `reasons`); on any API failure the rule
classifier is used so the loop keeps running.

## Tests

```bash
python -m newsbot.test
```

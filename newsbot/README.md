# newsbot — news-driven long-only equity bot (max one-week hold)

`newsbot` gathers evidence about each ticker from many sources (news APIs, press-release feeds,
provider sentiment, reported earnings vs consensus, FDA and clinical-trial events, social chatter),
aggregates it per ticker, scores the whole bundle with an AI model (or a deterministic rule ensemble),
and turns qualifying scores into **long** signals. A signal can place the order itself, send a
Telegram message, hit a webhook, or any combination. Every position carries three exits from the
moment it is filled:

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
* **run** — live polling of every configured source, paper or real Alpaca account, or notify-only.

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
cp config.example.yaml config.yaml   # edit universe / sources / scoring / actions / broker
python -m newsbot sources --config config.yaml    # fetch every source once, see what comes back
python -m newsbot run --config config.yaml        # Ctrl-C to stop; state survives restarts
python -m newsbot status --state newsbot_state.json

# 5. build a real backtest dataset (free Alpaca paper account) and dry-run the scorer on it
python -m newsbot backfill --source alpaca --tickers AAPL,MSFT --start 2022-01-01 --out news.json
python -m newsbot backtest --prices AAPL=aapl.csv --news news.json
python -m newsbot score --config config.yaml --news news.json --ticker AAPL --as-of 2023-05-05T00:00:00Z
```

`newsbot/data/sample_news.json` is **synthetic**: headlines were placed on real GOOG trading days
purely to exercise the pipeline. Its backtest result says nothing about the strategy's edge. Supply
your own history (same JSON/CSV schema: `published`, `tickers`, `headline`, `summary`) for a real test.

## Architecture

```
sources.py      NewsSource      -> File | RSS | YahooFinanceRSS | GoogleNewsRSS | NasdaqRSS
providers/      APISource       -> Finnhub (news, earnings calendar) | AlphaVantage sentiment | Polygon |
                                   Marketaux | NewsAPI | FMP (news, calendar) | Nasdaq calendar |
                                   FDA press RSS | openFDA approvals | ClinicalTrials.gov | StockTwits | Reddit
alpaca.py       Alpaca          news (+ historical backfill), latest prices, bracket-order broker
aggregator.py   EvidenceStore   per-ticker window of NewsItems -> Bundle (items + numeric features)
scoring.py      Scorer          -> RuleScorer (ensemble) | ClaudeScorer (AI reads the whole bundle)
classifiers/    Classifier      per-headline regex scoring (bundle features + backtests)
signals.py      SignalEngine    score -> Signal(ticker, target_pct, stop_pct, max_hold_days, ttl)
actions.py      Action          -> TradeAction | TelegramAction | WebhookAction | LogAction
prices.py       PriceFeed       -> StaticPriceFeed | CSVPriceFeed | YFinancePriceFeed | AlpacaPriceFeed
brokers.py      Broker          -> PaperBroker | AlpacaBroker (bracket orders, paper endpoint by default)
bot.py          NewsTradingBot  tick(): poll -> store -> score dirty tickers -> actions -> exits -> entries
state.py        BotState        JSON persistence (positions, pending, evidence, decisions, trade log)
strategy.py     NewsStrategy    backtesting.py Strategy + run_news_backtest() for many tickers
replay.py       run_replay()    live loop over history with SimClock
config.py       build_bot()     wire everything from a YAML/JSON dict
```

### Evidence, bundles and scoring

Every item carries a `kind`:

| kind | examples | role |
|------|----------|------|
| `news`, `filing` | headlines, press releases, Nasdaq wire | **trigger** (fresh within `max_news_age_minutes`) |
| `earnings_result` | Finnhub / FMP calendar: EPS and revenue vs consensus, surprise % | **trigger** (fresh within `max_event_age_hours`) |
| `regulatory` | FDA press release, openFDA approval, clinical-trial status change | **trigger** |
| `sentiment` | Alpha Vantage / Marketaux per-ticker sentiment, Polygon insights | context |
| `earnings_upcoming` | scheduled report date and time | context |
| `social` | one aggregate per poll: StockTwits bull/bear counts, Reddit mentions | context (weak) |

On each tick every new item goes into the `EvidenceStore` (default 24 h window per ticker). Tickers that
received a *fresh trigger* are scored **once per tick** over their whole bundle. The bundle exposes
features (recency-weighted rule scores, provider sentiment mean, latest earnings surprise, regulatory
prior, social tilt, source count) and a chronological text rendering.

* `RuleScorer` combines those features deterministically (weights in `scoring.py`).
* `ClaudeScorer` sends the rendering plus the features to the model with a strict JSON schema and
  returns `score / category / confidence / reasons`. On any API failure it falls back to `RuleScorer`
  and tags the reasons with `fallback:rules`. Duplicated headlines across providers are one event.

Every scoring pass is logged in the state file (`decisions`) whether or not it produced a signal, so
you can audit what the model saw and decided: `python -m newsbot status`.

### Actions

`actions:` in the config lists what a signal does. `trade` queues it for execution by the bot;
`telegram` and `webhook` notify (signal, entry and exit events are each optional). Without a
`trade` action the bot runs **notify-only**: it scores and alerts but never places an order.

### Signal rules (`SignalEngine`)

* `RuleClassifier` sums the weights of every matching pattern (`classifiers/rules.py`) and clamps to
  [−1, 1]; bearish patterns are negative so "beats estimates **but cuts guidance**" nets negative.
  Add rows to `DEFAULT_RULES` (regex, category, weight) to extend it.
* A signal is created when the bundle `score >= min_score` and the ticker is in `universe` (if set)
  and the category passes the allow/deny lists.
* `target_pct` scales with the score when `scale_target_by_score` is on: ×1.0 at 0.5, ×1.25 at 1.0.
* Signals live for `signal_ttl_hours` (18 h) so an after-close earnings release is bought at the
  next open; `max_chase_pct` skips the entry if price already gapped more than 5 % above the price
  seen when the signal was created.

### Live loop (`NewsTradingBot.tick`)

1. fetch evidence newer than `now − window` from every source (each source paces itself to its free
   tier), drop already-seen ids, store the rest;
2. score every ticker that got a fresh trigger, push qualifying signals to the actions, queue them
   if a `trade` action is configured (one per ticker; none for a ticker already held/queued);
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

## AI scoring (optional but recommended)

`pip install anthropic`, set `ANTHROPIC_API_KEY`, and use `scoring: {type: claude}`. One request per
scoring pass (per ticker with fresh evidence, per tick), so cost scales with news flow rather than
poll frequency. Test it offline on a news file without touching the live loop:

```bash
python -m newsbot score --config config.yaml --news news.json --ticker AAPL --as-of 2023-05-05T00:00:00Z
```

## Telegram

Create a bot with @BotFather, get the token, send the bot a message, then read your chat id from
`https://api.telegram.org/bot<TOKEN>/getUpdates`. Export `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
and add `- type: telegram` under `actions`. Remove `- type: trade` for alerts without orders.

## Tests

```bash
python -m newsbot.test
```

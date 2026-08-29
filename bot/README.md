inclusion-bot — index-inclusion signals over Telegram
=====================================================

A small bot that watches the **published rulebooks of major US and European
equity indices** for stocks that are about to become eligible for inclusion —
the "index effect" anticipation trade developed and backtested in
[`doc/examples/Index Inclusion Strategy.py`](../doc/examples/Index%20Inclusion%20Strategy.py)
— and pushes **buy/sell signals to a Telegram chat**. Every buy signal carries
an entry price, an expected exit price and date, and the theoretical P&L on a
configured notional; every position gets a matching sell signal (flows done,
candidacy failed, or the ~1-month maximum hold reached).

> ⚠️ **Educational software, not investment advice.** The signals are
> rule-based screens over public data; the "expected" exit prices rest on
> *assumed* effect sizes you configure yourself; and the modern index effect
> in heavily arbitraged large caps is small (see the notebook's discussion of
> Greenwood & Sammon). Verify every signal against the index provider's
> official announcements before acting on it.

What fires a signal
-------------------

| Index | Region | Rule watched | BUY when | Exit |
|:--|:--|:--|:--|:--|
| FTSE 100 | EUR (UK) | quarterly review: non-member ranked ≤ 90 by full cap at the cutoff (Tue before 1st Friday) is added automatically | a non-member ranks inside the band ahead of / at the cutoff | first session after the 3rd-Friday effective date |
| DAX 40 | EUR (DE) | reviews implemented after the 3rd Friday of Mar/Jun/Sep/Dec; fast entry at free-float rank ≤ 33 (every review), regular entry ≤ 40 (Mar & Sep) | a non-member ranks inside the applicable band | first session after the effective date |
| Nasdaq-100 | US | annual December reconstitution (top 75 added; ranks from end-November) + quarterly rank reviews since 2026 | a Nasdaq-listed non-financial non-member ranks ≤ 75 | first session after the effective date |
| S&P 500 | US | committee + eligibility gates: cap ≥ $22.7B (Jul 2025 level), positive GAAP earnings last quarter & trailing 4 | a non-member crosses the gates (a *watchlist* signal — committee decides the timing) | on falling back below the gates, or the max-hold backstop |

Portfolio rules mirror the backtest: **one theoretical position per index at a
time**, entries screened daily, and a **21-session (~1 month) maximum hold**.

The signal lifecycle for a rank-based index:

    pre-cutoff (default 10 sessions)   cutoff…effective          after effective
    "predicted addition"          →    "projected addition"  →   SELL
    rank ≤ band − buffer               ranks locked; INFO        (or SELL earlier if the
                                       message sent              rank decays out of the band)

Setup
-----

1. **Create the Telegram bot**: talk to [@BotFather](https://t.me/botfather),
   `/newbot`, and keep the token. Message your new bot once, then get your
   chat id, e.g. from `https://api.telegram.org/bot<TOKEN>/getUpdates`.
2. **Install and test** (Python ≥ 3.10):

       cd bot
       pip install -r requirements.txt
       python -m inclusion_bot selftest                 # offline demo cycle
       export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
       python -m inclusion_bot test-telegram            # "✅ connected"
       python -m inclusion_bot scan --dry-run           # full scan, printed only

3. **Schedule one scan per weekday after the US close** (22:15 UTC covers all
   four indices) — via `deploy/crontab.example`, a systemd timer, or the
   serverless `deploy/github-action.yml.example`.

`scan` writes `bot_state.json` (open positions, dedupe keys, closed history)
and caches the day's Wikipedia/Yahoo fetches under `.cache/`. Signals are
never re-sent for the same review, and a sell always references its entry.

Configuration
-------------

Defaults live in `inclusion_bot/config.py`; override any of them with
`--config config.json` (see `config.example.json`). The knobs that matter:

* `notional` — position size used for the theoretical P&L in messages.
* `indices.<KEY>.expected_effect` — the assumed addition run-up used for the
  expected exit price. Defaults are deliberately modest (FTSE 2%, DAX 1.5%,
  NDX 1%, SPX 0.5%); they are **your assumption, not a promise**.
* `indices.<KEY>.buffer` / `pre_cutoff_days` — how deep inside the entry band
  a stock must rank, and how early the hunt starts (the selectivity/
  opportunity dial explored in the notebook).
* `indices.<KEY>.enabled` — switch an index off entirely.
* Credentials come only from `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`.

Data sources & honest limitations
---------------------------------

* **Membership and candidate pools** come from Wikipedia's constituent lists
  (S&P 500/400, Nasdaq-100, FTSE 100/250, DAX/MDAX), parsed tolerantly;
  **prices, market caps, venues and float** come from Yahoo Finance via
  `yfinance` — both unofficial sources that can lag or hiccup. The bot
  refuses to rank on a day when less than 80% of a universe has data, and
  reports per-index failures to the chat rather than staying silent.
* **Ranks are computed within the covered universe** (members + the candidate
  pool), not the exchange's full list — accurate near the entry bands, which
  is where signals live. DAX uses Yahoo float shares for free-float caps with
  a full-cap fallback.
* **Calendars ignore exchange holidays** (pure weekday math) and the DAX /
  Nasdaq-100-quarterly announcement dates are approximations; cutoffs and
  effective dates are the load-bearing dates and follow the rulebooks. The
  Nasdaq-100's post-2026 quarterly review mechanics are approximated with the
  annual top-75 band (flagged `medium` confidence in messages).
* **No official announcement feed is parsed** (yet): between cutoff and
  effective the bot projects additions from its own locked ranks and tells
  you to verify against the provider's announcement. Deletions are not
  shorted; that's a natural extension, borrow costs permitting.
* Thresholds drift (S&P raises its cap floor several times a year) — revisit
  `config.py` against the current methodology documents periodically.

Development
-----------

Pure logic (calendars, screener, state machine, formatting) is fully covered
by offline tests with fixture data — including checks against real historical
review dates:

    python -m unittest discover -s bot/tests -v

The data layer funnels through two cached functions so everything else stays
testable without network access.

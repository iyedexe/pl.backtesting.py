# From backtest to live PMCC — roadmap and safety rails

This directory contains the **execution scaffold** for running the PMCC
strategy for real. It deliberately reuses the *same* decision core
(`pmcc.strategy.PMCC.decide()`) as the backtest, so the logic you tested is
the logic that trades.

**Status: paper-ready scaffold, NOT a finished trading system.** The
`PaperBroker` path works offline today; the Interactive Brokers adapter is an
untested skeleton written offline.

## Components

| file          | what it does |
|---------------|--------------|
| `state.py`    | JSON position/intent state, reconciled against broker truth each run |
| `brokers.py`  | `Broker` protocol + offline `PaperBroker` (simulated fills) |
| `executor.py` | one decision cycle: broker → `MarketView` → `decide()` → marketable-limit orders, with safety rails |
| `ib_adapter.py` | Interactive Brokers skeleton for Euronext options (**untested**) |
| `run_live.py` | CLI entry point, one invocation per trading day |

## Safety rails already built in

- **Dry-run by default** — you must pass `--send` for any order to go out.
- **Kill switch** — create a file named `STOP` next to the state file and the
  executor refuses to act.
- **Price bands** — orders whose limit deviates >25% from the model mid are
  rejected (fat-finger/stale-quote guard).
- **Order cap** per cycle, and live-port refusal in the IB adapter unless
  `allow_live=True` is passed explicitly.

## The path to real money (do not skip steps)

1. **Offline dry-run** (works today):
   `python -m pmcc.live.run_live --ticker MC.PA --broker paper`
2. **Offline paper loop with fills**: add `--send`. Run daily for a while;
   inspect `pmcc_state_*.json` history.
3. **IB paper account**: install `ib_insync`, run TWS paper (port 7497),
   fix contract details per underlying (`reqContractDetails`) — trading
   class, multiplier (some Paris options are 10-lot!), exchange routing.
   Feed `--closes` with a real daily-closes CSV you maintain.
4. **Weeks of paper trading**: compare each fill against the backtest
   assumptions (spread paid vs the 2%+€0.03 model, premium levels vs model).
   If real spreads are wider, re-run the backtest sensitivity with the
   measured numbers — the strategy's edge is spread-sensitive.
5. **Go-live checklist** (only after 3–4 clean paper months):
   - long-dated (12–24 m) series actually quoted on your underlying, with
     tolerable spreads (many Paris names have LEAPS listed but barely traded);
   - assignment handling rehearsed (early assignment around the May dividend
     season is *normal* for ITM short calls — roll before ex-div when the
     short's extrinsic < the dividend);
   - position sizing: one underlying at a time, and remember the backtest's
     leveraged variant (`sizing='budget'`) hit −58% portfolio drawdowns —
     use `match_stock` sizing;
   - tax treatment of option income in your jurisdiction.

## What still needs building for full automation

- [ ] real IV surface from live option chains (the executor currently prices
      from realised vol; `pmcc.pricing.implied_vol` is ready for quote-based
      calibration)
- [ ] dividend calendar feed (ex-dates) for early-assignment defence
- [ ] fill monitoring / partial-fill handling beyond one-shot limit orders
- [ ] multi-underlying portfolio runner + aggregate risk caps
- [ ] alerting (order rejected, reconciliation mismatch, margin call)

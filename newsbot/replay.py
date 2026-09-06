"""
Drive the *live* bot code path over historical CSV prices and a news file with a
simulated clock. Complements `newsbot.strategy` (which uses the backtest engine
and intraday high/low for stops) by exercising the exact ingestion -> signal ->
broker -> exit logic that runs in production, one tick per bar.

Daily bars: a tick fires at 09:30 New York each bar day and fills at that bar's
Open, so an after-close headline on day D is bought at D+1's open.
Intraday bars: a tick fires at each bar's timestamp and fills at its Close.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import pandas as pd

from .bot import NewsTradingBot
from .brokers import PaperBroker
from .clock import NY, SimClock
from .config import build_bot, merge_config
from .prices import CSVPriceFeed, load_ohlc_csv
from .sources import FileNewsSource
from .state import BotState

log = logging.getLogger(__name__)


def _is_daily(index: pd.DatetimeIndex) -> bool:
    return bool(((index.hour == 0) & (index.minute == 0)).all())


def run_replay(prices: Mapping[str, Union[str, Path, pd.DataFrame]], news: Union[str, Path, FileNewsSource],
               config: Optional[Dict[str, Any]] = None, *, cash: Optional[float] = None,
               state_path: Optional[Union[str, Path]] = None) -> NewsTradingBot:
    frames = {t.upper(): (p if isinstance(p, pd.DataFrame) else load_ohlc_csv(p)) for t, p in prices.items()}
    source = news if isinstance(news, FileNewsSource) else FileNewsSource(news)
    cfg = merge_config(config)
    cfg['bot']['market_hours_only'] = False
    cfg['bot']['max_news_age_minutes'] = 3 * 24 * 60   # one tick per bar: the overnight gap must not expire news
    if not cfg.get('universe'):
        cfg['universe'] = list(frames)

    all_ts = sorted(set().union(*[set(pd.DatetimeIndex(df.index)) for df in frames.values()]))
    daily = _is_daily(pd.DatetimeIndex(all_ts))
    clock = SimClock(all_ts[0])
    feed = CSVPriceFeed(frames, clock, field='Open' if daily else 'Close')
    broker = PaperBroker(feed, cash=float(cash if cash is not None else cfg['broker'].get('cash', 100000)),
                         clock=clock, commission=float(cfg['broker'].get('commission', 0.0)),
                         slippage_bps=float(cfg['broker'].get('slippage_bps', 0.0)), always_open=True)
    bot = build_bot(cfg, clock=clock, sources=[source], price_feed=feed, broker=broker, state=BotState(state_path))

    for ts in all_ts:
        ts = pd.Timestamp(ts)
        if daily:
            local = ts.tz_localize(NY) if ts.tz is None else ts.tz_convert(NY)
            clock.set(local.replace(hour=9, minute=30) + timedelta(0))
        else:
            clock.set(ts)
        bot.tick()
    log.info('replay finished: %s', bot.summary())
    return bot

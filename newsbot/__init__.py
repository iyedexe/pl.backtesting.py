"""
newsbot — a news-driven, long-only equity trading bot with a max one-week
holding period, built on top of backtesting.py.

    from newsbot import build_bot, load_config
    bot = build_bot(load_config('config.yaml'))
    bot.run()

See `newsbot/README.md`.
"""
from .actions import (Action, CallbackAction, DiscordAction, LogAction, SignalReport, SlackAction, TelegramAction,
                      TradeAction, WebhookAction)
from .aggregator import Bundle, EvidenceStore
from .bot import NewsTradingBot
from .brokers import Broker, BrokerError, PaperBroker
from .classifiers import Classifier, RuleClassifier
from .clock import SimClock, SystemClock
from .config import build_bot, load_config, merge_config
from .models import Classification, ClosedTrade, ExitReason, NewsItem, Position, Signal
from .prices import CSVPriceFeed, PriceFeed, StaticPriceFeed
from .scoring import ClaudeScorer, RuleScorer, Scorer
from .signals import SignalEngine
from .sources import FileNewsSource, NewsSource, RSSNewsSource, YahooFinanceRSS
from .state import BotState
from .universe import MARKETS, Market, Universe

_BACKTEST_EXPORTS = ('NewsStrategy', 'align_news_scores', 'run_news_backtest')


def __getattr__(name):
    """`backtesting` is only needed for backtests; the live bot runs without it."""
    if name in _BACKTEST_EXPORTS:
        from . import strategy  # noqa: PLC0415
        return getattr(strategy, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'Action', 'CallbackAction', 'DiscordAction', 'LogAction', 'SignalReport', 'SlackAction', 'TelegramAction',
    'TradeAction', 'WebhookAction', 'MARKETS', 'Market', 'Universe',
    'Bundle', 'EvidenceStore', 'ClaudeScorer', 'RuleScorer', 'Scorer',
    'NewsTradingBot', 'Broker', 'BrokerError', 'PaperBroker', 'Classifier', 'RuleClassifier',
    'SimClock', 'SystemClock', 'build_bot', 'load_config', 'merge_config',
    'Classification', 'ClosedTrade', 'ExitReason', 'NewsItem', 'Position', 'Signal',
    'CSVPriceFeed', 'PriceFeed', 'StaticPriceFeed', 'SignalEngine',
    'FileNewsSource', 'NewsSource', 'RSSNewsSource', 'YahooFinanceRSS', 'BotState',
    'NewsStrategy', 'align_news_scores', 'run_news_backtest',
]

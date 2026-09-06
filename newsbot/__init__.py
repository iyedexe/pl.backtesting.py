"""
newsbot — a news-driven, long-only equity trading bot with a max one-week
holding period, built on top of backtesting.py.

    from newsbot import build_bot, load_config
    bot = build_bot(load_config('config.yaml'))
    bot.run()

See `newsbot/README.md`.
"""
from .bot import NewsTradingBot
from .brokers import Broker, BrokerError, PaperBroker
from .classifiers import Classifier, RuleClassifier
from .clock import SimClock, SystemClock
from .config import build_bot, load_config, merge_config
from .models import Classification, ClosedTrade, ExitReason, NewsItem, Position, Signal
from .prices import CSVPriceFeed, PriceFeed, StaticPriceFeed
from .signals import SignalEngine
from .sources import FileNewsSource, NewsSource, RSSNewsSource, YahooFinanceRSS
from .state import BotState
from .strategy import NewsStrategy, align_news_scores, run_news_backtest

__all__ = [
    'NewsTradingBot', 'Broker', 'BrokerError', 'PaperBroker', 'Classifier', 'RuleClassifier',
    'SimClock', 'SystemClock', 'build_bot', 'load_config', 'merge_config',
    'Classification', 'ClosedTrade', 'ExitReason', 'NewsItem', 'Position', 'Signal',
    'CSVPriceFeed', 'PriceFeed', 'StaticPriceFeed', 'SignalEngine',
    'FileNewsSource', 'NewsSource', 'RSSNewsSource', 'YahooFinanceRSS', 'BotState',
    'NewsStrategy', 'align_news_scores', 'run_news_backtest',
]

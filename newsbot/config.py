"""
Build a fully wired `NewsTradingBot` from a YAML/JSON config dict.
See `config.example.yaml` at the repo root for every option.
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .bot import NewsTradingBot
from .brokers import Broker, PaperBroker
from .classifiers import Classifier, RuleClassifier
from .clock import Clock, SystemClock
from .prices import CSVPriceFeed, PriceFeed, StaticPriceFeed, YFinancePriceFeed
from .signals import SignalEngine
from .sources import FileNewsSource, NewsSource, RSSNewsSource, TickerExtractor, YahooFinanceRSS
from .state import BotState

log = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    'universe': [],
    'sources': [{'type': 'yahoo'}],
    'classifier': {'type': 'rules'},
    'signals': {'min_score': 0.5, 'target_pct': 0.05, 'stop_pct': 0.03, 'max_hold_days': 7,
                'signal_ttl_hours': 18, 'scale_target_by_score': True, 'categories': [], 'blocked_categories': []},
    'prices': {'type': 'yfinance'},
    'broker': {'type': 'paper', 'cash': 100000, 'commission': 0.0, 'slippage_bps': 5, 'paper': True},
    'bot': {'max_positions': 5, 'position_size_pct': 0.2, 'risk_per_trade_pct': None, 'max_position_value': None,
            'fractional_shares': False, 'max_news_age_minutes': 30, 'max_chase_pct': 0.05,
            'market_hours_only': True, 'poll_interval_seconds': 60, 'state_file': 'newsbot_state.json'},
}


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml  # noqa: PLC0415
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return merge_config(data)


def merge_config(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


class _AlpacaHolder:
    """Share one authenticated Alpaca client between news, prices and broker."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._client = None

    def client(self):
        if self._client is None:
            from .alpaca import AlpacaClient  # noqa: PLC0415
            a = self.cfg.get('alpaca', {})
            self._client = AlpacaClient(a.get('key'), a.get('secret'),
                                        paper=bool(self.cfg['broker'].get('paper', True)))
        return self._client


def build_classifier(cfg: Dict[str, Any]) -> Classifier:
    c = cfg.get('classifier', {})
    kind = c.get('type', 'rules')
    if kind == 'rules':
        return RuleClassifier()
    if kind == 'claude':
        from .classifiers.claude import ClaudeClassifier  # noqa: PLC0415
        kwargs = {}
        if c.get('model'):
            kwargs['model'] = c['model']
        if c.get('effort'):
            kwargs['effort'] = c['effort']
        return ClaudeClassifier(**kwargs)
    raise ValueError(f'unknown classifier type {kind!r}')


def build_sources(cfg: Dict[str, Any], alpaca: _AlpacaHolder) -> List[NewsSource]:
    universe = [t.upper() for t in cfg.get('universe') or []]
    extractor = TickerExtractor(universe)
    out: List[NewsSource] = []
    for s in cfg.get('sources', []):
        kind = s.get('type')
        if kind == 'file':
            out.append(FileNewsSource(s['path']))
        elif kind == 'rss':
            out.append(RSSNewsSource(s['url'], name=s.get('name'), tickers=s.get('tickers'), extractor=extractor))
        elif kind == 'yahoo':
            tickers = s.get('tickers') or universe
            if not tickers:
                raise ValueError('yahoo source needs `tickers` or a non-empty `universe`')
            out.append(YahooFinanceRSS(tickers))
        elif kind == 'alpaca':
            from .alpaca import AlpacaNewsSource  # noqa: PLC0415
            out.append(AlpacaNewsSource(alpaca.client(), symbols=s.get('symbols') or universe or None,
                                        limit=int(s.get('limit', 50))))
        else:
            raise ValueError(f'unknown news source type {kind!r}')
    return out


def build_price_feed(cfg: Dict[str, Any], alpaca: _AlpacaHolder, clock: Clock) -> PriceFeed:
    p = cfg.get('prices', {})
    kind = p.get('type', 'yfinance')
    if kind == 'static':
        return StaticPriceFeed(p.get('prices', {}))
    if kind == 'csv':
        return CSVPriceFeed.from_files(p['files'], clock)
    if kind == 'yfinance':
        return YFinancePriceFeed(clock=clock)
    if kind == 'alpaca':
        from .alpaca import AlpacaPriceFeed  # noqa: PLC0415
        return AlpacaPriceFeed(alpaca.client(), feed=p.get('feed', 'iex'))
    raise ValueError(f'unknown price feed type {kind!r}')


def build_broker(cfg: Dict[str, Any], alpaca: _AlpacaHolder, feed: PriceFeed, clock: Clock) -> Broker:
    b = cfg.get('broker', {})
    kind = b.get('type', 'paper')
    if kind == 'paper':
        return PaperBroker(feed, cash=float(b.get('cash', 100000)), clock=clock,
                           commission=float(b.get('commission', 0.0)), slippage_bps=float(b.get('slippage_bps', 0.0)),
                           always_open=bool(b.get('always_open', False)))
    if kind == 'alpaca':
        from .alpaca import AlpacaBroker  # noqa: PLC0415
        if not b.get('paper', True):
            log.warning('LIVE Alpaca trading enabled (broker.paper = false)')
        return AlpacaBroker(alpaca.client())
    raise ValueError(f'unknown broker type {kind!r}')


def build_engine(cfg: Dict[str, Any], classifier: Optional[Classifier] = None) -> SignalEngine:
    s = cfg.get('signals', {})
    return SignalEngine(classifier or build_classifier(cfg),
                        min_score=float(s.get('min_score', 0.5)),
                        target_pct=float(s.get('target_pct', 0.05)),
                        stop_pct=float(s.get('stop_pct', 0.03)),
                        max_hold_days=float(s.get('max_hold_days', 7)),
                        signal_ttl=timedelta(hours=float(s.get('signal_ttl_hours', 18))),
                        scale_target_by_score=bool(s.get('scale_target_by_score', True)),
                        universe=cfg.get('universe') or None,
                        categories=s.get('categories') or None,
                        blocked_categories=s.get('blocked_categories') or None)


def build_bot(config: Optional[Dict[str, Any]] = None, *, clock: Optional[Clock] = None,
              sources: Optional[List[NewsSource]] = None, price_feed: Optional[PriceFeed] = None,
              broker: Optional[Broker] = None, state: Optional[BotState] = None,
              classifier: Optional[Classifier] = None) -> NewsTradingBot:
    """Wire a bot from config. Explicit keyword components override the config (used by replay & tests)."""
    cfg = merge_config(config)
    clock = clock or SystemClock()
    alpaca = _AlpacaHolder(cfg)
    feed = price_feed or build_price_feed(cfg, alpaca, clock)
    broker = broker or build_broker(cfg, alpaca, feed, clock)
    sources = sources if sources is not None else build_sources(cfg, alpaca)
    b = cfg['bot']
    if state is None:
        state = BotState(b.get('state_file'))
    return NewsTradingBot(
        sources=sources, engine=build_engine(cfg, classifier), broker=broker, price_feed=feed, state=state, clock=clock,
        max_positions=int(b.get('max_positions', 5)),
        position_size_pct=float(b.get('position_size_pct', 0.2)),
        risk_per_trade_pct=(float(b['risk_per_trade_pct']) if b.get('risk_per_trade_pct') else None),
        max_position_value=(float(b['max_position_value']) if b.get('max_position_value') else None),
        fractional_shares=bool(b.get('fractional_shares', False)),
        max_news_age=timedelta(minutes=float(b.get('max_news_age_minutes', 30))),
        max_chase_pct=(float(b['max_chase_pct']) if b.get('max_chase_pct') is not None else None),
        market_hours_only=bool(b.get('market_hours_only', True)),
        poll_interval=float(b.get('poll_interval_seconds', 60)),
    )

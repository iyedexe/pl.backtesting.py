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

from .actions import Action, LogAction, TelegramAction, TradeAction, WebhookAction
from .aggregator import EvidenceStore
from .bot import NewsTradingBot
from .brokers import Broker, PaperBroker
from .classifiers import Classifier, RuleClassifier
from .clock import Clock, SystemClock
from .prices import CSVPriceFeed, PriceFeed, StaticPriceFeed, YFinancePriceFeed
from .scoring import RuleScorer, Scorer
from .signals import SignalEngine
from .sources import (FileNewsSource, GoogleNewsRSS, NasdaqRSS, NewsSource, RSSNewsSource, TickerExtractor,
                      YahooFinanceRSS)
from .state import BotState

log = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    'universe': [],
    'aliases': {},                       # ticker -> company names, for sources that never print tickers
    'api_keys': {},                      # provider -> key (else FINNHUB_API_KEY etc. from the environment)
    'sources': [{'type': 'yahoo'}],
    'classifier': {'type': 'rules'},     # per-headline rule scoring (features + backtests)
    'scoring': {'type': 'rules', 'model': 'claude-opus-5', 'effort': 'medium', 'window_hours': 24,
                'max_event_age_hours': 12},
    'actions': [{'type': 'trade'}],
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


PROVIDER_KEY_NAMES = {'finnhub': 'finnhub', 'finnhub_earnings': 'finnhub', 'alphavantage': 'alphavantage',
                      'polygon': 'polygon', 'marketaux': 'marketaux', 'newsapi': 'newsapi', 'fmp': 'fmp',
                      'fmp_earnings': 'fmp'}


def build_extractor(cfg: Dict[str, Any]) -> TickerExtractor:
    universe = [t.upper() for t in cfg.get('universe') or []]
    return TickerExtractor(universe, aliases=cfg.get('aliases') or {})


def build_source(s: Dict[str, Any], cfg: Dict[str, Any], alpaca: '_AlpacaHolder',
                 extractor: Optional[TickerExtractor] = None) -> NewsSource:
    from . import providers as P  # noqa: PLC0415, N812
    kind = s.get('type')
    universe = [t.upper() for t in cfg.get('universe') or []]
    tickers = [t.upper() for t in (s.get('tickers') or s.get('symbols') or universe)]
    extractor = extractor or build_extractor(cfg)
    common: Dict[str, Any] = {}
    if s.get('min_interval_seconds') is not None:
        common['min_interval'] = float(s['min_interval_seconds'])
    api_kw: Dict[str, Any] = dict(common)
    key = s.get('api_key') or (cfg.get('api_keys') or {}).get(PROVIDER_KEY_NAMES.get(kind or '', ''))
    if key:
        api_kw['api_key'] = key
    if s.get('base_url'):
        api_kw['base_url'] = s['base_url']

    def need_tickers():
        if not tickers:
            raise ValueError(f'{kind} source needs `tickers` or a non-empty `universe`')
        return tickers

    if kind == 'file':
        return FileNewsSource(s['path'])
    if kind == 'rss':
        return RSSNewsSource(s['url'], name=s.get('name'), tickers=s.get('tickers'), extractor=extractor,
                             kind=s.get('kind', 'news'), **common)
    if kind == 'yahoo':
        return YahooFinanceRSS(need_tickers(), **common)
    if kind == 'google':
        return GoogleNewsRSS(need_tickers(), **common)
    if kind == 'nasdaq_rss':
        return NasdaqRSS(need_tickers(), **common)
    if kind == 'alpaca':
        from .alpaca import AlpacaNewsSource  # noqa: PLC0415
        return AlpacaNewsSource(alpaca.client(), symbols=tickers or None, limit=int(s.get('limit', 50)))
    if kind == 'finnhub':
        return P.FinnhubNews(need_tickers(), **api_kw)
    if kind == 'finnhub_earnings':
        return P.FinnhubEarningsCalendar(tickers, **api_kw)
    if kind == 'alphavantage':
        return P.AlphaVantageNews(tickers, **api_kw)
    if kind == 'polygon':
        return P.PolygonNews(tickers, **api_kw)
    if kind == 'marketaux':
        return P.MarketauxNews(tickers, **api_kw)
    if kind == 'newsapi':
        return P.NewsAPINews(tickers, extractor=extractor, queries=s.get('queries'), **api_kw)
    if kind == 'fmp':
        return P.FMPNews(tickers, **api_kw)
    if kind == 'fmp_earnings':
        return P.FMPEarningsCalendar(tickers, **api_kw)
    if kind == 'nasdaq_earnings':
        return P.NasdaqEarningsCalendar(tickers, **api_kw)
    if kind == 'fda':
        return P.FDAPressRSS(extractor, url=s.get('url'), **common)
    if kind == 'openfda':
        return P.OpenFDAApprovals(extractor, **api_kw)
    if kind == 'clinicaltrials':
        sponsors = s.get('sponsors') or {t: names for t, names in (cfg.get('aliases') or {}).items()
                                         if not tickers or t.upper() in tickers}
        if not sponsors:
            raise ValueError('clinicaltrials source needs `sponsors` ({ticker: [names]}) or `aliases`')
        return P.ClinicalTrialsSource(sponsors, **api_kw)
    if kind == 'stocktwits':
        return P.StockTwitsStream(need_tickers(), **api_kw)
    if kind == 'reddit':
        return P.RedditMentions(need_tickers(), subreddits=s.get('subreddits'), **api_kw)
    raise ValueError(f'unknown news source type {kind!r}')


def build_sources(cfg: Dict[str, Any], alpaca: '_AlpacaHolder') -> List[NewsSource]:
    extractor = build_extractor(cfg)
    return [build_source(s, cfg, alpaca, extractor) for s in cfg.get('sources', [])]


def build_scorer(cfg: Dict[str, Any]) -> Scorer:
    sc = cfg.get('scoring', {})
    kind = sc.get('type', 'rules')
    if kind == 'rules':
        return RuleScorer()
    if kind == 'claude':
        from .scoring import ClaudeScorer  # noqa: PLC0415
        kwargs: Dict[str, Any] = {}
        if sc.get('model'):
            kwargs['model'] = sc['model']
        if sc.get('effort'):
            kwargs['effort'] = sc['effort']
        return ClaudeScorer(**kwargs)
    raise ValueError(f'unknown scoring type {kind!r}')


def build_actions(cfg: Dict[str, Any]) -> List[Action]:
    out: List[Action] = []
    for a in cfg.get('actions') or [{'type': 'trade'}]:
        kind = a.get('type')
        events = a.get('events')
        if kind == 'trade':
            out.append(TradeAction())
        elif kind == 'telegram':
            out.append(TelegramAction(a.get('token'), a.get('chat_id'), events=events,
                                      include_evidence=bool(a.get('include_evidence', True))))
        elif kind == 'webhook':
            out.append(WebhookAction(a['url'], events=events, headers=a.get('headers')))
        elif kind == 'log':
            out.append(LogAction(events))
        else:
            raise ValueError(f'unknown action type {kind!r}')
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
              classifier: Optional[Classifier] = None, scorer: Optional[Scorer] = None,
              actions: Optional[List[Action]] = None) -> NewsTradingBot:
    """Wire a bot from config. Explicit keyword components override the config (used by replay & tests)."""
    cfg = merge_config(config)
    clock = clock or SystemClock()
    alpaca = _AlpacaHolder(cfg)
    feed = price_feed or build_price_feed(cfg, alpaca, clock)
    broker = broker or build_broker(cfg, alpaca, feed, clock)
    sources = sources if sources is not None else build_sources(cfg, alpaca)
    b = cfg['bot']
    sc = cfg.get('scoring', {})
    if state is None:
        state = BotState(b.get('state_file'))
    engine = build_engine(cfg, classifier)
    store = EvidenceStore(window=timedelta(hours=float(sc.get('window_hours', 24))), classifier=engine.classifier,
                          universe=engine.universe or None)
    return NewsTradingBot(
        sources=sources, engine=engine, broker=broker, price_feed=feed, state=state, clock=clock,
        store=store, scorer=scorer or build_scorer(cfg), actions=actions if actions is not None else build_actions(cfg),
        max_event_age=timedelta(hours=float(sc.get('max_event_age_hours', 12))),
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

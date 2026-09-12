"""Alpha Vantage NEWS_SENTIMENT: ticker-tagged articles with provider sentiment. Free tier ~25 requests/day,
so the default interval is one call per hour covering the whole universe."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import KIND_SENTIMENT, NewsItem, to_utc
from ._base import APISource, _f


class AlphaVantageNews(APISource):
    name = 'alphavantage'
    ENV_KEYS = ('ALPHAVANTAGE_API_KEY',)
    BASE = 'https://www.alphavantage.co/query'
    DEFAULT_INTERVAL = 3600.0
    MIN_RELEVANCE = 0.3
    #: used when no tickers are configured (market mode)
    DEFAULT_TOPICS = 'earnings,mergers_and_acquisitions,financial_markets'

    def __init__(self, tickers=None, *, topics: Optional[str] = None, **kw):
        super().__init__(tickers, **kw)
        self.topics = topics

    def _fetch(self, since: datetime) -> List[NewsItem]:
        topics = self.topics or (None if self.tickers else self.DEFAULT_TOPICS)
        data = self._get(self.BASE, function='NEWS_SENTIMENT', tickers=','.join(self.tickers) or None, topics=topics,
                         time_from=since.strftime('%Y%m%dT%H%M'), sort='LATEST', limit=200, apikey=self.api_key) or {}
        return self.parse(data, set(self.tickers), self.MIN_RELEVANCE)

    @staticmethod
    def parse(data: Dict[str, Any], universe: set, min_relevance: float = 0.3) -> List[NewsItem]:
        items: List[NewsItem] = []
        for a in data.get('feed', []):
            try:
                published = to_utc(datetime.strptime(a['time_published'], '%Y%m%dT%H%M%S'))
            except (KeyError, ValueError):
                continue
            per_ticker = {}
            for ts in a.get('ticker_sentiment', []):
                sym = (ts.get('ticker') or '').upper()
                rel = _f(ts.get('relevance_score')) or 0.0
                if sym and (not universe or sym in universe) and rel >= min_relevance:
                    per_ticker[sym] = {'sentiment': _f(ts.get('ticker_sentiment_score')),
                                       'label': ts.get('ticker_sentiment_label'), 'relevance': rel}
            if not per_ticker:
                continue
            items.append(NewsItem(id=f'alphavantage:{a.get("url")}', headline=a.get('title', ''), published=published,
                                  tickers=list(per_ticker), summary=(a.get('summary') or '')[:2000],
                                  source=f'alphavantage/{a.get("source", "")}', url=a.get('url', ''),
                                  kind=KIND_SENTIMENT,
                                  meta={'overall_sentiment': _f(a.get('overall_sentiment_score')),
                                        'overall_label': a.get('overall_sentiment_label'),
                                        'ticker_sentiment': per_ticker}))
        return items

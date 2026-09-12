"""NewsAPI.org `everything` search. Free tier is delayed ~24h and non-commercial, ~100 requests/day.
Tickers are attached by text extraction (aliases recommended)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..models import KIND_NEWS, NewsItem, to_utc
from ..sources import TickerExtractor
from ._base import APISource


class NewsAPINews(APISource):
    name = 'newsapi'
    ENV_KEYS = ('NEWSAPI_API_KEY',)
    BASE = 'https://newsapi.org/v2/everything'
    DEFAULT_INTERVAL = 900.0

    def __init__(self, tickers: Optional[Iterable[str]] = None, *, extractor: Optional[TickerExtractor] = None,
                 queries: Optional[Mapping[str, str]] = None, **kw):
        super().__init__(tickers, **kw)
        self.extractor = extractor or TickerExtractor(self.tickers)
        self.queries = dict(queries or {})       # ticker -> search phrase, e.g. {'AAPL': '"Apple Inc"'}

    MARKET_QUERY = ('"beats estimates" OR "raises guidance" OR "FDA approval" OR "to be acquired" OR '
                    '"record revenue" OR "quarterly results"')

    def _query(self) -> str:
        terms = [self.queries.get(t, f'"{t}"') for t in self.tickers]
        return (' OR '.join(terms) if terms else self.MARKET_QUERY)[:490]   # NewsAPI caps q at 500 chars

    def _fetch(self, since: datetime) -> List[NewsItem]:
        data = self._get(self.BASE, q=self._query(), **{'from': since.strftime('%Y-%m-%dT%H:%M:%S')},
                         sortBy='publishedAt', language='en', pageSize=100, apiKey=self.api_key) or {}
        return self.parse(data, self.extractor)

    @staticmethod
    def parse(data: Dict[str, Any], extractor: TickerExtractor) -> List[NewsItem]:
        items: List[NewsItem] = []
        for a in data.get('articles', []):
            if not a.get('title') or not a.get('publishedAt'):
                continue
            text = f'{a["title"]} {a.get("description") or ""}'
            tickers = extractor.extract(text)
            if not tickers:
                continue
            items.append(NewsItem(id=f'newsapi:{a.get("url")}', headline=a['title'], published=to_utc(a['publishedAt']),
                                  tickers=tickers, summary=(a.get('description') or '')[:2000],
                                  source=f'newsapi/{(a.get("source") or {}).get("name", "")}', url=a.get('url', ''),
                                  kind=KIND_NEWS))
        return items

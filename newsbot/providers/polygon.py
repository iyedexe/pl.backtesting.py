"""Polygon.io reference news (Benzinga-sourced, delayed on the free tier, 5 calls/min).
One call per poll over the whole universe; `insights` carries per-ticker sentiment when present."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from ..models import KIND_NEWS, NewsItem, to_utc
from ._base import APISource


class PolygonNews(APISource):
    name = 'polygon'
    ENV_KEYS = ('POLYGON_API_KEY',)
    BASE = 'https://api.polygon.io/v2/reference/news'
    DEFAULT_INTERVAL = 60.0

    def _fetch(self, since: datetime) -> List[NewsItem]:
        params: Dict[str, Any] = {'published_utc.gte': since.strftime('%Y-%m-%dT%H:%M:%SZ'), 'order': 'desc',
                                  'limit': 100, 'apiKey': self.api_key}
        if len(self.tickers) == 1:
            params['ticker'] = self.tickers[0]
        data = self._get(self.BASE, **params) or {}
        return self.parse(data, set(self.tickers))

    @staticmethod
    def parse(data: Dict[str, Any], universe: set) -> List[NewsItem]:
        items: List[NewsItem] = []
        for r in data.get('results', []):
            tickers = [t.upper() for t in r.get('tickers', []) if not universe or t.upper() in universe]
            if not tickers or not r.get('published_utc'):
                continue
            sentiment = {i['ticker'].upper(): i.get('sentiment') for i in r.get('insights', []) or []
                         if i.get('ticker') and i['ticker'].upper() in tickers}
            items.append(NewsItem(id=f'polygon:{r.get("id")}', headline=r.get('title', ''),
                                  published=to_utc(r['published_utc']), tickers=tickers,
                                  summary=(r.get('description') or '')[:2000],
                                  source=f'polygon/{(r.get("publisher") or {}).get("name", "")}',
                                  url=r.get('article_url', ''), kind=KIND_NEWS,
                                  meta={'sentiment_labels': sentiment} if sentiment else {}))
        return items

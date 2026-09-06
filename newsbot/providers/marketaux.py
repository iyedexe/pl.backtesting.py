"""Marketaux: entity-tagged news with per-symbol sentiment. Free tier ~100 requests/day."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from ..models import KIND_SENTIMENT, NewsItem, to_utc
from ._base import APISource, _f


class MarketauxNews(APISource):
    name = 'marketaux'
    ENV_KEYS = ('MARKETAUX_API_KEY',)
    BASE = 'https://api.marketaux.com/v1/news/all'
    DEFAULT_INTERVAL = 900.0

    def _fetch(self, since: datetime) -> List[NewsItem]:
        data = self._get(self.BASE, symbols=','.join(self.tickers) or None, filter_entities='true',
                         published_after=since.strftime('%Y-%m-%dT%H:%M'), language='en', limit=50,
                         api_token=self.api_key) or {}
        return self.parse(data, set(self.tickers))

    @staticmethod
    def parse(data: Dict[str, Any], universe: set) -> List[NewsItem]:
        items: List[NewsItem] = []
        for a in data.get('data', []):
            sent = {}
            for e in a.get('entities', []) or []:
                sym = (e.get('symbol') or '').upper()
                if sym and (not universe or sym in universe):
                    sent[sym] = _f(e.get('sentiment_score'))
            if not sent or not a.get('published_at'):
                continue
            items.append(NewsItem(id=f'marketaux:{a.get("uuid")}', headline=a.get('title', ''),
                                  published=to_utc(a['published_at']), tickers=list(sent),
                                  summary=(a.get('description') or a.get('snippet') or '')[:2000],
                                  source=f'marketaux/{a.get("source", "")}', url=a.get('url', ''),
                                  kind=KIND_SENTIMENT, meta={'ticker_sentiment': sent}))
        return items

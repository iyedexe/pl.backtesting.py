"""Financial Modeling Prep: stock news and earnings calendar. FMP has been migrating from /api/v3 to /stable
endpoints; the defaults below target /stable and can be overridden with `base_url`."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from ..models import KIND_NEWS, NewsItem, to_utc
from ._base import APISource, earnings_result_item, earnings_upcoming_item


class FMPNews(APISource):
    name = 'fmp'
    ENV_KEYS = ('FMP_API_KEY',)
    BASE = 'https://financialmodelingprep.com/stable/news/stock'
    DEFAULT_INTERVAL = 300.0

    def _fetch(self, since: datetime) -> List[NewsItem]:
        data = self._get(self.BASE, symbols=','.join(self.tickers) or None, limit=100, apikey=self.api_key) or []
        return self.parse(data if isinstance(data, list) else data.get('content', []), set(self.tickers))

    @staticmethod
    def parse(rows: List[Dict[str, Any]], universe: set) -> List[NewsItem]:
        items: List[NewsItem] = []
        for r in rows:
            sym = (r.get('symbol') or '').upper()
            if not sym or (universe and sym not in universe) or not r.get('publishedDate') or not r.get('title'):
                continue
            items.append(NewsItem(id=f'fmp:{r.get("url") or r["title"]}', headline=r['title'],
                                  published=to_utc(r['publishedDate']), tickers=[sym],
                                  summary=(r.get('text') or '')[:2000],
                                  source=f'fmp/{r.get("site") or r.get("publisher", "")}',
                                  url=r.get('url', ''), kind=KIND_NEWS))
        return items


class FMPEarningsCalendar(APISource):
    name = 'fmp-earnings'
    ENV_KEYS = ('FMP_API_KEY',)
    BASE = 'https://financialmodelingprep.com/stable/earnings-calendar'
    DEFAULT_INTERVAL = 900.0
    LOOKAHEAD_DAYS = 10

    def _fetch(self, since: datetime) -> List[NewsItem]:
        to = (datetime.now(timezone.utc) + timedelta(days=self.LOOKAHEAD_DAYS)).date().isoformat()
        data = self._get(self.BASE, **{'from': since.date().isoformat()}, to=to, apikey=self.api_key) or []
        return self.parse(data, set(self.tickers))

    @staticmethod
    def parse(rows: List[Dict[str, Any]], universe: set) -> List[NewsItem]:
        items: List[NewsItem] = []
        now = datetime.now(timezone.utc)
        for r in rows:
            sym = (r.get('symbol') or '').upper()
            if not sym or (universe and sym not in universe) or not r.get('date'):
                continue
            day = to_utc(r['date'])
            eps_actual = r.get('epsActual', r.get('eps'))
            eps_est = r.get('epsEstimated')
            hour = (r.get('time') or '').lower()
            if eps_actual is not None:
                when = day.replace(hour=21, minute=5) if hour == 'amc' else day.replace(hour=12, minute=0)
                items.append(earnings_result_item(
                    sym, min(when, now), source='fmp', eps_actual=eps_actual, eps_estimate=eps_est,
                    revenue_actual=r.get('revenueActual', r.get('revenue')), revenue_estimate=r.get('revenueEstimated'),
                    extra={'hour': hour}))
            else:
                items.append(earnings_upcoming_item(sym, day, source='fmp', hour=hour, eps_estimate=eps_est))
        return items

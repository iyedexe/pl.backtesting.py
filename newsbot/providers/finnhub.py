"""Finnhub (https://finnhub.io): company news and the earnings calendar. Free tier: 60 calls/min."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from ..models import KIND_NEWS, NewsItem, to_utc
from ._base import APISource, earnings_result_item, earnings_upcoming_item


class FinnhubNews(APISource):
    name = 'finnhub'
    ENV_KEYS = ('FINNHUB_API_KEY',)
    BASE = 'https://finnhub.io/api/v1'
    DEFAULT_INTERVAL = 60.0          # one sweep of the universe per minute

    def _fetch(self, since: datetime) -> List[NewsItem]:
        out: List[NewsItem] = []
        frm, to = since.date().isoformat(), datetime.now(timezone.utc).date().isoformat()
        for t in self.tickers:
            data = self._get(f'{self.BASE}/company-news', symbol=t, **{'from': frm}, to=to, token=self.api_key)
            out += self.parse(data or [], t)
        return out

    @staticmethod
    def parse(rows: List[Dict[str, Any]], ticker: str) -> List[NewsItem]:
        items = []
        for r in rows:
            if not r.get('headline') or r.get('datetime') is None:
                continue
            items.append(NewsItem(id=f'finnhub:{r.get("id") or r["url"]}', headline=r['headline'],
                                  published=to_utc(int(r['datetime'])), tickers=[ticker],
                                  summary=r.get('summary') or '', source=f'finnhub/{r.get("source", "")}',
                                  url=r.get('url') or '', kind=KIND_NEWS,
                                  meta={'category': r.get('category')}))
        return items


class FinnhubEarningsCalendar(APISource):
    """Reported results (EPS/revenue vs estimate -> KIND_EARNINGS_RESULT) and upcoming report dates."""
    name = 'finnhub-earnings'
    ENV_KEYS = ('FINNHUB_API_KEY',)
    BASE = 'https://finnhub.io/api/v1'
    DEFAULT_INTERVAL = 900.0
    LOOKAHEAD_DAYS = 10

    def _fetch(self, since: datetime) -> List[NewsItem]:
        frm = since.date().isoformat()
        to = (datetime.now(timezone.utc) + timedelta(days=self.LOOKAHEAD_DAYS)).date().isoformat()
        data = self._get(f'{self.BASE}/calendar/earnings', **{'from': frm}, to=to, token=self.api_key) or {}
        return self.parse(data, set(self.tickers))

    @staticmethod
    def parse(data: Dict[str, Any], universe: set) -> List[NewsItem]:
        items: List[NewsItem] = []
        now = datetime.now(timezone.utc)
        for r in data.get('earningsCalendar', []):
            sym = (r.get('symbol') or '').upper()
            if not sym or (universe and sym not in universe) or not r.get('date'):
                continue
            day = to_utc(r['date'])
            hour = r.get('hour') or ''
            if r.get('epsActual') is not None:
                # bmo = before market open (~12:00 UTC), amc = after close (~21:05 UTC)
                when = day.replace(hour=21, minute=5) if hour == 'amc' else day.replace(hour=12, minute=0)
                items.append(earnings_result_item(
                    sym, min(when, now), source='finnhub', eps_actual=r['epsActual'], eps_estimate=r.get('epsEstimate'),
                    revenue_actual=r.get('revenueActual'), revenue_estimate=r.get('revenueEstimate'),
                    extra={'quarter': r.get('quarter'), 'year': r.get('year'), 'hour': hour}))
            else:
                items.append(earnings_upcoming_item(sym, day, source='finnhub', hour=hour,
                                                    eps_estimate=r.get('epsEstimate')))
        return items

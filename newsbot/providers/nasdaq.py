"""Nasdaq.com earnings calendar (no key). The endpoint expects browser-like headers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from ..models import NewsItem, to_utc
from ._base import APISource, earnings_upcoming_item


class NasdaqEarningsCalendar(APISource):
    name = 'nasdaq-earnings'
    ENV_KEYS = ()
    BASE = 'https://api.nasdaq.com/api/calendar/earnings'
    DEFAULT_INTERVAL = 3600.0
    LOOKAHEAD_DAYS = 7

    def _headers(self) -> Dict[str, str]:
        return {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/120 Safari/537.36',
                'Accept': 'application/json, text/plain, */*', 'Accept-Language': 'en-US,en;q=0.9',
                'Origin': 'https://www.nasdaq.com', 'Referer': 'https://www.nasdaq.com/'}

    def _fetch(self, since: datetime) -> List[NewsItem]:
        out: List[NewsItem] = []
        today = datetime.now(timezone.utc).date()
        for d in range(self.LOOKAHEAD_DAYS + 1):
            day = today + timedelta(days=d)
            if day.weekday() >= 5:
                continue
            data = self._get(self.BASE, date=day.isoformat()) or {}
            out += self.parse(data, day.isoformat(), set(self.tickers))
        return out

    @staticmethod
    def parse(data: Dict[str, Any], day: str, universe: set) -> List[NewsItem]:
        items: List[NewsItem] = []
        rows = ((data.get('data') or {}).get('rows')) or []
        for r in rows:
            sym = (r.get('symbol') or '').upper()
            if not sym or (universe and sym not in universe):
                continue
            t = (r.get('time') or '')
            hour = 'amc' if 'after' in t else 'bmo' if 'pre' in t else ''
            eps = str(r.get('epsForecast') or '').replace('$', '').replace('(', '-').replace(')', '')
            items.append(earnings_upcoming_item(sym, to_utc(day), source='nasdaq', hour=hour, eps_estimate=eps,
                                                extra={'name': r.get('name'), 'analysts': r.get('noOfEsts')}))
        return items

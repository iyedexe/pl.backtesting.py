"""
Alpaca integration (https://alpaca.markets): news (Benzinga feed), latest
prices, and a broker that submits bracket orders. Paper trading by default.

Credentials come from `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` (or constructor args).
Only `requests` is required.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from .brokers import Broker, BrokerError, BrokerPosition
from .models import Fill, NewsItem, to_utc
from .prices import PriceFeed
from .sources import NewsSource

log = logging.getLogger(__name__)

PAPER_URL = 'https://paper-api.alpaca.markets'
LIVE_URL = 'https://api.alpaca.markets'
DATA_URL = 'https://data.alpaca.markets'


class AlpacaClient:
    def __init__(self, key: Optional[str] = None, secret: Optional[str] = None, *, paper: bool = True,
                 timeout: float = 15.0):
        self.key = key or os.environ.get('APCA_API_KEY_ID', '')
        self.secret = secret or os.environ.get('APCA_API_SECRET_KEY', '')
        if not self.key or not self.secret:
            raise BrokerError('Alpaca credentials missing: set APCA_API_KEY_ID and APCA_API_SECRET_KEY')
        self.trading_url = PAPER_URL if paper else LIVE_URL
        self.timeout = timeout
        import requests  # noqa: PLC0415
        self._s = requests.Session()
        self._s.headers.update({'APCA-API-KEY-ID': self.key, 'APCA-API-SECRET-KEY': self.secret,
                                'Accept': 'application/json'})

    def _req(self, method: str, url: str, **kw):
        r = self._s.request(method, url, timeout=self.timeout, **kw)
        if r.status_code >= 400:
            raise BrokerError(f'{method} {url} -> {r.status_code}: {r.text[:300]}')
        return r.json() if r.text else None

    def get(self, path: str, base: str = '', **params):
        return self._req('GET', (base or self.trading_url) + path, params=params or None)

    def post(self, path: str, body: dict):
        return self._req('POST', self.trading_url + path, json=body)

    def delete(self, path: str):
        return self._req('DELETE', self.trading_url + path)


class AlpacaNewsSource(NewsSource):
    name = 'alpaca'

    def __init__(self, client: AlpacaClient, symbols: Optional[Iterable[str]] = None, limit: int = 50):
        self.client = client
        self.symbols = [s.upper() for s in symbols] if symbols else []
        self.limit = limit

    def fetch(self, since: Optional[datetime] = None) -> List[NewsItem]:
        params: Dict[str, object] = {'limit': self.limit, 'sort': 'desc', 'include_content': 'false'}
        if since is not None:
            params['start'] = to_utc(since).strftime('%Y-%m-%dT%H:%M:%SZ')
        if self.symbols:
            params['symbols'] = ','.join(self.symbols)
        try:
            data = self.client.get('/v1beta1/news', base=DATA_URL, **params) or {}
        except Exception as e:  # noqa: BLE001
            log.warning('alpaca news fetch failed: %s', e)
            return []
        items = []
        for n in data.get('news', []):
            items.append(NewsItem(id=f'alpaca:{n["id"]}', headline=n.get('headline', ''),
                                  published=to_utc(n.get('created_at') or n.get('updated_at')),
                                  tickers=n.get('symbols', []), summary=n.get('summary', '') or '',
                                  source=n.get('source', 'alpaca'), url=n.get('url', '') or ''))
        items.sort(key=lambda i: i.published)
        return items


class AlpacaPriceFeed(PriceFeed):
    def __init__(self, client: AlpacaClient, feed: str = 'iex'):
        self.client = client
        self.feed = feed

    def price(self, ticker: str) -> Optional[float]:
        try:
            data = self.client.get(f'/v2/stocks/{ticker.upper()}/trades/latest', base=DATA_URL, feed=self.feed)
            return float(data['trade']['p'])
        except Exception as e:  # noqa: BLE001
            log.warning('alpaca price for %s failed: %s', ticker, e)
            return None


class AlpacaBroker(Broker):
    manages_exits = True   # bracket orders: take-profit and stop legs live on the exchange side

    def __init__(self, client: AlpacaClient, *, fill_wait_seconds: float = 10.0):
        self.client = client
        self.fill_wait_seconds = fill_wait_seconds

    def _account(self) -> dict:
        return self.client.get('/v2/account')

    def cash(self) -> float:
        return float(self._account()['cash'])

    def equity(self) -> float:
        return float(self._account()['equity'])

    def positions(self) -> Dict[str, BrokerPosition]:
        out = {}
        for p in self.client.get('/v2/positions') or []:
            out[p['symbol']] = BrokerPosition(p['symbol'], float(p['qty']), float(p['avg_entry_price']))
        return out

    def is_market_open(self, now: Optional[datetime] = None) -> bool:
        try:
            return bool(self.client.get('/v2/clock')['is_open'])
        except BrokerError as e:
            log.warning('alpaca clock failed: %s', e)
            return False

    def _wait_fill(self, order: dict, fallback_price: Optional[float]) -> Fill:
        deadline = time.monotonic() + self.fill_wait_seconds
        while True:
            if order.get('filled_avg_price'):
                break
            if time.monotonic() > deadline:
                log.warning('order %s not filled after %.0fs; using reference price',
                            order['id'], self.fill_wait_seconds)
                break
            time.sleep(1.0)
            order = self.client.get(f'/v2/orders/{order["id"]}')
        price = float(order.get('filled_avg_price') or fallback_price or 0)
        qty = float(order.get('filled_qty') or order.get('qty') or 0)
        filled_at = order.get('filled_at')
        when = to_utc(filled_at) if filled_at else datetime.now(timezone.utc)
        return Fill(order['symbol'], qty, price, when, order_id=order['id'])

    def buy(self, ticker: str, qty: float, *, take_profit=None, stop_loss=None, reference_price=None) -> Fill:
        body: Dict[str, object] = {'symbol': ticker.upper(), 'qty': str(int(qty)), 'side': 'buy',
                                   'type': 'market', 'time_in_force': 'day'}
        if take_profit and stop_loss:
            body.update({'order_class': 'bracket',
                         'take_profit': {'limit_price': f'{take_profit:.2f}'},
                         'stop_loss': {'stop_price': f'{stop_loss:.2f}'}})
        order = self.client.post('/v2/orders', body)
        return self._wait_fill(order, reference_price)

    def sell(self, ticker: str, qty: float) -> Fill:
        ticker = ticker.upper()
        # Cancel any open bracket legs first, otherwise Alpaca rejects the position close.
        for o in self.client.get('/v2/orders', status='open', symbols=ticker) or []:
            try:
                self.client.delete(f'/v2/orders/{o["id"]}')
            except BrokerError as e:
                log.warning('cancel %s failed: %s', o['id'], e)
        order = self.client.delete(f'/v2/positions/{ticker}')
        if not order:
            raise BrokerError(f'close position {ticker} returned no order')
        return self._wait_fill(order, None)

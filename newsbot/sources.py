"""
News sources. Each source yields `NewsItem`s newer than a given timestamp.

- `FileNewsSource`   : JSON/CSV file, for replay and tests.
- `RSSNewsSource`    : any RSS/Atom feed; tickers are extracted from the text
                       ("(NASDAQ: XYZ)", "$XYZ", or a configured symbol list).
- `YahooFinanceRSS`  : per-ticker Yahoo Finance headline feeds.
- `AlpacaNewsSource` : see `newsbot.alpaca` (Benzinga feed via Alpaca's data API).
"""
from __future__ import annotations

import csv
import json
import logging
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .http import RateLimiter, get_text
from .models import KIND_NEWS, NewsItem, to_utc
from .universe import EXCHANGE_SUFFIX

log = logging.getLogger(__name__)


class NewsSource(ABC):
    name: str = 'source'

    @abstractmethod
    def fetch(self, since: Optional[datetime] = None) -> List[NewsItem]:
        """Return items published after `since` (all items if None). Must not raise on transient errors."""


# ---------------------------------------------------------------- tickers

_EXCHANGE_RX = re.compile(r'\(\s*(' + '|'.join(re.escape(k) for k in sorted(EXCHANGE_SUFFIX, key=len, reverse=True))
                          + r')\s*:\s*([A-Z0-9]{1,6}(?:[.-][A-Z0-9]{1,2})?\.?)\s*\)', re.IGNORECASE)
_CASHTAG_RX = re.compile(r'(?<![\w$])\$([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b')


def exchange_tag_ticker(exchange: str, symbol: str) -> str:
    """"(Euronext Paris: AIR)" -> "AIR.PA"; "(NASDAQ: AAPL)" -> "AAPL"; "(LSE: BP.)" -> "BP.L"."""
    suffix = EXCHANGE_SUFFIX.get(exchange.upper().strip(), '')
    sym = symbol.upper().rstrip('.')
    if suffix and not sym.endswith(suffix):
        sym = sym.replace('.', '-') + suffix     # BRK.B on a foreign exchange is rare; keep US dots for US
    return sym if suffix else symbol.upper()


class TickerExtractor:
    """Find ticker symbols mentioned in free text: "(NASDAQ: XYZ)", "$XYZ", bare known symbols,
    and company-name aliases (`{'AAPL': ['Apple', 'Apple Inc.']}`), which matter for sources that
    never print tickers (FDA, general news)."""

    def __init__(self, known: Optional[Iterable[str]] = None, aliases: Optional[Mapping[str, Iterable[str]]] = None):
        self.known: Set[str] = {k.upper() for k in known} if known else set()
        self._known_rx = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(self.known)) + r')\b') \
            if self.known else None
        self._aliases: List[Tuple[re.Pattern, str]] = []
        for ticker, names in (aliases or {}).items():
            for name in ([names] if isinstance(names, str) else names):
                self._aliases.append((re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE), ticker.upper()))

    def extract(self, text: str) -> List[str]:
        found: List[str] = [exchange_tag_ticker(ex, sym) for ex, sym in _EXCHANGE_RX.findall(text)]
        found += [m.upper() for m in _CASHTAG_RX.findall(text)]
        if self._known_rx:
            found += self._known_rx.findall(text)
        for rx, ticker in self._aliases:
            if rx.search(text):
                found.append(ticker)
        seen: Set[str] = set()
        return [t for t in found if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]


# ---------------------------------------------------------------- file

class FileNewsSource(NewsSource):
    """Reads a JSON list (or CSV) of news records. Handy for replay/backtests/tests."""
    name = 'file'

    def __init__(self, path, items: Optional[Sequence[NewsItem]] = None):
        self.path = Path(path) if path else None
        self._items: List[NewsItem] = list(items) if items is not None else self._load()

    def _load(self) -> List[NewsItem]:
        assert self.path is not None
        if self.path.suffix.lower() == '.csv':
            with open(self.path, newline='', encoding='utf-8') as f:
                records = list(csv.DictReader(f))
        else:
            with open(self.path, encoding='utf-8') as f:
                data = json.load(f)
            records = data['news'] if isinstance(data, dict) else data
        items = [NewsItem.from_dict(r) for r in records]
        items.sort(key=lambda i: i.published)
        return items

    @property
    def items(self) -> List[NewsItem]:
        return list(self._items)

    def fetch(self, since: Optional[datetime] = None) -> List[NewsItem]:
        if since is None:
            return list(self._items)
        since = to_utc(since)
        return [i for i in self._items if i.published > since]


# ---------------------------------------------------------------- rss

def parse_rss(xml_text: str, *, source: str = 'rss', kind: str = KIND_NEWS) -> List[NewsItem]:
    """Parse RSS 2.0 or Atom into NewsItems (tickers left empty)."""
    root = ET.fromstring(xml_text)
    items: List[NewsItem] = []

    def _text(el, *names):
        for n in names:
            child = el.find(n)
            if child is not None and (child.text or '').strip():
                return child.text.strip()
        return ''

    def _date(s: str) -> Optional[datetime]:
        if not s:
            return None
        try:
            return to_utc(parsedate_to_datetime(s))
        except (TypeError, ValueError):
            pass
        try:
            return to_utc(s)
        except ValueError:
            return None

    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('.//item') or root.findall('.//atom:entry', ns)
    for e in entries:
        title = _text(e, 'title', 'atom:title')
        if not title:
            continue
        link_el = e.find('atom:link', ns)
        link = _text(e, 'link') or (link_el.get('href', '') if link_el is not None else '')
        published = _date(_text(e, 'pubDate', 'dc:date', 'atom:published', 'atom:updated', 'published', 'updated'))
        if published is None:
            continue
        guid = _text(e, 'guid', 'atom:id') or link or f'{title}|{published.isoformat()}'
        summary = re.sub(r'<[^>]+>', ' ', _text(e, 'description', 'atom:summary', 'atom:content', 'summary'))
        items.append(NewsItem(id=f'{source}:{guid}', headline=title, published=published,
                              summary=' '.join(summary.split())[:2000], source=source, url=link, kind=kind))
    return items


class RSSNewsSource(NewsSource):
    """Generic RSS/Atom source. `tickers` pins every item to fixed symbols (per-ticker feeds);
    otherwise symbols are extracted from the text."""

    def __init__(self, url: str, *, name: Optional[str] = None, tickers: Optional[Iterable[str]] = None,
                 extractor: Optional[TickerExtractor] = None, timeout: float = 10.0, kind: str = KIND_NEWS,
                 min_interval: float = 0.0, strip_suffix: bool = False):
        self.url = url
        self.name = name or f'rss:{re.sub(r"^https?://", "", url)[:40]}'
        self.fixed_tickers = [t.upper() for t in tickers] if tickers else []
        self.extractor = extractor or TickerExtractor()
        self.timeout = timeout
        self.kind = kind
        self.strip_suffix = strip_suffix       # Google News appends " - Publisher" to titles
        self._limiter = RateLimiter(min_interval)

    def _download(self) -> Optional[str]:
        return get_text(self.url, timeout=self.timeout)

    def fetch(self, since: Optional[datetime] = None) -> List[NewsItem]:
        if not self._limiter.ready():
            return []
        xml_text = self._download()
        if not xml_text:
            return []
        try:
            items = parse_rss(xml_text, source=self.name, kind=self.kind)
        except Exception as e:  # noqa: BLE001 — parse errors must not kill the loop
            log.warning('%s: parse failed: %s', self.name, e)
            return []
        out = []
        for it in items:
            if self.strip_suffix and ' - ' in it.headline:
                it.headline = it.headline.rsplit(' - ', 1)[0]
            it.tickers = self.fixed_tickers or self.extractor.extract(f'{it.headline} {it.summary}')
            if since is not None and it.published <= to_utc(since):
                continue
            out.append(it)
        return out


TickerProvider = Union[Iterable[str], Callable[[], Iterable[str]]]


class PerTickerRSS(NewsSource):
    """One RSS feed per ticker built from a URL template. `tickers` may be a callable (e.g. the bot's
    currently active tickers in market mode); feeds are created lazily and capped at `max_tickers`."""
    name = 'per-ticker-rss'
    URL = ''
    KIND = KIND_NEWS
    STRIP_SUFFIX = False

    def __init__(self, tickers: TickerProvider, timeout: float = 10.0, min_interval: float = 0.0,
                 url: Optional[str] = None, max_tickers: int = 50):
        self._template = url or self.URL
        self._provider = tickers if callable(tickers) else (lambda: list(tickers))  # type: ignore[misc]
        self._timeout, self._min_interval, self.max_tickers = timeout, min_interval, max_tickers
        self._feeds: Dict[str, RSSNewsSource] = {}

    def _feed(self, t: str) -> RSSNewsSource:
        if t not in self._feeds:
            self._feeds[t] = RSSNewsSource(self._template.format(ticker=t, ticker_lower=t.lower()),
                                           name=f'{self.name}:{t}', tickers=[t], timeout=self._timeout,
                                           kind=self.KIND, min_interval=self._min_interval,
                                           strip_suffix=self.STRIP_SUFFIX)
        return self._feeds[t]

    @property
    def feeds(self) -> List[RSSNewsSource]:
        return [self._feed(t.upper()) for t in list(self._provider())[:self.max_tickers]]

    def fetch(self, since: Optional[datetime] = None) -> List[NewsItem]:
        out: List[NewsItem] = []
        for f in self.feeds:
            out += f.fetch(since)
        return out


class YahooFinanceRSS(PerTickerRSS):
    """Yahoo Finance headline feed per ticker (no key, minutes of latency)."""
    name = 'yahoo'
    URL = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'


class GoogleNewsRSS(PerTickerRSS):
    """Google News search feed per ticker (no key). Titles carry a " - Publisher" suffix that is stripped."""
    name = 'google'
    URL = 'https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en'
    STRIP_SUFFIX = True


class NasdaqRSS(PerTickerRSS):
    """Nasdaq.com press releases + articles per ticker (no key)."""
    name = 'nasdaq'
    URL = 'https://www.nasdaq.com/feed/rssoutbound?symbol={ticker}'
    KIND = 'filing'


#: Market-wide press-release / regulatory-news feeds. Company tickers are extracted from
#: "(EXCHANGE: SYMBOL)" tags in the text. URLs change occasionally; override with `url:` if one 404s.
WIRE_FEEDS: Dict[str, Dict[str, str]] = {
    'globenewswire_earnings': {
        'url': 'https://www.globenewswire.com/RssFeed/subjectcode/12-Earnings%20Releases%20And%20Operating%20Results/'
               'feedTitle/GlobeNewswire%20-%20Earnings%20Releases%20And%20Operating%20Results',
        'kind': 'filing', 'markets': 'us'},
    'globenewswire_public': {
        'url': 'https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20about%20Public'
               '%20Companies', 'kind': 'filing', 'markets': 'us'},
    'globenewswire_europe': {
        'url': 'https://www.globenewswire.com/RssFeed/region/Europe/feedTitle/GlobeNewswire%20-%20News%20from%20Europe',
        'kind': 'filing', 'markets': 'eu,uk'},
    'prnewswire_earnings': {
        'url': 'https://www.prnewswire.com/rss/financial-services-latest-news/earnings-list.rss',
        'kind': 'filing', 'markets': 'us'},
    'prnewswire_all': {'url': 'https://www.prnewswire.com/rss/news-releases-list.rss', 'kind': 'filing',
                       'markets': 'us'},
    'businesswire_earnings': {
        'url': 'https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEFpRXA%3D%3D', 'kind': 'filing',
        'markets': 'us'},
    'accesswire': {'url': 'https://www.accesswire.com/rss/latest', 'kind': 'filing', 'markets': 'us'},
    'eqs_adhoc': {'url': 'https://www.eqs-news.com/rss/news?news_type=adhoc', 'kind': 'filing', 'markets': 'eu'},
    'lse_rns': {'url': 'https://www.investegate.co.uk/rss/announcements.rss', 'kind': 'filing', 'markets': 'uk'},
    'sec_8k': {'url': 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&count=100&output=atom',
               'kind': 'filing', 'markets': 'us'},
}


class WireFeed(RSSNewsSource):
    """A named entry of `WIRE_FEEDS`."""

    def __init__(self, feed: str, extractor: TickerExtractor, *, url: Optional[str] = None, **kw):
        if feed not in WIRE_FEEDS:
            raise ValueError(f'unknown wire feed {feed!r}; choose from {sorted(WIRE_FEEDS)}')
        spec = WIRE_FEEDS[feed]
        kw.setdefault('min_interval', 120.0)
        kw.setdefault('kind', spec['kind'])
        super().__init__(url or spec['url'], name=f'wire:{feed}', extractor=extractor, **kw)
        self.markets = spec['markets'].split(',')


class CompositeNewsSource(NewsSource):
    name = 'composite'

    def __init__(self, sources: Iterable[NewsSource]):
        self.sources = list(sources)

    def fetch(self, since: Optional[datetime] = None) -> List[NewsItem]:
        out: List[NewsItem] = []
        for s in self.sources:
            try:
                out += s.fetch(since)
            except Exception as e:  # noqa: BLE001
                log.warning('%s: fetch failed: %s', getattr(s, 'name', s), e)
        out.sort(key=lambda i: i.published)
        return out

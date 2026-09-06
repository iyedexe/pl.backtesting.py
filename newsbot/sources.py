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
from typing import Iterable, List, Optional, Sequence, Set

from .models import NewsItem, to_utc

log = logging.getLogger(__name__)


class NewsSource(ABC):
    name: str = 'source'

    @abstractmethod
    def fetch(self, since: Optional[datetime] = None) -> List[NewsItem]:
        """Return items published after `since` (all items if None). Must not raise on transient errors."""


# ---------------------------------------------------------------- tickers

_EXCHANGE_RX = re.compile(r'\((?:NASDAQ|NYSE|NYSE American|NYSE ?MKT|AMEX|TSX|TSXV|OTCQB|OTCQX|OTC)\s*:\s*'
                          r'([A-Z]{1,5}(?:[.-][A-Z])?)\)', re.IGNORECASE)
_CASHTAG_RX = re.compile(r'(?<![\w$])\$([A-Z]{1,5})\b')


class TickerExtractor:
    """Find ticker symbols mentioned in free text."""

    def __init__(self, known: Optional[Iterable[str]] = None):
        self.known: Set[str] = {k.upper() for k in known} if known else set()
        self._known_rx = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(self.known)) + r')\b') \
            if self.known else None

    def extract(self, text: str) -> List[str]:
        found: List[str] = []
        for rx in (_EXCHANGE_RX, _CASHTAG_RX):
            found += [m.upper() for m in rx.findall(text)]
        if self._known_rx:
            found += self._known_rx.findall(text)
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

def parse_rss(xml_text: str, *, source: str = 'rss') -> List[NewsItem]:
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
                              summary=' '.join(summary.split())[:2000], source=source, url=link))
    return items


class RSSNewsSource(NewsSource):
    """Generic RSS/Atom source. Uses `requests` (already a dependency) with a short timeout."""

    def __init__(self, url: str, *, name: Optional[str] = None, tickers: Optional[Iterable[str]] = None,
                 extractor: Optional[TickerExtractor] = None, timeout: float = 10.0):
        self.url = url
        self.name = name or f'rss:{re.sub(r"^https?://", "", url)[:40]}'
        self.fixed_tickers = [t.upper() for t in tickers] if tickers else []
        self.extractor = extractor or TickerExtractor()
        self.timeout = timeout

    def _download(self) -> str:
        import requests  # noqa: PLC0415
        r = requests.get(self.url, timeout=self.timeout, headers={'User-Agent': 'newsbot/1.0'})
        r.raise_for_status()
        return r.text

    def fetch(self, since: Optional[datetime] = None) -> List[NewsItem]:
        try:
            xml_text = self._download()
            items = parse_rss(xml_text, source=self.name)
        except Exception as e:  # noqa: BLE001 — network / parse errors must not kill the loop
            log.warning('%s: fetch failed: %s', self.name, e)
            return []
        out = []
        for it in items:
            it.tickers = self.fixed_tickers or self.extractor.extract(f'{it.headline} {it.summary}')
            if since is not None and it.published <= to_utc(since):
                continue
            out.append(it)
        return out


class YahooFinanceRSS(NewsSource):
    """One Yahoo Finance headline feed per ticker."""
    name = 'yahoo'
    URL = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'

    def __init__(self, tickers: Iterable[str], timeout: float = 10.0):
        self.feeds = [RSSNewsSource(self.URL.format(ticker=t), name=f'yahoo:{t}', tickers=[t], timeout=timeout)
                      for t in tickers]

    def fetch(self, since: Optional[datetime] = None) -> List[NewsItem]:
        out: List[NewsItem] = []
        for f in self.feeds:
            out += f.fetch(since)
        return out


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

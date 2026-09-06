"""
Core data types shared by every layer of the news bot:
news items in, classifications/signals in the middle, positions and trades out.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_utc(value: Any) -> datetime:
    """Coerce a datetime / ISO string / epoch to a tz-aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, str):
        s = value.strip()
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        dt = datetime.fromisoformat(s)
    else:  # pandas.Timestamp and friends
        dt = value.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _dt(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.isoformat()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    return None if value is None else to_utc(value)


class ExitReason(str, Enum):
    TARGET = 'target'
    STOP = 'stop'
    TIME = 'time'
    BROKER = 'broker'   # closed on the broker side (e.g. bracket leg filled)
    MANUAL = 'manual'


@dataclass
class NewsItem:
    """A single headline as delivered by a `NewsSource`."""
    id: str
    headline: str
    published: datetime
    tickers: List[str] = field(default_factory=list)
    summary: str = ''
    source: str = ''
    url: str = ''

    def __post_init__(self):
        self.published = to_utc(self.published)
        self.tickers = [t.upper().strip() for t in self.tickers if t and t.strip()]

    @property
    def text(self) -> str:
        return f'{self.headline}. {self.summary}'.strip(' .') + '.'

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['published'] = self.published.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NewsItem':
        tickers = d.get('tickers') or d.get('symbols') or []
        if isinstance(tickers, str):
            tickers = [t for t in tickers.replace(';', ',').split(',')]
        if not tickers and d.get('ticker'):
            tickers = [d['ticker']]
        return cls(
            id=str(d.get('id') or uuid.uuid5(uuid.NAMESPACE_URL, f"{d.get('headline')}|{d.get('published')}")),
            headline=str(d.get('headline') or d.get('title') or ''),
            published=to_utc(d.get('published') or d.get('created_at') or d.get('date') or utcnow()),
            tickers=list(tickers),
            summary=str(d.get('summary') or d.get('description') or ''),
            source=str(d.get('source') or ''),
            url=str(d.get('url') or d.get('link') or ''),
        )


@dataclass
class Classification:
    """Output of a `Classifier`. `score` is in [-1, 1]; only positive scores can open longs."""
    score: float
    category: str = 'none'
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)

    @property
    def is_bullish(self) -> bool:
        return self.score > 0


@dataclass
class Signal:
    """An actionable long entry derived from one news item for one ticker."""
    ticker: str
    news_id: str
    headline: str
    score: float
    category: str
    created_at: datetime
    expires_at: datetime
    target_pct: float
    stop_pct: float
    max_hold_days: float
    reference_price: Optional[float] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def target_price(self, entry_price: float) -> float:
        return round(entry_price * (1 + self.target_pct), 4)

    def stop_price(self, entry_price: float) -> float:
        return round(entry_price * (1 - self.stop_pct), 4)

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['created_at'] = _dt(self.created_at)
        d['expires_at'] = _dt(self.expires_at)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Signal':
        d = dict(d)
        d['created_at'] = _parse_dt(d['created_at'])
        d['expires_at'] = _parse_dt(d['expires_at'])
        return cls(**d)


@dataclass
class Position:
    """An open long position managed by the bot."""
    ticker: str
    qty: float
    entry_price: float
    entry_time: datetime
    target_price: float
    stop_price: float
    exit_by: datetime
    signal_id: str = ''
    news_id: str = ''
    headline: str = ''
    broker_order_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['entry_time'] = _dt(self.entry_time)
        d['exit_by'] = _dt(self.exit_by)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Position':
        d = dict(d)
        d['entry_time'] = _parse_dt(d['entry_time'])
        d['exit_by'] = _parse_dt(d['exit_by'])
        return cls(**d)

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.qty


@dataclass
class ClosedTrade:
    ticker: str
    qty: float
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    reason: str
    headline: str = ''
    signal_id: str = ''

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def return_pct(self) -> float:
        return (self.exit_price / self.entry_price - 1) * 100 if self.entry_price else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['entry_time'] = _dt(self.entry_time)
        d['exit_time'] = _dt(self.exit_time)
        d['pnl'] = round(self.pnl, 4)
        d['return_pct'] = round(self.return_pct, 4)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ClosedTrade':
        d = {k: v for k, v in d.items() if k not in ('pnl', 'return_pct')}
        d['entry_time'] = _parse_dt(d['entry_time'])
        d['exit_time'] = _parse_dt(d['exit_time'])
        return cls(**d)


@dataclass
class Fill:
    ticker: str
    qty: float
    price: float
    time: datetime
    order_id: Optional[str] = None

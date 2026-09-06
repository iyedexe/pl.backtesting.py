"""Social streams (StockTwits, Reddit). Emitted as ONE aggregate KIND_SOCIAL item per ticker per poll
(bull/bear counts, message rate, sample headlines) rather than one item per post."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from ..models import KIND_SOCIAL, NewsItem, to_utc
from ._base import APISource


class StockTwitsStream(APISource):
    name = 'stocktwits'
    ENV_KEYS = ()
    BASE = 'https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json'
    DEFAULT_INTERVAL = 300.0

    def _fetch(self, since: datetime) -> List[NewsItem]:
        out: List[NewsItem] = []
        for t in self.tickers:
            data = self._get(self.BASE.format(ticker=t)) or {}
            item = self.parse(data, t, since)
            if item:
                out.append(item)
        return out

    @staticmethod
    def parse(data: Dict[str, Any], ticker: str, since: datetime) -> Optional[NewsItem]:
        msgs = data.get('messages') or []
        recent = []
        for m in msgs:
            try:
                created = to_utc(m['created_at'])
            except (KeyError, ValueError):
                continue
            if created > since:
                recent.append((created, m))
        if not recent:
            return None
        bull = sum(1 for _, m in recent if ((m.get('entities') or {}).get('sentiment') or {}).get('basic') == 'Bullish')
        bear = sum(1 for _, m in recent if ((m.get('entities') or {}).get('sentiment') or {}).get('basic') == 'Bearish')
        latest = max(c for c, _ in recent)
        span_h = max((latest - min(c for c, _ in recent)).total_seconds() / 3600, 0.1)
        sample = [m.get('body', '')[:120] for _, m in sorted(recent, key=lambda cm: -cm[0].timestamp())[:5]]
        return NewsItem(id=f'stocktwits:{ticker}:{max(m.get("id", 0) for _, m in recent)}',
                        headline=f'StockTwits {ticker}: {len(recent)} posts, {bull} bullish / {bear} bearish',
                        published=latest, tickers=[ticker], summary=' | '.join(sample), source='stocktwits',
                        kind=KIND_SOCIAL,
                        meta={'posts': len(recent), 'bull': bull, 'bear': bear,
                              'posts_per_hour': round(len(recent) / span_h, 1),
                              'bull_ratio': round(bull / (bull + bear), 2) if bull + bear else None})


class RedditMentions(APISource):
    name = 'reddit'
    ENV_KEYS = ()
    BASE = 'https://www.reddit.com/r/{sub}/search.json'
    DEFAULT_INTERVAL = 300.0

    def __init__(self, tickers: Optional[Iterable[str]] = None, *, subreddits: Optional[Iterable[str]] = None, **kw):
        super().__init__(tickers, **kw)
        self.subreddits = list(subreddits or ['wallstreetbets', 'stocks', 'investing'])

    def _headers(self) -> Dict[str, str]:
        return {'User-Agent': 'python:newsbot:1.0 (news trading research)'}

    def _fetch(self, since: datetime) -> List[NewsItem]:
        out: List[NewsItem] = []
        for t in self.tickers:
            posts: List[Dict[str, Any]] = []
            for sub in self.subreddits:
                data = self._get(self.BASE.format(sub=sub), q=t, sort='new', restrict_sr=1, limit=50, t='week') or {}
                posts += [c.get('data', {}) for c in (data.get('data') or {}).get('children', [])]
            item = self.parse(posts, t, since)
            if item:
                out.append(item)
        return out

    @staticmethod
    def parse(posts: List[Dict[str, Any]], ticker: str, since: datetime) -> Optional[NewsItem]:
        recent = [p for p in posts if p.get('created_utc') and
                  datetime.fromtimestamp(p['created_utc'], tz=timezone.utc) > since]
        if not recent:
            return None
        latest = max(datetime.fromtimestamp(p['created_utc'], tz=timezone.utc) for p in recent)
        score = sum(int(p.get('score') or 0) for p in recent)
        comments = sum(int(p.get('num_comments') or 0) for p in recent)
        titles = [p.get('title', '')[:120] for p in sorted(recent, key=lambda p: -(p.get('score') or 0))[:5]]
        return NewsItem(id=f'reddit:{ticker}:{max(p.get("id", "") for p in recent)}',
                        headline=f'Reddit {ticker}: {len(recent)} posts, {score} upvotes, {comments} comments',
                        published=latest, tickers=[ticker], summary=' | '.join(titles), source='reddit',
                        kind=KIND_SOCIAL,
                        meta={'posts': len(recent), 'upvotes': score, 'comments': comments})

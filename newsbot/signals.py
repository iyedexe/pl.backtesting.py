"""
Turn classified news into `Signal`s: which ticker to buy, how far the profit
target and protective stop sit from the fill, and by when the position must be
closed (the "one week max" rule).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Set

from .classifiers import Classifier, RuleClassifier
from .models import NewsItem, Signal

log = logging.getLogger(__name__)


class SignalEngine:
    def __init__(self, classifier: Optional[Classifier] = None, *,
                 min_score: float = 0.5,
                 target_pct: float = 0.05,
                 stop_pct: float = 0.03,
                 max_hold_days: float = 7,
                 signal_ttl: timedelta = timedelta(hours=18),
                 scale_target_by_score: bool = True,
                 universe: Optional[Iterable[str]] = None,
                 categories: Optional[Iterable[str]] = None,
                 blocked_categories: Optional[Iterable[str]] = None):
        """
        min_score: classification score at or above which a long is taken.
        target_pct / stop_pct: exit distances from the fill price.
        max_hold_days: hard time exit, in calendar days after entry (7 = one week).
        signal_ttl: how long an unfilled signal stays valid (18h lets an after-close
            earnings release be bought at the next open).
        scale_target_by_score: if True, stronger catalysts get a wider target
            (x1.0 at score 0.5, x1.25 at score 1.0).
        universe: optional whitelist of tickers; empty/None = trade any ticker the news names.
        categories / blocked_categories: optional catalyst allow/deny lists.
        """
        self.classifier = classifier or RuleClassifier()
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_pct = stop_pct
        self.max_hold_days = max_hold_days
        self.signal_ttl = signal_ttl
        self.scale_target_by_score = scale_target_by_score
        self.universe: Set[str] = {t.upper() for t in universe} if universe else set()
        self.categories: Set[str] = set(categories) if categories else set()
        self.blocked_categories: Set[str] = set(blocked_categories) if blocked_categories else set()

    def effective_target_pct(self, score: float) -> float:
        if not self.scale_target_by_score:
            return self.target_pct
        return round(self.target_pct * (0.75 + 0.5 * max(0.0, min(1.0, score))), 6)

    def evaluate(self, item: NewsItem, now: Optional[datetime] = None) -> List[Signal]:
        """Return zero or more signals (one per eligible ticker) for a news item."""
        tickers = [t for t in item.tickers if not self.universe or t in self.universe]
        if not tickers:
            return []
        c = self.classifier.classify(item)
        log.debug('classified %r -> %.2f %s %s', item.headline, c.score, c.category, c.reasons)
        if c.score < self.min_score:
            return []
        if self.categories and c.category not in self.categories:
            return []
        if c.category in self.blocked_categories:
            return []
        now = now or item.published
        return [Signal(ticker=t, news_id=item.id, headline=item.headline, score=c.score, category=c.category,
                       created_at=now, expires_at=now + self.signal_ttl,
                       target_pct=self.effective_target_pct(c.score), stop_pct=self.stop_pct,
                       max_hold_days=self.max_hold_days)
                for t in tickers]

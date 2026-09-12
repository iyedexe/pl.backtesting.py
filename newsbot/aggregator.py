"""
Per-ticker evidence aggregation.

Every source drops `NewsItem`s into the `EvidenceStore`. When a *trigger* item
(headline, filing, regulatory event, reported earnings) arrives for a ticker, the
store hands back a `Bundle`: every item about that ticker inside the look-back
window plus numeric features distilled from provider metadata. The scorer
(`newsbot.scoring`) then turns the bundle into one score.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Union

from .classifiers import Classifier, RuleClassifier
from .universe import Universe, universe_arg
from .models import (KIND_EARNINGS_RESULT, KIND_EARNINGS_UPCOMING, KIND_REGULATORY, KIND_SENTIMENT, KIND_SOCIAL,
                     NewsItem, to_utc)

_LABEL_SCORES = {'positive': 0.5, 'bullish': 0.5, 'somewhat-bullish': 0.25, 'neutral': 0.0,
                 'somewhat-bearish': -0.25, 'negative': -0.5, 'bearish': -0.5}


def _sentiment_for(item: NewsItem, ticker: str) -> Optional[float]:
    """Extract a provider sentiment in [-1, 1] for `ticker` from an item's meta, if any."""
    m = item.meta or {}
    ts = m.get('ticker_sentiment') or {}
    v = ts.get(ticker)
    if isinstance(v, dict):
        v = v.get('sentiment')
    if isinstance(v, (int, float)):
        return max(-1.0, min(1.0, float(v)))
    labels = m.get('sentiment_labels') or {}
    if ticker in labels and isinstance(labels[ticker], str):
        return _LABEL_SCORES.get(labels[ticker].lower())
    if isinstance(m.get('overall_sentiment'), (int, float)):
        return max(-1.0, min(1.0, float(m['overall_sentiment'])))
    return None


@dataclass
class Bundle:
    ticker: str
    items: List[NewsItem]                  # ascending by published, all inside the window
    trigger: NewsItem                      # newest trigger item
    as_of: datetime
    features: Dict[str, Any] = field(default_factory=dict)

    def by_kind(self, *kinds: str) -> List[NewsItem]:
        return [i for i in self.items if i.kind in kinds]

    @property
    def sources(self) -> List[str]:
        seen: Dict[str, None] = {}
        for i in self.items:
            seen.setdefault(i.source.split('/')[0], None)
        return list(seen)

    def describe(self, max_items: int = 25) -> str:
        """Compact, chronological text rendering for an LLM prompt."""
        lines = []
        items = self.items[-max_items:]
        for i in items:
            age_h = (self.as_of - i.published).total_seconds() / 3600
            extra = ''
            if i.kind == KIND_SENTIMENT:
                s = _sentiment_for(i, self.ticker)
                extra = f' [provider sentiment {s:+.2f}]' if s is not None else ''
            elif i.kind == KIND_EARNINGS_RESULT:
                extra = (f' [eps surprise {i.meta.get("eps_surprise_pct")}%, '
                         f'revenue surprise {i.meta.get("revenue_surprise_pct")}%]')
            elif i.kind == KIND_SOCIAL:
                extra = f' [{i.meta}]'
            mark = '*' if i is self.trigger else ' '
            summary = f' — {i.summary[:240]}' if i.summary else ''
            lines.append(f'{mark}[{i.kind}] {age_h:5.1f}h ago | {i.source}: {i.headline}{summary}{extra}')
        return '\n'.join(lines)


class EvidenceStore:
    def __init__(self, window: timedelta = timedelta(hours=24), classifier: Optional[Classifier] = None,
                 universe: Union[Universe, Iterable[str], None] = None):
        self.window = window
        self.classifier = classifier or RuleClassifier()
        self.universe: Universe = universe_arg(universe)
        self._items: Dict[str, Dict[str, NewsItem]] = defaultdict(dict)

    # -- storage ---------------------------------------------------------
    def add(self, item: NewsItem) -> List[str]:
        affected = []
        for t in item.tickers:
            if self.universe and t not in self.universe:
                continue
            self._items[t][item.id] = item
            affected.append(t)
        return affected

    def prune(self, now: datetime) -> None:
        cutoff = now - self.window
        for t in list(self._items):
            self._items[t] = {k: v for k, v in self._items[t].items() if v.published >= cutoff}
            if not self._items[t]:
                del self._items[t]

    def items_for(self, ticker: str, now: Optional[datetime] = None) -> List[NewsItem]:
        items = list(self._items.get(ticker.upper(), {}).values())
        if now is not None:
            cutoff = now - self.window
            items = [i for i in items if cutoff <= i.published <= now]
        return sorted(items, key=lambda i: i.published)

    def tickers(self) -> List[str]:
        return sorted(self._items)

    def to_dict(self) -> Dict[str, List[dict]]:
        return {t: [i.to_dict() for i in d.values()] for t, d in self._items.items()}

    def load_dict(self, d: Dict[str, List[dict]]) -> None:
        self._items = defaultdict(dict)
        for t, rows in (d or {}).items():
            for r in rows:
                it = NewsItem.from_dict(r)
                self._items[t][it.id] = it

    # -- bundles ---------------------------------------------------------
    def bundle(self, ticker: str, now: datetime, trigger: Optional[NewsItem] = None) -> Optional[Bundle]:
        now = to_utc(now)
        items = self.items_for(ticker, now)
        if trigger is None:
            triggers = [i for i in items if i.is_trigger]
            if not triggers:
                return None
            trigger = triggers[-1]
        b = Bundle(ticker=ticker.upper(), items=items, trigger=trigger, as_of=now)
        b.features = self.features(b)
        return b

    def features(self, b: Bundle) -> Dict[str, Any]:
        f: Dict[str, Any] = {'n_items': len(b.items), 'n_sources': len(b.sources), 'sources': b.sources}
        # rule scores on trigger items, recency weighted (half-life 6h)
        rule = []
        for i in b.items:
            if not i.is_trigger or i.kind == KIND_EARNINGS_RESULT:
                continue
            c = self.classifier.classify(i)
            age_h = max(0.0, (b.as_of - i.published).total_seconds() / 3600)
            rule.append({'id': i.id, 'score': c.score, 'category': c.category, 'weight': 0.5 ** (age_h / 6)})
        f['rule'] = rule
        f['trigger_rule'] = next((r for r in rule if r['id'] == b.trigger.id), None)
        f['rule_max'] = max((r['score'] for r in rule), default=0.0)
        f['rule_min'] = min((r['score'] for r in rule), default=0.0)
        wsum = sum(r['weight'] for r in rule)
        f['rule_weighted'] = round(sum(r['score'] * r['weight'] for r in rule) / wsum, 3) if wsum else 0.0
        # provider sentiment
        sents = [s for s in (_sentiment_for(i, b.ticker) for i in b.items) if s is not None]
        f['sentiment_mean'] = round(sum(sents) / len(sents), 3) if sents else None
        f['sentiment_n'] = len(sents)
        # earnings
        results = b.by_kind(KIND_EARNINGS_RESULT)
        f['earnings'] = results[-1].meta if results else None
        upcoming = b.by_kind(KIND_EARNINGS_UPCOMING)
        f['earnings_upcoming'] = upcoming[-1].meta if upcoming else None
        # regulatory priors (e.g. clinical-trial status)
        reg = [float(i.meta['prior']) for i in b.by_kind(KIND_REGULATORY)
               if isinstance(i.meta.get('prior'), (int, float))]
        f['regulatory_prior'] = round(sum(reg) / len(reg), 3) if reg else None
        # social
        soc = b.by_kind(KIND_SOCIAL)
        if soc:
            ratios = [float(i.meta['bull_ratio']) for i in soc if isinstance(i.meta.get('bull_ratio'), (int, float))]
            f['social'] = {'items': len(soc), 'bull_ratio': round(sum(ratios) / len(ratios), 2) if ratios else None,
                           'posts': sum(int(i.meta.get('posts') or 0) for i in soc)}
        else:
            f['social'] = None
        return f

"""
Bundle scorers: turn a per-ticker evidence `Bundle` into one `Classification`.

- `RuleScorer`  : deterministic ensemble of the per-headline rule scores, provider sentiment,
                  reported earnings surprise, regulatory priors and social tilt.
- `ClaudeScorer`: the AI model reads the whole bundle (plus the rule features as a prior) and
                  returns strict JSON. Falls back to `RuleScorer` on any API failure.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .aggregator import Bundle
from .models import Classification

log = logging.getLogger(__name__)


class Scorer(ABC):
    @abstractmethod
    def score(self, bundle: Bundle) -> Classification:
        ...


class RuleScorer(Scorer):
    def __init__(self, *, w_trigger: float = 0.6, w_weighted: float = 0.3, w_sentiment: float = 0.3,
                 w_social: float = 0.1, w_regulatory: float = 0.5):
        self.w_trigger, self.w_weighted, self.w_sentiment = w_trigger, w_weighted, w_sentiment
        self.w_social, self.w_regulatory = w_social, w_regulatory

    @staticmethod
    def earnings_component(e: Optional[Dict[str, Any]]) -> float:
        if not e:
            return 0.0
        s = e.get('eps_surprise_pct')
        r = e.get('revenue_surprise_pct')
        out = 0.0
        if isinstance(s, (int, float)):
            out += 0.5 if s > 5 else 0.25 if s > 1 else -0.6 if s < -5 else -0.35 if s < -1 else 0.0
        if isinstance(r, (int, float)):
            out += 0.2 if r > 2 else -0.25 if r < -2 else 0.0
        return out

    def score(self, b: Bundle) -> Classification:
        f = b.features
        reasons: List[str] = []
        total = 0.0
        trig = f.get('trigger_rule')
        if trig:
            total += self.w_trigger * trig['score']
            reasons.append(f"trigger {trig['category']} {trig['score']:+.2f}")
        if f.get('rule'):
            total += self.w_weighted * f['rule_weighted']
            reasons.append(f"headlines(weighted) {f['rule_weighted']:+.2f} over {len(f['rule'])}")
        if f.get('sentiment_mean') is not None:
            total += self.w_sentiment * f['sentiment_mean']
            reasons.append(f"provider sentiment {f['sentiment_mean']:+.2f} (n={f['sentiment_n']})")
        e = self.earnings_component(f.get('earnings'))
        if e:
            total += e
            reasons.append(f"earnings surprise {e:+.2f} ({f['earnings'].get('verdict')})")
        if f.get('regulatory_prior') is not None:
            total += self.w_regulatory * f['regulatory_prior']
            reasons.append(f"regulatory prior {f['regulatory_prior']:+.2f}")
        soc = f.get('social') or {}
        if soc.get('bull_ratio') is not None:
            tilt = (soc['bull_ratio'] - 0.5) * 2
            total += self.w_social * tilt
            reasons.append(f"social tilt {tilt:+.2f}")
        score = max(-1.0, min(1.0, total))
        category = trig['category'] if trig else (b.trigger.kind if b.trigger else 'none')
        if f.get('earnings') and abs(e) >= abs(trig['score'] if trig else 0):
            category = 'earnings_' + str(f['earnings'].get('verdict', 'result'))
        confidence = min(1.0, 0.3 + 0.15 * f.get('n_sources', 1) + (0.1 if f.get('earnings') else 0))
        return Classification(score=round(score, 3), category=category, confidence=round(confidence, 2),
                              reasons=reasons)


SYSTEM_PROMPT = """You are the signal desk of a systematic equity trading bot that only takes LONG positions
held for at most one week, with a take-profit and a stop-loss attached.

You receive every piece of evidence gathered about ONE ticker over the last day: headlines and press
releases from several providers, provider-computed sentiment, reported earnings versus consensus,
regulatory events, scheduled earnings dates and social-media activity. The line marked with * is the
newest trigger event. You also receive deterministic rule features as a prior.

Judge whether a disciplined trader should buy now for a multi-day hold. Score from -1.0 (very bearish)
to 1.0 (very bullish); 0.0 means no tradeable edge. Weigh primary evidence (reported numbers, regulator
decisions, definitive deals) above commentary; treat social chatter as weak context; discount
information that is old or already widely reported. A large price move that already happened is not a
reason to buy. Duplicated headlines across providers are one event, not several."""

OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'score': {'type': 'number'},
        'category': {'type': 'string', 'description': 'snake_case catalyst label, e.g. earnings_beat'},
        'confidence': {'type': 'number'},
        'reasons': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['score', 'category', 'confidence', 'reasons'],
    'additionalProperties': False,
}


class ClaudeScorer(Scorer):
    def __init__(self, model: str = 'claude-opus-5', *, client=None, fallback: Optional[Scorer] = None,
                 effort: str = 'medium', max_items: int = 25):
        if client is None:
            import anthropic  # noqa: PLC0415  (optional dependency)
            client = anthropic.Anthropic()
        self._client = client
        self._model = model
        self._fallback = RuleScorer() if fallback is None else fallback
        self._effort = effort
        self._max_items = max_items

    def score(self, bundle: Bundle) -> Classification:
        try:
            return self._score(bundle)
        except Exception as e:  # noqa: BLE001 — any SDK/network/parsing failure
            log.warning('Claude scoring failed (%s: %s); using rule scorer', type(e).__name__, e)
            c = self._fallback.score(bundle)
            c.reasons.insert(0, 'fallback:rules')
            return c

    def prompt(self, b: Bundle) -> str:
        feats = {k: v for k, v in b.features.items() if k not in ('rule',)}
        return (f'Ticker: {b.ticker}\nAs of: {b.as_of.isoformat(timespec="minutes")}\n\n'
                f'Evidence (oldest first, * = newest trigger):\n{b.describe(self._max_items)}\n\n'
                f'Rule-based features (prior):\n{json.dumps(feats, default=str)}')

    def _score(self, bundle: Bundle) -> Classification:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=[{'type': 'text', 'text': SYSTEM_PROMPT, 'cache_control': {'type': 'ephemeral'}}],
            messages=[{'role': 'user', 'content': self.prompt(bundle)}],
            output_config={'effort': self._effort, 'format': {'type': 'json_schema', 'schema': OUTPUT_SCHEMA}},
        )
        if response.stop_reason == 'refusal':
            raise RuntimeError('model refused the request')
        text = next(b.text for b in response.content if b.type == 'text')
        data = json.loads(text)
        score = max(-1.0, min(1.0, float(data['score'])))
        return Classification(score=round(score, 3), category=str(data.get('category') or 'none'),
                              confidence=max(0.0, min(1.0, float(data.get('confidence', 0.5)))),
                              reasons=[str(r) for r in data.get('reasons', [])][:8])

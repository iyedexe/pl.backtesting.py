"""
Optional LLM classifier backed by the Anthropic SDK (`pip install anthropic`).

The model returns strict JSON (score, category, confidence, reasons). On any
API error the classifier falls back to the rule classifier so the bot keeps
running; set `fallback=None` to surface errors instead.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from ..models import Classification, NewsItem
from .base import Classifier
from .rules import RuleClassifier

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an equity news analyst for a systematic trading desk.
Given a headline (and optional summary) about a listed company, judge whether the news is a
catalyst that a disciplined trader would buy for a holding period of up to one week.

Score from -1.0 (very bearish) to 1.0 (very bullish). 0.0 means no tradeable information.
Typical bullish catalysts: earnings beat with raised guidance, FDA approval, takeover offer,
large contract win, index inclusion, analyst upgrade. Bearish: earnings miss, guidance cut,
dilutive offering, failed trial, investigation, executive departure, bankruptcy.
Be conservative: routine PR, product announcements and opinion pieces score near 0.
Never treat a price move that already happened as a reason to buy."""

OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'score': {'type': 'number', 'description': 'From -1.0 (very bearish) to 1.0 (very bullish).'},
        'category': {'type': 'string', 'description': 'Short snake_case catalyst label, e.g. earnings_beat.'},
        'confidence': {'type': 'number', 'description': 'From 0.0 to 1.0.'},
        'reasons': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['score', 'category', 'confidence', 'reasons'],
    'additionalProperties': False,
}


class ClaudeClassifier(Classifier):
    def __init__(self, model: str = 'claude-opus-5', *, client=None,
                 fallback: Optional[Classifier] = RuleClassifier(), effort: str = 'low'):
        if client is None:
            import anthropic  # noqa: PLC0415  (optional dependency)
            client = anthropic.Anthropic()
        self._client = client
        self._model = model
        self._fallback = fallback
        self._effort = effort

    def classify(self, item: NewsItem) -> Classification:
        try:
            return self._classify(item)
        except Exception as e:  # noqa: BLE001 — any SDK/network/parsing failure
            if self._fallback is None:
                raise
            log.warning('Claude classification failed (%s: %s); using fallback classifier', type(e).__name__, e)
            return self._fallback.classify(item)

    def _classify(self, item: NewsItem) -> Classification:
        user = f'Tickers: {", ".join(item.tickers) or "unknown"}\nHeadline: {item.headline}'
        if item.summary:
            user += f'\nSummary: {item.summary[:2000]}'
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': user}],
            output_config={'effort': self._effort, 'format': {'type': 'json_schema', 'schema': OUTPUT_SCHEMA}},
        )
        if response.stop_reason == 'refusal':
            raise RuntimeError('model refused the request')
        text = next(b.text for b in response.content if b.type == 'text')
        data = json.loads(text)
        score = max(-1.0, min(1.0, float(data['score'])))
        return Classification(score=round(score, 3), category=str(data.get('category') or 'none'),
                              confidence=max(0.0, min(1.0, float(data.get('confidence', 0.5)))),
                              reasons=[str(r) for r in data.get('reasons', [])])

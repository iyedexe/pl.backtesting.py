from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Classification, NewsItem


class Classifier(ABC):
    """Turns a news item into a bullish/bearish `Classification`."""

    @abstractmethod
    def classify(self, item: NewsItem) -> Classification:
        ...

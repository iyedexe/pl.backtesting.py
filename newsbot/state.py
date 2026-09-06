"""JSON persistence so the bot can be restarted without losing positions or re-trading old headlines."""
from __future__ import annotations

import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union

from .models import ClosedTrade, Position, Signal


class BotState:
    MAX_SEEN = 20_000

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self.path = Path(path) if path else None
        self.positions: Dict[str, Position] = {}
        self.pending: Dict[str, Signal] = {}
        self.closed: List[ClosedTrade] = []
        self.seen: 'OrderedDict[str, None]' = OrderedDict()
        self.last_poll: Optional[str] = None
        if self.path and self.path.exists():
            self.load()

    # -- seen news -------------------------------------------------------
    def mark_seen(self, news_id: str) -> None:
        self.seen[news_id] = None
        while len(self.seen) > self.MAX_SEEN:
            self.seen.popitem(last=False)

    def has_seen(self, news_id: str) -> bool:
        return news_id in self.seen

    # -- io --------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            'positions': [p.to_dict() for p in self.positions.values()],
            'pending': [s.to_dict() for s in self.pending.values()],
            'closed': [t.to_dict() for t in self.closed],
            'seen': list(self.seen.keys()),
            'last_poll': self.last_poll,
        }

    def load_dict(self, d: dict) -> None:
        self.positions = {p['ticker']: Position.from_dict(p) for p in d.get('positions', [])}
        self.pending = {s['id']: Signal.from_dict(s) for s in d.get('pending', [])}
        self.closed = [ClosedTrade.from_dict(t) for t in d.get('closed', [])]
        self.seen = OrderedDict((k, None) for k in d.get('seen', []))
        self.last_poll = d.get('last_poll')

    def load(self) -> None:
        assert self.path is not None
        with open(self.path, encoding='utf-8') as f:
            self.load_dict(json.load(f))

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix='.state-', suffix='.json')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, self.path)

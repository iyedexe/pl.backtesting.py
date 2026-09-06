"""
Persistent strategy state for live execution.

The executor is stateless between runs (it is meant to be invoked once per
day, e.g. from cron, shortly before the Paris close): everything it needs to
remember lives in a small JSON file, and broker positions are the source of
truth that the stored state is reconciled against on every run.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
from dataclasses import dataclass, field

from ..pricing import OptionSpec


@dataclass
class LiveState:
    ticker: str
    strategy: str = 'PMCC'
    params: dict = field(default_factory=dict)
    # last known intended position (reconciled against the broker each run)
    leaps: dict | None = None      # {'strike':, 'expiry': 'YYYY-MM-DD', 'qty':}
    short: dict | None = None
    last_run: str | None = None
    history: list = field(default_factory=list)   # action log (append-only)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str, ticker: str) -> 'LiveState':
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            if raw.get('ticker') != ticker:
                raise ValueError(f'state file {path} is for {raw.get("ticker")}, '
                                 f'not {ticker}')
            return cls(**raw)
        return cls(ticker=ticker)

    def save(self, path: str) -> None:
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(dataclasses.asdict(self), f, indent=2, default=str)
        os.replace(tmp, path)

    # ------------------------------------------------------------- helpers
    def record(self, action: str, **details) -> None:
        self.history.append({'ts': dt.datetime.now(dt.timezone.utc).isoformat(),
                             'action': action, **details})

    @staticmethod
    def spec_to_dict(spec: OptionSpec, qty: int) -> dict:
        return {'strike': spec.strike, 'expiry': spec.expiry.isoformat(),
                'qty': qty}

    @staticmethod
    def dict_to_spec(d: dict) -> tuple[OptionSpec, int]:
        return (OptionSpec(float(d['strike']),
                           dt.date.fromisoformat(d['expiry'])), int(d['qty']))

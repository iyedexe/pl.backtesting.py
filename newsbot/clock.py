"""Time abstraction so the same bot code runs live, in replay, and in tests."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .models import to_utc

NY = ZoneInfo('America/New_York')
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


class Clock:
    def now(self) -> datetime:
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def sleep(self, seconds: float) -> None:
        import time as _time  # noqa: PLC0415
        _time.sleep(seconds)


class SimClock(Clock):
    """Manually advanced clock for replay and tests."""

    def __init__(self, start):
        self._now = to_utc(start)

    def now(self) -> datetime:
        return self._now

    def set(self, when) -> None:
        self._now = to_utc(when)

    def advance(self, **kwargs) -> None:
        self._now += timedelta(**kwargs)

    def sleep(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


def is_regular_session(when: datetime) -> bool:
    """US equities regular session (Mon-Fri 09:30-16:00 New York). Exchange holidays are NOT handled;
    use a broker-side clock (e.g. Alpaca `/v2/clock`) when that matters. For other markets see
    `newsbot.universe.Universe.is_open`."""
    local = to_utc(when).astimezone(NY)
    return local.weekday() < 5 and REGULAR_OPEN <= local.time() < REGULAR_CLOSE


def next_session_open(when: datetime) -> datetime:
    local = to_utc(when).astimezone(NY)
    candidate = local.replace(hour=9, minute=30, second=0, microsecond=0)
    if local.time() >= REGULAR_OPEN:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.astimezone(timezone.utc)

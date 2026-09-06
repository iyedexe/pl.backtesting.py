"""Small HTTP helpers shared by the API-backed news providers."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

DEFAULT_UA = 'newsbot/1.0 (+https://github.com/iyedexe/pl.backtesting.py)'


def env_key(*names: str, explicit: Optional[str] = None) -> str:
    """Return the explicit key, else the first environment variable among `names` that is set, else ''."""
    if explicit:
        return explicit
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ''


class RateLimiter:
    """Allow at most one call per `min_interval` seconds. `ready()` is non-blocking."""

    def __init__(self, min_interval: float):
        self.min_interval = float(min_interval)
        self._last = 0.0

    def ready(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if now - self._last >= self.min_interval:
            self._last = now
            return True
        return False


def get_json(url: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
             timeout: float = 15.0) -> Optional[Any]:
    """GET a JSON document. Returns None (and logs) on any network/HTTP/decoding error."""
    import requests  # noqa: PLC0415
    h = {'User-Agent': DEFAULT_UA, 'Accept': 'application/json'}
    if headers:
        h.update(headers)
    try:
        r = requests.get(url, params=params, headers=h, timeout=timeout)
        if r.status_code >= 400:
            log.warning('GET %s -> %s: %s', url.split('?')[0], r.status_code, r.text[:200])
            return None
        return r.json()
    except Exception as e:  # noqa: BLE001
        log.warning('GET %s failed: %s', url.split('?')[0], e)
        return None


def get_text(url: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
             timeout: float = 15.0) -> Optional[str]:
    import requests  # noqa: PLC0415
    h = {'User-Agent': DEFAULT_UA}
    if headers:
        h.update(headers)
    try:
        r = requests.get(url, params=params, headers=h, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:  # noqa: BLE001
        log.warning('GET %s failed: %s', url.split('?')[0], e)
        return None

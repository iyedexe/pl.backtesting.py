from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from ..http import RateLimiter, env_key, get_json
from ..models import NewsItem, to_utc
from ..sources import NewsSource

log = logging.getLogger(__name__)


class APISource(NewsSource):
    """Common plumbing: key lookup, rate limiting, `since` defaulting, JSON GET."""
    name = 'api'
    ENV_KEYS: tuple = ()
    DEFAULT_INTERVAL = 60.0
    LOOKBACK = timedelta(hours=24)

    def __init__(self, tickers: Optional[Iterable[str]] = None, *, api_key: Optional[str] = None,
                 min_interval: Optional[float] = None, timeout: float = 15.0, base_url: Optional[str] = None):
        self.tickers = [t.upper() for t in tickers] if tickers else []
        self.api_key = env_key(*self.ENV_KEYS, explicit=api_key)
        if self.ENV_KEYS and not self.api_key:
            log.warning('%s: no API key (set %s); source disabled', self.name, ' or '.join(self.ENV_KEYS))
        self.timeout = timeout
        self._limiter = RateLimiter(self.DEFAULT_INTERVAL if min_interval is None else min_interval)
        if base_url:
            self.BASE = base_url  # type: ignore[attr-defined]

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) or not self.ENV_KEYS

    def _get(self, url: str, **params) -> Optional[Any]:
        return get_json(url, params={k: v for k, v in params.items() if v is not None}, timeout=self.timeout,
                        headers=self._headers())

    def _headers(self) -> Dict[str, str]:
        return {}

    def _since(self, since: Optional[datetime]) -> datetime:
        return to_utc(since) if since is not None else datetime.now(timezone.utc) - self.LOOKBACK

    def fetch(self, since: Optional[datetime] = None) -> List[NewsItem]:
        if not self.enabled or not self._limiter.ready():
            return []
        try:
            items = self._fetch(self._since(since))
        except Exception as e:  # noqa: BLE001
            log.warning('%s: fetch failed: %s', self.name, e)
            return []
        cutoff = to_utc(since) if since is not None else None
        items = [i for i in items if cutoff is None or i.published > cutoff]
        items.sort(key=lambda i: i.published)
        return items

    def _fetch(self, since: datetime) -> List[NewsItem]:
        raise NotImplementedError


def _f(value: Any) -> Optional[float]:
    try:
        return None if value in (None, '', 'None') else float(value)
    except (TypeError, ValueError):
        return None


def earnings_result_item(symbol: str, when: datetime, *, source: str, eps_actual, eps_estimate,
                         revenue_actual=None, revenue_estimate=None,
                         extra: Optional[Dict[str, Any]] = None) -> NewsItem:
    """Build a KIND_EARNINGS_RESULT item with a computed surprise so the scorer has numbers, not prose."""
    from ..models import KIND_EARNINGS_RESULT  # noqa: PLC0415
    ea, ee = _f(eps_actual), _f(eps_estimate)
    ra, re_ = _f(revenue_actual), _f(revenue_estimate)
    surprise = (ea - ee) / abs(ee) if ea is not None and ee else None
    rev_surprise = (ra - re_) / abs(re_) if ra is not None and re_ else None
    if surprise is None:
        verdict = 'reported'
    else:
        verdict = 'beat' if surprise > 0.01 else 'miss' if surprise < -0.01 else 'inline'
    headline = (f'{symbol} reported EPS {ea if ea is not None else "n/a"} '
                f'vs {ee if ee is not None else "n/a"} estimate')
    if surprise is not None:
        headline += f' ({verdict}, {surprise * 100:+.1f}%)'
    if rev_surprise is not None:
        headline += f'; revenue {rev_surprise * 100:+.1f}% vs estimate'
    meta = {'eps_actual': ea, 'eps_estimate': ee,
            'eps_surprise_pct': None if surprise is None else round(surprise * 100, 2),
            'revenue_actual': ra, 'revenue_estimate': re_,
            'revenue_surprise_pct': None if rev_surprise is None else round(rev_surprise * 100, 2), 'verdict': verdict}
    meta.update(extra or {})
    return NewsItem(id=f'{source}:earnings:{symbol}:{when.date()}', headline=headline, published=when,
                    tickers=[symbol], source=source, kind=KIND_EARNINGS_RESULT, meta=meta)


def earnings_upcoming_item(symbol: str, report_date: datetime, *, source: str, hour: str = '',
                           eps_estimate=None, extra: Optional[Dict[str, Any]] = None) -> NewsItem:
    from ..models import KIND_EARNINGS_UPCOMING  # noqa: PLC0415
    meta = {'report_date': report_date.date().isoformat(), 'hour': hour, 'eps_estimate': _f(eps_estimate)}
    meta.update(extra or {})
    return NewsItem(id=f'{source}:upcoming:{symbol}:{report_date.date()}',
                    headline=f'{symbol} scheduled to report earnings on {report_date.date()} {hour}'.strip(),
                    published=min(report_date, datetime.now(timezone.utc)), tickers=[symbol],
                    source=source, kind=KIND_EARNINGS_UPCOMING, meta=meta)

"""FDA press announcements (RSS, no key) and openFDA drug approvals (no key, 240 req/min).
FDA text names companies, not tickers, so both sources rely on `aliases` in the ticker extractor."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import KIND_REGULATORY, NewsItem, to_utc
from ..sources import RSSNewsSource, TickerExtractor
from ._base import APISource


class FDAPressRSS(RSSNewsSource):
    URL = 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds-fda/press-releases/rss.xml'

    def __init__(self, extractor: TickerExtractor, url: Optional[str] = None, **kw):
        kw.setdefault('min_interval', 600.0)
        super().__init__(url or self.URL, name='fda', extractor=extractor, kind=KIND_REGULATORY, **kw)


class OpenFDAApprovals(APISource):
    """Drugs@FDA submissions approved recently (`submission_status: AP`)."""
    name = 'openfda'
    ENV_KEYS = ()
    BASE = 'https://api.fda.gov/drug/drugsfda.json'
    DEFAULT_INTERVAL = 1800.0

    def __init__(self, extractor: TickerExtractor, **kw):
        super().__init__(None, **kw)
        self.extractor = extractor

    def _fetch(self, since: datetime) -> List[NewsItem]:
        frm = since.strftime('%Y%m%d')
        to = datetime.now(since.tzinfo).strftime('%Y%m%d')
        data = self._get(self.BASE, search=f'submissions.submission_status_date:[{frm}+TO+{to}]+AND+'
                                           f'submissions.submission_status:AP', limit=100) or {}
        return self.parse(data, self.extractor)

    @staticmethod
    def parse(data: Dict[str, Any], extractor: TickerExtractor) -> List[NewsItem]:
        items: List[NewsItem] = []
        for r in data.get('results', []):
            sponsor = r.get('sponsor_name') or ''
            tickers = extractor.extract(sponsor)
            if not tickers:
                continue
            products = r.get('products') or []
            brand = products[0].get('brand_name') if products else r.get('application_number', '')
            for s in r.get('submissions') or []:
                if s.get('submission_status') != 'AP' or not s.get('submission_status_date'):
                    continue
                try:
                    when = to_utc(datetime.strptime(s['submission_status_date'], '%Y%m%d'))
                except ValueError:
                    continue
                stype = s.get('submission_type', '')
                cls = s.get('submission_class_code_description') or ''
                items.append(NewsItem(id=f'openfda:{r.get("application_number")}:{s.get("submission_number")}',
                                      headline=f'FDA approved {brand} ({sponsor}): {stype} {cls}'.strip(),
                                      published=when, tickers=tickers, source='openfda', kind=KIND_REGULATORY,
                                      meta={'submission_type': stype, 'class': cls, 'sponsor': sponsor}))
        return items

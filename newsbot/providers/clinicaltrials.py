"""ClinicalTrials.gov API v2 (no key): studies by sponsor whose status changed recently.
Emits a KIND_REGULATORY item on terminal status changes (completed / terminated / suspended / withdrawn)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping

from ..models import KIND_REGULATORY, NewsItem, to_utc
from ._base import APISource

INTERESTING = {'COMPLETED': 0.2, 'TERMINATED': -0.6, 'SUSPENDED': -0.5, 'WITHDRAWN': -0.4, 'ACTIVE_NOT_RECRUITING': 0.0}


class ClinicalTrialsSource(APISource):
    name = 'clinicaltrials'
    ENV_KEYS = ()
    BASE = 'https://clinicaltrials.gov/api/v2/studies'
    DEFAULT_INTERVAL = 1800.0

    def __init__(self, sponsors: Mapping[str, Iterable[str]], **kw):
        """sponsors: ticker -> sponsor names as they appear on ClinicalTrials.gov."""
        super().__init__(list(sponsors), **kw)
        self.sponsors = {t.upper(): ([n] if isinstance(n, str) else list(n)) for t, n in sponsors.items()}

    def _fetch(self, since: datetime) -> List[NewsItem]:
        out: List[NewsItem] = []
        for ticker, names in self.sponsors.items():
            for name in names:
                params = {'query.spons': name, 'pageSize': 50,
                          'filter.advanced': f'AREA[LastUpdatePostDate]RANGE[{since.date()},MAX]',
                          'fields': 'NCTId,BriefTitle,OverallStatus,Phase,LastUpdatePostDate,WhyStopped'}
                data = self._get(self.BASE, **params) or {}
                out += self.parse(data, ticker)
        return out

    @staticmethod
    def parse(data: Dict[str, Any], ticker: str) -> List[NewsItem]:
        items: List[NewsItem] = []
        for st in data.get('studies', []):
            proto = st.get('protocolSection', {})
            ident = proto.get('identificationModule', {})
            status = proto.get('statusModule', {})
            overall = status.get('overallStatus', '')
            if overall not in INTERESTING:
                continue
            upd = (status.get('lastUpdatePostDateStruct') or {}).get('date')
            if not upd:
                continue
            phases = (proto.get('designModule') or {}).get('phases') or []
            why = status.get('whyStopped') or ''
            title = ident.get('briefTitle', '')
            items.append(NewsItem(id=f'ct:{ident.get("nctId")}:{overall}:{upd}',
                                  headline=f'Clinical trial {overall.lower().replace("_", " ")}: {title}'
                                           + (f' ({", ".join(phases)})' if phases else ''),
                                  published=to_utc(upd), tickers=[ticker], summary=why, source='clinicaltrials',
                                  url=f'https://clinicaltrials.gov/study/{ident.get("nctId")}', kind=KIND_REGULATORY,
                                  meta={'status': overall, 'phases': phases, 'prior': INTERESTING[overall]}))
        return items

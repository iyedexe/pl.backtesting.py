"""
API-backed evidence providers. Every provider is a `NewsSource`: `fetch(since)` returns
`NewsItem`s (with `kind` and `meta`) and never raises on network trouble.

Keys are read from the environment unless passed explicitly:

    FINNHUB_API_KEY, ALPHAVANTAGE_API_KEY, POLYGON_API_KEY, MARKETAUX_API_KEY,
    NEWSAPI_API_KEY, FMP_API_KEY

Free-tier quotas are enforced with `min_interval` (seconds between calls) per provider.
"""
from .alphavantage import AlphaVantageNews
from .clinicaltrials import ClinicalTrialsSource
from .fda import FDAPressRSS, OpenFDAApprovals
from .finnhub import FinnhubEarningsCalendar, FinnhubNews
from .fmp import FMPEarningsCalendar, FMPNews
from .marketaux import MarketauxNews
from .nasdaq import NasdaqEarningsCalendar
from .newsapi import NewsAPINews
from .polygon import PolygonNews
from .social import RedditMentions, StockTwitsStream

__all__ = [
    'AlphaVantageNews', 'ClinicalTrialsSource', 'FDAPressRSS', 'OpenFDAApprovals', 'FinnhubEarningsCalendar',
    'FinnhubNews', 'FMPEarningsCalendar', 'FMPNews', 'MarketauxNews', 'NasdaqEarningsCalendar', 'NewsAPINews',
    'PolygonNews', 'RedditMentions', 'StockTwitsStream',
]

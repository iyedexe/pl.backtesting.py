"""
What the bot is allowed to trade: an explicit ticker list, or whole markets.

    universe: [AAPL, MSFT]                 # explicit list
    universe: {markets: [us]}              # every US-listed ticker the news mentions
    universe: {markets: [eu, uk], exclude: [XYZ.L]}
    universe: us                           # shorthand

Tickers use Yahoo Finance style suffixes (AIR.PA, BMW.DE, BP.L, NESN.SW ...); US tickers
have no suffix. Each market also carries its regular trading session for `market_hours_only`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Union
from zoneinfo import ZoneInfo

from .models import to_utc


@dataclass(frozen=True)
class Market:
    name: str
    suffixes: FrozenSet[str]          # '' = no suffix (US)
    tz: str
    open: time
    close: time
    countries: FrozenSet[str] = frozenset()   # ISO codes for providers with a country filter

    def is_open(self, when: datetime) -> bool:
        local = to_utc(when).astimezone(ZoneInfo(self.tz))
        return local.weekday() < 5 and self.open <= local.time() < self.close


EU_SUFFIXES = frozenset({'.PA', '.AS', '.BR', '.LS', '.MI', '.DE', '.F', '.MC', '.SW', '.ST', '.CO', '.HE', '.OL',
                         '.VI', '.IR', '.WA', '.PR', '.AT'})
MARKETS: Dict[str, Market] = {
    'us': Market('us', frozenset({''}), 'America/New_York', time(9, 30), time(16, 0), frozenset({'us'})),
    'eu': Market('eu', EU_SUFFIXES, 'Europe/Paris', time(9, 0), time(17, 30),
                 frozenset({'fr', 'nl', 'be', 'pt', 'it', 'de', 'es', 'ch', 'se', 'dk', 'fi', 'no', 'at', 'ie', 'pl'})),
    'uk': Market('uk', frozenset({'.L'}), 'Europe/London', time(8, 0), time(16, 30), frozenset({'gb'})),
    'ca': Market('ca', frozenset({'.TO', '.V'}), 'America/Toronto', time(9, 30), time(16, 0), frozenset({'ca'})),
}
MARKETS['europe'] = MARKETS['eu']
MARKETS['all'] = Market('all', frozenset(s for m in ('us', 'eu', 'uk', 'ca') for s in MARKETS[m].suffixes),
                        'UTC', time(0, 0), time(23, 59),
                        frozenset(c for m in ('us', 'eu', 'uk', 'ca') for c in MARKETS[m].countries))

#: "(EXCHANGE: TICKER)" tags in press releases -> ticker suffix.
EXCHANGE_SUFFIX: Dict[str, str] = {
    'NASDAQ': '', 'NYSE': '', 'NYSE AMERICAN': '', 'NYSE MKT': '', 'NYSEMKT': '', 'AMEX': '', 'NYSE ARCA': '',
    'OTC': '', 'OTCQB': '', 'OTCQX': '', 'OTC PINK': '', 'CBOE': '',
    'TSX': '.TO', 'TSXV': '.V', 'TSX-V': '.V', 'TSX VENTURE': '.V', 'CSE': '.CN',
    'LSE': '.L', 'LON': '.L', 'AIM': '.L', 'LONDON': '.L',
    'EURONEXT': '.PA', 'EURONEXT PARIS': '.PA', 'PARIS': '.PA', 'EPA': '.PA',
    'EURONEXT AMSTERDAM': '.AS', 'AMSTERDAM': '.AS', 'AMS': '.AS',
    'EURONEXT BRUSSELS': '.BR', 'BRUSSELS': '.BR', 'EBR': '.BR',
    'EURONEXT LISBON': '.LS', 'LISBON': '.LS', 'EURONEXT MILAN': '.MI', 'BORSA ITALIANA': '.MI', 'MILAN': '.MI',
    'BIT': '.MI', 'XETRA': '.DE', 'FSE': '.DE', 'FRANKFURT': '.DE', 'FWB': '.DE', 'ETR': '.DE', 'FRA': '.F',
    'SIX': '.SW', 'SWX': '.SW', 'SIX SWISS EXCHANGE': '.SW', 'SWISS': '.SW',
    'BME': '.MC', 'MADRID': '.MC', 'NASDAQ STOCKHOLM': '.ST', 'STOCKHOLM': '.ST', 'OMX': '.ST', 'STO': '.ST',
    'NASDAQ COPENHAGEN': '.CO', 'COPENHAGEN': '.CO', 'CPH': '.CO', 'NASDAQ HELSINKI': '.HE', 'HELSINKI': '.HE',
    'OSLO': '.OL', 'OSLO BORS': '.OL', 'OSE': '.OL', 'EURONEXT OSLO': '.OL', 'VIENNA': '.VI', 'WIENER BORSE': '.VI',
    'EURONEXT DUBLIN': '.IR', 'DUBLIN': '.IR', 'WSE': '.WA', 'WARSAW': '.WA', 'ATHENS': '.AT',
}


def ticker_suffix(ticker: str) -> str:
    t = ticker.upper()
    if '.' in t:
        return t[t.rindex('.'):]
    return ''


@dataclass
class Universe:
    tickers: FrozenSet[str] = frozenset()        # explicit list mode
    markets: List[Market] = field(default_factory=list)
    exclude: FrozenSet[str] = frozenset()

    @classmethod
    def from_config(cls, value: Any) -> 'Universe':
        if value is None:
            return cls()
        if isinstance(value, str):
            value = {'markets': [value]}
        if isinstance(value, (list, tuple, set, frozenset)):
            return cls(tickers=frozenset(str(t).upper() for t in value))
        if isinstance(value, dict):
            names = value.get('markets') or value.get('market') or []
            if isinstance(names, str):
                names = [names]
            markets = []
            for n in names:
                if str(n).lower() not in MARKETS:
                    raise ValueError(f'unknown market {n!r}; choose from {sorted(MARKETS)}')
                markets.append(MARKETS[str(n).lower()])
            return cls(tickers=frozenset(str(t).upper() for t in value.get('tickers') or []), markets=markets,
                       exclude=frozenset(str(t).upper() for t in value.get('exclude') or []))
        raise ValueError(f'cannot interpret universe {value!r}')

    # -- membership ----------------------------------------------------
    @property
    def is_market_mode(self) -> bool:
        return bool(self.markets) and not self.tickers

    @property
    def restricts(self) -> bool:
        return bool(self.tickers or self.markets)

    def __bool__(self) -> bool:
        return self.restricts

    def __contains__(self, ticker: object) -> bool:
        if not isinstance(ticker, str):
            return False
        t = ticker.upper()
        if t in self.exclude:
            return False
        if self.tickers:
            return t in self.tickers
        if self.markets:
            return any(ticker_suffix(t) in m.suffixes for m in self.markets)
        return True

    def filter(self, tickers: Iterable[str]) -> List[str]:
        return [t for t in tickers if t in self]

    @property
    def explicit(self) -> List[str]:
        """Ticker list for per-ticker sources ([] in market mode)."""
        return sorted(self.tickers)

    @property
    def countries(self) -> List[str]:
        return sorted({c for m in self.markets for c in m.countries})

    # -- sessions --------------------------------------------------------
    def market_for(self, ticker: str) -> Optional[Market]:
        sfx = ticker_suffix(ticker)
        for m in self.markets or MARKETS.values():
            if sfx in m.suffixes and m.name != 'all':
                return m
        return None

    def is_open(self, when: datetime, ticker: Optional[str] = None) -> bool:
        """Regular session check. With a ticker: that ticker's market. Without: any configured market
        (US by default). Exchange holidays are not modelled."""
        if ticker:
            m = self.market_for(ticker)
            if m is not None:
                return m.is_open(when)
        markets = self.markets or [MARKETS['us']]
        return any(m.is_open(when) for m in markets)

    def describe(self) -> str:
        if self.tickers:
            return f'{len(self.tickers)} tickers'
        if self.markets:
            return 'markets ' + '+'.join(m.name for m in self.markets)
        return 'any ticker'


def universe_arg(value: Union[Universe, Iterable[str], None]) -> Universe:
    return value if isinstance(value, Universe) else Universe.from_config(list(value) if value else None)

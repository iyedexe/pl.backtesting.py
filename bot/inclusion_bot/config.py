"""Bot configuration: per-index rulebook parameters and runtime settings.

Everything here can be overridden by a JSON file passed as ``--config``;
keys mirror the dataclass fields. Telegram credentials come from the
environment (``TELEGRAM_BOT_TOKEN``, ``TELEGRAM_CHAT_ID``) so they never
live in a file that might get committed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, replace


@dataclass(frozen=True)
class IndexConfig:
    key: str                  # short id used in state/messages, e.g. 'FTSE100'
    name: str                 # human name
    region: str               # 'US' / 'EUR'
    currency: str             # currency of quoted prices ('GBp', 'EUR', 'USD')
    members_page: str         # Wikipedia page listing current constituents
    candidates_page: str      # Wikipedia page for the candidate pool (or '' = derived)
    yahoo_suffix: str         # appended to bare tickers for Yahoo ('', '.L', '.DE')
    add_rank: int             # inclusion band: non-member at this rank or better qualifies
    fail_rank: int            # abandon a candidate whose rank decays beyond this
    buffer: int = 2           # pre-cutoff safety margin: require rank <= add_rank - buffer
    pre_cutoff_days: int = 10          # start hunting this many business days before cutoff
    expected_effect: float = .02       # assumed addition run-up, announcement -> effective
    use_float_cap: bool = False        # rank by free-float cap (DAX) instead of full cap
    exclude_financials: bool = False   # Nasdaq-100 style sector exclusion
    exchange_prefixes: tuple = ()      # restrict candidates to these Yahoo exchange codes
    min_cap: float = 0.0               # eligibility floor (S&P 500 watchlist)
    require_profitability: bool = False  # S&P 500: positive last quarter + trailing 4Q
    enabled: bool = True


# --- The four rulebooks -----------------------------------------------------
# Sources (verified Aug 2026, see bot/README.md): FTSE UK Index Series ground
# rules; DAX (ISS STOXX) equity index guide; Nasdaq-100 methodology; S&P U.S.
# indices methodology. Thresholds drift over time - review them periodically.

FTSE100 = IndexConfig(
    key='FTSE100', name='FTSE 100', region='EUR', currency='GBp',
    members_page='FTSE_100_Index', candidates_page='FTSE_250_Index',
    yahoo_suffix='.L',
    add_rank=90,      # auto-inclusion at rank <= 90 by full market cap
    fail_rank=100,
    expected_effect=.020,
)

DAX40 = IndexConfig(
    key='DAX40', name='DAX 40', region='EUR', currency='EUR',
    members_page='DAX', candidates_page='MDAX',
    yahoo_suffix='.DE',
    add_rank=40,      # regular entry (Mar/Sep); fast entry <= 33 checked quarterly
    fail_rank=45,
    expected_effect=.015,
    use_float_cap=True,   # DAX ranks by free-float market cap
)
DAX_FAST_ENTRY_RANK = 33  # quarterly fast-entry band (Jun/Dec reviews)

NDX100 = IndexConfig(
    key='NDX100', name='Nasdaq-100', region='US', currency='USD',
    members_page='Nasdaq-100', candidates_page='',   # candidates derived, see universe.py
    yahoo_suffix='',
    add_rank=75,      # top 75 by cap are added at reconstitution
    fail_rank=90,
    expected_effect=.010,
    exclude_financials=True,
    exchange_prefixes=('NMS', 'NGM', 'NCM'),  # Nasdaq-listed only
)

SPX500 = IndexConfig(
    key='SPX500', name='S&P 500', region='US', currency='USD',
    members_page='List_of_S%26P_500_companies',
    candidates_page='List_of_S%26P_400_companies',
    yahoo_suffix='',
    add_rank=0, fail_rank=0,           # not rank-based: committee + eligibility gates
    expected_effect=.005,
    min_cap=22.7e9,                    # unadjusted company market cap, July 2025 update
    require_profitability=True,
)

INDICES = (FTSE100, DAX40, NDX100, SPX500)


@dataclass
class BotConfig:
    notional: float = 10_000           # per-signal size used for theoretical P&L
    max_open_per_index: int = 1        # "one stock at a time", per index
    max_hold_sessions: int = 21        # ~1 calendar month backstop, as in the example
    state_path: str = 'bot_state.json'
    cache_dir: str = '.cache'
    telegram_token: str = field(default_factory=lambda: os.environ.get('TELEGRAM_BOT_TOKEN', ''))
    telegram_chat_id: str = field(default_factory=lambda: os.environ.get('TELEGRAM_CHAT_ID', ''))
    dry_run: bool = False
    indices: tuple = INDICES

    @classmethod
    def load(cls, path: str | None = None) -> 'BotConfig':
        cfg = cls()
        if not path:
            return cfg
        with open(path) as f:
            raw = json.load(f)
        per_index = raw.pop('indices', {})
        known = {f.name for f in fields(cls)}
        for k, v in raw.items():
            if k not in known:
                raise KeyError(f'Unknown config key: {k!r}')
            setattr(cfg, k, v)
        if per_index:
            updated = []
            for idx in cfg.indices:
                overrides = per_index.get(idx.key, {})
                bad = set(overrides) - {f.name for f in fields(IndexConfig)}
                if bad:
                    raise KeyError(f'Unknown index config key(s) for {idx.key}: {bad}')
                updated.append(replace(idx, **overrides) if overrides else idx)
            cfg.indices = tuple(updated)
        return cfg

"""CLI entry point.

    python -m inclusion_bot scan [--dry-run] [--config cfg.json] [--today YYYY-MM-DD]
    python -m inclusion_bot selftest        # offline demo on synthetic data
    python -m inclusion_bot test-telegram   # send a hello message

Run `scan` once per day after the US close (e.g. 22:00 UTC) from cron,
systemd, or a scheduled GitHub Action - see bot/README.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback

import pandas as pd

from . import notify, screener, universe
from . import state as st
from .config import BotConfig


def build_market(idx, cfg) -> tuple[pd.DataFrame, set]:
    """Assemble the ranking universe for one index from live data."""
    members, candidates = universe.load_index_universe(
        idx, cfg.indices, cfg.cache_dir)
    member_syms = {m['symbol'] for m in members}
    names = {r['symbol']: r['name'] for r in members + candidates}
    symbols = sorted(names)

    md = universe.fetch_market_data(symbols, cfg.cache_dir)
    universe.assert_coverage(md, symbols)
    md = md.assign(name=[names.get(s, s) for s in md.index])

    if idx.exchange_prefixes:      # e.g. Nasdaq-100: candidates must be Nasdaq-listed
        keep = md.index.isin(member_syms) | md['exchange'].isin(idx.exchange_prefixes)
        md = md[keep]

    if idx.use_float_cap:          # DAX ranks by free-float market cap
        float_caps = universe.fetch_float_caps(list(md.index), cfg.cache_dir)
        md = md.assign(rank_cap=[float_caps.get(s) or c
                                 for s, c in md['cap'].items()])
    else:
        md = md.assign(rank_cap=md['cap'])
    return md, member_syms


def cmd_scan(args) -> int:
    cfg = BotConfig.load(args.config)
    cfg.dry_run = cfg.dry_run or args.dry_run
    today = (dt.date.fromisoformat(args.today) if args.today
             else dt.date.today())
    state = st.load(cfg.state_path)

    signals, failures = [], []
    for idx in cfg.indices:
        if not idx.enabled:
            continue
        try:
            market, members = build_market(idx, cfg)
            profitability = (universe.check_profitability
                             if idx.require_profitability else (lambda s: True))
            signals += screener.scan_index(idx, cfg, market, members, today,
                                           state, profitability)
        except Exception as exc:
            failures.append(f'{idx.key}: {exc}')
            traceback.print_exc()

    for sig in signals:
        notify.send(notify.format_signal(sig), cfg.telegram_token,
                    cfg.telegram_chat_id, cfg.dry_run)
    if failures:
        notify.send('⚠️ inclusion-bot scan errors:\n' + '\n'.join(failures),
                    cfg.telegram_token, cfg.telegram_chat_id, cfg.dry_run)

    st.save(state, cfg.state_path)
    print(f'{today}: {len(signals)} signal(s), {len(failures)} index failure(s), '
          f'{len(state["open"])} open position(s)')
    return 1 if failures and not signals else 0


def cmd_selftest(args) -> int:
    """Offline demonstration: a synthetic FTSE-like universe around the next
    review produces one full BUY → INFO → SELL cycle, printed as messages."""
    import numpy as np
    from .calendars import ftse100_reviews, prev_bday
    cfg = BotConfig()
    cfg.dry_run = True
    idx = next(i for i in cfg.indices if i.key == 'FTSE100')
    state = st.load('/nonexistent-selftest')

    rng = np.random.default_rng(7)
    n = 130
    caps = np.sort(rng.lognormal(23, 1, n))[::-1]
    market = pd.DataFrame({
        'price': rng.uniform(100, 2000, n).round(1),
        'cap': caps, 'rank_cap': caps,
        'name': [f'Selftest Plc {i}' for i in range(n)],
        'currency': 'GBp', 'exchange': 'LSE',
    }, index=[f'TST{i:03d}.L' for i in range(n)])
    members = set(market.index[:100]) - {market.index[85]}  # rank-86 non-member
    members |= {market.index[112]}

    review = next(r for r in ftse100_reviews((dt.date.today().year,
                                              dt.date.today().year + 1))
                  if r.cutoff > dt.date.today())
    for label, day in [('pre-cutoff scan', prev_bday(review.cutoff, 3)),
                       ('cutoff-day scan', review.cutoff),
                       ('post-effective scan', review.exit_date)]:
        print(f'===== {label} ({day}) =====')
        for sig in screener.scan_index(idx, cfg, market, members, day, state):
            notify.send(notify.format_signal(sig), '', '', dry_run=True)
    print('selftest OK')
    return 0


def cmd_test_telegram(args) -> int:
    cfg = BotConfig.load(args.config)
    ok = notify.send('✅ inclusion-bot is connected.', cfg.telegram_token,
                     cfg.telegram_chat_id, args.dry_run)
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='inclusion_bot', description=__doc__)
    p.add_argument('command', nargs='?', default='scan',
                   choices=['scan', 'selftest', 'test-telegram'])
    p.add_argument('--config', help='JSON config overriding defaults')
    p.add_argument('--dry-run', action='store_true',
                   help='print messages instead of sending to Telegram')
    p.add_argument('--today', help='override the scan date (YYYY-MM-DD)')
    args = p.parse_args(argv)
    return {'scan': cmd_scan, 'selftest': cmd_selftest,
            'test-telegram': cmd_test_telegram}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())

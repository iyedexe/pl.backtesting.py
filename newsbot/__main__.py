"""
Command line:
  python -m newsbot classify "Acme beats estimates, raises full-year guidance"
  python -m newsbot backtest --prices GOOG=backtesting/test/GOOG.csv --news newsbot/data/sample_news.json
  python -m newsbot replay   --prices GOOG=backtesting/test/GOOG.csv --news newsbot/data/sample_news.json
  python -m newsbot run      --config config.yaml [--once]
  python -m newsbot status   --state newsbot_state.json
  python -m newsbot sources  --config config.yaml            # fetch each configured source once
  python -m newsbot score    --config config.yaml --news evidence.json --ticker AAPL   # dry-run the scorer
  python -m newsbot backfill --source alpaca --tickers AAPL,MSFT --start 2022-01-01 --out news.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict

from .bot import positions_table
from .classifiers import RuleClassifier
from .config import build_bot, load_config, merge_config
from .models import NewsItem, to_utc, utcnow
from .prices import load_ohlc_csv
from .state import BotState


def _parse_prices(pairs) -> Dict[str, str]:
    out = {}
    for p in pairs or []:
        if '=' in p:
            t, path = p.split('=', 1)
        else:
            path = p
            t = Path(p).stem
        out[t.upper()] = path
    if not out:
        raise SystemExit('need at least one --prices TICKER=path.csv')
    return out


def _strategy_overrides(args) -> dict:
    keys = ('min_score', 'target_pct', 'stop_pct', 'hold_bars', 'size')
    return {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}


def cmd_classify(args):
    item = NewsItem(id='cli', headline=args.text, published=utcnow(), tickers=args.ticker or [])
    c = RuleClassifier().classify(item)
    print(json.dumps({'score': c.score, 'category': c.category, 'confidence': c.confidence, 'reasons': c.reasons},
                     indent=1))


def cmd_backtest(args):
    from .strategy import run_news_backtest  # noqa: PLC0415
    prices = {t: load_ohlc_csv(p) for t, p in _parse_prices(args.prices).items()}
    res = run_news_backtest(prices, _load_news(args.news), cash=args.cash,
                            commission=args.commission, **_strategy_overrides(args))
    for ticker, st in res['stats'].items():
        print(f'\n=== {ticker} ===')
        print(st.drop(['_strategy', '_equity_curve', '_trades'], errors='ignore').to_string())
    if not res['trades'].empty:
        cols = ['Ticker', 'EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'ReturnPct', 'PnL', 'Tag']
        print('\n=== trades ===')
        print(res['trades'][[c for c in cols if c in res['trades']]].to_string(index=False))
    print('\n=== summary ===')
    print(json.dumps(res['summary'], indent=1))
    if args.plot:
        for ticker, bt in res['backtests'].items():
            bt.plot(filename=f'newsbot_backtest_{ticker}.html', open_browser=False)
            print(f'plot written to newsbot_backtest_{ticker}.html')


def _load_news(path):
    from .sources import FileNewsSource  # noqa: PLC0415
    return FileNewsSource(path).items


def cmd_replay(args):
    from .replay import run_replay  # noqa: PLC0415
    cfg = load_config(args.config) if args.config else merge_config(None)
    for k in ('min_score', 'target_pct', 'stop_pct'):
        if getattr(args, k, None) is not None:
            cfg['signals'][k] = getattr(args, k)
    bot = run_replay(_parse_prices(args.prices), args.news, cfg, cash=args.cash, state_path=args.state)
    print('\n=== closed trades ===')
    for t in bot.state.closed:
        print(f'{t.ticker:<6} {t.entry_time:%Y-%m-%d} -> {t.exit_time:%Y-%m-%d}  {t.reason:<6} '
              f'{t.entry_price:9.2f} -> {t.exit_price:9.2f}  {t.return_pct:+6.2f}%  {t.headline[:60]}')
    if bot.state.positions:
        print('\n=== still open ===')
        print(positions_table(bot.state.positions.values(), bot.feed))
    print('\n=== summary ===')
    print(json.dumps(bot.summary(), indent=1))


def cmd_run(args):
    cfg = load_config(args.config)
    if args.state:
        cfg['bot']['state_file'] = args.state
    bot = build_bot(cfg)
    if args.once:
        bot.tick()
        print(json.dumps(bot.summary(), indent=1))
    else:
        bot.run(max_ticks=args.max_ticks)


def cmd_sources(args):
    from .config import _AlpacaHolder, build_extractor, build_source  # noqa: PLC0415
    from .models import to_utc  # noqa: PLC0415
    cfg = load_config(args.config)
    since = utcnow() - timedelta(hours=args.since_hours)
    alpaca = _AlpacaHolder(cfg)
    extractor = build_extractor(cfg)
    total = 0
    for spec in cfg.get('sources', []):
        name = spec.get('type')
        try:
            src = build_source(spec, cfg, alpaca, extractor)
            items = src.fetch(since)
        except Exception as e:  # noqa: BLE001
            print(f'{name:<18} ERROR {type(e).__name__}: {e}')
            continue
        total += len(items)
        print(f'{name:<18} {len(items):4d} item(s)')
        for it in items[-args.show:]:
            print(f'    {to_utc(it.published):%m-%d %H:%M} [{it.kind}] {"/".join(it.tickers)[:20]:<20} '
                  f'{it.headline[:90]}')
    print(f'\ntotal: {total}')


def cmd_score(args):
    from .aggregator import EvidenceStore  # noqa: PLC0415
    from .config import build_engine, build_scorer  # noqa: PLC0415
    cfg = load_config(args.config) if args.config else merge_config(None)
    items = _load_news(args.news)
    engine = build_engine(cfg)
    store = EvidenceStore(window=timedelta(hours=args.window_hours), classifier=engine.classifier)
    for it in items:
        store.add(it)
    scorer = build_scorer(cfg)
    tickers = [args.ticker.upper()] if args.ticker else store.tickers()
    now = utcnow() if not args.as_of else to_utc(args.as_of)
    for t in tickers:
        b = store.bundle(t, now)
        if b is None:
            print(f'{t}: no trigger evidence in window')
            continue
        c = scorer.score(b)
        print(f'\n=== {t}  score {c.score:+.2f}  [{c.category}]  confidence {c.confidence:.2f}  '
              f'items={len(b.items)} sources={b.sources}')
        for r in c.reasons:
            print(f'  - {r}')
        if args.verbose:
            print(b.describe())
        sig = engine.build(t, c, b.trigger, now)
        print('  -> SIGNAL' if sig else '  -> no signal (below min_score or filtered)')


def cmd_backfill(args):
    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    start, end = to_utc(args.start), (to_utc(args.end) if args.end else utcnow())
    items = []
    if args.source == 'alpaca':
        from .alpaca import AlpacaClient, AlpacaNewsSource  # noqa: PLC0415
        src = AlpacaNewsSource(AlpacaClient(), symbols=tickers)
        for n, it in enumerate(src.backfill(start, end), 1):
            items.append(it)
            if n % 500 == 0:
                print(f'  {n} items, up to {it.published:%Y-%m-%d}', file=sys.stderr)
    elif args.source == 'finnhub':
        from .providers import FinnhubNews  # noqa: PLC0415
        src = FinnhubNews(tickers, min_interval=0)
        day = start
        while day < end:
            chunk_end = min(day + timedelta(days=30), end)
            for t in tickers:
                data = src._get(f'{src.BASE}/company-news', symbol=t, **{'from': day.date().isoformat()},
                                to=chunk_end.date().isoformat(), token=src.api_key)
                items += FinnhubNews.parse(data or [], t)
            day = chunk_end
    else:
        raise SystemExit(f'unknown backfill source {args.source}')
    seen = set()
    out = []
    for it in sorted(items, key=lambda i: i.published):
        if it.id in seen:
            continue
        seen.add(it.id)
        out.append(it.to_dict())
    if args.append and Path(args.out).exists():
        with open(args.out, encoding='utf-8') as f:
            old = json.load(f)
        old = old['news'] if isinstance(old, dict) else old
        ids = {r.get('id') for r in old}
        out = old + [r for r in out if r['id'] not in ids]
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {len(out)} items to {args.out}')


def cmd_status(args):
    state = BotState(args.state)
    print(f'last poll: {state.last_poll}')
    print(f'\nopen positions ({len(state.positions)}):')
    print(positions_table(state.positions.values()))
    print(f'\npending signals ({len(state.pending)}):')
    for s in state.pending.values():
        print(f'  {s.ticker:<6} score={s.score:.2f} {s.category:<18} '
              f'expires {s.expires_at:%Y-%m-%d %H:%M}  {s.headline[:70]}')
    print(f'\nclosed trades ({len(state.closed)}):')
    for t in state.closed[-20:]:
        print(f'  {t.ticker:<6} {t.reason:<6} {t.return_pct:+6.2f}%  pnl={t.pnl:9.2f}  {t.headline[:60]}')
    print(f'\nrecent scoring decisions ({len(state.decisions)}):')
    for d in state.decisions[-15:]:
        flag = 'SIGNAL' if d.get('signal') else '      '
        print(f'  {d["time"][:16]} {flag} {d["ticker"]:<6} {d["score"]:+.2f} {d["category"]:<18} '
              f'{d.get("items", 0):3d} items  {d["trigger"][:60]}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='newsbot', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-v', '--verbose', action='store_true')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('classify', help='score a headline with the rule classifier')
    p.add_argument('text')
    p.add_argument('--ticker', action='append')
    p.set_defaults(fn=cmd_classify)

    def _common(p):
        p.add_argument('--prices', action='append', required=True, metavar='TICKER=path.csv')
        p.add_argument('--news', required=True, help='JSON or CSV news file')
        p.add_argument('--cash', type=float, default=100_000)
        p.add_argument('--min-score', dest='min_score', type=float)
        p.add_argument('--target-pct', dest='target_pct', type=float)
        p.add_argument('--stop-pct', dest='stop_pct', type=float)

    p = sub.add_parser('backtest', help='backtest with the backtesting.py engine')
    _common(p)
    p.add_argument('--commission', type=float, default=0.0005)
    p.add_argument('--hold-bars', dest='hold_bars', type=int)
    p.add_argument('--size', type=float)
    p.add_argument('--plot', action='store_true')
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser('replay', help='replay the live bot pipeline over CSV prices')
    _common(p)
    p.add_argument('--config')
    p.add_argument('--state', help='write replay state to this JSON file')
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser('run', help='run the bot (paper or live per config)')
    p.add_argument('--config', required=True)
    p.add_argument('--state')
    p.add_argument('--once', action='store_true', help='single tick then exit')
    p.add_argument('--max-ticks', dest='max_ticks', type=int)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser('sources', help='fetch every configured source once and show what came back')
    p.add_argument('--config', required=True)
    p.add_argument('--since-hours', dest='since_hours', type=float, default=24)
    p.add_argument('--show', type=int, default=3, help='headlines to print per source')
    p.set_defaults(fn=cmd_sources)

    p = sub.add_parser('score', help='score evidence from a news file with the configured scorer (dry run)')
    p.add_argument('--news', required=True)
    p.add_argument('--config')
    p.add_argument('--ticker')
    p.add_argument('--as-of', dest='as_of', help='ISO time to evaluate at (default: now)')
    p.add_argument('--window-hours', dest='window_hours', type=float, default=24)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser('backfill', help='download historical news into the JSON format used by backtest/replay')
    p.add_argument('--source', choices=['alpaca', 'finnhub'], default='alpaca')
    p.add_argument('--tickers', required=True, help='comma separated')
    p.add_argument('--start', required=True)
    p.add_argument('--end')
    p.add_argument('--out', required=True)
    p.add_argument('--append', action='store_true')
    p.set_defaults(fn=cmd_backfill)

    p = sub.add_parser('status', help='show persisted state')
    p.add_argument('--state', default='newsbot_state.json')
    p.set_defaults(fn=cmd_status)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s', stream=sys.stderr)
    args.fn(args)


if __name__ == '__main__':
    main()

"""
Daily PMCC executor CLI.

Paper dry-run against bundled data (safe, offline)::

    python -m pmcc.live.run_live --ticker MC.PA --broker paper

Paper broker with sending enabled (still offline, simulated fills)::

    python -m pmcc.live.run_live --ticker MC.PA --broker paper --send

Interactive Brokers paper account (UNTESTED skeleton -- read
``pmcc/live/ib_adapter.py`` first; requires TWS/Gateway + ib_insync)::

    python -m pmcc.live.run_live --ticker MC.PA --broker ib --closes closes.csv

State is kept in ``pmcc_state_<TICKER>.json`` in --state-dir (default cwd).
Create a file named ``STOP`` next to it to freeze the executor.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from .. import data as datamod
from ..strategy import PMCC, PMCCParams
from .brokers import PaperBroker
from .executor import LiveExecutor, surface_from_history


def load_closes(args) -> pd.Series:
    if args.closes:
        s = pd.read_csv(args.closes, index_col=0, parse_dates=True).iloc[:, 0]
        return s.dropna()
    # offline fallback: bundled history (ends 2015 -- fine for dry-runs only)
    adj = datamod.load_adjusted_prices()[args.ticker].dropna()
    meta = datamod.UNIVERSE[args.ticker]
    return datamod.reconstruct_raw_prices(adj, meta.div_yield)


def main() -> None:
    ap = argparse.ArgumentParser(description='Daily PMCC live/paper executor')
    ap.add_argument('--ticker', required=True, choices=list(datamod.UNIVERSE))
    ap.add_argument('--broker', choices=['paper', 'ib'], default='paper')
    ap.add_argument('--send', action='store_true',
                    help='actually place orders (default: dry-run)')
    ap.add_argument('--closes', help='CSV of recent daily closes (date,close); '
                                     'required for meaningful live use')
    ap.add_argument('--state-dir', default='.')
    ap.add_argument('--rate', type=float, default=0.02, help='EUR short rate')
    ap.add_argument('--short-delta', type=float, default=0.25)
    ap.add_argument('--leaps-delta', type=float, default=0.80)
    args = ap.parse_args()

    closes = load_closes(args)
    spot = float(closes.iloc[-1])

    if args.broker == 'paper':
        surface = surface_from_history(closes)

        def model_mid(_t, spec, _s=surface, _spot=spot, _r=args.rate):
            from ..strategy import MarketView
            import datetime as dt
            meta = datamod.UNIVERSE[args.ticker]
            view = MarketView(dt.date.today(), _spot, _r, meta.div_yield, _s)
            return view.price(spec)

        broker = PaperBroker(prices={args.ticker: spot}, model_mid=model_mid,
                             state_file=os.path.join(
                                 args.state_dir,
                                 f'paper_broker_{args.ticker.replace(".", "_")}.json'))
    else:
        from .ib_adapter import IBBroker
        broker = IBBroker()

    strategy = PMCC(PMCCParams(short_target_delta=args.short_delta,
                               leaps_target_delta=args.leaps_delta))
    state_path = os.path.join(args.state_dir,
                              f'pmcc_state_{args.ticker.replace(".", "_")}.json')
    ex = LiveExecutor(broker, strategy, args.ticker, state_path,
                      closes=closes, r=args.rate, dry_run=not args.send)
    orders = ex.run_once()
    print(f'{len(orders)} order(s) this cycle; state -> {state_path}')


if __name__ == '__main__':
    main()

"""
Index-inclusion strategy ("trading the index effect") as a reusable package.

This is the machinery of the tutorial notebook
``doc/examples/Index Inclusion Strategy.py`` — a synthetic point-in-time
market with a mechanical FTSE-100-style rulebook, a look-ahead-free screener
that predicts additions ahead of the announcement, a stitched single-tape
representation for `backtesting.py`, and the `IndexInclusion` strategy —
factored into functions so that it can be swept and re-seeded programmatically
(see ``examples/run_index_inclusion.py``). The live-data counterpart is the
``inclusion-bot`` workspace package under ``bot/``.
"""

from .screener import build_tape, find_windows, rank_daily
from .simulate import Market, MarketParams, make_market
from .strategy import IndexInclusion, run_backtest

__all__ = [
    'IndexInclusion',
    'Market',
    'MarketParams',
    'build_tape',
    'find_windows',
    'make_market',
    'rank_daily',
    'run_backtest',
]

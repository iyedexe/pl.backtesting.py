"""The `backtesting.py` strategy and a one-call backtest runner."""

from __future__ import annotations

import pandas as pd

from backtesting import Backtest, Strategy

from .screener import build_tape, find_windows
from .simulate import Market


class IndexInclusion(Strategy):
    """Buy at Monday's open when a window switched ``Signal`` on at the Friday
    close; exit when the window says so or after ``hold_limit`` sessions."""

    hold_limit = 21                    # max sessions in a position (~1 calendar month)

    def init(self):
        self._last_window = -1

    def next(self):
        if self.position:
            held = len(self.data) - 1 - self.trades[-1].entry_bar
            if not self.data.Signal[-1] or held + 1 >= self.hold_limit:
                self.position.close()
        elif self.data.Signal[-1]:
            window = int(self.data.Window[-1])
            if window != self._last_window:
                self._last_window = window
                self.buy()


def run_backtest(market: Market, *, buffer: int = 2, pre_cutoff: int = 10,
                 hold_limit: int = 21, cash: float = 100_000,
                 commission: float = .002) -> pd.Series:
    """Screen -> tape -> Backtest for one parameter set; returns the stats Series."""
    windows = find_windows(market, buffer=buffer, pre_cutoff=pre_cutoff)
    tape, _ = build_tape(market, windows)
    bt = Backtest(tape, IndexInclusion, cash=cash, commission=commission,
                  finalize_trades=True)
    stats = bt.run(hold_limit=hold_limit)
    stats['_n_windows'] = len(windows)
    return stats

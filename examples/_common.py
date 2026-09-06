"""Shared helpers for the strategy example scripts (style, paths, output)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Importing the plotting module applies the project's validated chart style
# (palette, recessive grid, one axis per chart) to every matplotlib figure.
from pairs_trading import plotting  # noqa: F401
from pairs_trading.plotting import (  # noqa: F401
    AXIS,
    DIVERGING,
    INK,
    INK2,
    MUTED,
    SEQUENTIAL,
    SERIES,
    SURFACE,
)

FIG_DIR = Path(__file__).resolve().parent / 'figures'
TAB_DIR = Path(__file__).resolve().parent / 'tables'


def out_dirs() -> tuple[Path, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR, TAB_DIR


def save_table(df: pd.DataFrame, name: str) -> Path:
    _, tab = out_dirs()
    path = tab / f'{name}.csv'
    df.to_csv(path)
    return path


def print_table(df: pd.DataFrame, title: str, floatfmt: str = '.2f') -> None:
    print(f'\n{title}\n' + '-' * len(title))
    print(df.to_markdown(floatfmt=floatfmt))

"""
Pairs-trading research lab built on top of `backtesting.py`.

Subpackages/modules:

- `pairs_trading.data` — loaders for the vendored multi-asset daily panels
- `pairs_trading.stats` — cointegration, hedge-ratio and mean-reversion diagnostics
- `pairs_trading.signals` — z-score entry/exit/stop signal state machine
- `pairs_trading.engine` — two-leg, dollar-neutral pair backtest engine
- `pairs_trading.metrics` — performance and significance statistics
- `pairs_trading.walkforward` — rolling formation/trading out-of-sample harness
- `pairs_trading.screening` — universe-wide cointegration pair scans
- `pairs_trading.btpy_adapter` — cross-validation via the bundled backtesting.py engine
- `pairs_trading.experiments` — the five asset-class studies
- `pairs_trading.cli` — the `pairs` command-line interface
"""

from . import config, data, engine, metrics, signals, stats

__all__ = ['config', 'data', 'engine', 'metrics', 'signals', 'stats']

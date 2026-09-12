"""Configuration dataclasses shared across the research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class SignalConfig:
    """Z-score trading rule thresholds (in spread standard deviations)."""
    entry: float = 2.0      # open when |z| exceeds this
    exit: float = 0.0       # close when z reverts through this (toward 0)
    stop: float = 4.0       # close when z blows out beyond this (divergence stop)
    z_window: int = 60      # rolling window for the z-score mean/std
    max_holding: int | None = None  # time stop in bars (None = disabled)


@dataclass(frozen=True)
class CostModel:
    """Per-side proportional trading cost, in basis points of traded notional.

    The cost is charged on *each* leg's traded notional at entry and at exit, so a
    round trip on a dollar-neutral pair costs roughly ``4 * per_side_bp`` of the
    allocated (half-gross) notional. Slippage is folded into the same number.
    """
    per_side_bp: float = 5.0

    def rate(self) -> float:
        return self.per_side_bp / 1e4


@dataclass(frozen=True)
class EngineConfig:
    """Execution assumptions for the two-leg engine."""
    cost: CostModel = field(default_factory=CostModel)
    leverage: float = 1.0        # gross exposure as a multiple of current equity
    execution_lag: int = 1       # bars between signal close and execution close
    min_beta: float = 0.05       # skip entries when hedge ratio is outside
    max_beta: float = 20.0       # [min_beta, max_beta] (degenerate hedges)
    periods_per_year: int = 252  # 365 for crypto


@dataclass(frozen=True)
class WalkForwardConfig:
    """Rolling estimation harness: estimate on `formation`, trade on `trading`."""
    formation: int = 252         # bars used to fit beta and gate the pair
    trading: int = 63            # bars traded before re-estimation
    coint_pvalue_gate: float = 0.05   # trade a window only if formation EG p < gate
    min_half_life: float = 1.0        # bars; reject near-instant mean reversion (noise)
    max_half_life: float = 60.0       # bars; reject glacial mean reversion
    use_kalman: bool = False          # dynamic hedge ratio instead of formation OLS
    # State drift/bar for the Kalman hedge; far slower than Chan's 1e-4 because we
    # filter *log* prices — larger deltas let beta-noise x log-price swamp the spread.
    kalman_delta: float = 1e-7
    gate: bool = True                 # apply the cointegration/half-life gate at all
    z_mode: str = 'rolling'           # 'rolling' (causal window) or 'frozen' (GGR-style)


@dataclass(frozen=True)
class PairSpec:
    """A concrete pair within one of the vendored panels."""
    asset_class: str      # crypto | stocks | forex | commodities | cross | stocks_long
    a: str                # dependent leg (y in the cointegrating regression)
    b: str                # independent leg (x)
    label: str = ''

    @property
    def name(self) -> str:
        return self.label or f'{self.a}-{self.b}'


#: Defensible per-side cost assumptions per asset class, in bp of traded notional
#: (fee + half-spread + slippage; see report for sourcing).
DEFAULT_COSTS_BP = {
    'crypto': 12.0,       # Binance spot taker 10bp + ~2bp spread/slippage (majors)
    'stocks': 5.0,        # commission-free era; effective cost ~4.5bp/trade for S&P names
    'stocks_long': 5.0,
    'forex': 2.0,         # G10 crosses ~2-4bp; majors ~1bp — use 2bp conservatively
    'commodities': 5.0,   # ~2bp for CL/GC futures; 5bp conservative for spot proxies
    'cross': 12.0,        # max of the two legs, dominated by the crypto leg where present
}

PERIODS_PER_YEAR = {
    'crypto': 365,
    'stocks': 252,
    'stocks_long': 252,
    'forex': 252,
    'commodities': 252,
    'cross': 252,
}


def engine_config_for(asset_class: str, cost_bp: float | None = None,
                      **overrides) -> EngineConfig:
    """Build an :class:`EngineConfig` with per-asset-class defaults."""
    cfg = EngineConfig(
        cost=CostModel(cost_bp if cost_bp is not None
                       else DEFAULT_COSTS_BP.get(asset_class, 5.0)),
        periods_per_year=PERIODS_PER_YEAR.get(asset_class, 252),
    )
    return replace(cfg, **overrides) if overrides else cfg

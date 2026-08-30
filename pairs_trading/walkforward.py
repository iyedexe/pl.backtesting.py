"""Rolling formation/trading (walk-forward) harness.

This is the core anti-look-ahead device of the project, following the template
of Gatev, Goetzmann & Rouwenhorst (2006): all parameters used while trading
window *k* — the hedge ratio, the cointegration gate, the z-score history —
are estimated strictly from data before window *k* or accumulated causally
within it. Positions are forced flat at window boundaries (time stop), so no
trade ever relies on a relationship estimated after its entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import EngineConfig, SignalConfig, WalkForwardConfig
from .engine import TRADE_COLUMNS, backtest_pair
from .signals import generate_signals, zscore, zscore_frozen
from .stats import KalmanHedge, adf_pvalue, engle_granger, half_life, ols_hedge

WINDOW_COLUMNS = ['start', 'end', 'gate_pvalue', 'half_life', 'beta', 'traded',
                  'reject_reason', 'n_trades', 'window_return']


@dataclass
class WalkForwardResult:
    name: str
    returns: pd.Series               # stitched out-of-sample per-bar returns
    equity: pd.Series                # compounded OOS equity curve (starts at 1)
    trades: pd.DataFrame             # all round trips across windows
    windows: pd.DataFrame            # one row per formation/trading window
    meta: dict = field(default_factory=dict)

    @property
    def gate_pass_rate(self) -> float:
        return float(self.windows['traded'].mean()) if len(self.windows) else float('nan')


def kalman_state_paths(la: pd.Series, lb: pd.Series,
                       init_bars: int, delta: float):
    """One continuous, causal Kalman pass over the full pair history.

    The state is initialized from an OLS fit on the first ``init_bars``
    (which the walk-forward never trades — they are formation-only), then the
    filter runs once over everything. The hedge ratio at *t* therefore uses
    information up to *t* only. The intercept is **pinned** at its OLS value
    (``alpha_drift=False``): a random-walk alpha would absorb the very mean
    reversion the strategy trades (Kalman innovations are ~white by
    construction). The tradable spread is ``la - alpha0 - beta_t * lb``; its
    slow level moves are handled by the z-score's own mean. Restarting the
    filter every window would be wrong for a slow ``delta`` (it could never
    converge from a diffuse prior); continuity is the point of the filter.
    """
    alpha0, beta0 = ols_hedge(la.iloc[:init_bars], lb.iloc[:init_bars])
    kf = KalmanHedge(delta=delta, beta0=beta0, alpha0=alpha0, p0=1e-4,
                     alpha_drift=False)
    hist = kf.filter(la, lb)
    beta_path = hist['beta']
    spread_path = la - alpha0 - beta_path * lb
    return beta_path, spread_path


def _window_signals(la: pd.Series, lb: pd.Series, form: slice, trade: slice,
                    sig_cfg: SignalConfig, wf_cfg: WalkForwardConfig,
                    kalman_paths=None):
    """Compute (side, beta_series, diag) for one window, causally."""
    la_f, lb_f = la.iloc[form], lb.iloc[form]
    diag: dict = {}
    n_form = len(la_f)
    if wf_cfg.use_kalman:
        if kalman_paths is None:
            kalman_paths = kalman_state_paths(la, lb, n_form, wf_cfg.kalman_delta)
        beta_path, spread_path = kalman_paths
        beta_series = beta_path.iloc[trade]
        beta = float(beta_path.iloc[form].iloc[-1])
        spread_all = pd.concat([spread_path.iloc[form], spread_path.iloc[trade]])
        diag['beta'] = beta
        spread_f = spread_all.iloc[:n_form]
        diag['half_life'] = half_life(spread_f)
        # Gate on the spread this variant actually trades: ADF on the filtered
        # formation spread (slightly liberal: plain ADF critical values on a
        # spread with an estimated hedge ratio).
        diag['pvalue'] = adf_pvalue(spread_f)
    else:
        alpha, beta = ols_hedge(la_f, lb_f)
        beta_series = pd.Series(beta, index=la.index[trade])
        spread_all = pd.concat([la_f, la.iloc[trade]]) - alpha - beta * \
            pd.concat([lb_f, lb.iloc[trade]])
        diag['beta'] = beta
        spread_f = spread_all.iloc[:n_form]
        diag['half_life'] = half_life(spread_f)
    if wf_cfg.z_mode == 'frozen':
        z_all = zscore_frozen(spread_all, float(spread_f.mean()),
                              float(spread_f.std(ddof=1)))
    else:
        z_all = zscore(spread_all, sig_cfg.z_window)
    z_trade = z_all.iloc[n_form:]
    sigs = generate_signals(z_trade, sig_cfg)
    return sigs, beta_series, diag


def walk_forward_pair(prices: pd.DataFrame,
                      sig_cfg: SignalConfig,
                      eng_cfg: EngineConfig,
                      wf_cfg: WalkForwardConfig,
                      name: str = 'pair') -> WalkForwardResult:
    """Walk-forward backtest of a single, pre-specified pair.

    ``prices`` must have columns ['a', 'b']. Each trading window is gated on
    *formation-period* statistics only: Engle-Granger p-value and spread
    half-life. Windows that fail the gate stay flat (their zero returns remain
    in the stitched curve — capital committed but unused, GGR's conservative
    "committed capital" convention).
    """
    la = np.log(prices['a'].astype(float))
    lb = np.log(prices['b'].astype(float))
    n = len(prices)
    all_returns: list[pd.Series] = []
    all_trades: list[pd.DataFrame] = []
    window_rows: list[dict] = []
    kalman_paths = (kalman_state_paths(la, lb, wf_cfg.formation,
                                       wf_cfg.kalman_delta)
                    if wf_cfg.use_kalman else None)

    for start in range(wf_cfg.formation, n, wf_cfg.trading):
        end = min(start + wf_cfg.trading, n)
        if end - start < 5:
            break
        form = slice(start - wf_cfg.formation, start)
        trade = slice(start, end)
        idx_trade = prices.index[trade]
        row: dict = {'start': idx_trade[0], 'end': idx_trade[-1]}

        sigs, beta_series, diag = _window_signals(la, lb, form, trade, sig_cfg,
                                                  wf_cfg, kalman_paths)
        if 'pvalue' in diag:                      # Kalman: ADF on filtered spread
            gate_p = diag['pvalue']
        else:                                     # static hedge: Engle-Granger
            eg = engle_granger(prices['a'].iloc[form], prices['b'].iloc[form],
                               try_both_orientations=False)
            gate_p = eg.pvalue
        row['gate_pvalue'] = gate_p
        row['half_life'] = hl = diag['half_life']
        row['beta'] = beta = diag['beta']

        reject = ''
        if wf_cfg.gate:
            # `not (p <= gate)` (rather than `p > gate`) so a NaN p-value —
            # degenerate formation data — rejects instead of slipping through.
            if not (gate_p <= wf_cfg.coint_pvalue_gate):
                reject = 'coint'
            elif not (wf_cfg.min_half_life <= hl <= wf_cfg.max_half_life):
                reject = 'half_life'
            elif not (eng_cfg.min_beta <= beta <= eng_cfg.max_beta):
                reject = 'beta'
        row['traded'] = traded = reject == ''
        row['reject_reason'] = reject

        if traded:
            res = backtest_pair(prices.iloc[trade], sigs['side'], beta_series,
                                eng_cfg, reasons=sigs['reason'], name=name)
            rets = res.returns
            trades = res.trades.assign(window=idx_trade[0])
        else:
            rets = pd.Series(0.0, index=idx_trade)
            trades = pd.DataFrame(columns=TRADE_COLUMNS)
        row['n_trades'] = len(trades)
        row['window_return'] = float((1 + rets).prod() - 1)
        window_rows.append(row)
        all_returns.append(rets)
        if len(trades):
            all_trades.append(trades)

    returns = (pd.concat(all_returns) if all_returns
               else pd.Series(dtype=float))
    equity = (1 + returns).cumprod()
    trades = (pd.concat(all_trades, ignore_index=True) if all_trades
              else pd.DataFrame(columns=[*TRADE_COLUMNS, 'window']))
    windows = pd.DataFrame(window_rows, columns=WINDOW_COLUMNS)
    return WalkForwardResult(name=name, returns=returns, equity=equity,
                             trades=trades, windows=windows,
                             meta={'sig': sig_cfg, 'eng': eng_cfg, 'wf': wf_cfg})


def walk_forward_portfolio(panel: pd.DataFrame,
                           sig_cfg: SignalConfig,
                           eng_cfg: EngineConfig,
                           wf_cfg: WalkForwardConfig,
                           select_top: int = 5,
                           corr_min: float = 0.8,
                           name: str = 'portfolio') -> WalkForwardResult:
    """Gatev-style walk-forward *portfolio*: re-select pairs every window.

    Each window: screen the whole universe on the formation period only
    (correlation prefilter, then Engle-Granger + half-life gate), take the
    ``select_top`` pairs by EG p-value, trade them equal-weighted through the
    trading window, then re-select. Selection is therefore fully out-of-sample.

    Note: the screen's EG test is orientation-optimized (the better of both
    regressand choices), so its 5% level is slightly more liberal than the
    fixed-orientation gate in :func:`walk_forward_pair`. Kalman mode is not
    supported here — a per-window filter restart would defeat the filter.
    """
    from .screening import screen_panel  # local import to avoid cycle

    if wf_cfg.use_kalman:
        raise NotImplementedError('walk_forward_portfolio supports the OLS '
                                  'estimator only')
    dates = panel.index
    n = len(dates)
    all_returns: list[pd.Series] = []
    all_trades: list[pd.DataFrame] = []
    window_rows: list[dict] = []
    selections: list[dict] = []

    for start in range(wf_cfg.formation, n, wf_cfg.trading):
        end = min(start + wf_cfg.trading, n)
        if end - start < 5:
            break
        form_panel = panel.iloc[start - wf_cfg.formation:start]
        idx_trade = dates[start:end]
        scr = screen_panel(form_panel, corr_min=corr_min,
                           min_obs=int(wf_cfg.formation * 0.9))
        if wf_cfg.gate:
            ok = scr[(scr['eg_pvalue'] <= wf_cfg.coint_pvalue_gate)
                     & scr['half_life'].between(wf_cfg.min_half_life,
                                                wf_cfg.max_half_life)
                     & scr['beta'].between(eng_cfg.min_beta, eng_cfg.max_beta)]
        else:
            ok = scr
        chosen = ok.nsmallest(select_top, 'eg_pvalue') if len(ok) else ok
        selections.append({'start': idx_trade[0],
                           'pairs': [f'{r.y}/{r.x}' for r in chosen.itertuples()]})
        pair_rets = []
        for r in chosen.itertuples():
            cols = panel[[r.y, r.x]].iloc[start - wf_cfg.formation:end].dropna()
            sub = cols.rename(columns={r.y: 'a', r.x: 'b'})
            if len(sub) < wf_cfg.formation * 0.9:
                continue
            la, lb = np.log(sub['a']), np.log(sub['b'])
            n_form = (sub.index < idx_trade[0]).sum()
            sigs, beta_series, _ = _window_signals(
                la, lb, slice(0, n_form), slice(n_form, len(sub)), sig_cfg, wf_cfg)
            res = backtest_pair(sub.iloc[n_form:], sigs['side'], beta_series,
                                eng_cfg, reasons=sigs['reason'],
                                name=f'{r.y}/{r.x}')
            pair_rets.append(res.returns.reindex(idx_trade).fillna(0.0))
            if len(res.trades):
                all_trades.append(res.trades.assign(window=idx_trade[0],
                                                    pair=f'{r.y}/{r.x}'))
        if pair_rets:
            rets = pd.concat(pair_rets, axis=1).mean(axis=1)
        else:
            rets = pd.Series(0.0, index=idx_trade)
        window_rows.append({
            'start': idx_trade[0], 'end': idx_trade[-1],
            'gate_pvalue': float(chosen['eg_pvalue'].median()) if len(chosen) else np.nan,
            'half_life': float(chosen['half_life'].median()) if len(chosen) else np.nan,
            'beta': np.nan, 'traded': bool(len(chosen)),
            'reject_reason': '' if len(chosen) else 'no_pairs',
            'n_trades': int(sum(len(t) for t in all_trades
                                if len(t) and t['window'].iloc[0] == idx_trade[0])),
            'window_return': float((1 + rets).prod() - 1),
        })
        all_returns.append(rets)

    returns = pd.concat(all_returns) if all_returns else pd.Series(dtype=float)
    equity = (1 + returns).cumprod()
    trades = (pd.concat(all_trades, ignore_index=True) if all_trades
              else pd.DataFrame(columns=[*TRADE_COLUMNS, 'window', 'pair']))
    windows = pd.DataFrame(window_rows, columns=WINDOW_COLUMNS)
    return WalkForwardResult(name=name, returns=returns, equity=equity,
                             trades=trades, windows=windows,
                             meta={'selections': selections, 'sig': sig_cfg,
                                   'eng': eng_cfg, 'wf': wf_cfg})

"""
Backtest the news-trading rules with the in-repo `backtesting` engine.

`NewsStrategy` reads pre-aligned news scores (one per bar), buys at the next
bar's open when a bar carries a score >= `min_score` (limit-capped so a huge
gap is not chased), attaches a take-profit and stop-loss relative to the fill,
and force-closes after `hold_bars` bars (5 daily bars = one trading week).
Multi-ticker backtests run one `Backtest` per ticker.

Alignment rule: a headline published at time T is acted on at the close of the
last bar whose timestamp <= T and filled at the following bar's open. For daily
bars this means an after-close release on day D is bought at the open of D+1.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from backtesting import Backtest, Strategy

from .classifiers import Classifier, RuleClassifier
from .models import NewsItem, to_utc

NewsLike = Union[Sequence[NewsItem], pd.DataFrame, Sequence[dict], pd.Series]


def _as_items(news: NewsLike) -> Sequence[NewsItem]:
    if isinstance(news, pd.DataFrame):
        return [NewsItem.from_dict(r) for r in news.to_dict('records')]
    return [n if isinstance(n, NewsItem) else NewsItem.from_dict(n) for n in news]


def _index_to_utc(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')


def align_news_scores(index: pd.Index, news: NewsLike, *, ticker: Optional[str] = None,
                      classifier: Optional[Classifier] = None) -> np.ndarray:
    """Return one score per bar (0 where no news). If several items land on the same bar,
    the one with the largest absolute score wins."""
    if isinstance(news, pd.Series):          # already scored: timestamp -> score
        items = [NewsItem(id=str(i), headline='', published=to_utc(ts), tickers=[ticker or ''])
                 for i, ts in enumerate(news.index)]
        scores: Iterable[float] = (float(v) for v in news.values)
    else:
        classifier = classifier or RuleClassifier()
        items = [it for it in _as_items(news) if ticker is None or ticker.upper() in it.tickers]
        scores = (classifier.classify(it).score for it in items)
    utc_index = _index_to_utc(index)
    out = np.zeros(len(index))
    for item, score in zip(items, scores):
        pos = int(utc_index.searchsorted(item.published, side='right')) - 1
        if 0 <= pos < len(out) and abs(score) > abs(out[pos]):
            out[pos] = score
    return out


class NewsStrategy(Strategy):
    """
    Parameters (override via `Backtest.run(**kwargs)` or subclassing):
      news        : NewsItems / DataFrame / dicts / scored Series, or a precomputed per-bar score array.
      ticker      : restrict `news` to items mentioning this ticker (None = use all).
      classifier  : Classifier instance (defaults to RuleClassifier).
      min_score   : long when the bar's score >= this.
      target_pct / stop_pct : exit distances from the reference (signal-bar close) price.
      scale_target_by_score : widen the target for stronger catalysts (x1.0 @0.5 .. x1.25 @1.0).
      hold_bars   : force-close after this many bars in the trade (5 daily bars = 1 week).
      max_hold_days: alternative calendar-day limit (None = use hold_bars only).
      max_chase_pct: entry is a limit at signal-close x (1 + max_chase_pct); None = market order at next open.
      size        : fraction of equity per trade (0 < size < 1).
    """
    news: Any = None
    ticker: Optional[str] = None
    classifier: Optional[Classifier] = None
    min_score = 0.5
    target_pct = 0.05
    stop_pct = 0.03
    scale_target_by_score = True
    hold_bars = 5
    max_hold_days: Optional[float] = None
    max_chase_pct: Optional[float] = 0.05
    size = 0.95

    def init(self):
        if self.news is None:
            raise ValueError('NewsStrategy needs `news`')
        if isinstance(self.news, np.ndarray):
            scores = np.asarray(self.news, dtype=float)
            assert len(scores) == len(self.data), 'precomputed score array must match data length'
        else:
            scores = align_news_scores(self.data.df.index, self.news, ticker=self.ticker, classifier=self.classifier)
        self._scores = scores
        self.news_score = self.I(lambda s: s, scores, name='News score', overlay=False, color='orange')

    def _target_pct(self, score: float) -> float:
        if not self.scale_target_by_score:
            return self.target_pct
        return self.target_pct * (0.75 + 0.5 * min(1.0, max(0.0, score)))

    def next(self):
        i = len(self.data) - 1
        now = self.data.index[-1]
        # 1. An entry limit placed on the previous bar that did not fill is stale: cancel it.
        for order in list(self.orders):
            if not order.is_contingent:
                order.cancel()
        # 2. Attach stop/target to trades filled at this bar's open, relative to the *actual* fill price
        #    (setting them at order time would use the signal-bar close and get run over by an opening gap).
        for trade in self.trades:
            if trade.sl is None and trade.tp is None:
                score = float(trade.tag) if trade.tag is not None else self.min_score
                trade.sl = trade.entry_price * (1 - self.stop_pct)
                trade.tp = trade.entry_price * (1 + self._target_pct(score))
        # 3. Time exit: close order placed so the trade spans exactly `hold_bars` sessions.
        for trade in self.trades:
            if i - trade.entry_bar >= self.hold_bars - 1:
                trade.close()
            elif self.max_hold_days is not None and isinstance(now, pd.Timestamp) \
                    and now >= trade.entry_time + timedelta(days=self.max_hold_days):
                trade.close()
        # 4. New signal on this bar -> buy at next open, but not above `max_chase_pct` over this close.
        score = float(self._scores[i])
        if score >= self.min_score and not self.position:
            price = float(self.data.Close[-1])
            limit = price * (1 + self.max_chase_pct) if self.max_chase_pct is not None else None
            self.buy(size=self.size, limit=limit, tag=round(score, 3))


def run_news_backtest(prices: Mapping[str, pd.DataFrame], news: NewsLike, *, cash: float = 100_000,
                      commission: float = 0.0005, finalize_trades: bool = True,
                      **strategy_params) -> Dict[str, Any]:
    """
    Run `NewsStrategy` once per ticker. Returns a dict with per-ticker `stats`, per-ticker `backtests`
    (for `.plot()`), a combined `trades` DataFrame and an aggregate `summary` dict.
    """
    stats: Dict[str, pd.Series] = {}
    backtests: Dict[str, Backtest] = {}
    frames = []
    for ticker, df in prices.items():
        bt = Backtest(df, NewsStrategy, cash=cash, commission=commission, finalize_trades=finalize_trades)
        st = bt.run(news=news, ticker=ticker, **strategy_params)
        stats[ticker] = st
        backtests[ticker] = bt
        trades = st['_trades'].copy()
        trades.insert(0, 'Ticker', ticker)
        frames.append(trades)
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {'stats': stats, 'backtests': backtests, 'trades': trades, 'summary': summarize_trades(trades)}


def summarize_trades(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades.empty:
        return {'trades': 0}
    ret = trades['ReturnPct']
    wins = ret > 0
    return {
        'trades': int(len(trades)),
        'win_rate_pct': round(100 * wins.mean(), 2),
        'avg_return_pct': round(100 * ret.mean(), 3),
        'median_return_pct': round(100 * ret.median(), 3),
        'best_pct': round(100 * ret.max(), 3),
        'worst_pct': round(100 * ret.min(), 3),
        'total_pnl': round(float(trades['PnL'].sum()), 2),
        'profit_factor': (round(float(trades.loc[wins, 'PnL'].sum() / -trades.loc[~wins, 'PnL'].sum()), 3)
                          if (~wins).any() and trades.loc[~wins, 'PnL'].sum() < 0 else None),
        'avg_hold_bars': round(float((trades['ExitBar'] - trades['EntryBar']).mean()), 2),
        'max_hold_bars': int((trades['ExitBar'] - trades['EntryBar']).max()),
    }

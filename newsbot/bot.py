"""
The trading loop.

Each `tick()`:
  1. pulls fresh headlines from every source, scores them, and queues `Signal`s;
  2. checks open positions for target / stop / time exits;
  3. fills queued signals while there is room (max positions, cash, market open);
  4. persists state.

Every dependency (sources, prices, broker, clock) is injected so the identical
code path runs live, in replay against CSV data, and in unit tests.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional

from .brokers import Broker, BrokerError
from .clock import Clock, SystemClock
from .models import ClosedTrade, ExitReason, NewsItem, Position, Signal, to_utc
from .prices import PriceFeed
from .signals import SignalEngine
from .sources import CompositeNewsSource, NewsSource
from .state import BotState

log = logging.getLogger(__name__)


class NewsTradingBot:
    def __init__(self, *,
                 sources: Iterable[NewsSource],
                 engine: SignalEngine,
                 broker: Broker,
                 price_feed: PriceFeed,
                 state: Optional[BotState] = None,
                 clock: Optional[Clock] = None,
                 max_positions: int = 5,
                 position_size_pct: float = 0.2,
                 risk_per_trade_pct: Optional[float] = None,
                 max_position_value: Optional[float] = None,
                 fractional_shares: bool = False,
                 max_news_age: timedelta = timedelta(minutes=30),
                 max_chase_pct: Optional[float] = 0.05,
                 market_hours_only: bool = True,
                 poll_interval: float = 60.0,
                 on_event: Optional[Callable[[str, dict], None]] = None):
        """
        max_positions: concurrent open longs.
        position_size_pct: notional per position as a fraction of equity.
        risk_per_trade_pct: if set, also cap size so that a stop-out loses at most this fraction of equity.
        max_news_age: ignore headlines older than this when first seen (protects against stale feeds at startup).
        max_chase_pct: skip an entry if price already ran more than this above the price seen at signal time.
        market_hours_only: only trade while `broker.is_market_open()`; signals wait in the queue until then.
        """
        self.source = CompositeNewsSource(sources)
        self.engine = engine
        self.broker = broker
        self.feed = price_feed
        self.state = state or BotState()
        self.clock = clock or SystemClock()
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_value = max_position_value
        self.fractional_shares = fractional_shares
        self.max_news_age = max_news_age
        self.max_chase_pct = max_chase_pct
        self.market_hours_only = market_hours_only
        self.poll_interval = poll_interval
        self._on_event = on_event

    # ------------------------------------------------------------ helpers
    def _emit(self, kind: str, **payload) -> None:
        if self._on_event:
            self._on_event(kind, payload)

    def _market_open(self, now: datetime) -> bool:
        return (not self.market_hours_only) or self.broker.is_market_open(now)

    # ------------------------------------------------------------ 1. news
    def ingest(self, items: Iterable[NewsItem], now: Optional[datetime] = None) -> List[Signal]:
        """Score news items and queue signals. Returns the newly queued signals."""
        now = now or self.clock.now()
        queued: List[Signal] = []
        for item in items:
            if self.state.has_seen(item.id):
                continue
            if item.published > now:          # replay: not published yet
                continue
            self.state.mark_seen(item.id)
            if now - item.published > self.max_news_age:
                log.debug('skipping stale news (%s): %s', item.published, item.headline)
                continue
            for sig in self.engine.evaluate(item, now):
                if sig.ticker in self.state.positions:
                    log.info('signal for %s ignored: already holding', sig.ticker)
                    continue
                if any(p.ticker == sig.ticker for p in self.state.pending.values()):
                    continue
                sig.reference_price = self.feed.price(sig.ticker)
                self.state.pending[sig.id] = sig
                queued.append(sig)
                log.info('SIGNAL %s score=%.2f %s tp=+%.1f%% sl=-%.1f%% :: %s',
                         sig.ticker, sig.score, sig.category, sig.target_pct * 100, sig.stop_pct * 100, sig.headline)
                self._emit('signal', signal=sig, item=item)
        return queued

    def poll_news(self, now: Optional[datetime] = None) -> List[Signal]:
        now = now or self.clock.now()
        since = now - self.max_news_age
        return self.ingest(self.source.fetch(since), now)

    # ------------------------------------------------------------ 2. exits
    def _close(self, pos: Position, reason: ExitReason, now: datetime, exit_price_hint: Optional[float] = None) -> None:
        if reason is ExitReason.BROKER:
            exit_price = exit_price_hint if exit_price_hint is not None else pos.entry_price
            exit_time = now
        else:
            try:
                fill = self.broker.sell(pos.ticker, pos.qty)
            except BrokerError as e:
                log.error('exit %s (%s) failed: %s', pos.ticker, reason.value, e)
                return
            exit_price, exit_time = fill.price, fill.time
        trade = ClosedTrade(ticker=pos.ticker, qty=pos.qty, entry_price=pos.entry_price, entry_time=pos.entry_time,
                            exit_price=exit_price, exit_time=exit_time, reason=reason.value,
                            headline=pos.headline, signal_id=pos.signal_id)
        self.state.closed.append(trade)
        del self.state.positions[pos.ticker]
        log.info('EXIT %s (%s) x%g @ %.4f  pnl=%.2f (%.2f%%) held %s', pos.ticker, reason.value, pos.qty, exit_price,
                 trade.pnl, trade.return_pct, exit_time - pos.entry_time)
        self._emit('exit', trade=trade, position=pos)

    def manage_positions(self, now: Optional[datetime] = None) -> None:
        now = now or self.clock.now()
        if not self.state.positions:
            return
        broker_positions = self.broker.positions() if self.broker.manages_exits else None
        for pos in list(self.state.positions.values()):
            px = self.feed.price(pos.ticker)
            if broker_positions is not None and pos.ticker not in broker_positions:
                # A bracket leg (take-profit or stop) filled on the broker's side.
                self._close(pos, ExitReason.BROKER, now, exit_price_hint=px)
                continue
            if not self._market_open(now):
                continue
            if px is None:
                log.warning('no price for %s; cannot evaluate exits', pos.ticker)
                continue
            if broker_positions is None:      # we manage target/stop ourselves (paper)
                if px >= pos.target_price:
                    self._close(pos, ExitReason.TARGET, now)
                    continue
                if px <= pos.stop_price:
                    self._close(pos, ExitReason.STOP, now)
                    continue
            if now >= pos.exit_by:
                self._close(pos, ExitReason.TIME, now)

    # ------------------------------------------------------------ 3. entries
    def size_for(self, signal: Signal, price: float) -> float:
        equity = self.broker.equity()
        cash = self.broker.cash()
        notional = equity * self.position_size_pct
        if self.max_position_value is not None:
            notional = min(notional, self.max_position_value)
        notional = min(notional, cash)
        qty = notional / price
        if self.risk_per_trade_pct and signal.stop_pct > 0:
            qty = min(qty, equity * self.risk_per_trade_pct / (price * signal.stop_pct))
        if not self.fractional_shares:
            qty = math.floor(qty)
        return max(0.0, qty)

    def execute_signals(self, now: Optional[datetime] = None) -> List[Position]:
        now = now or self.clock.now()
        opened: List[Position] = []
        for sig in sorted(self.state.pending.values(), key=lambda s: -s.score):
            if sig.is_expired(now):
                log.info('signal %s for %s expired unfilled', sig.id, sig.ticker)
                del self.state.pending[sig.id]
                continue
            if sig.ticker in self.state.positions:
                del self.state.pending[sig.id]
                continue
            if len(self.state.positions) >= self.max_positions:
                break
            if not self._market_open(now):
                break
            px = self.feed.price(sig.ticker) or sig.reference_price
            if not px:
                log.warning('no price for %s; signal stays queued', sig.ticker)
                continue
            if (self.max_chase_pct is not None and sig.reference_price
                    and px > sig.reference_price * (1 + self.max_chase_pct)):
                log.info('skipping %s: price %.2f already %.1f%% above signal price %.2f', sig.ticker, px,
                         (px / sig.reference_price - 1) * 100, sig.reference_price)
                del self.state.pending[sig.id]
                continue
            qty = self.size_for(sig, px)
            if qty <= 0:
                log.info('skipping %s: position size rounds to zero', sig.ticker)
                del self.state.pending[sig.id]
                continue
            try:
                fill = self.broker.buy(sig.ticker, qty, take_profit=sig.target_price(px),
                                       stop_loss=sig.stop_price(px), reference_price=px)
            except BrokerError as e:
                log.error('entry %s failed: %s', sig.ticker, e)
                del self.state.pending[sig.id]
                continue
            pos = Position(ticker=sig.ticker, qty=fill.qty, entry_price=fill.price, entry_time=fill.time,
                           target_price=sig.target_price(fill.price), stop_price=sig.stop_price(fill.price),
                           exit_by=fill.time + timedelta(days=sig.max_hold_days),
                           signal_id=sig.id, news_id=sig.news_id, headline=sig.headline, broker_order_id=fill.order_id)
            self.state.positions[pos.ticker] = pos
            del self.state.pending[sig.id]
            opened.append(pos)
            log.info('ENTRY %s x%g @ %.4f  target=%.4f stop=%.4f exit_by=%s', pos.ticker, pos.qty, pos.entry_price,
                     pos.target_price, pos.stop_price, pos.exit_by.isoformat(timespec='minutes'))
            self._emit('entry', position=pos, signal=sig)
        return opened

    # ------------------------------------------------------------ loop
    def tick(self) -> None:
        now = self.clock.now()
        self.poll_news(now)
        self.manage_positions(now)
        self.execute_signals(now)
        self.state.last_poll = now.isoformat()
        self.state.save()

    def run(self, max_ticks: Optional[int] = None) -> None:
        ticks = 0
        log.info('news bot started: %d source(s), max %d positions, %s',
                 len(self.source.sources), self.max_positions, type(self.broker).__name__)
        try:
            while max_ticks is None or ticks < max_ticks:
                try:
                    self.tick()
                except Exception:  # noqa: BLE001 — keep the loop alive; the error is logged with traceback
                    log.exception('tick failed')
                ticks += 1
                if max_ticks is not None and ticks >= max_ticks:
                    break
                self.clock.sleep(self.poll_interval)
        except KeyboardInterrupt:
            log.info('stopped by user')
        finally:
            self.state.save()

    # ------------------------------------------------------------ reporting
    def summary(self) -> Dict[str, object]:
        closed = self.state.closed
        wins = [t for t in closed if t.pnl > 0]
        return {
            'equity': round(self.broker.equity(), 2),
            'cash': round(self.broker.cash(), 2),
            'open_positions': len(self.state.positions),
            'pending_signals': len(self.state.pending),
            'closed_trades': len(closed),
            'realized_pnl': round(sum(t.pnl for t in closed), 2),
            'win_rate_pct': round(100 * len(wins) / len(closed), 1) if closed else None,
            'avg_return_pct': round(sum(t.return_pct for t in closed) / len(closed), 3) if closed else None,
            'exits': {r.value: sum(1 for t in closed if t.reason == r.value) for r in ExitReason
                      if any(t.reason == r.value for t in closed)},
        }


def positions_table(positions: Iterable[Position], feed: Optional[PriceFeed] = None) -> str:
    rows = ['ticker    qty   entry     target    stop      exit_by             last      pnl']
    for p in positions:
        px = feed.price(p.ticker) if feed else None
        pnl = f'{p.unrealized_pnl(px):9.2f}' if px is not None else '        -'
        rows.append(f'{p.ticker:<7}{p.qty:6g} {p.entry_price:9.2f} {p.target_price:9.2f} {p.stop_price:9.2f} '
                    f'{to_utc(p.exit_by).strftime("%Y-%m-%d %H:%M"):<19} {px if px is not None else "-":>9} {pnl}')
    return '\n'.join(rows)

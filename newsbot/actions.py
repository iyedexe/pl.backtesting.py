"""
What happens when a signal fires. Configure any combination:

  - `TradeAction`    : the bot queues the signal and places the order itself (paper or broker).
  - `TelegramAction` : sends a message to a chat (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).
  - `WebhookAction`  : POSTs the signal as JSON to a URL.
  - `LogAction`      : just logs (default when nothing else is configured).

Without a `TradeAction` the bot is notify-only: it scores, alerts, and never trades.
"""
from __future__ import annotations

import json
import logging
from abc import ABC
from typing import Callable, Iterable, Optional

from .http import env_key
from .models import Classification, ClosedTrade, Position, Signal

log = logging.getLogger(__name__)

EVENTS = ('signal', 'entry', 'exit')


class Action(ABC):
    name = 'action'
    executes_trades = False

    def __init__(self, events: Optional[Iterable[str]] = None):
        self.events = set(events) if events else set(EVENTS)

    def on_signal(self, signal: Signal, classification: Classification, evidence_text: str = '') -> None:
        pass

    def on_entry(self, position: Position) -> None:
        pass

    def on_exit(self, trade: ClosedTrade) -> None:
        pass


class TradeAction(Action):
    name = 'trade'
    executes_trades = True


class LogAction(Action):
    name = 'log'

    def on_signal(self, signal, classification, evidence_text=''):
        log.info('ACTION signal %s score=%.2f %s reasons=%s', signal.ticker, classification.score,
                 classification.category, '; '.join(classification.reasons))


class CallbackAction(Action):
    """Route events to plain functions (tests, notebooks)."""
    name = 'callback'

    def __init__(self, on_signal: Optional[Callable] = None, on_entry: Optional[Callable] = None,
                 on_exit: Optional[Callable] = None):
        super().__init__()
        self._s, self._e, self._x = on_signal, on_entry, on_exit

    def on_signal(self, signal, classification, evidence_text=''):
        if self._s:
            self._s(signal, classification, evidence_text)

    def on_entry(self, position):
        if self._e:
            self._e(position)

    def on_exit(self, trade):
        if self._x:
            self._x(trade)


def format_signal(signal: Signal, c: Classification, evidence_text: str = '', max_evidence: int = 6) -> str:
    ref = f'\nRef price: {signal.reference_price:.2f}' if signal.reference_price else ''
    tgt = ''
    if signal.reference_price:
        tgt = (f' (target {signal.target_price(signal.reference_price):.2f}, '
               f'stop {signal.stop_price(signal.reference_price):.2f})')
    lines = [f'📈 LONG signal {signal.ticker}  score {c.score:+.2f}  [{c.category}]  conf {c.confidence:.2f}',
             f'Trigger: {signal.headline}',
             f'Plan: +{signal.target_pct * 100:.1f}% target / -{signal.stop_pct * 100:.1f}% stop / '
             f'max {signal.max_hold_days:g} days{tgt}{ref}',
             f'Valid until: {signal.expires_at:%Y-%m-%d %H:%M} UTC']
    if c.reasons:
        lines.append('Why: ' + '; '.join(c.reasons[:5]))
    if evidence_text:
        ev = evidence_text.strip().splitlines()[-max_evidence:]
        lines.append('Evidence:\n' + '\n'.join(ev))
    return '\n'.join(lines)


def format_entry(p: Position) -> str:
    return (f'✅ ENTERED {p.ticker} x{p.qty:g} @ {p.entry_price:.2f}\n'
            f'target {p.target_price:.2f} / stop {p.stop_price:.2f} / exit by {p.exit_by:%Y-%m-%d %H:%M} UTC')


def format_exit(t: ClosedTrade) -> str:
    emoji = '🟢' if t.pnl >= 0 else '🔴'
    return (f'{emoji} EXIT {t.ticker} ({t.reason}) x{t.qty:g} {t.entry_price:.2f} -> {t.exit_price:.2f}  '
            f'{t.return_pct:+.2f}%  pnl {t.pnl:+.2f}')


class TelegramAction(Action):
    name = 'telegram'
    API = 'https://api.telegram.org/bot{token}/sendMessage'

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None, *,
                 events: Optional[Iterable[str]] = None, include_evidence: bool = True,
                 sender: Optional[Callable[[str], None]] = None, timeout: float = 10.0):
        super().__init__(events)
        self.token = env_key('TELEGRAM_BOT_TOKEN', explicit=token)
        self.chat_id = env_key('TELEGRAM_CHAT_ID', explicit=chat_id)
        self.include_evidence = include_evidence
        self.timeout = timeout
        self._sender = sender
        self.sent: list = []
        if not sender and not (self.token and self.chat_id):
            log.warning('telegram: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; messages will only be logged')

    def send(self, text: str) -> None:
        self.sent.append(text)
        if self._sender:
            self._sender(text)
            return
        if not (self.token and self.chat_id):
            log.info('telegram (dry-run):\n%s', text)
            return
        import requests  # noqa: PLC0415
        try:
            r = requests.post(self.API.format(token=self.token), timeout=self.timeout,
                              json={'chat_id': self.chat_id, 'text': text[:4000], 'disable_web_page_preview': True})
            if r.status_code >= 400:
                log.warning('telegram send failed: %s %s', r.status_code, r.text[:200])
        except Exception as e:  # noqa: BLE001
            log.warning('telegram send failed: %s', e)

    def on_signal(self, signal, classification, evidence_text=''):
        if 'signal' in self.events:
            self.send(format_signal(signal, classification, evidence_text if self.include_evidence else ''))

    def on_entry(self, position):
        if 'entry' in self.events:
            self.send(format_entry(position))

    def on_exit(self, trade):
        if 'exit' in self.events:
            self.send(format_exit(trade))


class WebhookAction(Action):
    name = 'webhook'

    def __init__(self, url: str, *, events: Optional[Iterable[str]] = None, headers: Optional[dict] = None,
                 sender: Optional[Callable[[dict], None]] = None, timeout: float = 10.0):
        super().__init__(events)
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self._sender = sender

    def post(self, payload: dict) -> None:
        if self._sender:
            self._sender(payload)
            return
        import requests  # noqa: PLC0415
        try:
            requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout).raise_for_status()
        except Exception as e:  # noqa: BLE001
            log.warning('webhook %s failed: %s', self.url, e)

    def on_signal(self, signal, classification, evidence_text=''):
        if 'signal' in self.events:
            self.post({'event': 'signal', 'signal': signal.to_dict(), 'score': classification.score,
                       'category': classification.category, 'confidence': classification.confidence,
                       'reasons': classification.reasons, 'evidence': evidence_text})

    def on_entry(self, position):
        if 'entry' in self.events:
            self.post({'event': 'entry', 'position': position.to_dict()})

    def on_exit(self, trade):
        if 'exit' in self.events:
            self.post({'event': 'exit', 'trade': json.loads(json.dumps(trade.to_dict()))})

"""
What happens when a signal fires. Configure any combination:

  - `TradeAction`    : the bot queues the signal and places the order itself (paper or broker).
  - `SlackAction`    : posts to a Slack incoming webhook (SLACK_WEBHOOK_URL).
  - `DiscordAction`  : posts to a Discord webhook (DISCORD_WEBHOOK_URL).
  - `TelegramAction` : sends to a chat (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).
  - `WebhookAction`  : POSTs the full report as JSON to any URL.
  - `LogAction`      : just logs.

Every messenger receives a `SignalReport`: estimated entry, target, stop, exit deadline, size,
risk/reward, the model's score and reasons, and the evidence behind it. Without a `TradeAction`
the bot is notify-only: it scores, alerts, and never trades.
"""
from __future__ import annotations

import json
import logging
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from .http import env_key
from .models import Classification, ClosedTrade, Position, Signal

log = logging.getLogger(__name__)

EVENTS = ('signal', 'entry', 'exit')


@dataclass
class SignalReport:
    """Everything a human (or a downstream system) needs to act on a signal."""
    signal: Signal
    classification: Classification
    now: datetime
    reference_price: Optional[float] = None
    entry_limit: Optional[float] = None          # do not chase above this
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    exit_by: Optional[datetime] = None           # time exit if filled now
    suggested_qty: Optional[float] = None
    notional: Optional[float] = None
    risk_amount: Optional[float] = None          # loss if stopped out
    sources: List[str] = field(default_factory=list)
    n_items: int = 0
    evidence_lines: List[str] = field(default_factory=list)
    url: str = ''
    market: str = ''
    market_tz: str = 'UTC'
    mode: str = 'notify-only'

    @property
    def ticker(self) -> str:
        return self.signal.ticker

    @property
    def risk_reward(self) -> float:
        return round(self.signal.target_pct / self.signal.stop_pct, 2) if self.signal.stop_pct else 0.0

    def _local(self, dt: Optional[datetime]) -> str:
        if dt is None:
            return 'n/a'
        try:
            local = dt.astimezone(ZoneInfo(self.market_tz))
        except Exception:  # noqa: BLE001
            local = dt
        return f'{local:%a %d %b %H:%M} {local.tzname() or ""}'.strip()

    @staticmethod
    def _px(v: Optional[float]) -> str:
        return 'n/a' if v is None else f'{v:,.2f}'

    def text(self, max_evidence: int = 6) -> str:
        s, c = self.signal, self.classification
        head = (f'LONG {self.ticker}  score {c.score:+.2f}  [{c.category}]  confidence {c.confidence:.0%}'
                + (f'  ({self.market.upper()})' if self.market else ''))
        lines = [f'📈 {head}',
                 f'Trigger: {s.headline}' + (f'\n{self.url}' if self.url else ''),
                 '',
                 f'Entry:  ~{self._px(self.reference_price)} now, limit {self._px(self.entry_limit)} '
                 f'(do not chase above)',
                 f'Target: {self._px(self.target_price)} (+{s.target_pct * 100:.1f}%)',
                 f'Stop:   {self._px(self.stop_price)} (-{s.stop_pct * 100:.1f}%)   R:R {self.risk_reward}',
                 f'Exit by: {self._local(self.exit_by)} (max {s.max_hold_days:g} days) or on target/stop',
                 f'Signal valid until: {self._local(s.expires_at)}']
        if self.suggested_qty:
            lines.append(f'Size:   {self.suggested_qty:g} shares ≈ {self._px(self.notional)} '
                         f'(risk ≈ {self._px(self.risk_amount)} at stop)')
        if c.reasons:
            lines += ['', 'Why:'] + [f'  • {r}' for r in c.reasons[:6]]
        lines.append('')
        lines.append(f'Evidence: {self.n_items} item(s) from {", ".join(self.sources) or "n/a"}')
        for ev in self.evidence_lines[-max_evidence:]:
            lines.append(f'  {ev[:200]}')
        lines.append(f'Mode: {self.mode}')
        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        c = self.classification
        return {
            'event': 'signal', 'ticker': self.ticker, 'side': 'long', 'time': self.now.isoformat(),
            'score': c.score, 'category': c.category, 'confidence': c.confidence, 'reasons': c.reasons,
            'headline': self.signal.headline, 'url': self.url, 'market': self.market,
            'reference_price': self.reference_price, 'entry_limit': self.entry_limit,
            'target_price': self.target_price, 'stop_price': self.stop_price,
            'target_pct': self.signal.target_pct, 'stop_pct': self.signal.stop_pct, 'risk_reward': self.risk_reward,
            'max_hold_days': self.signal.max_hold_days, 'exit_by': self.exit_by.isoformat() if self.exit_by else None,
            'valid_until': self.signal.expires_at.isoformat(),
            'suggested_qty': self.suggested_qty, 'notional': self.notional, 'risk_amount': self.risk_amount,
            'sources': self.sources, 'n_items': self.n_items, 'evidence': self.evidence_lines,
            'signal': self.signal.to_dict(), 'mode': self.mode,
        }


class Action(ABC):
    name = 'action'
    executes_trades = False

    def __init__(self, events: Optional[Iterable[str]] = None):
        self.events = set(events) if events else set(EVENTS)

    def on_signal(self, report: SignalReport) -> None:
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

    def on_signal(self, report):
        log.info('ACTION signal\n%s', report.text())


class CallbackAction(Action):
    """Route events to plain functions (tests, notebooks)."""
    name = 'callback'

    def __init__(self, on_signal: Optional[Callable] = None, on_entry: Optional[Callable] = None,
                 on_exit: Optional[Callable] = None):
        super().__init__()
        self._s, self._e, self._x = on_signal, on_entry, on_exit

    def on_signal(self, report):
        if self._s:
            self._s(report)

    def on_entry(self, position):
        if self._e:
            self._e(position)

    def on_exit(self, trade):
        if self._x:
            self._x(trade)


def format_entry(p: Position) -> str:
    return (f'✅ ENTERED {p.ticker} x{p.qty:g} @ {p.entry_price:.2f}\n'
            f'target {p.target_price:.2f} / stop {p.stop_price:.2f} / exit by {p.exit_by:%Y-%m-%d %H:%M} UTC')


def format_exit(t: ClosedTrade) -> str:
    emoji = '🟢' if t.pnl >= 0 else '🔴'
    return (f'{emoji} EXIT {t.ticker} ({t.reason}) x{t.qty:g} {t.entry_price:.2f} -> {t.exit_price:.2f}  '
            f'{t.return_pct:+.2f}%  pnl {t.pnl:+.2f}')


class MessengerAction(Action):
    """Base for chat notifiers: subclasses implement `_deliver(text)`."""
    name = 'messenger'
    MAX_LEN = 4000

    def __init__(self, *, events: Optional[Iterable[str]] = None, include_evidence: bool = True,
                 sender: Optional[Callable[[str], None]] = None, timeout: float = 10.0):
        super().__init__(events)
        self.include_evidence = include_evidence
        self.timeout = timeout
        self._sender = sender
        self.sent: List[str] = []

    @property
    def configured(self) -> bool:
        return True

    def send(self, text: str) -> None:
        text = text[:self.MAX_LEN]
        self.sent.append(text)
        if self._sender:
            self._sender(text)
            return
        if not self.configured:
            log.info('%s (dry-run, not configured):\n%s', self.name, text)
            return
        try:
            self._deliver(text)
        except Exception as e:  # noqa: BLE001
            log.warning('%s send failed: %s', self.name, e)

    def _deliver(self, text: str) -> None:
        raise NotImplementedError

    def on_signal(self, report):
        if 'signal' in self.events:
            self.send(report.text(max_evidence=6 if self.include_evidence else 0))

    def on_entry(self, position):
        if 'entry' in self.events:
            self.send(format_entry(position))

    def on_exit(self, trade):
        if 'exit' in self.events:
            self.send(format_exit(trade))


class TelegramAction(MessengerAction):
    name = 'telegram'
    API = 'https://api.telegram.org/bot{token}/sendMessage'
    MAX_LEN = 4000

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None, **kw):
        super().__init__(**kw)
        self.token = env_key('TELEGRAM_BOT_TOKEN', explicit=token)
        self.chat_id = env_key('TELEGRAM_CHAT_ID', explicit=chat_id)
        if not self._sender and not self.configured:
            log.warning('telegram: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; messages will only be logged')

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def _deliver(self, text: str) -> None:
        import requests  # noqa: PLC0415
        r = requests.post(self.API.format(token=self.token), timeout=self.timeout,
                          json={'chat_id': self.chat_id, 'text': text, 'disable_web_page_preview': True})
        if r.status_code >= 400:
            raise RuntimeError(f'{r.status_code} {r.text[:200]}')


class SlackAction(MessengerAction):
    """Slack incoming webhook (https://api.slack.com/messaging/webhooks)."""
    name = 'slack'
    MAX_LEN = 3800

    def __init__(self, webhook_url: Optional[str] = None, **kw):
        super().__init__(**kw)
        self.webhook_url = env_key('SLACK_WEBHOOK_URL', explicit=webhook_url)
        if not self._sender and not self.configured:
            log.warning('slack: SLACK_WEBHOOK_URL not set; messages will only be logged')

    @property
    def configured(self) -> bool:
        return bool(self.webhook_url)

    def _deliver(self, text: str) -> None:
        import requests  # noqa: PLC0415
        r = requests.post(self.webhook_url, timeout=self.timeout, json={'text': f'```{text}```'})
        if r.status_code >= 400:
            raise RuntimeError(f'{r.status_code} {r.text[:200]}')


class DiscordAction(MessengerAction):
    """Discord channel webhook."""
    name = 'discord'
    MAX_LEN = 1900

    def __init__(self, webhook_url: Optional[str] = None, **kw):
        super().__init__(**kw)
        self.webhook_url = env_key('DISCORD_WEBHOOK_URL', explicit=webhook_url)
        if not self._sender and not self.configured:
            log.warning('discord: DISCORD_WEBHOOK_URL not set; messages will only be logged')

    @property
    def configured(self) -> bool:
        return bool(self.webhook_url)

    def _deliver(self, text: str) -> None:
        import requests  # noqa: PLC0415
        r = requests.post(self.webhook_url, timeout=self.timeout, json={'content': f'```{text}```'})
        if r.status_code >= 400:
            raise RuntimeError(f'{r.status_code} {r.text[:200]}')


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

    def on_signal(self, report):
        if 'signal' in self.events:
            self.post(report.to_dict())

    def on_entry(self, position):
        if 'entry' in self.events:
            self.post({'event': 'entry', 'position': position.to_dict()})

    def on_exit(self, trade):
        if 'exit' in self.events:
            self.post({'event': 'exit', 'trade': json.loads(json.dumps(trade.to_dict()))})


def build_report(signal: Signal, c: Classification, *, now: datetime, reference_price: Optional[float],
                 max_chase_pct: Optional[float], equity: Optional[float], qty: Optional[float],
                 sources: List[str], n_items: int, evidence_lines: List[str], url: str = '',
                 market: str = '', market_tz: str = 'UTC', mode: str = 'notify-only') -> SignalReport:
    ref = reference_price
    limit = ref * (1 + max_chase_pct) if ref and max_chase_pct is not None else ref
    target = signal.target_price(ref) if ref else None
    stop = signal.stop_price(ref) if ref else None
    notional = qty * ref if qty and ref else None
    risk = qty * ref * signal.stop_pct if qty and ref else None
    return SignalReport(signal=signal, classification=c, now=now, reference_price=ref, entry_limit=limit,
                        target_price=target, stop_price=stop, exit_by=now + timedelta(days=signal.max_hold_days),
                        suggested_qty=qty or None, notional=notional, risk_amount=risk, sources=sources,
                        n_items=n_items, evidence_lines=evidence_lines, url=url, market=market, market_tz=market_tz,
                        mode=mode)

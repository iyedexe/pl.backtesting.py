"""Telegram delivery: message formatting and the Bot API call.

Messages use Telegram HTML parse mode; all dynamic strings are escaped. In
``--dry-run`` (or with no credentials configured) messages are printed to
stdout instead of sent, so the whole pipeline can be exercised safely.
"""
from __future__ import annotations

import html

DISCLAIMER = '⚠️ Educational signal, not investment advice.'


def _n(x: float | None, digits: int = 2) -> str:
    return 'n/a' if x is None else f'{x:,.{digits}f}'


def _pct(x: float) -> str:
    return f'{x:+.1%}'


def format_signal(sig) -> str:
    e = html.escape
    if sig.action == 'INFO':
        return (f'ℹ️ <b>{e(sig.index_name)}</b> {e(sig.review_label)}\n'
                + '\n'.join(e(n) for n in sig.notes))

    if sig.action == 'SELL':
        move = (_pct(sig.price / sig.entry_price - 1)
                if sig.price and sig.entry_price else 'n/a')
        lines = [
            f'🔴 <b>SELL — {e(sig.symbol)}</b> ({e(sig.name)}) · {e(sig.index_name)}',
            f'Reason: {e(sig.reason)}',
            f'Entry {_n(sig.entry_price)} {e(sig.currency)} ({e(sig.entry_date)}) '
            f'→ now {_n(sig.price)} {e(sig.currency)} ({move})',
        ]
        if sig.realized_pnl is not None:
            lines.append(f'Theoretical P&amp;L on {_n(sig.notional, 0)}: '
                         f'<b>{sig.realized_pnl:+,.0f}</b> (before costs)')
        return '\n'.join(lines + [DISCLAIMER])

    # BUY
    what = {'predicted': 'predicted addition (ahead of the cutoff)',
            'projected': 'projected addition (ranks locked at cutoff)',
            'watchlist': 'newly eligible - inclusion watchlist'}[sig.kind]
    lines = [f'🟢 <b>BUY — {e(sig.symbol)}</b> ({e(sig.name)})',
             f'<b>{e(sig.index_name)}</b> · {e(what)}']
    if sig.rank is not None:
        lines.append(f'Rank <b>{sig.rank}</b>, inside the ≤{sig.band} entry band')
    if sig.cutoff:
        lines.append(f'Review cutoff {sig.cutoff} · effective after close {sig.effective}')
    lines.append(f'Entry: <b>{_n(sig.price)} {e(sig.currency)}</b> (last)')
    if sig.expected_exit_date:
        lines.append(f'Expected exit: <b>{_n(sig.expected_exit_price)} '
                     f'{e(sig.currency)}</b> around {sig.expected_exit_date}')
    else:
        lines.append(f'Expected exit: <b>{_n(sig.expected_exit_price)} '
                     f'{e(sig.currency)}</b> on inclusion (timing at the '
                     f'committee\'s discretion)')
    lines.append(f'Assumed index effect {_pct(sig.effect)} → theoretical P&amp;L '
                 f'on {_n(sig.notional, 0)}: <b>{sig.theoretical_pnl:+,.0f}</b> '
                 f'(before costs)')
    lines.append(f'Confidence: {e(sig.confidence)}')
    lines += [e(n) for n in sig.notes]
    return '\n'.join(lines + [DISCLAIMER])


def send(text: str, token: str, chat_id: str, dry_run: bool = False) -> bool:
    if dry_run or not (token and chat_id):
        print('--- telegram message (dry-run) ---')
        print(text)
        print()
        return True
    import requests
    r = requests.post(
        f'https://api.telegram.org/bot{token}/sendMessage',
        json={'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML',
              'disable_web_page_preview': True},
        timeout=30)
    ok = r.ok and r.json().get('ok', False)
    if not ok:
        print(f'Telegram send failed: {r.status_code} {r.text[:300]}')
    return ok

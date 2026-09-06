"""Persistent bot state: open signal positions, dedupe keys, closed history.

Plain JSON on disk, written atomically. The state is what turns a stateless
daily scan into a strategy: it remembers which signals were already sent,
which theoretical positions are open (so sells reference their entries), and
what got closed and why.
"""
from __future__ import annotations

import json
import os

EMPTY = {'open': {}, 'sent': [], 'closed': []}


def load(path: str) -> dict:
    if not os.path.exists(path):
        return {k: (dict(v) if isinstance(v, dict) else list(v))
                for k, v in EMPTY.items()}
    with open(path) as f:
        state = json.load(f)
    for key, default in EMPTY.items():
        state.setdefault(key, dict(default) if isinstance(default, dict) else list(default))
    return state


def save(state: dict, path: str):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)


def open_positions(state: dict, index_key: str) -> dict:
    return {pid: pos for pid, pos in state['open'].items()
            if pos['index'] == index_key}


def was_sent(state: dict, key: str) -> bool:
    return key in state['sent']


def mark_sent(state: dict, key: str):
    if key not in state['sent']:
        state['sent'].append(key)


def unmark_sent(state: dict, key: str):
    if key in state['sent']:
        state['sent'].remove(key)

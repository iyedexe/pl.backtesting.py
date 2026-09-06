"""
Index-inclusion signal bot.

Watches major US and European index rulebooks for stocks that are about to
become eligible for inclusion (the "index effect" anticipation trade from
``doc/examples/Index Inclusion Strategy.py``) and pushes buy/sell signals
to a Telegram chat, with entry price, expected exit price/date, and the
theoretical P&L on a configured notional.

Educational software. Signals are rule-based screens over public data, the
"expected" numbers are configurable assumptions, and nothing here is
investment advice.
"""

__version__ = '0.1.0'

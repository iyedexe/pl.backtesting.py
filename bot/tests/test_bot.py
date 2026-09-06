"""Offline tests for the inclusion bot: calendars against known real-world
dates, the screener's full signal lifecycle on fixture data, Wikipedia table
parsing, and message formatting. No network access required.

Run from the repo root:  python -m unittest discover -s bot/tests -v
"""
import datetime as dt
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inclusion_bot import notify, screener, state as st  # noqa: E402
from inclusion_bot.calendars import (  # noqa: E402
    active_review, bdays_between, dax40_reviews, ftse100_reviews,
    ndx100_reviews, next_bday, prev_bday)
from inclusion_bot.config import BotConfig, FTSE100, SPX500  # noqa: E402
from inclusion_bot.universe import _parse_constituents, to_yahoo  # noqa: E402


def review_at(reviews, cutoff):
    return next(r for r in reviews if r.cutoff == cutoff)


class TestCalendars(unittest.TestCase):

    def test_ftse_december_2025_matches_real_announcement(self):
        # LSEG's December 2025 UK Index Series review was announced Wed Dec 3.
        rv = review_at(ftse100_reviews([2025]), dt.date(2025, 12, 2))
        self.assertEqual(rv.cutoff.weekday(), 1)                # Tuesday
        self.assertEqual(rv.announce, dt.date(2025, 12, 3))
        self.assertEqual(rv.effective, dt.date(2025, 12, 19))   # 3rd Friday
        self.assertEqual(rv.exit_date, dt.date(2025, 12, 22))   # next Monday

    def test_ftse_september_2026(self):
        rv = review_at(ftse100_reviews([2026]), dt.date(2026, 9, 1))
        self.assertEqual(rv.effective, dt.date(2026, 9, 18))
        self.assertEqual(rv.threshold, 90)

    def test_ndx_annual_2024_matches_real_dates(self):
        # Palantir/MicroStrategy/Axon: announced Fri Dec 13 2024, effective
        # before the open on Mon Dec 23 (3rd Friday was Dec 20).
        rv = next(r for r in ndx100_reviews([2024])
                  if r.label == 'annual reconstitution')
        self.assertEqual(rv.cutoff, dt.date(2024, 11, 29))
        self.assertEqual(rv.announce, dt.date(2024, 12, 13))
        self.assertEqual(rv.effective, dt.date(2024, 12, 20))
        self.assertEqual(rv.exit_date, dt.date(2024, 12, 23))

    def test_dax_thresholds_by_month(self):
        reviews = dax40_reviews([2026])
        march = next(r for r in reviews if r.effective.month == 3)
        june = next(r for r in reviews if r.effective.month == 6)
        self.assertEqual((march.threshold, march.label),
                         (40, 'regular review'))
        self.assertEqual((june.threshold, june.label),
                         (33, 'fast-entry check'))
        self.assertEqual(march.cutoff, dt.date(2026, 2, 27))  # last Feb bday

    def test_phases(self):
        rv = review_at(ftse100_reviews([2026]), dt.date(2026, 9, 1))
        self.assertEqual(rv.phase(dt.date(2026, 8, 25), 10), 'pre_cutoff')
        self.assertIsNone(rv.phase(dt.date(2026, 7, 1), 10))
        self.assertEqual(rv.phase(rv.cutoff, 10), 'post_cutoff')
        self.assertEqual(rv.phase(dt.date(2026, 9, 17), 10), 'post_cutoff')
        self.assertEqual(rv.phase(rv.effective, 10), 'effective')
        self.assertIsNone(rv.phase(rv.exit_date, 10))

    def test_active_review_finds_window(self):
        found = active_review('FTSE100', dt.date(2026, 8, 25), 10)
        self.assertIsNotNone(found)
        review, phase = found
        self.assertEqual(review.cutoff, dt.date(2026, 9, 1))
        self.assertEqual(phase, 'pre_cutoff')
        self.assertIsNone(active_review('SPX500', dt.date(2026, 8, 25), 10))

    def test_bday_helpers(self):
        friday = dt.date(2026, 9, 18)
        self.assertEqual(next_bday(friday), dt.date(2026, 9, 21))
        self.assertEqual(prev_bday(dt.date(2026, 9, 21)), friday)
        self.assertEqual(bdays_between(dt.date(2026, 9, 14), friday), 4)


def fixture_market(n=130, candidate_rank=86, seed=7):
    """A FTSE-like universe: members are the top 100 by cap, except the stock
    at `candidate_rank` (a non-member inside the entry band - our candidate)
    who displaced the member now sitting at rank 113."""
    rng = np.random.default_rng(seed)
    caps = np.sort(rng.lognormal(23, 1, n))[::-1]
    market = pd.DataFrame({
        'price': rng.uniform(100, 2000, n).round(1),
        'cap': caps, 'rank_cap': caps,
        'name': [f'Fixture & Co {i}' for i in range(n)],
        'currency': 'GBp', 'exchange': 'LSE',
    }, index=[f'FIX{i:03d}.L' for i in range(n)])
    members = set(market.index[:100])
    candidate = market.index[candidate_rank - 1]
    members.discard(candidate)
    members.add(market.index[112])
    return market, members, candidate


class TestScreener(unittest.TestCase):

    def setUp(self):
        self.cfg = BotConfig()
        self.idx = FTSE100
        self.review = review_at(ftse100_reviews([2026]), dt.date(2026, 9, 1))
        self.market, self.members, self.candidate = fixture_market()
        self.state = st.load('/nonexistent')

    def scan(self, day):
        return screener.scan_index(self.idx, self.cfg, self.market,
                                   self.members, day, self.state)

    def test_full_buy_info_sell_cycle(self):
        pre = prev_bday(self.review.cutoff, 3)
        buys = self.scan(pre)
        self.assertEqual([s.action for s in buys], ['BUY'])
        buy = buys[0]
        self.assertEqual((buy.symbol, buy.kind, buy.rank),
                         (self.candidate, 'predicted', 86))
        self.assertEqual(buy.band, 88)                       # 90 - buffer 2
        self.assertEqual(buy.expected_exit_date, dt.date(2026, 9, 21))
        self.assertAlmostEqual(buy.expected_exit_price, buy.price * 1.02)
        self.assertAlmostEqual(buy.theoretical_pnl,
                               self.cfg.notional * self.idx.expected_effect)
        self.assertEqual(len(self.state['open']), 1)

        self.assertEqual(self.scan(next_bday(pre)), [])      # no duplicates

        infos = self.scan(self.review.cutoff)                # ranks locked
        self.assertEqual([s.action for s in infos], ['INFO'])
        self.assertIn(self.candidate, infos[0].notes[0])
        self.assertEqual(self.scan(self.review.cutoff), [])  # info deduped

        self.market.loc[self.candidate, 'price'] *= 1.03     # nice run-up
        sells = self.scan(self.review.exit_date)
        self.assertEqual([s.action for s in sells], ['SELL'])
        self.assertIn('Effective day has passed', sells[0].reason)
        self.assertAlmostEqual(sells[0].realized_pnl,
                               self.cfg.notional * .03, delta=1)
        self.assertEqual(self.state['open'], {})

    def test_candidacy_failure_exit(self):
        pre = prev_bday(self.review.cutoff, 8)
        self.scan(pre)
        # candidate's cap collapses -> rank beyond fail_rank, before cutoff
        self.market.loc[self.candidate, ['cap', 'rank_cap']] = \
            self.market['cap'].iloc[-1] * .5
        sells = self.scan(prev_bday(self.review.cutoff, 2))
        # the freed slot has no other in-band non-member to fill it, so the
        # scan yields exactly the one SELL
        self.assertEqual([s.action for s in sells], ['SELL'])
        self.assertIn('Candidacy failed', sells[0].reason)
        self.assertEqual(self.state['open'], {})

    def test_one_position_at_a_time(self):
        # two non-members inside the band: only the better-ranked is bought
        second = self.market.index[87]
        self.members.discard(second)
        buys = self.scan(prev_bday(self.review.cutoff, 3))
        self.assertEqual(len([s for s in buys if s.action == 'BUY']), 1)
        self.assertEqual(buys[0].rank, 86)

    def test_max_hold_backstop(self):
        pre = prev_bday(self.review.cutoff, 3)
        self.scan(pre)
        pos = next(iter(self.state['open'].values()))
        pos['entry_date'] = str(prev_bday(pre, 30))          # pretend it's old
        pos['expected_exit_date'] = str(dt.date(2099, 1, 1))
        sells = self.scan(next_bday(pre))
        self.assertEqual([s.action for s in sells], ['SELL'])
        self.assertIn('Maximum holding period', sells[0].reason)


class TestWatchlist(unittest.TestCase):

    def setUp(self):
        self.cfg = BotConfig()
        self.idx = SPX500
        caps = [30e9, 25e9, 20e9, 10e9]
        self.market = pd.DataFrame({
            'price': [100., 200., 300., 400.],
            'cap': caps, 'rank_cap': caps,
            'name': ['Alpha', 'Beta', 'Gamma', 'Delta'],
            'currency': 'USD', 'exchange': 'NMS',
        }, index=['AAA', 'BBB', 'CCC', 'DDD'])
        self.members = {'AAA'}          # BBB (25B) is the eligible non-member
        self.state = st.load('/nonexistent')

    def scan(self, day, profit=lambda s: True):
        return screener.scan_index(self.idx, self.cfg, self.market,
                                   self.members, day, self.state, profit)

    def test_watchlist_cycle_and_resignal(self):
        day = dt.date(2026, 8, 3)
        buys = self.scan(day)
        self.assertEqual([(s.action, s.symbol, s.kind) for s in buys],
                         [('BUY', 'BBB', 'watchlist')])
        self.assertIsNone(buys[0].expected_exit_date)
        self.assertAlmostEqual(buys[0].expected_exit_price, 200 * 1.005)

        # cap falls below the threshold -> SELL, and the key is re-armed
        self.market.loc['BBB', ['cap', 'rank_cap']] = 20e9
        sells = self.scan(next_bday(day))
        self.assertIn('below the market-cap', sells[0].reason)
        self.market.loc['BBB', ['cap', 'rank_cap']] = 26e9
        again = self.scan(next_bday(day, 2))
        self.assertEqual([s.symbol for s in again if s.action == 'BUY'],
                         ['BBB'])

    def test_profitability_gate(self):
        self.assertEqual(self.scan(dt.date(2026, 8, 3),
                                   profit=lambda s: False), [])
        buys = self.scan(dt.date(2026, 8, 4), profit=lambda s: None)
        self.assertTrue(any('could not be verified' in n
                            for n in buys[0].notes))


class TestUniverseParsing(unittest.TestCase):

    def test_to_yahoo(self):
        self.assertEqual(to_yahoo('BRK.B', ''), 'BRK-B')
        self.assertEqual(to_yahoo('AZN', '.L'), 'AZN.L')
        self.assertEqual(to_yahoo('AIR.DE', '.DE'), 'AIR.DE')
        self.assertEqual(to_yahoo('shel[1]', '.L'), 'SHEL.L')

    def test_parse_constituents_picks_ticker_table(self):
        rows = ''.join(f'<tr><td>Company {i}</td><td>T{i:02d}</td>'
                       f'<td>Industrials</td></tr>' for i in range(15))
        html = ('<table><tr><th>Year</th></tr><tr><td>1999</td></tr></table>'
                '<table><tr><th>Company</th><th>Ticker</th>'
                f'<th>GICS Sector</th></tr>{rows}</table>')
        parsed = _parse_constituents(html)
        self.assertEqual(len(parsed), 15)
        self.assertEqual(parsed[0], {'ticker': 'T00', 'name': 'Company 0',
                                     'sector': 'Industrials'})


class TestNotify(unittest.TestCase):

    def test_buy_message_contents_and_escaping(self):
        market, members, candidate = fixture_market()
        cfg, state = BotConfig(), st.load('/nonexistent')
        review = review_at(ftse100_reviews([2026]), dt.date(2026, 9, 1))
        buy = screener.scan_index(FTSE100, cfg, market, members,
                                  prev_bday(review.cutoff, 3), state)[0]
        text = notify.format_signal(buy)
        for needle in ('BUY', buy.symbol, 'Entry:', 'Expected exit:',
                       'theoretical P&amp;L', '+200', 'Rank <b>86</b>',
                       str(review.exit_date), notify.DISCLAIMER,
                       'Fixture &amp; Co'):
            self.assertIn(needle, text)
        self.assertTrue(notify.send(text, '', '', dry_run=True))


if __name__ == '__main__':
    unittest.main()

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.test import GOOG

from newsbot import (
    BotState, ClosedTrade, ExitReason, FileNewsSource, NewsItem, NewsStrategy, NewsTradingBot, PaperBroker,
    RuleClassifier, SignalEngine, SimClock, StaticPriceFeed, align_news_scores, build_bot, merge_config,
    run_news_backtest,
)
from newsbot.brokers import BrokerError
from newsbot.clock import is_regular_session, next_session_open
from newsbot.prices import CSVPriceFeed
from newsbot.replay import run_replay
from newsbot.sources import TickerExtractor, parse_rss

SAMPLE_NEWS = Path(__file__).parent.parent / 'data' / 'sample_news.json'
T0 = datetime(2024, 3, 4, 14, 35, tzinfo=timezone.utc)   # a Monday, 09:35 New York


def item(headline, tickers=('ACME',), published=None, summary='', id=None):
    published = published or T0 - timedelta(minutes=1)
    return NewsItem(id=id or headline, headline=headline, published=published, tickers=list(tickers), summary=summary)


class TestRuleClassifier(unittest.TestCase):
    def setUp(self):
        self.c = RuleClassifier()

    def score(self, text):
        return self.c.classify_text(text)

    def test_bullish(self):
        for text, cat in [
            ('Acme beats estimates, raises FY guidance', 'guidance_raise'),
            ('Acme posts record quarterly revenue', 'earnings_beat'),
            ('FDA approves Acme drug for lung cancer', 'fda_approval'),
            ('Acme to be acquired by Globex for $5 billion', 'acquisition_target'),
            ('Acme upgraded to Buy at Goldman', 'analyst_upgrade'),
            ('Acme to join S&P 500', 'index_inclusion'),
        ]:
            c = self.score(text)
            self.assertGreater(c.score, 0, text)
            self.assertEqual(c.category, cat, text)

    def test_bearish(self):
        for text in ['Acme misses estimates', 'Acme cuts guidance', 'Acme prices $100 million public offering',
                     'Acme downgraded to Sell', 'Acme trial fails to meet primary endpoint',
                     'Acme files for Chapter 11', 'SEC investigates Acme']:
            self.assertLess(self.score(text).score, 0, text)

    def test_mixed_nets_negative(self):
        c = self.score('Acme beats estimates but cuts full-year guidance')
        self.assertLess(c.score, 0)
        self.assertEqual(c.category, 'guidance_cut')

    def test_neutral(self):
        c = self.score('Acme to present at investor conference')
        self.assertEqual(c.score, 0)
        self.assertEqual(c.category, 'none')

    def test_clamped(self):
        c = self.score('Acme beats estimates, record revenue, raises guidance, FDA approval, buyback and dividend hike')
        self.assertEqual(c.score, 1.0)


class TestSignalEngine(unittest.TestCase):
    def test_signal_geometry(self):
        eng = SignalEngine(min_score=0.5, target_pct=0.04, stop_pct=0.02, max_hold_days=7,
                           signal_ttl=timedelta(hours=18), scale_target_by_score=False)
        sigs = eng.evaluate(item('Acme beats estimates and raises guidance'), now=T0)
        self.assertEqual(len(sigs), 1)
        s = sigs[0]
        self.assertEqual(s.ticker, 'ACME')
        self.assertAlmostEqual(s.target_price(100), 104)
        self.assertAlmostEqual(s.stop_price(100), 98)
        self.assertEqual(s.expires_at, T0 + timedelta(hours=18))
        self.assertEqual(s.max_hold_days, 7)

    def test_target_scales_with_score(self):
        eng = SignalEngine(target_pct=0.04)
        self.assertAlmostEqual(eng.effective_target_pct(0.5), 0.04)
        self.assertAlmostEqual(eng.effective_target_pct(1.0), 0.05)

    def test_filters(self):
        eng = SignalEngine(min_score=0.5, universe=['MSFT'])
        self.assertEqual(eng.evaluate(item('Acme beats estimates and raises guidance')), [])
        self.assertEqual(len(eng.evaluate(item('MSFT beats estimates and raises guidance', tickers=['MSFT']))), 1)
        self.assertEqual(eng.evaluate(item('Acme misses estimates')), [])
        eng = SignalEngine(categories=['fda_approval'])
        self.assertEqual(eng.evaluate(item('Acme beats estimates and raises guidance')), [])
        eng = SignalEngine(blocked_categories=['guidance_raise'])
        self.assertEqual(eng.evaluate(item('Acme beats estimates and raises guidance')), [])
        two = SignalEngine().evaluate(item('Acme beats estimates and raises guidance', tickers=['A', 'B']))
        self.assertEqual([s.ticker for s in two], ['A', 'B'])


class TestPaperBroker(unittest.TestCase):
    def test_round_trip(self):
        clock = SimClock(T0)
        feed = StaticPriceFeed({'ACME': 100})
        b = PaperBroker(feed, cash=10_000, clock=clock, commission=1.0, always_open=True)
        fill = b.buy('ACME', 50)
        self.assertEqual(fill.price, 100)
        self.assertAlmostEqual(b.cash(), 10_000 - 5_000 - 1)
        feed.set('ACME', 110)
        self.assertAlmostEqual(b.equity(), 4_999 + 5_500)
        with self.assertRaises(BrokerError):
            b.buy('ACME', 1000)
        with self.assertRaises(BrokerError):
            b.sell('ACME', 60)
        fill = b.sell('ACME', 50)
        self.assertEqual(fill.price, 110)
        self.assertAlmostEqual(b.cash(), 4_999 + 5_500 - 1)
        self.assertEqual(b.positions(), {})

    def test_market_hours(self):
        b = PaperBroker(StaticPriceFeed(), clock=SimClock(T0))
        self.assertTrue(b.is_market_open(T0))
        self.assertFalse(b.is_market_open(T0 + timedelta(hours=8)))        # after close
        self.assertFalse(b.is_market_open(T0 + timedelta(days=5)))         # Saturday
        self.assertTrue(is_regular_session(T0))
        self.assertEqual(next_session_open(T0 + timedelta(days=4, hours=8)).weekday(), 0)  # Fri evening -> Monday


class _BotFixture:
    def make(self, cash=100_000, **kw):
        self.clock = SimClock(T0)
        self.feed = StaticPriceFeed({'ACME': 100.0, 'BETA': 50.0})
        self.broker = PaperBroker(self.feed, cash=cash, clock=self.clock)
        self.source = FileNewsSource(None, items=kw.pop('items', []))
        self.engine = SignalEngine(min_score=0.5, target_pct=0.05, stop_pct=0.03, max_hold_days=7,
                                   signal_ttl=timedelta(hours=18), scale_target_by_score=False)
        params = dict(sources=[self.source], engine=self.engine, broker=self.broker, price_feed=self.feed,
                      clock=self.clock, max_positions=2, position_size_pct=0.2, max_news_age=timedelta(minutes=30))
        params.update(kw)
        self.bot = NewsTradingBot(**params)
        return self.bot


class TestBot(_BotFixture, unittest.TestCase):
    def test_entry_then_target_exit(self):
        bot = self.make(items=[item('Acme beats estimates and raises guidance')])
        bot.tick()
        self.assertIn('ACME', bot.state.positions)
        pos = bot.state.positions['ACME']
        self.assertEqual(pos.qty, 200)                       # 20% of 100k / $100
        self.assertAlmostEqual(pos.target_price, 105)
        self.assertAlmostEqual(pos.stop_price, 97)
        self.assertEqual(pos.exit_by, T0 + timedelta(days=7))
        self.feed.set('ACME', 104.9)
        self.clock.advance(minutes=1)
        bot.tick()
        self.assertIn('ACME', bot.state.positions)
        self.feed.set('ACME', 105.2)
        bot.tick()
        self.assertNotIn('ACME', bot.state.positions)
        self.assertEqual(bot.state.closed[0].reason, ExitReason.TARGET.value)
        self.assertAlmostEqual(bot.state.closed[0].pnl, 200 * 5.2)
        self.assertEqual(bot.summary()['exits'], {'target': 1})

    def test_stop_exit(self):
        bot = self.make(items=[item('Acme beats estimates and raises guidance')])
        bot.tick()
        self.feed.set('ACME', 96.5)
        bot.tick()
        self.assertEqual(bot.state.closed[0].reason, 'stop')

    def test_time_exit_one_week(self):
        bot = self.make(items=[item('Acme beats estimates and raises guidance')])
        bot.tick()
        self.feed.set('ACME', 101)
        self.clock.set(T0 + timedelta(days=6, hours=23))
        bot.tick()
        self.assertIn('ACME', bot.state.positions)
        self.clock.set(T0 + timedelta(days=7))                # same weekday one week later, market open
        bot.tick()
        self.assertEqual(bot.state.closed[0].reason, 'time')
        self.assertEqual(bot.state.closed[0].exit_time, T0 + timedelta(days=7))

    def test_after_hours_news_waits_for_open(self):
        after_close = T0.replace(hour=21, minute=5)           # 16:05 New York, Monday
        self.make(items=[item('Acme beats estimates and raises guidance', published=after_close)])
        self.clock.set(after_close + timedelta(minutes=2))
        self.bot.tick()
        self.assertEqual(len(self.bot.state.pending), 1)
        self.assertEqual(self.bot.state.positions, {})
        self.clock.set(after_close + timedelta(hours=17))     # 09:05 NY next day: still closed
        self.bot.tick()
        self.assertEqual(self.bot.state.positions, {})
        self.clock.set(after_close + timedelta(hours=17, minutes=30))
        self.bot.tick()
        self.assertIn('ACME', self.bot.state.positions)
        self.assertEqual(self.bot.state.pending, {})

    def test_signal_expiry_and_stale_news(self):
        after_close = T0.replace(hour=21, minute=5)
        self.make(items=[item('Acme beats estimates and raises guidance', published=after_close)])
        self.clock.set(after_close + timedelta(minutes=2))
        self.bot.tick()
        self.clock.set(after_close + timedelta(hours=19))     # past the 18h TTL (Tuesday 11:05 NY, market open)
        self.bot.tick()
        self.assertEqual(self.bot.state.pending, {})
        self.assertEqual(self.bot.state.positions, {})
        # stale news (older than max_news_age) is marked seen but never traded
        stale = item('Acme posts record revenue', published=T0 - timedelta(hours=2), id='stale')
        self.bot.ingest([stale], now=T0)
        self.assertTrue(self.bot.state.has_seen('stale'))
        self.assertEqual(self.bot.state.pending, {})

    def test_dedupe_and_position_limits(self):
        items = [item('Acme beats estimates and raises guidance', id='a1'),
                 item('Acme beats estimates and raises guidance', id='a1'),   # duplicate id
                 item('Acme raises dividend and announces buyback', id='a2'),  # already holding ACME
                 item('Beta wins $300 million contract award', tickers=['BETA'], id='b1'),
                 item('Gamma FDA approval granted', tickers=['GAMA'], id='g1')]
        bot = self.make(items=items)
        self.feed.set('GAMA', 10)
        bot.tick()
        self.assertEqual(set(bot.state.positions), {'ACME', 'GAMA'})   # max_positions=2, highest scores first
        self.assertEqual([s.ticker for s in bot.state.pending.values()], ['BETA'])

    def test_chase_protection_and_risk_sizing(self):
        bot = self.make(items=[item('Acme beats estimates and raises guidance')], max_chase_pct=0.02,
                        risk_per_trade_pct=0.005)
        bot.poll_news()
        self.feed.set('ACME', 103)
        bot.execute_signals()
        self.assertEqual(bot.state.positions, {})               # ran away: skipped
        bot2 = self.make(items=[item('Acme beats estimates and raises guidance')], risk_per_trade_pct=0.005)
        bot2.tick()
        # risk cap: 0.5% of 100k = $500 / (100 * 3%) = 166 shares < 200 from notional cap
        self.assertEqual(bot2.state.positions['ACME'].qty, 166)

    def test_state_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'state.json')
            bot = self.make(items=[item('Acme beats estimates and raises guidance')], state=BotState(path))
            bot.tick()
            self.assertTrue(os.path.exists(path))
            restored = BotState(path)
            self.assertIn('ACME', restored.positions)
            self.assertEqual(restored.positions['ACME'].exit_by, T0 + timedelta(days=7))
            self.assertTrue(restored.has_seen('Acme beats estimates and raises guidance'))
            with open(path) as f:
                self.assertEqual(json.load(f)['positions'][0]['ticker'], 'ACME')
            t = ClosedTrade('ACME', 1, 100, T0, 105, T0, 'target')
            self.assertEqual(ClosedTrade.from_dict(t.to_dict()), t)


class TestSources(unittest.TestCase):
    RSS = '''<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>
      <item><title>Acme Corp (NASDAQ: ACME) Beats Estimates</title><link>http://x/1</link>
        <guid>1</guid><pubDate>Mon, 04 Mar 2024 14:30:00 GMT</pubDate>
        <description>&lt;p&gt;Revenue rose 20%.&lt;/p&gt;</description></item>
      <item><title>No date</title></item>
    </channel></rss>'''

    def test_parse_rss(self):
        items = parse_rss(self.RSS, source='t')
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].published, datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc))
        self.assertEqual(items[0].summary, 'Revenue rose 20%.')
        self.assertEqual(items[0].id, 't:1')

    def test_ticker_extractor(self):
        ex = TickerExtractor(known=['MSFT'])
        self.assertEqual(ex.extract('Acme (NASDAQ: ACME) and $TSLA, MSFT and (NYSE:BRK.B) again $TSLA'),
                         ['ACME', 'BRK.B', 'TSLA', 'MSFT'])
        self.assertEqual(TickerExtractor().extract('nothing here'), [])

    def test_file_source(self):
        src = FileNewsSource(SAMPLE_NEWS)
        self.assertGreater(len(src.items), 30)
        self.assertTrue(all(i.tickers == ['GOOG'] for i in src.items))
        self.assertEqual(src.fetch(since=src.items[-1].published), [])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'n.csv')
            with open(p, 'w') as f:
                f.write('published,ticker,headline\n2024-03-04T14:00:00Z,ACME,Acme beats estimates\n')
            self.assertEqual(FileNewsSource(p).items[0].tickers, ['ACME'])

    def test_news_item_from_dict_variants(self):
        it = NewsItem.from_dict({'title': 'x', 'date': '2024-03-04', 'symbols': 'a, b'})
        self.assertEqual(it.tickers, ['A', 'B'])
        self.assertEqual(it.published.tzinfo, timezone.utc)


class TestPriceFeeds(unittest.TestCase):
    def test_csv_feed_is_clock_aware(self):
        clock = SimClock('2005-01-01')
        feed = CSVPriceFeed({'GOOG': GOOG}, clock)
        self.assertEqual(feed.price('GOOG'), float(GOOG.loc[:'2005-01-01', 'Close'].iloc[-1]))
        clock.set('2004-01-01')
        self.assertIsNone(feed.price('GOOG'))
        self.assertIsNone(feed.price('NOPE'))


class TestBacktest(unittest.TestCase):
    def test_align_scores(self):
        idx = pd.date_range('2024-03-04', periods=5, freq='D')
        d5 = datetime(2024, 3, 5, 21, 5, tzinfo=timezone.utc)
        news = [item('Acme beats estimates', tickers=['ACME'], published=d5),
                item('Acme misses estimates', tickers=['ACME'], published=d5 + timedelta(hours=1), id='m'),
                item('Other beats estimates', tickers=['OTHR'], published=datetime(2024, 3, 6, tzinfo=timezone.utc)),
                item('Too early', tickers=['ACME'], published=datetime(2024, 3, 1, tzinfo=timezone.utc), id='e')]
        scores = align_news_scores(idx, news, ticker='ACME')
        self.assertEqual(list(np.nonzero(scores)[0]), [1])
        self.assertLess(scores[1], 0)                           # largest |score| on the bar wins
        series = pd.Series([0.9], index=[pd.Timestamp('2024-03-07 12:00')])
        self.assertEqual(list(align_news_scores(idx, series)), [0, 0, 0, 0.9, 0])

    def test_strategy_respects_week_and_exits(self):
        res = run_news_backtest({'GOOG': GOOG}, FileNewsSource(SAMPLE_NEWS).items, commission=0.0,
                                min_score=0.5, target_pct=0.05, stop_pct=0.03, hold_bars=5)
        trades = res['trades']
        self.assertGreater(len(trades), 10)
        self.assertLessEqual((trades.ExitBar - trades.EntryBar).max(), 5)
        self.assertTrue((trades.Size > 0).all())                # long only
        # every exit is a stop (~-3%), a target (>= +3.75%) or a time exit in between
        self.assertGreaterEqual(trades.ReturnPct.min(), -0.03 - 0.06)   # allow gap through stop
        self.assertEqual(res['summary']['trades'], len(trades))
        self.assertIn('Return [%]', res['stats']['GOOG'])
        stats = res['stats']['GOOG']
        self.assertIsInstance(stats['_strategy'], NewsStrategy)

    def test_no_news_no_trades(self):
        res = run_news_backtest({'GOOG': GOOG}, [], finalize_trades=False)
        self.assertEqual(res['summary'], {'trades': 0})


class TestReplayAndConfig(unittest.TestCase):
    def test_replay_runs_live_pipeline(self):
        bot = run_replay({'GOOG': GOOG}, SAMPLE_NEWS, {'signals': {'min_score': 0.5}})
        closed = bot.state.closed
        self.assertGreater(len(closed), 10)
        for t in closed:
            self.assertLessEqual(t.exit_time - t.entry_time, timedelta(days=7))
            self.assertIn(t.reason, ('target', 'stop', 'time'))
        self.assertEqual(bot.state.positions, {})

    def test_build_bot_from_config(self):
        cfg = merge_config({'universe': ['ACME'], 'sources': [{'type': 'file', 'path': str(SAMPLE_NEWS)}],
                            'prices': {'type': 'static', 'prices': {'ACME': 10}},
                            'broker': {'type': 'paper', 'cash': 5000},
                            'bot': {'max_positions': 3, 'state_file': None}})
        bot = build_bot(cfg, clock=SimClock(T0))
        self.assertEqual(bot.max_positions, 3)
        self.assertEqual(bot.broker.cash(), 5000)
        self.assertEqual(bot.engine.universe, {'ACME'})
        with self.assertRaises(ValueError):
            build_bot(merge_config({'sources': [{'type': 'nope'}], 'prices': {'type': 'static'}}))


if __name__ == '__main__':
    unittest.main(verbosity=2)

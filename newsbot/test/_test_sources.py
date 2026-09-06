"""Tests for provider parsers, evidence aggregation, scorers, actions and the multi-source bot flow."""
import json
import unittest
from datetime import datetime, timedelta, timezone

from newsbot import (
    Bundle, CallbackAction, EvidenceStore, NewsItem, NewsTradingBot, PaperBroker, RuleScorer, SignalEngine, SimClock,
    StaticPriceFeed, TelegramAction, TradeAction, WebhookAction, build_bot, merge_config,
)
from newsbot.models import KIND_EARNINGS_RESULT, KIND_EARNINGS_UPCOMING, KIND_REGULATORY, KIND_SENTIMENT, KIND_SOCIAL
from newsbot.providers import (AlphaVantageNews, ClinicalTrialsSource, FinnhubEarningsCalendar, FinnhubNews,
                               FMPEarningsCalendar, MarketauxNews, NasdaqEarningsCalendar, NewsAPINews,
                               OpenFDAApprovals, PolygonNews, RedditMentions, StockTwitsStream)
from newsbot.scoring import ClaudeScorer
from newsbot.sources import FileNewsSource, TickerExtractor

T0 = datetime(2024, 3, 4, 14, 35, tzinfo=timezone.utc)   # Monday 09:35 New York
SINCE = T0 - timedelta(hours=24)


def item(headline, tickers=('ACME',), published=None, **kw):
    return NewsItem(id=kw.pop('id', headline), headline=headline, published=published or T0 - timedelta(minutes=1),
                    tickers=list(tickers), **kw)


class TestProviderParsers(unittest.TestCase):
    def test_finnhub_news(self):
        rows = [{'id': 1, 'headline': 'ACME beats estimates', 'datetime': int(T0.timestamp()), 'summary': 's',
                 'source': 'Reuters', 'url': 'http://x', 'category': 'company'},
                {'headline': '', 'datetime': 1}]
        items = FinnhubNews.parse(rows, 'ACME')
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, 'finnhub:1')
        self.assertEqual(items[0].published, T0)
        self.assertEqual(items[0].tickers, ['ACME'])
        self.assertTrue(items[0].is_trigger)

    def test_finnhub_calendar(self):
        data = {'earningsCalendar': [
            {'symbol': 'ACME', 'date': '2024-03-01', 'hour': 'amc', 'epsActual': 1.10, 'epsEstimate': 1.00,
             'revenueActual': 105, 'revenueEstimate': 100, 'quarter': 1, 'year': 2024},
            {'symbol': 'ACME', 'date': '2024-06-01', 'hour': 'bmo', 'epsActual': None, 'epsEstimate': 1.2},
            {'symbol': 'ZZZ', 'date': '2024-03-01', 'epsActual': 1},
        ]}
        items = FinnhubEarningsCalendar.parse(data, {'ACME'})
        self.assertEqual([i.kind for i in items], [KIND_EARNINGS_RESULT, KIND_EARNINGS_UPCOMING])
        res = items[0]
        self.assertEqual(res.meta['verdict'], 'beat')
        self.assertAlmostEqual(res.meta['eps_surprise_pct'], 10.0)
        self.assertAlmostEqual(res.meta['revenue_surprise_pct'], 5.0)
        self.assertEqual(res.published.hour, 21)
        self.assertIn('beat', res.headline)
        self.assertEqual(items[1].meta['report_date'], '2024-06-01')

    def test_alphavantage(self):
        data = {'feed': [{'title': 'ACME soars', 'url': 'http://a', 'time_published': '20240304T140000',
                          'summary': 's', 'source': 'Benzinga', 'overall_sentiment_score': 0.4,
                          'overall_sentiment_label': 'Bullish',
                          'ticker_sentiment': [{'ticker': 'ACME', 'relevance_score': '0.9',
                                                'ticker_sentiment_score': '0.55', 'ticker_sentiment_label': 'Bullish'},
                                               {'ticker': 'OTHR', 'relevance_score': '0.1',
                                                'ticker_sentiment_score': '0.9'}]},
                         {'title': 'bad date', 'time_published': 'nope'}]}
        items = AlphaVantageNews.parse(data, {'ACME', 'OTHR'})
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].kind, KIND_SENTIMENT)
        self.assertEqual(items[0].tickers, ['ACME'])                   # OTHR below relevance threshold
        self.assertEqual(items[0].meta['ticker_sentiment']['ACME']['sentiment'], 0.55)

    def test_polygon(self):
        data = {'results': [{'id': 'p1', 'title': 't', 'published_utc': '2024-03-04T14:00:00Z',
                             'tickers': ['ACME', 'ZZZ'],
                             'publisher': {'name': 'Benzinga'}, 'article_url': 'u', 'description': 'd',
                             'insights': [{'ticker': 'ACME', 'sentiment': 'positive'}]}]}
        items = PolygonNews.parse(data, {'ACME'})
        self.assertEqual(items[0].tickers, ['ACME'])
        self.assertEqual(items[0].meta['sentiment_labels'], {'ACME': 'positive'})
        self.assertEqual(items[0].source, 'polygon/Benzinga')

    def test_marketaux(self):
        data = {'data': [{'uuid': 'm1', 'title': 't', 'published_at': '2024-03-04T14:00:00.000000Z', 'url': 'u',
                          'source': 'x.com', 'description': 'd',
                          'entities': [{'symbol': 'ACME', 'sentiment_score': 0.7},
                                       {'symbol': 'ZZZ', 'sentiment_score': -1}]}]}
        items = MarketauxNews.parse(data, {'ACME'})
        self.assertEqual(items[0].meta['ticker_sentiment'], {'ACME': 0.7})

    def test_newsapi(self):
        data = {'articles': [{'title': 'Acme Corp beats estimates', 'description': 'x',
                              'publishedAt': '2024-03-04T14:00:00Z',
                              'url': 'u', 'source': {'name': 'CNBC'}},
                             {'title': 'Unrelated', 'publishedAt': '2024-03-04T14:00:00Z'}]}
        items = NewsAPINews.parse(data, TickerExtractor(aliases={'ACME': ['Acme Corp']}))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].tickers, ['ACME'])

    def test_fmp_calendar(self):
        rows = [{'symbol': 'ACME', 'date': '2024-03-01', 'epsActual': 0.9, 'epsEstimated': 1.0, 'time': 'bmo'},
                {'symbol': 'ACME', 'date': '2024-06-01', 'epsEstimated': 1.0, 'time': 'amc'}]
        items = FMPEarningsCalendar.parse(rows, set())
        self.assertEqual(items[0].meta['verdict'], 'miss')
        self.assertEqual(items[0].published.hour, 12)
        self.assertEqual(items[1].kind, KIND_EARNINGS_UPCOMING)

    def test_nasdaq_calendar(self):
        data = {'data': {'rows': [{'symbol': 'ACME', 'name': 'Acme', 'time': 'time-after-hours', 'epsForecast': '$1.25',
                                   'noOfEsts': '12'}]}}
        items = NasdaqEarningsCalendar.parse(data, '2024-03-07', {'ACME'})
        self.assertEqual(items[0].meta['hour'], 'amc')
        self.assertEqual(items[0].meta['eps_estimate'], 1.25)

    def test_openfda(self):
        data = {'results': [{'application_number': 'NDA1', 'sponsor_name': 'ACME PHARMA INC',
                             'products': [{'brand_name': 'CUREALL'}],
                             'submissions': [{'submission_status': 'AP', 'submission_status_date': '20240301',
                                              'submission_type': 'SUPPL', 'submission_number': '5',
                                              'submission_class_code_description': 'Efficacy'},
                                             {'submission_status': 'TA', 'submission_status_date': '20240201'}]}]}
        items = OpenFDAApprovals.parse(data, TickerExtractor(aliases={'ACME': ['Acme Pharma']}))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].kind, KIND_REGULATORY)
        self.assertIn('CUREALL', items[0].headline)

    def test_clinicaltrials(self):
        data = {'studies': [{'protocolSection': {
            'identificationModule': {'nctId': 'NCT1', 'briefTitle': 'Trial of X'},
            'statusModule': {'overallStatus': 'TERMINATED', 'lastUpdatePostDateStruct': {'date': '2024-03-01'},
                             'whyStopped': 'futility'},
            'designModule': {'phases': ['PHASE3']}}},
            {'protocolSection': {'identificationModule': {'nctId': 'NCT2'},
                                 'statusModule': {'overallStatus': 'RECRUITING'}}}]}
        items = ClinicalTrialsSource.parse(data, 'ACME')
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].meta['prior'], -0.6)
        self.assertIn('PHASE3', items[0].headline)

    def test_stocktwits_and_reddit(self):
        msgs = [{'id': i, 'body': f'msg {i}', 'created_at': (T0 - timedelta(minutes=i)).isoformat(),
                 'entities': {'sentiment': {'basic': 'Bullish' if i % 3 else 'Bearish'}}} for i in range(1, 10)]
        old = {'id': 99, 'body': 'old', 'created_at': '2020-01-01T00:00:00Z'}
        st = StockTwitsStream.parse({'messages': msgs + [old]}, 'ACME', SINCE)
        self.assertEqual(st.kind, KIND_SOCIAL)
        self.assertEqual(st.meta['posts'], 9)
        self.assertEqual(st.meta['bull'], 6)
        self.assertEqual(st.meta['bull_ratio'], 0.67)
        self.assertIsNone(StockTwitsStream.parse({'messages': []}, 'ACME', SINCE))
        posts = [{'id': 'a', 'title': 'ACME to the moon', 'created_utc': T0.timestamp(), 'score': 10,
                  'num_comments': 3}]
        rd = RedditMentions.parse(posts, 'ACME', SINCE)
        self.assertEqual(rd.meta, {'posts': 1, 'upvotes': 10, 'comments': 3})


class TestAggregator(unittest.TestCase):
    def make_store(self):
        store = EvidenceStore(window=timedelta(hours=24))
        store.add(item('Acme beats estimates and raises guidance', id='n1', source='finnhub/Reuters',
                       published=T0 - timedelta(hours=2)))
        store.add(item('Acme Q1 results', id='s1', kind=KIND_SENTIMENT, source='alphavantage',
                       meta={'ticker_sentiment': {'ACME': {'sentiment': 0.6}}}, published=T0 - timedelta(hours=1)))
        store.add(item('ACME reported EPS 1.1 vs 1.0', id='e1', kind=KIND_EARNINGS_RESULT, source='finnhub',
                       meta={'eps_surprise_pct': 10.0, 'revenue_surprise_pct': 3.0, 'verdict': 'beat'},
                       published=T0 - timedelta(hours=3)))
        store.add(item('StockTwits ACME', id='so1', kind=KIND_SOCIAL, source='stocktwits',
                       meta={'bull_ratio': 0.8, 'posts': 20}, published=T0 - timedelta(minutes=30)))
        store.add(item('Too old', id='old', published=T0 - timedelta(hours=30)))
        store.add(item('Other ticker', id='o1', tickers=['OTHR']))
        return store

    def test_bundle_and_features(self):
        store = self.make_store()
        b = store.bundle('ACME', T0)
        self.assertIsInstance(b, Bundle)
        self.assertEqual([i.id for i in b.items], ['e1', 'n1', 's1', 'so1'])   # window applied, chronological
        self.assertEqual(b.trigger.id, 'n1')                     # newest trigger (sentiment/social are not)
        f = b.features
        self.assertEqual(f['n_sources'], 3)                                     # finnhub, alphavantage, stocktwits
        self.assertGreater(f['trigger_rule']['score'], 0.9)
        self.assertEqual(f['sentiment_mean'], 0.6)
        self.assertEqual(f['earnings']['verdict'], 'beat')
        self.assertEqual(f['social']['bull_ratio'], 0.8)
        self.assertIn('*[news]', b.describe())
        self.assertIsNone(store.bundle('NOPE', T0))
        # persistence round trip
        store2 = EvidenceStore()
        store2.load_dict(json.loads(json.dumps(store.to_dict())))
        self.assertEqual([i.id for i in store2.items_for('ACME')], [i.id for i in store.items_for('ACME')])
        store.prune(T0)
        self.assertNotIn('old', [i.id for i in store.items_for('ACME')])

    def test_universe_filter(self):
        store = EvidenceStore(universe=['ACME'])
        self.assertEqual(store.add(item('x', tickers=['ACME', 'ZZZ'])), ['ACME'])


class TestScorers(unittest.TestCase):
    def test_rule_scorer_combines_sources(self):
        store = TestAggregator().make_store()
        b = store.bundle('ACME', T0)
        c = RuleScorer().score(b)
        self.assertGreaterEqual(c.score, 0.9)
        self.assertTrue(any('earnings surprise' in r for r in c.reasons))
        self.assertTrue(any('provider sentiment' in r for r in c.reasons))
        # a bearish miss drags a bullish headline down
        store = EvidenceStore()
        store.add(item('Acme posts record revenue', id='n', published=T0 - timedelta(hours=1)))
        store.add(item('miss', id='e', kind=KIND_EARNINGS_RESULT, published=T0 - timedelta(hours=2),
                       meta={'eps_surprise_pct': -12.0, 'revenue_surprise_pct': -4.0, 'verdict': 'miss'}))
        c2 = RuleScorer().score(store.bundle('ACME', T0))
        self.assertLess(c2.score, c.score - 0.5)
        self.assertEqual(c2.category, 'earnings_miss')

    def test_claude_scorer_with_fake_client_and_fallback(self):
        class Block:
            type = 'text'

            def __init__(self, text):
                self.text = text

        class Resp:
            stop_reason = 'end_turn'

            def __init__(self, text):
                self.content = [Block(text)]

        calls = []

        class Messages:
            def create(self, **kw):
                calls.append(kw)
                if kw.get('fail'):
                    raise RuntimeError('boom')
                return Resp(json.dumps({'score': 0.72, 'category': 'earnings_beat', 'confidence': 0.8,
                                        'reasons': ['beat and raise', 'sentiment positive']}))

        class Client:
            messages = Messages()

        b = TestAggregator().make_store().bundle('ACME', T0)
        sc = ClaudeScorer(client=Client(), model='claude-opus-5')
        c = sc.score(b)
        self.assertEqual((c.score, c.category, c.confidence), (0.72, 'earnings_beat', 0.8))
        self.assertEqual(calls[0]['model'], 'claude-opus-5')
        self.assertEqual(calls[0]['output_config']['format']['type'], 'json_schema')
        self.assertIn('Acme beats estimates', calls[0]['messages'][0]['content'])
        self.assertIn('"earnings"', calls[0]['messages'][0]['content'])

        class Broken:
            class messages:  # noqa: N801
                @staticmethod
                def create(**kw):
                    raise RuntimeError('api down')

        c2 = ClaudeScorer(client=Broken()).score(b)
        self.assertEqual(c2.reasons[0], 'fallback:rules')
        self.assertGreater(c2.score, 0.5)


class _Fixture:
    def make(self, actions=None, items=(), scorer=None, **kw):
        self.clock = SimClock(T0)
        self.feed = StaticPriceFeed({'ACME': 100.0, 'BETA': 50.0})
        self.broker = PaperBroker(self.feed, cash=100_000, clock=self.clock)
        self.source = FileNewsSource(None, items=list(items))
        engine = SignalEngine(min_score=0.5, target_pct=0.05, stop_pct=0.03, scale_target_by_score=False)
        params = dict(sources=[self.source], engine=engine, broker=self.broker, price_feed=self.feed, clock=self.clock,
                      max_positions=2, actions=actions, scorer=scorer)
        params.update(kw)
        self.bot = NewsTradingBot(**params)
        return self.bot


class TestMultiSourceBot(_Fixture, unittest.TestCase):
    def test_context_sources_raise_the_score_and_only_triggers_score(self):
        weak = item('Acme announces partnership with Globex', id='n1')            # rule score 0.2 alone
        sent = item('Acme upbeat', id='s1', kind=KIND_SENTIMENT, source='alphavantage',
                    meta={'ticker_sentiment': {'ACME': {'sentiment': 0.9}}}, published=T0 - timedelta(hours=3))
        earn = item('ACME EPS beat', id='e1', kind=KIND_EARNINGS_RESULT, source='finnhub',
                    meta={'eps_surprise_pct': 8.0, 'verdict': 'beat'}, published=T0 - timedelta(hours=4))
        social = item('StockTwits ACME', id='so1', kind=KIND_SOCIAL, source='stocktwits',
                      meta={'bull_ratio': 0.75, 'posts': 12}, published=T0 - timedelta(hours=2))
        bot = self.make(items=[sent, social])
        bot.tick()
        self.assertEqual(bot.state.decisions, [])                 # context only: nothing scored
        self.assertEqual(len(bot.store.items_for('ACME')), 2)
        self.source._items += [earn, weak]
        self.clock.advance(minutes=1)
        bot.tick()
        self.assertEqual(len(bot.state.decisions), 1)             # two triggers, one scoring pass
        d = bot.state.decisions[0]
        self.assertEqual(d['trigger'], weak.headline)             # newest trigger wins
        self.assertTrue(d['signal'])
        self.assertEqual(d['items'], 4)
        self.assertGreater(d['score'], 0.5)                       # 0.6*0.2 + 0.3*0.2 + 0.3*0.9 + 0.5 earnings + social
        self.assertIn('ACME', bot.state.positions)
        self.assertEqual(bot.summary()['mode'], 'trade')

    def test_one_scoring_pass_per_ticker_per_tick(self):
        seen = []
        scorer = RuleScorer()
        orig = scorer.score
        scorer.score = lambda b: seen.append(b.trigger.id) or orig(b)  # type: ignore[method-assign]
        items = [item('Acme beats estimates', id='a'), item('Acme raises guidance', id='b'),
                 item('Beta FDA approval granted', id='c', tickers=['BETA'])]
        bot = self.make(items=items, scorer=scorer)
        bot.tick()
        self.assertEqual(sorted(seen), ['b', 'c'])                # newest trigger per ticker, once each

    def test_notify_only_mode_with_telegram_and_webhook(self):
        sent, posted, cb = [], [], []
        tg = TelegramAction(sender=sent.append)
        wh = WebhookAction('http://example/hook', sender=posted.append)
        bot = self.make(actions=[tg, wh, CallbackAction(on_signal=lambda s, c, e: cb.append(s.ticker))],
                        items=[item('Acme beats estimates and raises guidance')])
        bot.tick()
        self.assertEqual(bot.summary()['mode'], 'notify-only')
        self.assertEqual(bot.state.positions, {})
        self.assertEqual(bot.state.pending, {})
        self.assertEqual(cb, ['ACME'])
        self.assertEqual(len(sent), 1)
        self.assertIn('LONG signal ACME', sent[0])
        self.assertIn('target 105.00, stop 97.00', sent[0])
        self.assertIn('Evidence:', sent[0])
        self.assertEqual(posted[0]['event'], 'signal')
        self.assertEqual(posted[0]['signal']['ticker'], 'ACME')

    def test_trade_plus_telegram_entry_and_exit_messages(self):
        sent = []
        tg = TelegramAction(sender=sent.append, events=['entry', 'exit'])
        bot = self.make(actions=[TradeAction(), tg], items=[item('Acme beats estimates and raises guidance')])
        bot.tick()
        self.assertIn('ACME', bot.state.positions)
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].startswith('✅ ENTERED ACME'))
        self.feed.set('ACME', 106)
        bot.tick()
        self.assertEqual(len(sent), 2)
        self.assertIn('EXIT ACME (target)', sent[1])

    def test_failing_action_does_not_block_trading(self):
        class Bad(CallbackAction):
            def on_signal(self, *a):
                raise RuntimeError('telegram down')
        bot = self.make(actions=[Bad(), TradeAction()], items=[item('Acme beats estimates and raises guidance')])
        bot.tick()
        self.assertIn('ACME', bot.state.positions)

    def test_stale_event_is_context_not_trigger(self):
        old_earn = item('ACME EPS beat', id='e1', kind=KIND_EARNINGS_RESULT, published=T0 - timedelta(hours=20),
                        meta={'eps_surprise_pct': 8.0, 'verdict': 'beat'})
        bot = self.make(items=[old_earn])
        bot.tick()
        self.assertEqual(bot.state.decisions, [])
        self.assertEqual(len(bot.store.items_for('ACME')), 1)

    def test_evidence_survives_restart(self):
        bot = self.make(items=[item('Acme upbeat', id='s1', kind=KIND_SENTIMENT, source='av',
                                    meta={'ticker_sentiment': {'ACME': {'sentiment': 0.5}}})])
        bot.tick()
        state = bot.state
        bot2 = NewsTradingBot(sources=[], engine=bot.engine, broker=self.broker, price_feed=self.feed, clock=self.clock,
                              state=state)
        self.assertEqual([i.id for i in bot2.store.items_for('ACME')], ['s1'])


class TestConfigBuild(unittest.TestCase):
    def test_every_source_and_action_type_builds(self):
        cfg = merge_config({
            'universe': ['ACME', 'PFE'], 'aliases': {'PFE': ['Pfizer']},
            'api_keys': {k: 'k' for k in ('finnhub', 'alphavantage', 'polygon', 'marketaux', 'newsapi', 'fmp')},
            'sources': [{'type': t} for t in (
                'yahoo', 'google', 'nasdaq_rss', 'finnhub', 'finnhub_earnings', 'alphavantage', 'polygon', 'marketaux',
                'newsapi', 'fmp', 'fmp_earnings', 'nasdaq_earnings', 'fda', 'openfda', 'clinicaltrials', 'stocktwits',
                'reddit')]
            + [{'type': 'rss', 'url': 'http://x/rss', 'kind': 'filing', 'min_interval_seconds': 30}],
            'scoring': {'type': 'rules', 'window_hours': 12},
            'actions': [{'type': 'trade'}, {'type': 'telegram', 'token': 't', 'chat_id': 'c', 'events': ['signal']},
                        {'type': 'webhook', 'url': 'http://x'}, {'type': 'log'}],
            'prices': {'type': 'static', 'prices': {'ACME': 10}},
            'bot': {'state_file': None},
        })
        bot = build_bot(cfg, clock=SimClock(T0))
        self.assertEqual(len(bot.source.sources), 18)
        self.assertEqual([a.name for a in bot.actions], ['trade', 'telegram', 'webhook', 'log'])
        self.assertTrue(bot.execute_trades)
        self.assertEqual(bot.store.window, timedelta(hours=12))
        self.assertIsInstance(bot.scorer, RuleScorer)
        with self.assertRaises(ValueError):
            build_bot(merge_config({'actions': [{'type': 'sms'}], 'prices': {'type': 'static'},
                                    'bot': {'state_file': None}}))

    def test_provider_without_key_is_disabled_not_fatal(self):
        src = FinnhubNews(['ACME'], api_key='')
        self.assertFalse(src.enabled)
        self.assertEqual(src.fetch(), [])


class TestBackfill(unittest.TestCase):
    def test_alpaca_backfill_paginates(self):
        from newsbot.alpaca import AlpacaNewsSource

        class FakeClient:
            pages = {None: {'news': [{'id': 1, 'headline': 'a', 'created_at': '2022-01-01T10:00:00Z',
                                      'symbols': ['ACME']}], 'next_page_token': 'p2'},
                     'p2': {'news': [{'id': 2, 'headline': 'b', 'created_at': '2022-01-02T10:00:00Z',
                                      'symbols': ['ACME']}]}}
            calls = []

            def get(self, path, base='', **params):
                self.calls.append(params)
                return self.pages[params.get('page_token')]

        client = FakeClient()
        items = list(AlpacaNewsSource(client, symbols=['ACME']).backfill(datetime(2022, 1, 1, tzinfo=timezone.utc)))
        self.assertEqual([i.id for i in items], ['alpaca:1', 'alpaca:2'])
        self.assertEqual(client.calls[0]['sort'], 'asc')
        self.assertEqual(client.calls[1]['page_token'], 'p2')


if __name__ == '__main__':
    unittest.main(verbosity=2)

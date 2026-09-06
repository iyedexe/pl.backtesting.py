import itertools

import numpy as np
import pytest

from index_inclusion import (
    IndexInclusion,
    build_tape,
    find_windows,
    make_market,
    rank_daily,
    run_backtest,
)
from index_inclusion.simulate import MarketParams


@pytest.fixture(scope='module')
def market():
    return make_market(MarketParams(start='2014-01-01', end='2019-12-31', seed=11))


def test_market_is_reproducible_for_a_seed(market):
    again = make_market(MarketParams(start='2014-01-01', end='2019-12-31', seed=11))
    np.testing.assert_array_equal(market.close, again.close)
    assert market.reviews == again.reviews
    other = make_market(MarketParams(start='2014-01-01', end='2019-12-31', seed=12))
    assert not np.array_equal(market.close, other.close)


def test_index_size_is_maintained_at_every_review(market):
    assert len(market.initial_members) == market.params.index_size
    for _, members in market.membership:
        assert len(members) == market.params.index_size


def test_membership_is_point_in_time(market):
    # membership consulted before a review's effective date must not include
    # that review's additions
    rv = market.reviews[3]
    before = market.members_at(rv['effective'])
    after = market.members_at(rv['effective'] + 1)
    for i in rv['adds']:
        assert i not in before and i in after


def test_ranks_are_competition_ranks(market):
    r = rank_daily(market)
    assert r.min() == 1 and r.max() == market.n_stocks
    assert sorted(r[:, 0]) == list(range(1, market.n_stocks + 1))


def test_windows_never_overlap_and_end_before_segment(market):
    windows = find_windows(market)
    assert windows
    for w1, w2 in itertools.pairwise(windows):
        assert w2['start'] > w1['end'] + 1
    tape, _ = build_tape(market, windows)
    assert tape.index.equals(market.calendar)
    assert set(tape['Window'][tape['Signal'] == 1]) == set(range(len(windows)))


def test_backtest_trades_never_span_a_splice(market):
    windows = find_windows(market)
    tape, seg_id = build_tape(market, windows)
    from backtesting import Backtest
    stats = Backtest(tape, IndexInclusion, cash=100_000, commission=.002,
                     finalize_trades=True).run()
    trades = stats['_trades']
    assert len(trades) > 5
    assert (seg_id[trades.EntryBar] == seg_id[trades.ExitBar]).all()
    assert (trades.EntryBar.values[1:] >= trades.ExitBar.values[:-1]).all()
    assert (trades.ExitBar - trades.EntryBar <= IndexInclusion.hold_limit).all()


def test_run_backtest_captures_the_planted_effect(market):
    st = run_backtest(market)
    assert st['# Trades'] > 5
    assert st['Win Rate [%]'] > 60      # a 5% planted effect is easy money
    assert st['_n_windows'] >= st['# Trades']

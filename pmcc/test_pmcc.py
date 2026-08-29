"""Unit tests for the PMCC backtest components.  Run: python -m pmcc.test_pmcc"""
import datetime as dt
import math
import unittest

from .pricing import (bs_delta, bs_price, monthly_expiries, next_expiry_with_dte,
                      norm_ppf, snap_strike, strike_for_delta, third_friday)
from .vol import VolSurface


class TestPricing(unittest.TestCase):
    def test_bs_known_value(self):
        # Hull-style reference: S=100, K=100, T=1, r=5%, sigma=20%, q=0
        c = bs_price(100, 100, 1.0, 0.05, 0.0, 0.20, True)
        self.assertAlmostEqual(c, 10.4506, places=3)
        p = bs_price(100, 100, 1.0, 0.05, 0.0, 0.20, False)
        self.assertAlmostEqual(p, 5.5735, places=3)

    def test_put_call_parity_with_yield(self):
        S, K, T, r, q, s = 87.0, 92.5, 0.7, 0.03, 0.045, 0.31
        c = bs_price(S, K, T, r, q, s, True)
        p = bs_price(S, K, T, r, q, s, False)
        lhs = c - p
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        self.assertAlmostEqual(lhs, rhs, places=8)

    def test_norm_ppf_roundtrip(self):
        from .pricing import norm_cdf
        for p in (0.001, 0.01, 0.2, 0.5, 0.8, 0.975, 0.9999):
            self.assertAlmostEqual(norm_cdf(norm_ppf(p)), p, places=9)

    def test_strike_for_delta_roundtrip(self):
        S, T, r, q, s = 60.0, 1.5, 0.02, 0.05, 0.28
        for target in (0.8, 0.5, 0.25):
            K = strike_for_delta(S, T, r, q, s, target)
            d = bs_delta(S, K, T, r, q, s, True)
            self.assertAlmostEqual(d, target, places=6)
        # deep ITM call strike is below spot, OTM above
        self.assertLess(strike_for_delta(S, T, r, q, s, 0.8), S)
        self.assertGreater(strike_for_delta(S, T, r, q, s, 0.25), S)

    def test_snap_strike(self):
        self.assertEqual(snap_strike(7.3), 7.5)
        self.assertEqual(snap_strike(18.2), 18.0)
        self.assertEqual(snap_strike(47.4), 48.0)
        self.assertEqual(snap_strike(88.7), 87.5)
        self.assertEqual(snap_strike(147.0), 145.0)
        self.assertEqual(snap_strike(303.0), 300.0)

    def test_intrinsic_at_expiry(self):
        self.assertEqual(bs_price(50, 40, 0.0, 0.02, 0.03, 0.3, True), 10)
        self.assertEqual(bs_price(50, 60, 0.0, 0.02, 0.03, 0.3, True), 0)


class TestCalendar(unittest.TestCase):
    def test_third_fridays(self):
        self.assertEqual(third_friday(2008, 6), dt.date(2008, 6, 20))
        self.assertEqual(third_friday(2015, 1), dt.date(2015, 1, 16))
        self.assertEqual(third_friday(2000, 4), dt.date(2000, 4, 21))
        self.assertEqual(third_friday(2012, 2), dt.date(2012, 2, 17))

    def test_next_expiry_dte_window(self):
        today = dt.date(2007, 3, 1)
        e = next_expiry_with_dte(today, 20, 50)
        self.assertEqual(e, dt.date(2007, 4, 20))
        e_leaps = next_expiry_with_dte(today, 540)
        self.assertGreaterEqual((e_leaps - today).days, 540)
        self.assertLessEqual((e_leaps - today).days, 540 + 31)

    def test_expiry_list_monotone(self):
        es = monthly_expiries(dt.date(2000, 1, 1), dt.date(2016, 12, 31))
        self.assertEqual(len(es), 17 * 12)
        self.assertTrue(all(b > a for a, b in zip(es, es[1:])))


class TestSurface(unittest.TestCase):
    def test_skew_direction(self):
        surf = VolSurface(atm_short=0.30, atm_long=0.25, skew_slope=0.10)
        F, T = 100.0, 0.25
        self.assertGreater(surf.iv(80, F, T), surf.iv(100, F, T))   # ITM call richer
        self.assertLess(surf.iv(120, F, T), surf.iv(100, F, T))    # OTM call cheaper

    def test_term_blend(self):
        surf = VolSurface(atm_short=0.40, atm_long=0.25, skew_slope=0.0)
        self.assertAlmostEqual(surf.atm(0.02), 0.40, delta=0.01)
        self.assertLess(surf.atm(2.0), 0.30)   # long end pulled toward 0.25


class TestEngineInvariants(unittest.TestCase):
    def test_buyhold_matches_analytic(self):
        """Zero-cost BuyHold total return == price return + dividend accrual."""
        from .engine import CostModel, Engine
        from .strategy import BuyHold
        costs = CostModel(opt_commission=0, opt_half_spread_pct=0,
                          opt_half_spread_min=0, stock_fee_pct=0, ftt_pct=0)
        eng = Engine('MC.PA', prem=1.1, costs=costs, start='2005-01-01',
                     end='2007-12-31', initial_cash=1e7)  # large cash: rounding negligible
        res = eng.run(BuyHold())
        got = res.equity.iloc[-1] / res.equity.iloc[0]
        # analytic: raw price return, dividends continuously reinvested, tiny
        # residual cash earning r ignored at this cash size
        px = eng.df['raw']
        yrs = (px.index[-1] - px.index[0]).days / 365.0
        want = (px.iloc[-1] / px.iloc[0]) * math.exp(eng.q * yrs)
        self.assertAlmostEqual(got / want, 1.0, delta=0.01)

    def test_pmcc_engine_smoke(self):
        from .engine import Engine
        from .strategy import PMCC
        eng = Engine('MC.PA', prem=1.1, start='2004-01-01', end='2006-12-31')
        res = eng.run(PMCC())
        self.assertFalse(res.equity.isna().any())
        self.assertGreater(res.counters['n_short_cycles'], 20)
        self.assertGreater(res.counters['short_premium_received'], 0)
        self.assertGreater(res.equity.iloc[-1], 50_000)  # not blown up in a bull market


if __name__ == '__main__':
    unittest.main()

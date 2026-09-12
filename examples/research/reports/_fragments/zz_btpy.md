
## Cross-check against backtesting.py

The same z-score rule, run through the bundled `backtesting.py` engine on the *price-ratio* approximation (single instrument, beta=1, next-open execution, 2x per-side commission), against the two-leg engine on the same ratio signals. The two implementations are independent, so agreement on trade count and return sign validates the signal/timing plumbing; level differences reflect the documented approximation.

|                          |   return_pct |   sharpe |   max_dd_pct |   n_trades |   win_rate_pct |
|:-------------------------|-------------:|---------:|-------------:|-----------:|---------------:|
| KO/PEP backtesting.py    |        19.25 |     0.41 |        12.79 |      20.00 |          75.00 |
| KO/PEP two-leg engine    |        10.60 |     0.51 |         5.18 |      20.00 |          75.00 |
| WTI/BRENT backtesting.py |      1727.66 |     0.37 |        34.79 |     234.00 |          67.52 |
| WTI/BRENT two-leg engine |       304.57 |     0.42 |        20.24 |     233.00 |          66.09 |

*(Note: this cross-check is a full-sample single split with a rolling z — it validates plumbing, not performance; both runs here share the same in-sample caveats.)*


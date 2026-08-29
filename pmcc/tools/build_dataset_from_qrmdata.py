"""
Build the bundled PMCC dataset from the CRAN `qrmdata` R package.

Provenance
----------
Source: https://github.com/cran/qrmdata (read-only CRAN mirror), package
version 2025-07-24-3, GPL-2/3, by M. Hofert, K. Hornik, A. J. McNeil.
Datasets used:
  * ``EURSTX_const.rda`` -- daily *adjusted* closes (Yahoo! Finance style,
    dividend + split adjusted, anchored so the last observation equals the
    raw close) of EURO STOXX 50 constituents, 2000-01-03 .. 2015-12-31.
    The 20 tickers with the ``.PA`` suffix are the French members.
  * ``CAC.rda``     -- CAC 40 index levels, 1990 .. 2015.
  * ``VIX.rda``     -- CBOE VIX index, 1990 .. 2015.
  * ``SP500.rda``   -- S&P 500 levels, 1950 .. 2015 (used together with VIX
    to estimate the implied/realised volatility premium).

Usage::

    git clone --depth 1 https://github.com/cran/qrmdata /tmp/qrmdata
    python -m pmcc.tools.build_dataset_from_qrmdata /tmp/qrmdata/data

Requires: ``pip install rdata pandas numpy``.

Output (written to ``pmcc/data/``):
  * ``prices_adj_fr_2000_2015.csv.gz`` -- adjusted closes, French tickers + ^FCHI
  * ``vix_sp500_1990_2015.csv.gz``     -- VIX and S&P 500 daily closes
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, os.pardir, 'data')


def _zoo_ctor(obj, attrs):
    """rdata constructor turning R xts/zoo objects into DataFrames."""
    vals = np.asarray(obj)
    idx = attrs.get('index')
    if idx is not None:
        idx = pd.to_datetime(np.asarray(idx, dtype='float64'), unit='s')
    if vals.ndim == 1:
        vals = vals[:, None]
    cols = None
    try:
        cols = [str(c) for c in obj.coords['dim_1'].values]
    except Exception:
        pass
    return pd.DataFrame(vals, index=idx, columns=cols)


def read_xts(path):
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d = rdata.read_rda(path, constructor_dict={'xts': _zoo_ctor, 'zoo': _zoo_ctor})
    return d


def main(rda_dir):
    os.makedirs(OUT_DIR, exist_ok=True)

    const = read_xts(os.path.join(rda_dir, 'EURSTX_const.rda'))['EURSTX_const']
    cac = read_xts(os.path.join(rda_dir, 'CAC.rda'))['CAC']
    vix = read_xts(os.path.join(rda_dir, 'VIX.rda'))['VIX']
    sp500 = read_xts(os.path.join(rda_dir, 'SP500.rda'))['SP500']

    fr = [t for t in const.columns if t.endswith('.PA')]
    px = const[fr].copy()
    px.index.name = 'Date'
    # Join the CAC 40 index over the same window
    px['^FCHI'] = cac.reindex(px.index).iloc[:, 0]

    # ------------------------------------------------------------------
    # Deterministic data repairs (documented; verified by inspection):
    #  * CS.PA (AXA): the 4-for-1 split of 2001-05-16 is missing from the
    #    adjustment -- rescale the pre-split segment by 1/4.
    #  * GLE.PA (Societe Generale): same for its 4-for-1 split of 2000-05-11.
    #  * SAF.PA: data before June 2005 is pre-merger Sagem (incl. an
    #    unadjusted reverse split) -- Safran only exists from May 2005.
    #  * EI.PA (Essilor): dozens of one-day half/double bad ticks -> drop.
    #  * ENGI.PA: pre-2005 data predates the Gaz de France IPO and contains
    #    multi-hundred-percent artifacts -> drop.
    #  * UL.PA (Unibail): series ends mid-2013 -> drop.
    # ------------------------------------------------------------------
    px.loc[:'2001-05-15', 'CS.PA'] *= 0.25
    px.loc[:'2000-05-10', 'GLE.PA'] *= 0.25
    px.loc[:'2005-05-31', 'SAF.PA'] = np.nan
    px = px.drop(columns=['EI.PA', 'ENGI.PA', 'UL.PA'])

    # Basic hygiene: forward-fill tiny gaps (<= 3 consecutive missing days)
    px = px.ffill(limit=3)

    # Validation: flag any remaining move that looks like a clean split ratio
    for c in px.columns:
        s = px[c].dropna()
        ratio = (s / s.shift()).dropna()
        for split in (2, 3, 4, 5, 0.5, 1 / 3, 0.25, 0.2):
            sus = ratio[(ratio / split - 1).abs() < 0.01]
            for d, v in sus.items():
                print(f"WARNING possible unadjusted split {c} {d.date()} ratio={v:.3f}")

    report = []
    for c in px.columns:
        s = px[c].dropna()
        r = s.pct_change().abs()
        report.append((c, str(s.index[0].date()), str(s.index[-1].date()),
                       int(px[c].isna().sum()), round(float(r.max()) * 100, 1)))
    rep = pd.DataFrame(report, columns=['ticker', 'first', 'last', 'n_nan', 'max_abs_ret_%'])
    print(rep.to_string(index=False))

    out1 = os.path.join(OUT_DIR, 'prices_adj_fr_2000_2015.csv.gz')
    px.round(6).to_csv(out1, compression='gzip')
    print('wrote', out1, px.shape)

    vs = pd.DataFrame({'VIX': vix.iloc[:, 0], 'SP500': sp500.reindex(vix.index).iloc[:, 0]})
    vs.index.name = 'Date'
    vs = vs.dropna(how='all')
    out2 = os.path.join(OUT_DIR, 'vix_sp500_1990_2015.csv.gz')
    vs.round(6).to_csv(out2, compression='gzip')
    print('wrote', out2, vs.shape)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/qrmdata/data')

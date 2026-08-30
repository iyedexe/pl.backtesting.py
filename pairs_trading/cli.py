"""`pairs` — command-line interface of the pairs-trading research lab.

Examples::

    pairs screen --study crypto --top 15
    pairs run --study commodities --quick
    pairs all
    pairs report
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings


def _cmd_screen(args):
    from .experiments import SCREEN_PARAMS, study_panel
    from .screening import screen_panel
    panel = study_panel(args.study)
    corr_min, min_obs, cap = SCREEN_PARAMS[args.study]
    scr = screen_panel(panel, corr_min=args.corr_min or corr_min,
                       min_obs=min_obs, max_pairs_tested=cap)
    with_p = scr[scr['eg_pvalue'] < 0.05]
    print(f'{args.study}: {panel.shape[1]} assets, {len(scr)} pairs screened, '
          f'{len(with_p)} pass EG at 5%')
    print(scr.head(args.top).to_string(index=False,
                                       float_format=lambda v: f'{v:.3f}'))


def _cmd_run(args):
    from .experiments import ALL_STUDIES, run_btpy_crosscheck, run_study
    studies = ALL_STUDIES if args.study == 'all' else [args.study]
    t0 = time.time()
    for s in studies:
        t = time.time()
        frag = run_study(s, quick=args.quick)
        print(f'[{s}] done in {time.time() - t:.1f}s -> {frag}')
    if args.study == 'all':
        frag = run_btpy_crosscheck(quick=args.quick)
        print(f'[btpy cross-check] -> {frag}')
        _cmd_report(args)
    print(f'total {time.time() - t0:.1f}s')


def _cmd_report(_args):
    from .report import build_report
    out = build_report()
    print(f'report -> {out}')


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='.*maximum lag.*')
    p = argparse.ArgumentParser(prog='pairs', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    studies = ['crypto', 'stocks', 'forex', 'commodities', 'cross']

    ps = sub.add_parser('screen', help='screen a universe for cointegrated pairs')
    ps.add_argument('--study', choices=studies, required=True)
    ps.add_argument('--top', type=int, default=15)
    ps.add_argument('--corr-min', type=float, default=None)
    ps.set_defaults(func=_cmd_screen)

    pr = sub.add_parser('run', help='run one study (or all) end to end')
    pr.add_argument('--study', choices=[*studies, 'all'], required=True)
    pr.add_argument('--quick', action='store_true',
                    help='smaller universes, no sensitivity grids')
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser('all', help='run every study and build the report')
    pa.add_argument('--quick', action='store_true')
    pa.set_defaults(func=_cmd_run, study='all')

    pp = sub.add_parser('report', help='assemble research/reports/report.md')
    pp.set_defaults(func=_cmd_report)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())

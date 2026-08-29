"""Review calendars for the supported indices, straight from the rulebooks.

All date arithmetic is plain weekday math: exchange holidays are ignored, so a
cutoff/effective date can occasionally land one business day off (e.g. around
Easter). That is fine for a signal bot that scans daily - phases span days -
but is called out in the README.

Rulebook timings implemented here (verified Aug 2026):

* FTSE 100 - quarterly reviews (Mar/Jun/Sep/Dec); ranks taken at the close of
  the Tuesday before the first Friday of the review month; changes announced
  the next day and effective after the close of the third Friday.
* DAX 40 - quarterly reviews implemented after the third Friday; ranks taken
  at the review cutoff, the last trading day of the preceding month. The
  fast-entry band (rank <= 33) applies at every review; the wider regular
  entry band (rank <= 40) only in March and September. Announcement lands
  about a week after the cutoff (approximated as +5 business days).
* Nasdaq-100 - annual reconstitution: ranks as of the last trading day of
  November, announced around the second Friday of December, effective before
  the open on the Monday after the third Friday. Since 2026 there are also
  rank-based quarterly reviews in Mar/Jun/Sep, approximated here with the
  same month-end cutoff pattern and the top-75 entry band.
* S&P 500 - no schedule: additions happen at the committee's discretion, so
  the bot treats it as a calendar-less watchlist (see screener.py).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

FRIDAY = 4


def _nth_friday(year: int, month: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta((FRIDAY - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def next_bday(d: dt.date, n: int = 1) -> dt.date:
    while n:
        d += dt.timedelta(1)
        if d.weekday() < 5:
            n -= 1
    return d


def prev_bday(d: dt.date, n: int = 1) -> dt.date:
    while n:
        d -= dt.timedelta(1)
        if d.weekday() < 5:
            n -= 1
    return d


def bdays_between(a: dt.date, b: dt.date) -> int:
    """Business days from `a` to `b` (positive when b is later)."""
    step = 1 if b >= a else -1
    n = 0
    while a != b:
        a += dt.timedelta(step)
        if a.weekday() < 5:
            n += step
    return n


def _last_bday_of_prev_month(year: int, month: int) -> dt.date:
    return prev_bday(dt.date(year, month, 1))


@dataclass(frozen=True)
class Review:
    index_key: str
    label: str            # e.g. 'quarterly review', 'annual reconstitution'
    threshold: int        # entry band: non-member at this rank or better gets added
    cutoff: dt.date       # ranks are taken at this day's close
    announce: dt.date     # changes made public (approximate for DAX/NDX quarterly)
    effective: dt.date    # changes implemented at/after this day's close
    confidence: str       # 'high' | 'medium'

    @property
    def exit_date(self) -> dt.date:
        """First session after the trackers' effective-day trading - our exit."""
        return next_bday(self.effective)

    def phase(self, today: dt.date, pre_cutoff_days: int) -> str | None:
        """Where `today` falls in this review's life, or None if outside."""
        if self.cutoff <= today < self.effective:
            return 'post_cutoff'     # ranks locked, additions computable
        if today < self.cutoff and bdays_between(today, self.cutoff) <= pre_cutoff_days:
            return 'pre_cutoff'      # prediction window
        if self.effective <= today < self.exit_date:
            return 'effective'
        return None


def ftse100_reviews(years) -> list[Review]:
    out = []
    for year in years:
        for month in (3, 6, 9, 12):
            first_friday = _nth_friday(year, month, 1)
            cutoff = first_friday - dt.timedelta(days=3)          # the Tuesday before
            out.append(Review('FTSE100', 'quarterly review', threshold=90,
                              cutoff=cutoff, announce=next_bday(cutoff),
                              effective=_nth_friday(year, month, 3),
                              confidence='high'))
    return out


def dax40_reviews(years) -> list[Review]:
    out = []
    for year in years:
        for month in (3, 6, 9, 12):
            regular = month in (3, 9)
            cutoff = _last_bday_of_prev_month(year, month)
            out.append(Review(
                'DAX40',
                'regular review' if regular else 'fast-entry check',
                threshold=40 if regular else 33,
                cutoff=cutoff, announce=next_bday(cutoff, 5),
                effective=_nth_friday(year, month, 3),
                confidence='high'))
    return out


def ndx100_reviews(years) -> list[Review]:
    out = []
    for year in years:
        out.append(Review('NDX100', 'annual reconstitution', threshold=75,
                          cutoff=_last_bday_of_prev_month(year, 12),
                          announce=_nth_friday(year, 12, 2),
                          effective=_nth_friday(year, 12, 3),
                          confidence='high'))
        for month in (3, 6, 9):
            cutoff = _last_bday_of_prev_month(year, month)
            out.append(Review('NDX100', 'quarterly review', threshold=75,
                              cutoff=cutoff, announce=next_bday(cutoff, 5),
                              effective=_nth_friday(year, month, 3),
                              confidence='medium'))
    return out


_CALENDARS = {
    'FTSE100': ftse100_reviews,
    'DAX40': dax40_reviews,
    'NDX100': ndx100_reviews,
}


def active_review(index_key: str, today: dt.date,
                  pre_cutoff_days: int) -> tuple[Review, str] | None:
    """The review whose window covers `today`, with its phase, if any."""
    builder = _CALENDARS.get(index_key)
    if builder is None:
        return None
    for review in sorted(builder((today.year - 1, today.year, today.year + 1)),
                         key=lambda r: r.cutoff):
        phase = review.phase(today, pre_cutoff_days)
        if phase:
            return review, phase
    return None

"""Corpus policy: which split a case belongs to, and which hour of a day it is taken at.

Both are decided HERE and written into the record at generation time. Nothing downstream
re-derives either -- a split inferred from a filename or a date at training time is a split
that can silently disagree with the one the case was generated under, and the whole point
of a split is that it cannot move.
"""
from __future__ import annotations

import datetime as dt
import hashlib

# ============================================================================ SPLITS ====
#
# HARD-CODED BY CALENDAR MONTH, AND ASSIGNED AT GENERATION. Whole years to val and test so
# a split boundary never falls inside a synoptic system, and so seasonal coverage is
# complete on each side rather than sampled.
#
# The four 2026 months are training-side padding at the end of the record; they are
# alternating months so they do not form one contiguous season.
#
# A month not named here is NOT IN THE CORPUS. That is deliberate and it is checked rather
# than defaulted: `split_of` refuses an unlisted month instead of guessing, so a case can
# never be generated into a split nobody chose.
_TRAIN_YEARS = (2021, 2022, 2023)
_TRAIN_2026_MONTHS = (2, 4, 6, 8)
VAL_YEAR = 2024
TEST_YEAR = 2025

SPLITS = {}
for _y in _TRAIN_YEARS:
    for _m in range(1, 13):
        SPLITS[(_y, _m)] = "train"
for _m in _TRAIN_2026_MONTHS:
    SPLITS[(2026, _m)] = "train"
for _m in range(1, 13):
    SPLITS[(VAL_YEAR, _m)] = "val"
    SPLITS[(TEST_YEAR, _m)] = "test"

# === NO CASE CAN LAND IN TWO SPLITS, ASSERTED AT IMPORT ================================
# A dict cannot hold one key twice, so the real risk is the RANGES overlapping while the
# dict quietly keeps whichever was written last. Count the months each split claims and
# check the total against the number of distinct keys: if any month were claimed twice the
# sum would exceed it.
_CLAIMED = ([(y, m) for y in _TRAIN_YEARS for m in range(1, 13)]
            + [(2026, m) for m in _TRAIN_2026_MONTHS]
            + [(VAL_YEAR, m) for m in range(1, 13)]
            + [(TEST_YEAR, m) for m in range(1, 13)])
if len(_CLAIMED) != len(set(_CLAIMED)):
    _dupes = sorted({k for k in _CLAIMED if _CLAIMED.count(k) > 1})
    raise AssertionError(f"a month is claimed by more than one split: {_dupes}")
SPLIT_MONTH_COUNTS = {s: sum(1 for v in SPLITS.values() if v == s)
                      for s in ("train", "val", "test")}
if SPLIT_MONTH_COUNTS != {"train": 40, "val": 12, "test": 12}:
    raise AssertionError(f"split month counts are {SPLIT_MONTH_COUNTS}, expected "
                         f"{{'train': 40, 'val': 12, 'test': 12}}")

# HRRR v4 begins here; nothing before it is forceable at all.
HRRR_V4_START = dt.date(2020, 12, 2)

# === THE ONE PER-DAY HOUR CAP ==========================================================
# 2026-08-31 is the last day of the record and the analyses past midday do not exist yet,
# so the draw is capped rather than allowed to spend the pool on hours that cannot return
# data. This is a data-availability fact about one day, not a screening rule.
HOUR_CAPS = {dt.date(2026, 8, 31): 12}

# =========================================================================== SCREENS ====
ZI_MIN_M = 300.0        # below: the 30 m receptor leaves the surface layer
ZI_MAX_M = 1250.0       # above: the 3660 m box cannot hold the layer at L >= 2 z_i
MAX_DZIDT_REL = 15.0    # %/h; a stationarity screen, independent of the z_i value
MAX_ZOL = 0.0           # z/L must be NEGATIVE: the corpus contains no stable cases, and
                        # docs/history/stable-regime.md is why -- a stable BL laminarises at
                        # this grid and there is no stationary state to sample.


def split_of(when):
    """'train' | 'val' | 'test' for a date or datetime. Refuses an unlisted month."""
    d = when.date() if isinstance(when, dt.datetime) else when
    s = SPLITS.get((d.year, d.month))
    if s is None:
        _tr = "/".join(str(y) for y in _TRAIN_YEARS)
        _t26 = "/".join(f"2026-{m:02d}" for m in _TRAIN_2026_MONTHS)
        raise ValueError(
            f"{d.isoformat()} is in no split. The corpus is train {_tr} plus {_t26}, "
            f"val {VAL_YEAR}, test {TEST_YEAR}. A month outside that is not part of the "
            f"corpus and a case must not be generated for it -- assigning one would put it "
            f"in a split nobody chose. See lpdm/corpus.py:SPLITS.")
    if d < HRRR_V4_START:
        raise ValueError(f"{d.isoformat()} is before HRRR v4 ({HRRR_V4_START}); there is "
                         f"no analysis to force a case with.")
    return s


def hour_cap(day):
    """The highest UTC hour drawable on this day (23 unless the day is capped)."""
    return HOUR_CAPS.get(day, 23)


def hour_pool(day):
    """The day's full pool of drawable round hours, in order."""
    return list(range(0, hour_cap(day) + 1))


def _seed_for(day):
    """A stable per-day RNG seed, so a re-run draws the SAME sequence.

    Derived from the date alone -- not from the clock, the machine or the process -- so a
    month re-run after a failure reproduces its own selection instead of quietly sampling a
    different corpus. The corpus is generated across several rented machines and resumed
    after interruptions; a selection that is not reproducible is not auditable.
    """
    h = hashlib.sha256(day.isoformat().encode()).digest()
    return int.from_bytes(h[:8], "big")


class DayDraw:
    """Sample this day's hours WITHOUT REPLACEMENT, uniformly, reproducibly.

    An hour is marked used the moment it is drawn, whatever happens to it: missing HRRR,
    a failed screen and a successful case all consume it. So the pool strictly shrinks and
    the day terminates -- either at an accepted hour or with the pool exhausted, which is a
    MISSING DAY with a reason and not a retry.

        d = DayDraw(date(2023, 7, 15))
        while (h := d.draw()) is not None:
            ...            # evaluate hour h
            d.reject(h, "z_i 1480 m outside [300, 1250]")   # or d.accept(h)
    """

    def __init__(self, day, rng=None):
        import random
        self.day = day
        self.pool = hour_pool(day)
        self.used = []
        self.reasons = {}
        self.accepted = None
        self._rng = rng if rng is not None else random.Random(_seed_for(day))

    def draw(self):
        """The next hour, drawn uniformly from what is left. None when exhausted."""
        if not self.pool:
            return None
        h = self._rng.choice(self.pool)
        self.pool.remove(h)
        self.used.append(h)
        return h

    def reject(self, hour, reason):
        self.reasons[hour] = reason

    def accept(self, hour):
        self.accepted = hour

    @property
    def exhausted(self):
        return not self.pool and self.accepted is None

    def summary(self):
        return {
            "date": self.day.isoformat(),
            "pool_size": len(hour_pool(self.day)),
            "hour_cap": hour_cap(self.day),
            "drawn": list(self.used),
            "accepted_hour": self.accepted,
            "rejected": dict(sorted(self.reasons.items())),
        }


def screen(hpbl, shtfl, dzidt_rel, zm=30.0, L=None, zol=None):
    """None if the hour is acceptable, else the reason it is not.

    Order matters only for which reason gets reported first; all four are checked. `zol` is
    z_m/L if the caller has it; otherwise the SIGN of the sensible heat flux stands in for
    it, which is all the z/L < 0 screen actually needs.
    """
    if hpbl is None:
        return "HRRR returned no HPBL"
    if zol is None:
        # SHTFL is positive UPWARD out of the surface in the HRRR analysis, so a positive
        # value is an unstable surface layer, i.e. z/L < 0. That is the whole content of
        # the z/L < 0 screen; the magnitude of z/L is not being thresholded.
        if shtfl is None:
            return "HRRR returned no SHTFL, so the stability sign is unknown"
        if shtfl <= 0.0:
            return (f"z/L >= 0: SHTFL {shtfl:+.1f} W/m2 is not an unstable surface layer, "
                    f"and the corpus contains no stable or exactly-neutral cases")
    elif zol >= MAX_ZOL:
        return f"z/L = {zol:+.3f} >= {MAX_ZOL:.2f}"
    if not (ZI_MIN_M <= hpbl <= ZI_MAX_M):
        return f"z_i {hpbl:.0f} m outside [{ZI_MIN_M:.0f}, {ZI_MAX_M:.0f}]"
    if dzidt_rel is None:
        return "dz_i/dt could not be formed (a neighbouring analysis hour is missing)"
    if abs(dzidt_rel) >= MAX_DZIDT_REL:
        return f"|dz_i/dt| {abs(dzidt_rel):.1f} %/h >= {MAX_DZIDT_REL:.0f}"
    return None

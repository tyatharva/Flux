#!/usr/bin/env python3
"""Draw the day's case hour, WITHOUT REPLACEMENT, from the 24 round hours.

  usage: bin/pick_hour.py 2023-07-15
         bin/pick_hour.py 2023-07-15 --json results/hours/2023-07-15.json

Prints the accepted timestamp on stdout (`2023-07-15T19:00`) and exits 0, or prints nothing
and exits 3 for a MISSING DAY. Everything else goes to stderr, so the caller can use the
stdout verbatim.

=== WITHOUT REPLACEMENT, AND AN HOUR IS SPENT WHATEVER HAPPENS TO IT ===

A pool of the day's round hours; draw uniformly from what is left; the drawn hour is marked
used immediately -- missing HRRR, a failed screen and an accepted case all consume it
equally. So the pool strictly shrinks and the day terminates: at an accepted hour, or with
the pool empty, which is a MISSING DAY with a reason rather than a retry.

The alternative -- pick midday and retry elsewhere on failure -- is neither reproducible nor
unbiased: it silently prefers whatever hours happen to survive, which at this site means
mornings and evenings, exactly when z_i moves fastest. Uniform sampling without replacement
has no such preference. There is NO rose weighting and NO direction stratification: the
weather supplies the rose, and stratifying on direction would be choosing the corpus's
input distribution rather than observing it.

=== THE DRAW IS REPRODUCIBLE ===

Seeded from the DATE alone, so re-running a month after a failure re-draws the same
sequence rather than quietly sampling a different corpus (lpdm/corpus.py:_seed_for).

=== THE SOUNDING IS THE ANALYSIS VALID AT EXACTLY T, NOT T-1 ===

This script only chooses T; bin/hrrr_sounding.py fetches at `fxx=0`, the analysis valid at
the timestamp it is given, and bin/get_case.sh gives it T. The reason it must be T and not
T-1 is that FORCING IS CONSTANT THROUGH THE RUN: the LES is initialised from the sounding
and then integrates 1.25 h under a fixed geostrophic wind and a fixed surface flux, so the
atmosphere never evolves from a T-1 state toward a T state. A T-1 sounding would produce a
footprint labelled T that represents T-1 meteorology, and the label is what the emulator is
trained against. The screening below reads HPBL at T-1 and T+1 too, but only to form
dz_i/dt; neither ever initialises anything.
"""
import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpdm.corpus import DayDraw, screen, split_of  # noqa: E402


class Screener:
    """HPBL/SHTFL at the tower for an analysis hour, fetched once and cached.

    A draw needs its own hour plus the two neighbours, for dz_i/dt. Neighbours are shared
    between adjacent draws, so the cache is what keeps an exhausted 24-hour pool near 26
    fetches instead of 72.
    """

    def __init__(self, cache_dir="data/hrrr", verbose=True):
        self.cache_dir = cache_dir
        self.verbose = verbose
        self._c = {}
        self.n_fetch = 0

    def at(self, when):
        """The screening fields at `when`, or None if HRRR has nothing there."""
        key = when.strftime("%Y%m%d%H")
        if key in self._c:
            return self._c[key]
        from enumerate_times import screen_hour
        try:
            self.n_fetch += 1
            v = screen_hour(when, self.cache_dir)
        except Exception as e:                       # noqa: BLE001 - any failure is "absent"
            if self.verbose:
                print(f"      HRRR at {when:%Y-%m-%d %H}Z unavailable: "
                      f"{type(e).__name__}: {str(e)[:90]}", file=sys.stderr)
            v = None
        self._c[key] = v
        return v

    def dzidt_rel(self, when):
        """d(z_i)/dt at `when` as %/h, centred; None if a neighbour is missing."""
        c = self.at(when)
        if c is None or c.get("hpbl") is None:
            return None
        a = self.at(when - dt.timedelta(hours=1))
        b = self.at(when + dt.timedelta(hours=1))
        if a is None or b is None:
            return None
        zi = c["hpbl"]
        if zi <= 0:
            return None
        return 100.0 * (b["hpbl"] - a["hpbl"]) / 2.0 / zi


class StubScreener:
    """The Screener's interface, answered offline from a hash. FOR PLUMBING ONLY.

    WHAT IT IS FOR. Dry-running a whole machine -- 8 months, ~243 days, a shared queue over
    8 workers -- needs the day loop, the draw, the screen reasons and the per-day JSON to
    run for real, and needs the HRRR fetch NOT to, because 243 days is thousands of network
    round trips and a rented box is not where that should be discovered.

    WHAT IT IS NOT. It is not a simulation of the weather and no corpus decision may come
    from it. `bin/run_corpus_machine.py --stub` is the only caller, its records are stamped
    `stub: true`, and `--stub-screen` is refused unless that flag was passed.

    THE DISTRIBUTION IS CHOSEN TO EXERCISE THE SCHEDULER, NOT TO RESEMBLE THE SITE. A
    uniform per-day yield would make every worker finish at the same time and would test
    nothing about rebalancing, so the acceptance probability VARIES BY CALENDAR MONTH --
    high in summer, low in winter, which is the direction the real screens go -- and each
    hour's fields are drawn from the day-and-hour hash. Some days exhaust their pool and
    return MISSING with a real reason, which is the outcome the queue has to absorb.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.n_fetch = 0
        self._c = {}

    @staticmethod
    def _h(when, salt):
        import hashlib
        b = hashlib.sha256(f"{salt}|{when:%Y%m%d%H}".encode()).digest()
        return int.from_bytes(b[:6], "big") / float(1 << 48)      # uniform [0, 1)

    def at(self, when):
        key = when.strftime("%Y%m%d%H")
        if key in self._c:
            return self._c[key]
        self.n_fetch += 1
        # ~4% of hours have no analysis at all, and whole days occasionally go: both are
        # real HRRR outcomes and both are paths the day loop has to handle.
        if self._h(when, "absent") < 0.04 or self._h(when.replace(hour=0), "dayout") < 0.02:
            v = None
        else:
            import math
            # Seasonal amplitude, and a daytime window. The constants are chosen so the
            # yield lands near the 80% PROJECT_BRIEF.md records for the real screens -- a yield
            # near 0 or near 1 would make every worker finish together and test nothing.
            warm = 0.35 + 0.65 * 0.5 * (1.0 - math.cos(2 * math.pi * (when.month - 1) / 12.0))
            # A PER-DAY SYNOPTIC FACTOR, and it is the point of the whole stub. Without it
            # the yield is 98% and flat, every worker gets ~30 cases a month and finishes
            # with every other worker -- so the shared queue and a rigid month-per-GPU
            # assignment would be indistinguishable and the rebalancing claim untested.
            # Overcast days suppress the surface flux entirely, more often in winter, which
            # is what puts real variance into how much work a month is.
            overcast = 0.40 - 0.28 * (warm - 0.35) / 0.65
            if self._h(when.replace(hour=0), "synoptic") < overcast:
                warm *= 0.04          # enough to drive SHTFL negative all day
            diel = max(0.0, math.sin(math.pi * (when.hour - 6) / 12.0))
            # SMOOTH in hour, deliberately: dz_i/dt is a CENTRED difference over the
            # neighbours, so a z_i with hour-to-hour hash noise in it fails the 15 %/h
            # screen almost everywhere and the day loop then only ever exercises one
            # rejection reason. The per-hour hash enters the OFFSET, not the shape.
            base = 260.0 + 120.0 * self._h(when.replace(hour=0), "ziday")
            v = {"hpbl": base + 620.0 * warm * diel,
                 "shtfl": -30.0 + 330.0 * warm * diel,
                 "wdir": 360.0 * self._h(when, "dir"),
                 "wspd": 2.0 + 10.0 * self._h(when, "spd")}
        self._c[key] = v
        return v

    def dzidt_rel(self, when):
        c = self.at(when)
        if c is None or c.get("hpbl") is None:
            return None
        a, b = self.at(when - dt.timedelta(hours=1)), self.at(when + dt.timedelta(hours=1))
        if a is None or b is None:
            return None
        return 100.0 * (b["hpbl"] - a["hpbl"]) / 2.0 / c["hpbl"]


def pick(day, screener, verbose=True):
    """(timestamp or None, the draw's own record)."""
    d = DayDraw(day)
    if verbose:
        print(f"  {day.isoformat()} ({split_of(day)}): pool of {len(d.pool)} hours",
              file=sys.stderr)
    while True:
        h = d.draw()
        if h is None:
            break
        when = dt.datetime.combine(day, dt.time(hour=h))
        c = screener.at(when)
        if c is None:
            d.reject(h, "HRRR analysis missing for this hour")
            if verbose:
                print(f"    {h:02d}Z  missing", file=sys.stderr)
            continue
        why = screen(c.get("hpbl"), c.get("shtfl"), screener.dzidt_rel(when))
        if why is None:
            d.accept(h)
            if verbose:
                print(f"    {h:02d}Z  ACCEPTED  z_i {c['hpbl']:.0f} m, "
                      f"SHTFL {c.get('shtfl', float('nan')):+.0f} W/m2, "
                      f"dz_i/dt {screener.dzidt_rel(when):+.1f} %/h, "
                      f"wind from {c['wdir']:.0f} deg", file=sys.stderr)
            rec = d.summary()
            rec["accepted"] = {
                "timestamp": when.strftime("%Y-%m-%dT%H:00"),
                "zi_m": c.get("hpbl"), "shtfl_wm2": c.get("shtfl"),
                "dzidt_rel_per_h": screener.dzidt_rel(when),
                "wdir_deg": c.get("wdir"), "wspd_ms": c.get("wspd"),
            }
            rec["n_hrrr_fetches"] = screener.n_fetch
            return when, rec
        d.reject(h, why)
        if verbose:
            print(f"    {h:02d}Z  rejected: {why}", file=sys.stderr)
    rec = d.summary()
    rec["accepted"] = None
    rec["n_hrrr_fetches"] = screener.n_fetch
    miss = sum(1 for v in rec["rejected"].values() if v.startswith("HRRR analysis missing"))
    rec["missing_reason"] = (
        "the whole day's HRRR is missing" if miss == rec["pool_size"] else
        f"the pool of {rec['pool_size']} hours was exhausted with no acceptance "
        f"({miss} missing from HRRR, {rec['pool_size'] - miss} screened out)")
    return None, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="YYYY-MM-DD")
    ap.add_argument("--json", default=None, help="write the full draw record here")
    ap.add_argument("--cache", default="data/hrrr")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--stub-screen", action="store_true",
                    help="answer the screen offline from a hash instead of fetching HRRR. "
                         "PLUMBING ONLY -- requires FLUX_STUB=1 in the environment so it "
                         "cannot be reached by an ordinary corpus command, and every "
                         "record it leads to is stamped stub:true.")
    a = ap.parse_args()

    day = dt.date.fromisoformat(a.date)
    split_of(day)                    # refuses a day outside the corpus before any fetch
    if a.stub_screen and os.environ.get("FLUX_STUB") != "1":
        print("FATAL: --stub-screen needs FLUX_STUB=1. It answers the meteorological "
              "screen from a hash; nothing that reaches the corpus may use it.",
              file=sys.stderr)
        return 64
    scr = StubScreener(not a.quiet) if a.stub_screen else Screener(a.cache, not a.quiet)
    when, rec = pick(day, scr, not a.quiet)
    if a.stub_screen:
        rec["STUB_SCREEN"] = True
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(rec, open(a.json, "w"), indent=1, default=float)
    if when is None:
        print(f"  DAY MISSING: {rec['missing_reason']}", file=sys.stderr)
        return 3
    print(when.strftime("%Y-%m-%dT%H:00"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

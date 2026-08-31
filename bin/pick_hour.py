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
    a = ap.parse_args()

    day = dt.date.fromisoformat(a.date)
    split_of(day)                    # refuses a day outside the corpus before any fetch
    when, rec = pick(day, Screener(a.cache, not a.quiet), not a.quiet)
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

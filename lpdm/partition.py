"""Which machine generates which month. Deterministic, total, and checkable in one call.

    from lpdm.partition import MONTHS, machine_of, months_for, describe
    months_for(3)        -> the eight (year, month) pairs machine 3 owns
    machine_of(2023, 7)  -> 3
    print(describe())    -> the full 8x8 table, printed at startup

=== WHY A RULE AND NOT A TABLE ===

A hand-written 64-entry table is a thing that can be edited wrong, and the failure is
silent in the worst way: a month assigned to two machines is generated twice (wasted
rental, and two records for one case racing for the same filename), a month assigned to
none is simply absent from the corpus and nothing ever says so. The rule below is one
line, and the properties that matter -- TOTAL and DISJOINT -- are asserted at import
against `lpdm.corpus.SPLITS` rather than trusted.

=== THE RULE: order by (calendar month, year), then round robin ===

Sort every corpus month by CALENDAR MONTH first and year second -- Jan21, Jan22, Jan23,
Jan24, Jan25, Feb21, ... -- and machine m takes indices m, m+8, m+16, ...

The obvious rule is chronological order, and it was written first and rejected on its own
output. `gcd(8, 12) = 4`, so stepping 8 months at a time walks an orbit of only THREE
calendar months: machine 0 drew Jan, May and Sep and nothing else, machine 3 drew Apr, Aug
and Dec. Seven of the eight machines were missing an entire season. Sorting by calendar
month instead breaks the resonance -- each calendar month has 5 or 6 instances, fewer than
the 8 machines, so its instances land on different machines and the offset walks.

Three properties, all checkable in the table `describe()` prints:

  EVERY MACHINE HOLDS ALL FOUR SEASONS, and eight DISTINCT calendar months. On rented
  hardware a machine will be lost, and losing one must not delete a season from the
  corpus. Under the chronological rule losing machine 6 would have taken most of the
  record's July, March and November; under this one it takes one month in eight, spread.

  NO MACHINE HOLDS MORE THAN A QUARTER OF ANY SPLIT. Measured on the printed table:
  4-6 of the 40 train months, 0-3 of the 12 val, 0-3 of the 12 test -- so the worst single
  machine loss costs 25% of val or of test and 15% of train. Note the 0s: this rule does
  NOT give every machine every split, and the chronological one did. That was traded for
  the season property above, deliberately, because a split is a partition of MONTHS the
  training code re-derives from the calendar, whereas a missing season is missing PHYSICS
  that nothing downstream can reconstruct. A per-split partition -- one machine takes val
  -- is the obvious alternative and is the worst of the three: losing that machine deletes
  the whole split.

  DAY COUNTS ARE WITHIN A FEW PERCENT (237-246 of ~243). Not by construction and not
  relied upon -- see below.

What the rule deliberately does NOT do is balance DAYS exactly (28 vs 31) or balance
EXPECTED YIELD (a February in the corpus yields fewer cases than a July). Neither matters,
because within a machine the work is a SHARED QUEUE over all of its days rather than one
month pinned per GPU -- so a light month simply frees its worker sooner. Balancing across
machines would matter if machines had to finish together; they do not.
"""
from __future__ import annotations

import calendar
import datetime as _dt

from lpdm.corpus import SPLITS, split_of

N_MACHINES = 8

# THE ONE ORDERING THE RULE DEPENDS ON: calendar month major, year minor. Derived from
# SPLITS rather than restated, so a month added to the corpus is partitioned automatically
# and the assertions below re-run against it at import.
MONTHS = tuple(sorted(SPLITS.keys(), key=lambda k: (k[1], k[0])))

# The same months in date order, for printing and for iterating a machine's work.
MONTHS_CHRONOLOGICAL = tuple(sorted(SPLITS.keys()))


def machine_of(year, month):
    """Which machine owns (year, month). Raises for a month outside the corpus."""
    key = (int(year), int(month))
    try:
        return MONTHS.index(key) % N_MACHINES
    except ValueError:
        raise ValueError(
            f"{key[0]}-{key[1]:02d} is not in the corpus, so no machine owns it. The "
            f"corpus months are defined by lpdm/corpus.py:SPLITS.") from None


def months_for(machine):
    """The (year, month) pairs this machine owns, in chronological order."""
    m = int(machine)
    if not 0 <= m < N_MACHINES:
        raise ValueError(f"--machine must be 0..{N_MACHINES - 1}, got {machine}")
    return sorted(k for i, k in enumerate(MONTHS) if i % N_MACHINES == m)


def days_in(year, month):
    return calendar.monthrange(int(year), int(month))[1]


def month_str(ym):
    return f"{ym[0]}-{ym[1]:02d}"


# === TOTAL AND DISJOINT, ASSERTED AT IMPORT ============================================
# Not a comment claiming it: the union of every machine's months must be exactly the
# corpus, with no month appearing twice. A partition that silently drops or duplicates a
# month is the failure this module exists to prevent, and it costs a microsecond to rule
# out at import instead of discovering it from a short corpus at the end of a rental.
_ASSIGNED = [ym for m in range(N_MACHINES) for ym in months_for(m)]
if len(_ASSIGNED) != len(set(_ASSIGNED)):
    _d = sorted({k for k in _ASSIGNED if _ASSIGNED.count(k) > 1})
    raise AssertionError(f"a month is owned by more than one machine: {_d}")
if set(_ASSIGNED) != set(MONTHS):
    _miss = sorted(set(MONTHS) - set(_ASSIGNED))
    _extra = sorted(set(_ASSIGNED) - set(MONTHS))
    raise AssertionError(f"the partition is not total: missing {_miss}, extra {_extra}")
if len(MONTHS) != N_MACHINES * (len(MONTHS) // N_MACHINES):
    raise AssertionError(
        f"{len(MONTHS)} corpus months do not divide evenly over {N_MACHINES} machines; "
        f"the round-robin still covers every month exactly once, but the machines carry "
        f"different counts. Check that this is intended before running.")
MONTHS_PER_MACHINE = len(MONTHS) // N_MACHINES

# === AND THE SEASON PROPERTY IS ASSERTED, NOT JUST CLAIMED =============================
# This is the property the chronological rule silently failed. It is cheap to check and
# the whole reason the ordering is what it is, so it is checked at import: a future edit
# to SPLITS that reintroduces a resonance fails loudly here instead of quietly shipping a
# partition where one dead machine costs a season.
for _m in range(N_MACHINES):
    _seasons = {(ym[1] % 12) // 3 for ym in months_for(_m)}
    if len(_seasons) != 4:
        raise AssertionError(
            f"machine {_m} holds only {len(_seasons)} of 4 seasons "
            f"({sorted(months_for(_m))}). Losing it would delete a season from the "
            f"corpus. See the ordering note at the top of this file.")

# And the split-concentration bound, for the same reason: it is the number the "losing one
# machine" argument rests on, so it is checked rather than quoted from a table that was
# true when it was written.
_SPLIT_TOTALS = {}
for _ym in MONTHS:
    _SPLIT_TOTALS[split_of(_dt.date(_ym[0], _ym[1], 1))] = \
        _SPLIT_TOTALS.get(split_of(_dt.date(_ym[0], _ym[1], 1)), 0) + 1
MAX_SPLIT_SHARE = 0.0
for _m in range(N_MACHINES):
    _c = {}
    for _ym in months_for(_m):
        _k = split_of(_dt.date(_ym[0], _ym[1], 1))
        _c[_k] = _c.get(_k, 0) + 1
    for _sp, _n in _c.items():
        MAX_SPLIT_SHARE = max(MAX_SPLIT_SHARE, _n / _SPLIT_TOTALS[_sp])
if MAX_SPLIT_SHARE > 0.30:
    raise AssertionError(
        f"one machine holds {MAX_SPLIT_SHARE:.0%} of a split; losing it would cost more "
        f"than the 30% this partition is designed to bound.")


def summary(machine):
    """{'months': [...], 'days': N, 'splits': {...}} for one machine."""
    ms = months_for(machine)
    sp = {}
    for ym in ms:
        k = split_of(_dt.date(ym[0], ym[1], 1))
        sp[k] = sp.get(k, 0) + 1
    return {"machine": machine, "months": [month_str(x) for x in ms],
            "n_months": len(ms), "days": sum(days_in(*x) for x in ms),
            "splits": sp,
            "n_calendar_months": len({x[1] for x in ms}),
            "n_seasons": len({(x[1] % 12) // 3 for x in ms})}


def describe(highlight=None):
    """The whole 8 x 8 assignment as printable text. Printed at startup, every run.

    THE OPERATOR HAS TO BE ABLE TO CHECK COVERAGE WITHOUT TRUSTING THIS FILE, so the
    totals at the bottom are recomputed from the printed rows rather than from the rule.
    """
    L = [f"=== MONTH -> MACHINE: {len(MONTHS)} corpus months over {N_MACHINES} machines "
         f"({MONTHS_PER_MACHINE} each) ===",
         "  rule: sort by (calendar month, year), then index % 8. Chronological order was",
         "        tried first and rejected: gcd(8,12)=4 gave each machine only 3 distinct",
         "        calendar months and left 7 of 8 missing a season.",
         ""]
    seen = []
    for m in range(N_MACHINES):
        s = summary(m)
        seen += months_for(m)
        mark = " <== THIS MACHINE" if highlight is not None and m == highlight else ""
        sp = " ".join(f"{k}:{v}" for k, v in sorted(s["splits"].items()))
        L.append(f"  machine {m}:  {'  '.join(s['months'])}")
        L.append(f"              {s['days']:4d} days   {sp}   "
                 f"seasons {s['n_seasons']}/4   {s['n_calendar_months']} distinct "
                 f"calendar months{mark}")
    L.append("")
    L.append(f"  COVERAGE CHECK, recomputed from the rows above: {len(seen)} assignments, "
             f"{len(set(seen))} distinct, corpus has {len(MONTHS)} months "
             f"-> {'OK, each exactly once' if len(seen) == len(set(seen)) == len(MONTHS) else 'BROKEN'}")
    L.append(f"  total days across all machines: {sum(days_in(*x) for x in MONTHS)}")
    L.append(f"  largest share of any one split held by a single machine: "
             f"{MAX_SPLIT_SHARE:.0%} (bounded at 30%); every machine holds 4/4 seasons")
    return "\n".join(L)


if __name__ == "__main__":
    print(describe())

#!/usr/bin/env python3
"""The live view of a corpus machine. Reads the progress file; never touches the run.

    corpus_progress                      # watch /out/progress.json, redraw in place
    corpus_progress --out /out --once    # print once and exit (for a script or a pipe)

=== WHY THIS IS A SEPARATE PROCESS ===

An SSH session drops. A run that exists only as a foreground process attached to that
session drops with it, and on a rented box that is the whole rental. So the orchestrator
never draws anything it needs: it writes `progress.json` on the MOUNTED VOLUME after every
state change, and this renders it. Start it, kill it, reconnect an hour later and start it
again -- the run does not know or care, and nothing is lost by not watching.

The file is written to a temp name and renamed, so a read never catches a half-written
record. A read that fails anyway is retried rather than reported, because a torn read is
not news.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time

BAR = "#"


def _hms(h):
    if h is None:
        return "  --  "
    s = int(h * 3600)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m"


def render(d, width):
    L = []
    c = d.get("counts", {})
    tot = d.get("n_days_total") or 1
    done = c.get("done", 0)
    frac = min(1.0, done / tot)
    nb = max(10, min(48, width - 30))
    stub = "  *** STUB RUN -- not corpus records ***" if d.get("stub") else ""
    L.append(f"CORPUS MACHINE {d.get('machine')}   "
             f"{'FINISHED' if d.get('finished') else 'running'}   "
             f"pid {d.get('pid')}{stub}")
    L.append(f"months: {' '.join(d.get('months', []))}")
    L.append("")
    L.append(f"[{BAR * int(nb * frac):<{nb}}] {done}/{tot} days  {100 * frac:5.1f}%")
    L.append(f"  cases {c.get('case', 0):<5} missing {c.get('missing', 0):<5} "
             f"failed {c.get('failed', 0):<5} skipped {c.get('skipped', 0):<5}")
    gph = d.get("gpu_h_per_case")
    L.append(f"  elapsed {_hms(d.get('elapsed_h'))}   "
             f"projected total {_hms(d.get('projected_total_h'))}   "
             f"finish {d.get('projected_finish_utc') or '--'}")
    L.append(f"  mean GPU-h per case (occupancy): "
             f"{gph if gph is None else f'{gph:.3f}'}")
    h = d.get("host") or {}
    if h:
        L.append(f"  host: peak RSS {h.get('cgroup_peak_gb')} GB of "
                 f"{h.get('mem_total_gb')} GB   MemAvailable low-water "
                 f"{h.get('mem_available_min_gb')} GB   swap {h.get('swap_used_peak_gb')} GB")
    L.append("")
    L.append("  GPU  state          month    day         stage")
    for g in sorted(d.get("gpus", {}), key=lambda x: int(x)):
        v = d["gpus"][g]
        L.append(f"  {g:>3}  {v.get('state', 'idle'):<14} {v.get('month', '-'):<8} "
                 f"{v.get('day', '-'):<11} {v.get('stage', '-')}")
    rec = d.get("recent") or []
    if rec:
        L.append("")
        L.append("  recent:")
        for r in rec[:8]:
            L.append(f"    {r[:width - 6]}")
    al = d.get("alerts") or []
    if al:
        L.append("")
        for m in al:
            L.append(f"  *** {m[:width - 8]}")
    return [ln[:width] for ln in L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    path = os.path.join(a.out, "progress.json")

    first = True
    while True:
        try:
            d = json.load(open(path))
        except (OSError, ValueError):
            if a.once:
                print(f"no readable {path} yet", file=sys.stderr)
                return 1
            time.sleep(a.interval)
            continue
        w = shutil.get_terminal_size((100, 30)).columns
        lines = render(d, w)
        if a.once:
            print("\n".join(lines))
            return 0
        # Redraw IN PLACE: home the cursor and clear to end of screen, rather than
        # scrolling. A clear-screen each tick flickers and loses whatever the operator
        # scrolled back to look at.
        sys.stdout.write(("\033[2J" if first else "") + "\033[H" + "\n".join(lines)
                         + "\033[J\n")
        sys.stdout.flush()
        first = False
        if d.get("finished"):
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()

#!/usr/bin/env python3
"""A schema-valid training record with no LES, no LPDM and no HRRR. PLUMBING ONLY.

    FLUX_STUB=1 bin/stub_case.py 2023-07-15T19:00 --npz-dir /out/pairs_npz

WHAT IT IS FOR. `bin/run_corpus_machine.py --stub` dry-runs a whole machine -- eight
months, ~243 days, a shared queue over eight workers, the progress file, the resume path
and the manifest -- and every one of those is orchestration rather than physics. Doing it
with real cases costs ~120 GPU-h per machine; doing it with `STUB_LES=1` still needs an
HRRR fetch per day. This writes the one artifact the orchestration is judged on, so the
dry run finishes in seconds and still exercises every path that touches the record.

WHAT IT IS NOT, AND THE ARTIFACT SAYS SO ITSELF. `meta.stub = true`, which
`bin/check_npz.py` REFUSES unless asked for a stub. The target raster is an analytic blob,
not a footprint. FLUX_STUB=1 is required in the environment so this cannot be reached by an
ordinary corpus command.

THE DELAY IS DELIBERATE AND IT IS NOT REALISM. A stub that returns instantly makes every
worker finish at the same instant, so a shared queue and a rigid month-per-GPU assignment
produce identical timelines and the scheduling is untested. A few milliseconds, VARIED per
case from the case's own hash, is enough to make workers finish unevenly and force the
queue to actually hand out work. It is deterministic, so a re-run reproduces the same
timeline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.corpus import split_of  # noqa: E402

N, PAD, NG = 128, 3, 122
SCALARS = ("h", "ustar", "sigma_v", "L", "sin_wdir", "cos_wdir")


def _u(tag, salt):
    b = hashlib.sha256(f"{salt}|{tag}".encode()).digest()
    return int.from_bytes(b[:6], "big") / float(1 << 48)


def _raster(tag, salt, x0, y0, sx, sy, amp):
    """A blob on the 122 grid, zero-padded to 128. The pad is structural zero."""
    g = np.zeros((N, N), dtype=np.float64)
    j, i = np.meshgrid(np.arange(NG), np.arange(NG), indexing="ij")
    r = ((i - x0) / sx) ** 2 + ((j - y0) / sy) ** 2
    g[PAD:PAD + NG, PAD:PAD + NG] = amp * np.exp(-r)
    return g.astype(np.float32)


def build(ts, git_commit=None):
    when = dt.datetime.fromisoformat(ts)
    tag = "case_" + when.strftime("%Y%m%d%H")
    split = split_of(when.date())
    wdir = 360.0 * _u(tag, "dir")
    th = np.radians(wdir)
    h = 400.0 + 700.0 * _u(tag, "h")
    ustar = 0.20 + 0.45 * _u(tag, "us")
    sigv = ustar * (1.7 + 0.9 * _u(tag, "sv"))
    L = -(8.0 + 400.0 * _u(tag, "L"))

    # The peak sits upwind, which is the one structural property a consumer might lean on.
    d = 4.0 + 6.0 * _u(tag, "peak")
    cx = NG / 2.0 + d * np.sin(th)
    cy = NG / 2.0 + d * np.cos(th)
    kl = _raster(tag, "k", cx, cy, 5.0, 8.0, 1.0e-4)
    tg = kl + _raster(tag, "t", cx + 2.0, cy - 1.5, 7.0, 10.0, 2.0e-5)
    # Signed, like the real target: negative lobes are physical and nothing clips them.
    tg = (tg - _raster(tag, "n", cx - 6.0, cy + 5.0, 9.0, 12.0, 6.0e-6)).astype(np.float32)

    meta = {
        "format": "flux-footprint-pair/1",
        "STUB_WARNING": "NOT A CORPUS RECORD. No LES, no LPDM, no HRRR. bin/stub_case.py.",
        "stub": True,
        "run_id": tag + "-stub",
        "parent_case": tag,
        "split": split,
        "split_key": f"{when.year}-{when.month:02d}",
        "datetime": when.strftime("%Y-%m-%dT%H:00"),
        "gate_state": "STUB",
        "zi_accepted_drifting": False,
        "zi_achieved_m": float(h),
        "h_estimator": "tke_peak_fraction",
        "integral": float(0.9 + 0.3 * _u(tag, "int")),
        "peak_x_m": float(30.0 * d),
        "centroid_dist_m": float(30.0 * (d + 4.0)),
        "array_share": float(0.02 + 0.30 * _u(tag, "arr")),
        "git_commit": git_commit,
        "kljun_source": "STUB (analytic blob; the real channel is third_party/FFP v1.42)",
        "grid": {"n": NG, "pad": PAD, "n_padded": N, "dx_m": 30.0, "dy_m": 30.0,
                 "domain_m": 3660.0},
        "receptor": {"z_m": 30.0, "z_agl_m": 30.0},
        "scalar_names": list(SCALARS),
        "inv_L": float(1.0 / L),
        "wdir_deg": float(wdir),
    }
    sc = np.array([h, ustar, sigv, L, np.sin(th), np.cos(th)], dtype=np.float32)
    return tag, sc, kl, tg, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timestamp", help="YYYY-MM-DDTHH:00")
    ap.add_argument("--npz-dir", default="pairs_npz")
    ap.add_argument("--delay-ms", default="3,25",
                    help="LO,HI -- a per-case delay drawn from the case hash, so workers "
                         "finish unevenly and the queue is actually exercised")
    ap.add_argument("--git-commit", default=None)
    a = ap.parse_args()
    if os.environ.get("FLUX_STUB") != "1":
        print("FATAL: bin/stub_case.py needs FLUX_STUB=1. It writes a record that is not a "
              "corpus record; it must not be reachable from an ordinary corpus command.",
              file=sys.stderr)
        return 64

    tag, sc, kl, tg, meta = build(a.timestamp, a.git_commit)
    lo, hi = (float(x) for x in a.delay_ms.split(","))
    time.sleep((lo + (hi - lo) * _u(tag, "delay")) / 1000.0)

    os.makedirs(a.npz_dir, exist_ok=True)
    out = os.path.join(a.npz_dir, tag + ".npz")
    # WRITTEN THROUGH A FILE OBJECT, not a path. np.savez_compressed APPENDS `.npz` to a
    # path that does not already end in it, so a `<out>.tmp.<pid>` path silently becomes
    # `<out>.tmp.<pid>.npz` and the os.replace below fails on a file that is not there.
    # A handle also keeps the temporary name outside the `*.npz` glob the resume scan uses.
    tmp = out + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, scalars=sc, kljun=kl, target=tg,
                            meta=np.array(json.dumps(meta)))
    os.replace(tmp, out)          # atomic: a killed worker never leaves a half record
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

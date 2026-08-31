#!/usr/bin/env python3
"""Validate a training record against the format, and print one in full.

The corpus is generated on rented machines that are destroyed afterwards, so a malformed
record is not something anyone gets to go back and fix. This is the check that runs at the
end of every case, on the file that is about to be the only thing that survives.

  usage: bin/check_npz.py pairs_npz/case_2023071519.npz
         bin/check_npz.py pairs_npz/case_2023071519.npz --print
         bin/check_npz.py pairs_npz/*.npz --quiet
         bin/check_npz.py <f> --expect-split train --expect-datetime 2023-07-15T19:00
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.corpus import split_of  # noqa: E402

N = 128
PAD = 3
SCALARS = ("h", "ustar", "sigma_v", "L", "sin_wdir", "cos_wdir")
REQUIRED_META = (
    "format", "run_id", "parent_case", "split", "split_key", "datetime",
    "gate_state", "zi_accepted_drifting", "zi_achieved_m", "h_estimator",
    "integral", "peak_x_m", "centroid_dist_m", "array_share",
    "git_commit", "kljun_source", "grid", "receptor", "scalar_names", "inv_L",
)


def check(path, expect_split=None, expect_dt=None, allow_stub=False):
    bad = []

    def fail(msg):
        bad.append(msg)

    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        for k in ("scalars", "kljun", "target", "meta"):
            if k not in keys:
                fail(f"missing array '{k}'")
        if bad:
            return bad, None
        extra = keys - {"scalars", "kljun", "target", "meta"}
        if extra:
            fail(f"unexpected arrays {sorted(extra)}; the record is meant to be exactly "
                 f"four entries")
        sc = np.asarray(z["scalars"])
        kl = np.asarray(z["kljun"])
        tg = np.asarray(z["target"])
        meta = json.loads(str(z["meta"]))

    if sc.shape != (6,):
        fail(f"scalars is {sc.shape}, expected (6,)")
    if sc.dtype != np.float32:
        fail(f"scalars dtype is {sc.dtype}, expected float32")
    for nm, arr in (("kljun", kl), ("target", tg)):
        if arr.shape != (N, N):
            fail(f"{nm} is {arr.shape}, expected ({N}, {N})")
        if arr.dtype != np.float32:
            fail(f"{nm} dtype is {arr.dtype}, expected float32")
        if not np.isfinite(arr).all():
            fail(f"{nm} carries {int((~np.isfinite(arr)).sum())} non-finite cells")
    # THE PAD MUST BE STRUCTURAL ZERO ON BOTH CHANNELS. 122 -> 128 is a zero-pad and not a
    # resize; if anything has leaked into the border, the two channels are not on the cells
    # the record says they are.
    for nm, arr in (("kljun", kl), ("target", tg)):
        if arr.shape == (N, N):
            border = np.concatenate([arr[:PAD].ravel(), arr[-PAD:].ravel(),
                                     arr[:, :PAD].ravel(), arr[:, -PAD:].ravel()])
            if np.any(border != 0.0):
                fail(f"{nm} has {int((border != 0).sum())} non-zero cells in the "
                     f"{PAD}-cell pad; the pad is structural and must be exactly zero")

    # L is legitimately +/-inf at exactly neutral; everything else must be finite.
    for i, nm in enumerate(SCALARS):
        if nm == "L":
            continue
        if not np.isfinite(sc[i]):
            fail(f"scalar '{nm}' is {sc[i]}")
    if abs(float(sc[4]) ** 2 + float(sc[5]) ** 2 - 1.0) > 1e-5:
        fail(f"sin_wdir^2 + cos_wdir^2 = {sc[4]**2 + sc[5]**2:.6f}, not 1")

    for k in REQUIRED_META:
        if k not in meta:
            fail(f"meta is missing '{k}'")
    if meta.get("scalar_names") and tuple(meta["scalar_names"]) != SCALARS:
        fail(f"meta.scalar_names {tuple(meta['scalar_names'])} != {SCALARS}")
    g = meta.get("grid") or {}
    if g.get("n") != 122 or g.get("pad") != PAD or g.get("n_padded") != N:
        fail(f"meta.grid says n={g.get('n')} pad={g.get('pad')} "
             f"n_padded={g.get('n_padded')}, expected 122/{PAD}/{N}")
    # THE GEOMETRY, ASSERTED IN THE RECORD ITSELF. A case that ran on a retired grid is
    # complete, plausible and wrong; this is the one place it can be caught after the fact.
    if g.get("dx_m") is not None and abs(float(g["dx_m"]) - 30.0) > 1e-6:
        fail(f"meta.grid.dx_m is {g['dx_m']}, not the 30 m production spacing")
    if g.get("domain_m") is not None and abs(float(g["domain_m"]) - 3660.0) > 1e-3:
        fail(f"meta.grid.domain_m is {g['domain_m']}, not 3660 m")
    if meta.get("split") not in ("train", "val", "test"):
        fail(f"meta.split is {meta.get('split')!r}")
    if meta.get("h_estimator") != "tke_peak_fraction":
        fail(f"meta.h_estimator is {meta.get('h_estimator')!r}")
    # A STUBBED RECORD IS NOT A CORPUS RECORD, AND THE ARTIFACT SAYS SO ITSELF.
    if meta.get("stub") and not allow_stub:
        fail("meta.stub is true: this record was produced with the LES and the LPDM "
             "STUBBED (bin/stub_footprint.py). Its target raster is Kljun plus a smooth "
             "perturbation, not a simulation. Pass --allow-stub only when verifying "
             "plumbing.")

    # The split must agree with the record's OWN datetime, independently of whoever wrote it.
    stamp = meta.get("datetime")
    if stamp:
        try:
            when = dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
            der = split_of(when.date())
            if der != meta.get("split"):
                fail(f"meta.split is {meta.get('split')!r} but {when.date()} belongs to "
                     f"{der!r}")
            if when.minute or when.second:
                fail(f"meta.datetime {stamp} is not a round hour")
            if expect_dt:
                want = dt.datetime.fromisoformat(expect_dt)
                if when.replace(tzinfo=None) != want:
                    fail(f"meta.datetime {when.replace(tzinfo=None)} != requested {want}")
        except ValueError as e:
            fail(f"meta.datetime {stamp!r}: {e}")
    if expect_split and meta.get("split") != expect_split:
        fail(f"meta.split {meta.get('split')!r} != requested {expect_split!r}")
    return bad, meta


def show(path, meta):
    with np.load(path, allow_pickle=True) as z:
        sc, kl, tg = z["scalars"], z["kljun"], z["target"]
    da = (meta["grid"]["dx_m"] or 0) * (meta["grid"]["dy_m"] or 0)
    print(f"=== {path}  ({os.path.getsize(path)/1e3:.0f} kB) ===")
    print(f"  arrays: scalars{tuple(sc.shape)} {sc.dtype} | kljun{tuple(kl.shape)} "
          f"{kl.dtype} | target{tuple(tg.shape)} {tg.dtype} | meta (json)")
    print("  scalars:")
    for nm, v in zip(SCALARS, sc):
        print(f"      {nm:<10}{float(v):>14.6f}")
    print(f"  target : sum*dA {float(tg.sum())*da:.6f}  min {float(tg.min()):.4e}  "
          f"max {float(tg.max()):.4e}  negative cells {int((tg < 0).sum())}")
    print(f"  kljun  : sum*dA {float(kl.sum())*da:.6f}  min {float(kl.min()):.4e}  "
          f"max {float(kl.max()):.4e}")
    print("  meta:")
    print("\n".join("      " + ln for ln in json.dumps(meta, indent=1, sort_keys=True,
                                                       default=float).splitlines()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--print", dest="show", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--expect-split", default=None)
    ap.add_argument("--expect-datetime", default=None)
    ap.add_argument("--allow-stub", action="store_true",
                    help="accept a record made with the LES stubbed. Verification only.")
    a = ap.parse_args()

    rc = 0
    for p in a.npz:
        bad, meta = check(p, a.expect_split, a.expect_datetime, a.allow_stub)
        if bad:
            rc = 1
            print(f"*** {p}: {len(bad)} problem(s)")
            for b in bad:
                print(f"      {b}")
        elif not a.quiet:
            print(f"  OK {os.path.basename(p)}  split={meta['split']:<6} "
                  f"{meta['datetime']}  h={meta['zi_achieved_m']:.0f} m  "
                  f"integral={meta['integral']:.3f}")
        if a.show and meta:
            show(p, meta)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

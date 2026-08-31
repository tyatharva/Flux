#!/usr/bin/env python3
"""A SYNTHETIC stage-7 output, for verifying the pipeline's plumbing with no GPU.

WHAT THIS IS FOR, AND WHAT IT CANNOT TELL YOU. It stands in for the LES and the backward
LPDM so the rest of the path -- sounding, forcing, surface, seed pick, restart, the training
record, the split, the manifest, the scratch cleanup -- can be exercised end to end on a
machine with no GPU. It says nothing whatever about the physics: the target raster it writes
is Kljun's own footprint plus a smooth perturbation, not a simulation of anything. A case
built with it is NOT a corpus case and must never reach `pairs_npz/` on a production run.
`--stub` is stamped into the footprint JSON and `bin/run_month.sh` refuses to record one.

Everything it CAN check, it checks by construction rather than by imitation:

  * the raster is built on THE CASE'S OWN GRID, read out of the grid directory the case
    actually used -- so a stubbed run on the wrong geometry produces the wrong raster and
    bin/check_npz.py catches it, exactly as a real one would;
  * the cell EDGES are the real ones, so make_pair's "Kljun on identical cells" assertion is
    a real assertion here too;
  * `stats` comes from the case's own HRRR forcing, so the six scalars are this case's, and
    L, h and the direction are the ones the sounding implies.

  usage: stub_footprint.py --grid data/case_grids/<tag> --forcing results/forcing/<tag>.json
                           --outdir results/corpus --tag <tag> --z-target 28.5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm import kljun_ffp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--forcing", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--z-target", type=float, default=28.5)
    a = ap.parse_args()

    fc = json.load(open(a.forcing))
    lab = fc["labels"]

    # ---- THE GRID, READ OFF THE CASE'S OWN SURFACE ---------------------------------------
    # Never a literal and never a default: if the case ran on a retired geometry this raster
    # comes out the wrong size and the schema check downstream fails, which is the point.
    z0 = np.load(os.path.join(a.grid, "z0m.npy"))
    ny, nx = z0.shape
    meta_p = os.path.join(a.grid, "meta.json")
    dx = None
    if os.path.exists(meta_p):
        try:
            gm = json.load(open(meta_p))
            dx = float(gm.get("dx") or gm.get("dx_m"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            dx = None
    if dx is None:
        dx = float(os.environ.get("DX", "30"))
    xc = (np.arange(nx) - (nx - 1) / 2.0) * dx
    yc = (np.arange(ny) - (ny - 1) / 2.0) * dx
    xe = np.concatenate([xc - dx / 2.0, [xc[-1] + dx / 2.0]])
    ye = np.concatenate([yc - dx / 2.0, [yc[-1] + dx / 2.0]])

    # ---- THE SCALARS, FROM THIS CASE'S OWN SOUNDING --------------------------------------
    h = float(lab["zi_m"])
    ust = float(lab["ustar_estimate"])
    L = float(lab["L_estimate"])
    wdir = float(lab.get("predicted_10m_dir_deg", lab["G_dir_from_deg"]))
    # sigma_v/u* runs 1.9-2.3 in this project's own windows; the mid-range is used so the
    # Kljun evaluation below is inside its published validity envelope.
    sig_v = 2.1 * ust
    # A log-law wind at the receptor over the domain's own geometric-mean z0.
    z0g = float(np.exp(np.log(z0).mean()))
    u_mean = max(ust / 0.4 * np.log(max(a.z_target, 2 * z0g) / z0g), 0.5)

    prof = kljun_ffp.ffp_profile(a.z_target, h, L, ust, sig_v, umean=u_mean)
    ang = np.radians(wdir)
    kl = kljun_ffp.footprint_on_static(xe, ye, ang, a.z_target, h, ust, sig_v,
                                       umean=u_mean, L=L, prof=prof)
    # THE "TARGET" IS KLJUN PLUS A SMOOTH, SIGNED PERTURBATION. It is not a footprint. It
    # exists so the target channel is distinguishable from the Kljun channel (a record whose
    # two channels were identical would pass a shape check that a real one might not) and so
    # the negative lobe a real target carries is present in the plumbing.
    X, Y = np.meshgrid(xc, yc)
    r = np.hypot(X, Y) / max(float(np.max(np.abs(xc))), 1.0)
    tg = kl * (1.0 + 0.25 * np.cos(3.0 * np.arctan2(Y, X)) * np.exp(-2.0 * r)) \
        - 0.02 * kl.max() * np.exp(-((r - 0.35) / 0.10) ** 2)

    da = dx * dx
    integral = float(tg.sum() * da)
    asym = 1.0 - a.z_target / h
    k = int(np.argmax(tg))
    pj, pi = divmod(k, nx)

    os.makedirs(a.outdir, exist_ok=True)
    base = os.path.join(a.outdir, a.tag)
    np.savez_compressed(base + ".npz", xc=xc, yc=yc, xe=xe, ye=ye,
                        les=tg.astype(np.float64), kljun=kl.astype(np.float64))
    fp = {
        "stub": True,
        "stub_note": ("SYNTHETIC. The target raster is Kljun plus a smooth perturbation, "
                      "not an LES footprint. bin/stub_footprint.py."),
        "stats": {
            "u_mean": u_mean, "ustar": ust, "sigma_v": sig_v, "h": h, "L": L,
            "wdir": wdir, "sigma_w": 1.25 * ust, "sigma_u": 2.4 * ust,
            "sigma_v_resolved": 0.9 * sig_v, "sigma_w_resolved": 1.0 * ust,
            "e_sgs": 0.1 * ust ** 2, "htFlux": float(lab["wth_virtual"]),
            "theta0": 300.0, "n_dumps": 541,
            "z_recept": a.z_target, "k_recept": 3, "d_recept": 0.0,
            "z_eff": a.z_target, "U": u_mean * np.sin(ang), "V": u_mean * np.cos(ang),
        },
        "zm": a.z_target, "zm_agl": a.z_target, "z_target": a.z_target, "d_recept": 0.0,
        "wind_angle": wdir,
        "integral_les": integral,
        "integral_kljun": float(kl.sum() * da),
        "integral_asymptote": float(asym),
        "les": {"peak_x": float(np.hypot(xc[pi], yc[pj])), "x80": 900.0,
                "area80_ha": 25.0, "centroid_dist": 400.0, "centroid_bearing": wdir},
        "kljun": {"peak_x": float(prof["x_peak"])},
        "cover_share": {"solar array": 0.05},
        "cover_share_se": {"solar array": 0.005},
        "floor": {"health": {"stub": True}},
        "profiles": {"zlev": None, "ww_prof": None, "esgs_prof": None, "tke_prof": None},
        "rel_seconds": 1800.0, "tback": 900.0,
        "kljun_source": "official FFP v1.42 (third_party/FFP), via lpdm/kljun_ffp.py",
    }
    json.dump(fp, open(base + ".json", "w"), indent=1, default=float)
    print(f"  STUB footprint {base}.json ({ny} x {nx} at dx = {dx:.1f} m), "
          f"integral {integral:.3f}, h {h:.0f} m, L {L:.1f} m, wdir {wdir:.1f} deg")
    print(f"  *** THIS IS NOT A SIMULATION. It verifies plumbing and geometry only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

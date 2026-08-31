#!/usr/bin/env python3
"""Assemble ONE (input, target) training record. Stage 8, the last of the pipeline.

    input   Kljun's scalars, every one of them read off the LES window itself
    target  the 122 x 122 LPDM flux footprint on that same window

=== THE INPUTS COME FROM THE LES, NOT FROM THE SOUNDING ===

This is the load-bearing decision of the whole corpus design and it is worth stating
plainly. The sounding chooses which state to simulate; it does not describe the state that
results. u*, U(z_m), sigma_v, L, z_i and the wind direction are all read back off the
window by lpdm/les_stats.py:window_stats, so:

  * an imperfectly-matched seed, or an adjustment that has not fully closed, moves where a
    case LANDS in input space without making the pair wrong -- input and target are
    measured on the same fields, so they are consistent by construction;
  * the Ekman turning, the inertial oscillation and the entrainment growth are all
    absorbed as LABELS rather than errors, which is what PROJECT_BRIEF.md already requires for
    direction ("Achieved direction is not forcing direction");
  * and the emulator is being taught the map the LES actually realises, which is the only
    map the LPDM footprint corresponds to.

The sounding's own numbers are still carried, as `forcing`, so the achieved-vs-requested
gap is measurable across the corpus instead of assumed small.

=== SPLIT BY RUN, NEVER BY SAMPLE ===

`run_id` is written into every record and is the ONLY legitimate grouping key for a
train/test split. The effective sample size for generalisation is the number of LES runs,
not the number of footprint cells or sub-windows drawn from one run (PROJECT_BRIEF.md, ML model).

=== WHAT IS NOT AN INPUT ===

z_0 and the receptor height are properties of the site, identical in every case, so they
are recorded as provenance rather than offered as features -- a constant column is not a
predictor, and this is a single-tower emulator by design.

usage: make_pair.py --tag <case> --footprint results/<case>.json
                    [--forcing results/forcing/<case>.json] [--seed results/pick/<case>.json]
                    [--outdir pairs]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm import kljun_ffp

# Kljun's scalar inputs, and nothing else. PROJECT_BRIEF.md: "Inputs are Kljun's scalars only."
KLJUN_INPUTS = ("u_mean", "ustar", "sigma_v", "h", "L", "wdir")

# ---- THE .npz TRAINING RECORD ---------------------------------------------------------
# 122 -> 128 BY ZERO-PADDING 3 CELLS ON EVERY SIDE. 128 is what an FNO wants (its spectral
# transform is happiest on a power of two) and 122 + 3 + 3 = 128 exactly, so no cell is
# resampled, rescaled or dropped: every value in the padded array is either a real LES
# column or a structural zero. The pad extent is written into the record so the loss can
# mask it rather than learn to reproduce a border of zeros.
RASTER_N = 122
RASTER_PAD = 3
RASTER_OUT = RASTER_N + 2 * RASTER_PAD

# THE SIX SCALARS, IN ORDER, AND THE ORDER IS PART OF THE FORMAT.
SCALAR_NAMES = ("h", "ustar", "sigma_v", "L", "sin_wdir", "cos_wdir")


def _m(v):
    """A length in metres for a human-readable warning, or '?' if it was never recorded."""
    return "?" if v is None else f"{float(v):.0f}"


def _split_of_case(a, parent):
    """'train' | 'val' | 'test', ASSIGNED AT GENERATION and written into the record.

    The split is a property of the case's calendar month (lpdm/corpus.py:SPLITS) and it is
    resolved here, once, rather than being derivable downstream from a filename or a date.
    A split re-derived at training time is a split that can silently disagree with the one
    the case was generated under -- and the entire value of a split is that it cannot move.

    The driver passes `--split` because it decided the split BEFORE spending any GPU time.
    That value is not trusted blindly: it is checked against what the case's own timestamp
    implies, and a disagreement is fatal. Either one alone would be a single point of
    failure -- the driver's could be stale, and re-deriving here would defeat the purpose.
    """
    import datetime as dt
    from lpdm.corpus import split_of

    stamp = None
    if a.forcing and os.path.exists(a.forcing):
        try:
            stamp = (json.load(open(a.forcing)).get("provenance") or {}).get("valid_time")
        except (OSError, json.JSONDecodeError):
            stamp = None
    if stamp is None:
        # The tag is case_YYYYMMDDHH by construction (bin/run_corpus_case.sh).
        m_ = re.match(r"^case_(\d{4})(\d{2})(\d{2})(\d{2})$", parent)
        if m_:
            stamp = f"{m_.group(1)}-{m_.group(2)}-{m_.group(3)}T{m_.group(4)}:00:00Z"
    if stamp is None:
        if a.split:
            print(f"  WARNING: no valid_time and no parseable tag, so the split is the "
                  f"driver's '{a.split}' with nothing to check it against")
            return a.split
        raise SystemExit(
            f"FATAL: cannot determine the split for '{parent}': there is no forcing JSON "
            f"to read a valid_time from and the tag is not case_YYYYMMDDHH. A record "
            f"without a split cannot go into the corpus -- assigning one downstream is "
            f"exactly what lpdm/corpus.py exists to prevent.")
    when = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    derived = split_of(when.date())
    if a.split and a.split != derived:
        raise SystemExit(
            f"FATAL: the driver asked for split '{a.split}' but {when.date().isoformat()} "
            f"belongs to '{derived}' (lpdm/corpus.py:SPLITS). One of the two is stale, and "
            f"guessing which would put a case in a split nobody chose.")
    return derived


def _git_commit(root):
    """The commit the corpus was generated at. None rather than a guess if unavailable."""
    head = os.path.join(root, ".git", "HEAD")
    try:
        with open(head) as f:
            ref = f.read().strip()
        if ref.startswith("ref: "):
            p = os.path.join(root, ".git", ref[5:])
            with open(p) as f:
                return f.read().strip()
        return ref
    except OSError:
        return None


def _pad(a):
    """122^2 -> 128^2, zero-padded 3 cells on every side. Refuses anything else."""
    a = np.asarray(a, dtype=np.float32)
    if a.shape != (RASTER_N, RASTER_N):
        raise ValueError(
            f"the raster is {a.shape}, not ({RASTER_N}, {RASTER_N}). The padding to "
            f"{RASTER_OUT}^2 is exact by construction at 122 and is not a resize: at any "
            f"other size it would either crop real cells or need interpolation, and both "
            f"would be silent.")
    return np.pad(a, RASTER_PAD, mode="constant", constant_values=0.0)


def write_training_npz(a, rec, st, fp, npz_path, target, kljun_raster, xc, yc, z0_geom):
    """One self-contained .npz per window, plus a per-machine manifest line."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # THE KLJUN CHANNEL IS RE-EVALUATED HERE, ON THE TARGET RASTER'S OWN EDGES.
    #
    # It could be copied out of the stage-5 .npz, and for a freshly produced case that is
    # the same array. It is recomputed anyway, for two reasons that both matter more than
    # the microsecond it costs. **Identity becomes structural rather than asserted**: the
    # official FFP is evaluated at `xe`/`ye`, the very edges whose midpoints are the
    # target's own cell centres, so the two cannot be on different cells -- where a
    # copied array only carries a promise that they were. And **records regenerated from
    # older runs get the official FFP** rather than whatever Kljun the run was made with;
    # the reimplementation this project used until now is 1.25x wide in sigma_y whenever
    # |L| > 5000, which is exactly the flat/neutral end of the corpus.
    #
    # The stored array is still compared against, and the difference reported -- silently
    # replacing a channel would be the same class of mistake as silently keeping it.
    if target.shape != (len(yc), len(xc)):
        raise ValueError(f"raster {target.shape} does not match its own "
                         f"{len(yc)} x {len(xc)} coordinate axes")
    with np.load(npz_path) as z:
        for nm in ("xc", "yc", "xe", "ye"):
            if nm not in z:
                raise KeyError(
                    f"{npz_path} carries no {nm}; the Kljun channel has to be evaluated on "
                    f"the target's own cell edges and there is no way to do that without "
                    f"them. This is not something to approximate.")
        xe, ye = np.asarray(z["xe"], float), np.asarray(z["ye"], float)
        if not (np.array_equal(np.asarray(z["xc"]), xc)
                and np.array_equal(np.asarray(z["yc"]), yc)):
            raise ValueError("the stage-5 arrays disagree with their own axes")
    # The edges must be the cell boundaries of exactly these centres, or "same cells" is
    # a word rather than a fact.
    if len(xe) != len(xc) + 1 or len(ye) != len(yc) + 1:
        raise ValueError(f"edges ({len(xe)}, {len(ye)}) do not bound centres "
                         f"({len(xc)}, {len(yc)})")
    for nm, e, c in (("x", xe, xc), ("y", ye, yc)):
        mid = 0.5 * (e[1:] + e[:-1])
        if not np.allclose(mid, c, rtol=0, atol=1e-9):
            raise ValueError(f"the {nm} edges' midpoints are not the {nm} cell centres "
                             f"(max |diff| {np.max(np.abs(mid - c)):.3e} m)")

    zm_eff = float(fp.get("zm"))
    ang = np.radians(float(fp["wind_angle"]))
    prof = kljun_ffp.ffp_profile(zm_eff, float(st["h"]), float(st["L"]),
                                 float(st["ustar"]), float(st["sigma_v"]),
                                 umean=float(st["u_mean"]))
    kl_official = kljun_ffp.footprint_on_static(
        xe, ye, ang, zm_eff, float(st["h"]), float(st["ustar"]), float(st["sigma_v"]),
        umean=float(st["u_mean"]), L=float(st["L"]), prof=prof)
    stored = np.asarray(kljun_raster, dtype=np.float64)
    den = max(float(np.abs(stored).max()), 1e-300)
    d_stored = float(np.abs(kl_official - stored).max()) / den
    if d_stored > 1e-12:
        print(f"  the Kljun channel was RE-EVALUATED with the official FFP and differs "
              f"from the array stage 5 stored by {d_stored:.2e} of its peak "
              f"(integral {kl_official.sum() * abs(xc[1]-xc[0]) * abs(yc[1]-yc[0]):.4f} "
              f"against {stored.sum() * abs(xc[1]-xc[0]) * abs(yc[1]-yc[0]):.4f}). That is "
              f"expected for a record regenerated from a run made before the official FFP "
              f"was vendored.")
    kljun_raster = kl_official

    L = float(st["L"])
    wdir = float(st["wdir"])
    th = np.radians(wdir)
    scalars = np.array([float(st["h"]), float(st["ustar"]), float(st["sigma_v"]), L,
                        float(np.sin(th)), float(np.cos(th))], dtype=np.float32)
    # L IS UNBOUNDED AND IS +/-inf AT EXACTLY NEUTRAL, which is a legitimate state and not
    # corruption -- but it cannot go into a network. The vector is written with L because
    # that is the named format; 1/L is written beside it in the meta, is finite everywhere,
    # and is the form the similarity functions actually use. ML_TARGETS.md says the loader
    # substitutes it. Loud here so it can never be a surprise there.
    inv_L = (1.0 / L) if np.isfinite(L) else 0.0
    if not np.isfinite(scalars).all():
        print(f"  WARNING: scalars carry a non-finite value "
              f"({dict(zip(SCALAR_NAMES, scalars))}). L = {L} is legitimate at exactly "
              f"neutral; the training loader must use meta['inv_L'] = {inv_L} in its "
              f"place. Every other non-finite is a fault.")

    fl = (fp.get("floor") or {}).get("health") or {}
    cover = fp.get("cover_share", {}) or {}
    meta = {
        "format": "flux-footprint-pair/1",
        "run_id": a.tag,
        "parent_case": rec["parent"],
        "window_index": rec["window_index"],
        "split_key": rec["split_key"],
        # ASSIGNED AT GENERATION, NEVER DERIVED DOWNSTREAM. See _split_of_case.
        "split": rec["split"],
        # A STUBBED RUN CAN NEVER MASQUERADE AS A CORPUS RECORD. bin/stub_footprint.py
        # stamps `stub` into the footprint JSON and it is carried here, so the flag lives
        # in the artifact rather than in whoever remembers how it was made.
        # bin/check_npz.py FAILS a stubbed record unless it is asked for one.
        "stub": bool(fp.get("stub", False)),
        "datetime": rec.get("forcing", {}).get("valid_time"),
        "gate_state": (rec.get("seed") or {}).get("gate_state", "unjudged"),
        "gate_indeterminate": (rec.get("seed") or {}).get("gate_indeterminate", []),
        "gate_drifting": (rec.get("seed") or {}).get("gate_drifting", []),
        # === z_i: DRIFTING AND ACCEPTED, ON THE NEUTRAL RUNGS, BY DESIGN =================
        # A separate boolean from gate_state so a training loader can filter or weight on
        # it without parsing prose. It marks the ONE limit this project holds to be
        # unsatisfiable rather than failed: a neutral Ekman layer with no capping inversion
        # deepens for several inertial periods (17.6 h each here), so no affordable spin-up
        # reaches an equilibrium depth. The seed is frozen at a FIXED simulated-time ceiling
        # instead -- deterministic and reproducible -- and z_i is a weak input at a 30 m
        # receptor: Kljun's only z_i channel, 1/(1 - z_m/h), spans ~5% over h = 400-1200 m.
        #
        # THIS PAIR'S OWN SCALARS ARE NOT AFFECTED. h comes from window_stats over exactly
        # the 30 minutes the footprint covers, not from the seed. What the acceptance shapes
        # is the corpus's z_i DISTRIBUTION, which is why zi_achieved_m is recorded per case.
        "zi_accepted_drifting": bool(
            (rec.get("seed") or {}).get("zi_accepted_drifting", False)),
        "zi_achieved_m": float(st["h"]),
        "seed_zi_achieved_m": (rec.get("seed") or {}).get("seed_zi_achieved_m"),
        "seed_zi_peakfrac_m": (rec.get("seed") or {}).get("seed_zi_peakfrac_m"),
        # z_i's OWN GATE ROW, because the DRIFTING/INDETERMINATE bucket is a threshold
        # applied at one scoring width and moves with it (measured: +5.76 %/h DRIFTING over
        # 2.0 h, +4.97 %/h INDETERMINATE over 1.5 h, same run). Filter on the trend.
        "zi_gate_verdict": (rec.get("seed") or {}).get("zi_gate_verdict"),
        "zi_trend_pct_per_h": (rec.get("seed") or {}).get("zi_trend_pct_per_h"),
        "zi_trend_se_pct_per_h": (rec.get("seed") or {}).get("zi_trend_se_pct_per_h"),
        "zi_trend_limit_pct_per_h": (rec.get("seed") or {}).get("zi_trend_limit_pct_per_h"),
        "gate_score_h": (rec.get("seed") or {}).get("gate_score_h"),
        "seed_job": (rec.get("seed") or {}).get("job"),
        # -- the diagnostics a record is filtered or weighted by, without opening the JSON
        "integral": fp.get("integral_les"),
        "integral_kljun": fp.get("integral_kljun"),
        "integral_asymptote": fp.get("integral_asymptote"),
        "peak_x_m": (fp.get("les") or {}).get("peak_x"),
        "centroid_dist_m": (fp.get("les") or {}).get("centroid_dist"),
        "centroid_bearing_deg": (fp.get("les") or {}).get("centroid_bearing"),
        "array_share": cover.get("solar array"),
        "array_share_se": (fp.get("cover_share_se") or {}).get("solar array"),
        "cover_share": cover,
        # -- provenance
        "git_commit": _git_commit(root),
        "kljun_source": fp.get("kljun_source",
                               "official FFP v1.42 (third_party/FFP) via lpdm/kljun_ffp.py"),
        "ffp_validity": kljun_ffp.ffp_validity(
            zm_eff, float(st["h"]), float(st["L"]), float(st["ustar"]),
            float(st["sigma_v"]), umean=float(st["u_mean"])),
        "kljun_x_peak_m": float(prof["x_peak"]),
        "kljun_reeval_vs_stored": d_stored,
        # -- the grid, so a record is interpretable with nothing else present
        "grid": {"n": RASTER_N, "pad": RASTER_PAD, "n_padded": RASTER_OUT,
                 "dx_m": float(xc[1] - xc[0]) if len(xc) > 1 else None,
                 "dy_m": float(yc[1] - yc[0]) if len(yc) > 1 else None,
                 "domain_m": float(len(xc) * (xc[1] - xc[0])) if len(xc) > 1 else None,
                 "x_centres_m": [float(xc[0]), float(xc[-1])],
                 "y_centres_m": [float(yc[0]), float(yc[-1])],
                 "frame": "north-up map, receptor at the origin, NOT wind-aligned",
                 "z0_geometric_m": z0_geom,
                 "grid_dir": a.grid},
        # -- the receptor, in BOTH frames, because the padding moves its index
        "receptor": {"z_m": fp.get("zm"), "z_agl_m": fp.get("zm_agl"),
                     "z_target_m": fp.get("z_target"),
                     "d_recept_m": fp.get("d_recept"),
                     "ij_122": _receptor_ij(xc, yc),
                     "ij_128": tuple(v + RASTER_PAD for v in _receptor_ij(xc, yc))},
        # -- the inputs, named, plus the substitution the loader needs
        "scalar_names": list(SCALAR_NAMES),
        "inv_L": inv_L,
        "L_m": L,
        "wdir_deg": wdir,
        "wdir_convention": "meteorological, the direction the wind comes FROM, degrees",
        "u_mean_ms": float(st["u_mean"]),
        # -- THE ESTIMATOR BEHIND h, NAMED IN THE RECORD. h has two definitions in this
        # project that differ by 7-21% (a fixed TKE threshold for the seed gate, a
        # peak-fraction one for the corpus inputs), and a corpus that does not say which it
        # used is a corpus whose h channel cannot be reproduced.
        "h_estimator": "tke_peak_fraction",
        "h_estimator_note": ("5% of the resolved-TKE profile's own peak, bounded by the "
                             "decay minimum (lpdm/les_stats.py:bl_depth). The seed "
                             "stationarity gate uses a FIXED 0.01 m2/s2 threshold instead, "
                             "because it scores a trend; the two differ by 7-21%."),
        "closure": rec["closure"],
        "floor_health": fl,
        "warnings": rec.get("warnings", []),
    }
    os.makedirs(a.npz_dir, exist_ok=True)
    outp = os.path.join(a.npz_dir, f"{a.tag}.npz")
    np.savez_compressed(
        outp,
        scalars=scalars,
        kljun=_pad(kljun_raster),
        target=_pad(target),
        meta=np.array(json.dumps(meta, default=float)),
    )
    # THE MANIFEST IS PER MACHINE, because that is the unit that gets shipped back. It says
    # which cases this machine produced and at which commit and grid, so a corpus assembled
    # from several rented boxes can be checked for gaps and for version skew rather than
    # assumed homogeneous.
    man = os.path.join(a.npz_dir, "manifest.json")
    m = {"format": "flux-footprint-manifest/1", "cases": {}}
    if os.path.exists(man):
        try:
            m = json.load(open(man))
            m.setdefault("cases", {})
        except (OSError, json.JSONDecodeError):
            print(f"  WARNING: {man} was unreadable and is being rewritten")
    m["git_commit"] = meta["git_commit"]
    m["grid"] = meta["grid"]
    m["host"] = os.uname().nodename
    m["cases"][a.tag] = {"parent": rec["parent"], "window_index": rec["window_index"],
                         "split": rec["split"],
                         "datetime": meta["datetime"], "file": os.path.basename(outp),
                         "integral": meta["integral"], "gate_state": meta["gate_state"],
                         # ENUMERABLE WITHOUT OPENING 1825 npz FILES. The z_i distribution
                         # is the thing to check before training on a corpus whose neutral
                         # seeds were frozen mid-growth, so it goes in the manifest line.
                         "zi_achieved_m": meta["zi_achieved_m"],
                         "zi_accepted_drifting": meta["zi_accepted_drifting"]}
    tmp = man + f".tmp.{os.getpid()}"
    json.dump(m, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, man)
    print(f"  npz {outp}  ({os.path.getsize(outp)/1e3:.0f} kB); "
          f"manifest {man}: {len(m['cases'])} case(s)")
    return outp


def _receptor_ij(xc, yc):
    """The receptor's (i, j) in the 122^2 frame: the cell whose centre is nearest 0, 0."""
    return (int(np.argmin(np.abs(np.asarray(xc)))), int(np.argmin(np.abs(np.asarray(yc)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="the case id; becomes run_id")
    ap.add_argument("--grid", default=None,
                    help="the surface directory this case ran on. Its z0m.npy sets the "
                         "recorded geometric-mean z0; without it the field is written as "
                         "null rather than as a number from another grid.")
    ap.add_argument("--footprint", required=True,
                    help="the .json stage5_footprint.py wrote (its .npz sits beside it)")
    ap.add_argument("--forcing", default=None)
    ap.add_argument("--split", default=os.environ.get("CASE_SPLIT") or None,
                    help="the split the DRIVER assigned before spending GPU time. Checked "
                         "against the case's own date; a disagreement is fatal. Defaults to "
                         "$CASE_SPLIT so bin/get_case.sh does not have to thread it.")
    ap.add_argument("--seed", default=None, help="the pick_seed.py JSON")
    ap.add_argument("--outdir", default="pairs")
    ap.add_argument("--npz-dir", default=None,
                    help="write the self-contained .npz training record here (and append "
                         "to its manifest.json). The corpus is generated on rented "
                         "machines that share no filesystem, so a record that references "
                         "results/corpus/<tag>.npz does not survive the trip; this one "
                         "carries its own scalars, both rasters and all of its metadata.")
    ap.add_argument("--copy-npz", action="store_true",
                    help="copy the footprint arrays into the pair rather than "
                         "referencing them; the corpus is then self-contained")
    a = ap.parse_args()

    if not os.path.exists(a.footprint):
        print(f"FATAL: {a.footprint} does not exist", file=sys.stderr)
        return 2
    fp = json.load(open(a.footprint))
    npz_path = a.footprint[:-5] + ".npz"
    if not os.path.exists(npz_path):
        print(f"FATAL: {npz_path} does not exist; the footprint arrays are the TARGET",
              file=sys.stderr)
        return 2

    st = fp.get("stats") or {}
    missing = [k for k in KLJUN_INPUTS if st.get(k) is None]
    if missing:
        print(f"FATAL: the window stats carry no {missing}; without them there is no "
              f"input vector and the record would be a target with no features",
              file=sys.stderr)
        return 2
    inputs = {k: float(st[k]) for k in KLJUN_INPUTS}
    # inf is a legitimate value of L (exactly neutral) and is NOT corruption -- but it
    # cannot go into a model, so it is carried as 1/L, which is finite everywhere and is
    # the form the similarity functions actually use.
    inputs["inv_L"] = (1.0 / inputs["L"]) if np.isfinite(inputs["L"]) else 0.0
    for k, v in inputs.items():
        if k != "L" and not np.isfinite(v):
            print(f"FATAL: input {k} = {v} is not finite", file=sys.stderr)
            return 2

    with np.load(npz_path) as z:
        target = np.asarray(z["les"], dtype=np.float64)
        kljun = np.asarray(z["kljun"], dtype=np.float64)
        xc, yc = np.asarray(z["xc"]), np.asarray(z["yc"])
    if not np.isfinite(target).all():
        print("FATAL: the footprint target is not finite", file=sys.stderr)
        return 2
    if target.shape != (len(yc), len(xc)) and target.shape != (len(xc), len(yc)):
        print(f"FATAL: target {target.shape} does not match the "
              f"{len(yc)} x {len(xc)} raster", file=sys.stderr)
        return 2

    # SPLIT BY PARENT CASE, NOT BY TAG. A case now runs TWO sampling windows inside one
    # FastEddy invocation and yields tags <case>_w0 and <case>_w1. They share a seed, an
    # adjustment, a sounding and a surface, so splitting on the tag would put one in train
    # and the other in validation and leak nearly everything that makes them what they are.
    # The effective sample size for generalisation is the number of PARENTS.
    parent = re.sub(r"_w\d+$", "", a.tag)
    widx = None
    m_ = re.search(r"_w(\d+)$", a.tag)
    if m_:
        widx = int(m_.group(1))
    z0_geom = None
    if a.grid:
        _z0p = os.path.join(a.grid, "z0m.npy")
        if os.path.exists(_z0p):
            z0_geom = float(np.exp(np.log(np.load(_z0p)).mean()))
        else:
            print(f"  WARNING: no z0m.npy in {a.grid}; z0_geometric_m recorded as null")
    else:
        print("  WARNING: no --grid given; z0_geometric_m recorded as null rather than "
              "as a number carried over from another grid")
    rec = {
        "run_id": a.tag,
        "parent": parent,
        "window_index": widx,
        "split_key": parent,     # SPLIT BY RUN. Never by sample, never by window.
        "split": _split_of_case(a, parent),
        "inputs": inputs,
        "target": {"file": os.path.basename(npz_path) if a.copy_npz else npz_path,
                   "array": "les", "shape": list(target.shape),
                   "units": "1/m^2, normalised so the raster integrates to "
                            f"{float(fp.get('integral_les', float('nan'))):.4f}",
                   "reference": "kljun"},
        "site": {"z_recept_m": fp.get("zm"), "z_agl_m": fp.get("zm_agl"),
                 "d_recept_m": fp.get("d_recept"), "z_target_m": fp.get("z_target"),
                 # READ OFF THE GRID THIS CASE RAN ON, never a literal. It was 0.1435 --
                 # the 122^2 @ 16 m value -- and it is 0.0615 at 30 m, because the box
                 # takes in more lake. A constant that is really a grid property is
                 # FASTEDDY_TRAPS.md 19, and it has bitten five times in one day.
                 "z0_geometric_m": z0_geom,
                 "note": "constant across the corpus: a single-tower emulator"},
        "closure": {"sgs_most": fp.get("sgs_most"), "mode": fp.get("sgs_most_mode"),
                    "form": fp.get("sgs_most_form"),
                    "subgrid_weight": fp.get("sgs_subgrid_weight"),
                    "eps_consistent": fp.get("sgs_eps_consistent"),
                    "tback_s": fp.get("tback"), "rel_seconds": fp.get("rel_seconds"),
                    "note": "the near field is closure-dominated at z/Delta ~ 1; the "
                            "sigma_w anchor is worth 46-66% shape L1 against a 38% "
                            "sampling floor (PROJECT_BRIEF.md)"},
        "diagnostics": {"integral_les": fp.get("integral_les"),
                        "integral_kljun": fp.get("integral_kljun"),
                        # THE ASYMPTOTE IS 1 - z_m/z_i, NOT 1 (Steinfeld et al. 2008,
                        # after Horst & Weil 1992). At 30 m in an 800 m boundary layer
                        # that is 3.75%, the size of effects this project gates on.
                        "integral_asymptote": fp.get("integral_asymptote"),
                        "integral_over_asymptote": fp.get("integral_over_asymptote"),
                        "touchdowns": fp.get("touchdowns"),
                        "overlap80_kljun": fp.get("overlap_kljun"),
                        "cover_share": fp.get("cover_share", {}),
                        # THE SHARE TRAVELS WITH ITS OWN SAMPLING SPREAD. A share quoted
                        # without an SE cannot be compared to anything: the h defect moved
                        # the array share 0.8 points against a 3.66-point SE and looked
                        # like a result. PROJECT_BRIEF.md: score a second moment against its own
                        # sampling spread, never against a number you picked.
                        "cover_share_se": fp.get("cover_share_se", {}),
                        "cover_share_groups": fp.get("cover_share_groups_n"),
                        # THE PER-CASE QC STAMP, so a suspect pair is identifiable from
                        # the training record alone rather than only from a log that was
                        # deleted with the fields.
                        "floor_health": ((fp.get("floor") or {}).get("health") or {}),
                        "wind_angle": fp.get("wind_angle")},
    }

    if a.forcing and os.path.exists(a.forcing):
        fc = json.load(open(a.forcing))
        lab = fc["labels"]
        rec["forcing"] = {
            "valid_time": fc["provenance"].get("valid_time"),
            "representable": fc.get("representable"),
            "requested": {"zi_m": lab["zi_m"], "G": lab["G_speed"],
                          "G_dir_from_deg": lab["G_dir_from_deg"],
                          "predicted_10m_dir_deg": lab["predicted_10m_dir_deg"],
                          "hrrr_10m_dir_deg": lab["hrrr_10m_dir_deg"],
                          "wth_virtual": lab["wth_virtual"], "bowen": lab["bowen"]},
            # ACHIEVED MINUS REQUESTED. Not an error term -- the achieved value is the
            # input -- but the distribution of these across 1825 cases is the only way to
            # see whether 30 minutes of adjustment is actually enough.
            "achieved_minus_requested": {
                "zi_m": (None if st.get("h") is None
                         else round(float(st["h"]) - float(lab["zi_m"]), 2)),
                "dir_deg": (None if st.get("wdir") is None else round(
                    float(((float(st["wdir"]) - float(lab["predicted_10m_dir_deg"])
                            + 180.0) % 360.0) - 180.0), 2))},
        }
        if fc.get("representable") is False:
            rec.setdefault("warnings", []).append(
                "the sounding's z_i exceeds what the domain supports; this pair is "
                "domain-constrained and should be excluded or reported as such")

    if a.seed and os.path.exists(a.seed):
        pk = json.load(open(a.seed))
        ch = pk["chosen"]
        rec["seed"] = {"job": ch["job"], "rot": ch["rot"],
                       "d_dir_deg": ch["d_dir_deg"],
                       "seed_zi_m": ch["seed_zi_m"],
                       "labelled_by": ch["labelled_by"],
                       "regime_match": ch["regime_match"],
                       # THE SEED'S GATE STATE TRAVELS WITH EVERY PAIR. A pair built on a
                       # seed whose stationarity was never established is not wrong, but
                       # it is qualified, and the qualification has to survive in the
                       # training record rather than only in a log.
                       "gate_state": ch.get("gate_state", "unjudged"),
                       "gate_indeterminate": ch.get("gate_indeterminate", []),
                       "gate_drifting": ch.get("gate_drifting", []),
                       # z_i ACCEPTED AS DRIFTING, BY DESIGN, ON THE NEUTRAL RUNGS.
                       # A separate flag from gate_state so a consumer can tell the
                       # unsatisfiable-limit case apart from an unexplained drift admitted
                       # by hand: a neutral Ekman layer has no equilibrium depth at any
                       # affordable spin-up, so the seed is frozen at a fixed 2.0 sim-h
                       # ceiling instead. The seed's own achieved depth travels with it.
                       "zi_accepted_drifting": bool(ch.get("zi_accepted_drifting", False)),
                       "drift_reason": ch.get("drift_reason"),
                       "seed_zi_achieved_m": ch.get("seed_zi_achieved_m"),
                       "seed_zi_peakfrac_m": ch.get("seed_zi_peakfrac_m"),
                       # z_i's gate row travels too: whether it reads DRIFTING or
                       # INDETERMINATE depends on the scoring width, so the bucket alone
                       # is not the evidence and the trend is.
                       "zi_gate_verdict": ch.get("zi_gate_verdict"),
                       "zi_trend_pct_per_h": ch.get("zi_trend_pct_per_h"),
                       "zi_trend_se_pct_per_h": ch.get("zi_trend_se_pct_per_h"),
                       "zi_trend_limit_pct_per_h": ch.get("zi_trend_limit_pct_per_h"),
                       "gate_score_h": ch.get("gate_score_h"),
                       # WHERE THE SEED'S HEADING CAME FROM. The adjustment does not close
                       # a direction gap -- measured, it widened one by 10.5 deg -- so the
                       # seed is carried forward at its own drift rate. Recording the
                       # frozen heading, the rate and the horizon makes that reconstructable
                       # per pair instead of only in aggregate.
                       "seed_dir_deg": ch.get("seed_dir_deg"),
                       "seed_dir_frozen_deg": ch.get("seed_dir_frozen_deg"),
                       "dwdir_dt_deg_per_h": ch.get("dwdir_dt_deg_per_h"),
                       "project_h": ch.get("project_h")}
        # THE WARNING BELONGS WHERE `ch` EXISTS. It was written into the --forcing block,
        # which runs BEFORE the seed is read, so it raised UnboundLocalError and killed
        # stage 8 on two finished cases -- after the GPU and the LPDM had both succeeded.
        if ch.get("gate_state") == "INDETERMINATE":
            rec.setdefault("warnings", []).append(
                "the seed this pair restarts from has UNESTABLISHED stationarity: "
                + ", ".join(ch.get("gate_indeterminate") or [])
                + " could not be resolved against their own limits in a 3.0 h spin-up. "
                  "Nothing was drifting; nothing was established either.")
        elif ch.get("zi_accepted_drifting"):
            rec.setdefault("warnings", []).append(
                f"the seed this pair restarts from is DRIFTING in z_i and was accepted "
                f"anyway, by design: a neutral Ekman layer with no capping inversion has no "
                f"equilibrium depth, so the limit is unsatisfiable rather than failed. The "
                f"seed was frozen at a fixed simulated-time ceiling and reached "
                f"z_i = {_m(ch.get('seed_zi_achieved_m'))} m (fixed-threshold) / "
                f"{_m(ch.get('seed_zi_peakfrac_m'))} m (peak-fraction, the corpus "
                f"currency). "
                f"THIS PAIR'S OWN INPUTS ARE UNAFFECTED -- they come from window_stats over "
                f"the same 30 minutes as the footprint -- but the corpus's z_i DISTRIBUTION "
                f"is a property of where the seeds were frozen; check it before training.")
        elif ch.get("gate_state") == "DRIFTING":
            rec.setdefault("warnings", []).append(
                "the seed this pair restarts from is DRIFTING in "
                + ", ".join(ch.get("gate_drifting") or [])
                + " and was admitted by a manual --allow-drifting. This is NOT the "
                  "by-design z_i acceptance: the seed is known to be moving in a "
                  "footprint-controlling parameter and the case starts mid-transient.")

    # ---- THE SELF-CONTAINED TRAINING RECORD ------------------------------------------
    # One .npz per window, carrying everything a training example needs and referencing
    # nothing. The corpus is about to be generated on RENTED machines that share no
    # filesystem with this one and with each other, so a record that points at
    # results/corpus/<tag>.npz is a record that does not survive the trip. ~130 kB each,
    # ~2900 of them, ~380 MB for the whole corpus.
    if a.npz_dir:
        write_training_npz(a, rec, st, fp, npz_path, target, kljun, xc, yc, z0_geom)

    os.makedirs(a.outdir, exist_ok=True)
    if a.copy_npz:
        shutil.copyfile(npz_path, os.path.join(a.outdir, os.path.basename(npz_path)))
    out = os.path.join(a.outdir, f"{a.tag}.json")
    json.dump(rec, open(out, "w"), indent=1, sort_keys=True)
    # One line per case, so 1825 records are enumerable without opening 1825 files.
    #
    # REWRITTEN, NOT APPENDED. bin/run_corpus.sh is resumable and a case can legitimately
    # be re-run -- a fixed seed, a corrected sounding, a killed analysis. A blind append
    # would put the same run_id in the index twice with different inputs, and the second
    # copy would be silently over-weighted by any consumer that iterates the file. The
    # line for this run_id is replaced in place and every other line is preserved
    # byte-for-byte, so a re-run costs nothing and concurrent cases do not clobber
    # each other's rows.
    idx = os.path.join(a.outdir, "index.jsonl")
    row = json.dumps({"run_id": a.tag, "record": os.path.basename(out),
                      "target": rec["target"]["file"],
                      "valid_time": rec.get("forcing", {}).get("valid_time"),
                      "inputs": inputs})
    keep, replaced = [], False
    if os.path.exists(idx):
        for ln in open(idx):
            ln = ln.rstrip("\n")
            if not ln:
                continue
            try:
                same = json.loads(ln).get("run_id") == a.tag
            except json.JSONDecodeError:
                same = False           # keep an unparseable line rather than lose it
            if same:
                replaced = True
                continue
            keep.append(ln)
    keep.append(row)
    tmp = idx + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write("\n".join(keep) + "\n")
    os.replace(tmp, idx)               # atomic, so a kill cannot truncate the index

    print(out)
    print(f"  index {idx}: {len(keep)} record(s)"
          f"{' (this run_id replaced)' if replaced else ''}")
    print(f"  run_id {a.tag}   target {target.shape}  "
          f"integral {rec['diagnostics']['integral_les']}")
    print(f"  inputs: " + "  ".join(f"{k} {v:.4g}" for k, v in inputs.items()))
    if "forcing" in rec:
        d = rec["forcing"]["achieved_minus_requested"]
        print(f"  achieved - requested: z_i {d['zi_m']} m, direction {d['dir_deg']} deg")
    if "seed" in rec:
        print(f"  from seed {rec['seed']['job']} rot {rec['seed']['rot']} "
              f"({rec['seed']['d_dir_deg']:.1f} deg away)")
    for w in rec.get("warnings", []):
        print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

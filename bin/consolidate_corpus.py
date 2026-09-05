#!/usr/bin/env python3
"""Eight machines' worth of records -> one training file. The last step before ML.

    bin/consolidate_corpus.py --npz-dir pairs_npz --manifests manifests \
                              --out corpus_raw.h5

WHAT THIS IS FOR. The corpus is generated on eight rented boxes that are destroyed
afterwards, so what arrives is eight directories of ~40 kB `.npz` files and eight machine
manifests. Nothing merges them, nothing checks that the eight halves agree, and nothing
computes the normalisation the model needs. That is this.

=== WHAT IT REFUSES, AND WHY EACH REFUSAL IS FATAL RATHER THAN A WARNING ===

Every check below can only fail in a way that silently corrupts training, and none of them
can be fixed after the rented machines are gone -- so each is a hard stop with the record
named, not a warning in a log nobody reads twice.

  A SPLIT DISAGREEMENT. The split is re-derived HERE from the case's own datetime via
  lpdm/corpus.py:split_of and compared against the split the record was GENERATED under.
  They must agree. If they ever do not, one of two things is true -- the record was written
  under a different SPLITS table, or its datetime is not its datetime -- and both mean the
  train/test boundary is not where anyone thinks it is. That is the one error in a corpus
  that a good validation score will actively hide.

  A DUPLICATE CASE ID. Two machines must never produce the same case: the partition gives
  each month to exactly one machine. A duplicate means the partition was not respected --
  someone ran the wrong --machine, or two boxes shared a volume -- and the duplicate would
  be trained on twice while its month's real coverage is short.

  A STUBBED RECORD. `meta.stub` marks a record whose LES and LPDM were replaced by an
  analytic blob. It is refused unless --allow-stub, which exists only to test this script.

  A COUNT MISMATCH against the manifests. The manifests say how many cases each machine
  produced; the directory says how many arrived. A difference is a transfer that dropped
  files, and it is reported per machine so it is obvious which rsync to repeat.

=== NORMALISATION IS COMPUTED ON TRAIN ONLY ===

Statistics over the whole corpus leak val and test into the model's input scaling. It is a
small leak and it is the kind that never shows up as a failure -- the score is simply
better than it should be, uniformly, and nothing points at why. Computed over the TRAIN
split alone and written into the file, so the loader cannot recompute them differently.

The target and Kljun channels are normalised on a LOG-MODULUS scale rather than a linear
one, because a footprint spans orders of magnitude and is SIGNED (negative lobes are
physical -- docs/les/lpdm-and-footprint.md -- and carry 5.8-11.1% of |flux|). `asinh(x / s)` with `s` the train
median of |x| over non-pad cells is smooth through zero, preserves sign, and needs no
clipping. The constant `s` is stored; nothing else is applied.

=== THE PAD IS RECORDED SO THE LOSS CAN MASK IT ===

122 -> 128 is a zero-pad of 3 cells, not a resize. Those **1,500** border cells (16,384 -
14,884) are structural zero on both channels and carry no information; a loss that averages
over them is reporting a number **9.2%** of which is the model learning to output zero where
it was told to. The extent is written to `meta/pad_cells` and a ready-made boolean
`meta/valid_mask` is stored beside it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.corpus import SPLIT_MONTH_COUNTS, split_of  # noqa: E402

N, PAD, NG = 128, 3, 122
SCALARS = ("h", "ustar", "sigma_v", "L", "sin_wdir", "cos_wdir")
SPLITS = ("train", "val", "test")

# Scalar meta carried per record. Everything here is either a label the model conditions on
# or a number a later analysis will want to slice by; nothing here is derived at load time.
META_F = ("integral", "peak_x_m", "centroid_dist_m", "array_share", "zi_achieved_m",
          "inv_L", "wdir_deg")
META_S = ("datetime", "parent_case", "split", "gate_state", "run_id", "git_commit",
          "seed_job", "kljun_source", "h_estimator")   # seed_rot is written as int8 below


def die(msg):
    raise SystemExit(f"FATAL: {msg}")


def seed_of(meta):
    """The seed a case was restarted from, wherever make_pair happened to put it."""
    s = meta.get("seed")
    if isinstance(s, dict):
        return (str(s.get("job") or s.get("name") or ""), s.get("rot"))
    return (str(meta.get("seed_job") or ""), meta.get("seed_rot"))


def load_one(path, allow_stub):
    with np.load(path, allow_pickle=True) as z:
        missing = {"scalars", "kljun", "target", "meta"} - set(z.files)
        if missing:
            die(f"{path}: missing array(s) {sorted(missing)}")
        sc = np.asarray(z["scalars"], dtype=np.float32)
        kl = np.asarray(z["kljun"], dtype=np.float32)
        tg = np.asarray(z["target"], dtype=np.float32)
        meta = json.loads(str(z["meta"]))

    if meta.get("stub") and not allow_stub:
        die(f"{path}: meta.stub is true. This record's LES and LPDM were STUBBED; its "
            f"target is an analytic blob, not a footprint. It must not enter a training "
            f"file. (--allow-stub exists only to test this script.)")
    if sc.shape != (6,):
        die(f"{path}: scalars is {sc.shape}, expected (6,)")
    for nm, a in (("kljun", kl), ("target", tg)):
        if a.shape != (N, N):
            die(f"{path}: {nm} is {a.shape}, expected ({N}, {N})")
        if not np.isfinite(a).all():
            die(f"{path}: {nm} carries {int((~np.isfinite(a)).sum())} non-finite cells")
        border = np.concatenate([a[:PAD].ravel(), a[-PAD:].ravel(),
                                 a[:, :PAD].ravel(), a[:, -PAD:].ravel()])
        if np.any(border != 0.0):
            die(f"{path}: {nm} has {int((border != 0).sum())} non-zero cells in the "
                f"{PAD}-cell pad. The pad is structural: if it is not zero, the two "
                f"channels are not on the cells the record says they are.")
    for i, nm in enumerate(SCALARS):
        if nm != "L" and not np.isfinite(sc[i]):
            die(f"{path}: scalar '{nm}' is {sc[i]}")

    # === THE SPLIT, RE-DERIVED AND CROSS-CHECKED ======================================
    stamp = meta.get("datetime")
    if not stamp:
        die(f"{path}: meta.datetime is absent, so its split cannot be re-derived")
    when = dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).replace(tzinfo=None)
    derived = split_of(when.date())
    stored = meta.get("split")
    if stored != derived:
        die(f"{path}: SPLIT DISAGREEMENT. The record was generated as {stored!r} but its "
            f"own datetime {when.isoformat()} belongs to {derived!r} under "
            f"lpdm/corpus.py:SPLITS. One of the two is wrong and both readings put the "
            f"train/test boundary somewhere nobody chose. This is not repairable here.")
    if when.minute or when.second:
        die(f"{path}: meta.datetime {stamp} is not a round hour")
    g = meta.get("grid") or {}
    if (g.get("n"), g.get("pad"), g.get("n_padded")) != (NG, PAD, N):
        die(f"{path}: meta.grid is n={g.get('n')} pad={g.get('pad')} "
            f"n_padded={g.get('n_padded')}, expected {NG}/{PAD}/{N}")
    if g.get("dx_m") is not None and abs(float(g["dx_m"]) - 30.0) > 1e-6:
        die(f"{path}: meta.grid.dx_m is {g['dx_m']}, not the 30 m production spacing -- "
            f"this record came off a retired grid and its cells mean something else")
    return sc, kl, tg, meta, derived, when


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Merge the per-machine .npz records into one training file.")
    ap.add_argument("--npz-dir", default="pairs_npz",
                    help="directory of case_YYYYMMDDHH.npz, or a parent of per-machine "
                         "directories -- both are searched")
    ap.add_argument("--manifests", default="manifests",
                    help="directory of the eight machine manifests")
    ap.add_argument("--out", default="corpus_raw.h5")
    ap.add_argument("--allow-stub", action="store_true",
                    help="admit records marked meta.stub. FOR TESTING THIS SCRIPT ONLY; "
                         "the output is stamped so it can never pass for a corpus.")
    ap.add_argument("--compression", default="gzip", choices=("gzip", "lzf", "none"),
                    help="gzip+shuffle is the default; lzf is ~4x faster to write and ~10% "
                         "larger, none is for a quick look")
    ap.add_argument("--level", type=int, default=4, help="gzip level 1-9")
    a = ap.parse_args()

    try:
        import h5py
    except ImportError:
        die("h5py is not installed. The analysis stack lives in the container:\n"
            "  docker run --rm -v $PWD:/w -w /w ghcr.io/tyatharva/flux-seeds:corpus \\\n"
            "         python3 bin/consolidate_corpus.py --out /w/corpus_raw.h5")

    # ---- find the records ------------------------------------------------------------
    files = sorted(glob.glob(os.path.join(a.npz_dir, "*.npz")))
    if not files:
        files = sorted(glob.glob(os.path.join(a.npz_dir, "*", "*.npz")))
    files = [f for f in files if not f.endswith(".REJECTED")]
    if not files:
        die(f"no .npz under {a.npz_dir} (looked one level down too)")
    print(f"  {len(files)} record(s) under {a.npz_dir}")

    # ---- the manifests: what SHOULD have arrived --------------------------------------
    mans = sorted(glob.glob(os.path.join(a.manifests, "*.json")))
    expected, miss_by_split, mach_cases = 0, Counter(), {}
    if not mans:
        print(f"  WARNING: no manifests under {a.manifests}. The count check below cannot "
              f"run, so a transfer that dropped files will not be detected here.")
    for mp in mans:
        m = json.load(open(mp))
        mid = m.get("machine", os.path.basename(mp))
        n = sum(1 for v in m.get("days", {}).values() if v.get("status") == "case")
        mach_cases[mid] = n
        expected += n
        for v in m.get("days", {}).values():
            if v.get("status") != "case":
                miss_by_split[(v.get("split"), v.get("status"))] += 1
    if mans:
        print(f"  {len(mans)} manifest(s): {expected} case(s) expected across machines "
              # SORT ON str(k). `mid` falls back to a FILENAME when a manifest carries no
              # "machine" key, so the key set can be mixed int/str and a bare sorted() dies
              # with TypeError -- at the last step before ML, on a corpus that is fine.
              + ", ".join(f"m{k}={v}" for k, v in sorted(mach_cases.items(),
                                                         key=lambda kv: str(kv[0]))))

    # ---- load, check, and index -------------------------------------------------------
    recs, seen = [], {}
    for f in files:
        sc, kl, tg, meta, derived, when = load_one(f, a.allow_stub)
        # KEYED ON run_id, NOT parent_case. With N_WINDOWS = 2 a case yields TWO records
        # -- `<case>_w0` and `<case>_w1` -- which deliberately SHARE a parent_case and are
        # separate training pairs (docs/les/lpdm-and-footprint.md: "both footprints are separate training pairs,
        # tagged by parent"). Keying on the parent refuses a perfectly good two-window
        # corpus. run_id is unique per record and is still identical between two machines
        # that generated the same day, which is the failure this is for.
        tag = meta.get("run_id") or os.path.splitext(os.path.basename(f))[0]
        if tag in seen:
            die(f"DUPLICATE RECORD {tag}: {seen[tag]} and {f}. Each of the 64 corpus months "
                f"belongs to exactly one machine (lpdm/partition.py), so a duplicate means "
                f"two boxes generated the same day -- check that each was given a distinct "
                f"--machine, and that no two shared an output volume. (Two records of one "
                f"case at N_WINDOWS = 2 are NOT this: they differ in run_id.)")
        seen[tag] = f
        recs.append((when, tag, sc, kl, tg, meta, derived))

    # THE MANIFEST COUNTS CASES; AT N_WINDOWS = 2 THERE ARE TWO RECORDS PER CASE. Compare
    # distinct parent cases against the manifests, and report the windows-per-case factor
    # rather than failing a correct two-window corpus on a count that was never records.
    n_parent = len({(r[5].get("parent_case") or r[1]) for r in recs})
    if mans and n_parent != expected:
        short = {k: v for k, v in mach_cases.items()}
        die(f"COUNT MISMATCH: {n_parent} distinct case(s) on disk ({len(recs)} record(s)) "
            f"against {expected} the "
            f"manifests account for. A transfer dropped files. Per machine the manifests "
            f"say {short}; re-run the rsync for whichever is short. Nothing here can tell "
            f"which files are missing, but the manifests name every case: compare "
            f"`jq -r '.days[]|select(.status==\"case\").tag' manifests/*.json | sort` "
            f"against `ls {a.npz_dir}`.")

    recs.sort(key=lambda r: (r[0], r[1]))          # chronological, stable
    n = len(recs)
    if n != n_parent:
        print(f"  {n} record(s) over {n_parent} case(s) -- "
              f"{n / n_parent:.2f} windows per case (N_WINDOWS > 1)")
    print(f"  {n} record(s) pass every check")

    by_split = Counter(r[6] for r in recs)
    print(f"  by split: " + ", ".join(f"{s} {by_split.get(s, 0)}" for s in SPLITS))
    if by_split.get("train", 0) == 0:
        die("the train split is empty, so no normalisation can be computed from it")

    # ---- assemble ---------------------------------------------------------------------
    sc = np.stack([r[2] for r in recs]).astype(np.float32)
    kl = np.stack([r[3] for r in recs]).astype(np.float32)
    tg = np.stack([r[4] for r in recs]).astype(np.float32)
    split_i = np.array([SPLITS.index(r[6]) for r in recs], dtype=np.int8)
    train = split_i == 0

    valid = np.zeros((N, N), dtype=bool)
    valid[PAD:PAD + NG, PAD:PAD + NG] = True

    # ---- normalisation, TRAIN ONLY ----------------------------------------------------
    # Scalars: mean/std over train. `L` is +/-inf at exactly neutral, so the model is given
    # 1/L -- which is finite everywhere and monotone through neutral -- and the raw L is
    # kept in the file for reference rather than fed in.
    sc_tr = sc[train]
    with np.errstate(invalid="ignore"):
        sc_mean = np.nanmean(np.where(np.isfinite(sc_tr), sc_tr, np.nan), axis=0)
        sc_std = np.nanstd(np.where(np.isfinite(sc_tr), sc_tr, np.nan), axis=0)
    sc_std = np.where(sc_std > 0, sc_std, 1.0).astype(np.float32)
    sc_mean = np.nan_to_num(sc_mean).astype(np.float32)

    # Rasters: a signed log-modulus scale, `y = asinh(x / s)`.
    #
    # `s` IS THE MEDIAN OVER TRAIN RECORDS OF EACH RECORD'S PEAK |x|, NOT THE MEDIAN OVER
    # CELLS. The first version took the median of |x| across every valid cell of every
    # train record and produced s = 7.9e-23, which is not a scale, it is the noise floor:
    # a footprint is a compact blob in a 122^2 frame, so the median CELL is deep in a tail
    # that is orders of magnitude below the peak, and dividing by it would push every cell
    # that matters into the linear-in-log regime of asinh and blow the dynamic range.
    #
    # The per-record peak is the quantity the raster is actually made of, and taking its
    # median across records is robust to the one case with an anomalous spike. It maps a
    # typical peak to asinh(1) = 0.88 and leaves the tail near zero, which is what a
    # residual model wants. Sign is preserved: negative lobes are physical and carry
    # 5.8-11.1% of |flux| (docs/les/lpdm-and-footprint.md), so nothing is clipped or absolute-valued.
    def scale_of(arr):
        peaks = np.abs(arr[train][:, valid]).max(axis=1)
        peaks = peaks[peaks > 0]
        return float(np.median(peaks)) if peaks.size else 1.0
    s_kl, s_tg = scale_of(kl), scale_of(tg)

    print(f"  normalisation from the TRAIN split alone ({int(train.sum())} records):")
    print(f"    scalars  mean {np.array2string(sc_mean, precision=3)}")
    print(f"    scalars  std  {np.array2string(sc_std, precision=3)}")
    print(f"    asinh scale  kljun {s_kl:.4e}   target {s_tg:.4e}")

    # ---- write ------------------------------------------------------------------------
    comp = None if a.compression == "none" else a.compression
    # SHUFFLE IS ON, and it is worth 20% for nothing. The byte-shuffle filter groups the
    # like-significance bytes of adjacent float32s before the compressor sees them, and on
    # smooth rasters that is most of what gzip can find. MEASURED on 731 records: gzip-4
    # alone 1.32x, gzip-4 + shuffle 1.65x, gzip-9 + shuffle 1.67x for 3x the write time.
    copts = {"compression": comp, "shuffle": comp is not None}
    if comp == "gzip":
        copts["compression_opts"] = a.level
    ck = (min(32, n), N, N)                # chunk on the sample axis: how a loader reads

    if os.path.exists(a.out):
        os.remove(a.out)
    with h5py.File(a.out, "w") as h:
        h.attrs["format"] = "flux-footprint-corpus/1"
        h.attrs["n"] = n
        h.attrs["created_utc"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        h.attrs["stub"] = bool(a.allow_stub)
        # The corpus ships as TWO files with identical layout: this one, whose `target` is
        # the LES field as produced, and corpus_cone.h5, whose `target` has the periodic
        # wraparound cropped out (bin/mask_cone.py). A loader points at one or the other.
        h.attrs["variant"] = "raw"
        if a.allow_stub:
            h.attrs["STUB_WARNING"] = ("built with --allow-stub: one or more records are "
                                       "NOT simulations. This file is not a corpus.")
        h.create_dataset("scalars", data=sc, chunks=(min(256, n), 6), **copts)
        h.create_dataset("kljun", data=kl, chunks=ck, **copts)
        h.create_dataset("target", data=tg, chunks=ck, **copts)

        m = h.create_group("meta")
        vs = h5py.string_dtype("utf-8")
        for k in META_S:
            if k == "seed_job":
                vals = [seed_of(r[5])[0] for r in recs]
            elif k == "split":
                vals = [r[6] for r in recs]
            else:
                vals = [str(r[5].get(k, "")) for r in recs]
            m.create_dataset(k, data=np.array(vals, dtype=object), dtype=vs, **copts)
        for k in META_F:
            vals = [float(r[5].get(k)) if r[5].get(k) is not None else np.nan
                    for r in recs]
            m.create_dataset(k, data=np.array(vals, dtype=np.float32), **copts)
        # -1 means THE RECORD DID NOT CARRY IT, not rotation -1. bin/make_pair.py began
        # persisting `seed_rot` on 2026-08-31; anything written before that has the seed's
        # name but not which of the four 90-degree re-indexes was used, and the two are
        # only reproducible together.
        rots = [seed_of(r[5])[1] for r in recs]
        m.create_dataset("seed_rot", data=np.array(
            [int(x) if x is not None else -1 for x in rots], dtype=np.int8))
        m.attrs["seed_rot_note"] = ("-1 means the record predates make_pair.py persisting "
                                    "the rotation; it is not a rotation value")
        n_norot = sum(1 for x in rots if x is None)
        if n_norot:
            print(f"  NOTE: {n_norot} of {len(recs)} record(s) carry no seed rotation "
                  f"(written before make_pair.py persisted it); stored as -1")
        m.create_dataset("split_index", data=split_i)
        m.attrs["split_order"] = list(SPLITS)
        m.create_dataset("scalar_names", data=np.array(list(SCALARS), dtype=object),
                         dtype=vs)

        # THE PAD, SO A LOSS CAN MASK IT. Both the number and a ready-made mask: a
        # consumer that reads only one of the two cannot get them inconsistent.
        m.attrs["pad_cells"] = PAD
        m.attrs["grid_n"] = NG
        m.attrs["grid_n_padded"] = N
        m.attrs["pad_is_structural_zero"] = True
        m.attrs["pad_note"] = (
            f"122 -> 128 is a ZERO-PAD of {PAD} cells, not a resize. The border carries no "
            f"information; a loss averaged over it reports a number "
            f"{100 * (1 - NG * NG / (N * N)):.1f}% of which is the model learning to emit "
            f"zero where it was told to. Mask with meta/valid_mask.")
        m.create_dataset("valid_mask", data=valid, **copts)

        g0 = (recs[0][5].get("grid") or {})
        gg = h.create_group("grid")
        for k, v in (("n", NG), ("pad", PAD), ("n_padded", N),
                     ("dx_m", g0.get("dx_m", 30.0)), ("dy_m", g0.get("dy_m", 30.0)),
                     ("domain_m", g0.get("domain_m", 3660.0)),
                     ("receptor_z_m", (recs[0][5].get("receptor") or {}).get("z_m", 30.0))):
            gg.attrs[k] = v

        nm = h.create_group("norm")
        nm.attrs["computed_on"] = "train split only"
        nm.attrs["n_train"] = int(train.sum())
        nm.attrs["why"] = ("statistics over the whole corpus leak val and test into the "
                           "input scaling -- a small leak that never shows up as a "
                           "failure, only as a uniformly better score")
        nm.create_dataset("scalars_mean", data=sc_mean)
        nm.create_dataset("scalars_std", data=sc_std)
        nm.attrs["raster_transform"] = "y = arcsinh(x / s), signed, no clipping"
        nm.attrs["raster_scale_note"] = (
            "s is the median, over TRAIN records, of each record's PEAK |x| within the "
            "valid (non-pad) frame -- NOT the median over cells, which is a tail value "
            "orders of magnitude below the signal. asinh is used rather than log because "
            "the footprint is signed: negative lobes are physical and nothing clips them.")
        nm.attrs["kljun_scale"] = s_kl
        nm.attrs["target_scale"] = s_tg

        cm = h.create_group("counts")
        for s_ in SPLITS:
            cm.attrs[f"cases_{s_}"] = int(by_split.get(s_, 0))
        for (sp, st), c in miss_by_split.items():
            cm.attrs[f"days_{sp}_{st}"] = int(c)
        cm.attrs["expected_from_manifests"] = expected
        cm.attrs["n_manifests"] = len(mans)

    size = os.path.getsize(a.out)
    print()
    print(f"  === {a.out}: {n} records, {size / 1e6:.1f} MB "
          f"({a.compression}{'-' + str(a.level) if comp == 'gzip' else ''}) ===")
    raw = sc.nbytes + kl.nbytes + tg.nbytes
    print(f"      raw arrays {raw / 1e6:.1f} MB -> {100 * size / raw:.1f}% "
          f"({raw / size:.1f}x compression)")
    print(f"      per record {size / n / 1e3:.1f} kB")
    print()
    print("  cases and missing days by split")
    print(f"    {'split':6s} {'cases':>7s} {'missing':>9s} {'failed':>8s} "
          f"{'not reached':>12s}")
    for s_ in SPLITS:
        print(f"    {s_:6s} {by_split.get(s_, 0):7d} "
              f"{miss_by_split.get((s_, 'missing'), 0):9d} "
              f"{miss_by_split.get((s_, 'failed'), 0):8d} "
              f"{miss_by_split.get((s_, 'not reached'), 0):12d}")
    print(f"    (the corpus is {SPLIT_MONTH_COUNTS['train']}/{SPLIT_MONTH_COUNTS['val']}/"
          f"{SPLIT_MONTH_COUNTS['test']} months train/val/test)")
    if a.allow_stub:
        print("\n  *** --allow-stub: this file contains stubbed records and is NOT a "
              "corpus. h.attrs['stub'] is set. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

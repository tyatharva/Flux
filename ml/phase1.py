"""Phase 1: the exploration matrix, run concurrently on one GPU.

    python -m ml.phase1 --round 1 -K 4            # the built-in one-factor matrix
    python -m ml.phase1 --runs my_runs.json -K 4  # any {name: {overrides}} mapping
    python -m ml.phase1 --summarise               # rebuild the tables from run.json files

Every run is `python -m ml.train` in its own process; K of them at a time. A monitor
thread samples nvidia-smi every 5 s into gpu_util.csv, so the utilisation and the
concurrency that was actually achieved are recorded, not assumed. The baseline is run with
several seeds and every comparison is quoted in units of that seed spread (PROJECT_BRIEF.md
standing rule 5: a tolerance measured from one difference is not a tolerance).
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
OUT_DEFAULT = os.path.join(REPO, "results", "ml", "phase1")

BASE = dict(epochs=80, patience=20, eval_every=20, save_ckpt=False)

ROUND1 = {
    "b0_s0": dict(seed=0), "b0_s1": dict(seed=1), "b0_s2": dict(seed=2), "b0_s3": dict(seed=3),
    "modes8": dict(modes=8), "modes12": dict(modes=12), "modes24": dict(modes=24),
    "modes32": dict(modes=32),
    "width16": dict(width=16), "width64": dict(width=64),
    "depth2": dict(depth=2), "depth6": dict(depth=6),
    "local_none": dict(local="none"), "local_conv3x3": dict(local="conv3x3"),
    "head_direct": dict(head="direct"),
    "norm_record": dict(norm_mode="record"),
    "knee0.3": dict(knee=0.3), "knee3": dict(knee=3.0),
    "dist_none": dict(dist="none"), "dist_xy": dict(dist="lin_exp_xy"),
    "statics_none": dict(statics="none"), "statics_C": dict(statics="C"),
    "statics_rot90": dict(statics="B_rot90"),
    "stab_L": dict(stab="L"),
    "peak0.1": dict(lam_peak=0.1), "peak1": dict(lam_peak=1.0),
    "int0.1_tgt": dict(lam_int=0.1, int_ref="target"),
    "int1_tgt": dict(lam_int=1.0, int_ref="target"),
    "int0.1_asym": dict(lam_int=0.1, int_ref="asymptote"),
    "int1_asym": dict(lam_int=1.0, int_ref="asymptote"),
    "weight_north3": dict(weight="north3"),
}

FACTORS = {   # ablation name -> the factor it varies, for the summary grouping
    "modes": "modes", "width": "width", "depth": "depth", "local": "local",
    "head": "head", "norm": "norm_mode", "knee": "knee", "dist": "dist",
    "statics": "statics", "stab": "stab", "peak": "lam_peak", "int": "lam_int",
    "weight": "weight",
}


class GpuMonitor(threading.Thread):
    def __init__(self, path, every=5.0):
        super().__init__(daemon=True)
        self.path, self.every, self.stop = path, every, threading.Event()

    def run(self):
        with open(self.path, "a") as fh:
            w = csv.writer(fh)
            if os.path.getsize(self.path) == 0:
                w.writerow(["utc", "util_pct", "mem_used_mib", "n_procs"])
            while not self.stop.is_set():
                try:
                    q = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits"], text=True).strip().split(", ")
                    p = subprocess.check_output(
                        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                        text=True).strip().splitlines()
                    w.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                q[0], q[1], len([x for x in p if x.strip()])])
                    fh.flush()
                except Exception:
                    pass
                self.stop.wait(self.every)


def run_one(name, overrides, outdir, base):
    out = os.path.join(outdir, name)
    if os.path.exists(os.path.join(out, "run.json")):
        return name, "cached", 0.0
    cfg = dict(base)
    cfg.update(overrides)
    args = [PY, "-m", "ml.train", "--out", out] + [f"--set={k}={v}" for k, v in cfg.items()]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(out, "log.txt"), "w") as log:
        rc = subprocess.call(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    ok = rc == 0 and os.path.exists(os.path.join(out, "run.json"))
    return name, ("ok" if ok else f"FAILED rc={rc}"), time.time() - t0


def campaign(runs, outdir, K, base, tag):
    os.makedirs(outdir, exist_ok=True)
    mon = GpuMonitor(os.path.join(outdir, "gpu_util.csv"))
    mon.start()
    t0 = time.time()
    results = []
    print(f"phase1[{tag}]: {len(runs)} runs, K={K}, base={base}")
    with ThreadPoolExecutor(max_workers=K) as ex:
        futs = [ex.submit(run_one, n, o, outdir, base) for n, o in runs.items()]
        for f in futs:
            name, status, dt = f.result()
            print(f"  {name:16s} {status:12s} {dt:6.0f} s")
            results.append(dict(name=name, status=status, wall=dt))
    mon.stop.set()
    mon.join(timeout=10)
    wall = time.time() - t0
    with open(os.path.join(outdir, f"campaign_{tag}.json"), "w") as fh:
        json.dump(dict(tag=tag, K=K, base=base, wall_s=wall, runs=results,
                       started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))),
                  fh, indent=1)
    print(f"phase1[{tag}]: {wall:.0f} s wall for {len(runs)} runs at K={K}")
    return results


def gpu_summary(outdir, t_from=None):
    path = os.path.join(outdir, "gpu_util.csv")
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path)))
    if t_from:
        rows = [r for r in rows if r["utc"] >= t_from]
    if not rows:
        return {}
    u = np.array([float(r["util_pct"]) for r in rows])
    m = np.array([float(r["mem_used_mib"]) for r in rows])
    p = np.array([float(r["n_procs"]) for r in rows])
    return dict(n_samples=len(u), util_mean=float(u.mean()), util_p50=float(np.median(u)),
                util_p90=float(np.percentile(u, 90)), mem_mean_mib=float(m.mean()),
                mem_max_mib=float(m.max()), procs_mean=float(p.mean()),
                procs_max=int(p.max()))


def load_runs(outdir):
    out = {}
    for p in sorted(glob.glob(os.path.join(outdir, "*", "run.json"))):
        with open(p) as fh:
            out[os.path.basename(os.path.dirname(p))] = json.load(fh)
    return out


def summarise(outdir, baseline_prefix="b0_s"):
    runs = load_runs(outdir)
    if not runs:
        print("no runs")
        return
    keys = ["val_loss", "val_mse_ref", "composite", "composite_north"]
    rows = []
    for name, r in runs.items():
        rows.append(dict(name=name, val_loss=r["val_loss"], val_mse_ref=r["val_mse_ref"],
                         composite=r["composite"], composite_north=r["composite_north"],
                         gap_ratio=r["gap"]["loss_ratio"], best_epoch=r["best_epoch"],
                         epochs=r["epochs_run"], wall=r["wall_s"], params=r["n_params"],
                         **{"r_" + k: v for k, v in r["composite_ratios"].items()},
                         **{"rn_" + k: v for k, v in r["composite_north_ratios"].items()}))
    base = [x for x in rows if x["name"].startswith(baseline_prefix)]
    spread = {}
    for k in keys:
        v = np.array([x[k] for x in base])
        spread[k] = dict(mean=float(v.mean()) if len(v) else np.nan,
                         sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan, n=len(v))
    fields = ["name", "val_loss", "val_mse_ref", "composite", "composite_north", "gap_ratio",
              "best_epoch", "epochs", "wall", "params"] + \
             ["r_" + k for k in ("peak_x", "centroid", "overlap80", "array_share", "integral")] + \
             ["rn_" + k for k in ("peak_x", "centroid", "overlap80", "array_share", "integral")]
    with open(os.path.join(outdir, "summary.tsv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for x in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in x.items()})
    gpu = gpu_summary(outdir)
    lines = ["# Phase 1 summary", "",
             f"{len(rows)} runs in `{os.path.relpath(outdir, REPO)}`; baseline `{baseline_prefix}*` "
             f"n = {spread['val_loss']['n']}.", "",
             "## Baseline seed spread", "", "| quantity | mean | sd | n |", "|---|---|---|---|"]
    for k in keys:
        s = spread[k]
        lines.append(f"| {k} | {s['mean']:.6g} | {s['sd']:.3g} | {s['n']} |")
    if gpu:
        lines += ["", "## GPU", "",
                  f"utilisation mean {gpu['util_mean']:.0f}% / p50 {gpu['util_p50']:.0f}% / "
                  f"p90 {gpu['util_p90']:.0f}%; memory mean {gpu['mem_mean_mib']:.0f} MiB, "
                  f"max {gpu['mem_max_mib']:.0f} MiB; concurrent processes mean "
                  f"{gpu['procs_mean']:.1f}, max {gpu['procs_max']} ({gpu['n_samples']} samples)"]
    lines += ["", "## Runs, as z-scores against the baseline seed spread (negative = better "
              "for losses and composites)", "",
              "| run | val_mse_ref | z | composite | z | composite_north | z | gap x | "
              "best/epochs | r_array | rn_array | r_centroid | r_overlap | r_integral |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def z(k, v):
        s = spread[k]
        return (v - s["mean"]) / s["sd"] if s["sd"] and np.isfinite(s["sd"]) and s["sd"] > 0 \
            else np.nan
    for x in sorted(rows, key=lambda x: x["val_mse_ref"]):
        lines.append(
            f"| {x['name']} | {x['val_mse_ref']:.6g} | {z('val_mse_ref', x['val_mse_ref']):+.1f} | "
            f"{x['composite']:.3f} | {z('composite', x['composite']):+.1f} | "
            f"{x['composite_north']:.3f} | {z('composite_north', x['composite_north']):+.1f} | "
            f"{x['gap_ratio']:.2f} | {x['best_epoch']}/{x['epochs']} | "
            f"{x['r_array_share']:.2f} | {x['rn_array_share']:.2f} | {x['r_centroid']:.2f} | "
            f"{x['r_overlap80']:.2f} | {x['r_integral']:.2f} |")
    # rank correlation between the two selection criteria
    a = np.array([x["val_mse_ref"] for x in rows])
    b = np.array([x["composite"] for x in rows])
    from scipy.stats import spearmanr
    rho = spearmanr(a, b).correlation if len(rows) > 3 else np.nan
    rho_n = spearmanr(a, [x["composite_north"] for x in rows]).correlation if len(rows) > 3 \
        else np.nan
    lines += ["", f"Spearman rank correlation across runs: val_mse_ref vs composite "
              f"{rho:+.2f}, vs composite_north {rho_n:+.2f}."]
    with open(os.path.join(outdir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows, spread


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--runs", default=None, help="JSON {name: {overrides}}")
    ap.add_argument("-K", type=int, default=4)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    ap.add_argument("--base", default=None, help="JSON of base overrides for every run")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated run names to run")
    ap.add_argument("--baseline-prefix", default="b0_s",
                    help="runs whose names start with this are the seed-spread reference")
    a = ap.parse_args(argv)
    if a.summarise:
        summarise(a.outdir, a.baseline_prefix)
        return 0
    base = dict(BASE)
    if a.base:
        base.update(json.loads(a.base))
    if a.runs:
        with open(a.runs) as fh:
            runs = json.load(fh)
    elif a.round == 1:
        runs = dict(ROUND1)
    else:
        sys.exit("give --round 1, --runs FILE or --summarise")
    if a.only:
        keep = a.only.split(",")
        runs = {k: v for k, v in runs.items() if k in keep}
    campaign(runs, a.outdir, a.K, base, a.tag or (f"round{a.round}" if a.round else "custom"))
    summarise(a.outdir, a.baseline_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())

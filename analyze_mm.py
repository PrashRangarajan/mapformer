"""Analysis of the Memory Maze batch (MEMORY_MAZE_PREREG.md). Reads <root>/<ARM>_s<seed>/results.json.

  python3 -m mapformer.analyze_mm --root <run root> --floors-eval <json> --floors-held <json> [--out MM_VERDICTS.json]
  python3 -m mapformer.analyze_mm --selftest            (synthetic batches through every label; no run dirs)

Registered metric per run: R1 accuracy on non-local blind re-entries of the official eval split (results.json
eval.normal.nonlocal.R1). n = 8 seeds per arm (3600-3607). Two-sample exact permutation p on per-seed means (stats_core).

REGISTERED PRIMARY, P_E vs R1, d = mean(P_E) - mean(R1):
  - INCOMPLETE  (no label) if any of P_E, R1, P_odo has fewer than 8 finished runs.
  - UNREADABLE  if mean(P_odo) < floor_eval + 0.10: with given odometry the task is not learned, so no contrast reads.
  - PATH WINS   if d >= +0.05 and p < .05.
  - INDEX WINS  if d <= -0.05 and p < .05.
  - UNMEASURED  otherwise (the exact-t MDE is printed beside it).
REGISTERED SECONDARIES (Holm over the four permutation p's; same 0.05 effect bar): S1 P_E vs R4, S2 P_E vs GRU,
S3 P_odo vs P_E, S4 P_odo vs GPS. Label '<A> ABOVE <B>' / '<B> ABOVE <A>' / 'UNMEASURED' (A is the first named).
  An arm with fewer than 8 runs makes its secondary 'NOT RUN'.
DECLARED DESCRIPTIVES (no verdict): every arm minus the eval floor (the gate's best map-free floor on the same split);
the primary rule replicated on the held-out train-part0 1600-2899 split (labelled HELD, never replacing the primary);
R2; context destruction (frames swapped across episodes; step features shuffled in time); P_raw vs P_E if run;
offline probe (Walls accuracy %, Objects MSE) beside the constant baseline and the published 15x15 rows;
loss-curve tail (final-10% relative slope; DESCENDING if < -0.5% per 10% of training).
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

from mapformer.stats_core import perm2_p, mde
from mapformer.analyze_lrsweep import holm

ARMS = ("P_E", "P_odo", "P_raw", "R1", "R4", "GPS", "GRU")
N_REQ = 8
BAR = 0.05
SECONDARIES = (("S1", "P_E", "R4"), ("S2", "P_E", "GRU"), ("S3", "P_odo", "P_E"), ("S4", "P_odo", "GPS"))
PUBLISHED_15 = {"constant": (78.3, 64.8), "VAE (no memory)": (78.7, 56.2), "RSSM-TBTT (Pasukonis)": (80.5, 33.8),
                "RSSM-TBTT (Deng)": (81.7, 32.6), "TSSM-XL": (79.8, 35.1), "S4WM": (80.2, 34.8), "S5WM": (81.8, 25.3),
                "supervised oracle": (88.5, 9.2)}


def load(root):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "*_s*", "results.json"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(os.path.dirname(f)))
        if not m or m.group(1) not in ARMS or not os.path.exists(os.path.join(os.path.dirname(f), "DONE")):
            continue
        runs.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(f))
    return runs


def acc(r, split="eval", stratum="nonlocal", key="R1", mode="normal"):
    return r[split][mode][stratum][key]


def contrast(a, b):
    """d = mean(a) - mean(b), exact permutation p, exact-t MDE from the pooled sd."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(a.mean() - b.mean())
    p = perm2_p(b, a)["p"]
    sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)) if len(a) > 1 and len(b) > 1 else float("nan")
    return dict(d=d, p=p, mean_a=float(a.mean()), mean_b=float(b.mean()), sd=sd,
                mde=float(mde(sd, len(a)) * math.sqrt(2)) if sd == sd else float("nan"))


def primary_label(vals, floor):
    for k in ("P_E", "R1", "P_odo"):
        if len(vals.get(k, [])) < N_REQ:
            return dict(label=f"INCOMPLETE: {k} has {len(vals.get(k, []))}/{N_REQ} runs")
    odo = float(np.mean(vals["P_odo"]))
    if odo < floor + 0.10:
        return dict(label=f"UNREADABLE: P_odo {odo:.3f} < floor {floor:.3f} + 0.10 (the task is not learned with given "
                          f"odometry)", p_odo=odo)
    c = contrast(vals["P_E"], vals["R1"])
    if c["d"] >= BAR and c["p"] < 0.05:
        lab = "PATH WINS"
    elif c["d"] <= -BAR and c["p"] < 0.05:
        lab = "INDEX WINS"
    else:
        lab = "UNMEASURED"
    return dict(label=f"{lab}: P_E {c['mean_a']:.4f} vs R1 {c['mean_b']:.4f}, d {c['d']:+.4f}, perm p {c['p']:.5f}, "
                      f"MDE {c['mde']:.4f}", **c)


def secondaries(vals):
    out, ps, keys = {}, [], []
    for name, a, b in SECONDARIES:
        if len(vals.get(a, [])) < N_REQ or len(vals.get(b, [])) < N_REQ:
            out[name] = dict(label=f"NOT RUN: {a} vs {b}")
            continue
        out[name] = contrast(vals[a], vals[b]); out[name]["pair"] = (a, b)
        ps.append(out[name]["p"]); keys.append(name)
    for k, hp in zip(keys, holm(ps) if ps else []):
        c = out[k]; a, b = c["pair"]; c["p_holm"] = float(hp)
        if c["d"] >= BAR and hp < 0.05:
            lab = f"{a} ABOVE {b}"
        elif c["d"] <= -BAR and hp < 0.05:
            lab = f"{b} ABOVE {a}"
        else:
            lab = "UNMEASURED"
        c["label"] = f"{lab}: {a} {c['mean_a']:.4f} vs {b} {c['mean_b']:.4f}, d {c['d']:+.4f}, Holm p {hp:.5f}"
    return out


def tail_slope(hist):
    h = np.asarray(hist, float)
    k = max(2, int(round(0.1 * len(h))))
    if len(h) < 4:
        return float("nan")
    x = np.arange(k); y = h[-k:]
    s = np.polyfit(x, y, 1)[0] * k
    return float(s / max(abs(y.mean()), 1e-9))


def analyse(runs, floor_eval, floor_held, out=print):
    V = {}
    vals = {a: [acc(r) for _, r in sorted(runs.get(a, {}).items())] for a in ARMS}
    out("== per arm (eval split, non-local blind re-entries, R1; floor %.4f)" % floor_eval)
    for a in ARMS:
        if vals[a]:
            v = np.asarray(vals[a])
            out(f"  {a:6s} n={len(v)} mean {v.mean():.4f} sd {v.std(ddof=1) if len(v) > 1 else float('nan'):.4f} "
                f"above floor {v.mean() - floor_eval:+.4f}  seeds {sorted(runs[a])}")
    P = primary_label(vals, floor_eval)
    V["primary"] = P
    out(f"REGISTERED PRIMARY: {P['label']}")
    S = secondaries(vals)
    V["secondaries"] = S
    for k, c in S.items():
        out(f"REGISTERED SECONDARY {k}: {c['label']}")
    # descriptives
    D = {}
    held = {a: [acc(r, "held") for _, r in sorted(runs.get(a, {}).items()) if "held" in r] for a in ARMS}
    D["held_primary"] = primary_label(held, floor_held)
    out(f"DESCRIPTIVE HELD (train-part0 1600-2899; floor {floor_held:.4f}): {D['held_primary']['label']}")
    for a in ARMS:
        if not runs.get(a):
            continue
        rr = list(runs[a].values())
        D[a] = dict(
            R2=float(np.mean([acc(r, key="R2") for r in rr])),
            all_R1=float(np.mean([acc(r, stratum="all") for r in rr])),
            frames_swapped=float(np.mean([acc(r, mode="frames_swapped") for r in rr])),
            steps_shuffled=float(np.mean([acc(r, mode="steps_shuffled") for r in rr])),
            descending=int(sum(tail_slope(r["hist"]) < -0.005 for r in rr)),
            n=len(rr))
        if all("probe" in r for r in rr):
            for mode in ("blinded", "unblinded"):
                D[a][f"probe_{mode}"] = dict(
                    walls=float(np.mean([r["probe"][mode]["walls_acc"] for r in rr])),
                    objects=float(np.mean([r["probe"][mode]["objects_mse_sqdist"] for r in rr])))
            D[a]["probe_constant"] = rr[0]["probe"]["constant"]
        out(f"  {a:6s} {json.dumps(D[a])}")
    if len(vals["P_raw"]) >= 2 and len(vals["P_E"]) >= 2:
        D["P_raw_vs_P_E"] = contrast(vals["P_raw"], vals["P_E"])
        out(f"DESCRIPTIVE P_raw - P_E: {D['P_raw_vs_P_E']['d']:+.4f} (p {D['P_raw_vs_P_E']['p']:.4f})")
    out("published 15x15 offline-probe rows (Walls %, Objects MSE): " + json.dumps(PUBLISHED_15))
    V["descriptive"] = D
    return V


def head(label):
    return label.split(":", 1)[0].strip()


def selftest():
    fails = []

    def chk(c, m):
        print(("PASS " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    rng = np.random.default_rng(0)

    def mk(means, n=8, sd=0.01):
        runs = {}
        for a, mu in means.items():
            runs[a] = {}
            for s in range(n):
                x = float(np.clip(mu + rng.normal(0, sd), 0, 1))
                ev = {m: {"nonlocal": {"n": 100, "R1": x, "R2": x / 2}, "all": {"n": 120, "R1": x, "R2": x / 2}}
                      for m in ("normal", "frames_swapped", "steps_shuffled")}
                runs[a][3600 + s] = {"eval": ev, "held": {"normal": ev["normal"]}, "hist": list(np.linspace(5, 2, 50))}
        return runs

    base = dict(P_E=0.6, P_odo=0.9, R1=0.3, R4=0.4, GPS=0.85, GRU=0.5)
    reached = set()
    cases = {
        "PATH WINS": base,
        "INDEX WINS": dict(base, P_E=0.2),
        "UNMEASURED": dict(base, P_E=0.32),
        "UNREADABLE": dict(base, P_odo=0.1),
    }
    for want, means in cases.items():
        V = analyse(mk(means), 0.05, 0.05, out=lambda *a: None)
        reached.add(head(V["primary"]["label"]))
        chk(head(V["primary"]["label"]) == want, f"primary {want}: {V['primary']['label']}")
    V = analyse(mk({k: v for k, v in base.items() if k != "R1"}), 0.05, 0.05, out=lambda *a: None)
    chk(head(V["primary"]["label"]) == "INCOMPLETE", f"INCOMPLETE: {V['primary']['label']}")
    reached.add("INCOMPLETE")
    chk(reached == {"PATH WINS", "INDEX WINS", "UNMEASURED", "UNREADABLE", "INCOMPLETE"}, f"every primary label: {reached}")
    # secondaries: P_E above R4, GRU above P_E, P_odo vs P_E unmeasured (equal), GPS above P_odo
    V = analyse(mk(dict(P_E=0.6, P_odo=0.61, R1=0.3, R4=0.3, GPS=0.9, GRU=0.8)), 0.05, 0.05, out=lambda *a: None)
    S = V["secondaries"]
    chk(head(S["S1"]["label"]) == "P_E ABOVE R4", S["S1"]["label"])
    chk(head(S["S2"]["label"]) == "GRU ABOVE P_E", S["S2"]["label"])
    chk(head(S["S3"]["label"]) == "UNMEASURED", S["S3"]["label"])
    chk(head(S["S4"]["label"]) == "GPS ABOVE P_odo", S["S4"]["label"])
    V = analyse(mk({k: v for k, v in base.items() if k != "GPS"}), 0.05, 0.05, out=lambda *a: None)
    chk(head(V["secondaries"]["S4"]["label"]) == "NOT RUN", V["secondaries"]["S4"]["label"])
    # Holm actually adjusts: four p's, the adjusted ones are >= raw
    chk(all(c.get("p_holm", 1) >= c.get("p", 0) for c in S.values()), "Holm-adjusted p >= raw p")
    print(f"\n{len(fails)} FAIL(S)" if fails else "\nSELFTEST ALL PASS")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--floors-eval", help="gate json with the eval-split floors (diag_eval_split.json)")
    ap.add_argument("--floors-held", help="gate json with the held-split floors (mm_floors output)")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())

    def best_floor(p):
        r = json.load(open(p))
        return float(max(v["value"] for v in r["s2_all"].values()))

    V = analyse(load(a.root), best_floor(a.floors_eval), best_floor(a.floors_held))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(V, f, indent=1, default=str)


if __name__ == "__main__":
    main()

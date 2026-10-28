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
loss-curve slope over 60-85% of training, before the cosine LR collapses (Amendment 2, M3): DESCENDING if the linear
fit's change across the window is < -3% of the window's mean loss. Validated on the 17 pilot curves: every pilot's loss
falls 1.9-5.6% across the window while validation retrieval moves within about 1 SE, so the rule is a weak readout;
every secondary is budget-scoped at E = 600.
P_E_frozen (Amendment 2): descriptive arm, P_E with the fitted E map installed frozen; trainable parameters = R1's.
Guards (Amendment 2): exactly the registered seeds per arm, the registered config, one md5 set (unchanged during each
run) and one torch version across all runs; anything else is refused. Every printed primary / HELD label carries
"(loss pilot-informed, Amendment 1)" and the scope note; a significant |d| < 0.05 is flagged "below the 0.05 bar".
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


def holm(ps):
    """Holm-adjusted p-values (step-down), same order as ps (identical to analyze_lrsweep.holm; copied so this module
    imports only numpy / scipy / stats_core)."""
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1.0, run)
    return adj

ARMS = ("P_E", "P_odo", "P_raw", "R1", "R4", "GPS", "GRU", "P_E_frozen")
N_REQ = 8
SEEDS = tuple(range(3600, 3608))
REG_CFG = dict(epochs=600, loss_steps="both", batch=16, lr=3e-4, wd=0.05, train_range=[0, 1600], held_range=[1600, 2900],
               eval_n=1000, pilot=False, val_every=0, no_probe=False, no_held=False, probe_steps=10000)
PILOT_FLAG = "(loss pilot-informed, Amendment 1)"
SCOPE = ("scope: one path layer vs one index layer at E = 600, budget-scoped; the one-layer retrieval limits every arm "
         "(Amendment 2); S1 tests depth AND parameters (R4 2.95x); if S1 reads 'R4 ABOVE P_E', 'one path layer ~ three "
         "attention layers' does not hold here")
BAR = 0.05
SECONDARIES = (("S1", "P_E", "R4"), ("S2", "P_E", "GRU"), ("S3", "P_odo", "P_E"), ("S4", "P_odo", "GPS"))
PUBLISHED_15 = {"constant": (78.3, 64.8), "VAE (no memory)": (78.7, 56.2), "RSSM-TBTT (Pasukonis)": (80.5, 33.8),
                "RSSM-TBTT (Deng)": (81.7, 32.6), "TSSM-XL": (79.8, 35.1), "S4WM": (80.2, 34.8), "S5WM": (81.8, 25.3),
                "supervised oracle": (88.5, 9.2)}


def load(root, strict=True):
    """Finished runs <root>/<ARM>_s<seed>/{results.json, DONE}. strict: refuse (SystemExit) unless every run is at a
    registered seed, has the registered config (install 'frozen' for P_E_frozen, 'none' otherwise), carries the same
    end-of-run md5 set as every other run with no change during the run, and the same torch version."""
    runs, problems, md5s, torches = {}, [], set(), set()
    for d in sorted(glob.glob(os.path.join(root, "*_s*"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(d))
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        if arm not in ARMS:
            problems.append(f"unknown arm dir {d}"); continue
        if not (os.path.exists(os.path.join(d, "DONE")) and os.path.exists(os.path.join(d, "results.json"))):
            continue
        r = json.load(open(os.path.join(d, "results.json")))
        if seed not in SEEDS:
            problems.append(f"{arm} seed {seed} not registered")
        c = r["config"]
        want = dict(REG_CFG, install="frozen" if arm == "P_E_frozen" else "none")
        bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
        if bad:
            problems.append(f"{arm}_s{seed} config differs: {bad}")
        if r.get("md5_changed"):
            problems.append(f"{arm}_s{seed} code changed during the run: {r['md5_changed']}")
        md5s.add(json.dumps(r.get("md5"), sort_keys=True)); torches.add(r.get("env", {}).get("torch"))
        runs.setdefault(arm, {})[seed] = r
    if len(md5s) > 1:
        problems.append(f"{len(md5s)} different code md5 sets across runs")
    if len(torches) > 1:
        problems.append(f"mixed torch versions {sorted(map(str, torches))} (never pool workstation and cluster runs)")
    if strict and problems:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(problems))
    meta = dict(torch=sorted(map(str, torches)), n_md5_sets=len(md5s), problems=problems)
    return runs, meta


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
    sub = ""
    if c["d"] >= BAR and c["p"] < 0.05:
        lab = "PATH WINS"
    elif c["d"] <= -BAR and c["p"] < 0.05:
        lab = "INDEX WINS"
    else:
        lab = "UNMEASURED"
        if c["p"] < 0.05:
            sub = f" [below the 0.05 bar: significant |d| < 0.05, MDE {c['mde']:.4f}]"
    return dict(label=f"{lab}: P_E {c['mean_a']:.4f} vs R1 {c['mean_b']:.4f}, d {c['d']:+.4f}, perm p {c['p']:.5f}, "
                      f"MDE {c['mde']:.4f}{sub} {PILOT_FLAG}", **c)


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
            lab = "UNMEASURED" + (" [below the 0.05 bar]" if hp < 0.05 else "")
        c["label"] = (f"{lab}: {a} {c['mean_a']:.4f} vs {b} {c['mean_b']:.4f}, d {c['d']:+.4f}, Holm p {hp:.5f} "
                      f"[budget-scoped, E = 600]")
    return out


def window_slope(hist, a=0.60, b=0.85):
    """Relative change of the linear fit of the training loss across [a, b] of training (before the LR collapses)."""
    h = np.asarray(hist, float)
    i0, i1 = int(round(a * len(h))), int(round(b * len(h)))
    if i1 - i0 < 3:
        return float("nan")
    y = h[i0:i1]
    return float(np.polyfit(np.arange(len(y)), y, 1)[0] * len(y) / max(abs(y.mean()), 1e-9))


def descending(hist, thr=-0.03):
    return window_slope(hist) < thr


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
    out(f"  {SCOPE}")
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
            descending=int(sum(descending(r["hist"]) for r in rr)),
            n=len(rr))
        if all("probe" in r for r in rr):
            for mode in ("blinded", "unblinded"):
                D[a][f"probe_{mode}"] = dict(
                    walls=float(np.mean([r["probe"][mode]["walls_acc"] for r in rr])),
                    objects=float(np.mean([r["probe"][mode]["objects_mse_sqdist"] for r in rr])))
            D[a]["probe_constant"] = rr[0]["probe"]["constant"]
        out(f"  {a:6s} {json.dumps(D[a])}")
    if len(vals["P_E_frozen"]) >= 2 and len(vals["R1"]) >= 2:
        D["P_E_frozen_vs_R1"] = contrast(vals["P_E_frozen"], vals["R1"])
        out(f"DESCRIPTIVE P_E_frozen - R1 (fixed path phase vs index phase, equal trainable parameters): "
            f"{D['P_E_frozen_vs_R1']['d']:+.4f} (p {D['P_E_frozen_vs_R1']['p']:.4f})")
    if len(vals["P_raw"]) >= 2 and len(vals["P_E"]) >= 2:
        D["P_raw_vs_P_E"] = contrast(vals["P_raw"], vals["P_E"])
        out(f"DESCRIPTIVE P_raw - P_E: {D['P_raw_vs_P_E']['d']:+.4f} (p {D['P_raw_vs_P_E']['p']:.4f})")
    out("published 15x15 offline-probe rows (Walls %, Objects MSE): " + json.dumps(PUBLISHED_15))
    V["descriptive"] = D
    return V


def head(label):
    return label.split(":", 1)[0].strip()


def best_floor(p):
    r = json.load(open(p))
    return float(max(v["value"] for v in r["s2_all"].values()))


def selftest():
    import tempfile
    fails = []

    def chk(c, m):
        print(("PASS " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    rng = np.random.default_rng(0)
    quiet = lambda *a: None

    def ev_of(x):
        return {m: {"nonlocal": {"n": 100, "R1": x, "R2": x / 2}, "all": {"n": 120, "R1": x, "R2": x / 2}}
                for m in ("normal", "frames_swapped", "steps_shuffled")}

    def mk(means, n=8, sd=0.01, held=None, explicit=None):
        runs = {}
        for a, mu in means.items():
            runs[a] = {}
            for i, s_ in enumerate(SEEDS[:n]):
                x = explicit[a][i] if explicit and a in explicit else float(np.clip(mu + rng.normal(0, sd), 0, 1))
                hx = (held or {}).get(a, x)
                runs[a][s_] = {"eval": ev_of(x), "held": {"normal": ev_of(hx)["normal"]},
                               "hist": list(np.linspace(5, 2, 600))}
        return runs

    base = dict(P_E=0.6, P_odo=0.9, R1=0.3, R4=0.4, GPS=0.85, GRU=0.5)
    reached = set()
    cases = {"PATH WINS": base, "INDEX WINS": dict(base, P_E=0.2), "UNMEASURED": dict(base, P_E=0.32),
             "UNREADABLE": dict(base, P_odo=0.1)}
    for want, means in cases.items():
        V = analyse(mk(means), 0.05, 0.05, out=quiet)
        reached.add(head(V["primary"]["label"]))
        chk(head(V["primary"]["label"]) == want, f"primary {want}: {V['primary']['label']}")
        if want in ("PATH WINS", "INDEX WINS", "UNMEASURED"):
            chk(PILOT_FLAG in V["primary"]["label"], f"{want} carries the pilot-informed flag")
    V = analyse(mk({k: v for k, v in base.items() if k != "R1"}), 0.05, 0.05, out=quiet)
    chk(head(V["primary"]["label"]) == "INCOMPLETE", f"INCOMPLETE via R1: {V['primary']['label']}")
    reached.add("INCOMPLETE")
    V = analyse(mk({k: v for k, v in base.items() if k != "P_odo"}), 0.05, 0.05, out=quiet)
    chk(V["primary"]["label"].startswith("INCOMPLETE: P_odo"), f"INCOMPLETE via P_odo: {V['primary']['label']}")
    chk(reached == {"PATH WINS", "INDEX WINS", "UNMEASURED", "UNREADABLE", "INCOMPLETE"}, f"every primary label: {reached}")
    # d >= bar but p >= .05 -> UNMEASURED, no sub-label
    V = analyse(mk(base, explicit={"P_E": [0.0, 0.8] * 4, "R1": [0.3] * 8}), 0.05, 0.05, out=quiet)
    chk(head(V["primary"]["label"]) == "UNMEASURED" and V["primary"]["d"] >= BAR and V["primary"]["p"] >= 0.05
        and "below the 0.05 bar" not in V["primary"]["label"], f"d >= bar, p >= .05: {V['primary']['label']}")
    # significant but below the bar -> UNMEASURED [below the 0.05 bar]
    V = analyse(mk(base, explicit={"P_E": [0.31 + 0.001 * i for i in range(8)], "R1": [0.30 + 0.001 * i for i in range(8)]}),
                0.05, 0.05, out=quiet)
    chk(head(V["primary"]["label"]) == "UNMEASURED" and "below the 0.05 bar" in V["primary"]["label"],
        f"significant |d| < bar flagged: {V['primary']['label']}")
    # HELD replicate reads its own values
    V = analyse(mk(base, held={"P_E": 0.31, "R1": 0.30}), 0.05, 0.05, out=quiet)
    chk(head(V["primary"]["label"]) == "PATH WINS" and head(V["descriptive"]["held_primary"]["label"]) == "UNMEASURED",
        f"HELD replicate independent of eval: {V['descriptive']['held_primary']['label']}")
    # secondaries
    V = analyse(mk(dict(P_E=0.6, P_odo=0.61, R1=0.3, R4=0.3, GPS=0.9, GRU=0.8)), 0.05, 0.05, out=quiet)
    S = V["secondaries"]
    chk(head(S["S1"]["label"]) == "P_E ABOVE R4", S["S1"]["label"])
    chk(head(S["S2"]["label"]) == "GRU ABOVE P_E", S["S2"]["label"])
    chk(head(S["S3"]["label"]).startswith("UNMEASURED"), S["S3"]["label"])
    chk(head(S["S4"]["label"]) == "GPS ABOVE P_odo", S["S4"]["label"])
    V = analyse(mk({k: v for k, v in base.items() if k != "GPS"}), 0.05, 0.05, out=quiet)
    chk(head(V["secondaries"]["S4"]["label"]) == "NOT RUN", V["secondaries"]["S4"]["label"])
    adj = list(holm([0.01, 0.04, 0.03, 0.2]))
    chk(np.allclose(adj, [0.04, 0.09, 0.09, 0.2]), f"Holm step-down exact: {adj}")
    V = analyse(mk(dict(base, P_E_frozen=0.5)), 0.05, 0.05, out=quiet)
    chk("P_E_frozen_vs_R1" in V["descriptive"], "P_E_frozen - R1 descriptive present")
    chk(descending(list(np.linspace(5, 2, 600))) and not descending([3.0] * 600), "window-slope DESCENDING rule")
    # floors on the real gate JSONs
    here = os.path.dirname(os.path.abspath(__file__))
    fe = os.path.join(here, "docs/audits/2026-10-27/memory_maze_gate_v3/diag_eval_split.json")
    fh = os.path.join(here, "docs/audits/2026-10-27/memory_maze_batch/floors_held.json")
    if os.path.exists(fe) and os.path.exists(fh):
        chk(abs(best_floor(fe) - 0.0472) < 5e-4 and abs(best_floor(fh) - 0.0472) < 5e-4,
            f"best_floor eval {best_floor(fe):.5f}, held {best_floor(fh):.5f}")
    # load() guards
    def write_runs(root, mut=None):
        for a in ("P_E", "R1", "P_E_frozen"):
            for s_ in SEEDS:
                d = os.path.join(root, f"{a}_s{s_}"); os.makedirs(d, exist_ok=True)
                cfg = dict(REG_CFG, install="frozen" if a == "P_E_frozen" else "none")
                r = {"config": cfg, "md5": {"train_mm.py": "x"}, "md5_changed": [], "env": {"torch": "2.13.0"},
                     "eval": ev_of(0.3), "hist": [1.0] * 10}
                if mut:
                    mut(a, s_, r, root)
                json.dump(r, open(os.path.join(d, "results.json"), "w")); open(os.path.join(d, "DONE"), "w").close()

    def refused(mut):
        with tempfile.TemporaryDirectory() as t:
            write_runs(t, mut)
            try:
                load(t)
                return False
            except SystemExit:
                return True

    with tempfile.TemporaryDirectory() as t:
        write_runs(t)
        runs, meta = load(t)
        chk(sorted(runs) == ["P_E", "P_E_frozen", "R1"] and all(len(v) == 8 for v in runs.values())
            and meta["torch"] == ["2.13.0"], f"load accepts the registered set {meta}")
    chk(refused(lambda a, s_, r, root: os.makedirs(os.path.join(root, "P_E_s3608"), exist_ok=True) or
                (json.dump(r, open(os.path.join(root, "P_E_s3608", "results.json"), "w")),
                 open(os.path.join(root, "P_E_s3608", "DONE"), "w").close())), "load refuses a stray seed")
    chk(refused(lambda a, s_, r, root: r["config"].update(epochs=400) if s_ == 3603 else None), "load refuses a config change")
    chk(refused(lambda a, s_, r, root: r.update(md5={"train_mm.py": "y"}) if s_ == 3601 else None), "load refuses mixed md5")
    chk(refused(lambda a, s_, r, root: r.update(env={"torch": "2.10.0"}) if a == "R1" else None), "load refuses mixed torch")
    chk(refused(lambda a, s_, r, root: r.update(md5_changed=["model_mm.py"]) if s_ == 3600 else None),
        "load refuses a run whose code changed")
    chk(refused(lambda a, s_, r, root: r["config"].update(install="none") if a == "P_E_frozen" else None),
        "load refuses P_E_frozen without the frozen install")
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

    runs, meta = load(a.root)
    print(f"runs loaded: { {k: sorted(v) for k, v in runs.items()} }; torch {meta['torch']}")
    V = analyse(runs, best_floor(a.floors_eval), best_floor(a.floors_held))
    V["meta"] = meta
    if a.out:
        with open(a.out, "w") as f:
            json.dump(V, f, indent=1, default=str)


if __name__ == "__main__":
    main()

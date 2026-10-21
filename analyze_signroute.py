"""Registered analysis for SIGNROUTE_PREREG.md (G-2 Part A + fixed-rate breadth arms), as amended
(Amendment 1: the independent results-blind code check, findings 1, 2, 5-8, 12, 15).

  python3 -m mapformer.analyze_signroute               # reads runs/signroute/p0, writes the verdicts
  python3 -m mapformer.analyze_signroute --selftest    # every label and reading, incl. load() on disk

Built from the shared primitives (rule 5: import, do not re-derive): stats_core.classify_run
(SOLVED = final-5% training loss < 0.05), stats_core.fisher_solved, stats_core.perm2_p; the breadth
cells use analyze_robust.cell_label's thresholds. The PRIMARY rule is not cell_label: cell_label
gates on its reference arm solving >= 4/8 and returns UNREADABLE otherwise, which would report a
FAILING hybrid -- the refuting outcome -- as unreadable.

`better(a, b)` = arm a learns better than arm b by the house SIGNED-LEARNS-BETTER thresholds with a in
the reference slot: (Fisher < .05 and k_a > k_b) or (permutation p < .05 and acc_a - acc_b >= +0.05).

Per rate r in {1e-3, 3e-3}, n = 8 seeds per arm; signed S, abs A, hybrid H:
  G0   k_S >= 6, else UNREADABLE at r.
  G1   better(S, A), else NO COST TO EXPLAIN at r: the sign cost must be MEASURED in this batch
       before a hybrid-vs-abs difference has anything to explain (Amendment 1 finding 8: the earlier
       "k_A <= 4" passed with no measurable cost, e.g. 6/8 vs 4/8, Fisher p 0.6).
  P    hA = better(H, A); sH = better(S, H);
       full  = k_H >= k_S - 1 and acc_H >= acc_S - 0.01        (the house NO-SIGN-COST closeness)
       likeA = k_H <= k_A + 1 and acc_H <= acc_A + 0.01        (the same closeness, toward abs)
         SIGNED INCREMENT RESTORES   hA and full
         NON-NEGATIVE READOUT COSTS  sH and likeA
         PARTIAL                     hA or sH, but neither of the above (the hybrid sits measurably
                                     between the anchors; Amendment 1 finding 1: the first rule read a
                                     4/8 hybrid as a full restoration)
         UNMEASURED                  otherwise
  C    any abs_tied SOLVED run -> every reading prefixed CONTROL BROKEN (no solution exists in that
       class: construct_signroute.py, proof + measurement).

Overall reading (amended): UNREADABLE (no rate passes G0 and G1); MIXED (RESTORES at one rate,
COSTS at the other); PARTIAL (PARTIAL at any readable rate, otherwise); SUPPORTED (RESTORES at both);
SUPPORTED AT ONE RATE (RESTORES at one; the other UNMEASURED, UNREADABLE or NO COST TO EXPLAIN);
REFUTED (COSTS at both); REFUTED AT ONE RATE (COSTS at one, the other UNMEASURED or unreadable);
UNMEASURED.

WHAT "SIGNED INCREMENT RESTORES" DOES AND DOES NOT SHOW (Amendment 1 finding 2). The hybrid's zero
clock sits at k = 0, so it removes BOTH the missing collective route (S3) and the S1 + P3 wrap barrier
(a |.| channel cancels only at k = 2 pi, across pi from a small start), which SEARCH_THEORY's own S3
"Reading" calls the larger obstruction. A hybrid win therefore shows that a non-negative READOUT
MATRIX is costless and the signed INCREMENT is what matters; it does NOT separate S3 from S1/P3.

Breadth (exp, softplus) vs signed at each rate: cell_label's thresholds, with Holm applied to EACH
test separately over the 4 cells (Fisher p's together, permutation p's together) and a label kept only
if the test that fired survives (Amendment 1 finding 6: min(perm, Fisher) is not a valid p).
Reported "at the two reference rates", never as best-rate results.

Diverged runs (non-finite loss anywhere): class DIVERGED, never SOLVED, non-finite accuracy counted
as 0 (declared; as analyze_lrsweep). Every run is checked for its recipe and len(losses) == epochs.
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

from mapformer.analyze_robust import NLB, NSC, SLB, UNM
from mapformer.stats_core import classify_run, fisher_solved, perm2_p

REPO = "/home/prashr/mapformer"                    # rule 25: absolute
ROOT = f"{REPO}/runs/signroute/p0"
SEEDS = list(range(1800, 1808))
RATES = (1e-3, 3e-3)
EPOCHS = 300
PRIMARY = ("signed", "abs", "hybrid", "abs_tied")
BREADTH = ("exp", "softplus")
ARMS = PRIMARY + BREADTH
FLOORS = {"blank": 0.50, "ngram": 0.554, "retrace": 0.860}   # LRSWEEP_RESULTS.md, ring 32, +1/-1
RECIPE = {"size": 32, "n_steps": 128, "batch_size": 128, "n_batches": 98, "data_workers": 3,
          "n_trials": 400, "epochs": EPOCHS, "weight_decay": 0.05, "deltas": "1,-1",
          "freeze_move_latents": False, "init_from": None}

SIR, NRC, PART, PUNM = ("SIGNED INCREMENT RESTORES", "NON-NEGATIVE READOUT COSTS", "PARTIAL",
                        "UNMEASURED")
UNR, NCE = "UNREADABLE", "NO COST TO EXPLAIN"


def rate_tag(lr):
    """1e-3 -> lr1e-3. The driver checks its directory tags against this function."""
    return f"lr{lr:.0e}".replace("e-0", "e-")


def run_dir(root, arm, lr, seed):
    return os.path.join(root, f"{arm}_{rate_tag(lr)}_s{seed}")


def better(k_a, k_b, acc_a, acc_b, n):
    pf = fisher_solved(k_b, n, k_a, n)
    pp = perm2_p(np.asarray(acc_b, float), np.asarray(acc_a, float))["p"]
    d = float(np.mean(acc_a) - np.mean(acc_b))
    return (pf < 0.05 and k_a > k_b) or (pp < 0.05 and d >= 0.05), {"fisher_p": pf, "perm_p": pp, "d": d}


def primary_label(cell, n):
    """cell: {arm: {"k": SOLVED count, "acc": [per-seed acc at T]}} at one rate."""
    S, A, H = cell["signed"], cell["abs"], cell["hybrid"]
    if S["k"] < 6:
        return f"{UNR} (signed SOLVED {S['k']}/{n} < 6)", {}
    sA, st_sa = better(S["k"], A["k"], S["acc"], A["acc"], n)
    if not sA:
        return f"{NCE} (signed {S['k']}/{n} vs abs {A['k']}/{n}: no measured cost)", {"S_vs_A": st_sa}
    hA, st_ha = better(H["k"], A["k"], H["acc"], A["acc"], n)
    sH, st_sh = better(S["k"], H["k"], S["acc"], H["acc"], n)
    full = H["k"] >= S["k"] - 1 and np.mean(H["acc"]) >= np.mean(S["acc"]) - 0.01
    likeA = H["k"] <= A["k"] + 1 and np.mean(H["acc"]) <= np.mean(A["acc"]) + 0.01
    det = {"S_vs_A": st_sa, "H_vs_A": st_ha, "S_vs_H": st_sh, "full": bool(full), "likeA": bool(likeA)}
    if hA and full:
        return SIR, det
    if sH and likeA:
        return NRC, det
    if hA or sH:
        return PART, det
    return PUNM, det


def overall(labels, control_broken):
    """labels: {rate: primary label}. Fixed in SIGNROUTE_PREREG.md Sec. 5 (as amended)."""
    readable = [l for l in labels.values() if not (l.startswith(UNR) or l.startswith(NCE))]
    if not readable:
        out = UNR
    elif SIR in readable and NRC in readable:
        out = "MIXED"
    elif PART in readable:
        out = PART
    elif readable.count(SIR) == len(RATES):
        out = "SUPPORTED"
    elif SIR in readable:
        out = "SUPPORTED AT ONE RATE"
    elif readable.count(NRC) == len(RATES):
        out = "REFUTED"
    elif NRC in readable:
        out = "REFUTED AT ONE RATE"
    else:
        out = PUNM
    return ("CONTROL BROKEN: " + out) if control_broken else out


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


def breadth_label(ao_acc, so_acc, k_ao, k_so, pf, pp):
    """analyze_robust.cell_label's rule, written against GIVEN p-values so Holm-adjusted ones can be
    used per test. With raw p-values it is cell_label exactly (checked in the selftest)."""
    d = float(np.mean(ao_acc) - np.mean(so_acc))
    if k_so < 4:
        return f"UNREADABLE (SO SOLVED {k_so} < 4)"
    if (pf < 0.05 and k_so > k_ao) or (pp < 0.05 and d <= -0.05):
        return SLB
    if (pf < 0.05 and k_ao > k_so) or (pp < 0.05 and d >= 0.05):
        return NLB
    if k_so >= 6 and k_ao >= k_so - 1 and np.mean(ao_acc) >= np.mean(so_acc) - 0.01:
        return NSC
    return UNM


# ------------------------------------------------------------------------------ loading
def _fin(x):
    return float(x) if x is not None and np.isfinite(x) else 0.0


def load(root=ROOT, seeds=SEEDS):
    R, missing = {}, []
    for arm in ARMS:
        for lr in RATES:
            for s in seeds:
                d = run_dir(root, arm, lr, s)
                p = os.path.join(d, "eval.json")
                if not os.path.exists(p):
                    missing.append(p)
                    continue
                e = json.load(open(p))
                A = e["args"]
                assert (A["arm"], A["seed"]) == (arm, s) and abs(A["lr"] - lr) <= 1e-9 * lr, (d, A)
                bad = {k: A.get(k) for k, v in RECIPE.items() if A.get(k) != v}
                assert not bad, (d, bad)
                L = np.asarray(e["losses"], float)
                assert len(L) == EPOCHS, (d, len(L))
                div = not np.all(np.isfinite(L))
                k = classify_run(np.nan_to_num(L, nan=1e9, posinf=1e9)) if div else classify_run(L)
                R[(arm, lr, s)] = {"acc": _fin(e["eval"]["128"]["acc"]), "acc4": _fin(e["eval"]["512"]["acc"]),
                                   "final_loss": float(e["final_loss"]), "div": div,
                                   "solved": (k["registered"] == "SOLVED") and not div,
                                   "cls": "DIVERGED" if div else k["cls"],
                                   **{q: e.get(q) for q in ("u_ratio", "latent_cos", "clock_absmax",
                                                            "step_frac_negative", "readout_frac_negative",
                                                            "readout_row_sign_coherent")}}
    return R, missing


def cells(R, seeds=SEEDS):
    C = {}
    for lr in RATES:
        C[lr] = {}
        for arm in ARMS:
            rows = [R[(arm, lr, s)] for s in seeds]
            C[lr][arm] = {"k": sum(r["solved"] for r in rows), "acc": [r["acc"] for r in rows],
                          "acc4": [r["acc4"] for r in rows], "rows": rows}
    return C


def _med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return f"{np.median(v):+.3f}" if v else "  n/a"


# ------------------------------------------------------------------------------ analysis
def analyse(R, seeds=SEEDS, out=print):
    n = len(seeds)
    C = cells(R, seeds)
    labels, details = {}, {}
    for lr in RATES:
        labels[lr], details[lr] = primary_label(C[lr], n)
    tied_solved = sum(C[lr]["abs_tied"]["k"] for lr in RATES)
    broken = tied_solved > 0
    reading = overall(labels, broken)

    out("== SIGNROUTE: registered primary (hybrid vs abs and vs signed, gated on the in-batch anchors) ==")
    for lr in RATES:
        c = C[lr]
        out(f"lr {lr:g}: SOLVED signed {c['signed']['k']}/{n} abs {c['abs']['k']}/{n} hybrid "
            f"{c['hybrid']['k']}/{n} abs_tied {c['abs_tied']['k']}/{n} | acc T=128 signed "
            f"{np.mean(c['signed']['acc']):.4f} abs {np.mean(c['abs']['acc']):.4f} hybrid "
            f"{np.mean(c['hybrid']['acc']):.4f} abs_tied {np.mean(c['abs_tied']['acc']):.4f}")
        out(f"   primary: {labels[lr]}   {json.dumps(details[lr])}")
    out(f"\nC (control): abs_tied SOLVED {tied_solved}/{n * len(RATES)} -- "
        + ("CONTROL BROKEN: a class with no solution solved; find the bug or the proof error before "
           "reading anything else" if broken else "held (no abs_tied run solved, as the proof requires)"))
    out(f"\nREADING (registered): {reading}")
    out("   scope: 'at lr 1e-3 and 3e-3'. A SIGNED INCREMENT RESTORES reading shows the non-negative readout "
        "MATRIX is costless and the signed increment is what matters; it does not separate S3 (no route) "
        "from S1/P3 (the wrap barrier) -- the hybrid removes both.")

    out("\n== breadth (fixed rates, NOT best-rate claims): exp / softplus vs signed, "
        "cell_label thresholds, Holm per test over 4 cells ==")
    raw = []
    for lr in RATES:
        for arm in BREADTH:
            a, s = C[lr][arm], C[lr]["signed"]
            pf = fisher_solved(a["k"], n, s["k"], n)
            pp = perm2_p(np.asarray(s["acc"]), np.asarray(a["acc"]))["p"]
            raw.append((lr, arm, a, s, pf, pp))
    af, ap = holm([r[4] for r in raw]), holm([r[5] for r in raw])
    breadth = {}
    for (lr, arm, a, s, pf, pp), pfa, ppa in zip(raw, af, ap):
        lab_raw = breadth_label(a["acc"], s["acc"], a["k"], s["k"], pf, pp)
        lab = breadth_label(a["acc"], s["acc"], a["k"], s["k"], pfa, ppa)
        breadth[f"{arm}|{lr:g}"] = lab
        out(f"lr {lr:g} {arm:9s}: SOLVED {a['k']}/{n} acc {np.mean(a['acc']):.4f} vs signed {s['k']}/{n} | "
            f"raw {lab_raw} (Fisher {pf:.4f}, perm {pp:.4f}) | Holm (Fisher {pfa:.4f}, perm {ppa:.4f}) -> {lab}")

    out("\n== S2 diagnostics (medians per arm x rate; registered as descriptive) ==")
    out("   arm        lr     u_ratio  cos(zx,zy)  max clock  step<0  readout<0  row-sign-coherent")
    for lr in RATES:
        for arm in ARMS:
            rows = C[lr][arm]["rows"]
            out(f"   {arm:9s} {lr:g}  {_med(rows, 'u_ratio')}   {_med(rows, 'latent_cos')}     "
                f"{_med(rows, 'clock_absmax')}    {_med(rows, 'step_frac_negative')}  "
                f"{_med(rows, 'readout_frac_negative')}     {_med(rows, 'readout_row_sign_coherent')}")

    out("\n== rule 2: r(log10 final loss, acc) per eval length, finite runs ==")
    fin = [r for r in R.values() if not r["div"]]
    L = np.array([r["final_loss"] for r in fin])
    for key, nm in (("acc", "T=128"), ("acc4", "T=512")):
        A = np.array([r[key] for r in fin])
        rr = float(np.corrcoef(np.log10(L + 1e-6), A)[0, 1]) if len(A) > 2 and A.std() > 0 and L.std() > 0 \
            else float("nan")
        out(f"   {nm}: r = {rr:+.3f} over {len(A)} runs ({len(R) - len(fin)} diverged excluded)")

    out(f"\nfloors on record (ring 32, +1/-1): blank {FLOORS['blank']}, n-gram {FLOORS['ngram']}, "
        f"retrace {FLOORS['retrace']}")
    out("\nper seed (acc T=128 + class initial; D = DIVERGED), lr x arm:")
    for lr in RATES:
        for arm in ARMS:
            rows = C[lr][arm]["rows"]
            out(f"  {lr:g} {arm:9s} " + " ".join(f"{r['acc']:.3f}{r['cls'][0]}" for r in rows))

    return {"reading": reading, "primary": {f"{lr:g}": labels[lr] for lr in RATES},
            "primary_detail": {f"{lr:g}": details[lr] for lr in RATES}, "control_broken": broken,
            "abs_tied_solved": tied_solved, "breadth": breadth,
            "solved": {f"{arm}|{lr:g}": C[lr][arm]["k"] for lr in RATES for arm in ARMS},
            "n_diverged": int(sum(r["div"] for r in R.values()))}


# ------------------------------------------------------------------------------ selftest
def _loss_curve(solved, diverge=False):
    """300-epoch curve: SOLVED -> tail 1e-4; unsolved -> flat 0.3 (STALLED); diverged -> NaN tail."""
    L = list(np.linspace(1.5, 0.3, 150)) + ([1e-4] * 150 if solved else [0.3] * 150)
    if diverge:
        L[-10:] = [float("nan")] * 10
    return L


def _write_synthetic(root, spec, acc_lo=None, seed=0):
    """Write REAL eval.json files under `root` in the driver's naming, so load() and classify_run are
    exercised (Amendment 1 finding 15). spec: {(arm, lr): k solved}; acc_lo: {arm: acc of unsolved}."""
    rng = np.random.default_rng(seed)
    acc_lo = acc_lo or {}
    for arm in ARMS:
        for lr in RATES:
            k = spec.get((arm, lr), 0)
            for i, s in enumerate(SEEDS):
                sol = i < k
                div = (arm, lr, "div") in spec and i == len(SEEDS) - 1
                d = run_dir(root, arm, lr, s)
                os.makedirs(d, exist_ok=True)
                lo = acc_lo.get(arm, 0.90)
                acc = (1.0 if sol else lo) - 0.002 * rng.random()
                args = dict(RECIPE, arm=arm, lr=lr, seed=s)
                e = {"eval": {"128": {"acc": float("nan") if div else acc, "nll": 0.0},
                              "512": {"acc": 0.99 if sol else 0.80, "nll": 0.0}},
                     "arm": arm, "lr": lr, "seed": s, "args": args,
                     "losses": _loss_curve(sol and not div, diverge=div),
                     "final_loss": float("nan") if div else (1e-4 if sol else 0.3)}
                with open(os.path.join(d, "eval.json"), "w") as f:
                    json.dump(e, f)


def selftest():
    def ks(**kw):
        base = {}
        for lr in RATES:
            base.update({("signed", lr): 8, ("abs", lr): 2, ("abs_tied", lr): 0, ("hybrid", lr): 8,
                         ("exp", lr): 2, ("softplus", lr): 3})
        for key, v in kw.items():
            if key == "div":
                base[v] = True
                continue
            arm, r = key.rsplit("_", 1)
            base[(arm.replace("abstied", "abs_tied"), {"a": 1e-3, "b": 3e-3}[r])] = v
        return base

    silent = lambda *a, **k: None
    tmp = tempfile.mkdtemp(prefix="signroute_selftest_")
    cases = [
        ("SUPPORTED", ks(), None),
        ("SUPPORTED AT ONE RATE", ks(hybrid_b=4), None),                 # b: 4/8 vs 2/8, 8 vs 4 -> UNMEASURED
        ("REFUTED", ks(hybrid_a=2, hybrid_b=2), None),
        ("REFUTED AT ONE RATE", ks(hybrid_a=2, hybrid_b=4), None),
        ("MIXED", ks(hybrid_b=2), None),
        ("UNMEASURED", ks(hybrid_a=4, hybrid_b=4), None),
        ("UNREADABLE", ks(signed_a=5, signed_b=5), None),
        ("UNREADABLE", ks(abs_a=6, abs_b=6), None),                      # G1: no measured cost at either rate
        ("CONTROL BROKEN: SUPPORTED", ks(abstied_a=1), None),
        # Amendment 1 finding 1, the reviewer's three counterexamples -- none may read SUPPORTED / REFUTED:
        ("PARTIAL", ks(abs_a=1, abs_b=1, hybrid_a=4, hybrid_b=4), {"abs": 0.62}),
        ("PARTIAL", ks(abs_a=0, abs_b=0, hybrid_a=5, hybrid_b=5), None),
        ("PARTIAL", ks(abs_a=0, abs_b=0, hybrid_a=3, hybrid_b=3), {"abs": 0.62}),
    ]
    ok = True
    seen = set()
    for i, (want, spec, lo) in enumerate(cases):
        root = os.path.join(tmp, f"c{i}")
        _write_synthetic(root, spec, lo)
        R, missing = load(root)
        assert not missing, missing[:2]
        v = analyse(R, out=silent)
        got = v["reading"]
        ok &= got == want
        seen |= set(v["primary"].values())
        print(f"  {'ok ' if got == want else 'BAD'} want {want!r:28s} got {got!r:28s} primary {v['primary']}")
    for lab in (SIR, NRC, PART, PUNM):
        if lab not in seen:
            print(f"  BAD primary label never reached: {lab}"); ok = False
    if not any(s.startswith(UNR) for s in seen) or not any(s.startswith(NCE) for s in seen):
        print("  BAD UNREADABLE / NO COST TO EXPLAIN gate never reached"); ok = False
    # divergence: a NaN run must be DIVERGED, never SOLVED, and must not break the correlation
    root = os.path.join(tmp, "div")
    sp = ks(); sp[("hybrid", 1e-3, "div")] = True
    _write_synthetic(root, sp)
    R, _ = load(root)
    r = R[("hybrid", 1e-3, SEEDS[-1])]
    if not (r["div"] and not r["solved"] and r["cls"] == "DIVERGED" and r["acc"] == 0.0):
        print(f"  BAD diverged run handled wrongly: {r}"); ok = False
    v = analyse(R, out=silent)
    if v["n_diverged"] != 1:
        print(f"  BAD n_diverged {v['n_diverged']}"); ok = False
    # breadth: SLB and NSC reachable, and the given-p rule equals analyze_robust.cell_label on raw p's
    from mapformer.analyze_robust import cell_label
    bl = set()
    for sp, nm in ((ks(), "b0"), (ks(exp_a=8, exp_b=8, softplus_a=8, softplus_b=8), "b1"),
                   (ks(exp_a=0, exp_b=0, softplus_a=0, softplus_b=0), "b2")):
        root = os.path.join(tmp, nm)
        _write_synthetic(root, sp, {"exp": 0.80, "softplus": 0.80})
        R, _ = load(root)
        bl |= set(analyse(R, out=silent)["breadth"].values())
        C = cells(R)
        for lr in RATES:
            for arm in BREADTH:
                a, s = C[lr][arm], C[lr]["signed"]
                pf = fisher_solved(a["k"], 8, s["k"], 8)
                pp = perm2_p(np.asarray(s["acc"]), np.asarray(a["acc"]))["p"]
                ref = cell_label(a["acc"], s["acc"], a["k"], s["k"], 8)[0]
                mine = breadth_label(a["acc"], s["acc"], a["k"], s["k"], pf, pp)
                if ref != mine:
                    print(f"  BAD breadth_label != cell_label on raw p ({mine!r} vs {ref!r})"); ok = False
    for lab in (SLB, NSC):
        if lab not in bl:
            print(f"  BAD breadth label never reached: {lab}"); ok = False
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    R, missing = load(a.root)
    if missing:
        print(f"MISSING {len(missing)} runs, e.g. {missing[:3]}")
        sys.exit(2)
    v = analyse(R)
    json.dump(v, open(f"{REPO}/SIGNROUTE_VERDICTS.json", "w"), indent=2)


if __name__ == "__main__":
    main()

"""Registered analysis for SIGNSEP_PREREG.md: which of the two accounts of the |.| cost -- the missing collective
route (A, SEARCH_THEORY S3) or the wrap barrier (B, S1 + P3 + P4) -- carries it on the ring.

  python3 -m mapformer.analyze_signsep               # reads runs/signsep/p0, writes SIGNSEP_VERDICTS.json
  python3 -m mapformer.analyze_signsep --selftest    # every label and reading, through load() on disk

Shared primitives (rule 5: import, do not re-derive): stats_core.classify_run (SOLVED = final-5% training loss < 0.05),
stats_core.fisher_solved, stats_core.perm2_p; `better` is analyze_signroute.better (the house SIGNED-LEARNS-BETTER
thresholds with the first arm in the reference slot): (Fisher < .05 and k_a > k_b) or (permutation p < .05 and
acc_a - acc_b >= +0.05).

Per rate r in {1e-3, 3e-3}, n = 8 seeds per arm; S signed, A abs (anchors), W abs_wrap, F signed_frozen (the new
cells), P abs_wrap_past (control C1). All five start at the same point (model_signsep.build).
  G0  k_S >= 6, else UNREADABLE at r.
  G1  k_A <= 3 and better(S, A), else NO COST TO EXPLAIN at r.
  Class of X in {W, F}:
      S-LIKE      better(X, A) and not better(S, X) and k_X >= k_S - 1
      A-LIKE      better(S, X) and not better(X, A) and k_X <= k_A + 1
      PARTIAL     better(X, A) or better(S, X), but neither of the above
      UNMEASURED  otherwise
  C1  (amended) k_P >= 6 AND the median final tau_sum of P >= 1.55 (Amendment 3; P starts at 1.5 with every channel past pi, where
      |.| learns without any route -- MECHANISM AO_high -- so k_P alone cannot show that tau is usable; rule 15).
  Sheet use (amended): W "on the sheet" iff over its SOLVED runs the median |tau_sum| >= 1.5 and the median distance of
      |tau_sum| to 2 Z \ {0} <= 0.5.
  Half verdicts:  F A-LIKE -> ROUTE NEEDED, F S-LIKE -> ROUTE NOT NEEDED;
                  W S-LIKE on the sheet -> BARRIER CROSSED, off it -> W SOLVED OFF THE SHEET;
                  W A-LIKE -> C1 FAILED (k_P < 6), else BARRIER OR PARAMETERISATION (P's tau did not move), else
                  W DIVERGED (any W run diverged at that rate), else BARRIER BLOCKS.
  Reading at r (the 2x2):
      W S-LIKE (sheet), F A-LIKE        -> ROUTE          (account A: the route is what |.| lacks; with it the barrier
                                                           is crossed)
      W A-LIKE (BLOCKS), F S-LIKE       -> BARRIER        (account B: removing the barrier suffices; a route does not
                                                           get past it)
      W A-LIKE (BLOCKS), F A-LIKE       -> BOTH           (each missing piece alone blocks learning)
      W S-LIKE (sheet), F S-LIKE        -> NEITHER ALONE  (either remedy suffices; |.| fails because it lacks both)
      W S-LIKE off the sheet, or W A-LIKE not BLOCKS -> that half verdict (not definite)
      otherwise                         -> PARTIAL if any class is PARTIAL, else UNMEASURED
Overall (amended, finding 4): UNREADABLE (no rate passes G0 and G1); MIXED if W or F is S-LIKE at one readable rate and
A-LIKE at the other; the reading if both rates give the same definite reading; MIXED if two different definite
readings; "<reading> AT ONE RATE" if exactly one readable rate is definite; otherwise C1 FAILED / BARRIER OR
PARAMETERISATION / W DIVERGED / W SOLVED OFF THE SHEET / PARTIAL / UNMEASURED in that order of precedence.

Diverged runs (non-finite loss anywhere): DIVERGED, never SOLVED, non-finite accuracy counted 0 (as SIGNROUTE). Every
run's recipe and len(losses) == 300 are asserted.
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

from mapformer.analyze_signroute import better, holm  # noqa: F401  (holm kept for parity; not used by the primary)
from mapformer.stats_core import classify_run, fisher_solved

REPO = "/home/prashr/mapformer"                    # rule 25
ROOT = f"{REPO}/runs/signsep/p0"
SEEDS = list(range(2300, 2308))
RATES = (1e-3, 3e-3)
EPOCHS = 300
ARMS = ("signed", "abs", "abs_wrap", "signed_frozen", "abs_wrap_past")
FLOORS = {"blank": 0.50, "ngram": 0.554, "retrace": 0.860}   # LRSWEEP_RESULTS.md, ring 32, +1/-1
RECIPE = {"size": 32, "n_steps": 128, "batch_size": 128, "n_batches": 98, "data_workers": 3, "n_trials": 400,
          "epochs": EPOCHS, "weight_decay": 0.05, "deltas": "1,-1", "init_from": None,
          "tau_decay": "off"}     # Amendment 2
SL, AL, PART, UNM = "S-LIKE", "A-LIKE", "PARTIAL", "UNMEASURED"
UNR, NCE = "UNREADABLE", "NO COST TO EXPLAIN"
ROUTE, BARRIER, BOTH, NEITHER, C1F = "ROUTE", "BARRIER", "BOTH", "NEITHER ALONE", "C1 FAILED"
BOP, WDIV, OFF = "BARRIER OR PARAMETERISATION", "W DIVERGED", "W SOLVED OFF THE SHEET"
C1_TAU = 1.55          # C1: median final tau_sum of abs_wrap_past (start 1.5) must move toward the sheet (Amendment 3)
SHEET_TOL = 0.5        # W on the sheet: median over solved W runs of |tau_sum| distance to 2 Z \ {0} <= 0.5
DEFINITE = (ROUTE, BARRIER, BOTH, NEITHER)


def rate_tag(lr):
    """1e-3 -> lr1e-3 (the driver checks its directory tags against this function)."""
    return f"lr{lr:.0e}".replace("e-0", "e-")


def run_dir(root, arm, lr, seed):
    return os.path.join(root, f"{arm}_{rate_tag(lr)}_s{seed}")


def arm_class(X, S, A, n):
    xA, st_xa = better(X["k"], A["k"], X["acc"], A["acc"], n)
    sX, st_sx = better(S["k"], X["k"], S["acc"], X["acc"], n)
    det = {"X_vs_A": st_xa, "S_vs_X": st_sx, "better_X_A": bool(xA), "better_S_X": bool(sX)}
    if xA and not sX and X["k"] >= S["k"] - 1:
        return SL, det
    if sX and not xA and X["k"] <= A["k"] + 1:
        return AL, det
    if xA or sX:
        return PART, det
    return UNM, det


def _sheet_dist(t):
    """Distance of tau_sum from the nearest collective wrap sheet tau_sum in 2 Z \ {0} (|.| >= 1.5 required)."""
    a = abs(t)
    return abs(a - 2 * max(1, round(a / 2)))


def rate_reading(cell, n):
    """cell: {arm: {"k", "acc", "rows"}} at one rate -> (reading, details). Amendment 1 (findings 1, 2, 10): C1 also
    needs the control to MOVE tau toward the sheet; a W S-LIKE reading needs the solved W runs to sit on a sheet; a W
    A-LIKE reading is not read as the barrier if any W run diverged."""
    S, A = cell["signed"], cell["abs"]
    if S["k"] < 6:
        return f"{UNR} (signed SOLVED {S['k']}/{n} < 6)", {}
    sA, st = better(S["k"], A["k"], S["acc"], A["acc"], n)
    if not (A["k"] <= 3 and sA):
        return f"{NCE} (signed {S['k']}/{n} vs abs {A['k']}/{n}; better {sA})", {"S_vs_A": st}
    cW, dW = arm_class(cell["abs_wrap"], S, A, n)
    cF, dF = arm_class(cell["signed_frozen"], S, A, n)
    P, Wr = cell["abs_wrap_past"], cell["abs_wrap"]["rows"]
    tauP = [r["tau_sum"] for r in P["rows"] if r.get("tau_sum") is not None and np.isfinite(r["tau_sum"])]
    tauP_med = float(np.median(tauP)) if tauP else float("nan")
    c1_k = P["k"] >= 6
    c1_tau = bool(np.isfinite(tauP_med) and tauP_med >= C1_TAU)
    solvedW = [r["tau_sum"] for r in Wr if r["solved"] and r.get("tau_sum") is not None and np.isfinite(r["tau_sum"])]
    sheet_med = float(np.median([_sheet_dist(t) for t in solvedW])) if solvedW else float("nan")
    on_sheet = bool(solvedW and np.median([abs(t) for t in solvedW]) >= 1.5 and sheet_med <= SHEET_TOL)
    w_div = int(sum(r["div"] for r in Wr))
    half_route = {AL: "ROUTE NEEDED", SL: "ROUTE NOT NEEDED"}.get(cF, cF)
    if cW == SL:
        half_barrier = "BARRIER CROSSED" if on_sheet else OFF
    elif cW == AL:
        half_barrier = (C1F if not c1_k else BOP if not c1_tau else WDIV if w_div else "BARRIER BLOCKS")
    else:
        half_barrier = cW
    det = {"S_vs_A": st, "W": {"class": cW, **dW}, "F": {"class": cF, **dF}, "k_P": P["k"],
           "C1_solved": bool(c1_k), "C1_tau_median": tauP_med, "C1_tau": c1_tau,
           "W_solved_tau_sums": solvedW, "W_on_sheet": on_sheet, "W_diverged": w_div,
           "half_route": half_route, "half_barrier": half_barrier}
    if cW == SL and not on_sheet:
        return OFF, det
    if cW == AL and half_barrier != "BARRIER BLOCKS":
        return half_barrier, det
    table = {(SL, AL): ROUTE, (AL, SL): BARRIER, (AL, AL): BOTH, (SL, SL): NEITHER}
    if (cW, cF) in table:
        return table[(cW, cF)], det
    return (PART if PART in (cW, cF) else UNM), det


def overall(labels, details=None):
    """Amendment 1 finding 4: if W or F is S-LIKE at one readable rate and A-LIKE at the other, the overall reading is
    MIXED whatever the per-rate readings are."""
    readable = {r: l for r, l in labels.items() if not (l.startswith(UNR) or l.startswith(NCE))}
    if not readable:
        return UNR
    if details is not None and len(readable) == 2:
        for x in ("W", "F"):
            cls = {details[r][x]["class"] for r in readable}
            if SL in cls and AL in cls:
                return "MIXED"
    defin = [l for l in readable.values() if l in DEFINITE]
    if len(defin) == 2 and defin[0] == defin[1]:
        return defin[0]
    if len(defin) == 2:
        return "MIXED"
    if len(defin) == 1:
        return f"{defin[0]} AT ONE RATE"
    for lab in (C1F, BOP, WDIV, OFF, PART):
        if lab in readable.values():
            return lab
    return UNM


# ------------------------------------------------------------------------------ loading
def _fin(x):
    return float(x) if x is not None and np.isfinite(x) else 0.0


DIAG = ("u_ratio", "latent_cos", "clock_wrapped_rms", "cancel_share", "cancel_zero_share", "cancel_wrap_share",
        "tau_sum", "pre_frac_negative", "step_frac_negative")


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
                                   "k0_median": (e.get("start") or {}).get("k0_median"),
                                   **{q: e.get(q) for q in DIAG}}
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
    return f"{np.median(v):+.3f}" if v else "   n/a"


# ------------------------------------------------------------------------------ analysis
def analyse(R, seeds=SEEDS, out=print):
    n = len(seeds)
    C = cells(R, seeds)
    labels, details = {}, {}
    for lr in RATES:
        labels[lr], details[lr] = rate_reading(C[lr], n)
    reading = overall(labels, details)

    out("== SIGNSEP: registered 2x2 (W = route + barrier, F = no route + no barrier; anchors S, A; control C1) ==")
    for lr in RATES:
        c = C[lr]
        out(f"lr {lr:g}: SOLVED " + " ".join(f"{a} {c[a]['k']}/{n}" for a in ARMS)
            + " | acc T=128 " + " ".join(f"{a} {np.mean(c[a]['acc']):.4f}" for a in ARMS))
        out(f"   reading: {labels[lr]}")
        out(f"   {json.dumps(details[lr])}")
    out("diverged runs per arm x rate: " + ", ".join(f"{a}|{lr:g} {sum(r['div'] for r in C[lr][a]['rows'])}"
                                                     for lr in RATES for a in ARMS))
    out(f"\nREADING (registered): {reading}")
    out("   scope: ring 32, +1/-1, observation-step-free rank 4, the common |.| start, 300 epochs, 'at lr 1e-3 and "
        "3e-3'. ROUTE / BARRIER / BOTH / NEITHER ALONE are statements about these two constructed classes against "
        "the two anchors, nothing wider.")

    out("\n== diagnostics (medians per arm x rate; descriptive, registered as such) ==")
    out("   arm            lr     u_ratio  cos(zx,zy) clock_rms cancel  c_zero  c_wrap  tau_sum  pre<0   k0_med")
    for lr in RATES:
        for arm in ARMS:
            rows = C[lr][arm]["rows"]
            out(f"   {arm:14s} {lr:g}  " + "  ".join(_med(rows, q) for q in
                ("u_ratio", "latent_cos", "clock_wrapped_rms", "cancel_share", "cancel_zero_share",
                 "cancel_wrap_share", "tau_sum", "pre_frac_negative", "k0_median")))
    out("\n   W / W_past runs, tau_sum at the end per seed (S = SOLVED): the collective wrap sheet is tau_sum = +-2")
    for lr in RATES:
        for arm in ("abs_wrap", "abs_wrap_past"):
            rows = C[lr][arm]["rows"]
            out(f"   {lr:g} {arm:14s} " + " ".join(
                f"{(r['tau_sum'] if r.get('tau_sum') is not None else float('nan')):+.2f}{'S' if r['solved'] else '-'}"
                for r in rows))

    out("\n== rule 2: r(log10 final loss, acc) per eval length, finite runs ==")
    fin = [r for r in R.values() if not r["div"]]
    L = np.array([r["final_loss"] for r in fin])
    for key, nm in (("acc", "T=128"), ("acc4", "T=512")):
        Acc = np.array([r[key] for r in fin])
        rr = float(np.corrcoef(np.log10(L + 1e-6), Acc)[0, 1]) if len(Acc) > 2 and Acc.std() > 0 and L.std() > 0 \
            else float("nan")
        out(f"   {nm}: r = {rr:+.3f} over {len(Acc)} runs ({len(R) - len(fin)} diverged excluded)")
    out(f"\nfloors on record (ring 32, +1/-1): blank {FLOORS['blank']}, n-gram {FLOORS['ngram']}, "
        f"retrace {FLOORS['retrace']}")
    code = {"SOLVED": "S", "STALLED": "T", "DESCENDING": "D", "RISING": "R", "DIVERGED": "X"}
    out("\nper seed (acc T=128 + class: S SOLVED, T STALLED, D DESCENDING, R RISING, X DIVERGED), lr x arm:")
    for lr in RATES:
        for arm in ARMS:
            out(f"  {lr:g} {arm:14s} " + " ".join(f"{r['acc']:.3f}{code[r['cls']]}" for r in C[lr][arm]["rows"]))
    return {"reading": reading, "per_rate": {f"{lr:g}": labels[lr] for lr in RATES},
            "detail": {f"{lr:g}": details[lr] for lr in RATES},
            "solved": {f"{arm}|{lr:g}": C[lr][arm]["k"] for lr in RATES for arm in ARMS},
            "n_diverged": int(sum(r["div"] for r in R.values()))}


# ------------------------------------------------------------------------------ selftest
def _loss_curve(solved, diverge=False):
    L = list(np.linspace(1.5, 0.3, 150)) + ([1e-4] * 150 if solved else [0.3] * 150)
    if diverge:
        L[-10:] = [float("nan")] * 10
    return L


def _write_synthetic(root, spec, acc_lo=None, seed=0, tau=None):
    """REAL eval.json files under `root` in the driver's naming (load() and classify_run are exercised). spec:
    {(arm, lr): k solved}, (arm, lr, 'div') -> the last seed diverges; acc_lo: {arm: accuracy of unsolved runs}."""
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
                acc = (1.0 if sol else acc_lo.get(arm, 0.90)) - 0.002 * rng.random()
                e = {"eval": {"128": {"acc": float("nan") if div else acc, "nll": 0.0},
                              "512": {"acc": 0.99 if sol else 0.80, "nll": 0.0}},
                     "arm": arm, "lr": lr, "seed": s, "args": dict(RECIPE, arm=arm, lr=lr, seed=s),
                     "losses": _loss_curve(sol and not div, diverge=div),
                     "final_loss": float("nan") if div else (1e-4 if sol else 0.3),
                     "start": {"k0_median": 0.5}, "u_ratio": 0.1,
                     "tau_sum": (tau or {}).get(arm, 2.0) if arm.startswith("abs_wrap") else None}
                with open(os.path.join(d, "eval.json"), "w") as f:
                    json.dump(e, f)


def selftest():
    def ks(**kw):
        base = {}
        for lr in RATES:
            base.update({("signed", lr): 8, ("abs", lr): 1, ("abs_wrap", lr): 8, ("signed_frozen", lr): 1,
                         ("abs_wrap_past", lr): 8})
        short = {"S": "signed", "A": "abs", "W": "abs_wrap", "F": "signed_frozen", "P": "abs_wrap_past"}
        for key, v in kw.items():
            arm, r = key.split("_")
            base[(short[arm], {"a": 1e-3, "b": 3e-3}[r])] = v
        return base

    def both(**kw):            # the same counts at both rates
        out = {}
        for k, v in kw.items():
            out[f"{k}_a"] = v; out[f"{k}_b"] = v
        return ks(**out)

    def both_div(spec, arm):
        for lr in RATES:
            spec[(arm, lr, "div")] = True
        return spec

    silent = lambda *a, **k: None
    tmp = tempfile.mkdtemp(prefix="signsep_selftest_")
    cases = [
        (ROUTE, both(W=8, F=1), None),
        (BARRIER, both(W=1, F=8), None),
        (BOTH, both(W=1, F=1), None),
        (NEITHER, both(W=8, F=8), None),
        (C1F, both(W=1, F=1, P=3), None),
        (f"{ROUTE} AT ONE RATE", ks(W_b=4), None),                 # rate b: W 4/8 (8 vs 4 p .077, 4 vs 1 p .28) -> UNMEASURED
        ("MIXED", ks(W_b=1), None),                                 # a ROUTE, b BOTH
        (PART, both(W=5, F=1), {"abs_wrap": 0.99}),                 # W better than A (accuracy) but 5 < k_S - 1
        (UNM, both(W=4, F=4), None),
        (UNR, both(S=5), None),
        (UNR, both(A=4), None),                                     # G1: k_A > 3 at both rates
        (UNR, both(A=3, S=6), None),                                # G1: 6 vs 3 has no measured cost
        # Amendment 1 (findings 1, 2, 4, 10)
        (BOP, both(W=1, F=1), None, {"abs_wrap_past": 1.5}),        # C1 solved but tau never moved
        (OFF, both(W=8, F=1), None, {"abs_wrap": 0.3}),             # W solved off the collective sheet
        (WDIV, both_div(both(W=1, F=1), "abs_wrap"), None),         # a W failure that includes a divergence
        ("MIXED", ks(W_b=1, P_b=3, F_b=8), None),                   # a ROUTE, b C1 FAILED with F S-LIKE
        ("MIXED", ks(W_b=4, F_b=8), None),                          # a ROUTE, b UNMEASURED with F S-LIKE
        (f"{BOTH} AT ONE RATE", ks(W_a=1, F_a=1, W_b=4, F_b=1), None),
    ]
    ok = True
    seen_read, seen_cls = set(), set()
    for i, case in enumerate(cases):
        want, spec, lo = case[:3]
        tau = case[3] if len(case) > 3 else None
        root = os.path.join(tmp, f"c{i}")
        _write_synthetic(root, spec, lo, tau=tau)
        R, missing = load(root)
        assert not missing, missing[:2]
        v = analyse(R, out=silent)
        got = v["reading"]
        ok &= got == want
        seen_read |= set(v["per_rate"].values())
        for d in v["detail"].values():
            for x in ("W", "F"):
                if x in d:
                    seen_cls.add(d[x]["class"])
        print(f"  {'ok ' if got == want else 'BAD'} want {want!r:24s} got {got!r:24s} per rate {v['per_rate']}")
    for lab in (ROUTE, BARRIER, BOTH, NEITHER, C1F, BOP, WDIV, OFF, PART, UNM):
        if lab not in seen_read:
            print(f"  BAD per-rate reading never reached: {lab}"); ok = False
    if not any(s.startswith(UNR) for s in seen_read) or not any(s.startswith(NCE) for s in seen_read):
        print("  BAD UNREADABLE / NO COST TO EXPLAIN gate never reached"); ok = False
    for lab in (SL, AL, PART, UNM):
        if lab not in seen_cls:
            print(f"  BAD arm class never reached: {lab}"); ok = False
    # a NaN run is DIVERGED, never SOLVED, accuracy 0, excluded from rule 2
    root = os.path.join(tmp, "div")
    sp = both(W=8, F=1); sp[("abs_wrap", 1e-3, "div")] = True
    _write_synthetic(root, sp)
    R, _ = load(root)
    r = R[("abs_wrap", 1e-3, SEEDS[-1])]
    if not (r["div"] and not r["solved"] and r["cls"] == "DIVERGED" and r["acc"] == 0.0):
        print(f"  BAD diverged run handled wrongly: {r}"); ok = False
    v = analyse(R, out=silent)
    if v["n_diverged"] != 1:
        print(f"  BAD n_diverged {v['n_diverged']}"); ok = False
    # power facts quoted in the prereg (Fisher two-sided, as stats_core.fisher_solved computes it)
    for (a, b, sig) in ((8, 3, True), (7, 2, True), (8, 4, False), (7, 3, False), (6, 1, True), (6, 2, False)):
        p = fisher_solved(b, 8, a, 8)
        if (p < 0.05) != sig:
            print(f"  BAD power fact {a} vs {b}: p {p:.4f}"); ok = False
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
    json.dump(v, open(f"{REPO}/SIGNSEP_VERDICTS.json", "w"), indent=2)


if __name__ == "__main__":
    main()

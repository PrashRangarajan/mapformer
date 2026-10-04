"""Readouts and registered verdicts for TW_NORMSTEP_PREREG.md.

--readouts: compute per-run readouts (tw_normstep_readouts.readouts, CPU) for all 32 runs into TW_NORMSTEP.json,
then print the verdicts. Without it, read TW_NORMSTEP.json. --runs-dir / --seeds / --out for the pilot.
"""
import argparse
import json

import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, fisher_solved, mde, signflip_p

REPO = "/home/prashr/mapformer"
ARMS = ["MapWM", "NormStep", "NormStepNB", "DirOnly"]
ACC_MIN, DRIFT_MIN, CLOCK_SEED, OPT_MIN = 0.02, 4, 16, 0.05   # materiality thresholds (prereg + Amendment 1)


def collect(rdir, seeds, do_readouts, out):
    if not do_readouts:
        return json.load(open(out))
    from mapformer.tw_normstep_readouts import readouts
    torch.set_num_threads(8); res = {}
    for s in seeds:
        for a in ARMS:
            d = f"{rdir}/{a}_s{s}"; ev = json.load(open(f"{d}/eval.json"))
            ck = torch.load(f"{d}/{a}.pt", map_location="cpu", weights_only=False)
            c = classify_run(ck["losses"])
            r = readouts(f"{d}/{a}.pt")
            r.update(acc=ev["eval"]["1024"]["acc"], acc2048=ev["eval"]["2048"]["acc"], cls=c["registered"],
                     tail=float(c["tail"]))
            res[f"{a}_s{s}"] = r
            print(f"  {a:10s} s{s}: acc {r['acc']:.4f}  {r['cls']:10s}  drift {r['drift']:2d}/64  "
                  f"word drift {r['drift_opt_rad']:.4f} rad ({r['drift_opt']:2d}/64)  move {r['move']:.3f}"
                  f"  bias_step {r['bias_step'] if r['bias_step'] is None else round(r['bias_step'], 4)}"
                  f"  ablate {r['ablate_nondir']:.4f}  own map {r['acc_own']:.4f}  "
                  f"eval/train mode {r['acc40_eval']:.4f}/{r['acc40_train']:.4f}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    return res


def contrast(x, y, thr):
    """mean(y) - mean(x), exact permutation p, fires if p < .05 and |d| >= thr."""
    d = float(np.mean(y) - np.mean(x)); p = perm2_p(x, y)["p"]
    return d, p, (p < 0.05 and abs(d) >= thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", action="store_true")
    ap.add_argument("--runs-dir", default=f"{REPO}/runs/tw_normstep/p0")
    ap.add_argument("--seeds", default="10,11,12,13,14,15,16,17")
    ap.add_argument("--out", default=f"{REPO}/TW_NORMSTEP.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    res = collect(a.runs_dir, seeds, a.readouts, a.out)
    g = lambda arm, k: [res[f"{arm}_s{s}"][k] for s in seeds]
    n = len(seeds)

    print(f"\n== per arm (n={n}): T=1024 held-out acc | SOLVED | drift channels of 64 | clock seeds (drift > {CLOCK_SEED})"
          " | move ratio | non-direction steps zeroed | [T=2048] ==")
    for arm in ARMS:
        acc, dr = g(arm, "acc"), g(arm, "drift")
        print(f"  {arm:10s} {np.mean(acc):.4f} +/- {np.std(acc, ddof=1) if n > 1 else 0:.4f}  "
              f"{g(arm, 'cls').count('SOLVED')}/{n}  drift {np.mean(dr):5.1f} ({' '.join(map(str, dr))})  "
              f"clock {sum(d > CLOCK_SEED for d in dr)}/{n}  word drift {np.mean(g(arm, 'drift_opt_rad')):.4f} rad  "
              f"move {np.mean(g(arm, 'move')):.3f}  "
              f"ablate {np.mean(g(arm, 'ablate_nondir')):.4f}  own map {np.mean(g(arm, 'acc_own')):.4f}  "
              f"[{np.mean(g(arm, 'acc2048')):.4f}]  train-mode {np.mean(g(arm, 'acc40_train')):.4f} "
              f"(eval {np.mean(g(arm, 'acc40_eval')):.4f}, same 40 walks)")
    bs = g("NormStep", "bias_step")
    print(f"  NormStep bias step / direction step: {' '.join(f'{b:.4f}' for b in bs)}")
    for s_ in seeds:                                   # registered void check (Amendment 1: asserted)
        r = res[f"DirOnly_s{s_}"]
        assert r["move"] == 0.0, ("VOID: DirOnly non-direction step", s_, r["move"])
        assert abs(r["ablate_nondir"] - r["acc"]) <= 0.002, ("VOID: DirOnly ablation changed accuracy", s_)
    print("  void check: DirOnly non-direction steps exactly 0 and ablation a no-op on every seed -> OK")
    if n < 8:
        print("\n(pilot: no verdicts below n=8)"); return

    sol = lambda arm: g(arm, "cls").count("SOLVED")
    print("\n== primary A: accuracy, NormStep - MapWM (fires: perm p < .05 and |d| >= 0.02, or Fisher p < .05) ==")
    d, p, f = contrast(g("MapWM", "acc"), g("NormStep", "acc"), ACC_MIN)
    pf = fisher_solved(sol("MapWM"), n, sol("NormStep"), n); ds = sol("NormStep") - sol("MapWM")
    sd = np.sqrt((np.var(g("MapWM", "acc"), ddof=1) + np.var(g("NormStep", "acc"), ddof=1)) / 2)
    print(f"  d {d:+.4f}, perm p {p:.4f}; SOLVED {sol('NormStep')}/{n} vs {sol('MapWM')}/{n}, Fisher p {pf:.4f}; "
          f"MDE (exact t, pooled sd, 2-sample) {mde(sd, n) * np.sqrt(2):.4f}")
    m_ = max(mde(sd, n) * np.sqrt(2), ACC_MIN)
    if f:                                              # Amendment 1: accuracy decides the direction
        va = "NORMSTEP HURTS IN WORDS" if d < 0 else "NORMSTEP HELPS IN WORDS"
    elif pf < 0.05:
        va = f"SOLVED RATE {'LOWER' if ds < 0 else 'HIGHER'} FOR NORMSTEP (convergence; accuracy unmeasured below {m_:.4f})"
    else:
        va = f"NO DETECTABLE DIFFERENCE (unmeasured below {m_:.4f})"
    print(f"  REGISTERED A: {va}")

    print(f"\n== primary B (Amendment 1): per-word clock, drift_opt_rad (fires: perm p < .05 and |d| >= {OPT_MIN} rad) ==")
    d1, p1, f1 = contrast(g("MapWM", "drift_opt_rad"), g("NormStep", "drift_opt_rad"), OPT_MIN)
    d2, p2, f2 = contrast(g("NormStepNB", "drift_opt_rad"), g("NormStep", "drift_opt_rad"), OPT_MIN)
    print(f"  NormStep - MapWM      {d1:+.4f} rad, perm p {p1:.4f}  {'FIRES' if f1 else ''}")
    print(f"  NormStep - NormStepNB {d2:+.4f} rad, perm p {p2:.4f}  {'FIRES' if f2 else ''}")
    if f1 and d1 > 0:
        vb = "WORD-COUNT CLOCK (removing the bias removes it)" if (f2 and d2 > 0) else \
            "WORD-COUNT CLOCK, BIAS NOT SHOWN TO CAUSE IT"
    elif f1 and d1 < 0:
        vb = "LESS PER-WORD DRIFT THAN MAPWM"
    else:
        vb = "NO WORD-COUNT CLOCK DETECTED (sensitivity: an injected tick of 0.1 direction step adds 0.09-0.43 rad)"
    print(f"  REGISTERED B: {vb}")
    carries = not va.startswith("NORMSTEP HURTS") and not vb.startswith("WORD-COUNT")
    print(f"  COMPOSITE: {'NORMSTEP CARRIES OVER TO WORDS' if carries else 'NORMSTEP DOES NOT CARRY OVER CLEANLY'}"
          + (" (no per-word clock above the sensitivity limit)" if carries and vb.startswith("NO WORD") else ""))

    print(f"\n== declared secondary (formerly primary B): all-word drift channels (fires: p < .05, |d| >= {DRIFT_MIN});"
          " bimodal per seed, needs a swing of ~4 clock seeds: a non-firing is UNMEASURED ==")
    e1, q1, h1 = contrast(g("MapWM", "drift"), g("NormStep", "drift"), DRIFT_MIN)
    e2, q2, h2 = contrast(g("NormStepNB", "drift"), g("NormStep", "drift"), DRIFT_MIN)
    print(f"  NormStep - MapWM      {e1:+.2f} channels, perm p {q1:.4f}  {'FIRES' if h1 else 'unmeasured'}")
    print(f"  NormStep - NormStepNB {e2:+.2f} channels, perm p {q2:.4f}  {'FIRES' if h2 else 'unmeasured'}")

    print("\n== secondaries (no verdict) ==")
    for x, y in (("MapWM", "NormStepNB"), ("NormStep", "NormStepNB"), ("MapWM", "DirOnly"), ("NormStep", "DirOnly")):
        dd, pp, _ = contrast(g(x, "acc"), g(y, "acc"), 0)
        de, pe, _ = contrast(g(x, "drift"), g(y, "drift"), 0)
        print(f"  {y} - {x}: acc {dd:+.4f} (p {pp:.4f}), SOLVED {sol(y)} vs {sol(x)} (Fisher "
              f"{fisher_solved(sol(x), n, sol(y), n):.4f}); drift {de:+.2f} (p {pe:.4f})")
    cm, cn = sum(x > CLOCK_SEED for x in g("MapWM", "drift")), sum(x > CLOCK_SEED for x in g("NormStep", "drift"))
    print(f"  clock seeds NormStep {cn}/{n} vs MapWM {cm}/{n}, Fisher p {fisher_solved(cm, n, cn, n):.4f}")
    dt, pt_, _ = contrast(g("MapWM", "acc40_train"), g("NormStep", "acc40_train"), 0)
    print(f"  train-mode accuracy (40 walks) NormStep - MapWM {dt:+.4f}, perm p {pt_:.4f}; mode gap train - eval per arm: "
          + ", ".join(f"{a_} {np.mean(np.array(g(a_, 'acc40_train')) - np.array(g(a_, 'acc40_eval'))):+.4f}" for a_ in ARMS))
    for x_, y_ in (("MapWM", "NormStep"), ("MapWM", "NormStepNB")):
        dd = np.array(g(y_, "acc")) - np.array(g(x_, "acc")); do = np.array(g(y_, "drift_opt_rad")) - np.array(g(x_, "drift_opt_rad"))
        print(f"  paired by seed {y_} - {x_}: acc {dd.mean():+.4f} sign-flip p {signflip_p(dd)['p']:.4f}; "
              f"word drift {do.mean():+.4f} sign-flip p {signflip_p(do)['p']:.4f}")
    x = sum((g(arm, "tail") for arm in ARMS), []); y = sum((g(arm, "acc") for arm in ARMS), [])
    print(f"  r(final loss, acc) over {len(x)} runs: {np.corrcoef(x, y)[0, 1]:+.3f}")
    for arm in ARMS:
        r = np.corrcoef(g(arm, "drift"), g(arm, "acc"))[0, 1] if np.std(g(arm, "drift")) > 0 else float("nan")
        print(f"  r(drift, acc) within {arm}: {r:+.3f}")


if __name__ == "__main__":
    main()

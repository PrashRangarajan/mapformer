"""Registered verdicts for TW_AMBIG_PREREG.md.

--readouts: compute per-run readouts (tw_ambig_readouts.readouts) for every run into TW_AMBIG.json, then print the
verdicts. Without it, read TW_AMBIG.json. --runs-dir / --seeds / --out for the pilot. --smoke: run the decision
functions on synthetic data built to land in every branch (no runs read).

The decision functions `verdict_a(res, seeds, acc_key)` and `verdict_b(res, seeds)` are pure; the power script
(docs/audits/2026-10-08/tw_ambig_power.py) and the smoke test call them unchanged.
"""
import argparse
import json
import math

import numpy as np

from mapformer.stats_core import classify_run, perm2_p, perm2_ci, fisher_solved, mde, signflip_p

REPO = "/home/prashr/mapformer"
ARMS = ["MapWM", "RoleTag", "DirOnlyRole", "HSR", "CF2", "RoPE1", "RoPE2"]
NM_CLASSES = ["nat", "lead_near", "lead_far", "trail_near", "trail_far"]
NEAR, FAR = ["nat", "lead_near", "trail_near"], ["lead_far", "trail_far"]
ACC_MIN, NI_MARGIN, RATIO_SUPP, STEP_MIN, WIRING_TOL = 0.02, 0.03, 0.3, 0.01, 0.01


def contrast(x, y, thr=ACC_MIN):
    """mean(y) - mean(x); exact (or MC) permutation p; fires if p < .05 and |d| >= thr."""
    d = float(np.mean(y) - np.mean(x)); p = perm2_p(x, y)["p"]
    return d, p, (p < 0.05 and abs(d) >= thr)


def noninferior(ref, x, margin=NI_MARGIN):
    """x non-inferior to ref at `margin`: the permutation test of mean(x) - mean(ref) = -margin rejects (p < .05) with
    the observed difference above -margin (test inversion of perm2_ci at the margin)."""
    d = float(np.mean(x) - np.mean(ref))
    return bool(d > -margin and perm2_p(ref, x, shift=-margin)["p"] < 0.05)


def verdict_a(res, seeds, acc_key="acc"):
    g = lambda arm: [res[f"{arm}_s{s}"][acc_key] for s in seeds]
    sol = lambda arm: sum(res[f"{arm}_s{s}"]["cls"] == "SOLVED" for s in seeds)
    n = len(seeds); need = math.ceil(0.75 * n)
    out = {"n": n, "RoleTag_solved": sol("RoleTag")}
    dN, pN, fN = contrast(g("MapWM"), g("RoleTag")); dR, pR, fR = contrast(g("CF2"), g("HSR"))
    dG, pG, fG = contrast(g("RoleTag"), g("HSR"))
    ni_h = noninferior(g("RoleTag"), g("HSR")); ni_m = noninferior(g("RoleTag"), g("MapWM"))
    out.update(N=(dN, pN, fN), R=(dR, pR, fR), G=(dG, pG, fG), NI_HSR=ni_h, NI_MapWM=ni_m)
    if sol("RoleTag") < need:
        lab = f"UNINTERPRETABLE: THE ORACLE DID NOT LEARN (RoleTag SOLVED {sol('RoleTag')}/{n} < {need})"
    elif fN and dN > 0:
        if fR and dR > 0:
            if fG and dG < 0:
                lab = "CONTEXT STEP NEEDED; HSR RECOVERS PART OF IT (below the oracle)"
            elif fG and dG > 0:
                lab = "CONTEXT STEP NEEDED; HSR RECOVERS IT AND EXCEEDS THE ORACLE"
            elif ni_h:
                lab = f"CONTEXT STEP NEEDED; HSR RECOVERS IT (non-inferior to the oracle at {NI_MARGIN})"
            else:
                lab = f"CONTEXT STEP NEEDED; HSR RECOVERS IT (gap to the oracle unmeasured at {NI_MARGIN})"
        elif fR and dR < 0:
            lab = "CONTEXT STEP NEEDED; HSR BELOW ITS CONTEXT-FREE TWIN"
        else:
            lab = "CONTEXT STEP NEEDED; THE CONTEXT STEP FAILS TOO"
    elif fN and dN < 0:
        lab = "ORACLE BELOW CONTEXT-FREE (a defect of the oracle or the task; no claim)"
    elif ni_m:
        lab = f"CONTEXT-FREE COPES (MapWM non-inferior to the oracle at {NI_MARGIN})"
    else:
        lab = "UNMEASURED (the need is not shown and neither is non-inferiority)"
    q = []
    if "RoPE1_s%d" % seeds[0] in res and np.mean(g("MapWM")) - np.mean(g("RoPE1")) < ACC_MIN:
        q.append("MapWM AT THE INDEX FLOOR")
    if all(res[f"{a}_s{s}"][acc_key] >= 0.99 for a in ("RoleTag", "HSR", "MapWM") for s in seeds):
        q.append("CEILING: RoleTag, HSR and MapWM >= 0.99 on every seed")
    out["label"] = lab; out["qualifiers"] = q
    return out


def verdict_b(res, seeds):
    """Mechanism: does HSR ignore non-movement uses, per class (swap ratio at 15 tokens, seeds that learned a step)."""
    n = len(seeds); h = [res[f"HSR_s{s}"] for s in seeds]
    learned = [r for r in h if (r.get("move_change") or 0) >= STEP_MIN]
    out = {"learned": len(learned), "median_ratio": {}, "suppress": {}}
    for c in NM_CLASSES:
        v = [r[f"ratio_{c}"] for r in learned if r.get(f"ratio_{c}") is not None]
        med = float(np.median(v)) if v else float("nan")
        out["median_ratio"][c] = med; out["suppress"][c] = bool(v) and med <= RATIO_SUPP
    if len(learned) < n - 1:
        out["label"] = f"HSR DID NOT RELIABLY LEARN A STEP ({len(learned)}/{n} seeds with move change >= {STEP_MIN}); B NOT READ"
        return out
    s = out["suppress"]; yes = [c for c in NM_CLASSES if s[c]]
    if len(yes) == len(NM_CLASSES):
        lab = "HSR IGNORES NON-MOVEMENT USES AT EVERY CUE DISTANCE"
    elif all(s[c] for c in NEAR) and not any(s[c] for c in FAR):
        lab = "HSR IGNORES NEAR-CUE USES ONLY"
    elif all(s[c] for c in FAR) and not any(s[c] for c in NEAR):
        lab = "HSR IGNORES FAR-CUE USES ONLY"
    elif yes:
        lab = "HSR IGNORES SOME CLASSES: " + ", ".join(yes) + " (not: " + ", ".join(c for c in NM_CLASSES if not s[c]) + ")"
    else:
        lab = "HSR TREATS NON-MOVEMENT USES AS MOVES"
    out["label"] = lab
    return out


def void_checks(res, seeds):
    """Wiring (registered void conditions): context-free arms have swap ratio 1.00 +/- 0.01 on every class and seed;
    DirOnlyRole's non-movement swap is exactly 0. Returns a list of violations."""
    bad = []
    for s in seeds:
        for a in ("MapWM", "CF2", "RoleTag", "DirOnlyRole"):
            r = res.get(f"{a}_s{s}")
            if r is None:
                bad.append(f"missing {a}_s{s}"); continue
            for c in NM_CLASSES:
                v = r.get(f"ratio_{c}")
                if a in ("MapWM", "CF2") and (v is None or abs(v - 1) > WIRING_TOL):
                    bad.append(f"{a}_s{s} ratio_{c} {v}")
                if a == "DirOnlyRole" and r.get(f"{c}@15") != 0.0:
                    bad.append(f"{a}_s{s} {c}@15 {r.get(f'{c}@15')}")
    return bad


def collect(rdir, seeds, do_readouts, out, dev):
    if not do_readouts:
        return json.load(open(out))
    import torch
    from mapformer.tw_ambig_readouts import readouts
    torch.set_num_threads(8); res = {}
    for s in seeds:
        for a in ARMS:
            d = f"{rdir}/{a}_s{s}"; ev = json.load(open(f"{d}/eval.json"))
            ck = torch.load(f"{d}/{a}.pt", map_location="cpu", weights_only=False)
            c = classify_run(ck["losses"])
            r = readouts(f"{d}/{a}.pt", dev)
            r["acc_readout"] = r.pop("acc")
            e = ev["eval"]
            r.update(acc=e["1024"]["acc"], acc2048=e["2048"]["acc"], acc_tm=e["1024"]["acc_train_mode"],
                     cls=c["registered"], tail=float(c["tail"]), final_loss=ev["final_loss"])
            res[f"{a}_s{s}"] = r
            print(f"  {a:11s} s{s}: acc {r['acc']:.4f} (readout {r['acc_readout']:.4f}) train-mode {r['acc_tm']:.4f} "
                  f"{r['cls']:10s} clean/contam {r['acc_clean']:.3f}/{r['acc_contam']:.3f}"
                  + (f"  move {r['move_change']:.3f} ratios " + " ".join(f"{c} {r[f'ratio_{c}']:.2f}" for c in NM_CLASSES)
                     + f"  nm drift {r['nm_drift_rad']:.3f} rad, drift {r['drift']}/64, reliance {r['reliance']:+.3f}"
                     if "move_change" in r else ""), flush=True)
    json.dump(res, open(out, "w"), indent=1)
    return res


def report(res, seeds):
    n = len(seeds); g = lambda arm, k: [res[f"{arm}_s{s}"].get(k) for s in seeds]
    sol = lambda arm: sum(x == "SOLVED" for x in g(arm, "cls"))
    print(f"\n== per arm (n={n}): T=1024 held-out acc (eval mode) | SOLVED | train mode | rescored | clean / contaminated gaps"
          " | [T=2048] ==")
    for arm in ARMS:
        acc = g(arm, "acc")
        print(f"  {arm:11s} {np.mean(acc):.4f} +/- {np.std(acc, ddof=1) if n > 1 else 0:.4f}  {sol(arm)}/{n}  "
              f"tm {np.mean(g(arm, 'acc_tm')):.4f}  rescored {np.mean(g(arm, 'acc_rescored')):.4f}  "
              f"clean {np.nanmean([x if x is not None else np.nan for x in g(arm, 'acc_clean')]):.4f} / contam "
              f"{np.nanmean([x if x is not None else np.nan for x in g(arm, 'acc_contam')]):.4f}  "
              f"[{np.mean(g(arm, 'acc2048')):.4f}]  per seed " + " ".join(f"{x:.3f}" for x in acc))
    print("  floors (gate, this eval stream): best constant 0.5112, reversal-copy 0.6132; contaminated-path oracle 0.5998"
          " (an estimate, NOT a bound: a trained context-free model beat it by +0.19 on ctx3 lead/far)")
    dif = [abs(res[f"{a}_s{s}"]["acc_readout"] - res[f"{a}_s{s}"]["acc"]) for a in ARMS for s in seeds
           if "acc_readout" in res[f"{a}_s{s}"]]
    if dif:
        print(f"  consistency: readout re-computation of the registered accuracy, max |diff| {max(dif):.4f} "
              f"({'OK' if max(dif) <= 0.002 else 'FLAG: > 0.002, device numerics or a stream mismatch'})")
    bad = void_checks(res, seeds)
    print(f"  void check (wiring: context-free swap ratio 1.00 +/- {WIRING_TOL}, DirOnlyRole non-movement swap 0): "
          + ("OK" if not bad else "VOID -- " + "; ".join(bad[:10])))
    if n < 8:
        print("\n(pilot: no verdicts below n=8)"); return
    for key, name in (("acc", "eval mode, REGISTERED"), ("acc_tm", "train mode, registered mode check")):
        A = verdict_a(res, seeds, key)
        print(f"\n== primary A ({name}) ==")
        for k, nm_ in (("N", "need: RoleTag - MapWM"), ("R", "recovery: HSR - CF2"), ("G", "gap: HSR - RoleTag")):
            d, p, f = A[k]
            print(f"  {nm_:24s} {d:+.4f}  perm p {p:.4f}  {'FIRES' if f else ''}")
        ci = perm2_ci(g("RoleTag", key), g("HSR", key), step=0.002)
        print(f"  HSR - RoleTag 95% CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}]; non-inferior at {NI_MARGIN}: HSR {A['NI_HSR']}, "
              f"MapWM {A['NI_MapWM']}; RoleTag SOLVED {A['RoleTag_solved']}/{n}")
        print(f"  {'REGISTERED A' if key == 'acc' else 'MODE CHECK A'}: {A['label']}"
              + (f"  [{'; '.join(A['qualifiers'])}]" if A["qualifiers"] else ""))
        if key == "acc":
            la = A["label"]
        elif A["label"] != la:
            print(f"  -> MODE-DEPENDENT: eval mode reads '{la}', train mode '{A['label']}'")
    B = verdict_b(res, seeds)
    print(f"\n== primary B (mechanism, HSR swap ratio at +15 tokens, seeds with a step: {B['learned']}/{n}) ==")
    print("  median ratio per class: " + "  ".join(f"{c} {B['median_ratio'][c]:.3f}" for c in NM_CLASSES))
    print(f"  REGISTERED B: {B['label']}")

    print("\n== declared secondaries (no verdict) ==")
    for x, y in (("MapWM", "HSR"), ("MapWM", "CF2"), ("DirOnlyRole", "RoleTag"), ("MapWM", "DirOnlyRole"),
                 ("RoPE1", "MapWM"), ("RoPE1", "RoPE2"), ("RoPE2", "HSR")):
        d, p, _ = contrast(g(x, "acc"), g(y, "acc"), 0)
        print(f"  {y} - {x}: acc {d:+.4f} (perm p {p:.4f}); SOLVED {sol(y)} vs {sol(x)} (Fisher "
              f"{fisher_solved(sol(x), n, sol(y), n):.4f})")
    for arm in ("MapWM", "CF2", "HSR", "RoleTag", "DirOnlyRole"):
        print(f"  {arm:11s} nm drift {np.mean(g(arm, 'nm_drift_rad')):.3f} rad, core drift {np.mean(g(arm, 'core_drift_rad')):.3f}"
              f" rad, drift channels {np.mean(g(arm, 'drift')):.1f}/64, disp {np.mean(g(arm, 'disp')):.3f} rad, sharp "
              f"{np.mean(g(arm, 'sharp')):.3f}, reliance {np.mean(g(arm, 'reliance')):+.3f}, move change "
              f"{np.mean(g(arm, 'move_change')):.3f}")
    for c in NM_CLASSES:
        r15, r0 = g("HSR", f"ratio_{c}"), g("HSR", f"ratio0_{c}")
        print(f"  HSR {c:10s}: ratio at +15 {np.nanmedian([v if v is not None else np.nan for v in r15]):.3f}, at the word "
              f"{np.nanmedian([v if v is not None else np.nan for v in r0]):.3f} (gate-inside: both ~0; cancel-later: word ~1, +15 ~0)")
    far = [np.mean([res[f'HSR_s{s}'][f'ratio_{c}'] or 0 for c in FAR]) - np.mean([res[f'HSR_s{s}'][f'ratio_{c}'] or 0 for c in NEAR])
           for s in seeds]
    print(f"  HSR far - near ratio, paired by seed: {np.mean(far):+.3f}, sign-flip p {signflip_p(far)['p']:.4f}")
    for arm in ("MapWM", "CF2"):
        print(f"  {arm} contaminated-gap accuracy {np.mean(g(arm, 'acc_contam')):.4f} vs the contaminated-path oracle's 0.5182"
              " (above = the context-free step copes in part)")
    d, p, _ = contrast(g("RoleTag", "sharp"), g("MapWM", "sharp"), 0)
    print(f"  compromise step: sharp-channel share MapWM - RoleTag {d:+.3f} (perm p {p:.4f}); disp MapWM - RoleTag "
          f"{np.mean(g('MapWM', 'disp')) - np.mean(g('RoleTag', 'disp')):+.3f} rad")
    print(f"  HSR alpha: " + " ".join(f"{a:+.3f}" for a in g("HSR", "alpha")))
    x = sum((g(a, "tail") for a in ARMS), []); y = sum((g(a, "acc") for a in ARMS), [])
    print(f"  r(final loss, acc) over {len(x)} runs: {np.corrcoef(x, y)[0, 1]:+.3f}")
    sd = np.sqrt((np.var(g("MapWM", "acc"), ddof=1) + np.var(g("RoleTag", "acc"), ddof=1)) / 2)
    print(f"  MDE (exact t, pooled sd of MapWM / RoleTag, two-sample) {mde(sd, n) * np.sqrt(2):.4f}")


# ------------------------------------------------------------------ smoke test of every branch on synthetic data
def _syn(seeds, acc, solved=None, extra=None):
    res = {}
    for a in ARMS:
        for j, s in enumerate(seeds):
            v = acc[a] if np.isscalar(acc[a]) else acc[a][j]
            r = {"acc": float(v), "acc_tm": float(v), "cls": "SOLVED" if (solved or {}).get(a, v >= 0.95) else "STALLED"}
            # (synthetic: SOLVED follows accuracy >= 0.95 unless overridden; real runs use the loss rule)
            r.update((extra or {}).get(a, {}))
            res[f"{a}_s{s}"] = r
    return res


def smoke():
    S = list(range(8)); rng = np.random.default_rng(0)
    hi = lambda m=0.985: np.clip(m + 0.01 * rng.standard_normal(8), 0, 1)
    lo = lambda m=0.70: np.clip(m + 0.03 * rng.standard_normal(8), 0, 1)
    base = {"MapWM": lo(), "RoleTag": hi(), "DirOnlyRole": hi(0.97), "HSR": hi(), "CF2": lo(), "RoPE1": 0.51,
            "RoPE2": 0.77}
    cases = {
        "RECOVERS IT (non-inferior": dict(base),
        "RECOVERS PART OF IT": dict(base, HSR=hi(0.90)),
        "EXCEEDS THE ORACLE": (dict(base, RoleTag=hi(0.93), HSR=hi(0.99)), {"RoleTag": True}),
        "gap to the oracle unmeasured": dict(base, HSR=np.array([0.99, 0.99, 0.99, 0.99, 0.80, 0.99, 0.99, 0.80])),
        "BELOW ITS CONTEXT-FREE TWIN": dict(base, HSR=lo(0.60), CF2=lo(0.72)),
        "THE CONTEXT STEP FAILS TOO": dict(base, HSR=lo(0.70)),
        "ORACLE BELOW CONTEXT-FREE": dict(base, MapWM=hi(0.99), RoleTag=np.array([0.96] * 6 + [0.955, 0.955])),
        "CONTEXT-FREE COPES": dict(base, MapWM=hi(0.985)),
        "UNMEASURED": dict(base, MapWM=np.array([0.99, 0.70, 0.99, 0.99, 0.70, 0.99, 0.99, 0.99])),
        "UNINTERPRETABLE": dict(base, RoleTag=lo(0.80)),
    }
    ok = True
    for want, acc in cases.items():
        acc, sv = acc if isinstance(acc, tuple) else (acc, None)
        lab = verdict_a(_syn(S, acc, sv), S)["label"]; hit = want in lab; ok &= hit
        print(f"  A  want '{want}': got '{lab}'  {'OK' if hit else 'MISMATCH'}")
    q = verdict_a(_syn(S, dict(base, MapWM=0.515)), S)["qualifiers"]
    print(f"  A  qualifier floor: {q}  {'OK' if 'MapWM AT THE INDEX FLOOR' in q else 'MISMATCH'}"); ok &= 'MapWM AT THE INDEX FLOOR' in q
    q = verdict_a(_syn(S, dict(base, MapWM=0.995, RoleTag=0.996, HSR=0.995)), S)["qualifiers"]
    hit = any(x.startswith("CEILING") for x in q); ok &= hit
    print(f"  A  qualifier ceiling: {q}  {'OK' if hit else 'MISMATCH'}")

    def bres(ratios, move=0.2, n_learned=8):
        ex = {"HSR": {}}
        res = _syn(S, base)
        for j, s in enumerate(S):
            r = res[f"HSR_s{s}"]; r["move_change"] = move if j < n_learned else 0.001
            for c in NM_CLASSES:
                r[f"ratio_{c}"] = ratios[c]
        return res
    allr = lambda v: {c: v for c in NM_CLASSES}
    bcases = {"AT EVERY CUE DISTANCE": allr(0.05), "NEAR-CUE USES ONLY": dict(allr(0.05), lead_far=0.9, trail_far=0.8),
              "FAR-CUE USES ONLY": dict(allr(0.9), lead_far=0.05, trail_far=0.05),
              "SOME CLASSES": dict(allr(0.05), trail_far=0.9), "TREATS NON-MOVEMENT USES AS MOVES": allr(0.95)}
    for want, rt in bcases.items():
        lab = verdict_b(bres(rt), S)["label"]; hit = want in lab; ok &= hit
        print(f"  B  want '{want}': got '{lab}'  {'OK' if hit else 'MISMATCH'}")
    lab = verdict_b(bres(allr(0.05), n_learned=5), S)["label"]; hit = "DID NOT RELIABLY LEARN" in lab; ok &= hit
    print(f"  B  want 'DID NOT RELIABLY LEARN': got '{lab}'  {'OK' if hit else 'MISMATCH'}")
    vres = _syn(S, base, extra={"MapWM": {f"ratio_{c}": 1.0 for c in NM_CLASSES}, "CF2": {f"ratio_{c}": 1.0 for c in NM_CLASSES},
                                "DirOnlyRole": {f"{c}@15": 0.0 for c in NM_CLASSES}})
    v1 = void_checks(vres, S); vres["CF2_s3"]["ratio_nat"] = 0.97; v2 = void_checks(vres, S)
    hit = (not v1) and len(v2) == 1; ok &= hit
    print(f"  void check: clean -> {v1}; one CF2 ratio 0.97 -> {v2}  {'OK' if hit else 'MISMATCH'}")
    print(f"SMOKE {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs-dir", default=f"{REPO}/runs/tw_ambig/p0")
    ap.add_argument("--seeds", default=None, help="comma list; default = the registered seeds")
    ap.add_argument("--out", default=f"{REPO}/TW_AMBIG.json")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.smoke:
        raise SystemExit(0 if smoke() else 1)
    seeds = [int(s) for s in (a.seeds or ",".join(map(str, SEEDS))).split(",")]
    res = collect(a.runs_dir, seeds, a.readouts, a.out, a.device)
    report(res, seeds)


SEEDS = list(range(40, 48))

if __name__ == "__main__":
    main()

"""Readouts and registered verdicts for TW_STATECHANGE_PREREG.md.

--readouts: per-run readouts (tw_statechange_readouts.readouts, CPU) for all 32 runs and the floors on the same eval
stream, into TW_STATECHANGE.json; then the verdicts. Without it, read TW_STATECHANGE.json. --runs-dir / --seeds / --out
for the pilot. `analyse(res, fl, seeds)` is pure (smoke-tested on synthetic inputs: tw_statechange_smoke.py).
"""
import argparse
import json
import math

import numpy as np

from mapformer.stats_core import classify_run, perm2_p, fisher_solved, mde, signflip_p

REPO = "/home/prashr/mapformer"
ARMS = ["MapWM", "NormStep", "DirOnly", "RoPE"]
SEEDS = [50, 51, 52, 53, 54, 55, 56, 57]
EPOCHS = 900
ACC_MIN = 0.02          # accuracy contrasts: |d| >= 0.02 with perm p < .05 (TW_NORMSTEP's floor)
CEIL = 0.999            # CEILING: both arms >= 0.999 on every seed
SHIFT_OFF = 0.10        # moves per state clause (in-plane): OFF THE MAP if <= (aside calibration, see prereg)
OFF_K = 6               # seeds out of 8 for an arm-level OFF / ON geometry label
L_MIN = 0.01            # functional readout L_sc materiality (accuracy on T1s; declared secondary)
C_MARGIN = 0.10         # C: T2drop accuracy must exceed the floor by >= 0.10 (mean over seeds, sign-flip p < .05)
REG_V, ALT_V = "intact_rs", "intact"   # registered accuracy: re-scored (x 1/(1-p) at eval); eval mode -> FLAG lines


# ------------------------------------------------------------------------------------------------ state functions
def contrast_state(x, y, sx, sy, n, acc_min=ACC_MIN):
    """y vs x (lists of per-seed accuracies; sx / sy SOLVED counts). Accuracy decides BETTER / WORSE (perm p < .05 and
    |d| >= acc_min); a Fisher-only firing is a convergence statement (TW_NORMSTEP Amendment 1, item 3)."""
    d = float(np.mean(y) - np.mean(x)); p = perm2_p(x, y)["p"]; pf = fisher_solved(sx, n, sy, n)
    fa = p < 0.05 and abs(d) >= acc_min; ff = pf < 0.05
    sd = math.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2) if n > 1 else 0.0
    m_ = max(mde(sd, n) * math.sqrt(2), acc_min) if n > 1 else float("nan")
    info = f"d {d:+.4f}, perm p {p:.4f}; SOLVED {sy}/{n} vs {sx}/{n}, Fisher p {pf:.4f}"
    if fa and ff and (d > 0) != (sy > sx):
        return "CONFLICT", info
    if fa:
        return ("BETTER" if d > 0 else "WORSE"), info
    if ff:
        return f"SOLVED RATE {'HIGHER' if sy > sx else 'LOWER'} (convergence; accuracy unmeasured below {m_:.4f})", info
    if min(x) >= CEIL and min(y) >= CEIL:
        return "CEILING", info
    return f"NO DIFFERENCE (unmeasured below {m_:.4f})", info


def geom_label(shifts, thr=SHIFT_OFF, k=OFF_K):
    n_off = sum(s <= thr for s in shifts); n = len(shifts)
    if n_off >= k * n / 8:
        return "OFF", n_off
    if n - n_off >= k * n / 8:
        return "ON", n_off
    return "MIXED", n_off


def func_label(L, l_min=L_MIN):
    L = np.asarray(L, float); m = float(L.mean()); p = signflip_p(L)["p"]
    sd = float(L.std(ddof=1)) if L.size > 1 else 0.0
    md = max(mde(sd, L.size), l_min) if L.size > 1 else float("nan")
    if p < 0.05 and m >= l_min:
        return "COSTS", m, p, md
    if p < 0.05 and m <= -l_min:
        return "USED", m, p, md
    return "NO COST", m, p, md


def aside_label(sh, sa):
    """Paired by seed: state-clause shift minus aside shift in the same model (sign-flip, two-sided)."""
    dd = np.asarray(sh, float) - np.asarray(sa, float); p = signflip_p(dd)["p"]; m = float(dd.mean())
    if p < 0.05:
        return ("LARGER THAN ASIDES" if m > 0 else "SMALLER THAN ASIDES"), m, p
    return "NOT DISTINGUISHED FROM ASIDES", m, p


B_HEAD = {"OFF": "STATE VERBS STAY OFF THE MAP", "ON": "STATE VERBS MOVE THE MAP"}


def b_verdict(g, n_off, n, al, c_label):
    head = B_HEAD.get(g) or f"MIXED: STATE VERBS OFF THE MAP ON {n_off}/{n} SEEDS"
    head += f" ({al.lower()})"
    if c_label.startswith("STATE NOT SHOWN"):
        head += " [state not shown used by this arm (C): the clauses are not shown task-relevant to it]"
    return head


def c_label(acc_t2drop, F1, F2, margin=C_MARGIN):
    """F1: T2drop accuracy of the best location-free rule chosen on ALL revisit targets (the rule a model without path
    integration would apply); F2: the best location-free rule on T2drop alone (an upper bound no uniform rule reaches)."""
    a = np.asarray(acc_t2drop, float)
    out = {}
    for tag, F in (("F2", F2), ("F1", F1)):
        d = a - F; out[tag] = (float(d.mean()), signflip_p(d)["p"])
    if out["F2"][1] < 0.05 and out["F2"][0] >= margin:
        lab = "STATE BOUND TO PLACE"
    elif out["F1"][1] < 0.05 and out["F1"][0] >= margin:
        lab = "STATE USED, NOT SHOWN BEYOND THE LAST-DROP RULE"
    else:
        lab = "STATE NOT SHOWN USED"
    return lab, out


def a_headline(state):
    if state == "BETTER":
        return "PATH NEEDED FOR LOCATION: path integration beats index RoPE on unchanged-cell revisits"
    if state == "WORSE":
        return "INDEX ABOVE PATH ON LOCATION"
    return f"LOCATION CONTRAST {state}"


# ------------------------------------------------------------------------------------------------ analysis
def analyse(res, fl, seeds=SEEDS, epochs=EPOCHS, out=print):
    """res: {f'{arm}_s{seed}': per-run readouts + cls/tail/epochs/acc2048}; fl: floors dict. Returns verdicts."""
    n = len(seeds); V = {}
    missing = [f"{a}_s{s}" for s in seeds for a in ARMS if f"{a}_s{s}" not in res]
    short = [k for k, r in res.items() if r.get("epochs", epochs) != epochs]
    if missing or short:
        out(f"VOID: missing {missing}, not {epochs} epochs {short}"); return {"VOID": True}
    g = lambda arm, f: [f(res[f"{arm}_s{s}"]) for s in seeds]
    acc = lambda arm, k, v=REG_V: g(arm, lambda r: r["acc"][v][k])
    sol = lambda arm: g(arm, lambda r: r["cls"]).count("SOLVED")
    # void: DirOnly's state clauses exactly off the map
    for s in seeds:
        r = res[f"DirOnly_s{s}"]
        assert r["geom"]["shift_sc"] == 0.0 and r["geom"]["R_sc"] == 0.0, ("VOID: DirOnly state-clause step", s)
        assert r["L_sc"] == 0.0 and r["L_sc0"] == 0.0, ("VOID: DirOnly cancellation changed accuracy", s)
    out("void check: DirOnly state clauses have exactly zero net step and the cancellations are no-ops -> OK")

    from mapformer.tw_statechange_readouts import nonpath_floor, NONPATH_RULES
    best_all = max((r for r in NONPATH_RULES if r in fl), key=lambda r: fl[r]["all"])
    F1 = fl[best_all]["T2drop"]; F2, r2 = nonpath_floor(fl, "T2drop")
    out(f"\nfloors (eval stream): constant all {fl['constant']['all']:.4f} T1 {fl['constant']['T1']:.4f}; "
        f"reversal-copy T1 {fl['revcopy']['T1']:.4f}; best uniform location-free rule '{best_all}' (all "
        f"{fl[best_all]['all']:.4f}) -> T2drop F1 = {F1:.4f}; best rule on T2drop alone '{r2}' F2 = {F2:.4f}; "
        f"stale map (last saw) T2drop {fl['last_saw']['T2drop']:.4f}")

    out(f"\n== per arm (n={n}), re-scored (registered) [eval mode]: all | T1 | T1s | T2take | T2drop | T3 | SOLVED | T=2048 ==")
    for arm in ARMS:
        row = "  ".join(f"{np.mean(acc(arm, k)):.4f} [{np.mean(acc(arm, k, ALT_V)):.4f}]"
                        for k in ("all", "T1", "T1s", "T2take", "T2drop", "T3"))
        out(f"  {arm:9s} {row}  {sol(arm)}/{n}  [{np.mean(g(arm, lambda r: r['acc2048'])):.4f}]")
    for arm in ARMS[:3]:
        G = lambda f: np.mean(g(arm, f))
        out(f"  {arm:9s} shift/clause (moves, in-plane): state {G(lambda r: r['geom']['shift_sc']):.4f} (take "
            f"{G(lambda r: r['geom']['shift_take']):.4f}, drop {G(lambda r: r['geom']['shift_drop']):.4f}), aside "
            f"{G(lambda r: r['geom']['shift_aside']):.4f}; R state {G(lambda r: r['geom']['R_sc']):.3f} aside "
            f"{G(lambda r: r['geom']['R_aside']):.3f}; L_sc {G(lambda r: r['L_sc']):+.4f} [rs {G(lambda r: r['L_sc_rs']):+.4f}]"
            f"; drift state {G(lambda r: r['drift_sc']):.4f} aside {G(lambda r: r['drift_aside']):.4f} rad")

    flags = []
    def rescore_flag(name, lab_reg, lab_alt):
        if lab_reg.split(" (")[0] != lab_alt.split(" (")[0]:
            flags.append(name)
            return f" [FLAG: in eval mode without the dropout-scale correction {name} reads {lab_alt}; registered verdict unchanged]"
        return ""

    # ---- A
    out("\n== A LOCATION: MapWM - RoPE on T1 (unchanged-cell revisits), re-scored accuracy ==")
    sA, iA = contrast_state(acc("RoPE", "T1"), acc("MapWM", "T1"), sol("RoPE"), sol("MapWM"), n)
    sAr, _ = contrast_state(acc("RoPE", "T1", ALT_V), acc("MapWM", "T1", ALT_V), sol("RoPE"), sol("MapWM"), n)
    flo = sum(a <= fl["constant"]["T1"] + 0.01 for a in acc("RoPE", "T1"))
    out(f"  {iA}; RoPE within 0.01 of the constant floor on T1: {flo}/{n}")
    V["A"] = a_headline(sA) + rescore_flag("A", sA, sAr)
    out(f"  REGISTERED A: {V['A']}")

    # ---- C (before B: B carries C's qualifier)
    out(f"\n== C STATE BOUND TO PLACE: T2drop accuracy vs F2 = {F2:.4f} and F1 = {F1:.4f} (margin {C_MARGIN}, sign-flip) ==")
    Cl = {}
    for arm in ("MapWM", "NormStep"):
        lab, o = c_label(acc(arm, "T2drop"), F1, F2); labr, _ = c_label(acc(arm, "T2drop", ALT_V), F1, F2)
        Cl[arm] = lab
        out(f"  {arm}: T2drop {' '.join(f'{a:.3f}' for a in acc(arm, 'T2drop'))}; - F2 {o['F2'][0]:+.4f} (p {o['F2'][1]:.4f}),"
            f" - F1 {o['F1'][0]:+.4f} (p {o['F1'][1]:.4f})")
        V[f"C_{arm}"] = lab + rescore_flag(f"C {arm}", lab, labr)
        out(f"  REGISTERED C ({arm}): {V[f'C_{arm}']}")

    # ---- B
    out(f"\n== B STATE VERBS OFF THE MAP: field shift per state clause (the clause's net phase displacement projected on "
        f"the plane of the two axis half-steps, in moves; OFF <= {SHIFT_OFF} on >= {OFF_K}/8 seeds, ON > {SHIFT_OFF} on >= {OFF_K}/8), "
        f"and the same model's aside shift (paired sign-flip) ==")
    for arm in ("MapWM", "NormStep"):
        sh = g(arm, lambda r: r["geom"]["shift_sc"]); sa = g(arm, lambda r: r["geom"]["shift_aside"])
        gl, n_off = geom_label(sh); al, m_, p_ = aside_label(sh, sa)
        out(f"  {arm}: state shift {' '.join(f'{x:.3f}' for x in sh)} -> {gl} ({n_off}/{n} <= {SHIFT_OFF}); aside shift "
            f"{' '.join(f'{x:.3f}' for x in sa)}; state - aside {m_:+.4f} moves, sign-flip p {p_:.4f} -> {al}")
        V[f"B_{arm}"] = b_verdict(gl, n_off, n, al, Cl[arm])
        out(f"  REGISTERED B ({arm}): {V[f'B_{arm}']}")

    # ---- D
    out("\n== D STEP FORM: accuracy on all revisit targets, re-scored ==")
    for tag, y in (("D1 NormStep - MapWM", "NormStep"), ("D2 DirOnly - MapWM", "DirOnly")):
        s_, i_ = contrast_state(acc("MapWM", "all"), acc(y, "all"), sol("MapWM"), sol(y), n)
        sr, _ = contrast_state(acc("MapWM", "all", ALT_V), acc(y, "all", ALT_V), sol("MapWM"), sol(y), n)
        V[tag[:2]] = s_ + rescore_flag(tag[:2], s_, sr)
        out(f"  {tag}: {i_}\n  REGISTERED {tag[:2]}: {V[tag[:2]]}")

    # ---- the question, from A, B(MapWM), C(MapWM) -- no new test
    holds = (V["A"].startswith("PATH NEEDED") and V["B_MapWM"].startswith("STATE VERBS STAY OFF THE MAP")
             and not Cl["MapWM"].startswith("STATE NOT SHOWN"))
    V["SUMMARY"] = ("PREDICTION HOLDS (location needs path integration; learned steps keep state verbs off the map while "
                    "state is used)" if holds else "PREDICTION DOES NOT HOLD AS STATED (see A, B, C)")
    out(f"\n  SUMMARY (computed, no new test): {V['SUMMARY']}")
    if flags:
        out(f"  re-score flags on: {', '.join(flags)}")

    # ---- declared secondaries (no verdict)
    out("\n== secondaries (no verdict) ==")
    for arm in ("MapWM", "NormStep"):
        L = g(arm, lambda r: r["L_sc"]); Lr = g(arm, lambda r: r["L_sc_rs"]); La = g(arm, lambda r: r["L_aside"])
        f_, m_, p_, md = func_label(L); fr = func_label(Lr)
        out(f"  functional (net cancellation) {arm}: L_sc {' '.join(f'{x:+.4f}' for x in L)} mean {m_:+.4f} sign-flip p "
            f"{p_:.4f} -> {f_} (unmeasured below {md:.4f}); re-scored {fr[1]:+.4f} -> {fr[0]}; aside analogue L_aside "
            f"{np.mean(La):+.4f} (calibration, stored runs at >= 0.99: -0.001..-0.012)")
    for x, y in (("MapWM", "DirOnly"), ("MapWM", "NormStep")):
        for k in ("T1", "T1clean", "T2take", "T2drop", "T3"):
            d = np.mean(acc(y, k)) - np.mean(acc(x, k)); p = perm2_p(acc(x, k), acc(y, k))["p"]
            dr = np.mean(acc(y, k, ALT_V)) - np.mean(acc(x, k, ALT_V)); pr = perm2_p(acc(x, k, ALT_V), acc(y, k, ALT_V))["p"]
            out(f"  {y} - {x} on {k:7s}: re-scored {d:+.4f} (p {p:.4f}); eval mode {dr:+.4f} (p {pr:.4f})")
    for arm in ARMS[:3]:
        G = lambda f: g(arm, f)
        out(f"  {arm}: reliance {np.mean(G(lambda r: r['reliance'])):.3f}; clock channels "
            f"{' '.join(str(x) for x in G(lambda r: r['clock_channels']))}; carry_cos "
            f"{' '.join(f'{x:+.2f}' for x in G(lambda r: r['geom'].get('carry_cos', float('nan'))))}; L_sc0 (zeroed) "
            f"{np.mean(G(lambda r: r['L_sc0'])):+.4f}; L_sc on all {np.mean(G(lambda r: r['L_sc_all'])):+.4f}; "
            f"L_aside {np.mean(G(lambda r: r['L_aside'])):+.4f}; drift optional words {np.mean(G(lambda r: r['drift_optword'])):.4f}")
        cs = {c: np.mean([res[f'{arm}_s{s}']['geom']['class_step'][c] for s in seeds])
              for c in res[f"{arm}_s{seeds[0]}"]["geom"]["class_step"]}
        out(f"     step / direction half-step by class: " + ", ".join(f"{c} {v:.3f}" for c, v in cs.items()))
    x = sum((g(a, lambda r: r["tail"]) for a in ARMS), []); y = sum((acc(a, "all") for a in ARMS), [])
    out(f"  r(final-5% loss, acc) over {len(x)} runs: {np.corrcoef(x, y)[0, 1]:+.3f}")
    return V


# ------------------------------------------------------------------------------------------------ collection
def collect(rdir, seeds, out_json):
    import torch
    from mapformer import tw_statechange_readouts as R
    torch.set_num_threads(8)
    res = {}; W = Wd = None; fl = None
    for s in seeds:
        for a in ARMS:
            d = f"{rdir}/{a}_s{s}"; ev = json.load(open(f"{d}/eval.json"))
            ck = torch.load(f"{d}/{a}.pt", map_location="cpu", weights_only=False)
            if W is None:
                _m, _a, cfg = R.load(f"{d}/{a}.pt"); env = R.eval_env(cfg); W = R.walks(env); Wd = W[:40]
                fl = R.floors(env, W, ngram_env=R.make_env(0, cfg["size"], cfg["p_take"], cfg["p_drop"], cfg["state_vocab"]))
                assert all(R.replay_check(w, env) for w in W), "VOID: eval-stream bookkeeping"
            c = classify_run(ck["losses"])
            r = R.readouts(f"{d}/{a}.pt", W=W, Wd=Wd)
            r.update(cls=c["registered"], tail=float(c["tail"]), epochs=len(ck["losses"]),
                     acc_evaljson=ev["eval"]["1024"]["acc"], acc2048=ev["eval"]["2048"]["acc"])
            if abs(r["acc"]["intact"]["all"] - r["acc_evaljson"]) > 0.002:
                print(f"  NOTE {a}_s{s}: CPU readout accuracy {r['acc']['intact']['all']:.4f} vs trainer eval.json "
                      f"{r['acc_evaljson']:.4f} (device difference > 0.002)")
            res[f"{a}_s{s}"] = r
            print(f"  {a:9s} s{s}: acc {r['acc']['intact']['all']:.4f} T1 {r['acc']['intact']['T1']:.4f} T2drop "
                  f"{r['acc']['intact']['T2drop']:.4f} {r['cls']}" +
                  (f"  shift_sc {r['geom']['shift_sc']:.4f} L_sc {r['L_sc']:+.4f}" if "geom" in r else ""), flush=True)
    json.dump({"runs": res, "floors": fl}, open(out_json, "w"), indent=1, default=float)
    return res, fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", action="store_true")
    ap.add_argument("--runs-dir", default=f"{REPO}/runs/tw_statechange/p0")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--out", default=f"{REPO}/TW_STATECHANGE.json")
    ap.add_argument("--verdicts", default=f"{REPO}/TW_STATECHANGE_VERDICTS.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.readouts:
        res, fl = collect(a.runs_dir, seeds, a.out)
    else:
        d = json.load(open(a.out)); res, fl = d["runs"], d["floors"]
    if len(seeds) < 8:
        print("\n(pilot: no verdicts below n=8; per-run readouts above)"); return
    V = analyse(res, fl, seeds)
    json.dump(V, open(a.verdicts, "w"), indent=1)


if __name__ == "__main__":
    main()

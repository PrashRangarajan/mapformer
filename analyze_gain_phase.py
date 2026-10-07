"""Registered readouts for GAIN_PHASE_PREREG.md: the gain-phase map on the new-object task (2x2 STEP x SCORE).

Arms (model_gain_phase): W = MapWM (raw step, rotary score), N = NormStep (NormStep step, rotary), G = GainRaw (raw
step, gain score), GP = GainPhase (NormStep step, gain score). Rank 4, T = 1024, 900 epochs, LEAK's recipe; seeds
8-15 (n = 8 per arm), one batch.

Inputs: GAIN_PHASE_EVAL.json (gain_phase_eval.evaluate_run per run: acc = unseen-object accuracy x1, L_ms, S_id, ...)
and each checkpoint's per-epoch losses (SOLVED = stats_core.classify_run, final-5% loss < 0.05; speed = first epoch
at which the 10-epoch running mean loss is < 0.05, censored at E + 1 if never).

Registered decisions (GAIN_PHASE_PREREG.md, all two-sided at .05 unless stated):
  D1  STEP under the gain score: GP vs G; its positive control D1r: N vs W (LEAK's +0.0107 on fresh seeds).
  D2  SCORE at each step: G vs W and GP vs N.
  D3  LEAK DISSOCIATION (separate defects?): L_ms and S_id per arm.
  D4  COMBINATION: GP vs W (two-sided) and GP vs N (non-inferiority at MARGIN, SOLVED slack 0).
  D5  SPEED: GP vs N and G vs W.
Amendment 1 (independent audit, before launch): D3's gain-side labels are read only if the gain arms pass a convergence
gate (converged_gate; else "D3 UNMEASURED"); NEGATIVE L_ms branch; TOLERATED carries the distortion share; budget-scope
qualifiers on D1 / D2 / D4 (budget_flag); D1r's firing test on the D1 line; combo WORSE x AS GOOD names NormStep's own
deficit; a registered accuracy label that flips under the dropout-scale re-score is flagged on its line.
Amendment 2 (re-audit): the D3 gate no longer depends on the leak (S_id and accuracy-vs-MapWM criteria removed): it
reads theta reliance and a training-loss criterion, names the failed criterion, and is re-checked on the re-scored
reliance; scope qualifiers count every non-SOLVED run (STALLED too); D4's WORSE x AS GOOD wording depends on D1r.
Amendment 3 (final audit): the gate's convergence criterion is the LEAK-FREE loss nll_ms (Amendment 2's training tail
contained the leak's cost); the GainRaw-vs-MapWM scope qualifier reads nll_ms too; D3 carries GainRaw's qualifier.
Every branch is exercised on synthetic data by docs/audits/2026-10-06/gain_phase_smoke.py.
`python3 -m mapformer.analyze_gain_phase`
"""
import json
import os

import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, perm2_ci, fisher_solved

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/gain_phase/p0"
W, N, G, GP = "MapWM", "NormStep", "GainRaw", "GainPhase"
ARMS = [W, N, G, GP]
SEEDS = list(range(8, 16))
EPOCHS = 900
MIN_D = 0.005        # accuracy firing floor: half of LEAK's registered leak cost (+0.0107)
MARGIN = 0.005       # non-inferiority margin, GP vs N (same reasoning)
SOLVED_SLACK = 0     # GAIN_GRAIN Amendment 1 (D1): AS GOOD needs SOLVED(GP) >= SOLVED(N)
CEIL = 0.999
LEAK_ABSENT = 0.002  # median L_ms at or below: no leak (LEAK: NormStep max +0.0004, ActOnly 0)
LEAK_PRESENT = 0.005 # median L_ms at or above: leak present (LEAK: MapWM min +0.0074, median +0.0098)
SPEED_RATIO = 1.25   # geometric-mean ratio of epochs-to-0.05 that counts as faster / slower
GATE_RELIANCE = 0.2     # D3 gate: median theta reliance >= 0.2 (untrained models |.| <= 0.0001; LEAK checkpoints 0.92-0.97, V6)
GATE_NLL = 0.06         # D3 gate (Amendment 3): median LEAK-FREE loss nll_ms <= 0.06 -- eval-mode NLL at every revisit target
                        # with object steps mean-substituted (the L_ms substitution), test-pool stream. Amendment 2's
                        # training-tail criterion was NOT leak-independent (the auditor: ~97% of MapWM's eval loss is its
                        # leak; a 1.7x identity step crosses 0.25). Validation (gain_phase_lossgate_validate_out.txt): LEAK
                        # MapWM 0.007-0.038, NormStep 0.0003-0.0012, ActOnly 0.0006-0.0017; identity step x1.5-x3 leaves
                        # nll_ms unchanged (0.0071) while eval nll goes 0.125 -> 0.959; untrained models ~7; 30-epoch pilot
                        # models 0.080-3.41. 0.06 sits between MapWM's worst converged run (0.038) and the best short-pilot
                        # model (NormStep s111, 0.080).
REGIME_X = 1.0          # scope qualifier GainRaw vs MapWM (Amendment 3): median nll_ms(GainRaw) above every MapWM run's
REGIME_FLOOR = 0.01     # ... and > 0.01 (no flag for differences inside converged runs' spread)
N_MC = 200_000       # Monte Carlo relabellings when C(2n, n) > 250k (n >= 12); the power script lowers it
PRINT_CI = True      # the power script turns the (slow) CI walk off; it never changes a label


# ------------------------------------------------------------------------------------------- readout states
def acc_state(x, y):
    """y - x, two-sided permutation. 'CEIL' if both arms >= CEIL on every seed (rule 5: could not have gone the other
    way); 'POS' / 'NEG' if p < .05 and |d| >= MIN_D; else 'NONE'. Returns (state, d, p)."""
    x, y = np.asarray(x, float), np.asarray(y, float); d = y.mean() - x.mean()
    p = perm2_p(x, y, n_mc=N_MC)["p"]
    if x.min() >= CEIL and y.min() >= CEIL:
        return "CEIL", d, p
    if p < 0.05 and abs(d) >= MIN_D:
        return ("POS" if d > 0 else "NEG"), d, p
    return "NONE", d, p


def fisher_state(sx, nx, sy, ny):
    p = fisher_solved(sx, nx, sy, ny)
    if p < 0.05:
        return ("POS" if sy / ny > sx / nx else "NEG"), p
    return "NONE", p


def ni_p(ref, x, margin=MARGIN):
    """One-sided permutation p for H0: mean(x) - mean(ref) <= -margin (equal n: symmetric, so two-sided / 2)."""
    ref, x = np.asarray(ref, float), np.asarray(x, float); d = x.mean() - ref.mean()
    p2 = perm2_p(ref, x, shift=-margin, n_mc=N_MC)["p"]
    return p2 / 2 if d > -margin else 1 - p2 / 2


def contrast_state(x_acc, x_sol, y_acc, y_sol):
    """y vs x, two-sided on accuracy and SOLVED. Order: CONFLICT, BETTER, WORSE, CEILING, NO DIFFERENCE.
    CEILING only if both arms >= CEIL on every seed AND SOLVED does not fire. Returns (label, detail)."""
    nx, ny = len(x_acc), len(y_acc)
    a, d, pa = acc_state(x_acc, y_acc); f, pf = fisher_state(sum(x_sol), nx, sum(y_sol), ny)
    det = (f"d {d:+.4f} (perm p {pa:.4f}, {a}); SOLVED {sum(y_sol)}/{ny} vs {sum(x_sol)}/{nx} (Fisher p {pf:.4f}, {f})")
    if (a == "POS" and f == "NEG") or (a == "NEG" and f == "POS"):
        return "CONFLICT", det
    if a == "POS" or f == "POS":
        return "BETTER", det + " -- fires on " + " and ".join(w for w, s in (("accuracy", a), ("SOLVED", f)) if s == "POS")
    if a == "NEG" or f == "NEG":
        return "WORSE", det + " -- fires on " + " and ".join(w for w, s in (("accuracy", a), ("SOLVED", f)) if s == "NEG")
    if a == "CEIL":
        return "CEILING", det + " -- both arms >= 0.999 on every seed: no headroom"
    if not PRINT_CI:
        return "NO DIFFERENCE", det
    ci = perm2_ci(x_acc, y_acc, level=0.95, step=0.0005)
    return "NO DIFFERENCE", det + f" -- 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}], MDE {mde2(x_acc, y_acc):.4f}"


def mde2(x, y):
    from scipy import stats
    nx, ny = len(x), len(y); df = nx + ny - 2
    sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / df)
    return float((stats.t.ppf(0.975, df) + stats.t.ppf(0.8, df)) * sd * np.sqrt(1 / nx + 1 / ny))


def ni_state(ref_acc, ref_sol, x_acc, x_sol):
    """GP (x) against N (ref): WORSE (two-sided firing), BETTER, AS GOOD (non-inferior at MARGIN and SOLVED(x) >=
    SOLVED(ref) - SOLVED_SLACK), else UNDETERMINED; CONFLICT first."""
    lab, det = contrast_state(ref_acc, ref_sol, x_acc, x_sol)
    pn = ni_p(ref_acc, x_acc); det += f"; non-inferiority at -{MARGIN}: one-sided p {pn:.4f}"
    if lab in ("CONFLICT", "WORSE", "BETTER"):
        return lab, det
    if pn < 0.05 and sum(x_sol) >= sum(ref_sol) - SOLVED_SLACK:
        return "AS GOOD", det + (" (at ceiling)" if lab == "CEILING" else "")
    why = [w for w, c in ((f"accuracy not shown within {MARGIN}", pn >= 0.05),
                          (f"SOLVED {sum(x_sol)} below the reference's {sum(ref_sol)}", sum(x_sol) < sum(ref_sol) - SOLVED_SLACK)) if c]
    return "UNDETERMINED", det + " -- " + "; ".join(why)


def speed_epoch(losses, thr=0.05, win=10):
    r = np.convolve(np.asarray(losses, float), np.ones(win) / win, mode="valid"); i = np.nonzero(r < thr)[0]
    return int(i[0]) + win if len(i) else len(losses) + 1


def speed_state(x_ep, y_ep, E=EPOCHS):
    """y vs x on log(epochs to a 10-epoch running loss < 0.05), censored runs at E + 1. Two-sided permutation;
    FASTER / SLOWER if p < .05 and the geometric-mean ratio is beyond SPEED_RATIO; NEITHER CONVERGED if every run of
    both arms is censored; else NO DIFFERENCE."""
    x, y = np.log(np.asarray(x_ep, float)), np.log(np.asarray(y_ep, float))
    cx, cy = int((np.asarray(x_ep) > E).sum()), int((np.asarray(y_ep) > E).sum())
    ratio = float(np.exp(x.mean() - y.mean()))                       # > 1: y faster
    det = (f"geometric-mean epochs {np.exp(y.mean()):.0f} vs {np.exp(x.mean()):.0f} (x{ratio:.2f} faster); "
           f"censored {cy}/{len(y)} vs {cx}/{len(x)}")
    if cx == len(x) and cy == len(y):
        return "NEITHER CONVERGED", det
    p = perm2_p(x, y, n_mc=N_MC)["p"]; det += f"; perm p {p:.4f}"
    if p < 0.05 and ratio >= SPEED_RATIO:
        return "FASTER", det
    if p < 0.05 and ratio <= 1 / SPEED_RATIO:
        return "SLOWER", det
    return "NO DIFFERENCE", det


# ------------------------------------------------------------------------------------------- verdicts
def step_verdict(rep, d1):
    """(D1r: N vs W label, D1: GP vs G label) -> headline."""
    if rep != "BETTER":
        return (f"REPLICATION FAILED: NormStep vs MapWM is {rep} on these seeds (LEAK: BETTER), so the batch cannot "
                f"show whether the step effect carries over; under the gain score NormStep is {d1} (reported as it falls)")
    return {"BETTER": "NORMSTEP HELPS UNDER BOTH SCORES (the step fix does not depend on the score)",
            "CEILING": ("NORMSTEP HELPS UNDER THE ROTARY SCORE; UNDER THE GAIN SCORE THERE IS NO HEADROOM (GainRaw already "
                        "at ceiling: the gain score alone avoids the leak's accuracy cost; D3 says why)"),
            "NO DIFFERENCE": ("NORMSTEP HELPS UNDER THE ROTARY SCORE ONLY: no step effect detected under the gain score "
                              "(see its CI and MDE; D3 says whether GainRaw leaks)"),
            "WORSE": "NORMSTEP HELPS UNDER THE ROTARY SCORE BUT HURTS UNDER THE GAIN SCORE (interference)",
            "CONFLICT": "NORMSTEP HELPS UNDER THE ROTARY SCORE; UNDER THE GAIN SCORE accuracy and SOLVED disagree"}[d1]


def leak_side(raw_ms, norm_ms, raw_sid, norm_sid):
    """Does the step fix remove the leak at one score? NEGATIVE: either median L_ms < -LEAK_ABSENT (the identity step is
    used; Amendment 1). NO LEAK TO REMOVE: |median L_ms(raw)| <= LEAK_ABSENT. REMOVES: |median L_ms(norm)| <= LEAK_ABSENT
    and S_id(norm) < S_id(raw) (two-sided perm p < .05). Else INCOMPLETE."""
    p = perm2_p(raw_sid, norm_sid, n_mc=N_MC)["p"]; lower = np.mean(norm_sid) < np.mean(raw_sid)
    det = (f"median L_ms raw {np.median(raw_ms):+.4f} / NormStep {np.median(norm_ms):+.4f}; S_id raw "
           f"{np.mean(raw_sid):.4f} / NormStep {np.mean(norm_sid):.4f} (perm p {p:.4f})")
    if np.median(raw_ms) < -LEAK_ABSENT or np.median(norm_ms) < -LEAK_ABSENT:
        return "NEGATIVE", det
    if abs(np.median(raw_ms)) <= LEAK_ABSENT:
        return "NO LEAK TO REMOVE", det
    if abs(np.median(norm_ms)) <= LEAK_ABSENT and p < 0.05 and lower:
        return "REMOVES", det
    return "INCOMPLETE", det


def gain_leak(g_ms, w_sid, g_sid, g_resid=None, g_shift=None, w_shift=None):
    """The leak under the gain score with the raw step (GainRaw). PERSISTS: median L_ms(G) >= LEAK_PRESENT. NEGATIVE:
    median < -LEAK_ABSENT (Amendment 1). ABSENT: |median| <= LEAK_ABSENT, split by S_id(G) vs S_id(W): UNLEARNED if lower
    (perm p < .05), TOLERATED otherwise. PARTIAL in between. The detail carries GainRaw's distortion share and field shift
    (Amendment 1): a TOLERATED spread with distortion share >= 0.5 is mostly isotropic (cheap by V5), < 0.5 mostly a field
    shift the gain score ignores."""
    p = perm2_p(w_sid, g_sid, n_mc=N_MC)["p"]; med = float(np.median(g_ms))
    det = f"median L_ms(GainRaw) {med:+.4f}; S_id GainRaw {np.mean(g_sid):.4f} vs MapWM {np.mean(w_sid):.4f} (perm p {p:.4f})"
    if g_resid is not None:
        det += (f"; GainRaw distortion share {np.median(g_resid):.2f}, field shift {np.median(g_shift):.4f} cells "
                f"(MapWM {np.median(w_shift):.4f})")
    if med >= LEAK_PRESENT:
        return "PERSISTS", det
    if med < -LEAK_ABSENT:
        return "NEGATIVE", det
    if abs(med) <= LEAK_ABSENT:
        return ("ABSENT, UNLEARNED" if (p < 0.05 and np.mean(g_sid) < np.mean(w_sid)) else "ABSENT, TOLERATED"), det
    return "PARTIAL", det


def tolerated_note(g_resid, g_shift, w_shift):
    kind = "mostly isotropic spread" if np.median(g_resid) >= 0.5 else "mostly a field shift the gain score ignores"
    return (f"{kind}: GainRaw distortion share {np.median(g_resid):.2f}, field shift {np.median(g_shift):.4f} cells vs "
            f"MapWM {np.median(w_shift):.4f}")


def class_counts(cls):
    return (sum(c["cls"] == "SOLVED" for c in cls), sum(c["registered"] == "STALLED" for c in cls),
            sum(c["registered"] == "DESCENDING" for c in cls))


def converged_gate(a_rel, a_nll_ms, a_cls):
    """Amendment 3: a gain arm's D3 labels are read only if its LEAK-FREE loss is converged (median nll_ms <= GATE_NLL)
    and it USES THETA (median theta reliance >= GATE_RELIANCE). Neither criterion reads the leak: nll_ms removes the
    identity step's cost by the same substitution as L_ms (no S_id, no accuracy relative to MapWM, no training tail).
    Returns (ok, reason, detail); reason names the failed criterion ('' if ok)."""
    nll = float(np.median(a_nll_ms)); rel = float(np.median(a_rel))
    loss_ok, rel_ok = nll <= GATE_NLL, rel >= GATE_RELIANCE
    n = len(a_cls); k, st, d = class_counts(a_cls)
    tail = float(np.median([c["tail"] for c in a_cls]))
    det = (f"median leak-free loss nll_ms {nll:.4f} (<= {GATE_NLL}: {'ok' if loss_ok else 'FAILS'}), median theta reliance "
           f"{rel:.3f} (>= {GATE_RELIANCE}: {'ok' if rel_ok else 'FAILS'}); training tail {tail:.4f}; {k}/{n} SOLVED, "
           f"{st}/{n} STALLED, {d}/{n} DESCENDING")
    fl = [w for w, ok_ in ((f"leak-free loss above {GATE_NLL} (not converged)", loss_ok),
                           (f"theta reliance below {GATE_RELIANCE} (does not use theta)", rel_ok)) if not ok_]
    return bool(loss_ok and rel_ok), " and ".join(fl), det


def leak_verdict(rot, gain, gl, tol_note=""):
    """(step fix at rotary, step fix at gain, GainRaw's leak) -> D3 headline (gain-side labels only after the
    convergence gate; see analyse)."""
    if rot == "NO LEAK TO REMOVE":
        return "NO LEAK TO DISSOCIATE: MapWM's in-distribution leak is <= 0.002 on these seeds (LEAK: +0.0098); D3 void"
    if "NEGATIVE" in (rot, gain, gl):
        return (f"NEGATIVE L_ms (removing the identity step COSTS accuracy: the identity step is used) -- rotary: {rot}; "
                f"gain: {gain}; GainRaw: {gl} -- reported as it falls")
    step_ok = rot == "REMOVES" and gain in ("REMOVES", "NO LEAK TO REMOVE")
    if step_ok and gl == "PERSISTS":
        return ("SEPARATE DEFECTS: the step fix removes the leak under both scores; the score fix leaves it (it is in "
                "theta)")
    if step_ok and gl == "ABSENT, TOLERATED":
        return ("THE GAIN SCORE TOLERATES THE LEAK: the identity step is still in theta (S_id not lower than MapWM's) but "
                "costs no accuracy under the gain score; the step fix removes it under both scores"
                + (f" [{tol_note}]" if tol_note else ""))
    if step_ok and gl == "ABSENT, UNLEARNED":
        return ("THE GAIN SCORE ALSO REMOVES THE LEAK (training unlearns the identity step): not separate defects; the "
                "step fix removes it too")
    if step_ok and gl == "PARTIAL":
        return "PARTIAL: the step fix removes the leak under both scores; under the gain score the raw step leaks less"
    return (f"STEP FIX INCOMPLETE (rotary: {rot}; gain: {gain}); GainRaw's leak {gl} -- reported as it falls")


def combo_verdict(vs_w, vs_n, rep=None):
    """(GP vs W two-sided, GP vs N non-inferiority[, D1r label]) -> D4 headline."""
    if "CONFLICT" in (vs_w, vs_n):
        return f"CONFLICT (vs MapWM {vs_w}, vs NormStep {vs_n}): reported as it falls"
    if vs_w == "WORSE" and vs_n == "AS GOOD" and rep == "WORSE":
        return ("WORSE THAN MapWM BUT AS GOOD AS NormStep: NormStep is itself below MapWM on these seeds (D1r WORSE); "
                "the gain-phase map is not worse than its step fix")
    if vs_w == "WORSE" and vs_n == "AS GOOD":
        return (f"WORSE THAN MapWM BUT NON-INFERIOR TO NormStep (within {MARGIN}); NormStep vs MapWM is {rep}, so the "
                f"shortfall is not attributed to either fix")
    if vs_w == "WORSE":
        return "THE GAIN-PHASE MAP FAILS: worse than MapWM"
    if vs_w == "BETTER":
        return {"AS GOOD": "THE GAIN-PHASE MAP WORKS: better than MapWM and as good as NormStep",
                "BETTER": "THE GAIN-PHASE MAP WORKS AND BEATS NORMSTEP",
                "WORSE": "BETTER THAN MapWM, BUT THE GAIN SCORE COSTS AGAINST NormStep",
                "UNDETERMINED": "BETTER THAN MapWM; non-inferiority to NormStep undetermined"}[vs_n]
    return f"NO GAIN OVER MapWM DETECTED (vs MapWM {vs_w}; vs NormStep {vs_n})"


def budget_flag(arm, a_cls, ref, r_cls, by, a_nll=None, r_nll=None):
    """Training scope (Amendment 1; Amendment 2: every non-SOLVED run counts, STALLED as well as DESCENDING -- a run that
    flattened at high loss under cosine decay is not converged either, rule 3). by='solved' (GainPhase vs NormStep): flag if
    the gain arm has more non-SOLVED runs than its rotary counterpart. by='regime' (GainRaw vs MapWM; MapWM never reaches
    0.05): Amendment 3 -- flag if the gain arm's median LEAK-FREE loss nll_ms is above REGIME_X x MapWM's largest and above
    REGIME_FLOOR (the training tail would call a converged but leakier GainRaw "not converged").
    Returns the qualifier string ('' if none)."""
    n = len(a_cls); k, st, d = class_counts(a_cls); kr, str_, dr = class_counts(r_cls)
    if by == "solved":
        hit = (n - k) > (len(r_cls) - kr)
    else:
        hit = np.median(a_nll) > max(REGIME_X * np.max(r_nll), REGIME_FLOOR)
    if not hit:
        return ""
    extra = (f"; {ref} {kr}/{n} SOLVED, {str_}/{n} STALLED, {dr}/{n} DESCENDING" if by == "solved" else
             f"; median leak-free loss {np.median(a_nll):.4f} vs {ref}'s max {np.max(r_nll):.4f}")
    return f" (scoped to {EPOCHS} epochs, not converged: {arm} {k}/{n} SOLVED, {st}/{n} STALLED, {d}/{n} DESCENDING{extra})"


def fires_on(detail):
    return detail.split("-- fires on ")[1].split(" --")[0] if "-- fires on " in detail else "neither test"


def acc_labels(acc, sol):
    """The accuracy-bearing registered labels, for the dropout-scale re-score comparison."""
    return {"D1r": contrast_state(acc[W], sol[W], acc[N], sol[N])[0], "D1": contrast_state(acc[G], sol[G], acc[GP], sol[GP])[0],
            "D2_raw": contrast_state(acc[W], sol[W], acc[G], sol[G])[0],
            "D2_norm": contrast_state(acc[N], sol[N], acc[GP], sol[GP])[0],
            "D4_vs_W": contrast_state(acc[W], sol[W], acc[GP], sol[GP])[0], "D4_vs_N": ni_state(acc[N], sol[N], acc[GP], sol[GP])[0]}


# ------------------------------------------------------------------------------------------- main
def load_runs(path_eval=f"{REPO}/GAIN_PHASE_EVAL.json", runs=R):
    J = json.load(open(path_eval)); out = {}
    for a in ARMS:
        for s in SEEDS:
            l = torch.load(f"{runs}/{a}_s{s}/{a}.pt", map_location="cpu", weights_only=False)["losses"]
            out[(a, s)] = dict(J[f"{a}|{s}"], losses=l)
    return out


def analyse(D, seeds=SEEDS, E=EPOCHS, out=print):
    """D[(arm, seed)] -> per-run dict with acc, L_ms, S_id, ..., losses. Prints every registered verdict; returns them."""
    bad = [(a, s) for a in ARMS for s in seeds if (a, s) not in D or len(D[(a, s)]["losses"]) != E]
    if bad:
        out(f"VOID: missing or not {E} epochs: {bad}"); return {"void": bad}
    col = lambda a, k: np.array([D[(a, s)][k] for s in seeds], float)
    cls = {(a, s): classify_run(D[(a, s)]["losses"]) for a in ARMS for s in seeds}
    sol = {a: [cls[(a, s)]["cls"] == "SOLVED" for s in seeds] for a in ARMS}
    acc = {a: col(a, "acc") for a in ARMS}
    ep = {a: [speed_epoch(D[(a, s)]["losses"]) for s in seeds] for a in ARMS}
    cl = {a: [cls[(a, s)] for s in seeds] for a in ARMS}
    bf = {G: budget_flag(G, cl[G], W, cl[W], "regime", col(G, "nll_ms"), col(W, "nll_ms")),
          GP: budget_flag(GP, cl[GP], N, cl[N], "solved")}
    rescore = None
    if all("acc_rescored" in D[(a, s)] for a in ARMS for s in seeds):
        reg = acc_labels(acc, sol); rs = acc_labels({a: col(a, "acc_rescored") for a in ARMS}, sol)
        rescore = {k: (f" [FLAG: under the dropout-scale re-score {k} reads {rs[k]}; registered verdict unchanged]"
                       if rs[k] != reg[k] else "") for k in reg}
    rflag = lambda *ks: "".join(rescore[k] for k in ks) if rescore else " [re-scored accuracies absent: flags not computed]"
    out("== per arm (unseen-object accuracy x1; leak readouts; classes) ==")
    for a in ARMS:
        out(f"  {a:9s} acc {acc[a].mean():.4f} +/- {acc[a].std(ddof=1):.4f} (min {acc[a].min():.4f}) | SOLVED "
            f"{sum(sol[a])}/{len(seeds)} | L_ms median {np.median(col(a, 'L_ms')):+.4f} | S_id {col(a, 'S_id').mean():.4f} | "
            + (f"theta reliance median {np.median(col(a, 'reliance')):.3f} | " if "reliance" in D[(a, seeds[0])] else "")
            + f"final-5% loss {np.mean([cls[(a, s)]['tail'] for s in seeds]):.4f} | epochs to 0.05 "
            + " ".join(str(e) if e <= E else "-" for e in ep[a]))
    V = {}
    rep, drep = contrast_state(acc[W], sol[W], acc[N], sol[N]); d1, dd1 = contrast_state(acc[G], sol[G], acc[GP], sol[GP])
    V["D1r"], V["D1"] = rep, d1
    V["D1_headline"] = (step_verdict(rep, d1) + f" [D1r fires on {fires_on(drep)}]" + bf[G] + bf[GP]
                        + rflag("D1r", "D1"))
    out(f"\n== D1 STEP ==\n  D1r NormStep vs MapWM (rotary; positive control): {rep}: {drep}\n"
        f"  D1  GainPhase vs GainRaw (gain score): {d1}: {dd1}\n  REGISTERED D1: {V['D1_headline']}")
    s_raw, ds_raw = contrast_state(acc[W], sol[W], acc[G], sol[G]); s_nrm, ds_nrm = contrast_state(acc[N], sol[N], acc[GP], sol[GP])
    V["D2_raw"], V["D2_norm"] = s_raw, s_nrm
    out(f"\n== D2 SCORE (gain vs rotary) ==\n  raw step: GainRaw vs MapWM: {s_raw}: {ds_raw}\n"
        f"  NormStep step: GainPhase vs NormStep: {s_nrm}: {ds_nrm}\n"
        f"  REGISTERED D2: raw step -- GAIN SCORE {s_raw}{bf[G]}; NormStep step -- GAIN SCORE {s_nrm}{bf[GP]}"
        + rflag("D2_raw", "D2_norm"))
    V["D2_headline"] = f"raw step -- GAIN SCORE {s_raw}{bf[G]}; NormStep step -- GAIN SCORE {s_nrm}{bf[GP]}" + rflag("D2_raw", "D2_norm")
    rot, drot = leak_side(col(W, "L_ms"), col(N, "L_ms"), col(W, "S_id"), col(N, "S_id"))
    gai, dgai = leak_side(col(G, "L_ms"), col(GP, "L_ms"), col(G, "S_id"), col(GP, "S_id"))
    gl, dgl = gain_leak(col(G, "L_ms"), col(W, "S_id"), col(G, "S_id"), col(G, "resid"), col(G, "shift_cells"),
                        col(W, "shift_cells"))
    gate = {a: converged_gate(col(a, "reliance"), col(a, "nll_ms"), cl[a]) for a in (G, GP)}
    gflag = ""
    if all("reliance_rescored" in D[(a, s)] for a in (G, GP) for s in seeds):
        gr = {a: converged_gate(col(a, "reliance_rescored"), col(a, "nll_ms"), cl[a]) for a in (G, GP)}
        gflag = "".join(f" [FLAG: under the dropout-scale re-score the D3 gate for {a} reads "
                        f"{'PASS' if gr[a][0] else 'FAIL (' + gr[a][1] + ')'}; registered verdict unchanged]"
                        for a in (G, GP) if gr[a][0] != gate[a][0])
    V["D3_rot"], V["D3_gain"], V["D3_gainraw"] = rot, gai, gl
    V["D3_gate"] = {a: gate[a][0] for a in gate}
    unconv = [a for a in (G, GP) if not gate[a][0]]
    if rot != "NO LEAK TO REMOVE" and unconv:
        V["D3_headline"] = "D3 UNMEASURED: " + "; ".join(f"{a}: {gate[a][1]} ({gate[a][2]})" for a in unconv)
    else:
        V["D3_headline"] = leak_verdict(rot, gai, gl, tolerated_note(col(G, "resid"), col(G, "shift_cells"),
                                                                     col(W, "shift_cells")) if gl == "ABSENT, TOLERATED" else "")
    if rot != "NO LEAK TO REMOVE":                     # Amendment 3: no flags on a void D3; GainRaw's scope as on D1 / D2
        V["D3_headline"] += bf[G] + gflag
    out(f"\n== D3 LEAK DISSOCIATION ==\n  convergence gate (Amendment 3): GainRaw {'PASS' if gate[G][0] else 'FAIL'}: "
        f"{gate[G][2]}\n  convergence gate: GainPhase {'PASS' if gate[GP][0] else 'FAIL'}: {gate[GP][2]}\n"
        f"  step fix under the rotary score: {rot}: {drot}\n"
        f"  step fix under the gain score: {gai}: {dgai}{'' if not unconv else ' [label not read: gate]'}\n"
        f"  leak under the gain score, raw step: {gl}: {dgl}{'' if gate[G][0] else ' [label not read: gate]'}\n"
        f"  REGISTERED D3: {V['D3_headline']}")
    vw, dvw = contrast_state(acc[W], sol[W], acc[GP], sol[GP]); vn, dvn = ni_state(acc[N], sol[N], acc[GP], sol[GP])
    V["D4_vs_W"], V["D4_vs_N"] = vw, vn
    V["D4_headline"] = (combo_verdict(vw, vn, rep) + (" (non-inferiority at ceiling)" if vn == "AS GOOD" and "(at ceiling)" in dvn else "")
                        + bf[GP] + rflag("D4_vs_W", "D4_vs_N"))
    out(f"\n== D4 COMBINATION ==\n  GainPhase vs MapWM: {vw}: {dvw}\n  GainPhase vs NormStep (non-inferiority): {vn}: {dvn}\n"
        f"  REGISTERED D4: {V['D4_headline']}")
    sp_n, dsp_n = speed_state(ep[N], ep[GP], E); sp_w, dsp_w = speed_state(ep[W], ep[G], E)
    V["D5_norm"], V["D5_raw"] = sp_n, sp_w
    out(f"\n== D5 SPEED (epochs until the 10-epoch running training loss < 0.05) ==\n"
        f"  GainPhase vs NormStep: {sp_n}: {dsp_n}\n  GainRaw vs MapWM: {sp_w}: {dsp_w}\n"
        f"  REGISTERED D5: NormStep step -- gain score {sp_n}; raw step -- gain score {sp_w}")
    return V


def main():
    D = load_runs()
    print(f"GAIN_PHASE registered analysis; runs {R}; seeds {SEEDS[0]}-{SEEDS[-1]}\n"
          f"floors for the readout (object-identity accuracy at object revisits, test pool; "
          f"docs/audits/2026-10-06/gain_phase_floor_out.txt): retrace 0.518, last object 0.141, most frequent 0.099, "
          f"uniform over the sequence's 16 objects 0.0625")
    V = analyse(D)
    json.dump(V, open(f"{REPO}/GAIN_PHASE_VERDICTS.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Declared secondaries of GAIN_PHASE_PREREG.md (no verdict). Run by the driver AFTER the registered artifacts and the
done marker; reads GAIN_PHASE_EVAL.json and the checkpoints. Output: gain_phase_secondary_out.txt.
 (a) leak decomposition per arm: L_zero (LEAK's readout), L_ms, S_id, field shift (cells), distortion share, shared and
     blank step (gauge part).
 (b) remap decomposition for the gain arms (analytic, docs/audits/2026-10-05/remap_probe.py's three levers adapted):
     GAIN = spread of the key gain mu_k over the 1000 unseen codes (cv) and blank vs objects; WIDTH = none by construction
     (the kernel sum_c A_c cos(dtheta_c) is content-free; its spectrum share per 8-channel band is printed); SHIFT =
     the field shift an object's own step gives its key (shift_cells; the leak expressed as remapping).
 (c) r(final-5% loss, acc) over the 32 runs and within arm (rule 2); run classes; train-pool accuracy.
 (d) x2 / x4 embedding-side code norm per arm: construction checks for the NormStep-step arms (must equal x1), a real
     distribution shift for the raw-step arms (rule 10: robustness, not capability).
 (e) dropout-scale re-score (rescore_hook, GainKernelLayer registered): intact x1 accuracy with attention x 1/(1-p);
     per-arm means and the registered accuracy contrasts' differences on the re-scored values.
 (f) the STEP x SCORE interaction (GP - G) - (N - W) on accuracy, descriptive; Holm over the registered accuracy p's.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
import mapformer.rescore_hook as RH
import mapformer.analyze_gain_phase as A
from mapformer.stats_core import classify_run, perm2_p
from mapformer.gain_phase_eval import load, Probe, obj_acc, sequences
from mapformer.model_codes import set_pool

REPO = A.REPO; W, N, G, GP = A.W, A.N, A.G, A.GP
# usage: gain_phase_secondary.py [EVAL_JSON RUNS_DIR SEED ...]   (defaults: the batch; the pilot passes its own)
EVAL = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/GAIN_PHASE_EVAL.json"
RUNS = sys.argv[2] if len(sys.argv) > 2 else A.R
if len(sys.argv) > 3:
    A.SEEDS = [int(x) for x in sys.argv[3:]]
J = json.load(open(EVAL))
D = A.load_runs(EVAL, RUNS)
col = lambda a, k: np.array([J[f"{a}|{s}"][k] for s in A.SEEDS], float)

print("== (a) leak decomposition per arm (means over seeds; L in accuracy units, steps relative to an axis step) ==")
for a in A.ARMS:
    print(f"  {a:9s} L_zero {col(a, 'L_zero').mean():+.4f} | L_ms {col(a, 'L_ms').mean():+.4f} | S_id {col(a, 'S_id').mean():.4f} | "
          f"shift {col(a, 'shift_cells').mean():.4f} cells | distortion share {col(a, 'resid').mean():.3f} | shared "
          f"{col(a, 'shared').mean():.4f} | blank {col(a, 'blank').mean():.4f}")

print("\n== (b) remap decomposition, gain arms (per head, mean over seeds) ==")
for a in (G, GP):
    g = [J[f"{a}|{s}"]["gains"] for s in A.SEEDS]
    m = lambda k: np.mean([x[k] for x in g], axis=0)
    print(f"  {a:9s} GAIN: mu_k objects {np.round(m('muk_obj_mean'), 3)} cv over codes {np.round(m('muk_obj_cv'), 3)} | blank "
          f"{np.round(m('muk_blank'), 3)} | action keys {np.round(m('muk_act'), 3)} | action queries {np.round(m('muq_act'), 3)}")
    print(f"  {'':9s} WIDTH: none by construction; spectrum share per band fine -> coarse {np.round(m('band_share'), 3).tolist()}")
    print(f"  {'':9s} SHIFT: object-identity field shift {col(a, 'shift_cells').mean():.4f} cells (rms), distortion share "
          f"{col(a, 'resid').mean():.3f}")

print("\n== (c) loss vs accuracy; classes; train pool ==")
tails = {(a, s): classify_run(D[(a, s)]["losses"]) for a in A.ARMS for s in A.SEEDS}
x = [tails[(a, s)]["tail"] for a in A.ARMS for s in A.SEEDS]; y = [J[f"{a}|{s}"]["acc"] for a in A.ARMS for s in A.SEEDS]
print(f"  r(final-5% loss, acc) over {len(x)} runs: {np.corrcoef(x, y)[0, 1]:+.3f}; within arm: " + "  ".join(
    f"{a} " + (f"{np.corrcoef([tails[(a, s)]['tail'] for s in A.SEEDS], col(a, 'acc'))[0, 1]:+.3f}" if col(a, 'acc').std() > 0 else "n/a")
    for a in A.ARMS))
for a in A.ARMS:
    print(f"  {a:9s} classes " + ",".join(f"{c}:{sum(tails[(a, s)]['cls'] == c for s in A.SEEDS)}"
                                         for c in ("SOLVED", "STALLED", "DESCENDING", "RISING"))
          + f" | train-pool acc {col(a, 'train_x1').mean():.4f} | test {col(a, 'acc').mean():.4f}")

print("\n== (d) code norm x2 / x4 (embedding side) ==")
for a in A.ARMS:
    d2, d4 = col(a, "x2") - col(a, "acc"), col(a, "x4") - col(a, "acc")
    kind = "CONSTRUCTION CHECK (must be ~0)" if a in (N, GP) else "distribution shift"
    print(f"  {a:9s} x2 - x1 {d2.mean():+.4f} | x4 - x1 {d4.mean():+.4f} (max |.| {max(abs(d2).max(), abs(d4).max()):.4f}) -- {kind}")

print("\n== (e) dropout-scale re-score (attention x 1/(1-p), eval-only) ==")
RH.KNOWN |= {"mapformer.model_em_pope.GainKernelLayer"}
RH.install("auto")
toks, revs = sequences("test"); dev = "cuda:0" if torch.cuda.is_available() else "cpu"
re_acc = {}
for a in A.ARMS:
    re_acc[a] = []
    for s in A.SEEDS:
        m, _ = load(f"{RUNS}/{a}_s{s}/{a}.pt", a, dev); p = Probe(m); set_pool(m, "test")
        re_acc[a].append(obj_acc(p, toks, revs, "test", None, dev))
    print(f"  {a:9s} {col(a, 'acc').mean():.4f} -> {np.mean(re_acc[a]):.4f}")
print(f"  rescore hooks: {RH.STATS['hooked']} hooked, skipped classes {sorted(RH.STATS['skipped']) or 'none'}")
for x_, y_ in ((W, N), (G, GP), (W, G), (N, GP), (W, GP)):
    for lab, f in (("registered", lambda a: col(a, "acc")), ("re-scored", lambda a: np.array(re_acc[a]))):
        print(f"  {y_} - {x_} ({lab}): {f(y_).mean() - f(x_).mean():+.4f} (perm p {perm2_p(f(x_), f(y_))['p']:.4f})")

print("\n== (f) STEP x SCORE interaction, descriptive ==")
acc = {a: col(a, "acc") for a in A.ARMS}
it = (acc[GP].mean() - acc[G].mean()) - (acc[N].mean() - acc[W].mean())
print(f"  (GainPhase - GainRaw) - (NormStep - MapWM) = {it:+.4f} (step effect under the gain score minus under the rotary)")
ps = {"D1r": perm2_p(acc[W], acc[N])["p"], "D1": perm2_p(acc[G], acc[GP])["p"], "D2 raw": perm2_p(acc[W], acc[G])["p"],
      "D2 norm": perm2_p(acc[N], acc[GP])["p"], "D4 vs W": perm2_p(acc[W], acc[GP])["p"]}
order = sorted(ps, key=ps.get); k = len(order); run = 0.0; parts = []
for i, n_ in enumerate(order):
    run = max(run, min(1.0, (k - i) * ps[n_])); parts.append(f"{n_} {ps[n_]:.4f} -> {run:.4f}")
print("  Holm over the registered two-sided accuracy p's: " + "; ".join(parts))

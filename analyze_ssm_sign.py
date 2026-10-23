"""Registered analysis for G3_SSM_PREREG.md (G-3: the sign test in a one-layer diagonal complex SSM).

  python3 -m mapformer.analyze_ssm_sign               # reads runs/ssm_sign/p0, writes G3_SSM_VERDICTS.json
  python3 -m mapformer.analyze_ssm_sign --selftest    # every label and reading, through load() on disk

Shared primitives only (rule 5): stats_core.classify_run (SOLVED = final-5% training loss < 0.05), fisher_solved,
perm2_p; the contrast labels are analyze_robust.cell_label's thresholds written against given p-values
(analyze_signroute.breadth_label; equal to cell_label on raw p-values, checked in the selftest).

Layout: <root>/A/<arm>_moves_m<m>_s<seed>  (block A, registered primary: steps = moves, m in {32, 8}, lr 3e-3)
        <root>/A2/<arm>_moves_m32_lr1e-3_s<seed>  (block A2: the same five arms at lr 1e-3, m = 32)
        <root>/B/<arm>_all_m32_s<seed>     (block B, secondary: every token steps, m = 32)
        <root>/H/<cls>_built_moves_m32_s<seed>  (block H, rule 14 continuations from block A's signed m = 32 runs, 3e-3)
        <root>/H2/<cls>_built_moves_m32_lr1e-3_s<seed>  (block H2: the SAME constructions continued at lr 1e-3)

Per m, block A (n = 8; signed S, abs A, softplus P, exp X, index I):
  G0   k_S >= 6, else UNREADABLE at m (the signed SSM does not learn the map within the budget).
  GI   index must fail: k_I <= 1 and mean acc_S - mean acc_I >= 0.05, else TASK DOES NOT NEED INTEGRATION (UNREADABLE
       for the sign question). (With k_S >= 6, k_I <= 1 already implies Fisher p <= 0.041.)
  P    label(A vs S) by cell_label's thresholds (raw p): SIGNED LEARNS BETTER / NO SIGN COST /
       NON-NEGATIVE LEARNS BETTER / UNMEASURED. m = 32 is the PRIMARY (the attention anchors' ring); m = 8 secondary.
  BR   P and X vs S at each m, Holm per test (Fisher p's together, permutation p's together) over the 4 cells; a label is
       kept only if the test that fired survives.
READING (registered) from m = 32's P, with m = 8's appended:
  UNREADABLE (G0 or GI fails at 32) | SIGN COST IN THE SSM | NO SIGN COST IN THE SSM | NON-NEGATIVE LEARNS BETTER |
  UNMEASURED; suffix "; m = 8: <label>"; then "; lr 1e-3 (m = 32): <label>" from block A2 (same gates; abs raw p,
  softplus / exp Holm over the 3 cells).

Rule 14, blocks H (lr 3e-3) and H2 (lr 1e-3; joint read against A2), each with the rule below
(classes abs, softplus, exp; control = the signed construction at the same seed), on the n_s sources
that are SOLVED in block A (UNREADABLE if n_s < 6):
  E1   frozen: |frozen acc - source acc| <= 0.005 on every source, and the construction's move-angle error <= 1e-4 rad.
       Fails -> NO EXACT CONSTRUCTION (that class's from-scratch failure is then not shown to be search).
  E2   LOST (as WRAP_EXISTENCE / MAIN_APPENDIX2 D6): max per-epoch training loss in the continuation > 0.3, or > 10x the
       control's at that seed and > 0.05. HELD: SOLVED on >= n_s - 1, mean acc >= control - 0.01, LOST on <= 1;
       PERTURBED AND RE-FOUND: SOLVED on >= n_s - 1 and mean acc >= control - 0.01 but LOST on >= 2; NOT HELD otherwise.
  Joint, per class at m = 32: SIGN LEARNS BETTER from scratch + E1 + HELD -> "SEARCH COST (exists, kept, not found)".

Block B (secondary): G0 on B's signed; abs / softplus / exp vs signed by the same labels, Holm over the 3 cells per test.

Diverged runs (any non-finite loss): DIVERGED, never SOLVED, non-finite accuracy counted 0. Every run's recipe and
len(losses) are asserted.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

from mapformer.analyze_robust import NLB, NSC, SLB, UNM
from mapformer.analyze_signroute import breadth_label, holm
from mapformer.stats_core import classify_run, fisher_solved, perm2_p

REPO = "/home/prashr/mapformer"                    # rule 25: absolute
ROOT = f"{REPO}/runs/ssm_sign/p0"
SEEDS = list(range(2100, 2108))
SIZES = (32, 8)                                    # 32 = primary
LR = 3e-3                                          # fixed from the signed pilots (G3_SSM_PREREG.md Sec. 5)
LR2 = 1e-3                                         # block A2: the second rate, m = 32 only (G3_SSM_PREREG.md Sec. 5)
EPOCHS = 300
EPOCHS_H = 100
ARMS_A = ("signed", "abs", "softplus", "exp", "index")
ARMS_B = ("signed", "abs", "softplus", "exp")
CLASSES_H = ("abs", "softplus", "exp", "signed")
RECIPE = {"n_steps": 128, "batch_size": 128, "n_batches": 98, "data_workers": 3, "n_trials": 400,
          "weight_decay": 0.05, "deltas": "1,-1", "rank": 4, "decay": "none"}
UNR, NNI = "UNREADABLE", "TASK DOES NOT NEED INTEGRATION"
COST, NOCOST, REV = "SIGN COST IN THE SSM", "NO SIGN COST IN THE SSM", "NON-NEGATIVE LEARNS BETTER"
HELD, PRF, NHELD, NOEX = "HELD", "PERTURBED AND RE-FOUND", "NOT HELD", "NO EXACT CONSTRUCTION"
ATTN_REF = ("attention anchors on record (ring 32, AO, 300 epochs, never pooled): at lr 3e-3 LRSWEEP signed 8/8, |.| 2/8 "
            "(|.|'s best rate); SIGNROUTE exp 1/8 0.958, softplus 3/8 0.966. At lr 1e-3: |.| 1/8 0.917, exp 1/8 0.935, "
            "softplus 1/8 0.879, signed 8/8")


def run_dir(root, block, arm, steps, m, seed):
    tag = f"{arm}_built" if block in ("H", "H2") else arm
    lr = "_lr1e-3" if block in ("A2", "H2") else ""
    return os.path.join(root, block, f"{tag}_{steps}_m{m}{lr}_s{seed}")


def block_lr(block):
    return LR2 if block in ("A2", "H2") else LR


def better(k_a, k_b, acc_a, acc_b, n):
    """The house SIGNED-LEARNS-BETTER thresholds with a in the reference slot (as analyze_signroute.better)."""
    pf = fisher_solved(k_b, n, k_a, n)
    pp = perm2_p(np.asarray(acc_b, float), np.asarray(acc_a, float))["p"]
    d = float(np.mean(acc_a) - np.mean(acc_b))
    return (pf < 0.05 and k_a > k_b) or (pp < 0.05 and d >= 0.05), {"fisher_p": pf, "perm_p": pp, "d": d}


def _fin(x):
    return float(x) if x is not None and np.isfinite(x) else 0.0


def _row(p, want, epochs):
    e = json.load(open(p))
    A = e["args"]
    for k, v in want.items():
        assert A.get(k) == v if not isinstance(v, float) else abs(A.get(k) - v) <= 1e-9 * abs(v), (p, k, A.get(k), v)
    bad = {k: A.get(k) for k, v in RECIPE.items() if (A.get(k) != v if not isinstance(v, float)
                                                      else abs(A.get(k) - v) > 1e-9 * v)}
    assert not bad, (p, bad)
    assert A["epochs"] == epochs, (p, A["epochs"], epochs)
    L = np.asarray(e["losses"], float)
    assert len(L) == epochs, (p, len(L))
    div = not np.all(np.isfinite(L))
    k = classify_run(np.nan_to_num(L, nan=1e9, posinf=1e9)) if div else classify_run(L)
    return {"acc": _fin(e["eval"]["128"]["acc"]), "acc4": _fin(e["eval"]["512"]["acc"]),
            "final_loss": float(e["final_loss"]), "div": div, "losses": L,
            "max_loss": float(np.nanmax(np.nan_to_num(L, nan=1e9))),
            "solved": (k["registered"] == "SOLVED") and not div, "cls": "DIVERGED" if div else k["cls"],
            "frozen": e.get("frozen"),
            **{q: e.get(q) for q in ("latent_cos", "u_ratio", "cancel_share_nonblind", "wrap_cancel_share_nonblind",
                                     "move_step_min", "move_step_frac_negative", "obs_angle_wrapped_absmax")}}


def load(root=ROOT, seeds=SEEDS, need_h=True, need_b=True, need_a2=True):
    """need_a2 covers blocks A2 and H2 (the 1e-3 readouts); need_h covers H (and H2 when need_a2)."""
    R, missing = {}, []
    specs = [("A", a, "moves", m) for m in SIZES for a in ARMS_A]
    if need_a2:
        specs += [("A2", a, "moves", 32) for a in ARMS_A]
    if need_b:
        specs += [("B", a, "all", 32) for a in ARMS_B]
    if need_h:
        specs += [("H", c, "moves", 32) for c in CLASSES_H]
        if need_a2:
            specs += [("H2", c, "moves", 32) for c in CLASSES_H]
    for blk, arm, steps, m in specs:
        for s in seeds:
            p = os.path.join(run_dir(root, blk, arm, steps, m, s), "eval.json")
            if not os.path.exists(p):
                missing.append(p)
                continue
            want = {"arm": arm, "steps": steps, "size": m, "seed": s, "init_built": None, "lr": block_lr(blk)}
            if blk in ("H", "H2"):           # H2 continues the SAME constructions at lr 1e-3
                want["init_built"] = os.path.join(root, "H", "built", f"s{s}", f"{arm}_built.pt")
            R[(blk, arm, m, s)] = _row(p, want, EPOCHS_H if blk in ("H", "H2") else EPOCHS)
    return R, missing


def cell(R, blk, arm, m, seeds):
    rows = [R[(blk, arm, m, s)] for s in seeds]
    return {"k": sum(r["solved"] for r in rows), "acc": [r["acc"] for r in rows], "rows": rows}


def contrast_labels(cells, ref, arms, n, holm_adjust):
    """{arm: label} of arm vs ref by cell_label's thresholds; Holm per test over `arms` when holm_adjust."""
    raw = []
    for arm in arms:
        a = cells[arm]
        pf = fisher_solved(a["k"], n, ref["k"], n)
        pp = perm2_p(np.asarray(ref["acc"]), np.asarray(a["acc"]))["p"]
        raw.append((arm, a, pf, pp))
    af = holm([r[2] for r in raw]) if holm_adjust else [r[2] for r in raw]
    ap = holm([r[3] for r in raw]) if holm_adjust else [r[3] for r in raw]
    out = {}
    for (arm, a, pf, pp), pfa, ppa in zip(raw, af, ap):
        out[arm] = {"label": breadth_label(a["acc"], ref["acc"], a["k"], ref["k"], pfa, ppa),
                    "raw_label": breadth_label(a["acc"], ref["acc"], a["k"], ref["k"], pf, pp),
                    "fisher_p": pf, "perm_p": pp, "fisher_p_adj": float(pfa), "perm_p_adj": float(ppa),
                    "d": float(np.mean(a["acc"]) - np.mean(ref["acc"])), "k": a["k"], "k_ref": ref["k"]}
    return out


def gates(C, n):
    S, I = C["signed"], C["index"]
    if S["k"] < 6:
        return f"{UNR} (signed SOLVED {S['k']}/{n} < 6)"
    # k_S >= 6 and k_I <= 1 already give Fisher p <= 0.041, so the accuracy clause is the one that can bite: an index
    # arm that is not SOLVED but scores within 0.05 of signed would mean the task is mostly solvable without integration
    if I["k"] > 1 or np.mean(S["acc"]) - np.mean(I["acc"]) < 0.05:
        return f"{NNI} (index SOLVED {I['k']}/{n}, acc {np.mean(I['acc']):.3f})"
    return None


def reading(primary, secondary):
    def one(lab):
        return {SLB: COST, NSC: NOCOST, NLB: REV}.get(lab, lab)
    head = primary if primary.startswith(UNR) or primary.startswith(NNI) else one(primary)
    return f"{head}; m = 8: {secondary if secondary.startswith((UNR, NNI)) else one(secondary)}"


def existence(R, seeds, C32, hb="H"):
    """Block hb (H at 3e-3, H2 at 1e-3: the same constructions) on the sources SOLVED in block A (signed, m = 32)."""
    src = [s for s, r in zip(seeds, C32["signed"]["rows"]) if r["solved"]]
    n_s = len(src)
    res = {"n_sources": n_s, "sources": src}
    if n_s < 6:
        res["label"] = {c: f"{UNR} (n_s {n_s} < 6)" for c in CLASSES_H[:-1]}
        return res
    src_acc = {s: R[("A", "signed", 32, s)]["acc"] for s in src}
    ctl = {s: R[(hb, "signed", 32, s)] for s in src}
    res["label"], res["detail"] = {}, {}
    for c in CLASSES_H[:-1]:
        rows = {s: R[(hb, c, 32, s)] for s in src}
        fz = {s: rows[s]["frozen"] or {} for s in src}
        e1_acc = all(abs(fz[s].get("acc", -9) - src_acc[s]) <= 0.005 for s in src)
        e1_ang = all(((fz[s].get("construction") or {}).get("angle_err_moves", 9.0)) <= 1e-4 for s in src)
        lost = {s: rows[s]["max_loss"] > 0.3 or (rows[s]["max_loss"] > 10 * ctl[s]["max_loss"]
                                                  and rows[s]["max_loss"] > 0.05) for s in src}
        k = sum(rows[s]["solved"] for s in src)
        acc = float(np.mean([rows[s]["acc"] for s in src]))
        acc_c = float(np.mean([ctl[s]["acc"] for s in src]))
        nl = sum(lost.values())
        if not (e1_acc and e1_ang):
            lab = NOEX
        elif k >= n_s - 1 and acc >= acc_c - 0.01:
            lab = HELD if nl <= 1 else PRF
        else:
            lab = NHELD
        res["label"][c] = lab
        res["detail"][c] = {"E1_acc": e1_acc, "E1_angle": e1_ang, "solved": k, "lost": nl, "acc": acc,
                            "control_acc": acc_c,
                            "frozen_acc": [fz[s].get("acc") for s in src], "source_acc": [src_acc[s] for s in src]}
    return res


def analyse(R, seeds=SEEDS, out=print):
    n = len(seeds)
    v = {"A": {}, "B": {}, "gate": {}}
    out("== G-3, block A (steps = moves; registered primary m = 32, secondary m = 8; lr 3e-3, 300 epochs) ==")
    CA = {}
    for m in SIZES:
        C = {a: cell(R, "A", a, m, seeds) for a in ARMS_A}
        CA[m] = C
        g = gates(C, n)
        v["gate"][m] = g or "pass"
        out(f"m = {m}: SOLVED " + " ".join(f"{a} {C[a]['k']}/{n}" for a in ARMS_A) + " | acc T=128 "
            + " ".join(f"{a} {np.mean(C[a]['acc']):.4f}" for a in ARMS_A))
        out(f"   gates: {g or 'G0 and GI pass'}")
    # primary abs vs signed (raw p) per m; breadth with Holm over the 4 cells
    prim = {}
    for m in SIZES:
        lab = contrast_labels(CA[m], CA[m]["signed"], ["abs"], n, False)["abs"]
        prim[m] = v["gate"][m] if v["gate"][m] != "pass" else lab["label"]
        v["A"][f"abs|{m}"] = lab
        out(f"   P m = {m}: abs vs signed -> {prim[m]}  (Fisher {lab['fisher_p']:.4f}, perm {lab['perm_p']:.4f}, "
            f"d {lab['d']:+.4f})")
    raw = []
    for m in SIZES:
        for arm in ("softplus", "exp"):
            a, s = CA[m][arm], CA[m]["signed"]
            raw.append((m, arm, a, s, fisher_solved(a["k"], n, s["k"], n),
                        perm2_p(np.asarray(s["acc"]), np.asarray(a["acc"]))["p"]))
    af, ap = holm([r[4] for r in raw]), holm([r[5] for r in raw])
    for (m, arm, a, s, pf, pp), pfa, ppa in zip(raw, af, ap):
        lab = breadth_label(a["acc"], s["acc"], a["k"], s["k"], pfa, ppa)
        lab = v["gate"][m] if v["gate"][m] != "pass" else lab
        v["A"][f"{arm}|{m}"] = {"label": lab, "fisher_p": pf, "perm_p": pp, "fisher_p_adj": float(pfa),
                                "perm_p_adj": float(ppa), "k": a["k"], "d": float(np.mean(a["acc"]) - np.mean(s["acc"]))}
        out(f"   BR m = {m} {arm:8s}: SOLVED {a['k']}/{n} acc {np.mean(a['acc']):.4f} -> {lab} (Fisher {pf:.4f} / "
            f"Holm {pfa:.4f}, perm {pp:.4f} / Holm {ppa:.4f})")
    v["reading"] = reading(prim[32], prim[8])
    if all(("A2", a, 32, s) in R for a in ARMS_A for s in seeds):
        C2 = {a: cell(R, "A2", a, 32, seeds) for a in ARMS_A}
        g2 = gates(C2, n)
        out(f"\n== block A2 (registered secondary: the same arms at lr 1e-3, m = 32) ==")
        out("   SOLVED " + " ".join(f"{a} {C2[a]['k']}/{n}" for a in ARMS_A) + " | acc "
            + " ".join(f"{a} {np.mean(C2[a]['acc']):.4f}" for a in ARMS_A))
        out(f"   gates: {g2 or 'G0 and GI pass'}")
        labs2 = contrast_labels(C2, C2["signed"], ["abs", "softplus", "exp"], n, True)
        for a, l in labs2.items():
            if a == "abs":                      # the primary contrast, unadjusted, as at 3e-3
                l["label"] = l["raw_label"]
            l["label"] = g2 or l["label"]
            out(f"   {a:8s} vs signed -> {l['label']} (raw {l['raw_label']}; Fisher {l['fisher_p']:.4f} / Holm "
                f"{l['fisher_p_adj']:.4f}, perm {l['perm_p']:.4f} / Holm {l['perm_p_adj']:.4f})")
        v["A2"] = labs2
        v["reading"] += f"; lr 1e-3 (m = 32): " + (g2 if g2 else {SLB: COST, NSC: NOCOST, NLB: REV}.get(
            labs2["abs"]["label"], labs2["abs"]["label"]))
    out(f"\nREADING (registered): {v['reading']}")
    out(f"   {ATTN_REF}")

    for hb, fb, rate in (("H", "A", "3e-3"), ("H2", "A2", "1e-3")):
        if not all((hb, c, 32, s) in R for c in CLASSES_H for s in seeds):
            continue
        ex = existence(R, seeds, CA[32], hb)
        v["existence" if hb == "H" else "existence_lr1e-3"] = ex
        out(f"\n== rule 14, block {hb} (lr {rate}; continuations from the n_s = {ex['n_sources']} SOLVED signed m = 32 "
            f"sources of block A) ==")
        joint = {}
        for c, lab in ex["label"].items():
            d = (ex.get("detail") or {}).get(c, {})
            out(f"   {c:8s}: {lab}  {json.dumps({k: d[k] for k in ('E1_acc', 'E1_angle', 'solved', 'lost', 'acc', 'control_acc') if k in d})}")
            fr = (v["A"].get(f"{c}|32", {}) if fb == "A" else v.get("A2", {}).get(c, {})).get("label")
            joint[c] = "SEARCH COST" if fr == SLB and lab == HELD else None
            if joint[c]:
                out(f"      joint (lr {rate}): SEARCH COST (a {c} solution exists, is kept under training at this rate, "
                    f"and is not found from scratch at it)")
        v["joint" if hb == "H" else "joint_lr1e-3"] = joint

    if all(("B", a, 32, s) in R for a in ARMS_B for s in seeds):
        CB = {a: cell(R, "B", a, 32, seeds) for a in ARMS_B}
        g = f"{UNR} (signed SOLVED {CB['signed']['k']}/{n} < 6)" if CB["signed"]["k"] < 6 else None
        out("\n== block B (secondary; every token steps, m = 32) ==")
        out("   SOLVED " + " ".join(f"{a} {CB[a]['k']}/{n}" for a in ARMS_B) + " | acc "
            + " ".join(f"{a} {np.mean(CB[a]['acc']):.4f}" for a in ARMS_B))
        labs = contrast_labels(CB, CB["signed"], ["abs", "softplus", "exp"], n, True)
        for a, l in labs.items():
            l["label"] = g or l["label"]
            out(f"   {a:8s} vs signed -> {l['label']} (raw {l['raw_label']}; Fisher {l['fisher_p']:.4f} / Holm "
                f"{l['fisher_p_adj']:.4f}, perm {l['perm_p']:.4f} / Holm {l['perm_p_adj']:.4f})")
        v["B"] = labs

    out("\n== rule 2: r(log10 final loss, acc), finite runs, blocks A and B ==")
    fin = [r for k, r in R.items() if not r["div"] and k[0] != "H"]
    if len(fin) > 2:
        L = np.array([r["final_loss"] for r in fin])
        for key, nm in (("acc", "T=128"), ("acc4", "T=512")):
            A = np.array([r[key] for r in fin])
            rr = float(np.corrcoef(np.log10(L + 1e-6), A)[0, 1]) if A.std() > 0 and L.std() > 0 else float("nan")
            out(f"   {nm}: r = {rr:+.3f} over {len(A)} runs")
    code = {"SOLVED": "S", "STALLED": "T", "DESCENDING": "D", "RISING": "R", "DIVERGED": "X"}
    out("\nper seed (acc T=128 + class: S SOLVED, T STALLED, D DESCENDING, R RISING, X DIVERGED):")
    for (blk, arm, m) in sorted({k[:3] for k in R}):
        rows = [R[(blk, arm, m, s)] for s in seeds if (blk, arm, m, s) in R]
        med = lambda q: np.median([r[q] for r in rows if r.get(q) is not None]) if any(
            r.get(q) is not None for r in rows) else float("nan")
        out(f"  {blk} m{m:<2d} {arm:8s} " + " ".join(f"{r['acc']:.3f}{code[r['cls']]}" for r in rows)
            + f" | med cos(zx,zy) {med('latent_cos'):+.3f} cancel {med('cancel_share_nonblind'):.2f} "
              f"wrap-cancel {med('wrap_cancel_share_nonblind'):.2f}")
    v["solved"] = {f"{k[0]}|{k[1]}|{k[2]}": int(sum(R[(k[0], k[1], k[2], s)]["solved"] for s in seeds
                                                   if (k[0], k[1], k[2], s) in R)) for k in {kk[:3] for kk in R}}
    v["n_diverged"] = int(sum(r["div"] for r in R.values()))
    return v


# ------------------------------------------------------------------------------ selftest
def _curve(solved, epochs, diverge=False, spike=False, warm=False):
    """From scratch: 1.5 -> 0.3, then 1e-4 (SOLVED) or flat 0.3 (STALLED). Warm (block H): starts at the solution."""
    if warm:
        L = [1e-4 if solved else 0.2] * epochs
    else:
        L = list(np.linspace(1.5, 0.3, epochs // 2)) + ([1e-4] * (epochs - epochs // 2) if solved else
                                                        [0.3] * (epochs - epochs // 2))
    if spike:
        L[3] = 0.6
    if diverge:
        L[-5:] = [float("nan")] * 5
    return L


def _write(root, spec, lo=None, h=None, seed=0, hi=None):
    """Real eval.json files in the driver's layout. spec {(blk, arm, m): k solved}; lo {(blk, arm, m): unsolved acc};
    h {cls: {"lost": n spiked, "frozen_off": bool, "k": solved}}."""
    rng = np.random.default_rng(seed)
    lo, h, hi = lo or {}, h or {}, hi or {}
    specs = [("A", a, "moves", m) for m in SIZES for a in ARMS_A] + [("A2", a, "moves", 32) for a in ARMS_A] + \
            [("B", a, "all", 32) for a in ARMS_B] + [("H", c, "moves", 32) for c in CLASSES_H] + \
            [("H2", c, "moves", 32) for c in CLASSES_H]
    for blk, arm, steps, m in specs:
        isH = blk in ("H", "H2")
        ep = EPOCHS_H if isH else EPOCHS
        hh = h.get(arm, {}) if isH else {}
        k = hh.get("k", 8) if isH else spec.get((blk, arm, m), 8 if arm == "signed" else 0)
        for i, s in enumerate(SEEDS):
            sol = i < k
            div = spec.get(("div", blk, arm, m)) and i == len(SEEDS) - 1
            d = run_dir(root, blk, arm, steps, m, s)
            os.makedirs(d, exist_ok=True)
            acc = (hi.get((blk, arm, m), 1.0) if sol else lo.get((blk, arm, m), 0.95 if isH else 0.90)) \
                - 0.002 * rng.random()
            args = dict(RECIPE, arm=arm, steps=steps, size=m, seed=s, epochs=ep, lr=block_lr(blk),
                        init_built=(os.path.join(root, "H", "built", f"s{s}", f"{arm}_built.pt") if isH
                                    else None))
            frozen = None
            if isH:
                off = 0.05 if hh.get("frozen_off") else 0.0
                frozen = {"acc": 1.0 - off, "construction": {"angle_err_moves": 1e-6}}
            e = {"eval": {"128": {"acc": float("nan") if div else acc, "nll": 0.0},
                          "512": {"acc": 0.99 if sol else 0.8, "nll": 0.0}},
                 "arm": arm, "steps": steps, "size": m, "lr": block_lr(blk), "seed": s, "args": args,
                 "losses": _curve(sol and not div, ep, diverge=div, spike=(isH and i < hh.get("lost", 0)), warm=isH),
                 "final_loss": float("nan") if div else (1e-4 if sol else 0.3), "frozen": frozen}
            json.dump(e, open(os.path.join(d, "eval.json"), "w"))
    # block A's signed sources must have acc matching the frozen acc (1.0 - jitter): set them exactly to 1.0
    for s in SEEDS:
        p = os.path.join(run_dir(root, "A", "signed", "moves", 32, s), "eval.json")
        e = json.load(open(p))
        if e["eval"]["128"]["acc"] > 0.99:
            e["eval"]["128"]["acc"] = 1.0
        json.dump(e, open(p, "w"))


def selftest():
    silent = lambda *a, **k: None
    tmp = tempfile.mkdtemp(prefix="ssm_sign_selftest_")
    base = {("A", "abs", 32): 1, ("A", "abs", 8): 1, ("A", "softplus", 32): 1, ("A", "exp", 32): 1,
            ("A", "softplus", 8): 1, ("A", "exp", 8): 1, ("A", "index", 32): 0, ("A", "index", 8): 0,
            ("B", "abs", 32): 1, ("B", "softplus", 32): 1, ("B", "exp", 32): 1}
    lo_idx = {("A", "index", 32): 0.70, ("A", "index", 8): 0.70}

    def sp(**kw):
        d = dict(base)
        for k, val in kw.items():
            arm, m = k.rsplit("_", 1)
            d[("A", arm, int(m))] = val
        return d
    cases = [
        (f"{COST}; m = 8: {COST}", sp(), lo_idx, None, None),
        (f"{NOCOST}; m = 8: {COST}", sp(abs_32=8), lo_idx, None, None),
        (f"{REV}; m = 8: {COST}", sp(abs_32=8, signed_32=6), {**lo_idx, ("A", "signed", 32): 0.60}, None,
         {("A", "signed", 32): 0.93}),
        (f"{UNM}; m = 8: {COST}", sp(abs_32=5), lo_idx, None, None),
        (f"{UNR} (signed SOLVED 5/8 < 6); m = 8: {COST}", sp(signed_32=5), lo_idx, None, None),
        (f"{NNI} (index SOLVED 2/8, acc 0.774); m = 8: {COST}", sp(index_32=2), lo_idx, None, None),
        (f"{COST}; m = 8: {NNI} (index SOLVED 2/8, acc 0.774)", sp(index_8=2), lo_idx, None, None),
        # GI's second clause: index unsolved but within reach of signed on accuracy (no Fisher, no perm)
        (f"{NNI} (index SOLVED 0/8, acc 0.968); m = 8: {COST}", sp(signed_32=6, index_32=0),
         {**lo_idx, ("A", "signed", 32): 0.90, ("A", "index", 32): 0.969}, None, None),
    ]
    ok, seen_ex, seen_lab = True, set(), set()
    SUF = f"; lr 1e-3 (m = 32): {COST}"
    cases = [(w + SUF, sp_, lo_, h_, hi_) for (w, sp_, lo_, h_, hi_) in cases]
    cases += [(f"{COST}; m = 8: {COST}; lr 1e-3 (m = 32): {NOCOST}", {**sp(), ("A2", "abs", 32): 8}, lo_idx, None, None),
              (f"{COST}; m = 8: {COST}; lr 1e-3 (m = 32): {UNR} (signed SOLVED 4/8 < 6)",
               {**sp(), ("A2", "signed", 32): 4}, lo_idx, None, None)]
    for i, (want, spec, lo, h, hi) in enumerate(cases):
        root = os.path.join(tmp, f"c{i}")
        _write(root, spec, lo, h, hi=hi)
        R, missing = load(root)
        assert not missing, missing[:2]
        vv = analyse(R, out=silent)
        got = vv["reading"]
        seen_lab |= {x["label"] for x in vv["A"].values()}
        ok &= got == want
        print(f"  {'ok ' if got == want else 'BAD'} want {want!r}\n      got  {got!r}")
    for lab in (SLB, NSC, NLB, UNM):
        if lab not in seen_lab:
            print(f"  BAD block-A label never reached: {lab}")
            ok = False
    # existence labels: HELD, PERTURBED AND RE-FOUND, NOT HELD, NO EXACT CONSTRUCTION, and the joint SEARCH COST
    hcases = [({"abs": {"k": 8}, "softplus": {"k": 8, "lost": 3}, "exp": {"k": 5}}, {"abs": HELD, "softplus": PRF,
                                                                                   "exp": NHELD}),
              ({"abs": {"k": 8, "frozen_off": True}, "softplus": {"k": 8, "lost": 1}, "exp": {"k": 7}},
               {"abs": NOEX, "softplus": HELD, "exp": HELD})]
    for j, (h, want) in enumerate(hcases):
        root = os.path.join(tmp, f"h{j}")
        _write(root, sp(), lo_idx, h)
        R, _ = load(root)
        v = analyse(R, out=silent)
        got = v["existence"]["label"]
        seen_ex |= set(got.values())
        good = got == want
        if j == 0:
            good &= v["joint"]["abs"] == "SEARCH COST" and v["joint"]["exp"] is None
            good &= v["existence_lr1e-3"]["label"] == want and v["joint_lr1e-3"]["abs"] == "SEARCH COST"
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} existence want {want} got {got} joint {v['joint']}")
    # existence unreadable when fewer than 6 sources solved
    root = os.path.join(tmp, "h_unr")
    _write(root, sp(signed_32=5), lo_idx)
    R, _ = load(root)
    ex = analyse(R, out=silent)["existence"]["label"]
    good = all(x.startswith(UNR) for x in ex.values())
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} existence with n_s = 5 -> {ex}")
    # divergence: DIVERGED, never SOLVED, acc 0
    root = os.path.join(tmp, "div")
    s = sp()
    s[("div", "A", "abs", 32)] = True
    _write(root, s, lo_idx)
    R, _ = load(root)
    r = R[("A", "abs", 32, SEEDS[-1])]
    good = r["div"] and not r["solved"] and r["cls"] == "DIVERGED" and r["acc"] == 0.0
    good &= analyse(R, out=silent)["n_diverged"] == 1
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} diverged run handled")
    # block B labels, incl. NO SIGN COST and gate
    root = os.path.join(tmp, "b")
    s = sp()
    s[("B", "abs", 32)] = 8
    _write(root, s, lo_idx)
    R, _ = load(root)
    vb = analyse(R, out=silent)["B"]
    good = vb["abs"]["label"] == NSC and vb["exp"]["label"] == SLB
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} block B labels {[(a, l['label']) for a, l in vb.items()]}")
    # recipe guard: a wrong lr must raise
    root = os.path.join(tmp, "recipe")
    _write(root, sp(), lo_idx)
    p = os.path.join(run_dir(root, "A", "abs", "moves", 32, SEEDS[0]), "eval.json")
    e = json.load(open(p))
    e["args"]["lr"] = 1e-3                     # block A is registered at 3e-3
    json.dump(e, open(p, "w"))
    try:
        load(root)
        good = False
    except AssertionError:
        good = True
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} recipe guard raises on a wrong lr")
    # breadth_label equals cell_label on raw p
    from mapformer.analyze_robust import cell_label
    for ka, accs in ((0, [0.8] * 8), (8, [1.0] * 8), (3, [1.0] * 3 + [0.9] * 5)):
        so = [1.0] * 8
        pf = fisher_solved(ka, 8, 8, 8)
        pp = perm2_p(np.asarray(so), np.asarray(accs))["p"]
        ok &= breadth_label(accs, so, ka, 8, pf, pp) == cell_label(accs, so, ka, 8, 8)[0]
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--no-h", action="store_true", help="read blocks A and B only (stage 1 complete)")
    ap.add_argument("--no-b", action="store_true", help="block B was dropped before launch (G3_BLOCK_B=0)")
    ap.add_argument("--no-a2", action="store_true", help="block A2 was dropped before launch (G3_BLOCK_A2=0)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    R, missing = load(a.root, need_h=not a.no_h, need_b=not a.no_b, need_a2=not a.no_a2)
    if missing:
        print(f"MISSING {len(missing)} runs, e.g. {missing[:3]}")
        sys.exit(2)
    v = analyse(R)
    json.dump(v, open(f"{REPO}/G3_SSM_VERDICTS.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()

"""Registered analysis for G3_SSM_PREREG.md (G-3: the sign test in a one-layer diagonal complex SSM), as amended by
Amendment 1 (the independent check, docs/audits/2026-10-23/g3_ssm_audit.md).

  python3 -m mapformer.analyze_ssm_sign --config minimal_h2      # reads runs/ssm_sign/p0, writes G3_SSM_VERDICTS.json
  python3 -m mapformer.analyze_ssm_sign --selftest               # every label and reading, through load() on disk

Shared primitives only (rule 5): stats_core.classify_run (SOLVED = final-5% training loss < 0.05), fisher_solved,
perm2_p; the contrast labels are analyze_robust.cell_label's thresholds written against given p-values
(analyze_signroute.breadth_label; equal to cell_label on raw p-values, checked in the selftest).

CONFIGURATIONS (a launch-time choice, CONFIGS below; the driver passes the same name):
  minimal_h2  A at m = 32 + H + H2                     104 runs (recommended default)
  core        A at m = 32 and 8 + H                     112 runs
  full        A at m = 32 and 8 + A2 + B + H + H2       216 runs
  a2_after_minimal_h2  (Amendment 2) block A2 (40 runs) added to a finished minimal_h2 root; read as A at m = 32 + A2 +
              H + H2 (the readings `full` registers for those blocks: "; lr 1e-3 (m = 32): <label>", H2 joins A2)

Layout: <root>/A/<arm>_moves_m<m>_s<seed>              block A, primary: steps = moves, lr 3e-3
        <root>/A2/<arm>_moves_m32_lr1e-3_s<seed>       block A2: the same five arms at lr 1e-3, m = 32 (full only)
        <root>/B/<arm>_all_m32_s<seed>                 block B: every token steps, m = 32 (full only)
        <root>/H/<cls>_built_moves_m32_s<seed>         block H: rule 14 continuations from A's signed m = 32 runs, 3e-3
        <root>/H2/<cls>_built_moves_m32_lr1e-3_s<seed> block H2: the SAME constructions continued at lr 1e-3

RUNAWAY (Amendment 1.4): final move_step_absmax > 1e3, or any epoch after the 5% warm-up with training loss > 2.08 (the
memoryless arm's). Counted per cell; a contrast whose non-negative cell has k_RUNAWAY >= 2 keeps its label with the
suffix "(k RUNAWAY)" and licenses no learnability sentence (no joint is formed from it).

Per m, block A (n = 8; signed S, abs A, softplus P, exp X, index I):
  G0   k_S >= 6, else UNREADABLE at m.
  GI   k_I <= 1 and mean acc_S - mean acc_I >= 0.05, else TASK DOES NOT NEED INTEGRATION at m.
  P    abs vs S by cell_label's thresholds (raw p), GATED by G0 / GI (Amendment 1.1: the gated label is the one stored
       and read by the joint). m = 32 primary; m = 8 secondary when in the configuration.
  BR   softplus, exp vs S at each m, Holm per test over the 2 x |sizes| cells, gated.
READING: from m = 32's P [+ "; m = 8: <label>" if m = 8 is in the configuration] [+ "; lr 1e-3 (m = 32): <label>" if
  block A2 is]. Wording (Amendment 1.5): SIGN COST IN THE SSM reads "not specific to softmax attention"; NO SIGN COST IN
  THE SSM reads "not shown without softmax normalisation", not "attention-specific".

Rule 14, blocks H (3e-3) and H2 (1e-3), on the n_s sources SOLVED in block A at m = 32 (UNREADABLE if n_s < 6):
  E1   frozen |acc - source acc| <= 0.005 on every source, for the class AND for the signed control (Amendment 1.7),
       and the construction's move-angle error <= 1e-4 rad; else NO EXACT CONSTRUCTION.
  E2   (registered) LOST: max per-epoch continuation loss > 0.3, or > 10x the control's at that seed and > 0.05.
       HELD: SOLVED on >= n_s - 1, mean acc >= control - 0.01, LOST on <= 1; PERTURBED AND RE-FOUND: the same but LOST on
       >= 2; NOT HELD: neither (declared: not WRAP_EXISTENCE A4's statistical NOT HELD).
  E2'  (declared secondary, Amendment 1.3) the same with LOST' = max loss > 0.3 only; printed with each class's max-loss
       range beside the attention ranges (AO_bal 0.024-0.154, SO_bal 0.015-0.112).
  softplus / exp: any E2 other than HELD is read as a property of constructions (f) / (g) at omega' = 2 pi / ln 2, 2 pi,
       not of the class (Amendment 1.2); the class-level statement is made for abs only. Sensitivities printed.
  Joint (per class): from-scratch SIGNED LEARNS BETTER (gated, no RUNAWAY suffix) at the same rate, plus
       HELD -> SEARCH COST; PERTURBED AND RE-FOUND -> COST, SOLUTION PERTURBED AND RE-FOUND; NOT HELD -> COST, SOLUTION
       NOT HELD (Amendment 1.7: the two are distinguished; with 100-epoch continuations the line between them is set by
       the budget). H joins A (3e-3); H2 joins A2 when A2 is run, otherwise A's 3e-3 label as a declared CROSS-RATE joint.

Block B (full only): G0 on B's signed; abs / softplus / exp vs signed, Holm over the 3 cells per test; RUNAWAY applies.

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
LR = 3e-3                                          # fixed from the signed pilots (G3_SSM_PREREG.md Sec. 5)
LR2 = 1e-3                                         # blocks A2, H2
EPOCHS = 300
EPOCHS_H = 100
ARMS_A = ("signed", "abs", "softplus", "exp", "index")
ARMS_B = ("signed", "abs", "softplus", "exp")
CLASSES_H = ("abs", "softplus", "exp", "signed")
CONFIGS = {"minimal_h2": {"sizes": (32,), "a2": False, "b": False, "h2": True},
           "core": {"sizes": (32, 8), "a2": False, "b": False, "h2": False},
           "full": {"sizes": (32, 8), "a2": True, "b": True, "h2": True},
           # Amendment 2: A2 run into a finished minimal_h2 root ("base"); the driver launches A2 only and refuses
           # to re-run A / H / H2. The analysis reads it as any configuration with these blocks.
           "a2_after_minimal_h2": {"sizes": (32,), "a2": True, "b": False, "h2": True, "base": "minimal_h2"}}
RECIPE = {"n_steps": 128, "batch_size": 128, "n_batches": 98, "data_workers": 3, "n_trials": 400,
          "weight_decay": 0.05, "deltas": "1,-1", "rank": 4, "decay": "none"}
RUNAWAY_STEP, RUNAWAY_LOSS = 1e3, 2.08              # Amendment 1.4; 2.08 = the memoryless pilot's loss
UNR, NNI = "UNREADABLE", "TASK DOES NOT NEED INTEGRATION"
COST, NOCOST, REV = "SIGN COST IN THE SSM", "NO SIGN COST IN THE SSM", "NON-NEGATIVE LEARNS BETTER"
HELD, PRF, NHELD, NOEX = "HELD", "PERTURBED AND RE-FOUND", "NOT HELD", "NO EXACT CONSTRUCTION"
J_SEARCH, J_PRF, J_NHELD = "SEARCH COST", "COST, SOLUTION PERTURBED AND RE-FOUND", "COST, SOLUTION NOT HELD"
CONSTRUCTION_NOTE = {"softplus": "construction (f), omega' = 2 pi / ln 2", "exp": "construction (g), omega' = 2 pi"}
ATTN_REF = ("attention anchors on record (ring 32, AO, 300 epochs, never pooled): at lr 3e-3 LRSWEEP signed 8/8, |.| 2/8 "
            "(|.|'s best rate); SIGNROUTE exp 1/8 0.958, softplus 3/8 0.966. At lr 1e-3: |.| 1/8 0.917, exp 1/8 0.935, "
            "softplus 1/8 0.879, signed 8/8")
ATTN_MAXLOSS = "attention continuation max-loss ranges (WRAP_EXISTENCE ring, lr 1e-3): AO_bal 0.024-0.154, SO_bal 0.015-0.112"
WORDING = {COST: "not specific to softmax attention (it appears in unnormalised rotary linear attention = this SSM)",
           NOCOST: "not shown without softmax normalisation (a scope boundary at softmax, not 'attention-specific')"}


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
    warm = max(1, int(0.05 * epochs))                     # train.py's cosine warm-up is 5% of the steps
    step_max = e.get("move_step_absmax")
    runaway = bool((step_max is not None and step_max > RUNAWAY_STEP)
                   or np.any(np.nan_to_num(L[warm:], nan=1e9) > RUNAWAY_LOSS))
    return {"acc": _fin(e["eval"]["128"]["acc"]), "acc4": _fin(e["eval"]["512"]["acc"]),
            "final_loss": float(e["final_loss"]), "div": div, "losses": L,
            "max_loss": float(np.nanmax(np.nan_to_num(L, nan=1e9))),
            "solved": (k["registered"] == "SOLVED") and not div, "cls": "DIVERGED" if div else k["cls"],
            "frozen": e.get("frozen"), "runaway": runaway, "move_step_absmax": step_max,
            **{q: e.get(q) for q in ("latent_cos", "u_ratio", "cancel_share_nonblind", "wrap_cancel_share_nonblind",
                                     "move_step_min", "move_step_frac_negative", "obs_angle_wrapped_absmax")}}


def specs_of(cfg):
    sp = [("A", a, "moves", m) for m in cfg["sizes"] for a in ARMS_A]
    if cfg["a2"]:
        sp += [("A2", a, "moves", 32) for a in ARMS_A]
    if cfg["b"]:
        sp += [("B", a, "all", 32) for a in ARMS_B]
    sp += [("H", c, "moves", 32) for c in CLASSES_H]
    if cfg["h2"]:
        sp += [("H2", c, "moves", 32) for c in CLASSES_H]
    return sp


def load(root=ROOT, cfg=None, seeds=SEEDS, need_h=True):
    cfg = cfg or CONFIGS["full"]
    R, missing = {}, []
    for blk, arm, steps, m in specs_of(cfg):
        if blk in ("H", "H2") and not need_h:
            continue
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
    return {"k": sum(r["solved"] for r in rows), "acc": [r["acc"] for r in rows], "rows": rows,
            "kr": sum(r["runaway"] for r in rows)}


def _ra(c):
    return f" ({c['kr']} RUNAWAY)" if c["kr"] >= 2 else ""


def contrast_labels(cells, ref, arms, n, holm_adjust, gate=None):
    """{arm: {...}} of arm vs ref by cell_label's thresholds; Holm per test over `arms` when holm_adjust. "label" is the
    GATED base label; "shown" adds the RUNAWAY suffix; "licensed" is False when the suffix applies."""
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
        lab = gate or breadth_label(a["acc"], ref["acc"], a["k"], ref["k"], pfa, ppa)
        out[arm] = {"label": lab, "shown": lab + _ra(a), "licensed": a["kr"] < 2, "k_runaway": a["kr"],
                    "raw_label": breadth_label(a["acc"], ref["acc"], a["k"], ref["k"], pf, pp),
                    "fisher_p": pf, "perm_p": pp, "fisher_p_adj": float(pfa), "perm_p_adj": float(ppa),
                    "d": float(np.mean(a["acc"]) - np.mean(ref["acc"])), "k": a["k"], "k_ref": ref["k"]}
    return out


def gates(C, n):
    S, I = C["signed"], C["index"]
    if S["k"] < 6:
        return f"{UNR} (signed SOLVED {S['k']}/{n} < 6)"
    # k_S >= 6 and k_I <= 1 already give Fisher p <= 0.041, so the accuracy clause is the one that can bite
    if I["k"] > 1 or np.mean(S["acc"]) - np.mean(I["acc"]) < 0.05:
        return f"{NNI} (index SOLVED {I['k']}/{n}, acc {np.mean(I['acc']):.3f})"
    return None


def one(lab_entry):
    """Reading words for one gated contrast entry (base label + RUNAWAY suffix)."""
    lab = lab_entry["label"]
    if lab.startswith((UNR, NNI)):
        return lab
    return {SLB: COST, NSC: NOCOST, NLB: REV}.get(lab, lab) + _ra({"kr": lab_entry["k_runaway"]})


def existence(R, seeds, C32, hb="H"):
    """Block hb (H at 3e-3, H2 at 1e-3: the same constructions) on the sources SOLVED in block A (signed, m = 32)."""
    src = [s for s, r in zip(seeds, C32["signed"]["rows"]) if r["solved"]]
    n_s = len(src)
    res = {"n_sources": n_s, "sources": src}
    if n_s < 6:
        res["label"] = {c: f"{UNR} (n_s {n_s} < 6)" for c in CLASSES_H[:-1]}
        res["label_secondary"] = dict(res["label"])
        return res
    src_acc = {s: R[("A", "signed", 32, s)]["acc"] for s in src}
    ctl = {s: R[(hb, "signed", 32, s)] for s in src}
    ctl_e1 = all(abs(((ctl[s]["frozen"] or {}).get("acc", -9)) - src_acc[s]) <= 0.005 for s in src)
    res["control_E1"] = ctl_e1
    res["control_max_loss"] = [ctl[s]["max_loss"] for s in src]
    res["label"], res["label_secondary"], res["detail"] = {}, {}, {}
    for c in CLASSES_H[:-1]:
        rows = {s: R[(hb, c, 32, s)] for s in src}
        fz = {s: rows[s]["frozen"] or {} for s in src}
        e1_acc = all(abs(fz[s].get("acc", -9) - src_acc[s]) <= 0.005 for s in src)
        e1_ang = all(((fz[s].get("construction") or {}).get("angle_err_moves", 9.0)) <= 1e-4 for s in src)
        k = sum(rows[s]["solved"] for s in src)
        acc = float(np.mean([rows[s]["acc"] for s in src]))
        acc_c = float(np.mean([ctl[s]["acc"] for s in src]))
        lost = {s: rows[s]["max_loss"] > 0.3 or (rows[s]["max_loss"] > 10 * ctl[s]["max_loss"]
                                                  and rows[s]["max_loss"] > 0.05) for s in src}
        lost2 = {s: rows[s]["max_loss"] > 0.3 for s in src}

        def lab_of(L):
            if not (e1_acc and e1_ang and ctl_e1):
                return NOEX
            if k >= n_s - 1 and acc >= acc_c - 0.01:
                return HELD if sum(L.values()) <= 1 else PRF
            return NHELD
        res["label"][c], res["label_secondary"][c] = lab_of(lost), lab_of(lost2)
        sens = [(fz[s].get("construction") or {}).get("sensitivity_mean") for s in src]
        sens = [x for x in sens if x is not None]
        res["detail"][c] = {"E1_acc": e1_acc, "E1_angle": e1_ang, "E1_control": ctl_e1, "solved": k,
                            "lost": int(sum(lost.values())), "lost_abs_only": int(sum(lost2.values())), "acc": acc,
                            "control_acc": acc_c,
                            "max_loss_range": [min(rows[s]["max_loss"] for s in src),
                                               max(rows[s]["max_loss"] for s in src)],
                            "sensitivity_median": float(np.median(sens)) if sens else None,
                            "frozen_acc": [fz[s].get("acc") for s in src], "source_acc": [src_acc[s] for s in src]}
    return res


def joint_of(scratch, ex_lab):
    """Joint per class: scratch = the gated contrast entry (or None); ex_lab = the E2 label."""
    if scratch is None or scratch["label"] != SLB or not scratch["licensed"]:
        return None
    return {HELD: J_SEARCH, PRF: J_PRF, NHELD: J_NHELD}.get(ex_lab)


def analyse(R, cfg=None, seeds=SEEDS, out=print):
    cfg = cfg or CONFIGS["full"]
    sizes = cfg["sizes"]
    n = len(seeds)
    v = {"config": {k: (list(x) if isinstance(x, tuple) else x) for k, x in cfg.items()}, "A": {}, "gate": {}}
    out(f"== G-3, block A (steps = moves; lr 3e-3, 300 epochs; m = {', '.join(map(str, sizes))}; 32 primary) ==")
    CA = {}
    for m in sizes:
        C = {a: cell(R, "A", a, m, seeds) for a in ARMS_A}
        CA[m] = C
        v["gate"][m] = gates(C, n)
        out(f"m = {m}: SOLVED " + " ".join(f"{a} {C[a]['k']}/{n}" for a in ARMS_A) + " | RUNAWAY "
            + " ".join(f"{a} {C[a]['kr']}" for a in ARMS_A) + " | acc T=128 "
            + " ".join(f"{a} {np.mean(C[a]['acc']):.4f}" for a in ARMS_A))
        out(f"   gates: {v['gate'][m] or 'G0 and GI pass'}")
    for m in sizes:                     # P: abs vs signed, raw p, GATED (Amendment 1.1)
        lab = contrast_labels(CA[m], CA[m]["signed"], ["abs"], n, False, v["gate"][m])["abs"]
        v["A"][f"abs|{m}"] = lab
        out(f"   P m = {m}: abs vs signed -> {lab['shown']}  (raw {lab['raw_label']}; Fisher {lab['fisher_p']:.4f}, "
            f"perm {lab['perm_p']:.4f}, d {lab['d']:+.4f})")
    raw = []
    for m in sizes:
        for arm in ("softplus", "exp"):
            a, s = CA[m][arm], CA[m]["signed"]
            raw.append((m, arm, a, s, fisher_solved(a["k"], n, s["k"], n),
                        perm2_p(np.asarray(s["acc"]), np.asarray(a["acc"]))["p"]))
    af, ap = holm([r[4] for r in raw]), holm([r[5] for r in raw])
    for (m, arm, a, s, pf, pp), pfa, ppa in zip(raw, af, ap):
        lab = v["gate"][m] or breadth_label(a["acc"], s["acc"], a["k"], s["k"], pfa, ppa)
        v["A"][f"{arm}|{m}"] = {"label": lab, "shown": lab + _ra(a), "licensed": a["kr"] < 2, "k_runaway": a["kr"],
                                "fisher_p": pf, "perm_p": pp, "fisher_p_adj": float(pfa), "perm_p_adj": float(ppa),
                                "k": a["k"], "d": float(np.mean(a["acc"]) - np.mean(s["acc"]))}
        out(f"   BR m = {m} {arm:8s}: SOLVED {a['k']}/{n} acc {np.mean(a['acc']):.4f} -> {v['A'][f'{arm}|{m}']['shown']}"
            f" (Fisher {pf:.4f} / Holm {pfa:.4f}, perm {pp:.4f} / Holm {ppa:.4f})")
    rd = one(v["A"]["abs|32"])
    if 8 in sizes:
        rd += f"; m = 8: {one(v['A']['abs|8'])}"
    if cfg["a2"]:
        C2 = {a: cell(R, "A2", a, 32, seeds) for a in ARMS_A}
        g2 = gates(C2, n)
        out("\n== block A2 (lr 1e-3, m = 32; the index arm repeats A's gate at the second rate) ==")
        out("   SOLVED " + " ".join(f"{a} {C2[a]['k']}/{n}" for a in ARMS_A) + " | RUNAWAY "
            + " ".join(f"{a} {C2[a]['kr']}" for a in ARMS_A) + " | acc "
            + " ".join(f"{a} {np.mean(C2[a]['acc']):.4f}" for a in ARMS_A))
        out(f"   gates: {g2 or 'G0 and GI pass'}")
        # Sec. 6 / Amendment 3.1: softplus and exp by Holm per test over the 3 cells {abs, softplus, exp}; abs's own
        # label at raw p
        labs2 = contrast_labels(C2, C2["signed"], ["abs", "softplus", "exp"], n, True, g2)
        labs2["abs"] = contrast_labels(C2, C2["signed"], ["abs"], n, False, g2)["abs"]
        for a, l in labs2.items():
            out(f"   {a:8s} vs signed -> {l['shown']} (raw {l['raw_label']}; Fisher {l['fisher_p']:.4f} / adj "
                f"{l['fisher_p_adj']:.4f}, perm {l['perm_p']:.4f} / adj {l['perm_p_adj']:.4f})")
        v["A2"] = labs2
        # declared sensitivity (Amendment 3.1), not a registered label: Holm over the 2 cells {softplus, exp}
        sens2 = contrast_labels(C2, C2["signed"], ["softplus", "exp"], n, True, g2)
        v["A2_sensitivity_2cell"] = sens2
        out("   sensitivity (declared, not registered): Holm over 2 cells {softplus, exp}: "
            + "; ".join(f"{a} {l['shown']} (Fisher adj {l['fisher_p_adj']:.4f}, perm adj {l['perm_p_adj']:.4f})"
                        + ("" if l["label"] == labs2[a]["label"] else "  [DIFFERS from the registered label]")
                        for a, l in sens2.items()))
        rd += f"; lr 1e-3 (m = 32): {one(labs2['abs'])}"
    v["reading"] = rd
    out(f"\nREADING (registered): {rd}")
    head = v["A"]["abs|32"]["label"]
    if head in (SLB, NSC) and v["A"]["abs|32"]["licensed"]:
        out(f"   wording (Amendment 1.5): {WORDING[COST if head == SLB else NOCOST]}")
    elif not v["A"]["abs|32"]["licensed"]:
        out("   RUNAWAY >= 2 in the abs cell: no learnability sentence is licensed (Amendment 1.4)")
    out(f"   {ATTN_REF}")
    if 8 not in sizes:
        out("   m = 8 not in this configuration")

    for hb, rate in (("H", "3e-3"), ("H2", "1e-3")):
        if hb == "H2" and not cfg["h2"]:
            continue
        if not all((hb, c, 32, s) in R for c in CLASSES_H for s in seeds):
            continue
        ex = existence(R, seeds, CA[32], hb)
        key = "" if hb == "H" else "_lr1e-3"
        v["existence" + key] = ex
        out(f"\n== rule 14, block {hb} (lr {rate}; continuations from the n_s = {ex['n_sources']} SOLVED signed m = 32 "
            f"sources of block A) ==")
        if "control_E1" in ex:
            out(f"   signed control: E1 {ex['control_E1']}, max loss {min(ex['control_max_loss']):.3f}-"
                f"{max(ex['control_max_loss']):.3f}")
        if hb == "H":
            src_lab = {c: v["A"].get(f"{c}|32") for c in CLASSES_H[:-1]}
            jkind = "same rate (3e-3)"
        elif cfg["a2"]:
            src_lab = {c: v["A2"].get(c) for c in CLASSES_H[:-1]}
            jkind = "same rate (1e-3)"
        else:
            src_lab = {c: v["A"].get(f"{c}|32") for c in CLASSES_H[:-1]}
            jkind = "CROSS-RATE (from scratch at 3e-3, kept at 1e-3; declared)"
        joint = {}
        for c, lab in ex["label"].items():
            d = (ex.get("detail") or {}).get(c, {})
            note = (f"  [{CONSTRUCTION_NOTE[c]}: read as a property of the construction, not the class]"
                    if c in CONSTRUCTION_NOTE and lab != HELD and not lab.startswith(UNR) else "")
            out(f"   {c:8s}: E2 {lab}{note} | E2' (LOST = max > 0.3 only) {ex['label_secondary'][c]}")
            if d:
                out(f"            solved {d['solved']}/{ex['n_sources']}, LOST {d['lost']} (abs-only {d['lost_abs_only']}),"
                    f" acc {d['acc']:.4f} vs control {d['control_acc']:.4f}, max loss {d['max_loss_range'][0]:.3f}-"
                    f"{d['max_loss_range'][1]:.3f}, sensitivity median {d['sensitivity_median']}")
            j = joint_of(src_lab[c], lab)
            if j and c in CONSTRUCTION_NOTE and j != J_SEARCH:
                j += f" ({CONSTRUCTION_NOTE[c]})"
            joint[c] = j
            if j:
                out(f"      joint, {jkind}: {j}")
        out(f"   {ATTN_MAXLOSS}")
        v["joint" + key] = joint
        v["joint_kind" + key] = jkind

    if cfg["b"] and all(("B", a, 32, s) in R for a in ARMS_B for s in seeds):
        CB = {a: cell(R, "B", a, 32, seeds) for a in ARMS_B}
        g = f"{UNR} (signed SOLVED {CB['signed']['k']}/{n} < 6)" if CB["signed"]["k"] < 6 else None
        out("\n== block B (secondary; every token steps, m = 32) ==")
        out("   SOLVED " + " ".join(f"{a} {CB[a]['k']}/{n}" for a in ARMS_B) + " | RUNAWAY "
            + " ".join(f"{a} {CB[a]['kr']}" for a in ARMS_B) + " | acc "
            + " ".join(f"{a} {np.mean(CB[a]['acc']):.4f}" for a in ARMS_B))
        labs = contrast_labels(CB, CB["signed"], ["abs", "softplus", "exp"], n, True, g)
        for a, l in labs.items():
            out(f"   {a:8s} vs signed -> {l['shown']} (raw {l['raw_label']}; Fisher {l['fisher_p']:.4f} / Holm "
                f"{l['fisher_p_adj']:.4f}, perm {l['perm_p']:.4f} / Holm {l['perm_p_adj']:.4f})")
        v["B"] = labs

    out("\n== rule 2: r(log10 final loss, acc), finite runs, blocks A / A2 / B (Amendment 1.7) ==")
    fin = [r for k, r in R.items() if not r["div"] and k[0] in ("A", "A2", "B")]
    if len(fin) > 2:
        L = np.array([r["final_loss"] for r in fin])
        for key, nm in (("acc", "T=128"), ("acc4", "T=512")):
            A = np.array([r[key] for r in fin])
            rr = float(np.corrcoef(np.log10(L + 1e-6), A)[0, 1]) if A.std() > 0 and L.std() > 0 else float("nan")
            out(f"   {nm}: r = {rr:+.3f} over {len(A)} runs")
    code = {"SOLVED": "S", "STALLED": "T", "DESCENDING": "D", "RISING": "R", "DIVERGED": "X"}
    out("\nper seed (acc T=128 + class: S SOLVED, T STALLED, D DESCENDING, R RISING, X DIVERGED; ! = RUNAWAY):")
    for (blk, arm, m) in sorted({k[:3] for k in R}):
        rows = [R[(blk, arm, m, s)] for s in seeds if (blk, arm, m, s) in R]
        med = lambda q: np.median([r[q] for r in rows if r.get(q) is not None]) if any(
            r.get(q) is not None for r in rows) else float("nan")
        out(f"  {blk:2s} m{m:<2d} {arm:8s} " + " ".join(f"{r['acc']:.3f}{code[r['cls']]}{'!' if r['runaway'] else ''}"
                                                      for r in rows)
            + f" | med cos(zx,zy) {med('latent_cos'):+.3f} cancel {med('cancel_share_nonblind'):.2f} "
              f"wrap-cancel {med('wrap_cancel_share_nonblind'):.2f}")
    v["solved"] = {f"{k[0]}|{k[1]}|{k[2]}": int(sum(R[(k[0], k[1], k[2], s)]["solved"] for s in seeds
                                                   if (k[0], k[1], k[2], s) in R)) for k in sorted({kk[:3] for kk in R})}
    v["runaway"] = {f"{k[0]}|{k[1]}|{k[2]}": int(sum(R[(k[0], k[1], k[2], s)]["runaway"] for s in seeds
                                                    if (k[0], k[1], k[2], s) in R)) for k in sorted({kk[:3] for kk in R})}
    v["n_diverged"] = int(sum(r["div"] for r in R.values()))
    return v


# ------------------------------------------------------------------------------ selftest
def _curve(solved, epochs, diverge=False, spike=None, warm=False, runaway=False):
    """From scratch: 1.5 -> 0.3, then 1e-4 (SOLVED) or flat 0.3 (STALLED). Warm (blocks H, H2): starts at the solution;
    spike = the peak value put at epoch 3. runaway: the loss climbs above 2.08 after the warm-up."""
    if warm:
        L = [1e-4 if solved else 0.2] * epochs
    else:
        L = list(np.linspace(1.5, 0.3, epochs // 2)) + ([1e-4] * (epochs - epochs // 2) if solved else
                                                        [0.3] * (epochs - epochs // 2))
    if spike is not None:
        L[3] = spike
    if runaway:
        L[epochs // 3] = 2.5
    if diverge:
        L[-5:] = [float("nan")] * 5
    return L


def _write(root, spec, lo=None, h=None, seed=0, hi=None, cfg=None):
    """Real eval.json files in the driver's layout for every spec of `cfg` (default full).
    spec {(blk, arm, m): k solved, ("div", blk, arm, m): True, ("run", blk, arm, m): k runaway seeds};
    lo / hi {(blk, arm, m): unsolved / solved acc}; h {cls: {"lost": n spiked, "spike": value, "frozen_off": bool,
    "k": solved}} applied to H and H2 (cls "signed" = the control)."""
    rng = np.random.default_rng(seed)
    lo, h, hi = lo or {}, h or {}, hi or {}
    for blk, arm, steps, m in specs_of(cfg or CONFIGS["full"]):
        isH = blk in ("H", "H2")
        ep = EPOCHS_H if isH else EPOCHS
        hh = h.get(arm, {}) if isH else {}
        k = hh.get("k", 8) if isH else spec.get((blk, arm, m), 8 if arm == "signed" else 0)
        n_run = spec.get(("run", blk, arm, m), 0)
        for i, s in enumerate(SEEDS):
            sol = i < k
            run = (not sol) and i >= len(SEEDS) - n_run
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
                frozen = {"acc": 1.0 - off, "construction": {"angle_err_moves": 1e-6, "sensitivity_mean": 100.0}}
            sp = hh.get("spike", 0.6) if (isH and i < hh.get("lost", 0)) else (0.001 if isH else None)
            e = {"eval": {"128": {"acc": float("nan") if div else acc, "nll": 0.0},
                          "512": {"acc": 0.99 if sol else 0.8, "nll": 0.0}},
                 "arm": arm, "steps": steps, "size": m, "lr": block_lr(blk), "seed": s, "args": args,
                 "losses": _curve(sol and not div, ep, diverge=div, spike=sp, warm=isH, runaway=False),
                 "final_loss": float("nan") if div else (1e-4 if sol else 0.3), "frozen": frozen,
                 "move_step_absmax": 5e4 if run else 3.0}
            with open(os.path.join(d, "eval.json"), "w") as f:
                json.dump(e, f)
    p = os.path.join(run_dir(root, "A", "signed", "moves", 32, SEEDS[0]), "eval.json")
    for s in SEEDS:                       # block A's solved signed sources at exactly 1.0 = the frozen acc
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
    lo_idx = {("A", "index", 32): 0.70, ("A", "index", 8): 0.70, ("A2", "index", 32): 0.70}

    def sp(**kw):
        d = dict(base)
        for k, val in kw.items():
            arm, m = k.rsplit("_", 1)
            d[("A", arm, int(m))] = val
        return d
    F, MIN, CORE = CONFIGS["full"], CONFIGS["minimal_h2"], CONFIGS["core"]
    SUF = f"; lr 1e-3 (m = 32): {COST}"
    cases = [   # (want, spec, lo, h, hi, cfg)
        (f"{COST}; m = 8: {COST}{SUF}", sp(), lo_idx, None, None, F),
        (f"{NOCOST}; m = 8: {COST}{SUF}", sp(abs_32=8), lo_idx, None, None, F),
        (f"{REV}; m = 8: {COST}{SUF}", sp(abs_32=8, signed_32=6), {**lo_idx, ("A", "signed", 32): 0.60}, None,
         {("A", "signed", 32): 0.93}, F),
        (f"{UNM}; m = 8: {COST}{SUF}", sp(abs_32=5), lo_idx, None, None, F),
        (f"{UNR} (signed SOLVED 5/8 < 6); m = 8: {COST}{SUF}", sp(signed_32=5), lo_idx, None, None, F),
        (f"{NNI} (index SOLVED 2/8, acc 0.774); m = 8: {COST}{SUF}", sp(index_32=2), lo_idx, None, None, F),
        (f"{COST}; m = 8: {NNI} (index SOLVED 2/8, acc 0.774){SUF}", sp(index_8=2), lo_idx, None, None, F),
        (f"{NNI} (index SOLVED 0/8, acc 0.968); m = 8: {COST}{SUF}", sp(signed_32=6, index_32=0),
         {**lo_idx, ("A", "signed", 32): 0.90, ("A", "index", 32): 0.969}, None, None, F),
        (f"{COST}; m = 8: {COST}; lr 1e-3 (m = 32): {NOCOST}", {**sp(), ("A2", "abs", 32): 8}, lo_idx, None, None, F),
        (f"{COST}; m = 8: {COST}; lr 1e-3 (m = 32): {UNR} (signed SOLVED 4/8 < 6)",
         {**sp(), ("A2", "signed", 32): 4}, lo_idx, None, None, F),
        # configurations: m = 32 only, no A2 (minimal_h2); m = 32 and 8, no A2 (core)
        (COST, sp(), lo_idx, None, None, MIN),
        (f"{COST}; m = 8: {NOCOST}", sp(abs_8=8), lo_idx, None, None, CORE),
        # RUNAWAY (Amendment 1.4): abs at m = 32 with 2 runaway seeds keeps its label with the suffix
        (f"{COST} (2 RUNAWAY)", {**sp(), ("run", "A", "abs", 32): 2}, lo_idx, None, None, MIN),
    ]
    ok, seen_lab = True, set()
    for i, (want, spec, lo, h, hi, cfg) in enumerate(cases):
        root = os.path.join(tmp, f"c{i}")
        _write(root, spec, lo, h, hi=hi, cfg=cfg)
        R, missing = load(root, cfg)
        assert not missing, missing[:2]
        vv = analyse(R, cfg, out=silent)
        got = vv["reading"]
        seen_lab |= {x["label"] for x in vv["A"].values()}
        good = got == want
        if i == 12:          # a RUNAWAY cell licenses no joint
            good &= vv["joint"]["abs"] is None
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} want {want!r}\n      got  {got!r}")
    for lab in (SLB, NSC, NLB, UNM):
        if lab not in seen_lab:
            print(f"  BAD block-A label never reached: {lab}")
            ok = False
    # MAJOR 1: a GI failure at m = 32 forms no joint (the audit's c4_joint case: index 2/8)
    root = os.path.join(tmp, "gi_joint")
    _write(root, sp(index_32=2), lo_idx, cfg=MIN)
    R, _ = load(root, MIN)
    v = analyse(R, MIN, out=silent)
    good = v["reading"].startswith(NNI) and all(x is None for x in v["joint"].values()) \
        and all(x is None for x in v["joint_lr1e-3"].values())
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} GI failure -> no joint: {v['joint']} / {v['joint_lr1e-3']}")
    # existence labels and joints (full config: H joins A, H2 joins A2)
    hcases = [({"abs": {"k": 8}, "softplus": {"k": 8, "lost": 3}, "exp": {"k": 5}},
               {"abs": HELD, "softplus": PRF, "exp": NHELD},
               {"abs": J_SEARCH, "softplus": J_PRF, "exp": J_NHELD}),
              ({"abs": {"k": 8, "frozen_off": True}, "softplus": {"k": 8, "lost": 1}, "exp": {"k": 7}},
               {"abs": NOEX, "softplus": HELD, "exp": HELD}, {"abs": None, "softplus": J_SEARCH, "exp": J_SEARCH}),
              # the signed control fails E1 -> every class NO EXACT CONSTRUCTION (Amendment 1.7)
              ({"signed": {"k": 8, "frozen_off": True}}, {"abs": NOEX, "softplus": NOEX, "exp": NOEX},
               {"abs": None, "softplus": None, "exp": None})]
    for j, (h, want, wantj) in enumerate(hcases):
        root = os.path.join(tmp, f"h{j}")
        _write(root, sp(), lo_idx, h, cfg=F)
        R, _ = load(root, F)
        v = analyse(R, F, out=silent)
        got = v["existence"]["label"]
        gj = {c: (x.split(" (construction")[0] if x else x) for c, x in v["joint"].items()}
        good = got == want and gj == wantj and v["existence_lr1e-3"]["label"] == want
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} existence want {want} got {got}; joint {gj}")
    # E2' (MAJOR 3): a 0.1 peak against a 0.001 control is LOST by the 10x clause, not by the absolute one
    root = os.path.join(tmp, "e2p")
    _write(root, sp(), lo_idx, {"abs": {"k": 8, "lost": 8, "spike": 0.1}}, cfg=MIN)
    R, _ = load(root, MIN)
    v = analyse(R, MIN, out=silent)
    good = v["existence"]["label"]["abs"] == PRF and v["existence"]["label_secondary"]["abs"] == HELD
    good &= v["joint"]["abs"] == J_PRF and v["joint_kind_lr1e-3"].startswith("CROSS-RATE")
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} E2 {v['existence']['label']['abs']} vs E2' "
          f"{v['existence']['label_secondary']['abs']}; H2 joint kind {v['joint_kind_lr1e-3']}")
    # Amendment 2: A2 added to a minimal_h2 root -> the A2 suffix, A2's own labels (exp with a RUNAWAY suffix forms no
    # joint), and H2 joins A2 at the same rate (not CROSS-RATE)
    A2C = CONFIGS["a2_after_minimal_h2"]
    root = os.path.join(tmp, "a2c")
    _write(root, {**sp(), ("run", "A2", "exp", 32): 3}, lo_idx, cfg=A2C)
    R, missing = load(root, A2C)
    v = analyse(R, A2C, out=silent)
    good = not missing and v["reading"] == f"{COST}; lr 1e-3 (m = 32): {COST}"
    good &= v["joint_kind_lr1e-3"] == "same rate (1e-3)" and v["A2"]["exp"]["shown"] == f"{SLB} (3 RUNAWAY)"
    good &= v["joint_lr1e-3"]["abs"] == J_SEARCH and v["joint_lr1e-3"]["exp"] is None
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} a2_after_minimal_h2: {v['reading']!r}; H2 joint kind {v['joint_kind_lr1e-3']}; "
          f"A2 exp {v['A2']['exp']['shown']}; H2 joints {v['joint_lr1e-3']}")
    # Amendment 3 (MAJOR 1): the Holm family of block A2 is the 3 cells {abs, softplus, exp}; abs keeps raw p. Signed 6/8,
    # abs 1/8, softplus 0/8, exp 1/8, accuracies close (the permutation branch cannot fire): exp's Fisher p adjusted over
    # 3 cells is 0.0811 (no label), over 2 cells 0.041 (SIGNED LEARNS BETTER, the declared sensitivity line only)
    near = {("A2", a, 32): 0.98 for a in ("signed", "abs", "softplus", "exp")}
    root = os.path.join(tmp, "a2holm")
    _write(root, {**sp(), ("A2", "signed", 32): 6, ("A2", "abs", 32): 1, ("A2", "softplus", 32): 0,
                  ("A2", "exp", 32): 1}, {**lo_idx, **near}, cfg=A2C)
    R, _ = load(root, A2C)
    v = analyse(R, A2C, out=silent)
    ex3, ex2 = v["A2"]["exp"], v["A2_sensitivity_2cell"]["exp"]
    good = abs(ex3["fisher_p_adj"] - 0.0811) < 5e-4 and abs(ex2["fisher_p_adj"] - 0.0406) < 5e-4
    good &= ex3["label"] == UNM and ex2["label"] == SLB and v["A2"]["abs"]["label"] == SLB
    good &= abs(v["A2"]["abs"]["fisher_p_adj"] - v["A2"]["abs"]["fisher_p"]) < 1e-12      # abs at raw p
    good &= v["reading"] == f"{COST}; lr 1e-3 (m = 32): {COST}"
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} A2 Holm over 3 cells: exp adj {ex3['fisher_p_adj']:.4f} -> {ex3['label']}; "
          f"2-cell sensitivity adj {ex2['fisher_p_adj']:.4f} -> {ex2['label']}; abs {v['A2']['abs']['label']} at raw p")
    # Amendment 3 (MINOR 3): A2 gate failures (every H2 joint None), A2 UNMEASURED, A2 exp NON-NEGATIVE LEARNS BETTER
    a2cases = [("G0", {("A2", "signed", 32): 5}, {}, {}, f"{UNR} (signed SOLVED 5/8 < 6)"),
               ("GI", {("A2", "index", 32): 2}, {}, {}, f"{NNI} (index SOLVED 2/8"),
               ("UNM", {("A2", "abs", 32): 5}, {}, {}, UNM),
               ("NLB", {("A2", "signed", 32): 6, ("A2", "exp", 32): 8}, {("A2", "signed", 32): 0.60},
                {("A2", "signed", 32): 0.93}, None)]
    for nm, extra, lo_x, hi_x, want_suffix in a2cases:
        root = os.path.join(tmp, f"a2_{nm}")
        _write(root, {**sp(), **extra}, {**lo_idx, **lo_x}, hi=hi_x, cfg=A2C)
        R, _ = load(root, A2C)
        v = analyse(R, A2C, out=silent)
        suf = v["reading"].split("; lr 1e-3 (m = 32): ", 1)[1]
        if nm in ("G0", "GI"):
            good = suf.startswith(want_suffix) and all(x is None for x in v["joint_lr1e-3"].values())
            good &= all(x["label"].startswith(want_suffix[:12]) for x in v["A2"].values())
        elif nm == "UNM":
            good = suf == want_suffix and v["joint_lr1e-3"]["abs"] is None
        else:
            good = v["A2"]["exp"]["label"] == NLB and v["joint_lr1e-3"]["exp"] is None
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} A2 {nm}: suffix {suf!r}; A2 exp {v['A2']['exp']['shown']}; "
              f"H2 joints {v['joint_lr1e-3']}")
    # existence unreadable when fewer than 6 sources solved
    root = os.path.join(tmp, "h_unr")
    _write(root, sp(signed_32=5), lo_idx, cfg=MIN)
    R, _ = load(root, MIN)
    ex = analyse(R, MIN, out=silent)["existence"]["label"]
    good = all(x.startswith(UNR) for x in ex.values())
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} existence with n_s = 5 -> {ex}")
    # divergence: DIVERGED, never SOLVED, acc 0
    root = os.path.join(tmp, "div")
    s = sp()
    s[("div", "A", "abs", 32)] = True
    _write(root, s, lo_idx, cfg=MIN)
    R, _ = load(root, MIN)
    r = R[("A", "abs", 32, SEEDS[-1])]
    good = r["div"] and not r["solved"] and r["cls"] == "DIVERGED" and r["acc"] == 0.0
    good &= analyse(R, MIN, out=silent)["n_diverged"] == 1
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} diverged run handled")
    # RUNAWAY by the loss criterion (a post-warm-up epoch above 2.08) is detected
    L = _curve(False, EPOCHS, runaway=True)
    pth = os.path.join(run_dir(os.path.join(tmp, "div"), "A", "abs", "moves", 32, SEEDS[0]), "eval.json")
    e = json.load(open(pth))
    e["losses"], e["move_step_absmax"] = L, 3.0
    json.dump(e, open(pth, "w"))
    R, _ = load(os.path.join(tmp, "div"), MIN)
    good = R[("A", "abs", 32, SEEDS[0])]["runaway"] and not R[("A", "abs", 32, SEEDS[1])]["runaway"]
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} RUNAWAY by loss detected")
    # block B labels, incl. NO SIGN COST and a RUNAWAY suffix on exp
    root = os.path.join(tmp, "b")
    s = sp()
    s[("B", "abs", 32)] = 8
    s[("run", "B", "exp", 32)] = 3
    _write(root, s, lo_idx, cfg=F)
    R, _ = load(root, F)
    vb = analyse(R, F, out=silent)["B"]
    good = vb["abs"]["label"] == NSC and vb["exp"]["label"] == SLB and vb["exp"]["shown"] == f"{SLB} (3 RUNAWAY)"
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} block B labels {[(a, l['shown']) for a, l in vb.items()]}")
    # recipe guard: a wrong lr must raise
    root = os.path.join(tmp, "recipe")
    _write(root, sp(), lo_idx, cfg=MIN)
    p = os.path.join(run_dir(root, "A", "abs", "moves", 32, SEEDS[0]), "eval.json")
    e = json.load(open(p))
    e["args"]["lr"] = 1e-3                     # block A is registered at 3e-3
    json.dump(e, open(p, "w"))
    try:
        load(root, MIN)
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
    ap.add_argument("--config", choices=sorted(CONFIGS), default=None,
                    help="the launch-time configuration (required for a real reading)")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=f"{REPO}/G3_SSM_VERDICTS.json")
    ap.add_argument("--no-h", action="store_true", help="read stage 1 only (descriptive)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.config is None:
        ap.error("--config is required")
    cfg = CONFIGS[a.config]
    R, missing = load(a.root, cfg, need_h=not a.no_h)
    if missing:
        print(f"MISSING {len(missing)} runs, e.g. {missing[:3]}")
        sys.exit(2)
    v = analyse(R, cfg)
    v["config_name"] = a.config
    json.dump(v, open(a.out, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()

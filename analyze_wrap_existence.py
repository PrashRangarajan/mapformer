"""Registered analysis for WRAP_EXISTENCE_PREREG.md (P0, E1, E2, E3; then declared secondaries). CPU.
  python3 -m mapformer.analyze_wrap_existence                                   # the batch (seeds 600-607)
  python3 -m mapformer.analyze_wrap_existence --root runs/wrap_existence_pilot --seeds 690 --no-json   # pilot / smoke
"""
import argparse, json

import numpy as np
import torch

from mapformer import rescore_hook
from mapformer.environment_cancel import GridWorldCancel
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4, wrap_share
from mapformer.stats_core import perm2_p, signflip_p, fisher_solved, classify_run
from mapformer.train_cancel import evaluate

REPO = "/home/prashr/mapformer"
CLS = {"SO": MapFormerWM_SignedAO_r4, "SO_cont": MapFormerWM_SignedAO_r4, "AO": MapFormerWM_AbsAO_r4,
       "AO_wrap": MapFormerWM_AbsAO_r4}
FLOOR = 0.554                                                                    # CANCEL_RESULTS.md, p_plus 0.5


def load(root, arm, s):
    d = f"{root}/{arm}_s{s}"
    ev = json.load(open(f"{d}/eval.json")); b = torch.load(f"{d}/{arm}.pt", map_location="cpu", weights_only=False)
    m = CLS[arm](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32,
                 action_ids=(b["config"]["action_offset"], b["config"]["action_offset"] + 1))
    m.load_state_dict(b["model_state_dict"])
    return {"ev": ev, "losses": b["losses"], "model": m, "solved": classify_run(b["losses"])["registered"] == "SOLVED",
            "cls": classify_run(b["losses"])["cls"], "acc": ev["eval"]["128"]["acc"], "acc512": ev["eval"]["512"]["acc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/wrap_existence/p0")
    ap.add_argument("--seeds", default="600-607")
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1)); n = len(seeds)
    R = {(arm, s): load(a.root, arm, s) for arm in CLS for s in seeds}
    V = {}

    print("== per seed: accuracy T = 128 (eval mode) / class ==")
    print("seed | SO | AO | SO_cont | AO_wrap frozen -> trained")
    for s in seeds:
        w = R["AO_wrap", s]["ev"]
        print(f"{s} | {R['SO', s]['acc']:.4f} {R['SO', s]['cls']} | {R['AO', s]['acc']:.4f} {R['AO', s]['cls']} | "
              f"{R['SO_cont', s]['acc']:.4f} {R['SO_cont', s]['cls']} | {w['frozen_acc']:.4f} -> {R['AO_wrap', s]['acc']:.4f} "
              f"{R['AO_wrap', s]['cls']}")
    print(f"floor (CANCEL) {FLOOR}")

    # P0
    src = [s for s in seeds if R["SO", s]["solved"]]; ns = len(src)
    print(f"\nP0: SO SOLVED {ns}/{n} -> E1 / E2 read on seeds {src}")
    V["P0"] = {"n_solved_sources": ns, "sources": src}

    # E1
    if ns == 0:
        V["E1"] = "UNREADABLE (no solved source)"
    else:
        bad = []
        for s in src:
            w = R["AO_wrap", s]["ev"]
            gap = abs(w["frozen_acc"] - w["source_acc"]); err = w["construction"]["angle_err_max"]
            if not (gap <= 0.002 and err < 1e-2):
                bad.append((s, gap, err))
        V["E1"] = "EXISTS" if not bad else f"CONSTRUCTION FAILED {bad}"
        gaps = [abs(R['AO_wrap', s]['ev']['frozen_acc'] - R['AO_wrap', s]['ev']['source_acc']) for s in src]
        errs = [R['AO_wrap', s]['ev']['construction']['angle_err_max'] for s in src]
        print(f"E1: {V['E1']} -- max |frozen - source| {max(gaps):.4f}, max angle error {max(errs):.1e} rad")

    # E2
    if ns < 4:
        V["E2"] = f"UNREADABLE (n_s = {ns} < 4)"
        print(f"E2: {V['E2']}")
    else:
        aw = np.array([R["AO_wrap", s]["acc"] for s in src]); sc = np.array([R["SO_cont", s]["acc"] for s in src])
        k_aw = sum(R["AO_wrap", s]["solved"] for s in src); k_sc = sum(R["SO_cont", s]["solved"] for s in src)
        pf = fisher_solved(k_aw, ns, k_sc, ns); d = aw - sc; sf = signflip_p(d)
        if k_aw >= ns - 1 and aw.mean() >= sc.mean() - 0.01:
            lab = "HELD"
        elif (pf < 0.05 and k_sc > k_aw) or (sf["p"] < 0.05 and d.mean() <= -0.05):
            lab = "NOT HELD"
        else:
            lab = "PARTIAL"
        V["E2"] = lab
        print(f"E2: {lab} -- AO_wrap {aw.mean():.4f} SOLVED {k_aw}/{ns} vs SO_cont {sc.mean():.4f} SOLVED {k_sc}/{ns}; "
              f"paired d {d.mean():+.4f} (sign-flip p {sf['p']:.4f}, min p {sf['min_p']:.4f}); Fisher p {pf:.4f}")

    # E3
    so = np.array([R["SO", s]["acc"] for s in seeds]); ao = np.array([R["AO", s]["acc"] for s in seeds])
    k_so = sum(R["SO", s]["solved"] for s in seeds); k_ao = sum(R["AO", s]["solved"] for s in seeds)
    pf3 = fisher_solved(k_ao, n, k_so, n); d3 = ao.mean() - so.mean()
    pp3 = perm2_p(so, ao)["p"] if n > 1 else float("nan")
    if (pf3 < 0.05 and k_so > k_ao) or (pp3 < 0.05 and d3 <= -0.05):
        lab3 = "SIGNED LEARNS BETTER"
    elif k_ao >= k_so - 1 and ao.mean() >= so.mean() - 0.01:
        lab3 = "NO SIGN COST"
    else:
        lab3 = "UNMEASURED"
    V["E3"] = lab3
    print(f"E3: {lab3} -- AO {ao.mean():.4f} SOLVED {k_ao}/{n} vs SO {so.mean():.4f} SOLVED {k_so}/{n}; d {d3:+.4f} "
          f"(perm p {pp3:.4f}); Fisher p {pf3:.4f}")

    # secondaries
    print("\n== secondaries (declared; no verdict) ==")
    for arm in CLS:
        ws = [wrap_share(R[arm, s]["model"]) for s in seeds]
        print(f"wrap share {arm:8s}: median |wrap| {np.nanmedian([w[0] for w in ws]):.3f} rad, share < 0.1 rad "
              f"{np.nanmean([w[1] for w in ws]):.2f} (per seed {' '.join(f'{w[1]:.2f}' for w in ws)})")
    fw = [R["AO_wrap", s]["ev"]["frozen_wrap"][1] for s in seeds]
    print(f"AO_wrap wrap share at construction {np.mean(fw):.2f} -> after training "
          f"{np.mean([wrap_share(R['AO_wrap', s]['model'])[1] for s in seeds]):.2f}")
    for s in seeds:
        c = R["AO_wrap", s]["ev"]["construction"]
        print(f"  s{s}: |W_out| max signed {c['w_out_absmax_signed']:.2f} wrapped {c['w_out_absmax_wrapped']:.2f}; "
              f"|step| max signed {c['step_absmax_signed']:.2f} wrapped {c['step_absmax_wrapped']:.2f}")
    for arm in CLS:
        print(f"acc T = 512 (OOD) {arm:8s}: {np.mean([R[arm, s]['acc512'] for s in seeds]):.4f}")
    auc = np.array([np.mean(R["AO", s]["losses"]) - np.mean(R["SO", s]["losses"]) for s in seeds])
    print(f"loss AUC AO - SO (scratch, paired): {auc.mean():+.4f} (sign-flip p {signflip_p(auc)['p']:.4f}; "
          f"AO slower on {int((auc > 0).sum())}/{n})")
    print("cross-batch, labelled: RING_DIAL Abs_r4 (with observation steps) SOLVED 2/8 at p_plus 0.5; "
          f"AO here {k_ao}/{n}")
    rescore_hook.install("auto")
    te = GridWorldCancel(size=32, seed=10000, p_plus=0.5)
    for arm in CLS:
        rs = [evaluate(R[arm, s]["model"], te, 128, 400, "cpu", seed=10 ** 6)[0] for s in seeds]
        print(f"dropout re-score (1/(1-p)) {arm:8s}: {np.mean(rs):.4f}")
    if not a.no_json:
        json.dump(V, open(f"{REPO}/WRAP_EXISTENCE_VERDICTS.json", "w"), indent=2)


if __name__ == "__main__":
    main()

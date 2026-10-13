"""Registered analysis for WRAP_EXISTENCE_PREREG.md + Amendment 1 (P0, E1, E2 per construction, E3; then declared
secondaries). CPU.
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
CLS = {"SO": MapFormerWM_SignedAO_r4, "AO": MapFormerWM_AbsAO_r4, "SO_cont": MapFormerWM_SignedAO_r4,
       "SO_bal": MapFormerWM_SignedAO_r4, "AO_bal": MapFormerWM_AbsAO_r4, "AO_wrap": MapFormerWM_AbsAO_r4,
       "AO_wrapg": MapFormerWM_AbsAO_r4}
BUILT = ("SO_bal", "AO_bal", "AO_wrap", "AO_wrapg")
# E2: (constructed arm, its control, registered status)
E2_SPECS = [("AO_bal", "SO_bal", "PRIMARY"), ("AO_wrap", "SO_cont", "descriptive"), ("AO_wrapg", "SO_cont", "descriptive")]
FLOOR = 0.554                                                                    # CANCEL_RESULTS.md, p_plus 0.5
LOST_ABS, LOST_REL, LOST_MIN = 0.3, 10.0, 0.05     # Amendment 1


def load(root, arm, s):
    d = f"{root}/{arm}_s{s}"
    ev = json.load(open(f"{d}/eval.json")); b = torch.load(f"{d}/{arm}.pt", map_location="cpu", weights_only=False)
    m = CLS[arm](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32,
                 action_ids=(b["config"]["action_offset"], b["config"]["action_offset"] + 1))
    m.load_state_dict(b["model_state_dict"])
    c = classify_run(b["losses"])
    return {"ev": ev, "losses": np.asarray(b["losses"], float), "model": m, "solved": c["registered"] == "SOLVED",
            "cls": c["cls"], "acc": ev["eval"]["128"]["acc"], "acc512": ev["eval"]["512"]["acc"]}


def e1_ok(w):
    """Amendment 1: E1 on one source -- frozen accuracy within 0.002 of the source and max action-angle error < 1e-2 rad."""
    return abs(w["frozen_acc"] - w["source_acc"]) <= 0.002 and w["construction"]["angle_err_max"] < 1e-2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/wrap_existence/p0")
    ap.add_argument("--seeds", default="600-607")
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1)); n = len(seeds)
    R = {(arm, s): load(a.root, arm, s) for arm in CLS for s in seeds}
    V = {}

    print("== per seed: accuracy T = 128 (eval mode) and class; constructed arms frozen -> trained ==")
    print("seed | SO | AO | SO_cont | " + " | ".join(BUILT))
    for s in seeds:
        cells = [f"{R[x, s]['acc']:.4f} {R[x, s]['cls']}" for x in ("SO", "AO", "SO_cont")]
        cells += [f"{R[x, s]['ev']['frozen_acc']:.4f} -> {R[x, s]['acc']:.4f} {R[x, s]['cls']} "
                  f"(max loss {R[x, s]['losses'].max():.3f})" for x in BUILT]
        print(f"{s} | " + " | ".join(cells))
    mx = [R["SO_cont", s]["losses"].max() for s in seeds]
    print("max per-epoch loss SO_cont: " + " ".join(f"{v:.3f}" for v in mx))
    print(f"floor (CANCEL) {FLOOR}")

    # P0
    src = [s for s in seeds if R["SO", s]["solved"]]; ns = len(src)
    print(f"\nP0: SO SOLVED {ns}/{n} -> E1 / E2 read on seeds {src}")
    V["P0"] = {"n_solved_sources": ns, "sources": src}

    # E1, per construction
    V["E1"] = {}
    for x in BUILT:
        if ns == 0:
            V["E1"][x] = "UNREADABLE (no solved source)"
        else:
            bad = [s for s in src if not e1_ok(R[x, s]["ev"])]
            V["E1"][x] = "EXISTS" if not bad else f"CONSTRUCTION FAILED on {bad}"
            gaps = [abs(R[x, s]['ev']['frozen_acc'] - R[x, s]['ev']['source_acc']) for s in src]
            errs = [R[x, s]['ev']['construction']['angle_err_max'] for s in src]
            lds = [R[x, s]['ev']['logit_diff_max'] for s in src]
            print(f"E1 [{x}]: {V['E1'][x]} -- max |frozen - source| {max(gaps):.4f}, max angle error {max(errs):.1e} rad, "
                  f"max logit diff T=128 {max(l['128'] for l in lds):.1e} / T=512 {max(l['512'] for l in lds):.1e}")
    if ns == 0:
        print(f"E1: {V['E1']}")

    # E2, per construction (primary: AO_bal vs SO_bal), gated on that construction's E1 and on n_s >= 4
    V["E2"] = {}
    for x, ctl, status in E2_SPECS:
        if ns < 4:
            lab = f"UNREADABLE (n_s = {ns} < 4)"
        elif V["E1"][x] != "EXISTS" or (ctl in BUILT and V["E1"][ctl] != "EXISTS"):
            lab = "UNREADABLE (E1 not EXISTS for this construction or its control)"
        else:
            aw = np.array([R[x, s]["acc"] for s in src]); sc = np.array([R[ctl, s]["acc"] for s in src])
            k_aw = sum(R[x, s]["solved"] for s in src); k_sc = sum(R[ctl, s]["solved"] for s in src)
            lost = lambda arm, s: (R[arm, s]["losses"].max() > LOST_ABS or
                                   (R[arm, s]["losses"].max() > LOST_REL * R[ctl, s]["losses"].max()
                                    and R[arm, s]["losses"].max() > LOST_MIN))
            l_aw = sum(lost(x, s) for s in src); l_sc = sum(lost(ctl, s) for s in src)
            pf = fisher_solved(k_aw, ns, k_sc, ns); pl = fisher_solved(ns - l_aw, ns, ns - l_sc, ns)
            d = aw - sc; sf = signflip_p(d)
            if k_aw >= ns - 1 and aw.mean() >= sc.mean() - 0.01 and l_aw <= 1:
                lab = "HELD"
            elif ((pf < 0.05 and k_sc > k_aw) or (sf["p"] < 0.05 and d.mean() <= -0.05)
                  or (pl < 0.05 and l_aw > l_sc)):
                lab = "NOT HELD"
            else:
                lab = "PARTIAL"
            print(f"E2 [{x} vs {ctl}, {status}]: {lab} -- {x} {aw.mean():.4f} SOLVED {k_aw}/{ns} LOST {l_aw}/{ns} vs {ctl} "
                  f"{sc.mean():.4f} SOLVED {k_sc}/{ns} LOST {l_sc}/{ns}; paired d {d.mean():+.4f} (sign-flip p "
                  f"{sf['p']:.4f}, min p {sf['min_p']:.4f}); Fisher SOLVED p {pf:.4f}, Fisher not-LOST p {pl:.4f}")
        V["E2"][x] = {"label": lab, "control": ctl, "status": status}
        if lab.startswith("UNREADABLE"):
            print(f"E2 [{x} vs {ctl}, {status}]: {lab}")

    # E3, gated on SO SOLVED >= 4
    so = np.array([R["SO", s]["acc"] for s in seeds]); ao = np.array([R["AO", s]["acc"] for s in seeds])
    k_so = sum(R["SO", s]["solved"] for s in seeds); k_ao = sum(R["AO", s]["solved"] for s in seeds)
    pf3 = fisher_solved(k_ao, n, k_so, n); d3 = ao.mean() - so.mean()
    pp3 = perm2_p(so, ao)["p"] if n > 1 else float("nan")
    if k_so < 4:
        lab3 = f"UNREADABLE (SO SOLVED {k_so} < 4)"
    elif (pf3 < 0.05 and k_so > k_ao) or (pp3 < 0.05 and d3 <= -0.05):
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
    for x in BUILT:
        print(f"{x} wrap share at construction {np.nanmean([R[x, s]['ev']['frozen_wrap'][1] for s in seeds]):.2f} -> "
              f"after training {np.nanmean([wrap_share(R[x, s]['model'])[1] for s in seeds]):.2f}")
    for x in BUILT:
        for s in seeds:
            c = R[x, s]["ev"]["construction"]
            print(f"  {x} s{s}: |W_out| max signed {c['w_out_absmax_signed']:.2f} built {c['w_out_absmax_built']:.2f}; "
                  f"|W_in| {c['w_in_absmax_signed']:.2f} -> {c['w_in_absmax_built']:.2f}; |step| max {c['step_absmax_signed']:.2f}"
                  f" -> {c['step_absmax_built']:.2f}; |omega| max {c['omega_absmax_signed']:.2f} -> {c['omega_absmax_built']:.2f};"
                  f" omega < 0: {c['n_omega_negative']}; latent cos {c['latent_cos_signed']:+.3f}")
    # angle distance from the construction after training (wrap-aware, action angles)
    from mapformer.model_wrap_exist import action_angles
    from mapformer.train_wrap_exist import load_so, model_kw
    from mapformer.model_wrap_exist import wrap_construct
    for x in BUILT:
        ds = []
        for s in src:
            env = GridWorldCancel(size=32, seed=s, p_plus=0.5); kw = model_kw(env, 32)
            m0, _ = wrap_construct(load_so(f"{a.root}/SO_s{s}/SO.pt", env, 32), kw.pop("vocab_size"), kw.pop("action_ids"),
                                   mode=dict(SO_bal="signed_balanced", AO_bal="balanced", AO_wrap="min", AO_wrapg="gauge")[x], **kw)
            ds.append(float(torch.angle(torch.exp(1j * (action_angles(R[x, s]["model"]) - action_angles(m0)))).abs().max()))
        if ds:
            print(f"{x}: max wrap-aware action-angle change during training, per solved source: {' '.join(f'{v:.3f}' for v in ds)} rad")
    for arm in CLS:
        print(f"acc T = 512 (OOD) {arm:8s}: {np.mean([R[arm, s]['acc512'] for s in seeds]):.4f}")
    auc = np.array([np.mean(R["AO", s]["losses"]) - np.mean(R["SO", s]["losses"]) for s in seeds])
    print(f"loss AUC AO - SO (scratch, paired): {auc.mean():+.4f} (sign-flip p {signflip_p(auc)['p']:.4f}; "
          f"AO slower on {int((auc > 0).sum())}/{n})")
    print("cross-batch, labelled: RING_DIAL Abs_r4 (with observation steps) SOLVED 2/8 at p_plus 0.5; "
          f"AO here {k_ao}/{n}")
    # frozen perturbation sensitivity (Amendment 1 secondary): omega or W_out x (1 + eps N(0,1)), 3 draws, 200 trajectories
    te = GridWorldCancel(size=32, seed=10000, p_plus=0.5)
    for x in ("SO_bal", "AO_bal"):
        for pname in ("path_integrator.omega", "action_to_lie.w_out.weight"):
            for eps in (1e-3, 1e-2):
                accs = []
                for s in src:
                    env = GridWorldCancel(size=32, seed=s, p_plus=0.5); kw = model_kw(env, 32)
                    m0, _ = wrap_construct(load_so(f"{a.root}/SO_s{s}/SO.pt", env, 32), kw.pop("vocab_size"),
                                           kw.pop("action_ids"), mode=dict(SO_bal="signed_balanced", AO_bal="balanced")[x], **kw)
                    P = dict(m0.named_parameters())[pname]; P0 = P.data.clone(); g = torch.Generator().manual_seed(s)
                    for _ in range(3):
                        P.data = P0 * (1 + eps * torch.randn(P0.shape, generator=g))
                        accs.append(evaluate(m0, te, 128, 200, "cpu", seed=10 ** 6)[0])
                    P.data = P0
                if accs:
                    print(f"sensitivity {x} frozen, {pname} x (1 + {eps:g} N): mean acc {np.mean(accs):.4f} (min {np.min(accs):.4f})")
    rescore_hook.install("auto")
    for arm in CLS:
        rs = [evaluate(R[arm, s]["model"], te, 128, 400, "cpu", seed=10 ** 6)[0] for s in seeds]
        print(f"dropout re-score (1/(1-p)) {arm:8s}: {np.mean(rs):.4f}")
    if not a.no_json:
        json.dump(V, open(f"{REPO}/WRAP_EXISTENCE_VERDICTS.json", "w"), indent=2)


if __name__ == "__main__":
    main()

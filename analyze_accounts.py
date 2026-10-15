"""Registered analysis for ACCOUNTS_PREREG.md: T1 (D1, D2), T2 (F1, F2, reading), T4 (L1, L2); T3 (toy) summarised by
`runs/accounts/t3/summarize_t3.py`. CPU.
  python3 -m mapformer.analyze_accounts [--root runs/accounts] [--no-json]
"""
import argparse, json

import numpy as np
import torch

from mapformer.analyze_wrap_mech import compare as mech_compare
from mapformer.environment_cancel import GridWorldCancel
from mapformer.model_wrap_exist import (action_angles, usage_weights, wrap_share_weighted, wrap_construct,
                                        MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4)
from mapformer.train_wrap_exist import load_so, model_kw
from mapformer.stats_core import perm2_p, signflip_p, fisher_solved, classify_run

REPO = "/home/prashr/mapformer"
T1_SEEDS, T2_SEEDS, T4_SEEDS = range(600, 608), range(1000, 1008), range(1100, 1108)
T4_COPE_SEEDS = range(1100, 1116)                                    # Amendment 1: 16 seeds for the two CoPE arms


def fl(vals, f="{:.2f}"):
    return " ".join(f.format(v) for v in vals)


def t1(root, V):
    print("== T1: weight decay and the drift off the wrapped code (ring continuations) ==")
    global TOK
    te = GridWorldCancel(size=32, seed=10000, p_plus=0.5); np.random.seed(10 ** 6)
    TOK = torch.stack([te.generate_trajectory(128)[0] for _ in range(32)])
    R = {}
    for arm in ("AO_bal_wd", "AO_bal_wd0", "SO_bal_wd", "SO_bal_wd0"):
        for s in T1_SEEDS:
            d = f"{root}/t1/{arm}_s{s}"; base = arm.rsplit("_", 1)[0]
            ev = json.load(open(f"{d}/eval.json")); b = torch.load(f"{d}/{base}.pt", map_location="cpu", weights_only=False)
            cls = MapFormerWM_AbsAO_r4 if arm.startswith("AO") else MapFormerWM_SignedAO_r4
            m = cls(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(0, 1))
            m.load_state_dict(b["model_state_dict"])
            env = GridWorldCancel(size=32, seed=s, p_plus=0.5); kw = model_kw(env, 32)
            m0, _ = wrap_construct(load_so(f"{REPO}/runs/wrap_existence/p0/SO_s{s}/SO.pt", env, 32), kw.pop("vocab_size"),
                                   kw.pop("action_ids"), mode="balanced" if arm.startswith("AO") else "signed_balanced", **kw)
            ww = wrap_share_weighted(m, usage_weights(m, TOK)); ww0 = wrap_share_weighted(m0, usage_weights(m0, TOK))
            R[arm, s] = {"wrap": ev["wrap"][1], "wrap0": ev["frozen_wrap"][1], "wrapw": ww, "wrapw0": ww0,
                         "acc": ev["eval"]["128"]["acc"],
                         "acc4": ev["eval"]["512"]["acc"], "solved": classify_run(b["losses"])["registered"] == "SOLVED",
                         "maxloss": float(np.max(b["losses"]))}
    for arm in ("AO_bal_wd", "AO_bal_wd0", "SO_bal_wd", "SO_bal_wd0"):
        r = [R[arm, s] for s in T1_SEEDS]
        w0, w1 = [x["wrap0"] for x in r], [x["wrap"] for x in r]
        v0, v1 = [x["wrapw0"] for x in r], [x["wrapw"] for x in r]
        print(f"{arm:11s}: usage-weighted wrap share {np.mean(v0):.2f} -> {np.mean(v1):.2f} (per seed {fl(v1)}); unweighted "
              f"{np.mean(w0):.2f} -> {np.mean(w1):.2f}; acc "
              f"{np.mean([x['acc'] for x in r]):.4f}, 4T {np.mean([x['acc4'] for x in r]):.4f}; SOLVED "
              f"{sum(x['solved'] for x in r)}/8; max loss {fl([x['maxloss'] for x in r], '{:.3f}')}")
    for key, x, ref, fld in (("D1", "AO_bal_wd0", "AO_bal_wd", "wrapw"), ("D2", "SO_bal_wd0", "SO_bal_wd", "wrapw"),
                             ("D1u (unweighted, secondary)", "AO_bal_wd0", "AO_bal_wd", "wrap"),
                             ("D2u (unweighted, secondary)", "SO_bal_wd0", "SO_bal_wd", "wrap")):
        d = np.array([R[x, s][fld] - R[ref, s][fld] for s in T1_SEEDS]); sf = signflip_p(d)
        if d.mean() >= 0.30 and sf["p"] < 0.05:
            lab = "WEIGHT DECAY DRIVES THE DRIFT"
        elif abs(d.mean()) < 0.10:
            lab = "NO EFFECT DETECTED"
        else:
            lab = "UNMEASURED"
        V[key] = lab
        print(f"{key} ({x} - {ref}, final {fld} share, paired): {lab} -- mean {d.mean():+.3f} (sign-flip p {sf['p']:.4f}, "
              f"min p {sf['min_p']:.4f}); per seed {fl(d, '{:+.2f}')}")


def t2(root, V):
    print("\n== T2: is the signed escape from the |.| start collective? (frozen move latents) ==")
    seeds = list(T2_SEEDS)
    def load2(x, s):
        d = f"{root}/t2/{x}_s{s}"; ev = json.load(open(f"{d}/eval.json"))
        b = torch.load(f"{d}/{x}.pt", map_location="cpu", weights_only=False)
        m = MapFormerWM_SignedAO_r4(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32,
                                    action_ids=(b["config"]["action_offset"], b["config"]["action_offset"] + 1))
        if x.endswith("_fz"):
            m.freeze_move_latents()                         # creates the buffer the checkpoint carries
        m.load_state_dict(b["model_state_dict"]); c = classify_run(b["losses"])
        return {"model": m, "acc": ev["eval"]["128"]["acc"], "solved": c["registered"] == "SOLVED", "cls": c["cls"]}
    R = {(x, s): load2(x, s) for x in ("SO_same", "SO_absmag", "SO_same_fz", "SO_absmag_fz") for s in seeds}
    for x in ("SO_same", "SO_absmag", "SO_same_fz", "SO_absmag_fz"):
        opp = []
        for s in seeds:
            A = action_angles(R[x, s]["model"]).double().numpy()
            sel = np.abs(np.angle(np.exp(1j * A[0]))) > 0.05
            opp.append(float((np.sign(A[0]) != np.sign(A[1]))[sel].mean()) if sel.any() else float("nan"))
        print(f"{x:13s}: mean {np.mean([R[x, s]['acc'] for s in seeds]):.4f}, SOLVED {sum(R[x, s]['solved'] for s in seeds)}/8; "
              f"opposite-sign channels at the end {np.nanmean(opp):.2f} (per seed {fl(opp)})")
    k_same = sum(R["SO_same", s]["solved"] for s in seeds)
    labs = {}
    for key, x, ref in (("F1", "SO_absmag_fz", "SO_absmag"), ("F2", "SO_same_fz", "SO_same")):
        lab, txt, _ = mech_compare(R, seeds, x, ref)
        if k_same < 4:
            lab = f"UNREADABLE (SO_same SOLVED {k_same} < 4)"
        labs[key] = lab; V[key] = lab
        print(f"{key} ({x} vs {ref}): {lab} -- {txt}")
    if k_same < 4:
        V["T2 reading"] = "UNREADABLE"
    elif labs["F1"] == "WORSE" and labs["F2"] != "WORSE":
        V["T2 reading"] = "SUPPORTED"
    elif labs["F1"] == "NO EFFECT DETECTED":
        V["T2 reading"] = "NOT SUPPORTED"
    else:
        V["T2 reading"] = "AMBIGUOUS"
    print(f"T2 reading (collective-route account): {V['T2 reading']}")


def t4(root, V):
    print("\n== T4: CoPE on a line that never wraps ==")
    from mapformer.environment_cancel import GridWorldCancel
    R = {}
    SEEDS = {"CoPEF_line": T4_COPE_SEEDS, "CoPEF_ring": T4_COPE_SEEDS, "Signed_r4_line": T4_SEEDS, "RoPE_line": T4_SEEDS}
    for arm in ("CoPEF_line", "Signed_r4_line", "RoPE_line", "CoPEF_ring"):
        model_arm = arm.rsplit("_", 1)[0]
        for s in SEEDS[arm]:
            d = f"{root}/t4/{arm}_s{s}"
            ev = json.load(open(f"{d}/eval.json")); b = torch.load(f"{d}/{model_arm}.pt", map_location="cpu", weights_only=False)
            R[arm, s] = {"acc": ev["eval"]["128"]["acc"], "acc4": ev["eval"]["512"]["acc"],
                         "solved": classify_run(b["losses"])["registered"] == "SOLVED", "cls": classify_run(b["losses"])["cls"]}
    for arm in ("CoPEF_line", "Signed_r4_line", "RoPE_line", "CoPEF_ring"):
        accs = [R[arm, s]["acc"] for s in SEEDS[arm]]
        print(f"{arm:15s}: acc {np.mean(accs):.4f} (per seed {fl(accs, '{:.3f}')}), SOLVED "
              f"{sum(R[arm, s]['solved'] for s in SEEDS[arm])}/{len(SEEDS[arm])}")
    nc, n = len(T4_COPE_SEEDS), len(T4_SEEDS)
    kl = sum(R["CoPEF_line", s]["solved"] for s in T4_COPE_SEEDS); kr = sum(R["CoPEF_ring", s]["solved"] for s in T4_COPE_SEEDS)
    pf = fisher_solved(kr, nc, kl, nc)
    V["L1"] = ("LINE EASIER" if (pf < 0.05 and kl > kr) else ("RING EASIER" if (pf < 0.05 and kr > kl) else
               ("NO EFFECT DETECTED" if abs(kl - kr) <= 2 else "UNMEASURED")))
    print(f"L1 (CoPEF line vs CoPEF ring, SOLVED, n = {nc} each): {V['L1']} -- {kl}/{nc} vs {kr}/{nc} (Fisher p {pf:.4f})")
    a = np.array([R["CoPEF_line", s]["acc"] for s in T4_COPE_SEEDS]); r = np.array([R["Signed_r4_line", s]["acc"] for s in T4_SEEDS])
    ks = sum(R["Signed_r4_line", s]["solved"] for s in T4_SEEDS); pf2 = fisher_solved(kl, nc, ks, n); pp = perm2_p(r, a)["p"]
    d = a.mean() - r.mean()
    if ks < 4:
        V["L2"] = f"UNREADABLE (Signed_r4 line SOLVED {ks} < 4)"
    elif (pf2 < 0.05 and ks / n > kl / nc) or (pp < 0.05 and d <= -0.05):
        V["L2"] = "SIGNED LEARNS BETTER"
    elif (pf2 < 0.05 and kl / nc > ks / n) or (pp < 0.05 and d >= 0.05):
        V["L2"] = "COPE BETTER"
    elif kl / nc >= ks / n - 1 / n and a.mean() >= r.mean() - 0.01:
        V["L2"] = "NO SIGN COST"
    else:
        V["L2"] = "UNMEASURED"
    print(f"L2 (CoPEF line vs Signed_r4 line): {V['L2']} -- {a.mean():.4f} {kl}/{nc} vs {r.mean():.4f} {ks}/{n}; d {d:+.4f} "
          f"(perm p {pp:.4f}); Fisher p {pf2:.4f}")
    te = GridWorldCancel(size=256, seed=10000, p_plus=0.5); np.random.seed(10 ** 6); tot = blank = 0
    for _ in range(400):
        tok, _, rev = te.generate_trajectory(128); m = rev[1:]; tgt = tok[1:][m]
        tot += int(m.sum()); blank += int((tgt == te.unified_blank).sum())
    print(f"secondary: line revisit targets that are blank {blank / max(tot, 1):.3f} (predict-blank floor)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/accounts")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--parts", default="t1,t2,t4")
    a = ap.parse_args()
    V = {}
    for part in a.parts.split(","):
        {"t1": t1, "t2": t2, "t4": t4}[part](a.root, V)
    if not a.no_json:
        json.dump(V, open(f"{REPO}/ACCOUNTS_VERDICTS.json", "w"), indent=2)


if __name__ == "__main__":
    main()

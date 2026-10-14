"""Registered analysis for MECHANISM_PREREG.md + Amendment 1 (M1-M6, then declared secondaries). CPU.
  python3 -m mapformer.analyze_wrap_mech [--root runs/wrap_mech/p0] [--seeds 800-807] [--no-json]
"""
import argparse, json, os

import numpy as np
import torch

from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4, wrap_share, action_angles
from mapformer.stats_core import perm2_p, perm2_ci, signflip_p, fisher_solved, classify_run

REPO = "/home/prashr/mapformer"
ARMS = ("SO", "AO", "SO_same", "SO_absmag", "AO_absmag", "AO_high", "SO_high", "AO_mid")


def load(root, arm, s):
    d = f"{root}/{arm}_s{s}"
    ev = json.load(open(f"{d}/eval.json")); b = torch.load(f"{d}/{arm}.pt", map_location="cpu", weights_only=False)
    cls = MapFormerWM_SignedAO_r4 if arm.startswith("SO") else MapFormerWM_AbsAO_r4
    m = cls(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32,
            action_ids=(b["config"]["action_offset"], b["config"]["action_offset"] + 1))
    m.load_state_dict(b["model_state_dict"])
    c = classify_run(b["losses"])
    return {"ev": ev, "losses": np.asarray(b["losses"], float), "model": m, "solved": c["registered"] == "SOLVED",
            "cls": c["cls"], "acc": ev["eval"]["128"]["acc"], "acc4": ev["eval"]["512"]["acc"]}


def compare(R, seeds, x, ref):
    """Amendment 1 labels: X BETTER / X WORSE (Fisher p < .05 in that direction, or perm p < .05 with |d| >= 0.05),
    NO EFFECT DETECTED (SOLVED within 1 and |mean d| <= 0.01), UNMEASURED otherwise; with the permutation CI of d and a
    paired sign-flip p (descriptive: the arms share weights, stream and dropout at a seed)."""
    n = len(seeds)
    a = np.array([R[x, s]["acc"] for s in seeds]); r = np.array([R[ref, s]["acc"] for s in seeds])
    ka = sum(R[x, s]["solved"] for s in seeds); kr = sum(R[ref, s]["solved"] for s in seeds)
    pf = fisher_solved(ka, n, kr, n); pp = perm2_p(r, a)["p"]; d = float(a.mean() - r.mean())
    ci = perm2_ci(r, a, step=0.005, grid=(-1.0, 1.0)); sf = signflip_p(a - r)["p"]
    if (pf < 0.05 and ka > kr) or (pp < 0.05 and d >= 0.05):
        lab = "BETTER"
    elif (pf < 0.05 and kr > ka) or (pp < 0.05 and d <= -0.05):
        lab = "WORSE"
    elif abs(ka - kr) <= 1 and abs(d) <= 0.01:
        lab = "NO EFFECT DETECTED"
    else:
        lab = "UNMEASURED"
    txt = (f"{x} {a.mean():.4f} SOLVED {ka}/{n} vs {ref} {r.mean():.4f} SOLVED {kr}/{n}; d {d:+.4f} [95% CI {ci['lo']:+.3f}, "
           f"{ci['hi']:+.3f}] (perm p {pp:.4f}; paired sign-flip p {sf:.4f}, descriptive); Fisher p {pf:.4f}")
    return lab, txt, kr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/wrap_mech/p0")
    ap.add_argument("--seeds", default="800-807")
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1)); n = len(seeds)
    arms = [x for x in ARMS if os.path.exists(f"{a.root}/{x}_s{seeds[0]}/eval.json")]
    R = {(x, s): load(a.root, x, s) for x in arms for s in seeds}
    V = {}
    print("== per seed: accuracy T = 128 (eval mode) and class ==")
    print("seed | " + " | ".join(arms))
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{R[x, s]['acc']:.4f} {R[x, s]['cls'][:4]}" for x in arms))
    for x in arms:
        print(f"{x:10s}: mean {np.mean([R[x, s]['acc'] for s in seeds]):.4f}, SOLVED {sum(R[x, s]['solved'] for s in seeds)}/{n}")
    k_so = sum(R["SO", s]["solved"] for s in seeds)

    def reg(key, x, ref, names, gate=None):
        lab, txt, kr = compare(R, seeds, x, ref)
        lab = names.get(lab, lab)
        if gate is not None and not gate[0]:
            lab = f"UNREADABLE ({gate[1]})"
        V[key] = lab
        print(f"{key} ({x} vs {ref}): {lab} -- {txt}")

    print()
    k_same = sum(R["SO_same", s]["solved"] for s in seeds)
    reg("M1", "SO_absmag", "SO_same", {"WORSE": "HURTS", "BETTER": "HELPS"},
        gate=(k_same >= 4, f"SO_same SOLVED {k_same} < 4"))
    reg("M2", "AO_high", "AO_absmag", {"BETTER": "RESCUES", "WORSE": "WORSE"}, gate=(k_so >= 4, f"SO SOLVED {k_so} < 4"))
    if "AO_mid" in arms:
        reg("M2b", "AO_high", "AO_mid", {"BETTER": "SUM ABOVE PI BEATS A LARGE SUB-PI START", "WORSE": "WORSE"},
            gate=(k_so >= 4, f"SO SOLVED {k_so} < 4"))
    for x, ref in (("SO_same", "SO"), ("AO_absmag", "AO")):
        lab, txt, _ = compare(R, seeds, x, ref)
        lab = "NO DIFFERENCE DETECTED" if lab in ("NO EFFECT DETECTED", "UNMEASURED") else f"DIFFERS ({lab})"
        V[f"M3 {x} vs {ref}"] = lab
        print(f"M3 ({x} vs {ref}): {lab} -- {txt}")
    lab, txt, kr = compare(R, seeds, "AO", "SO")
    V["M4"] = (f"UNREADABLE (SO SOLVED {k_so} < 4)" if k_so < 4 else
               {"WORSE": "SIGNED LEARNS BETTER", "NO EFFECT DETECTED": "NO SIGN COST"}.get(lab, lab))
    print(f"M4 (AO vs SO): {V['M4']} -- {txt}")
    reg("M5", "AO_absmag", "SO_absmag", {"WORSE": "SIGN COST AT THE |.| START", "BETTER": "|.| BETTER AT THE |.| START"})
    if "SO_high" in arms:
        reg("M6", "AO_high", "SO_high", {"WORSE": "SIGN COST AT THE LARGE START", "BETTER": "|.| BETTER AT THE LARGE START"})
    if not a.no_json:
        json.dump(V, open(f"{REPO}/MECHANISM_VERDICTS.json", "w"), indent=2)

    print("\n== secondaries (declared; no verdict) ==")
    for x in arms:
        ws = [wrap_share(R[x, s]["model"])[1] for s in seeds]
        print(f"wrap share {x:10s}: {np.nanmean(ws):.2f} (per seed {' '.join(f'{w:.2f}' for w in ws)}); acc 4T "
              f"{np.mean([R[x, s]['acc4'] for s in seeds]):.4f}; loss AUC {np.mean([R[x, s]['losses'].mean() for s in seeds]):.3f}")
    for x in [y for y in ("SO", "SO_same", "SO_absmag", "SO_high") if y in arms]:
        rows = []
        for s in seeds:
            A0 = np.array(R[x, s]["ev"]["init_angles"]); A1 = action_angles(R[x, s]["model"]).numpy()
            d0 = np.abs(np.angle(np.exp(1j * (A0[0] - A0[1])))); canc = np.abs(np.angle(np.exp(1j * (A1[0] + A1[1])))) < 0.1
            big = np.abs(np.angle(np.exp(1j * A1[0]))) > 0.05; med = np.median(d0)
            hi_, lo_ = big & (d0 > med), big & (d0 <= med)
            rows.append((med, canc[hi_].mean() if hi_.any() else np.nan, canc[lo_].mean() if lo_.any() else np.nan,
                         float((np.sign(A0[0]) != np.sign(A0[1])).mean())))
        print(f"{x:10s}: median initial |a(+1) - a(-1)| {np.mean([r[0] for r in rows]):.3f} rad; final cancelling share, large vs "
              f"small initial difference {np.nanmean([r[1] for r in rows]):.2f} vs {np.nanmean([r[2] for r in rows]):.2f}; "
              f"initial opposite-sign share per seed {' '.join(f'{r[3]:.2f}' for r in rows)}")
    if "SO" in arms:                                         # per-seed opposite-sign share at init vs loss AUC (signed default)
        opp = [float((np.sign(np.array(R['SO', s]['ev']['init_angles'])[0]) != np.sign(np.array(R['SO', s]['ev']['init_angles'])[1])).mean()) for s in seeds]
        auc = [R["SO", s]["losses"].mean() for s in seeds]
        print(f"SO: r(initial opposite-sign share, loss AUC) over seeds {np.corrcoef(opp, auc)[0, 1]:+.2f}")
    if "SO_high" in arms:                                    # does the |.| ever bind from the large start?
        neg = [float((action_angles(R["SO_high", s]["model"]) < 0).double().mean()) for s in seeds]
        print(f"SO_high: share of (move, channel) angles < 0 at the end {np.mean(neg):.3f} (0 = the signed model stayed non-negative)")
    from mapformer import rescore_hook
    from mapformer.environment_cancel import GridWorldCancel
    from mapformer.train_cancel import evaluate
    rescore_hook.install("auto"); te = GridWorldCancel(size=32, seed=10000, p_plus=0.5)
    for x in arms:
        rs = [evaluate(R[x, s]["model"], te, 128, 400, "cpu", seed=10 ** 6)[0] for s in seeds]
        print(f"dropout re-score (1/(1-p)) {x:10s}: {np.mean(rs):.4f} (eval mode {np.mean([R[x, s]['acc'] for s in seeds]):.4f})")


if __name__ == "__main__":
    main()

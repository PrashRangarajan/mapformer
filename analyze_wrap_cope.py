"""Registered analysis for WRAP_COPE_PREREG.md (W1-W3, then declared secondaries). CPU.
  python3 -m mapformer.analyze_wrap_cope [--root runs/wrap_cope/p0] [--seeds 900-907] [--no-json]
"""
import argparse, json, math

import numpy as np
import torch

from mapformer.stats_core import perm2_p, perm2_ci, fisher_solved, classify_run
from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"
ARMS = ("CoPEF", "CoPEF_Hi", "CoPEF_Lo", "CoPEF_LoS")
FLOOR = 0.554


def compare(R, seeds, x, ref):
    n = len(seeds)
    a = np.array([R[x, s]["acc"] for s in seeds]); r = np.array([R[ref, s]["acc"] for s in seeds])
    ka = sum(R[x, s]["solved"] for s in seeds); kr = sum(R[ref, s]["solved"] for s in seeds)
    pf = fisher_solved(ka, n, kr, n); pp = perm2_p(r, a)["p"]; d = float(a.mean() - r.mean())
    ci = perm2_ci(r, a, step=0.005, grid=(-1.0, 1.0))
    if (pf < 0.05 and ka > kr) or (pp < 0.05 and d >= 0.05):
        lab = "BETTER"
    elif (pf < 0.05 and kr > ka) or (pp < 0.05 and d <= -0.05):
        lab = "WORSE"
    elif abs(ka - kr) <= 1 and abs(d) <= 0.01:
        lab = "NO EFFECT DETECTED"
    else:
        lab = "UNMEASURED"
    return lab, (f"{x} {a.mean():.4f} SOLVED {ka}/{n} vs {ref} {r.mean():.4f} SOLVED {kr}/{n}; d {d:+.4f} [95% CI "
                 f"{ci['lo']:+.3f}, {ci['hi']:+.3f}] (perm p {pp:.4f}); Fisher p {pf:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/wrap_cope/p0")
    ap.add_argument("--seeds", default="900-907")
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1)); n = len(seeds)
    R = {}
    for x in ARMS:
        for s in seeds:
            d = f"{a.root}/{x}_s{s}"
            b = torch.load(f"{d}/{x}.pt", map_location="cpu", weights_only=False); ev = json.load(open(f"{d}/eval.json"))
            m = VARIANT_MAP[x](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32)
            m.load_state_dict(b["model_state_dict"]); c = classify_run(b["losses"])
            R[x, s] = {"model": m, "losses": np.asarray(b["losses"], float), "acc": ev["eval"]["128"]["acc"],
                       "acc4": ev["eval"]["512"]["acc"], "nll": ev["eval"]["128"]["nll"],
                       "solved": c["registered"] == "SOLVED", "cls": c["cls"]}
    print("== per seed: accuracy T = 128 (eval mode) and class ==")
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{x} {R[x, s]['acc']:.4f} {R[x, s]['cls'][:4]}" for x in ARMS))
    for x in ARMS:
        print(f"{x:9s}: mean {np.mean([R[x, s]['acc'] for s in seeds]):.4f}, SOLVED {sum(R[x, s]['solved'] for s in seeds)}/{n}")
    V = {}
    mean = lambda x: np.mean([R[x, s]["acc"] for s in seeds])
    for key, x, ref in (("W1", "CoPEF_Hi", "CoPEF_Lo"), ("W1b", "CoPEF_Hi", "CoPEF_LoS"), ("W2", "CoPEF_Hi", "CoPEF"),
                        ("W3", "CoPEF_Lo", "CoPEF")):
        lab, txt = compare(R, seeds, x, ref)
        if max(mean(x), mean(ref)) < FLOOR + 0.05:                     # Amendment 1: both arms at the floor
            lab = "UNREADABLE (both arms within 0.05 of the floor)"
        V[key] = lab
        print(f"{key} ({x} vs {ref}): {lab} -- {txt}")
    for key, x, ref in (("W1n", "CoPEF_Hi", "CoPEF_Lo"), ("W1bn", "CoPEF_Hi", "CoPEF_LoS")):   # Amendment 1: held-out NLL
        a_, r_ = np.array([R[x, s]["nll"] for s in seeds]), np.array([R[ref, s]["nll"] for s in seeds])
        pp = perm2_p(r_, a_)["p"]; d = float(a_.mean() - r_.mean())
        lab = "LOWER NLL" if (pp < 0.05 and d < 0) else ("HIGHER NLL" if (pp < 0.05 and d > 0) else "NO DIFFERENCE DETECTED")
        V[key] = lab
        print(f"{key} ({x} vs {ref}, held-out NLL at T = 128): {lab} -- {a_.mean():.4f} vs {r_.mean():.4f}, d {d:+.4f} (perm p {pp:.4f})")
    if not a.no_json:
        json.dump(V, open(f"{REPO}/WRAP_COPE_VERDICTS.json", "w"), indent=2)
    print("\n== secondaries (declared; no verdict) ==")
    for x in ARMS:
        print(f"{x:9s}: acc 4T {np.mean([R[x, s]['acc4'] for s in seeds]):.4f}; loss AUC "
              f"{np.mean([R[x, s]['losses'].mean() for s in seeds]):.3f}")
    for x in ARMS:                                            # table power in each band at the end (DFT over p in 0..255)
        hi_p, lo_p, pi_p = [], [], []
        for s in seeds:
            pe = R[x, s]["model"].layers[0].attn.cope.pos_emb.detach()[0, :, :256]
            F = torch.fft.rfft(pe - pe.mean(1, keepdim=True), dim=1).abs().pow(2).sum(0)
            f = torch.fft.rfftfreq(256) * 2 * math.pi                    # angular frequency per count unit
            hi_p.append(float(F[(f >= 0.6 * math.pi) & (f <= 0.95 * math.pi)].sum() / F.sum()))
            lo_p.append(float(F[(f >= 0.1 * math.pi) & (f <= 0.45 * math.pi)].sum() / F.sum()))
            pi_p.append(float(F[f > 0.95 * math.pi].sum() / F.sum()))                 # period ~2: the constructed solution
        print(f"{x:9s}: table power share (counts 0-255) in the Hi band {np.mean(hi_p):.2f}, Lo band {np.mean(lo_p):.2f}, "
              f"near pi (> 0.95 pi) {np.mean(pi_p):.2f}")
    from mapformer import rescore_hook
    from mapformer.environment_cancel import GridWorldCancel
    from mapformer.train_cancel import evaluate
    rescore_hook.install("auto"); te = GridWorldCancel(size=32, seed=10000, p_plus=0.5)
    for x in ARMS:
        rs = [evaluate(R[x, s]["model"], te, 128, 400, "cpu", seed=10 ** 6)[0] for s in seeds]
        print(f"dropout re-score {x:9s}: {np.mean(rs):.4f} (eval mode {np.mean([R[x, s]['acc'] for s in seeds]):.4f})")
    print(f"rescore_hook STATS: hooked {rescore_hook.STATS['hooked']}, fired {rescore_hook.STATS['fired']}, skipped {sorted(rescore_hook.STATS['skipped'])}")


if __name__ == "__main__":
    main()

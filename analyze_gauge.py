"""Registered analysis for GAUGE_PREREG.md (G1, G2, reading; then declared secondaries). CPU.
  python3 -m mapformer.analyze_gauge [--root runs/gauge/p0] [--seeds 1300-1307] [--no-json]
"""
import argparse, json

import numpy as np
import torch

from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, action_angles
from mapformer.stats_core import perm2_p, perm2_ci, fisher_solved, classify_run

REPO = "/home/prashr/mapformer"
CELLS = ("SO_m", "AO_m", "SO_p", "AO_p", "RoPE_p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/gauge/p0")
    ap.add_argument("--seeds", default="1300-1307")
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1)); n = len(seeds)
    R = {}
    for c in CELLS:
        arm = c.split("_")[0]
        for s in seeds:
            d = f"{a.root}/{c}_s{s}"
            ev = json.load(open(f"{d}/eval.json")); b = torch.load(f"{d}/{arm}.pt", map_location="cpu", weights_only=False)
            k = classify_run(b["losses"])
            R[c, s] = {"acc": ev["eval"]["128"]["acc"], "acc4": ev["eval"]["512"]["acc"], "solved": k["registered"] == "SOLVED",
                       "cls": k["cls"], "auc": float(np.mean(b["losses"])), "blob": b}
    print("== per seed: accuracy T = 128 and class ==")
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{c} {R[c, s]['acc']:.4f} {R[c, s]['cls'][:4]}" for c in CELLS))
    for c in CELLS:
        print(f"{c:7s}: mean {np.mean([R[c, s]['acc'] for s in seeds]):.4f}, SOLVED {sum(R[c, s]['solved'] for s in seeds)}/{n}, "
              f"acc 4T {np.mean([R[c, s]['acc4'] for s in seeds]):.4f}, loss AUC {np.mean([R[c, s]['auc'] for s in seeds]):.3f}")
    V = {}
    for key, x, ref in (("G1", "AO_p", "SO_p"), ("G2", "AO_m", "SO_m")):
        av = np.array([R[x, s]["acc"] for s in seeds]); rv = np.array([R[ref, s]["acc"] for s in seeds])
        ka = sum(R[x, s]["solved"] for s in seeds); kr = sum(R[ref, s]["solved"] for s in seeds)
        pf = fisher_solved(ka, n, kr, n); pp = perm2_p(rv, av)["p"]; d = float(av.mean() - rv.mean())
        ci = perm2_ci(rv, av, step=0.005, grid=(-1.0, 1.0))
        if kr < 4:
            lab = f"UNREADABLE ({ref} SOLVED {kr} < 4)"
        elif (pf < 0.05 and kr > ka) or (pp < 0.05 and d <= -0.05):
            lab = "SIGNED LEARNS BETTER"
        elif ka >= kr - 1 and av.mean() >= rv.mean() - 0.01:
            lab = "NO SIGN COST"
        else:
            lab = "UNMEASURED"
        V[key] = lab
        print(f"{key} ({x} vs {ref}): {lab} -- {av.mean():.4f} SOLVED {ka}/{n} vs {rv.mean():.4f} SOLVED {kr}/{n}; d {d:+.4f} "
              f"[95% CI {ci['lo']:+.3f}, {ci['hi']:+.3f}] (perm p {pp:.4f}); Fisher p {pf:.4f}")
    ka_p = sum(R["AO_p", s]["solved"] for s in seeds); ka_m = sum(R["AO_m", s]["solved"] for s in seeds)
    pf3 = fisher_solved(ka_m, n, ka_p, n)
    V["G3"] = "AO SOLVES +1/+2 MORE" if (pf3 < 0.05 and ka_p > ka_m) else "NO DIFFERENCE DETECTED"
    print(f"G3 (AO_p vs AO_m, SOLVED): {V['G3']} -- {ka_p}/{n} vs {ka_m}/{n} (Fisher p {pf3:.4f})")
    # Amendment 1: T1 is representational; this test reads LEARNABILITY. The reading cannot refute T1.
    if V["G1"].startswith("UNREADABLE") or V["G2"].startswith("UNREADABLE"):
        V["reading"] = "UNREADABLE"
    elif V["G1"] == "NO SIGN COST" and V["G2"] == "SIGNED LEARNS BETTER" and V["G3"] == "AO SOLVES +1/+2 MORE":
        V["reading"] = "SUPPORTED (the learnability cost follows the relative sign)"
    elif V["G1"] == "SIGNED LEARNS BETTER":
        V["reading"] = "NOT SUPPORTED (a sign cost without opposite moves; T1 is representational and stands)"
    else:
        V["reading"] = "AS FALLS"
    print(f"Reading: {V['reading']}")
    if not a.no_json:
        json.dump(V, open(f"{REPO}/GAUGE_VERDICTS.json", "w"), indent=2)
    print("\n== secondaries (declared; no verdict) ==")
    print("floor: predict blank ~0.50 (p_empty 0.5); CANCEL's measured +1/-1 floor 0.554")
    # Amendment 1: accuracy by revisit type (backtrack: unwrapped net displacement 0 since the last visit; lap: +-32k, k != 0)
    # and gap, re-evaluated on a fresh stream (np seed 10**6, 200 trajectories, eval mode)
    from mapformer.environment_steps import GridWorldSteps
    from mapformer.train_variant import VARIANT_MAP
    from mapformer.model_wrap_exist import MapFormerWM_AbsAO_r4
    import torch.nn.functional as F
    def build(c, b):
        arm = c.split("_")[0]; kw = dict(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32)
        if arm == "RoPE":
            m = VARIANT_MAP["RoPE"](**kw)
        else:
            m = (MapFormerWM_SignedAO_r4 if arm == "SO" else MapFormerWM_AbsAO_r4)(
                action_ids=(b["config"]["action_offset"], b["config"]["action_offset"] + 1), **kw)
        m.load_state_dict(b["model_state_dict"]); return m.eval()
    for c in CELLS:
        deltas = (1, -1) if c.endswith("_m") else (1, 2)
        acc = {}
        for s in seeds:
            m = build(c, R[c, s]["blob"]); te = GridWorldSteps(size=32, seed=10000, p_first=0.5, deltas=deltas)
            np.random.seed(10 ** 6)
            with torch.no_grad():
                for _ in range(200):
                    tok, _, rev = te.generate_trajectory(128)
                    lp = m(tok[None, :-1]).argmax(-1)[0]
                    acts = (tok[0::2] - te.action_offset).numpy(); up = np.cumsum(np.array(deltas)[acts])   # unwrapped pos
                    last = {}
                    for t in range(128):
                        cell = int(up[t] % 32)
                        if cell in last:
                            gap = t - last[cell]; kind = "backtrack" if up[t] == up[last[cell]] else "lap"
                            gb = "gap<16" if gap < 16 else "gap>=16"
                            hit = int(lp[2 * t] == tok[2 * t + 1])
                            for key in (kind, gb):
                                acc.setdefault(key, []).append(hit)
                        last[cell] = t
        print(f"{c:7s} strata: " + "; ".join(f"{k} {np.mean(v):.3f} (n {len(v)})" for k, v in sorted(acc.items())))
    for c in ("SO_p", "SO_m"):
        same = []
        for s in seeds:
            b = R[c, s]["blob"]
            m = MapFormerWM_SignedAO_r4(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32,
                                        action_ids=(b["config"]["action_offset"], b["config"]["action_offset"] + 1))
            m.load_state_dict(b["model_state_dict"]); A = action_angles(m).double().numpy()
            W = np.angle(np.exp(1j * A))                                  # Amendment 1: wrapped to (-pi, pi]
            sel = np.abs(W[0]) > 0.05
            same.append(float((np.sign(W[0]) == np.sign(W[1]))[sel].mean()) if sel.any() else float("nan"))
        print(f"{c}: share of channels whose two (wrapped) move angles have the SAME sign at the end {np.nanmean(same):.2f} (init ~0.50-0.70) "
              f"(per seed {' '.join(f'{v:.2f}' for v in same)})")


if __name__ == "__main__":
    main()

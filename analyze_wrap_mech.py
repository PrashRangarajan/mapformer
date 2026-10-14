"""Registered analysis for MECHANISM_PREREG.md (M1-M4, then declared secondaries). CPU.
  python3 -m mapformer.analyze_wrap_mech [--root runs/wrap_mech/p0] [--seeds 800-807] [--no-json]
"""
import argparse, json

import numpy as np
import torch

from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4, wrap_share
from mapformer.stats_core import perm2_p, fisher_solved, classify_run

REPO = "/home/prashr/mapformer"
ARMS = ("SO", "AO", "SO_same", "SO_absmag", "AO_absmag", "AO_high")


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


def contrast(R, seeds, x, ref):
    n = len(seeds)
    a = np.array([R[x, s]["acc"] for s in seeds]); r = np.array([R[ref, s]["acc"] for s in seeds])
    ka = sum(R[x, s]["solved"] for s in seeds); kr = sum(R[ref, s]["solved"] for s in seeds)
    return a, r, ka, kr, fisher_solved(ka, n, kr, n), perm2_p(r, a)["p"], a.mean() - r.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/wrap_mech/p0")
    ap.add_argument("--seeds", default="800-807")
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1)); n = len(seeds)
    R = {(x, s): load(a.root, x, s) for x in ARMS for s in seeds}
    V = {}
    print("== per seed: accuracy T = 128 (eval mode) and class ==")
    print("seed | " + " | ".join(ARMS))
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{R[x, s]['acc']:.4f} {R[x, s]['cls'][:4]}" for x in ARMS))
    for x in ARMS:
        print(f"{x:10s}: mean {np.mean([R[x, s]['acc'] for s in seeds]):.4f}, SOLVED {sum(R[x, s]['solved'] for s in seeds)}/{n}")

    a1, r1, ka, kr, pf, pp, d = contrast(R, seeds, "SO_absmag", "SO_same")
    if (pf < 0.05 and kr > ka) or (pp < 0.05 and d <= -0.05):
        V["M1"] = "HURTS"
    elif ka >= kr - 1 and a1.mean() >= r1.mean() - 0.01:
        V["M1"] = "NO EFFECT"
    else:
        V["M1"] = "UNMEASURED"
    print(f"\nM1 (SO_absmag vs SO_same): {V['M1']} -- {a1.mean():.4f} SOLVED {ka}/{n} vs {r1.mean():.4f} SOLVED {kr}/{n}; "
          f"d {d:+.4f} (perm p {pp:.4f}); Fisher p {pf:.4f}")
    a2, r2, ka, kr, pf, pp, d = contrast(R, seeds, "AO_high", "AO_absmag")
    if (pf < 0.05 and ka > kr) or (pp < 0.05 and d >= 0.05):
        V["M2"] = "RESCUES"
    elif ka <= kr + 1 and a2.mean() <= r2.mean() + 0.01:
        V["M2"] = "NO EFFECT"
    else:
        V["M2"] = "UNMEASURED"
    print(f"M2 (AO_high vs AO_absmag): {V['M2']} -- {a2.mean():.4f} SOLVED {ka}/{n} vs {r2.mean():.4f} SOLVED {kr}/{n}; "
          f"d {d:+.4f} (perm p {pp:.4f}); Fisher p {pf:.4f}")
    V["M3"] = {}
    for x, ref in (("SO_same", "SO"), ("AO_absmag", "AO")):
        a3, r3, ka, kr, pf, pp, d = contrast(R, seeds, x, ref)
        lab = "DIFFERS" if (pf < 0.05 or (pp < 0.05 and abs(d) >= 0.05)) else "NO DIFFERENCE DETECTED"
        V["M3"][f"{x} vs {ref}"] = lab
        print(f"M3 ({x} vs {ref}): {lab} -- {a3.mean():.4f} SOLVED {ka}/{n} vs {r3.mean():.4f} SOLVED {kr}/{n}; d {d:+.4f} "
              f"(perm p {pp:.4f}); Fisher p {pf:.4f}")
    a4, r4, ka, kr, pf, pp, d = contrast(R, seeds, "AO", "SO")
    if kr < 4:
        V["M4"] = f"UNREADABLE (SO SOLVED {kr} < 4)"
    elif (pf < 0.05 and kr > ka) or (pp < 0.05 and d <= -0.05):
        V["M4"] = "SIGNED LEARNS BETTER"
    elif ka >= kr - 1 and a4.mean() >= r4.mean() - 0.01:
        V["M4"] = "NO SIGN COST"
    else:
        V["M4"] = "UNMEASURED"
    print(f"M4 (AO vs SO): {V['M4']} -- {a4.mean():.4f} SOLVED {ka}/{n} vs {r4.mean():.4f} SOLVED {kr}/{n}; d {d:+.4f} "
          f"(perm p {pp:.4f}); Fisher p {pf:.4f}")

    print("\n== secondaries (declared; no verdict) ==")
    for x in ARMS:
        ws = [wrap_share(R[x, s]["model"])[1] for s in seeds]
        print(f"wrap share {x:10s}: {np.nanmean(ws):.2f} (per seed {' '.join(f'{w:.2f}' for w in ws)}); acc 4T "
              f"{np.mean([R[x, s]['acc4'] for s in seeds]):.4f}; loss AUC {np.mean([R[x, s]['losses'].mean() for s in seeds]):.3f}")
    for x in ("SO", "SO_same", "SO_absmag"):                     # initial difference vs final cancellation, signed arms
        rows = []
        for s in seeds:
            A0 = np.array(R[x, s]["ev"]["init_angles"]); d0 = np.abs(np.angle(np.exp(1j * (A0[0] - A0[1]))))
            from mapformer.model_wrap_exist import action_angles
            A1 = action_angles(R[x, s]["model"]).numpy(); canc = np.abs(np.angle(np.exp(1j * (A1[0] + A1[1])))) < 0.1
            big = np.abs(np.angle(np.exp(1j * A1[0]))) > 0.05
            rows.append((np.median(d0), canc[big & (d0 > np.median(d0))].mean() if (big & (d0 > np.median(d0))).any() else np.nan,
                         canc[big & (d0 <= np.median(d0))].mean() if (big & (d0 <= np.median(d0))).any() else np.nan))
        print(f"{x:10s}: median initial |a(+1) - a(-1)| {np.mean([r[0] for r in rows]):.3f} rad; final cancelling share among "
              f"channels with large vs small initial difference {np.nanmean([r[1] for r in rows]):.2f} vs {np.nanmean([r[2] for r in rows]):.2f}")
    if not a.no_json:
        json.dump(V, open(f"{REPO}/MECHANISM_VERDICTS.json", "w"), indent=2)


if __name__ == "__main__":
    main()

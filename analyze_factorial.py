"""Registered analysis for FACTORIAL_PREREG.md (index RoPE at 1 / 4 layers / looped x4 vs path-integrated MapWM at 1 layer,
torus T = 1024).  python3 -m mapformer.analyze_factorial [--smoke]   (from /home/prashr, after the batch)"""
import argparse, json, sys
import numpy as np
import torch

from mapformer.stats_core import perm2_p, fisher_solved, classify_run

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/factorial"; SEEDS = [500, 501, 502, 503]; EPOCHS = 900; N = 4
# cell -> (root, variant, layers)
CELLS = {"RoPE_L1": ("L1", "RoPE", 1), "Path_L1": ("L1", "Vanilla_r4", 1), "RoPE_L4": ("L4", "RoPE", 4),
         "RoPE_Loop": ("LOOP", "RoPELooped", 1)}


def c0(path, rope):
    d = float(np.mean(path) - np.mean(rope)); p = perm2_p(rope, path)["p"]
    lab = "PATH WINS AT ONE LAYER" if d >= 0.05 and p < .05 else "AS IT FALLS"
    return f"{lab}: d {d:+.4f} (perm p {p:.4f}; n = 4 vs 4, only large splits readable)"


def catch_up(index, path, sol_i, sol_p, name, n_desc=0):
    """Amendment 1 (audit D1, D2): the 0.02-margin label is renamed (no equivalence test is possible at n = 4) and every
    label is scoped to the 900-epoch budget with the count of index runs still DESCENDING."""
    d = float(np.mean(path) - np.mean(index)); p = perm2_p(index, path)["p"]; pf = fisher_solved(sol_p, N, sol_i, N)
    tail = (f"Path_L1 - {name} {d:+.4f} (perm p {p:.4f}); SOLVED {name} {sol_i}/4 vs Path_L1 {sol_p}/4 (Fisher p {pf:.4f}) "
            f"[within 900 epochs; {name} still DESCENDING {n_desc}/4]")
    if np.mean(index) >= np.mean(path) - 0.02 and sol_i >= 3:
        return f"INDEX ({name}) WITHIN 0.02 OF PATH (margin rule, no equivalence test at n = 4): {tail}"
    if (d >= 0.05 and p < .05) or (pf < .05 and sol_p > sol_i):
        return f"INDEX ({name}) DOES NOT CATCH UP: {tail}"
    return f"UNMEASURED (n = 4): {tail}"


def helps(deep, base, name):
    d = float(np.mean(deep) - np.mean(base)); p = perm2_p(base, deep)["p"]
    return f"{'HELPS' if d >= 0.05 and p < .05 else 'AS IT FALLS'}: {name} - RoPE_L1 {d:+.4f} (perm p {p:.4f})"


def smoke():
    ok = True
    def chk(got, want):
        nonlocal ok
        ok &= want in got; print(f"  {'ok ' if want in got else 'BAD'} {got}")
    one = [0.998, 0.999, 0.997, 0.998]; low = [0.73, 0.72, 0.74, 0.73]
    chk(c0(one, low), "PATH WINS")
    chk(catch_up([0.99, 0.995, 0.99, 0.985], one, 4, 4, "RoPE_L4"), "WITHIN 0.02 OF PATH")
    chk(catch_up(low, one, 0, 4, "RoPE_L4"), "DOES NOT CATCH UP")
    chk(catch_up([0.999, 0.80, 0.998, 0.70], one, 2, 4, "RoPE_L4"), "UNMEASURED")
    chk(helps([0.9, 0.92, 0.91, 0.9], low, "RoPE_L4"), "HELPS")
    print("SMOKE", "ALL PASS" if ok else "FAILED"); return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    acc, solved, cls, out = {}, {}, {}, {}
    for cell, (root, var, nl) in CELLS.items():
        J = json.load(open(f"{REPO}/FACTORIAL_{root}.json"))
        by_seed = {int(r[0]): r for r in J[f"0.0|{var}|1024"]}
        acc[cell], solved[cell], cls[cell] = [], [], []
        for s in SEEDS:
            b = torch.load(f"{R}/{root}/p0/{var}_s{s}/{var}.pt", map_location="cpu", weights_only=False); c = b["config"]
            assert b["variant"] == var and b["seed"] == s and len(b["losses"]) == EPOCHS, (cell, s)
            assert c["n_layers"] == nl and c["n_steps"] == 1024 and c["epochs"] == EPOCHS and c.get("fast_attn"), (cell, s, c)
            acc[cell].append(float(by_seed[s][1])); k = classify_run(b["losses"])
            cls[cell].append(k["registered"]); solved[cell].append(k["registered"] == "SOLVED")
        print(f"  {cell:9s} acc T=1024 {np.mean(acc[cell]):.4f} (per seed {[round(x, 3) for x in acc[cell]]}) | SOLVED "
              f"{sum(solved[cell])}/4 | classes {cls[cell]}")
    print("  floors (RANK_MI stream): blank 0.507, best n-gram 0.576")
    print(f"  REGISTERED C0 (Path_L1 - RoPE_L1): {c0(acc['Path_L1'], acc['RoPE_L1'])}")
    for cell in ("RoPE_L4", "RoPE_Loop"):
        tag = "C1 (depth)" if cell == "RoPE_L4" else "C2 (loop)"
        print(f"  REGISTERED {tag}: {catch_up(acc[cell], acc['Path_L1'], sum(solved[cell]), sum(solved['Path_L1']), cell, cls[cell].count('DESCENDING'))}")
        print(f"  REGISTERED C3 ({cell}): {helps(acc[cell], acc['RoPE_L1'], cell)}")
    for root in ("L1", "L4", "LOOP"):
        try:
            S = json.load(open(f"{REPO}/FACTORIAL_{root}_STRATA.json"))
            for k, v in S.items():
                if k.endswith("|1024"):
                    parts = [f"{st} {v[st]['acc']:.3f}" for st in v if isinstance(v[st], dict) and v[st].get("acc") is not None]
                    print(f"  secondary strata {root} {k}: " + ", ".join(parts))
        except FileNotFoundError:
            pass
    own = {}
    for cell, (root, var, nl) in CELLS.items():                     # Amendment 1 (D3): own (training) map, secondary
        vals = []
        for sd in SEEDS:
            try:
                J = json.load(open(f"{REPO}/FACTORIAL_{root}_OWN_s{sd}.json")); vals.append(float(J[f"0.0|{var}|1024"][0][1]))
            except (FileNotFoundError, KeyError, IndexError):
                pass
        if vals:
            own[cell] = vals
            print(f"  secondary own (training) map {cell}: acc T=1024 {np.mean(vals):.4f} vs held-out {np.mean(acc[cell]):.4f}")
    json.dump({"acc": acc, "solved": solved, "cls": cls, "own_map": own}, open(f"{REPO}/FACTORIAL_VERDICTS.json", "w"), indent=1)


if __name__ == "__main__":
    main()

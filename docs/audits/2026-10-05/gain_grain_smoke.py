"""Smoke test of every registered branch of analyze_gain_grain.py on synthetic data (GAIN_GRAIN_PREREG.md, rule 29).
Part 1: each branch of granularity_state / sign_state / headroom_state is reached by a constructed case and the label
is asserted; granularity_verdict is printed for all 25 label pairs. Part 2: main() end to end on a synthetic run
directory (untrained models of every arm, fake per-epoch losses, fake eval JSON) in a temporary directory.
Output: gain_grain_smoke_out.txt."""
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
import mapformer.analyze_gain_grain as A
from mapformer import train_gain_grain  # noqa: F401
from mapformer.train_variant import VARIANT_MAP

rng = np.random.default_rng(0)
n = 16
ok_all = True


def near(m, sd, k=n, lo=0.0, hi=1.0):
    return np.clip(rng.normal(m, sd, k), lo, hi)


def expect(name, got, want):
    global ok_all
    ok = got.startswith(want)
    ok_all &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:62s} -> {got}")


ceil = np.ones(n); good = near(0.9995, 0.0008, lo=0.997)
print("== part 1: decision functions ==")
print(" granularity_state(ref = P, x = coarser gain)")
lab, det = A.granularity_state(good, 16, good.copy(), 16, n, n); expect("identical arms", lab, "AS GOOD")
lab, det = A.granularity_state(ceil, 16, ceil, 16, n, n); expect("both at 1.000 (ceiling)", lab, "AS GOOD")
print(f"      detail: {det}")
bad = np.concatenate([good[:10], near(0.85, 0.05, 6)])
lab, det = A.granularity_state(good, 16, bad, 10, n, n); expect("x 10/16 solved, failures ~0.85 (accuracy + SOLVED)", lab, "WORSE")
print(f"      detail: {det}")
lab, det = A.granularity_state(good, 16, np.clip(good - 0.012, 0, 1), 16, n, n); expect("x uniformly 0.012 lower, all solved (accuracy only)", lab, "WORSE")
lab, det = A.granularity_state(good, 16, good.copy(), 10, n, n); expect("x accuracy equal, SOLVED 10/16 (SOLVED only)", lab, "WORSE")
lab, det = A.granularity_state(bad, 10, good, 16, n, n); expect("reference falters, x solid", lab, "BETTER")
lab, det = A.granularity_state(good, 16, np.clip(good - 0.012, 0, 1), 10, n, n)
expect("x lower on accuracy and SOLVED (both NEG)", lab, "WORSE")
noisy = np.concatenate([good[:15], [0.93]])
lab, det = A.granularity_state(good, 16, noisy, 15, n, n); expect("one x seed at 0.93 (CI too wide)", lab, "UNDETERMINED")
print(f"      detail: {det}")
lab, det = A.granularity_state(good, 16, good.copy(), 14, n, n); expect("accuracy equal, SOLVED 14/16 (slack rule)", lab, "UNDETERMINED")
print(f"      detail: {det}")
conf_ref = np.concatenate([good[:12], near(0.97, 0.005, 4)])
lab, det = A.granularity_state(conf_ref, 4, np.concatenate([near(0.95, 0.003, 16)]), 16, n, n)
expect("accuracy NEG while SOLVED POS", lab, "CONFLICT")
print(f"      detail: {det}")

print(" granularity_verdict (all 25 label pairs)")
labs = ["AS GOOD", "BETTER", "WORSE", "UNDETERMINED", "CONFLICT"]
for s_ in labs:
    for m_ in labs:
        print(f"    S {s_:12s} M {m_:12s} -> {A.granularity_verdict(s_, m_)}")

print(" sign_state(E = MapEM, N = softplus)")
em = np.concatenate([near(0.997, 0.001, 10, hi=1.0), near(0.80, 0.1, 6)])
lab, det = A.sign_state(ceil, 16, ceil, 16, n, n); expect("both at ceiling", lab, "CEILING")
lab, det = A.sign_state(em, 10, good, 16, n, n); expect("N solid, E bimodal", lab, "NON-NEGATIVE BETTER")
print(f"      detail: {det}")
lab, det = A.sign_state(good, 16, em, 10, n, n); expect("N bimodal, E solid", lab, "NON-NEGATIVE WORSE")
lab, det = A.sign_state(em, 10, em[::-1].copy(), 10, n, n); expect("same distribution", lab, "NO DIFFERENCE DETECTED")
print(f"      detail: {det}")
lab, det = A.sign_state(np.concatenate([near(0.95, 0.003, 16)]), 16, conf_ref, 4, n, n)
expect("accuracy POS while SOLVED NEG", lab, "CONFLICT")
lab, det = A.sign_state(good, 16, good.copy(), 9, n, n); expect("SOLVED only, negative", lab, "NON-NEGATIVE WORSE")
print(" headroom_state(W, P)")
lab, det = A.headroom_state(bad, 10, good, 16, n, n); expect("MapWM deficit replicates", lab, "OK")
lab, det = A.headroom_state(good, 16, good.copy(), 16, n, n); expect("MapWM at P's level", lab, "NO HEADROOM")

print("\n== part 2: main() end to end on synthetic checkpoints (untrained weights, fake losses, fake eval) ==")
tmp = tempfile.mkdtemp(prefix="gain_grain_smoke_")
A.REPO = tmp; A.R = f"{tmp}/runs/gain_grain/p0"
J = {}
profile = {"Vanilla": (0.97, 10), "MapPoPE-Pair": (0.9995, 16), "GainScalar": (0.9995, 16), "GainMod4": (0.995, 15),
           "VanillaEM": (0.93, 10), "VanillaEM_NonNeg": (0.99, 15)}
for a in A.ARMS:
    mu, ns = profile[a]
    for T, dT in ((128, 0.0), (512, 0.03), (1024, 0.08)):
        J[f"0.0|{a}|{T}"] = [[s, float(np.clip(mu - dT + rng.normal(0, 0.005), 0, 1)), float(abs(rng.normal(0.05, 0.02)))] for s in A.SEEDS]
    for i, s in enumerate(A.SEEDS):
        torch.manual_seed(s)
        m = VARIANT_MAP[a](vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64)
        tail = 0.01 if i < ns else 0.3
        losses = list(np.concatenate([np.linspace(2.0, tail, 250), np.full(50, tail)]) + rng.normal(0, 1e-4, 300))
        d = f"{A.R}/{a}_s{s}"; os.makedirs(d)
        torch.save({"model_state_dict": m.state_dict(), "losses": losses, "variant": a, "seed": s,
                    "config": {"vocab_size": 21, "d_model": 128, "n_heads": 2, "n_layers": 1, "grid_size": 64,
                               "n_obs_types": 16, "p_empty": 0.5}}, f"{d}/{a}.pt")
json.dump(J, open(f"{tmp}/GAIN_GRAIN_EVAL.json", "w"))
A.main()
print(f"\n(synthetic directory {tmp}; numbers above are synthetic and mean nothing)")
print(f"\nPART 1 OVERALL: {'PASS' if ok_all else 'FAIL'}")

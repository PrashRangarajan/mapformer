"""Smoke test of every branch of analyze_gain_phase on synthetic data, and of main()'s path (load_runs + analyse) end to
end on 32 synthetic checkpoints. Output: gain_phase_smoke_out.txt (PASS / FAIL per case, then the exhaustive verdict
tables)."""
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
import mapformer.analyze_gain_phase as A

A.N_MC = 20_000
ok = True


def case(name, got, want):
    global ok
    g = got[0] if isinstance(got, tuple) else got
    good = g == want
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {g}" + ("" if good else f" (expected {want})"))


n = 8; T, F = [True] * n, [False] * n
hi = np.array([0.9998, 0.9996, 0.9995, 1.0, 0.9995, 1.0, 0.9998, 1.0])
lo = np.array([0.9893, 0.9918, 0.9861, 0.9875, 0.9838, 0.9919, 0.9908, 0.9911])
mid = lo + 0.003
print("== contrast_state (y vs x) ==")
case("BETTER on accuracy and SOLVED", A.contrast_state(lo, F, hi, T), "BETTER")
case("BETTER on accuracy only (both SOLVED)", A.contrast_state(lo, T, hi, T), "BETTER")
case("BETTER on SOLVED only (accuracy at ceiling-ish, tiny gap)", A.contrast_state(hi - 0.0003, [True] * 2 + [False] * 6, hi, T), "BETTER")
case("WORSE on accuracy", A.contrast_state(hi, T, lo, T), "WORSE")
case("WORSE on SOLVED only", A.contrast_state(hi, T, hi - 0.0003, [True] * 2 + [False] * 6), "WORSE")
case("CONFLICT", A.contrast_state(lo, T, hi, F), "CONFLICT")
case("CEILING", A.contrast_state(hi, T, hi[::-1], T), "CEILING")
case("NO DIFFERENCE (below MIN_D)", A.contrast_state(lo, F, mid, F), "NO DIFFERENCE")
case("NO DIFFERENCE (same)", A.contrast_state(lo, F, lo[::-1], F), "NO DIFFERENCE")
print("== ni_state (GP vs NormStep) ==")
case("AS GOOD (at ceiling)", A.ni_state(hi, T, hi[::-1], T), "AS GOOD")
case("AS GOOD (off ceiling)", A.ni_state(hi - 0.002, T, hi - 0.0021, T), "AS GOOD")
case("UNDETERMINED (one run unsolved, acc within)", A.ni_state(hi, T, hi, T[:7] + [False]), "UNDETERMINED")
case("UNDETERMINED (acc not within margin)", A.ni_state(hi, T, np.r_[hi[:6], 0.97, 0.995], T), "UNDETERMINED")
case("WORSE", A.ni_state(hi, T, lo, T), "WORSE")
case("BETTER", A.ni_state(lo, F, hi, T), "BETTER")
case("CONFLICT", A.ni_state(lo, T, hi, F), "CONFLICT")
print("== speed_state (y vs x) ==")
case("FASTER", A.speed_state([300, 320, 280, 310, 290, 305, 315, 295], [100, 110, 90, 105, 95, 100, 98, 102]), "FASTER")
case("SLOWER", A.speed_state([100, 110, 90, 105, 95, 100, 98, 102], [300, 320, 280, 310, 290, 305, 315, 295]), "SLOWER")
case("NO DIFFERENCE", A.speed_state([300, 320, 280, 310, 290, 305, 315, 295], [310, 300, 290, 300, 295, 315, 305, 285]), "NO DIFFERENCE")
case("NO DIFFERENCE (p < .05 but ratio < 1.25)", A.speed_state([300] * 8, [270, 271, 272, 273, 274, 275, 276, 277]), "NO DIFFERENCE")
case("NEITHER CONVERGED", A.speed_state([901] * 8, [901] * 8), "NEITHER CONVERGED")
case("FASTER with censoring", A.speed_state([901] * 8, [300, 320, 280, 310, 290, 305, 315, 295]), "FASTER")
case("speed_epoch censoring", A.speed_epoch([0.1] * 900), 901)
case("speed_epoch: 1-based last epoch of the first window with mean < 0.05", A.speed_epoch([0.1] * 50 + [0.01] * 850), 56)
print("== leak_side (raw, norm) ==")
wl = np.array([0.0106, 0.0082, 0.0139, 0.0124, 0.0159, 0.0078, 0.0089, 0.0074]); nl = np.array([0.0002, 0.0003, 0.0004, 0, 0.0004, 0, 0.0001, 0])
ws = np.array([0.0050, 0.0039, 0.0048, 0.0050, 0.0048, 0.0041, 0.0048, 0.0038]); ns = ws / 4.7
case("REMOVES", A.leak_side(wl, nl, ws, ns), "REMOVES")
case("NO LEAK TO REMOVE", A.leak_side(nl, nl, ns, ns), "NO LEAK TO REMOVE")
case("INCOMPLETE (norm still leaks)", A.leak_side(wl, wl / 2, ws, ns), "INCOMPLETE")
case("INCOMPLETE (S_id not lower)", A.leak_side(wl, nl, ws, ws[::-1]), "INCOMPLETE")
print("== gain_leak (GainRaw) ==")
case("PERSISTS", A.gain_leak(wl, ws, ws[::-1]), "PERSISTS")
case("ABSENT, UNLEARNED", A.gain_leak(nl, ws, ns), "ABSENT, UNLEARNED")
case("ABSENT, TOLERATED", A.gain_leak(nl, ws, ws[::-1]), "ABSENT, TOLERATED")
case("PARTIAL", A.gain_leak(wl / 3, ws, ws), "PARTIAL")

print("\n== step_verdict (D1r, D1): exhaustive ==")
for rep in ("BETTER", "NO DIFFERENCE", "CEILING", "WORSE", "CONFLICT"):
    for d1 in ("BETTER", "CEILING", "NO DIFFERENCE", "WORSE", "CONFLICT"):
        print(f"  {rep:13s} x {d1:13s} -> {A.step_verdict(rep, d1)}")
print("\n== leak_verdict (rotary, gain, GainRaw): exhaustive ==")
for rot in ("REMOVES", "NO LEAK TO REMOVE", "INCOMPLETE"):
    for gai in ("REMOVES", "NO LEAK TO REMOVE", "INCOMPLETE"):
        for gl in ("PERSISTS", "ABSENT, TOLERATED", "ABSENT, UNLEARNED", "PARTIAL"):
            print(f"  {rot:17s} x {gai:17s} x {gl:17s} -> {A.leak_verdict(rot, gai, gl)}")
print("\n== combo_verdict (GP vs MapWM, GP vs NormStep): exhaustive ==")
for vw in ("BETTER", "WORSE", "CEILING", "NO DIFFERENCE", "CONFLICT"):
    for vn in ("AS GOOD", "BETTER", "WORSE", "UNDETERMINED", "CONFLICT"):
        print(f"  {vw:13s} x {vn:12s} -> {A.combo_verdict(vw, vn)}")

print("\n== main path end to end: 32 synthetic checkpoints + synthetic GAIN_PHASE_EVAL.json ==")
tmp = tempfile.mkdtemp(prefix="gain_phase_smoke_")
rng = np.random.default_rng(0); J = {}
solved_curve = list(6.8 * np.exp(-np.arange(900) / 60) + 0.015)
desc_curve = list(6.8 * np.exp(-np.arange(900) / 120) + 0.07 + 0.02 * np.linspace(1, 0, 900))
for a in A.ARMS:
    for s in A.SEEDS:
        d = f"{tmp}/{a}_s{s}"; os.makedirs(d)
        raw = a in ("MapWM", "GainRaw")
        torch.save({"losses": desc_curve if raw else solved_curve}, f"{d}/{a}.pt")
        J[f"{a}|{s}"] = {"acc": float(lo[s - 8] if raw else hi[s - 8]), "L_ms": float(wl[s - 8] if raw else nl[s - 8]),
                         "S_id": float(ws[s - 8] if raw else ns[s - 8]), "L_zero": 0.0, "x2": 0.9, "x4": 0.8, "train_x1": 0.99}
json.dump(J, open(f"{tmp}/EVAL.json", "w"))
D = A.load_runs(f"{tmp}/EVAL.json", tmp)
V = A.analyse(D)
case("end to end: D1 headline", V["D1_headline"].split(" (")[0], "NORMSTEP HELPS UNDER BOTH SCORES")
case("end to end: D3 headline", V["D3_headline"].split(":")[0], "SEPARATE DEFECTS")
case("end to end: D4 headline", V["D4_headline"].split(":")[0], "THE GAIN-PHASE MAP WORKS")
case("end to end: D5 (raw arms never reach 0.05)", V["D5_raw"], "NEITHER CONVERGED")
D.pop(("GainPhase", 15))
case("VOID on a missing run", list(A.analyse(D, out=lambda *a: None)), ["void"])
print(f"\nALL {'PASS' if ok else 'FAIL'}")

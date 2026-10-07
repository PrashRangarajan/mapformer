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

print("== Amendment 1: NEGATIVE, |median| for ABSENT, gate, budget flags, tolerated note, rescore flags ==")
case("leak_side NEGATIVE (norm arm < -0.002)", A.leak_side(wl, -wl / 2, ws, ns), "NEGATIVE")
case("leak_side NEGATIVE (raw arm < -0.002)", A.leak_side(-wl, nl, ws, ns), "NEGATIVE")
case("leak_side NO LEAK TO REMOVE uses |median|", A.leak_side(-nl, nl, ns, ns), "NO LEAK TO REMOVE")
case("gain_leak NEGATIVE", A.gain_leak(-wl, ws, ws), "NEGATIVE")
case("gain_leak ABSENT at a small negative median", A.gain_leak(-nl, ws, ws[::-1]), "ABSENT, TOLERATED")
case("gain_leak PARTIAL just below -0.002 is NEGATIVE, not ABSENT", A.gain_leak(np.full(8, -0.0025), ws, ws), "NEGATIVE")
cl_s = [{"cls": "SOLVED", "registered": "SOLVED", "tail": 0.015}] * 8
cl_d = [{"cls": "DESCENDING", "registered": "DESCENDING", "tail": 0.07}] * 8
cl_hi = [{"cls": "DESCENDING", "registered": "DESCENDING", "tail": 0.5}] * 8
rel_hi, rel_0 = np.full(8, 0.9), np.full(8, 0.01)
cl_flat = [{"cls": "STALLED", "registered": "STALLED", "tail": 3.6}] * 8
print("== Amendment 2: leak-independent gate, scope flags counting STALLED ==")
wn = np.array([0.0071, 0.0078, 0.0141, 0.0124, 0.0242, 0.0113, 0.0155, 0.0377])   # LEAK MapWM nll_ms (validation)
nn = np.full(8, 0.0006)                                                               # LEAK NormStep-like nll_ms
print("== Amendment 3: gate on the leak-free loss nll_ms ==")
case("gate PASS (MapWM-like: nll_ms 0.007-0.038, reliance 0.9)", "|".join(map(str, A.converged_gate(rel_hi, wn, cl_d)[:2])), "True|")
case("gate PASS with a training tail of 0.28 (a big leak's loss) -- the tail is not read",
     "|".join(map(str, A.converged_gate(rel_hi, wn, [{"cls": "DESCENDING", "registered": "DESCENDING", "tail": 0.28}] * 8)[:2])), "True|")
case("gate FAIL: leak-free loss above 0.06", A.converged_gate(rel_hi, np.full(8, 0.5), cl_hi)[1], "leak-free loss above 0.06 (not converged)")
case("gate FAIL: theta unused although converged -> named", A.converged_gate(rel_0, nn, cl_s)[1], "theta reliance below 0.2 (does not use theta)")
case("gate FAIL: both", A.converged_gate(rel_0, np.full(8, 3.5), cl_flat)[1],
     "leak-free loss above 0.06 (not converged) and theta reliance below 0.2 (does not use theta)")
case("scope flag, solved basis: GP 2 DESCENDING vs N 0", bool(A.budget_flag("GainPhase", cl_s[:6] + cl_d[:2], "NormStep", cl_s, "solved")), True)
case("scope flag, solved basis: GP 2 STALLED (flat tail) vs N 0", bool(A.budget_flag("GainPhase", cl_s[:6] + cl_flat[:2], "NormStep", cl_s, "solved")), True)
case("scope flag, solved basis: equal -> none", A.budget_flag("GainPhase", cl_s, "NormStep", cl_s, "solved"), "")
case("scope flag, regime basis: GainRaw leak-free loss 0.05 above every MapWM run (max 0.0377)", bool(A.budget_flag("GainRaw", cl_hi, "MapWM", cl_d, "regime", np.full(8, 0.05), wn)), True)
case("scope flag, regime basis: flat-tail GainRaw with a high leak-free loss", bool(A.budget_flag("GainRaw", cl_flat, "MapWM", cl_d, "regime", np.full(8, 3.5), wn)), True)
case("scope flag, regime basis: a big leak (training tail 0.28) with MapWM's leak-free loss -> none",
     A.budget_flag("GainRaw", [{"cls": "DESCENDING", "registered": "DESCENDING", "tail": 0.28}] * 8, "MapWM", cl_d, "regime", wn, wn), "")
case("tolerated note: isotropic", A.tolerated_note(np.full(8, 0.9), ws, ws).split(":")[0], "mostly isotropic spread")
case("tolerated note: field shift", A.tolerated_note(np.full(8, 0.01), ws, ws).split(":")[0], "mostly a field shift the gain score ignores")
case("combo WORSE x AS GOOD with D1r WORSE names NormStep's deficit", A.combo_verdict("WORSE", "AS GOOD", "WORSE").split(":")[0], "WORSE THAN MapWM BUT AS GOOD AS NormStep")
case("combo WORSE x AS GOOD without D1r WORSE is softened", A.combo_verdict("WORSE", "AS GOOD", "NO DIFFERENCE").split(";")[0], "WORSE THAN MapWM BUT NON-INFERIOR TO NormStep (within 0.005)")
case("fires_on", A.fires_on("x -- fires on accuracy and SOLVED"), "accuracy and SOLVED")

print("\n== step_verdict (D1r, D1): exhaustive ==")
for rep in ("BETTER", "NO DIFFERENCE", "CEILING", "WORSE", "CONFLICT"):
    for d1 in ("BETTER", "CEILING", "NO DIFFERENCE", "WORSE", "CONFLICT"):
        print(f"  {rep:13s} x {d1:13s} -> {A.step_verdict(rep, d1)}")
print("\n== leak_verdict (rotary, gain, GainRaw): exhaustive ==")
for rot in ("REMOVES", "NO LEAK TO REMOVE", "INCOMPLETE", "NEGATIVE"):
    for gai in ("REMOVES", "NO LEAK TO REMOVE", "INCOMPLETE", "NEGATIVE"):
        for gl in ("PERSISTS", "ABSENT, TOLERATED", "ABSENT, UNLEARNED", "PARTIAL", "NEGATIVE"):
            print(f"  {rot:17s} x {gai:17s} x {gl:17s} -> {A.leak_verdict(rot, gai, gl)}")
print("\n== combo_verdict (GP vs MapWM, GP vs NormStep): exhaustive ==")
for vw in ("BETTER", "WORSE", "CEILING", "NO DIFFERENCE", "CONFLICT"):
    for vn in ("AS GOOD", "BETTER", "WORSE", "UNDETERMINED", "CONFLICT"):
        print(f"  {vw:13s} x {vn:12s} -> {A.combo_verdict(vw, vn)}")

solved_curve = list(6.8 * np.exp(-np.arange(900) / 60) + 0.015)
desc_curve = list(6.8 * np.exp(-np.arange(900) / 120) + 0.07 + 0.02 * np.linspace(1, 0, 900))
high_curve = list(6.8 * np.exp(-np.arange(900) / 300) + 0.3)          # still descending at 900 (tail ~0.6)
flat_curve = list(6.8 * np.exp(-np.arange(900) / 10) + 3.6)            # flattened at 3.6 (STALLED), pilot-like
leaky_curve = list(np.array(desc_curve) + 0.21)                        # 3x leak: training tail ~0.28 (auditor's measured mapping)
conv_flat_curve = list(6.8 * np.exp(-np.arange(900) / 10) + 0.09)      # converged, flat at 0.09 (STALLED)


def synth(variant):
    """32 synthetic checkpoints + EVAL.json. 'predicted': raw arms MapWM-like, NormStep arms NormStep-like.
    'unconverged': both gain arms untrained-like (acc 0.14, reliance 0, S_id 1.0, L_ms 0), loss still descending at 0.6.
    'rescore_flip': as predicted, but GainPhase's re-scored accuracy falls to MapWM's (a D4 / D2 label flips) and
                    GainRaw's re-scored reliance falls to 0.1 (the D3 gate flips).
    Amendment 2: 'flat_tail': GainPhase flattened at loss 3.6 (STALLED), acc 0.14, reliance 0.05.
    'tolerated_bigsid': GainRaw SOLVED, acc ~1, reliance 0.9, L_ms ~0, S_id and field shift 4x MapWM's.
    'persists_large': GainRaw acc MapWM's - 0.018, L_ms 3x, S_id 3x MapWM's; Amendment 3: training tail ~0.28 (the
                      auditor's measured loss mapping; was ~0.15), leak-free loss MapWM's.
    'persists_lms3' (M'): GainRaw L_ms 3x, S_id unchanged, training tail ~0.28, leak-free loss MapWM's.
    'converged_flat' (Q): GainRaw flat at training loss 0.09 (STALLED), leak 1.3x MapWM's, leak-free loss MapWM's.
    'lagging': GainRaw leak-free loss 0.05 (passes the 0.06 gate, above every MapWM run): D3 read, with GainRaw's qualifier.
    'solved_no_theta': GainRaw SOLVED, acc ~1, reliance 0.1."""
    tmp = tempfile.mkdtemp(prefix=f"gain_phase_smoke_{variant}_"); J = {}
    for a in A.ARMS:
        for s in A.SEEDS:
            d = f"{tmp}/{a}_s{s}"; os.makedirs(d); i = s - 8
            raw = a in ("MapWM", "GainRaw"); unc = variant == "unconverged" and a in ("GainRaw", "GainPhase")
            curve = high_curve if unc else (desc_curve if raw else solved_curve)
            if variant == "flat_tail" and a == "GainPhase":
                curve = flat_curve
            if a == "GainRaw" and variant in ("tolerated_bigsid", "solved_no_theta"):
                curve = solved_curve
            if a == "GainRaw" and variant in ("persists_large", "persists_lms3"):
                curve = leaky_curve
            if a == "GainRaw" and variant == "converged_flat":
                curve = conv_flat_curve
            torch.save({"losses": curve}, f"{d}/{a}.pt")
            r = {"acc": float(lo[i] if raw else hi[i]), "L_ms": float(wl[i] if raw else nl[i]),
                 "S_id": float(ws[i] if raw else ns[i]), "reliance": 0.9, "resid": 0.002, "shift_cells": float(ws[i] if raw else ns[i]),
                 "L_zero": 0.0, "x2": 0.9, "x4": 0.8, "train_x1": 0.99, "nll_ms": float(wn[i] if raw else nn[i])}
            if unc:
                r.update(acc=0.14, L_ms=0.0, S_id=1.0, reliance=0.0, resid=0.6, shift_cells=0.5, nll_ms=4.0)
            if variant == "flat_tail" and a == "GainPhase":
                r.update(acc=0.14, L_ms=0.03, S_id=1.3, reliance=0.05, resid=0.6, shift_cells=0.8, nll_ms=3.5)
            if a == "GainRaw" and variant == "tolerated_bigsid":
                r.update(acc=float(hi[i]), L_ms=float(nl[i]), S_id=float(4 * ws[i]), shift_cells=float(4 * ws[i]), nll_ms=float(nn[i]))
            if a == "GainRaw" and variant == "persists_large":
                r.update(acc=float(lo[i] - 0.018), L_ms=float(3 * wl[i]), S_id=float(3 * ws[i]), shift_cells=float(3 * ws[i]))
            if a == "GainRaw" and variant == "solved_no_theta":
                r.update(acc=float(hi[i]), L_ms=float(nl[i]), reliance=0.1, nll_ms=float(nn[i]))
            if a == "GainRaw" and variant == "persists_lms3":
                r.update(acc=float(lo[i] - 0.018), L_ms=float(3 * wl[i]))
            if a == "GainRaw" and variant == "converged_flat":
                r.update(acc=float(lo[i] - 0.003), L_ms=float(1.3 * wl[i]), S_id=float(1.3 * ws[i]))
            if a == "GainRaw" and variant == "lagging":
                r.update(nll_ms=0.05)
            if a == "MapWM" and variant == "d3_void":
                r.update(L_ms=float(nl[i]))
            r["acc_rescored"] = float(lo[i]) if (variant == "rescore_flip" and a == "GainPhase") else r["acc"]
            r["reliance_rescored"] = 0.1 if (variant in ("rescore_flip", "d3_void") and a == "GainRaw") else r["reliance"]
            J[f"{a}|{s}"] = r
    json.dump(J, open(f"{tmp}/EVAL.json", "w"))
    return A.load_runs(f"{tmp}/EVAL.json", tmp)


print("\n== main path end to end: 32 synthetic checkpoints + synthetic GAIN_PHASE_EVAL.json (predicted) ==")
D = synth("predicted")
V = A.analyse(D)
case("end to end: D1 headline", V["D1_headline"].split(" (")[0], "NORMSTEP HELPS UNDER BOTH SCORES")
case("end to end: D1 carries D1r's firing test", "[D1r fires on accuracy and SOLVED]" in V["D1_headline"], True)
case("end to end: D3 headline", V["D3_headline"].split(":")[0], "SEPARATE DEFECTS")
case("end to end: D4 headline", V["D4_headline"].split(":")[0], "THE GAIN-PHASE MAP WORKS")
case("end to end: no rescore flag when labels agree", "FLAG" in V["D4_headline"], False)
case("end to end: D5 (raw arms never reach 0.05)", V["D5_raw"], "NEITHER CONVERGED")
print("\n== end to end, unconverged gain arms ==")
V = A.analyse(synth("unconverged"))
case("unconverged: D3 UNMEASURED", V["D3_headline"].split(":")[0], "D3 UNMEASURED")
case("unconverged: D3 names the failed criteria", "GainRaw: leak-free loss above 0.06 (not converged) and theta reliance below 0.2" in V["D3_headline"], True)
case("unconverged: D4 carries the scope", "not converged: GainPhase" in V["D4_headline"], True)
case("unconverged: D2 carries both scopes", "not converged: GainRaw" in V["D2_headline"] and "not converged: GainPhase" in V["D2_headline"], True)
print("\n== end to end, Amendment 2 scenarios ==")
V = A.analyse(synth("flat_tail"))
case("flat tail: D4 qualified although 0/8 DESCENDING", "not converged: GainPhase 0/8 SOLVED, 8/8 STALLED" in V["D4_headline"], True)
case("flat tail: D3 UNMEASURED, GainPhase fails both criteria", "GainPhase: leak-free loss above 0.06 (not converged) and theta reliance below 0.2" in V["D3_headline"], True)
V = A.analyse(synth("tolerated_bigsid"))
case("TOLERATED with S_id 4x MapWM's is read (gate does not use S_id)", V["D3_headline"].split(":")[0], "THE GAIN SCORE TOLERATES THE LEAK")
case("TOLERATED carries the field-shift note", "mostly a field shift the gain score ignores" in V["D3_headline"], True)
V = A.analyse(synth("persists_large"))
case("large PERSISTS (3x MapWM's leak, acc 0.97, training tail 0.28) is read", V["D3_headline"].split(":")[0], "SEPARATE DEFECTS")
case("large PERSISTS: no 'not converged' qualifier (leak-free loss is MapWM's)", "not converged: GainRaw" in V["D3_headline"] + V["D2_headline"], False)
print("\n== end to end, Amendment 3 scenarios ==")
V = A.analyse(synth("persists_lms3"))
case("M' (L_ms x3, S_id unchanged, tail 0.28): SEPARATE DEFECTS, unqualified",
     (V["D3_headline"].split(":")[0], "not converged: GainRaw" in V["D3_headline"]) == ("SEPARATE DEFECTS", False), True)
V = A.analyse(synth("converged_flat"))
case("Q (converged GainRaw flat at 0.09, leak 1.3x): SEPARATE DEFECTS, unqualified",
     (V["D3_headline"].split(":")[0], "not converged: GainRaw" in V["D3_headline"]) == ("SEPARATE DEFECTS", False), True)
V = A.analyse(synth("lagging"))
case("lagging GainRaw (leak-free loss 0.05): D3 read and carries GainRaw's qualifier",
     (V["D3_headline"].split(":")[0], "not converged: GainRaw" in V["D3_headline"]) == ("SEPARATE DEFECTS", True), True)
V = A.analyse(synth("solved_no_theta"))
case("SOLVED but theta unused: named", "GainRaw: theta reliance below 0.2 (does not use theta)" in V["D3_headline"], True)
print("\n== end to end, a label that flips under the re-score (GainPhase re-scored down to MapWM's accuracy) ==")
V = A.analyse(synth("rescore_flip"))
case("rescore flip: D4 line flagged", "FLAG: under the dropout-scale re-score D4_vs_N reads WORSE" in V["D4_headline"], True)
case("rescore flip: D4 verdict unchanged", V["D4_headline"].split(":")[0], "THE GAIN-PHASE MAP WORKS")
case("rescore flip: D3 gate flag on the D3 line, verdict unchanged",
     ("the D3 gate for GainRaw reads FAIL" in V["D3_headline"], V["D3_headline"].split(":")[0]) == (True, "SEPARATE DEFECTS"), True)
V = A.analyse(synth("d3_void"))
case("void D3 (MapWM without a leak) carries no flag or qualifier",
     (V["D3_headline"].split(":")[0], "FLAG" in V["D3_headline"]) == ("NO LEAK TO DISSOCIATE", False), True)
D.pop(("GainPhase", 15))
case("VOID on a missing run", list(A.analyse(D, out=lambda *a: None)), ["void"])
print(f"\nALL {'PASS' if ok else 'FAIL'}")

"""Descriptive pilot summary (seed 1690, never pooled): eval accuracies, training classes, probe secondaries."""
import json, re, sys
import numpy as np
import torch
sys.path.insert(0, "/home/prashr")
from mapformer.analyze_delta_free import ARMS, load_run, probe_run, kernel_checks
from mapformer.stats_core import classify_run

R = "/home/prashr/mapformer/runs/delta_free_pilot"; S = 1690
raw = json.load(open(f"{R}/_PILOT_RAW.json"))
torch.set_num_threads(8)
for a in ARMS:
    accs = {T: raw[f"0.0|{a}|{T}"][0][1] for T in (128, 512, 1024)}
    m, b = load_run(R, a, S)
    l = np.asarray(b["losses"]); c = classify_run(l)
    ep = [float(x) for x in re.findall(r"\| ([0-9.]+)s$", open(f"{R}/{a}_s{S}.log").read(), re.M)]
    p = {**probe_run(m, S), **kernel_checks(m)}
    print(f"{a:20s} acc T128 {accs[128]:.4f} T512 {accs[512]:.4f} T1024 {accs[1024]:.4f} | {c['registered']} tail {c['tail']:.4f} "
          f"AUC {l.mean():.4f} | s/epoch median {np.median(ep):.2f} (n {len(ep)})")
    print(f"    probe acc {p['acc']:.4f} verify {p['verify']:.1e} | R1 {p['acc_R1']:.3f} R2 {p['acc_R2']:.3f} R3+ {p['acc_R3+']:.3f} "
          f"none {p['acc_none']:.3f} | off1 mass {p['mass_off1']:.3f}")
    print(f"    delta: deadzone {p['deadzone_share']:.3f} repinned {p['repinned_share']:.3f} |wrap| {p['abs_wrapped_mean']:.3f} "
          f"moved {p.get('moved_abs_mean', float('nan')):.3f} (max {p.get('moved_abs_max', float('nan')):.3f}) "
          f"outside {p['outside_clamp_share']:.3f}"
          + (f" | K3>K1 {p['frac_K3_gt_K1']:.3f} argmax " + " ".join(f"{d}:{n}" for d, n in enumerate(p['argmax_offset_hist']) if n)
             if 'frac_K3_gt_K1' in p else f" | C {p['C']:.3f} C0 {p['C0']:.3f} same-cell mass {p['mass_cell']:.3f}"))

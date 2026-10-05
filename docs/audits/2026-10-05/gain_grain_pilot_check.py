"""GAIN_GRAIN pilot readout (gain_grain_pilot.sh): (a) reproduction of the stored MAPPOPE_PAIR MapPoPE-Pair s10 run
through train_gain_grain (per-epoch losses and final weights); (b) per-epoch wall time at 8 concurrent jobs, per arm;
(c) sanity: run class, final-5% loss and T=128 held-out accuracy (100 walks) of the pilot runs (seeds 100-101, outside
the batch). Output: gain_grain_pilot_check_out.txt."""
import glob, json, os, re, sys
import numpy as np
import torch
sys.path.insert(0, "/home/prashr")
from mapformer.stats_core import classify_run

REPO = "/home/prashr/mapformer"; P = f"{REPO}/runs/gain_grain_pilot"
a = torch.load(f"{P}/repro/MapPoPE-Pair_s10/MapPoPE-Pair.pt", map_location="cpu", weights_only=False)
b = torch.load(f"{REPO}/runs/mappope_pair/p0/MapPoPE-Pair_s10/MapPoPE-Pair.pt", map_location="cpu", weights_only=False)
x, y = np.array(a["losses"]), np.array(b["losses"])
wd = max((a["model_state_dict"][k] - b["model_state_dict"][k]).abs().max().item() for k in b["model_state_dict"])
eq = int((x == y).sum())
print(f"(a) reproduction MapPoPE-Pair s10 (train_gain_grain vs stored mappope_pair): {eq}/{len(y)} epochs bitwise equal, "
      f"max |loss diff| {np.abs(x - y).max():.2e}, max |final weight diff| {wd:.2e} -> {'PASS' if eq == len(y) and wd == 0 else 'FAIL'}")
print("\n(b) per-epoch wall time at 8 concurrent (4 per GPU, 2 RTX 4090), median over logged epochs 50-300:")
for f in sorted(glob.glob(f"{P}/*/*.log")):
    t = [float(m.group(2)) for m in re.finditer(r"Epoch\s+(\d+)/300 .*\| ([\d.]+)s", open(f).read()) if int(m.group(1)) >= 50]
    print(f"  {os.path.basename(f):28s} {np.median(t):.2f} s/epoch  (n={len(t)})")
print("\n(c) pilot runs (seeds 100-101; READ after the prereg commit d332718):")
J = json.load(open(f"{P}/PILOT_EVAL.json"))
for k, rows in sorted(J.items()):
    arm = k.split("|")[1]
    for s, acc, nll in rows:
        c = classify_run(torch.load(f"{P}/p0/{arm}_s{s}/{arm}.pt", map_location="cpu", weights_only=False)["losses"])
        print(f"  {arm:17s} s{s}: {c['registered']:10s} final-5% loss {c['tail']:.4f}  T=128 acc {acc:.4f}  nll {nll:.4f}")

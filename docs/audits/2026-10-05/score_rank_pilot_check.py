"""SCORE_RANK pilot readout (runs/score_rank_pilot, written by score_rank_pilot.sh). (1) Reproduction: per-epoch losses
of Vanilla s0 trained through train_score_rank vs the stored runs/rank_mi/p0/Vanilla_s0 (pass = bitwise equal on every
epoch). (2) Timing: the per-epoch wall time train.py prints (one epoch, every 5th) at 8 concurrent (4/GPU) and 4
concurrent (2/GPU). (3) Pilot outcome at seeds 100/101 (40-epoch cosine schedule, NOT the batch recipe): final-5% loss
and class. Output: score_rank_pilot_check_out.txt next to this file."""
import glob, os, re, sys
import numpy as np
import torch
sys.path.insert(0, "/home/prashr")
from mapformer.stats_core import classify_run  # noqa: E402

P = "/home/prashr/mapformer/runs/score_rank_pilot"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_rank_pilot_check_out.txt")
lines = []
def log(s): print(s, flush=True); lines.append(s)
x = np.array(torch.load(f"{P}/repro/Vanilla_s0/Vanilla.pt", map_location="cpu", weights_only=False)["losses"])
y = np.array(torch.load("/home/prashr/mapformer/runs/rank_mi/p0/Vanilla_s0/Vanilla.pt", map_location="cpu", weights_only=False)["losses"])
ok = len(x) == len(y) == 900 and np.array_equal(x, y)
log(f"(1) reproduction Vanilla s0: {len(x)} vs {len(y)} epochs, max |diff| {np.abs(x - y).max():.2e}, bitwise equal: {ok} -> {'PASS' if ok else 'FAIL'}")
log("(2) per-epoch wall time (s), from the train.py prints:")
for tag in ("c8", "c4"):
    for f in sorted(glob.glob(f"{P}/{tag}/*.log")):
        t = [float(v) for v in re.findall(r"\| ([0-9.]+)s\n", open(f).read())]
        log(f"    {tag} {os.path.basename(f)[:-4]:24s} median {np.median(t):.1f} s/epoch over {len(t)} prints")
log("(3) pilot outcome (40-epoch schedule, seeds outside the batch):")
for f in sorted(glob.glob(f"{P}/c*/*/*.pt")):
    c = classify_run(torch.load(f, map_location="cpu", weights_only=False)["losses"])
    log(f"    {f[len(P) + 1:]:50s} final-5% loss {c['tail']:.4f} {c['registered']}")
open(OUT, "w").write("\n".join(lines) + "\n")

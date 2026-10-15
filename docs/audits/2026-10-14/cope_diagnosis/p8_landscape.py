"""P8: constructed landscape along delta (no learned weights). Gates per pair alpha = 1 + delta, beta = 1 - delta
(g+ = 0.9 + delta, g- = 0.9 - delta, obs 0.1, as the c3 construction), period-2 triangle table of amplitude M.
Reads accuracy and attention-mass loss on local and wrap-only revisits."""
import sys
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis")
import numpy as np, torch
torch.set_num_threads(8)
import lib_cope as lc
from func_cope import func_eval, make_table, gate_table

trajs = lc.trajectories(100)
print("M | delta | 32*delta mod 2 | acc all | acc local | acc wrap-only | loss all (-log share of obs-key attention mass on the right obs)")
for M in (1.0, 3.0, 10.0, 30.0):
    cp = make_table(M)
    for d in (0.0, 0.004, 0.008, 0.016, 0.024, 0.03125, 0.04, 0.05, 0.058, 0.0625):
        acc, loss, rows = func_eval(gate_table(0.9 + d, 0.9 - d, 0.1), cp, trajs)
        st = lc.strat_table(rows)
        print(f"{M:5.1f} | {d:.4f} | {32*d % 2:.3f} | {acc:.4f} | {st['local'][0]:.4f} | {st['wrap_only'][0]:.4f} | {loss:.4f}",
              flush=True)

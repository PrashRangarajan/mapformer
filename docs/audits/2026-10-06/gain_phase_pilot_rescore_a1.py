"""Amendment 1, CPU only: re-run the evaluation (now with theta reliance, acc_rescored) and the registered analysis on the
8 pilot checkpoints (runs/gain_phase_pilot/p0, 30-epoch schedule) -- a real-checkpoint check that the D3 convergence
gate reads the unconverged gain arms as UNMEASURED and the budget qualifiers print. Outputs in runs/gain_phase_pilot:
GAIN_PHASE_PILOT_EVAL_A1.json, eval_out_A1.txt (stdout here), analysis_out_A1.txt."""
import contextlib
import sys

import torch

sys.path.insert(0, "/home/prashr")
torch.set_num_threads(6)
import mapformer.analyze_gain_phase as A
from mapformer.eval_gain_phase import main as eval_main

R = "/home/prashr/mapformer/runs/gain_phase_pilot"
eval_main("cpu", runs=f"{R}/p0", seeds=[110, 111], out=f"{R}/GAIN_PHASE_PILOT_EVAL_A1.json")
A.SEEDS = [110, 111]
with open(f"{R}/analysis_out_A1.txt", "w") as f, contextlib.redirect_stdout(f):
    A.analyse(A.load_runs(f"{R}/GAIN_PHASE_PILOT_EVAL_A1.json", f"{R}/p0"), seeds=[110, 111], E=30)

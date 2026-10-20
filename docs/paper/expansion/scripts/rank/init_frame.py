"""ANALYSIS (post hoc, CPU): does the INITIAL frame conditioning / drift predict which rank-D runs solve at T = 1024?
Rebuilds each run's initial weights as train_variant does (torch.manual_seed(seed); np.random.seed(seed); env; model)
and reads head_geometry at init; outcome = registered class of the stored run. Output init_frame_out.txt."""
import glob, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
torch.set_num_threads(2)
from rank_geom import head_geometry  # noqa
from mapformer.train_variant import VARIANT_MAP  # noqa
from mapformer.stats_core import classify_run  # noqa
from mapformer.environment import GridWorld  # noqa
from mapformer.environment_nd import GridWorldND  # noqa
from scipy.stats import mannwhitneyu  # noqa
R = "/home/prashr/mapformer/runs"
SETS = [("score_rank shared r2", f"{R}/score_rank/p0/Vanilla_s{{s}}/Vanilla.pt", "Vanilla", range(10, 22), "torus", 64),
        ("rank_mi per-head r2", f"{R}/rank_mi/p0/Vanilla_r2ph_s{{s}}/Vanilla_r2ph.pt", "Vanilla_r2ph", range(8), "torus", 64),
        ("rank_sep block-diag r4", f"{R}/rank_sep/p0/Vanilla_r4mibd_s{{s}}/Vanilla_r4mibd.pt", "Vanilla_r4mibd", range(8), "torus", 64),
        ("rank_nd D2 G32 r2", f"{R}/rank_nd/D2/Vanilla_r2ph_s{{s}}/Vanilla_r2ph.pt", "Vanilla_r2ph", range(8), "nd", 32),
        ("rank3 per-head r3", f"{R}/rank3/p0/Vanilla_r3ph_s{{s}}/Vanilla_r3ph.pt", "Vanilla_r3ph", range(8), "torus", 64)]
L = []; pooled = {True: [], False: []}
for lab, pat, arm, seeds, env_kind, G in SETS:
    rows = []
    for s in seeds:
        b = torch.load(pat.format(s=s), map_location="cpu", weights_only=False)
        torch.manual_seed(s); np.random.seed(s)
        env = (GridWorld(size=G, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=s) if env_kind == "torus"
               else GridWorldND(dims=2, size=G, n_obs_types=16, p_empty=0.5, seed=s))
        m = VARIANT_MAP[arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=G).eval()
        g = head_geometry(m)
        sol = classify_run(b["losses"])["registered"] == "SOLVED"
        mi = max(h["indep"] for h in g); kmin = min(h["kappa"] for h in g)
        rows.append((s, sol, mi, kmin))
        if "r3" not in lab:
            pooled[sol].append(mi)
    L.append(f"{lab}: " + "; ".join(f"s{s} {'SOLV' if v else 'fail'} max-indep0 {mi:.2f} min-kappa0 {k:.2f}" for s, v, mi, k in rows))
p = mannwhitneyu(pooled[True], pooled[False], alternative="greater").pvalue if pooled[True] else float("nan")
L.append(f"rank-D pooled: best initial head indep, solved {np.round(sorted(pooled[True]), 2).tolist()} vs unsolved median "
         f"{np.median(pooled[False]):.2f} (n {len(pooled[False])}); Mann-Whitney one-sided p {p:.3f}")
open(f"{HERE}/init_frame_out.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))

# ---- initial drift (min over heads of kappa at init) vs outcome, rank-D sets (Prop. 3 predicts lower initial drift
# in the runs that solve). Seeds 0-7 share the base across the rank_mi / rank_sep / rank_nd sets, so the pooled test
# is not independent; the score_rank set (seeds 10-21) is.
import re
txt = open(f"{HERE}/init_frame_out.txt").read().splitlines()
sol, uns, sol_sr, uns_sr = [], [], [], []
for line in txt:
    if line.startswith("rank3") or line.startswith("rank-D"):
        continue
    for s, v, k in re.findall(r"s(\d+) (SOLV|fail) max-indep0 [\d.]+ min-kappa0 ([\d.]+)", line):
        (sol if v == "SOLV" else uns).append(float(k))
        if line.startswith("score_rank"):
            (sol_sr if v == "SOLV" else uns_sr).append(float(k))
p_all = mannwhitneyu(sol, uns, alternative="less").pvalue
p_sr = mannwhitneyu(sol_sr, uns_sr, alternative="less").pvalue
extra = [f"initial drift (min-kappa0): solved {sorted(sol)} vs unsolved median {np.median(uns):.2f} (n {len(uns)}); "
         f"one-sided Mann-Whitney p {p_all:.3f} (pooled, seeds shared across sets); score_rank alone (independent seeds) "
         f"solved {sorted(sol_sr)} vs unsolved {sorted(uns_sr)}, p {p_sr:.3f}"]
open(f"{HERE}/init_frame_out.txt", "a").write("\n".join(extra) + "\n"); print("\n".join(extra))

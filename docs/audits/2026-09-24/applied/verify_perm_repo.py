"""stats_core (repo) vs the pristine HEAD analyze_rank_matched.perm / perm_ci.
p-values compared with ==, CIs with ==, on every committed rank JSON contrast, on a
shift grid, and on random tie-heavy cases. Also checks the row sums the statistic is
built from are bitwise equal to the old per-combination x[list(idx)].sum()."""
import sys, json, time, itertools, importlib.util
import numpy as np
SP = __import__("os").environ["MF_SCRATCH"]  # holds base/mapformer: a git worktree of the pre-change commit
spec = importlib.util.spec_from_file_location("oldarm", f"{SP}/base/mapformer/analyze_rank_matched.py")
O = importlib.util.module_from_spec(spec); spec.loader.exec_module(O)
sys.path.insert(0, "/home/prashr")
from mapformer import stats_core as SC
from mapformer import analyze_rank_matched as N
M = "/home/prashr/mapformer"
n = ok = 0
cases = []
for js in ["RANK_MATCHED.json", "RANK_MATCHED_e900.json", "RANK_MATCHED_e900c.json", "RANK_PROJ_TRAIN.json",
           "RANK_PROJ_FROZEN.json", "RANK_SWEEP.json"]:
    J = json.load(open(f"{M}/{js}"))
    keys = sorted(J)
    for k1, k2 in itertools.combinations(keys, 2):
        if k1.split("|")[-1] != k2.split("|")[-1]: continue           # same length
        for col in (1, 2):
            a = [x[col] for x in sorted(J[k1])]; b = [x[col] for x in sorted(J[k2])]
            if None in a or None in b or len(a) + len(b) > 16: continue
            cases.append((js, k1, k2, col, a, b))
print(len(cases), "committed contrasts")
t0 = time.perf_counter()
for js, k1, k2, col, a, b in cases:
    for sh in (0.0, 0.05, 0.055, 0.085, 0.15, 0.155, -0.1):
        po = O.perm(a, b, sh); pn = SC.perm2_p(a, b, sh)["p"]; pm = N.perm(a, b, sh)
        n += 1; s = po == pn == pm; ok += s
        if not s: print("FAIL", js, k1, k2, col, sh, po, pn, pm)
rng = np.random.default_rng(0)
for _ in range(300):
    na, nb = rng.integers(2, 9), rng.integers(2, 9)
    pool = rng.choice([0.5, 0.75, 0.875, 0.9, 1.0, rng.random()], size=na + nb) if rng.random() < .7 else rng.random(na + nb)
    a, b = pool[:na].tolist(), pool[na:].tolist(); sh = float(rng.choice([0, 0.005 * rng.integers(-40, 40)]))
    po = O.perm(a, b, sh); pn = SC.perm2_p(a, b, sh)["p"]
    n += 1; s = po == pn; ok += s
    if not s: print("FAIL random", a, b, sh, po, pn)
print(f"p-values: {ok}/{n} identical ({time.perf_counter()-t0:.1f}s)")
# row sums: old x[list(idx)].sum() vs x[idx].sum(axis=1), bitwise, on real data
J = json.load(open(f"{M}/RANK_MATCHED_e900.json"))
bad = tot = 0
for T in (512, 1024, 2048):
    for col in (1, 2):
        a = np.array([x[col] for x in sorted(J[f"0.0|Vanilla|{T}"])]); b = np.array([x[col] for x in sorted(J[f"0.0|Vanilla_r4|{T}"])])
        for sh in np.arange(-0.6, 0.6 + 1e-9, 0.005):
            x = np.concatenate([a, b - sh]); idx = SC._relabellings(16, 8, 250_000, 200_000, 0)[0]
            new = x[idx].sum(axis=1); old = np.array([x[list(i)].sum() for i in idx[::97]])
            tot += len(old); bad += int((new[::97] != old).sum())
print(f"row sums: {tot - bad}/{tot} bitwise equal (sampled every 97th relabelling, 241 shifts x 6 contrasts)")
# CI
for js, key in (("RANK_MATCHED_e900.json", 1024), ("RANK_MATCHED_e900c.json", 1024), ("RANK_MATCHED.json", 1024)):
    J = json.load(open(f"{M}/{js}"))
    a = [x[1] for x in sorted(J[f"0.0|Vanilla|{key}"])]; b = [x[1] for x in sorted(J[f"0.0|Vanilla_r4|{key}"])]
    t = time.perf_counter(); co = O.perm_ci(a, b); t1 = time.perf_counter(); cn = N.perm_ci(a, b); t2 = time.perf_counter()
    walk = SC.perm2_ci(a, b, step=0.005)
    print(f"{js} CI HEAD {co} ({t1-t:.2f}s)  repo {cn[:2]} clipped={cn[2]} ({t2-t1:.3f}s)  {'IDENTICAL' if co == cn[:2] else 'DIFFERENT'}"
          f"  | unclipped walk from d0: [{walk['lo']:+.4f}, {walk['hi']:+.4f}]")

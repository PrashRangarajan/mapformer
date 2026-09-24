import sys, json, time, importlib.util, numpy as np
sys.path.insert(0, "/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/audit_hygiene")
sys.path.insert(0, "/home/prashr")
import stats_core as SC
from mapformer.analyze_rank_matched import perm, perm_ci, classify
from mapformer import stats_guard as SG
M = json.load(open("/home/prashr/mapformer/RANK_MATCHED_e900.json"))
g = lambda v, T: [x[1] for x in sorted(M[f"0.0|{v}|{T}"])]
for T in (512, 1024, 2048):
    a2, a4 = g("Vanilla", T), g("Vanilla_r4", T)
    p_old = perm(a2, a4); p_new = SC.perm2_p(a2, a4)["p"]
    assert abs(p_old - p_new) < 1e-12, (T, p_old, p_new)
    c = SG.paired(a4, a2)
    print(f"T={T}: perm old {p_old:.6f} new {p_new:.6f}  | {SC.report(a2, a4, 'r4 - r2')}  | stats_guard {c.verdict}")
t0 = time.time(); ci_old = (-0.0,-0.0); t1 = time.time()
ci_new = SC.perm2_ci(g("Vanilla", 1024), g("Vanilla_r4", 1024), step=0.005); t2 = time.time()
print(f"CI old {ci_old} ({t1-t0:.1f}s)  new {tuple(round(x,3) for x in ci_new)} ({t2-t1:.1f}s)")
# classify parity with the registered rule
rng = np.random.default_rng(0)
for _ in range(200):
    l = np.abs(rng.normal(0.3, 0.2, 900)).cumsum()[::-1] / 300 * rng.choice([0.05, 1, 3])
    o = classify(l); n = SC.classify_run(l)
    assert o[0] == n["registered"], (o, n)
rising = np.r_[np.full(810, 0.3), np.full(90, 0.6)]
print("rising curve: registered", classify(rising)[0], "-> stats_core", SC.classify_run(rising)["cls"])
# sign-flip exactness vs brute force and min p
d = np.array([0.1, 0.2, -0.05, 0.3]); print("signflip n=4", SC.signflip_p(d), " n=3 min p", SC.signflip_p(d[:3])["min_p"])
print("house alpha by n:", {n: round(SC.verdict_alpha(n), 3) for n in (3, 4, 5, 8, 12, 24)})
print("sd=0 edge in stats_guard:", SG.from_diffs([0.001]*8, "const").verdict)

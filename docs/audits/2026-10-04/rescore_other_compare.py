"""Registered vs re-scored (attention x 1/(1-p)) for text world, TW_NORMSTEP, H3 (cancel), leak, rank_nd, rank_wrap:
per-arm means and the registered accuracy contrasts (exact permutation). Inputs from rescore_trainer_evals.py and
rescore_nd.sh (runs_rescore/)."""
import json
import numpy as np
from mapformer.stats_core import perm2_p

R = "/home/prashr/mapformer"; O = f"{R}/runs_rescore"


def arms(d, key=lambda k: k.rsplit("_s", 1)[0]):
    out = {}
    for k, v in sorted(d.items(), key=lambda kv: int(kv[0].rsplit("_s", 1)[1])):
        out.setdefault(key(k), []).append(v)
    return {a: (np.array([x[1] for x in v]), np.array([x[2] for x in v])) for a, v in out.items()}


def con(label, A, x, y):
    (rx, ax), (ry, ay) = A[x], A[y]
    dr, pr = ry.mean() - rx.mean(), perm2_p(rx, ry)["p"]; da, pa = ay.mean() - ax.mean(), perm2_p(ax, ay)["p"]
    flip = (pr < 0.05) != (pa < 0.05) or np.sign(dr) != np.sign(da)
    print(f"  {label:42s} registered {dr:+.4f} (p {pr:.4f})   rescored {da:+.4f} (p {pa:.4f})  {'<-- CHANGES' if flip else ''}")


def table(A):
    for a, (r, s) in A.items():
        print(f"  {a:22s} {r.mean():.4f} -> {s.mean():.4f}   max per-seed change {np.max(s - r):+.4f} / {np.min(s - r):+.4f}")


print("== text world (registered A: path 1L - RoPE 1L) ==")
A = arms(json.load(open(f"{O}/textworld.json"))); table(A)
con("path 1L - RoPE 1L", A, "RoPE_L1", "Vanilla_r4_L1"); con("path 1L - RoPE 2L", A, "RoPE_L2", "Vanilla_r4_L1")
con("RoPE 2L - RoPE 1L", A, "RoPE_L1", "RoPE_L2")
F = {k: v for k, v in json.load(open(f"{O}/textworld.json")).items() if int(k.rsplit("_s", 1)[1]) >= 2}
con("fresh seeds s2-s7: path 1L - RoPE 1L", arms(F), "RoPE_L1", "Vanilla_r4_L1")

print("\n== TW_NORMSTEP (registered A: NormStep - MapWM) ==")
A = arms(json.load(open(f"{O}/tw_normstep.json"))); table(A)
con("NormStep - MapWM", A, "MapWM", "NormStep"); con("NormStepNB - MapWM", A, "MapWM", "NormStepNB")
con("DirOnly - MapWM", A, "MapWM", "DirOnly")

print("\n== H3 cancel (T=128) ==")
A = arms(json.load(open(f"{O}/cancel.json"))); PS = ["0.5", "0.75", "0.9", "1.0"]
for p in PS:
    pm_r, pm_s = A[f"Vanilla_L1_p{p}"][0].mean(), A[f"Vanilla_L1_p{p}"][1].mean()
    g = [(pm_r - A[f"RoPE_L{L}_p{p}"][0].mean(), pm_s - A[f"RoPE_L{L}_p{p}"][1].mean()) for L in (1, 2, 3)]
    kr = next((L for L, x in zip((1, 2, 3), g) if x[0] <= 0.01), ">3"); ks = next((L for L, x in zip((1, 2, 3), g) if x[1] <= 0.01), ">3")
    print(f"  p={p}: path1 {pm_r:.4f}->{pm_s:.4f}; index 1L {A[f'RoPE_L1_p{p}'][0].mean():.4f}->{A[f'RoPE_L1_p{p}'][1].mean():.4f}; "
          f"G2 {g[1][0]:+.4f}->{g[1][1]:+.4f}; G3 {g[2][0]:+.4f}->{g[2][1]:+.4f}; k {kr} -> {ks}")
for i, p in enumerate(PS):
    for q in PS[i + 1:]:
        con(f"index 1L a1({q}) - a1({p})", A, f"RoPE_L1_p{p}", f"RoPE_L1_p{q}")

print("\n== leak (x1 unseen-object accuracy; registered from LEAK.json) ==")
L = json.load(open(f"{O}/leak.json")); reg = json.load(open(f"{R}/LEAK.json"))["res"]
for k, v in L.items():
    arm, s = k.rsplit("_s", 1); v[0] = reg[arm][int(s)]["test|x1"]["intact"]
bad = [k for k, v in L.items() if abs(v[0] - v[1]) > 1e-9]
print(f"  rerun reproduces LEAK.json on {len(L) - len(bad)}/{len(L)}")
A = arms(L); table(A)
con("ActOnly - MapWM", A, "MapWM", "ActOnly"); con("NormStep - MapWM", A, "MapWM", "NormStep")

print("\n== rank_nd (T=1024) ==")
def nd(path, cfg):
    reg = json.load(open(path))
    return reg[cfg]["acc"]
for D, (a, b) in (("D2", ("Vanilla_r2ph", "Vanilla_r3ph")), ("D3", ("Vanilla_r3ph", "Vanilla_r4ph"))):
    regj = json.load(open(f"{R}/RANK_ND.json"))[D]["acc"]; nj = json.load(open(f"{O}/RANK_ND_none.json"))[D]["acc"]
    aj = json.load(open(f"{O}/RANK_ND_auto.json"))[D]["acc"]; A = {}
    for v in (a, b):
        r = np.array([regj[v]["1024"][str(s)] for s in range(8)]); n = np.array([nj[v]["1024"][str(s)] for s in range(8)])
        assert np.max(np.abs(r - n)) < 1e-12, ("nd rerun mismatch", D, v)
        A[v] = (r, np.array([aj[v]["1024"][str(s)] for s in range(8)]))
    table(A); con(f"{D}: rank D+1 - rank D", A, a, b)
print("  (rerun reproduces RANK_ND.json exactly)")

print("\n== rank_wrap (T=1024) ==")
A = {}
for D, N, v in ((2, 32, "Vanilla_r3ph"), (2, 10, "Vanilla_r3ph"), (3, 18, "Vanilla_r4ph"), (3, 10, "Vanilla_r4ph")):
    regj = json.load(open(f"{R}/runs/rank_wrap/N{N}/EVAL_D{D}.json"))[f"D{D}"]["acc"][v]["1024"]
    nj = json.load(open(f"{O}/RANK_WRAP_D{D}_N{N}_none.json"))[f"D{D}"]["acc"][v]["1024"]
    aj = json.load(open(f"{O}/RANK_WRAP_D{D}_N{N}_auto.json"))[f"D{D}"]["acc"][v]["1024"]
    r = np.array([regj[str(s)] for s in range(8)]); n = np.array([nj[str(s)] for s in range(8)])
    assert np.max(np.abs(r - n)) < 1e-12, ("wrap rerun mismatch", D, N)
    A[f"{D}{'L' if N > 10 else 'H'}"] = (r, np.array([aj[str(s)] for s in range(8)]))
table(A); print("  (rerun reproduces EVAL_D*.json exactly)")
con("3D wrap: 3L - 3H", A, "3H", "3L"); con("2D wrap: 2L - 2H", A, "2H", "2L"); con("high wrap: 3H - 2H", A, "2H", "3H")

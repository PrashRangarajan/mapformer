"""Original perm/perm_ci vs vectorised, on the committed RANK_MATCHED_e900.json and on
random data (including exact ties). Every p-value and CI bound must be equal (==)."""
import sys, json, time, importlib
import numpy as np
sys.path.insert(0, "/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/audit_efficiency")
O = importlib.import_module("mapformer.analyze_rank_matched"); N = importlib.import_module("mfprop.analyze_rank_matched")
M = json.load(open("/home/prashr/mapformer/RANK_MATCHED_e900.json"))
S = list(range(8)); ok = True
get = lambda v, T, i: [dict((x[0], x) for x in M[f"0.0|{v}|{T}"])[s][i] for s in S]
for T in (512, 1024, 2048):
    for i in (1, 2):
        a, b = get("Vanilla", T, i), get("Vanilla_r4", T, i)
        ok &= O.perm(a, b) == N.perm(a, b)
a, b = get("Vanilla", 1024, 1), get("Vanilla_r4", 1024, 1)
t = time.perf_counter(); co = O.perm_ci(a, b); t1 = time.perf_counter(); cn = N.perm_ci(a, b); t2 = time.perf_counter()
ok &= co == cn
print(f"perm_ci original {co} in {t1-t:.2f}s, vectorised {cn} in {t2-t1:.3f}s")
rng = np.random.default_rng(0)
for _ in range(30):
    a = rng.integers(0, 5, 8) / 7.0; b = rng.integers(0, 5, 8) / 7.0      # heavy ties
    for sh in (0.0, 0.125, -0.3):
        ok &= O.perm(a, b, sh) == N.perm(a, b, sh)
print("ALL IDENTICAL" if ok else "MISMATCH")

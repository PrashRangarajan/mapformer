"""Independent reimplementation (from the LRSWEEP_PREREG.md text) of the selection rule, the selection-aware null and the
reading; compared with analyze_lrsweep on random inputs."""
import itertools, math, numpy as np
from mapformer import analyze_lrsweep as L
from mapformer.analyze_robust import SLB, NSC, UNM, NLB

def sel_ref(solved, score, members):
    best = None
    for j in range(solved.shape[1]):                         # rates ascending
        k = sum(solved[p, j] for p in members); m = sum(score[p, j] for p in members) / len(members)
        key = (k, -round(m, 9))
        if best is None or key > best[0]:                    # strict: equal key keeps the lower rate
            best = (key, j)
    return best[1]

def b1_ref(S, SC, A, n):
    P = list(range(2 * n)); obs_ao = tuple(range(n, 2 * n))
    def stat(ao):
        so = [p for p in P if p not in ao]; ra, rs = sel_ref(S, SC, ao), sel_ref(S, SC, so)
        return (sum(S[p, ra] for p in ao) - sum(S[p, rs] for p in so),
                np.mean([A[p, ra] for p in ao]) - np.mean([A[p, rs] for p in so]))
    tk, ta = stat(obs_ao); hk = ha = M = 0
    for ao in itertools.combinations(P, n):
        k, a = stat(ao); M += 1; hk += abs(k) >= abs(tk); ha += abs(a) >= abs(ta) - 1e-12
    return M, hk / M, ha / M, sel_ref(S, SC, obs_ao), sel_ref(S, SC, tuple(range(n)))

rng = np.random.default_rng(7)
# 1) selection rule on random small groups with forced ties
bad = 0
for _ in range(3000):
    R = rng.integers(1, 6); m = rng.integers(1, 5)
    S = rng.random((m, R)) < rng.random(); SC = rng.choice([0.1, 0.2, 0.3], size=(m, R))
    if L.select(S, SC, np.arange(m)[None])[0] != sel_ref(S, SC, list(range(m))): bad += 1
print("selection rule: 3000 random groups (ties forced), mismatches", bad)
# 2) selection-aware null at n = 4 (C(8,4) = 70) on random data: exact match of p-values and selected rates
bad = 0
for _ in range(60):
    n, R = 4, rng.integers(1, 5)
    S = rng.random((2 * n, R)) < rng.random(); SC = rng.choice([0.01, 0.2, 0.25], size=(2 * n, R)); A = rng.random((2 * n, R))
    t = L.b1_test(S, SC, A, n); M, pk, pa, ra, rs = b1_ref(S, SC, A, n)
    ok = (t["n_relabel"] == M and abs(t["p_k_sel"] - pk) < 1e-12 and abs(t["p_acc_sel"] - pa) < 1e-12 and t["r_ao"] == ra and t["r_so"] == rs)
    bad += not ok
print("selection-aware null, n = 4, 60 random datasets: mismatches", bad)
# 3) n = 8 enumeration size and structure
idx, exact = L._relabellings(16, 8, 250_000, 200_000, 0)
rows = {tuple(r) for r in idx}
print("n = 8: relabellings", len(idx), "distinct", len(rows), "exact", exact, "= C(16,8)", math.comb(16, 8),
      "| observed AO set present:", tuple(range(8, 16)) in rows)
S = rng.random((16, 4)) < 0.5; SC = rng.random((16, 4)); A = rng.random((16, 4)); t = L.b1_test(S, SC, A, 8)
M, pk, pa, ra, rs = b1_ref(S, SC, A, 8)
print(f"n = 8 random dataset: module p {t['p_k_sel']:.6f}/{t['p_acc_sel']:.6f} vs reference {pk:.6f}/{pa:.6f}; M {t['n_relabel']} vs {M}")
# 4) reading: independent from the prereg text, over every label x every (r_ao, r_so) on the 4- and 5-rate grids
def read_ref(lab, ra, rs, rates):
    q = " [AO's best rate is the grid's top rate: the AO optimum is not bracketed; 'best' = best TESTED]" if ra == max(rates) else ""
    if lab.startswith("UNREADABLE"): return "UNREADABLE (the signed arm solves < 4/8 at every rate)" + q
    if lab == SLB: return "COST AT BEST RATES" + q
    if lab in (NSC, NLB):
        rev = " (reversed: NON-NEGATIVE LEARNS BETTER)" if lab == NLB else ""
        base = (f"COST CLOSES AT HIGHER LR (AO best {ra:g} > SO best {rs:g})" if ra > rs else f"NO COST AT BEST RATES (AO best {ra:g}, SO best {rs:g})")
        return base + rev + q
    return "UNMEASURED" + q
bad = tot = 0
for rates in ([1e-3, 3e-3, 1e-2, 3e-2], [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]):
    for lab in (SLB, NSC, NLB, UNM, "UNREADABLE (SO SOLVED 3 < 4)"):
        for ra in rates:
            for rs in rates:
                tot += 1; bad += L.reading(lab, ra, rs, rates) != read_ref(lab, ra, rs, rates)
print("reading: combinations", tot, "mismatches", bad)

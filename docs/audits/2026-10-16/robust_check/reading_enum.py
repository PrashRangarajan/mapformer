import itertools, collections
from mapformer.analyze_robust import reading, cell_label, CELLS, SLB, NSC, UNM
U = "UNREADABLE (SO SOLVED 3 < 4)"
def mine(L):   # independent, from the prereg text
    if L["ring|def"] == U: return "UNREADABLE"
    nsc = [c for c in CELLS if L[c] == NSC]
    if nsc:
        r = any(c.startswith("ring") for c in nsc); t = "torus|d256L2" in nsc
        return "RECIPE-DEPENDENT AND SCALE-DEPENDENT" if r and t else ("RECIPE-DEPENDENT" if r else "SCALE-DEPENDENT")
    if all(L[c] == SLB for c in CELLS): return "ROBUST"
    nu = sum(L[c] == U for c in CELLS)
    if nu >= 2: return "INSUFFICIENT"
    if nu == 1 and all(L[c] == SLB for c in CELLS if L[c] != U): return "ROBUST WHERE READABLE"
    return "PARTLY UNMEASURED"
cnt = collections.Counter(); bad = 0
for labs in itertools.product((SLB, NSC, UNM, U), repeat=5):
    L = dict(zip(CELLS, labs)); got = reading(L); exp = mine(L)
    head = got.split(" (")[0]
    if head != exp: bad += 1; print("MISMATCH", labs, got, exp)
    cnt[exp] += 1
print("combos 1024, mismatches", bad); print(dict(cnt))
# def NSC case
L = dict(zip(CELLS, (NSC, SLB, SLB, SLB, SLB))); print("def NSC ->", reading(L))
L = dict(zip(CELLS, (UNM, SLB, SLB, SLB, SLB))); print("def UNM ->", reading(L))
# cell-label boundary checks
n = 8
print("SO 8 AO 4 equal acc:", cell_label([1]*8, [1]*8, 4, 8, n)[0])
print("SO 8 AO 3:", cell_label([1]*8, [1]*8, 3, 8, n)[0])
print("SO 7 AO 2:", cell_label([1]*8, [1]*8, 2, 7, n)[0])
print("SO 4 AO 3 acc equal:", cell_label([0.9]*8, [0.9]*8, 3, 4, n)[0])
print("SO 4 AO 4 acc equal (weak NSC):", cell_label([0.8]*8, [0.8]*8, 4, 4, n)[0])
print("AO solves MORE (AO 8 SO 4), acc AO higher:", cell_label([1]*8, [0.9]*8, 8, 4, n)[0])
print("AO solves more but Fisher p<.05 and SO ahead impossible; AO 0 SO 4 ->", cell_label([0.8]*8, [0.9]*8, 0, 4, n))

"""Winner's curse on AO@best and a selection-free (cross-fit) guard for NO SIGN COST: each half of the seeds is scored at the
rate selected on the other half (rule 1), pooled over both halves -> a selection-free count out of 8 per arm."""
import sys, numpy as np
from sim_null import draw, n, R
from mapformer.analyze_lrsweep import b1_test, b1_decide, select

def crossfit_counts(S, SC, A):
    h = n // 2; ka = ks = 0; aa = []; sa = []
    for sel, tst in ((np.arange(h), np.arange(h, n)), (np.arange(h, n), np.arange(h))):
        ra = select(S, SC, (sel + n)[None])[0]; rs = select(S, SC, sel[None])[0]
        ka += S[tst + n, ra].sum(); ks += S[tst, rs].sum(); aa += list(A[tst + n, ra]); sa += list(A[tst, rs])
    return ka, ks, np.mean(aa), np.mean(sa)

def run(name, p_so, p_ao, sims, seed, **kw):
    rng = np.random.default_rng(seed); c = dict(nsc=0, nsc_cf=0, slb=0); kb = []; kc = []
    for _ in range(sims):
        S, SC, A = draw(rng, p_so, p_ao, **kw)
        t = b1_test(S, SC, A, n); lab, P = b1_decide(t, A, S, n)
        ka, ks, aa, sa = crossfit_counts(S, SC, A); kb.append(t["k_ao"]); kc.append(ka)
        nsc = lab == "NO SIGN COST"; c["nsc"] += nsc; c["slb"] += lab == "SIGNED LEARNS BETTER"
        c["nsc_cf"] += nsc and ka >= ks - 1 and aa >= sa - 0.01
    print(f"{name}: true AO best p {max(p_ao):.2f} (x8 = {8*max(p_ao):.1f}) | mean AO@best {np.mean(kb):.2f}, cross-fit AO {np.mean(kc):.2f} | "
          f"NSC {c['nsc']/sims:.3f}, NSC + cross-fit guard {c['nsc_cf']/sims:.3f} | SLB {c['slb']/sims:.3f}", flush=True)

sims = int(sys.argv[1])
run("AO (.125,.6,.6,.6)", (1,) * 4, (.125, .6, .6, .6), sims, 41)
run("AO (.125,.5,.75,.75)", (1,) * 4, (.125, .5, .75, .75), sims, 42)
run("AO (.125,.5,.75,.5)", (1,) * 4, (.125, .5, .75, .5), sims, 43)
run("AO (.125,.5,.9,.9)", (1,) * 4, (.125, .5, .9, .9), sims, 44)
run("AO (.125,.5,1,.9) true no cost at best", (1,) * 4, (.125, .5, 1, .9), sims, 45)
run("AO (.125,.5,.9,.5)", (1,) * 4, (.125, .5, .9, .5), sims, 46)

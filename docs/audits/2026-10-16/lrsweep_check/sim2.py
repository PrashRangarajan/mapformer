"""(1) composite nulls aimed at a FALSE 'SIGNED LEARNS BETTER' (equal best-rate solve probability, SO broad plateau below
ceiling, AO narrow); (2) a paired (within-seed swap, 2^8) selection-aware test beside the registered unpaired one;
(3) worst-case configurations for the power sentence."""
import sys, itertools, time
import numpy as np
from sim_null import draw, n, R
from mapformer.analyze_lrsweep import b1_test, b1_decide, select, _at
from mapformer.stats_core import fisher_solved

FLIPS = np.array(list(itertools.product([0, 1], repeat=n)), bool)          # (256, n): True = swap SO/AO at that seed

def paired_p(S, SC, A):
    so = np.arange(n)[None].repeat(len(FLIPS), 0); ao = so + n
    i_so = np.where(FLIPS, ao, so); i_ao = np.where(FLIPS, so, ao)
    ra, rs = select(S, SC, i_ao), select(S, SC, i_so)
    Tk = _at(S, i_ao, ra).sum(1) - _at(S, i_so, rs).sum(1)
    Ta = _at(A, i_ao, ra).mean(1) - _at(A, i_so, rs).mean(1)
    tk, ta = Tk[0], Ta[0]                                                     # row 0 = no flips = observed
    return (np.abs(Tk) >= abs(tk)).mean(), (np.abs(Ta) >= abs(ta) - 1e-12).mean()

def run(name, p_so, p_ao, sims, seed, **kw):
    rng = np.random.default_rng(seed); c = dict(sel=0, naive=0, dec=0, paired=0, slb=0, nlb=0, nsc=0, pslb=0); t0 = time.time()
    kao = []
    for _ in range(sims):
        S, SC, A = draw(rng, p_so, p_ao, **kw)
        t = b1_test(S, SC, A, n); lab, P = b1_decide(t, A, S, n); pk, pa = paired_p(S, SC, A)
        c["sel"] += t["p_k_sel"] < .05; c["naive"] += P["naive_fisher_p"] < .05; c["dec"] += P["p_k"] < .05
        c["paired"] += pk < .05
        c["slb"] += lab == "SIGNED LEARNS BETTER"; c["nlb"] += lab == "NON-NEGATIVE LEARNS BETTER"; c["nsc"] += lab == "NO SIGN COST"
        c["pslb"] += (pk < .05 and t["k_so"] > t["k_ao"]) or (pa < .05 and t["acc_ao"] - t["acc_so"] <= -0.05)
        kao.append(t["k_ao"])
    print(f"{name}: sims {sims} ({time.time()-t0:.0f}s) | SOLVED-route rejection: sel {c['sel']/sims:.3f} naive {c['naive']/sims:.3f} "
          f"decision {c['dec']/sims:.3f} paired-sel {c['paired']/sims:.3f} | label SLB {c['slb']/sims:.3f} NLB {c['nlb']/sims:.3f} "
          f"NSC {c['nsc']/sims:.3f} | paired-route SLB {c['pslb']/sims:.3f} | mean AO@best SOLVED {np.mean(kao):.2f}", flush=True)

if __name__ == "__main__":
    sims = int(sys.argv[1]); which = sys.argv[2]
    if which == "comp":
        run("C1 composite: SO (.6,.6,.6,.6) AO (.05,.05,.05,.6)", (.6,) * 4, (.05, .05, .05, .6), sims, 21)
        run("C2 composite: SO (.8,.8,.8,.8) AO (.05,.05,.05,.8)", (.8,) * 4, (.05, .05, .05, .8), sims, 22)
        run("C3 composite: SO (.8,.8,.8,.3) AO (.1,.3,.8,.3)", (.8, .8, .8, .3), (.1, .3, .8, .3), sims, 23)
        run("C4 composite: SO (.6,.6,.6,.6) AO (.05,.05,.05,.6) seed c2", (.6,) * 4, (.05, .05, .05, .6), sims, 24, c_seed=2.0)
        run("H0 paired seed effect c2 (.5,.6,.6,.4)", (.5, .6, .6, .4), (.5, .6, .6, .4), sims, 25, c_seed=2.0)
        run("H0 paired seed effect c4 (.7,.7,.7,.7)", (.7,) * 4, (.7,) * 4, sims, 26, c_seed=4.0)
    if which == "pow":
        run("P1 SO 1.0; AO (.125,.5,.5,.3)", (1,) * 4, (.125, .5, .5, .3), sims, 31)
        run("P2 SO 1.0; AO (.125,.3,.3,.3)", (1,) * 4, (.125, .3, .3, .3), sims, 32)
        run("P3 SO 1.0; AO (.125,.6,.6,.6) [NSC inflation]", (1,) * 4, (.125, .6, .6, .6), sims, 33)
        run("P4 SO 1.0; AO (.125,.5,.75,.75)", (1,) * 4, (.125, .5, .75, .75), sims, 34)
        run("P5 SO 1.0; AO (.125,.5,.5,.3) seed c2", (1,) * 4, (.125, .5, .5, .3), sims, 35, c_seed=2.0)
    if which == "worst":
        # AO 3/8 at every rate, on disjoint / rotating seeds; SO 8/8 everywhere
        for name, ao in (("AO 3/8 at each rate, same seeds", [[0, 1, 2]] * 4),
                         ("AO 3/8 at each rate, rotating seeds", [[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 3, 5]]),
                         ("AO 3/8 best, 2/8 elsewhere, disjoint", [[0, 1, 2], [3, 4], [5, 6], [7, 0]])):
            S = np.zeros((2 * n, R), bool); S[:n] = True
            for j, seeds in enumerate(ao):
                S[n + np.array(seeds), j] = True
            SC = np.where(S, 0.002, 0.2) + np.linspace(0, 1e-3, R)[None]; A = np.where(S, 1.0, 0.9)
            t = b1_test(S, SC, A, n); lab, P = b1_decide(t, A, S, n)
            print(f"{name}: AO {t['k_ao']}/8 vs SO {t['k_so']}/8; p_sel {t['p_k_sel']:.4f}, Fisher {P['naive_fisher_p']:.4f}, "
                  f"decision {P['p_k']:.4f}, paired {paired_p(S, SC, A)[0]:.4f} -> {lab}")

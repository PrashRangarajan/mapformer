"""Size and power of the registered B1 test (analyze_lrsweep.b1_test / b1_decide) under synthetic data-generating processes.
Per seed s a latent difficulty u_s ~ N(0,1) (shared by both arms and all rates when rho_seed > 0); per run
solved ~ Bernoulli(sigmoid(logit(p_arm,rate) + c*u_s + d*v_{s,arm} + e)), e ~ N(0, sd_run) independent per run."""
import sys, time
import numpy as np
from scipy.special import expit, logit
from mapformer.analyze_lrsweep import b1_test, b1_decide, label_from, select
from mapformer.stats_core import fisher_solved, perm2_p

n, R = 8, 4

def draw(rng, p_so, p_ao, c_seed=0.0, c_profile=0.0):
    u = rng.normal(size=n)                       # seed effect: shared by SO and AO at the seed (pairing)
    S = np.zeros((2 * n, R), bool)
    for a, p in ((0, p_so), (1, p_ao)):
        v = rng.normal(size=n)                   # profile effect: shared across rates within one arm at a seed
        for j in range(R):
            pj = np.clip(p[j], 1e-6, 1 - 1e-6)
            q = expit(logit(pj) + c_seed * u + c_profile * v) if (c_seed or c_profile) else np.full(n, pj)
            S[a * n:(a + 1) * n, j] = rng.random(n) < q
    SC = np.where(S, 0.002, 0.2) + rng.uniform(0, 0.05, S.shape)
    A = np.where(S, 0.995 + rng.normal(0, 0.003, S.shape), rng.uniform(0.80, 0.97, S.shape))
    return S, SC, np.clip(A, 0, 1)

def run(name, p_so, p_ao, sims, seed=0, **kw):
    rng = np.random.default_rng(seed); t0 = time.time()
    rej = {"sel_k": 0, "naive_k": 0, "dec_k": 0, "sel_a": 0, "naive_a": 0, "dec_a": 0}; labs = {}
    for _ in range(sims):
        S, SC, A = draw(rng, p_so, p_ao, **kw)
        t = b1_test(S, SC, A, n); lab, P = b1_decide(t, A, S, n)
        rej["sel_k"] += t["p_k_sel"] < .05; rej["naive_k"] += P["naive_fisher_p"] < .05; rej["dec_k"] += P["p_k"] < .05
        rej["sel_a"] += t["p_acc_sel"] < .05; rej["naive_a"] += P["naive_perm_p"] < .05; rej["dec_a"] += P["p_acc"] < .05
        labs[lab[:10]] = labs.get(lab[:10], 0) + 1
    print(f"{name}: sims {sims} ({time.time() - t0:.0f}s) | rejection rates (p<.05, two-sided): "
          + " ".join(f"{k} {v / sims:.3f}" for k, v in rej.items()) + " | labels " + str({k: round(v / sims, 3) for k, v in labs.items()}), flush=True)

if __name__ == "__main__":
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if which in ("all", "null"):
        run("H0 profile, iid, equal probs (0.5,0.6,0.6,0.4)", (0.5, .6, .6, .4), (0.5, .6, .6, .4), sims, 1)
        run("H0 profile, iid, flat 0.5", (.5,) * 4, (.5,) * 4, sims, 2)
        run("H0 profile + seed effect (paired, c 2)", (.5, .6, .6, .4), (.5, .6, .6, .4), sims, 3, c_seed=2.0)
        run("H0 profile + profile effect (c 2, no pairing)", (.5, .6, .6, .4), (.5, .6, .6, .4), sims, 4, c_profile=2.0)
        run("composite null: best equal 0.9, profiles differ", (.9, .9, .5, .1), (.1, .5, .9, .9), sims, 5)
        run("composite null: best equal 0.6, profiles differ", (.6, .6, .3, .05), (.05, .3, .6, .6), sims, 6)
        run("composite null: best equal 0.6, differ, profile effect", (.6, .6, .3, .05), (.05, .3, .6, .6), sims, 7, c_profile=2.0)
    if which in ("all", "power"):
        run("power: SO 1.0; AO ROBUST-like (.125,.5,.5,.3) iid", (1, 1, 1, 1), (.125, .5, .5, .3), sims, 11)
        run("power: SO 1.0; AO ROBUST-like (.125,.5,.5,.3) profile c2", (1, 1, 1, 1), (.125, .5, .5, .3), sims, 12, c_profile=2.0)
        run("power: SO 1.0; AO (.125,.3,.3,.3) iid", (1, 1, 1, 1), (.125, .3, .3, .3), sims, 13)
        run("power: SO 1.0; AO (.125,.2,.2,.2) iid", (1, 1, 1, 1), (.125, .2, .2, .2), sims, 14)
        run("power: SO 1.0; AO (0,.1,.1,.1) iid", (1, 1, 1, 1), (0, .1, .1, .1), sims, 15)
        run("closes: SO 1.0; AO (.125,.5,.9,.9) iid", (1, 1, 1, 1), (.125, .5, .9, .9), sims, 16)
        run("closes: SO 1.0; AO (.125,.5,.75,.75) iid", (1, 1, 1, 1), (.125, .5, .75, .75), sims, 17)
        run("selection-favours-NSC: SO 1.0; AO (.125,.6,.6,.6) iid", (1, 1, 1, 1), (.125, .6, .6, .6), sims, 18)

"""ANALYSIS: per-set summaries of sweep_final.json for RANK_THEORY.md checks C1-C4. Output summarise_final_out.txt.
C1 drift direction vs resolution (r_beta) in drifting rank-D heads (D >= 2), against the isotropic null E[r_beta] = 1
   and against a random-direction null drawn from each head's own frame Gram matrix.
C2 where the leftover latent clock sits in solved rank-(D+1) heads: off-plane share lam_perp / lam, against the
   random-direction expectation sqrt((r - D) / r), and how much of it the readout sees (readgain, 1 = read like the map).
C3 unsolved rank-(D+1) heads: the same, for the free route that was available and not used.
C4 1D: r_beta is identically 1 (no direction to drift along other than the map's one)."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f"{HERE}/sweep_final.json"))
rng = np.random.default_rng(0)
L = []


def q(x):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return "n=0"
    return f"n={len(x)} med {np.median(x):.3f} IQR {np.percentile(x, 25):.3f}-{np.percentile(x, 75):.3f}"


L.append("== C1: drifting heads (kappa > 0.01) -- r_beta = resolution along the drift / mean resolution (null 1) ==")
allD = []
for lab, S in J.items():
    D, r = S["D"], S["r"]
    if D < 2:
        continue
    rb = [h["r_beta"] for x in S["rows"] for h in x["heads"] if h["kappa"] > 0.01]
    rbc = [h["r_beta"] for x in S["rows"] for h in x["heads"] if h["kappa"] <= 0.01]
    if r == D:
        allD += rb
    L.append(f"  {lab:46s} drifting: {q(rb)}; frac<0.5 {np.mean(np.asarray(rb) < 0.5) if rb else float('nan'):.2f} | "
             f"clean-drift heads: {q(rbc)}")
# exact null: for a head with frame singular values s1 >= s2 (D=2), a uniformly random direction has
# r = 2 (s1^2 cos^2 + s2^2 sin^2) / (s1^2 + s2^2); P(r < 0.5) for indep = s2/s1
L.append(f"  pooled rank-D (D>=2) drifting heads: {q(allD)}; frac < 0.5: {np.mean(np.asarray(allD) < 0.5):.2f}")
# random-direction null using each drifting rank-D head's own indep (D=2 only)
nulls = []
for lab, S in J.items():
    if S["D"] == 2 and S["r"] == 2:
        for x in S["rows"]:
            for h in x["heads"]:
                if h["kappa"] > 0.01:
                    s2 = h["indep"] ** 2
                    th = rng.uniform(0, np.pi, 2000)
                    rr = 2 * (np.cos(th) ** 2 + s2 * np.sin(th) ** 2) / (1 + s2)
                    nulls.append(np.mean(rr < 0.5))
L.append(f"  random-direction null for the same heads (own frame conditioning): expected frac < 0.5 = {np.mean(nulls):.2f}")

L.append("\n== C2 / C3: rank > D heads -- leftover latent clock (lam = |c_lat| / |U_lat|), off-plane share, readout gain ==")
for lab, S in J.items():
    D, r = S["D"], S["r"]
    if r <= D:
        continue
    exp_off = np.sqrt((r - D) / r)
    for solved in (True, False):
        hs = [h for x in S["rows"] if x["solved"] == solved for h in x["heads"]]
        if not hs:
            continue
        lam = [h["lam"] for h in hs]; off = [h["lam_perp"] / h["lam"] for h in hs if h["lam"] > 0]
        rg = [h["readgain"] for h in hs]; kap = [h["kappa"] for h in hs]
        L.append(f"  {lab:40s} {'SOLVED' if solved else 'unsolved'} heads: lam {q(lam)} | off-plane share {q(off)} "
                 f"(random {exp_off:.2f}) | readgain {q(rg)} | kappa {q(kap)}")

L.append("\n== rank-D heads, for comparison: lam (latent clock) and readgain ==")
for lab, S in J.items():
    D, r = S["D"], S["r"]
    if r != D:
        continue
    for solved in (True, False):
        hs = [h for x in S["rows"] if x["solved"] == solved for h in x["heads"]]
        if hs:
            L.append(f"  {lab:40s} {'SOLVED' if solved else 'unsolved'}: lam {q([h['lam'] for h in hs])} | "
                     f"kappa {q([h['kappa'] for h in hs])} | indep {q([h['indep'] for h in hs])}")

L.append("\n== C4: 1D ring -- per-head kappa and frame (r_beta is 1 by construction in 1D) ==")
for lab, S in J.items():
    if S["D"] != 1:
        continue
    for x in S["rows"]:
        L.append(f"  {lab:34s} s{x['seed']} {x['cls'][:4]} {x['basin']:6s} kappa {[round(h['kappa'], 4) for h in x['heads']]} "
                 f"lam {[round(h['lam'], 4) for h in x['heads']]} |U| {[round(h['nU'], 2) for h in x['heads']]}")

open(f"{HERE}/summarise_final_out.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))

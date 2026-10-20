"""ANALYSIS C1 at run level: rank-D (D >= 2) runs with >= 1 drifting head (kappa > 0.01); per run, is the most aligned
drifting head's r_beta < 0.5? Null: drift direction uniform relative to each head's own frame (2D formula
r = 2 (cos^2 + s^2 sin^2) / (1 + s^2), s = indep; used as an approximation in 3D). Each checkpoint counted once."""
import json, numpy as np
J = json.load(open('sweep_final.json')); rng = np.random.default_rng(1)
obs = 0; exp = []; n = 0; seen = set()
for lab, S in J.items():
    if S['D'] < 2 or S['r'] != S['D']:
        continue
    for x in S['rows']:
        if x['ck'] in seen:
            continue
        seen.add(x['ck'])
        hs = [h for h in x['heads'] if h['kappa'] > 0.01]
        if not hs:
            continue
        n += 1; obs += min(h['r_beta'] for h in hs) < 0.5
        th = rng.uniform(0, np.pi, (4000, len(hs)))
        s2 = np.array([h['indep'] ** 2 for h in hs])
        v = 2 * (np.cos(th) ** 2 + s2 * np.sin(th) ** 2) / (1 + s2)
        exp.append(np.mean(v.min(1) < 0.5))
exp = np.array(exp)
sim = (rng.random((20000, len(exp))) < exp).sum(1)
print(f"rank-D runs with a drifting head: {n}; runs whose most aligned drifting head has r_beta < 0.5: {obs}; "
      f"expected under a random drift direction {exp.sum():.1f}; P(>= {obs}) = {np.mean(sim >= obs):.5f} (20000 sims)")

"""ANALYSIS C1 at run level, counting each NETWORK once (docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md, rank MINOR).
c1_runlevel.py de-duplicates by checkpoint path, but escape C ('r2 continued') is the rank_mi Vanilla s0-7 networks
trained 600 epochs further, so the same 8 networks enter twice. Same computation as c1_runlevel.py (same RNG seed and
draw order), with escape C dropped; a second line also drops loop_rank_e1800 (identical initialisation to rank_mi).
  cd .../scripts/rank && python3 c1_runlevel_once.py > c1_runlevel_once_out.txt"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def run(skip):
    J = json.load(open(f"{HERE}/sweep_final.json")); rng = np.random.default_rng(1)
    obs = 0; exp = []; n = 0; seen = set()
    for lab, S in J.items():
        if S['D'] < 2 or S['r'] != S['D'] or any(k in lab for k in skip):
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
    print(f"skip {skip}: rank-D runs with a drifting head: {n}; most aligned drifting head r_beta < 0.5: {obs}; "
          f"expected under a random drift direction {exp.sum():.1f}; P(>= {obs}) = {np.mean(sim >= obs):.5f} (20000 sims)")


run(())
run(("escape C",))
run(("escape C", "1800ep"))

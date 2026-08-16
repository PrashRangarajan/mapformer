"""Pre-flight gates for the two CSCG-derived tasks. CPU only, run BEFORE training.

Both score a single observation per scored event, drawn i.i.d. per cell per
episode, so an answer n-gram should be at chance BY CONSTRUCTION. The gates check
that, plus the task-specific risks:

  stitch  confound-vs-shared  the paper's own negative control. A model merging on
                              local appearance alone fails the confound case. If
                              the two cases have different FLOORS, the metric must
                              be reported split, not pooled.
  schema  interior leakage    scored events must land only on periphery cells the
                              agent actually saw; if interior cells leak in, the
                              answers are unknowable and the ceiling is not 1.0.
  both    last-obs / marginal / n-gram 1..5 / oracle
"""
import argparse, json
from collections import Counter, defaultdict
import numpy as np
from mapformer.environment_stitch import StitchWorld
from mapformer.environment_schema import SchemaWorld

def baselines(answers, n_obs, offset):
    a = [x - offset for x in answers]; n = len(a); c = Counter(a)
    marg = c.most_common(1)[0][1] / max(n, 1)
    last = sum(int(a[i] == a[i-1]) for i in range(1, n)) / max(n - 1, 1)
    ng, half = {}, n // 2
    for k in (1, 2, 3, 5):
        tab = defaultdict(Counter)
        for i in range(k, half): tab[tuple(a[i-k:i])][a[i]] += 1
        pred = {q: d.most_common(1)[0][0] for q, d in tab.items()}
        fb = c.most_common(1)[0][0]
        ok = sum(int(pred.get(tuple(a[i-k:i]), fb) == a[i]) for i in range(half+k, n))
        ng[k] = ok / max(n - half - k, 1)
    return dict(n=n, chance=1.0/n_obs, marginal=marg, last_obs=last, ngram=ng)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="CSCG_TASK_GATES.md")
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    lines = ["# CSCG-derived tasks -- pre-flight gates (CPU, no training)", ""]

    # ---- stitch, split by the paper's negative control ----
    env = StitchWorld(seed=10000)
    split = {}
    for cf in (False, True):
        ans, ns = [], []
        for _ in range(a.n_episodes):
            _t, _r, sp, an, info = env.generate_episode(rng=rng, force_confound=cf)
            ans += an; ns.append(len(sp))
        split["confound" if cf else "shared"] = dict(
            **baselines(ans, env.n_obs_types, env.obs_offset),
            scored_per_ep=float(np.mean(ns)))
    lines += ["## Transitive inference (stitch), split by the negative control", "",
              f"Two 8x6 rooms, {env.n_obs_types} observation types, shared 3x3 corner patch,",
              "plus a CONFOUNDING identical patch elsewhere in room A.",
              f"**chance = {1/env.n_obs_types:.4f}.**", "",
              "| start patch | scored/ep | marginal | last-obs | n-gram o1 | o3 | o5 |",
              "|---|---|---|---|---|---|---|"]
    for k, r in split.items():
        lines.append(f"| {k} | {r['scored_per_ep']:.1f} | {r['marginal']:.4f} | "
                     f"{r['last_obs']:.4f} | {r['ngram'][1]:.4f} | {r['ngram'][3]:.4f} | "
                     f"{r['ngram'][5]:.4f} |")
        print(f"stitch/{k:9s} chance={r['chance']:.4f} marg={r['marginal']:.4f} "
              f"last={r['last_obs']:.4f} ng1={r['ngram'][1]:.4f} scored/ep={r['scored_per_ep']:.1f}")

    # ---- schema ----
    envs = SchemaWorld(seed=10000)
    ans, ns, leak = [], [], 0
    for _ in range(a.n_episodes):
        _t, _r, sp, an, info = envs.generate_episode(rng=rng)
        ans += an; ns.append(len(sp))
    rs = baselines(ans, envs.n_obs_types, envs.obs_offset)
    lines += ["", "## Schema transfer (shortcut across unvisited interior)", "",
              f"{envs.h}x{envs.w} room, {envs.n_obs_types} observation types, periphery-only",
              f"exploration. **chance = {rs['chance']:.4f}.**", "",
              "| scored/ep | marginal | last-obs | n-gram o1 | o3 | o5 |",
              "|---|---|---|---|---|---|",
              f"| {np.mean(ns):.1f} | {rs['marginal']:.4f} | {rs['last_obs']:.4f} | "
              f"{rs['ngram'][1]:.4f} | {rs['ngram'][3]:.4f} | {rs['ngram'][5]:.4f} |"]
    print(f"schema        chance={rs['chance']:.4f} marg={rs['marginal']:.4f} "
          f"last={rs['last_obs']:.4f} ng1={rs['ngram'][1]:.4f} scored/ep={np.mean(ns):.1f}")
    open(a.out, "w").write("\n".join(lines) + "\n")
    json.dump({"stitch": split, "schema": rs}, open(a.out.replace(".md", ".json"), "w"),
              indent=2, default=str)
    print("\n".join(lines))

if __name__ == "__main__":
    main()

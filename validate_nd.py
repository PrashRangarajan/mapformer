"""Gates for the D-dimensional rank test. Run BEFORE any GPU time.

Rule 7: this CALLS GridWorldND, it does not reimplement the walk. A gate that
duplicates the task certifies a different task from the one the trainer runs.

Five gates:
  G1  measured chance      -- majority-class frequency over SCORED positions.
                              Report it beside every headline; do not assume
                              1/(K+1).
  G2  action-stream ngram  -- orders 1..5 on the ACTIONS ALONE. Four
                              demonstration tasks were voided by this check.
                              An ngram must not beat chance by much.
  G3  revisit rate         -- a simple random walk on Z^D is TRANSIENT for
                              D >= 3 (Polya), so revisits here exist only
                              because the torus is finite. If the rate
                              collapses the masked loss has no signal and a
                              "collapse" would be the task, not the rank.
  G4  scored positions     -- how many labels per trajectory, so the effective
                              sample size per arm is known.
  G5  phase resolution     -- the theory's premise, measured. NOT collisions:
                              a generic linear map is injective on any finite
                              set of lattice points, so exact collisions never
                              occur and the first version of this gate returned
                              0.0% everywhere and falsified the kernel argument
                              it was written to check. What actually binds is
                              SEPARATION. The phase is confined to an
                              r-dimensional subtorus, and packing N^D positions
                              into it forces

                                  min separation  ~  N^(-max(D/r, 1)),

                              verified by fitting the exponent below. r >= D
                              attains the floor of 1; r < D is strictly worse
                              at D/r. So GEOMETRY sets a hard threshold at
                              r = D and says nothing about r = 2 vs r = 4 in
                              2D, where both sit at the floor -- that gap is
                              optimisation, and the two mechanisms are
                              separable exactly this way.
"""
import argparse
import itertools
from collections import Counter, defaultdict

import numpy as np

from mapformer.environment_nd import GridWorldND


def _rollout(env, T, n, seed):
    np.random.seed(seed)
    toks, revs, locs = [], [], []
    for _ in range(n):
        t, _m, r = env.generate_trajectory(T)
        toks.append(t.numpy()); revs.append(r.numpy()); locs.append(list(env.visited_locations))
    return toks, revs, locs


def g1_chance(toks, revs):
    lab = [t[1::2][r[1::2]] for t, r in zip(toks, revs)]
    lab = np.concatenate([x for x in lab if len(x)])
    c = Counter(lab.tolist())
    return {"n_labels": int(len(lab)), "n_classes": len(c),
            "majority_freq": max(c.values()) / len(lab)}


def g2_action_ngram(toks, revs, max_order=5):
    """Predict the scored observation from the preceding ACTION tokens only."""
    out = {}
    acts = [t[0::2] for t in toks]
    obs = [t[1::2] for t in toks]
    mask = [r[1::2] for r in revs]
    n_tr = len(toks)
    split = max(1, int(0.7 * n_tr))
    for order in range(1, max_order + 1):
        table = defaultdict(Counter)
        for i in range(split):
            a, o, m = acts[i], obs[i], mask[i]
            for j in range(order - 1, len(o)):
                if m[j]:
                    table[tuple(a[j - order + 1:j + 1].tolist())][int(o[j])] += 1
        hit = tot = 0
        for i in range(split, n_tr):
            a, o, m = acts[i], obs[i], mask[i]
            for j in range(order - 1, len(o)):
                if not m[j]:
                    continue
                ctx = tuple(a[j - order + 1:j + 1].tolist())
                tot += 1
                if table.get(ctx) and table[ctx].most_common(1)[0][0] == int(o[j]):
                    hit += 1
        out[order] = hit / max(tot, 1)
    return out


def g3_g4(revs):
    per = [int(r[1::2].sum()) for r in revs]
    rate = [float(r[1::2].mean()) for r in revs]
    return float(np.mean(rate)), float(np.mean(per))


def g5_separation(env, r, trials=5, seed=0):
    """Scale-free minimum separation of the FULL position set after a random
    rank-r projection, plus the fitted exponent of its decay in N."""
    from scipy.spatial.distance import pdist

    def sep_at(N, D, r):
        pts = np.array(list(itertools.product(range(N), repeat=D)), dtype=float)
        out = []
        for s_ in range(trials):
            W = np.random.RandomState(seed + s_).randn(D, r)
            S = pts @ W
            S = S / S.std()                      # remove overall scale
            out.append(float(pdist(S).min()))
        return float(np.mean(out))

    D = env.dims
    here = sep_at(env.size, D, r)
    # fit the exponent over a small ladder of map sizes, capped so the point
    # count stays tractable
    Ns = [N for N in range(3, 33) if 2 <= N and N ** D <= 6000][-4:]
    slope = float("nan")
    if len(Ns) >= 3:
        vals = [sep_at(N, D, r) for N in Ns]
        slope = float(np.polyfit(np.log(Ns), np.log(vals), 1)[0])
    return {"min_sep": here, "fitted_exponent": slope,
            "predicted_exponent": -max(D / r, 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str,
                    default="2:32,3:10,5:4",
                    help="comma-separated D:size pairs")
    ap.add_argument("--n-obs-types", type=int, default=16)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("# Gates: D-dimensional torus rank test\n")
    print(f"T={a.steps}, {a.n_traj} trajectories per config, "
          f"n_obs_types={a.n_obs_types}, seed={a.seed}\n")
    rows = []
    for spec in a.configs.split(","):
        D, N = (int(v) for v in spec.split(":"))
        env = GridWorldND(dims=D, size=N, n_obs_types=a.n_obs_types, seed=a.seed)
        toks, revs, locs = _rollout(env, a.steps, a.n_traj, a.seed)
        ch = g1_chance(toks, revs)
        ng = g2_action_ngram(toks, revs)
        rate, per = g3_g4(revs)
        print(f"## D={D}, grid {N} ({env.n_cells} cells, vocab {env.unified_vocab_size}, "
              f"{env.n_actions} actions)\n")
        print(f"  G1 chance (majority class over {ch['n_labels']} labels): "
              f"{ch['majority_freq']:.4f}  [{ch['n_classes']} classes seen]")
        print("  G2 action-stream ngram, orders 1-5: " +
              "  ".join(f"{k}:{v:.3f}" for k, v in ng.items()))
        print(f"  G3 revisit rate: {rate:.3f}")
        print(f"  G4 scored positions per trajectory: {per:.1f}")
        for r in sorted({2, D, D + 2}):
            g5 = g5_separation(env, r)
            print(f"  G5 rank r={r}: min separation {g5['min_sep']:.4f}, "
                  f"decay exponent {g5['fitted_exponent']:+.2f} "
                  f"(predicted {g5['predicted_exponent']:+.2f})")
        worst = max(ng.values())
        verdict = "PASS" if worst < ch["majority_freq"] + 0.10 and rate > 0.05 else "CHECK"
        print(f"\n  verdict: {verdict}  (best ngram {worst:.3f} vs chance "
              f"{ch['majority_freq']:.3f}; revisit {rate:.3f})\n")
        rows.append((D, N, ch["majority_freq"], worst, rate, per))

    print("## Summary\n")
    print("| D | grid | cells | chance | best ngram | revisit rate | scored/traj |")
    print("|---|---|---|---|---|---|---|")
    for D, N, c, w, rate, per in rows:
        print(f"| {D} | {N} | {N**D} | {c:.3f} | {w:.3f} | {rate:.3f} | {per:.1f} |")


if __name__ == "__main__":
    main()

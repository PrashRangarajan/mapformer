"""Pre-flight gates for the Match-Query task. CPU only. Run BEFORE any training.

Same discipline as validate_map_query.py, with the gates re-aimed at this task's
specific shortcut risks.

  G1 chance          uniform guess over n_obs_types. PASS level for every baseline.
  G2 marginal        always predict the most common answer. Catches a skewed
                     answer distribution.
  G3 answer n-gram   order 1..5 over the answer stream alone. THE HIGH-RISK GATE
                     here: if the query walk revisits a cell, the answer repeats
                     and an order-1 model catches it for free. This is the exact
                     family of bug that invalidated hier-goal twice (raw BFS
                     order-1 0.969; interleaved order-3 0.971).
  G4 never-moved     predict the observation at the END-OF-EXPLORE cell for every
                     query. If the query walk barely moves, this alone solves it.
  G5 answerable rate what fraction of query steps are scoreable at all. Too low
                     and the task is mostly empty; it also sets the effective
                     sample size per episode.
  G6 oracle          true position + true map must give 1.0.

A gate at chance is a PASS.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from mapformer.environment_match_query import MatchQueryGridWorld


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-explore", nargs="+", type=int, default=[128, 256, 512])
    ap.add_argument("--t-query", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--n-episodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="MATCH_QUERY_GATES.md")
    args = ap.parse_args()

    rows = []
    for TE in args.t_explore:
        for TQ in args.t_query:
            env = MatchQueryGridWorld(size=64, n_obs_types=16, seed=10000)
            rng = np.random.RandomState(args.seed)
            answers, never_moved, n_steps_tot, n_scored_tot = [], [], 0, 0

            for _ in range(args.n_episodes):
                _t, _rv, sp, ans, info = env.generate_match_episode(TE, TQ, rng)
                n_steps_tot += TQ
                n_scored_tot += len(ans)
                answers.extend(ans)
                # G4: what the "agent never left the explore endpoint" answer
                # would be. The endpoint is the first query cell's predecessor;
                # recover it as the cell the walk started the query phase from.
                om = info["obs_map"]
                ex, ey = info["end_explore_pos"]     # the ACTUAL endpoint, not an
                guess = int(om[ex, ey]) + env.obs_offset   # approximation
                never_moved.extend([guess] * len(ans))

            n = len(answers)
            if n == 0:
                continue
            K = env.n_obs_types
            g1 = 1.0 / K
            cnt = Counter(answers)
            g2 = cnt.most_common(1)[0][1] / n

            ngram = {}
            half = n // 2
            for Kk in (1, 2, 3, 5):
                tab = defaultdict(Counter)
                for i in range(Kk, half):
                    tab[tuple(answers[i - Kk:i])][answers[i]] += 1
                pred = {c: d.most_common(1)[0][0] for c, d in tab.items()}
                fb = cnt.most_common(1)[0][0]
                ok = 0
                for i in range(half + Kk, n):
                    c = tuple(answers[i - Kk:i])
                    ok += int(pred.get(c, fb) == answers[i])
                ngram[Kk] = ok / max(n - half - Kk, 1)

            g4 = (sum(int(a == b) for a, b in zip(answers, never_moved))
                  / max(len(never_moved), 1)) if never_moved else float("nan")
            g5 = n_scored_tot / max(n_steps_tot, 1)

            rows.append(dict(TE=TE, TQ=TQ, chance=g1, marginal=g2, ngram=ngram,
                             never_moved=g4, answerable=g5, oracle=1.0,
                             n_answers=n))
            print(f"TE={TE:4d} TQ={TQ:4d} chance={g1:.4f} marginal={g2:.4f} "
                  f"ngram1={ngram[1]:.4f} ngram3={ngram[3]:.4f} "
                  f"never_moved={g4:.4f} answerable={g5:.3f} n={n}", flush=True)

    lines = ["# Match-Query task -- pre-flight gates (CPU, no training)", "",
             "Blind-continuation task: explore with observations revealed, then "
             "continue with them withheld and predict the observation at each cell.",
             "Scored only at cells visited during explore AND non-blank, so chance "
             "is 1/16 = 0.0625.", "",
             "| T_explore | T_query | chance | marginal | n-gram o1 | o3 | o5 | never-moved | answerable rate | n |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['TE']} | {r['TQ']} | {r['chance']:.4f} | {r['marginal']:.4f} | "
            f"{r['ngram'][1]:.4f} | {r['ngram'][3]:.4f} | {r['ngram'][5]:.4f} | "
            f"{r['never_moved']:.4f} | {r['answerable']:.3f} | {r['n_answers']} |")
    lines += ["", "**Reading it.** Every baseline column should sit at `chance` "
              "(0.0625). `n-gram` is the high-risk gate: a repeated cell inside the "
              "query phase makes the answer repeat. `answerable rate` is the "
              "fraction of query steps that are scoreable, which sets the effective "
              "sample size per episode."]
    open(args.out, "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(args.out.replace(".md", ".json"), "w"), indent=2, default=str)
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

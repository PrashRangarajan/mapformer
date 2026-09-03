"""Pre-flight gates for the algorithmic tasks. CPU only. Run BEFORE any training.

Standing rule 1: n-gram on the ANSWER STREAM ALONE at orders 1-5 before any new
task. Four planner tasks in this repo were voided by exactly this check after the
GPU had already been spent on them.

Per task, every baseline below must sit at the MEASURED chance rate:
  marginal      always predict the most common answer
  ngram k       predict from the previous k answers, fitted on half and tested on
                the other half
  echo-input    predict the input token at the scored position. For parity this is
                "output the current bit"; for copy it is "output the token you are
                looking at", which is the off-by-one a teacher-forced copy invites.
  repeat-prev   predict the previous ANSWER. For parity this is the strongest
                trivial strategy available, since the running parity changes only
                when the bit is 1.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mapformer.environment_algorithmic import WORLDS


def ngram_acc(ans, k):
    n = len(ans)
    half = n // 2
    tab = defaultdict(Counter)
    for i in range(k, half):
        tab[tuple(ans[i - k:i])][ans[i]] += 1
    pred = {c: d.most_common(1)[0][0] for c, d in tab.items()}
    fb = Counter(ans).most_common(1)[0][0]
    ok = sum(int(pred.get(tuple(ans[i - k:i]), fb) == ans[i])
             for i in range(half + k, n))
    return ok / max(n - half - k, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["parity", "copy"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--n-symbols", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ALGORITHMIC_GATES.md")
    a = ap.parse_args()

    rows = []
    for task in a.tasks:
        W = WORLDS[task]
        env = W(seed=1000) if task == "parity" else W(a.n_symbols, seed=1000)
        for L in a.lengths:
            rng = np.random.RandomState(a.seed)
            ans, echo, prev = [], [], []
            for _ in range(a.n_episodes):
                t, y, m = env.batch(1, L, rng)
                t, y, m = t[0].numpy(), y[0].numpy(), m[0].numpy()
                idx = np.flatnonzero(m)
                ans.extend(y[idx].tolist())
                echo.extend(t[idx].tolist())
                prev.extend(y[idx - 1].tolist())
            n = len(ans)
            cnt = Counter(ans)
            row = dict(task=task, L=L, n=n, chance=env.chance,
                       marginal=cnt.most_common(1)[0][1] / n,
                       echo=sum(int(x == y) for x, y in zip(echo, ans)) / n,
                       prev=sum(int(x == y) for x, y in zip(prev, ans)) / n,
                       ngram={k: ngram_acc(ans, k) for k in (1, 2, 3, 5)})
            rows.append(row)
            print(f"{task:7s} L={L:4d} chance={row['chance']:.4f} "
                  f"marginal={row['marginal']:.4f} "
                  + " ".join(f"ng{k}={row['ngram'][k]:.4f}" for k in (1, 2, 3, 5))
                  + f" echo={row['echo']:.4f} repeat-prev={row['prev']:.4f} n={n}",
                  flush=True)

    o = ["# Algorithmic tasks -- pre-flight gates (CPU, no training)", "",
         "Every baseline column must sit at `chance`. A gate AT chance is a PASS.", "",
         "| task | L | chance | marginal | ngram1 | ngram2 | ngram3 | ngram5 | "
         "echo-input | repeat-prev | n |", "|---" * 12 + "|"]
    worst = 0.0
    for r in rows:
        vals = [r["marginal"]] + [r["ngram"][k] for k in (1, 2, 3, 5)] + \
               [r["echo"], r["prev"]]
        worst = max(worst, max(v - r["chance"] for v in vals))
        o.append(f"| {r['task']} | {r['L']} | {r['chance']:.4f} | "
                 f"{r['marginal']:.4f} | "
                 + " | ".join(f"{r['ngram'][k]:.4f}" for k in (1, 2, 3, 5))
                 + f" | {r['echo']:.4f} | {r['prev']:.4f} | {r['n']} |")
    o += ["", "## Verdict", "",
          f"Largest excess of any trivial baseline over its own chance rate: "
          f"**{worst:+.4f}**.", ""]
    o += (["**PASS.** No trivial strategy beats chance by more than 0.02, so a "
           "trained model's score is not available for free.", ""] if worst <= 0.02
          else ["**FAIL.** A trivial strategy clears chance by more than 0.02. Fix "
                "the task before training on it.", ""])
    o += ["Note on parity specifically: `repeat-prev` is the strategy to watch. The "
          "running parity changes only when the bit is 1, so predicting the previous "
          "answer is right half the time -- which is chance here, but would NOT be "
          "if the bit distribution were skewed."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()

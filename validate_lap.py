"""Pre-flight gates for the Lap task. CPU only. Run BEFORE any training.

The decision is binary at each of the K lap boundaries: is the next token REWARD?
Exactly one boundary per episode is positive, so the baselines are:

  G1 always-no        never predict REWARD. hit=0, false_alarm=0, and boundary
                      accuracy = (K-1)/K. That last number is the reason boundary
                      accuracy alone is a BAD headline metric -- 0.75 for free.
  G2 always-yes       hit=1 but false_alarm=1. Shows why hit rate alone is also
                      not a metric.
  G3 random-boundary  pick one boundary uniformly. exact = 1/K.
  G4 n-gram           order 1..8 over the token stream. The track is periodic, so
                      a high-order n-gram sees an identical context at every lap
                      boundary and must answer the same way each time -> it
                      cannot hit the reward without false alarms. Measured, not
                      assumed: this is the gate hier-goal failed twice.
  G5 POSITIONAL       predict REWARD at a fixed token index. THE decisive gate.
                      With fixed loop_len the reward always lands at index K*L,
                      so this scores 1.0 and the task is void. With variable
                      loop_len it should collapse.
  G6 oracle           knows the true lap -> exact = 1.0.

`exact` (hit AND no false alarms) is the headline metric. Its floors are: 0 for
always-no, 0 for always-yes, 1/K for random-boundary.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from mapformer.environment_lap import LapWorld


def gates(fixed_loop, n_episodes, n_laps, seed):
    env = LapWorld(n_laps=n_laps, fixed_loop=fixed_loop, seed=10000)
    rng = np.random.RandomState(seed)
    eps = [env.generate_lap_episode(rng) for _ in range(n_episodes)]
    K = n_laps

    # --- G5 POSITIONAL: fit the single best "reward is at token index i" rule on
    # the first half, test on the second. This is the shortcut fixed_loop creates.
    half = n_episodes // 2
    idx_counts = Counter(e[3]["reward_index"] for e in eps[:half])
    best_idx = idx_counts.most_common(1)[0][0]
    pos_exact = 0
    for toks, dec_pos, dec_lab, info in eps[half:]:
        # rule fires REWARD at exactly one position; correct iff that position is
        # the true reward index AND it coincides with the true positive boundary
        hit = int(info["reward_index"] == best_idx)
        pos_exact += hit
    g5 = pos_exact / max(n_episodes - half, 1)

    # --- G4 n-gram over the token stream, predicting the boundary token
    stream = []
    for toks, _dp, _dl, _in in eps:
        stream.extend(int(t) for t in toks)
    ngram = {}
    hs = len(stream) // 2
    for order in (1, 2, 4, 8):
        tab = defaultdict(Counter)
        for i in range(order, hs):
            tab[tuple(stream[i - order:i])][stream[i]] += 1
        pred = {c: d.most_common(1)[0][0] for c, d in tab.items()}
        # score ONLY at decision points in the held-out half
        ok = tot = 0
        off = 0
        for toks, dec_pos, dec_lab, _in in eps:
            n = toks.shape[0]
            if off + n <= hs:
                off += n; continue
            for p, lab in zip(dec_pos, dec_lab):
                gi = off + p + 1                      # the token being predicted
                if gi < order or gi >= len(stream):
                    continue
                c = tuple(stream[gi - order:gi])
                guess = pred.get(c, env.obs_offset)
                ok += int((guess == env.reward_tok) == bool(lab))
                tot += 1
            off += n
        ngram[order] = ok / max(tot, 1)

    return dict(
        fixed_loop=fixed_loop,
        always_no=dict(hit=0.0, false_alarm=0.0, boundary_acc=(K - 1) / K, exact=0.0),
        always_yes=dict(hit=1.0, false_alarm=1.0, boundary_acc=1 / K, exact=0.0),
        random_boundary_exact=1.0 / K,
        ngram_boundary_acc=ngram,
        positional_exact=g5,
        oracle_exact=1.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=2000)
    ap.add_argument("--n-laps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="LAP_GATES.md")
    args = ap.parse_args()

    rows = [gates(fl, args.n_episodes, args.n_laps, args.seed) for fl in (True, False)]
    for r in rows:
        print(f"fixed_loop={r['fixed_loop']}  positional_exact={r['positional_exact']:.3f}  "
              f"ngram(boundary acc)={ {k: round(v,3) for k,v in r['ngram_boundary_acc'].items()} }")

    K = args.n_laps
    lines = [
        "# Lap task -- pre-flight gates (CPU, no training)", "",
        f"K={K} laps. Exactly one of the K lap boundaries is a REWARD boundary.",
        "Headline metric is **exact** = hit the right boundary AND no false alarms.", "",
        "| variant | positional-shortcut exact | n-gram boundary acc (o1/o2/o4/o8) | random-boundary exact | oracle |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        ng = r["ngram_boundary_acc"]
        lines.append(
            f"| loop_len {'FIXED' if r['fixed_loop'] else 'VARIABLE'} | "
            f"**{r['positional_exact']:.3f}** | "
            f"{ng[1]:.3f} / {ng[2]:.3f} / {ng[4]:.3f} / {ng[8]:.3f} | "
            f"{r['random_boundary_exact']:.3f} | {r['oracle_exact']:.3f} |")
    lines += [
        "", "## Floors that matter", "",
        f"- always-say-no: exact **0.000**, but boundary accuracy **{(K-1)/K:.3f}** for free.",
        "  This is why boundary accuracy alone must NOT be the headline.",
        "- always-say-yes: hit 1.000 but false-alarm 1.000, exact 0.000.",
        f"- random boundary: exact **{1.0/K:.3f}**.",
        "", "## Reading the positional gate", "",
        "`positional-shortcut exact` fits the single best 'REWARD is at token index i'",
        "rule on half the episodes and tests it on the other half. With a FIXED loop",
        "length the reward always lands at index K*loop_len, so this rule is perfect and",
        "the task measures nothing. Variable loop length is the operating point iff this",
        "collapses.",
    ]
    open(args.out, "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(args.out.replace(".md", ".json"), "w"), indent=2, default=str)
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

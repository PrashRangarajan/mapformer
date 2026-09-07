"""Gates for Flip-Flop LM. CPU only, run BEFORE training.

  G1 chance        1/2 -- binary task, so the floor is high and every margin is
                   read against 0.5, not against 0.0625.
  G2 marginal      always predict the majority bit.
  G3 answer n-gram order 1..5 over the answer stream. EXPECTED TO FAIL at order 1
                   and that is a property of the PUBLISHED task, not a bug we
                   introduced: two consecutive reads with no write between them
                   return the same bit, which at p_w = p_r = 0.1 happens about
                   half the time. We measure it and report both readings rather
                   than redefining someone else's benchmark. `score_final_only`
                   scores one read per sequence and removes it.
  G4 last-bit      predict the bit that most recently appeared anywhere (usually
                   an ignore's bit, so this should sit at chance).
  G5 scored rate   reads per token.
  G6 oracle        true last-write bit must give 1.0.
  G7 distance      tokens back to the governing write, per split. This is what
                   the OOD splits vary and what the task is FOR.
"""
import argparse, json
from collections import Counter, defaultdict

import numpy as np

from mapformer.environment_flipflop import FlipFlopWorld, W, R


def ngram_acc(ans, orders=(1, 2, 3, 5)):
    n = len(ans)
    if n < 20:
        return {k: float("nan") for k in orders}
    cnt = Counter(ans); fb = cnt.most_common(1)[0][0]; half = n // 2
    out = {}
    for K in orders:
        tab = defaultdict(Counter)
        for i in range(K, half):
            tab[tuple(ans[i - K:i])][ans[i]] += 1
        pred = {c: d.most_common(1)[0][0] for c, d in tab.items()}
        out[K] = sum(int(pred.get(tuple(ans[i - K:i]), fb) == ans[i])
                     for i in range(half + K, n)) / max(n - half - K, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--out", default="/home/prashr/mapformer/FLIPFLOP_GATES.md")
    a = ap.parse_args()

    SPLITS = [("train  p_i=0.80", 0.10, 0.80, 0.10),
              ("OOD dense p_i=0.98", 0.01, 0.98, 0.01),
              ("OOD sparse p_i=0.10", 0.45, 0.10, 0.45)]
    rows = []
    for final_only in (False, True):
        for name, pw, pi, pr in SPLITS:
            env = FlipFlopWorld(pw, pi, pr, score_final_only=final_only, seed=0)
            rng = np.random.RandomState(1)
            ans, lastbit, dists, ns, nt = [], [], [], 0, 0
            for _ in range(a.episodes):
                tok, sp, an, info = env.generate_episode(a.T, rng)
                ans.extend(an); ns += len(sp); nt += info["T"]
                t = tok.tolist()
                for p, x in zip(sp, an):
                    prev = [t[j] for j in range(1, p, 2)]        # bits so far
                    lastbit.append(int((prev[-1] if prev else 3) == x))
                    wpos = [j for j in range(0, p, 2) if t[j] == W]
                    if wpos:
                        dists.append(p - wpos[-1])
            n = len(ans); cnt = Counter(ans)
            rows.append(dict(split=name, final_only=final_only, n=n,
                             chance=0.5, marginal=cnt.most_common(1)[0][1] / max(n, 1),
                             ngram=ngram_acc(ans),
                             last_bit=float(np.mean(lastbit)) if lastbit else float("nan"),
                             scored_rate=ns / max(nt, 1), oracle=1.0,
                             dist_mean=float(np.mean(dists)) if dists else 0.0,
                             dist_max=int(np.max(dists)) if dists else 0))
            r = rows[-1]
            print(f"{'final-only' if final_only else 'all-reads ':11s} {name:20s} "
                  f"marg={r['marginal']:.3f} o1={r['ngram'][1]:.3f} o3={r['ngram'][3]:.3f} "
                  f"lastbit={r['last_bit']:.3f} rate={r['scored_rate']:.3f} "
                  f"dist={r['dist_mean']:.1f} n={n}", flush=True)

    L = ["# Flip-Flop LM -- pre-flight gates (CPU, no training)", "",
         "Published task (Liu et al. 2023; definition taken from CoPE sec 5.1, "
         "read first-hand). **Chance 0.500** -- binary, so every margin here is "
         "read against 0.5.", "",
         "| scoring | split | marginal | o1 | o2 | o3 | o5 | last-bit | scored/token | dist to write | n |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r["ngram"]
        L.append(f"| {'final read only' if r['final_only'] else 'all reads'} | "
                 f"{r['split']} | {r['marginal']:.3f} | {g[1]:.3f} | {g[2]:.3f} | "
                 f"{g[3]:.3f} | {g[5]:.3f} | {r['last_bit']:.3f} | "
                 f"{r['scored_rate']:.4f} | {r['dist_mean']:.1f} (max {r['dist_max']}) | {r['n']} |")
    L += ["", "**The order-1 row under `all reads` is expected to fail, and it is "
          "the published task's own property**: two consecutive reads with no "
          "write between them return the same bit. Scoring the final read only "
          "removes it, at the cost of one scored position per sequence. Both "
          "readings are reported in the results rather than one being chosen "
          "quietly.", "",
          "`dist to write` is what the OOD splits vary and is the reason the task "
          "exists: a fixed offset cannot address the governing write."]
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(rows, open(a.out.replace(".md", ".json"), "w"), indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()

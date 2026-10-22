"""Statistics quoted from the T = 1024 what/where test that had no committed generator (POST HOC, CPU, seconds).

Reads docs/audits/2026-10-20/causal_whatwhere_T1024.json (written by causal_whatwhere_T1024.py; SIGN_MATCHED
checkpoints, seeds 0-7 per arm) and docs/paper/expansion/scripts/language/tw_common_used_out.txt. Prints:
  - per arm: intact, token identity within type removed (pos_class), shared kernel x per-pair gain plus offset (gain);
    mean costs; seeds lower; paired t p; exact sign-flip p (2^8 sign patterns, two-sided);
  - SS(P)/SS(S) (the 'row_pos_share' ratio; the script's 'share of score variance explained by position alone');
  - Spearman rho between intact accuracy and the share of it kept with identity removed (pos_class / intact), over the
    16 non-negative runs (Abs_r4 + Pos_r4), with scipy's p and a permutation p (20k shuffles; resolution 5e-5);
  - the exact permutation p of the text-world Spearman +0.905 (n = 8, all 8! orderings).
Applied from docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md Part 6 MINOR-1/7/9 and Part 5 (tw_common_used).
  python3 whatwhere_stats.py > whatwhere_stats_out.txt
"""
import itertools
import json
import re

import numpy as np
from scipy import stats

REPO = "/home/prashr/mapformer"
D = json.load(open(f"{REPO}/docs/audits/2026-10-20/causal_whatwhere_T1024.json"))["runs"]


def signflip_p(d):
    d = np.asarray(d); obs = abs(d.mean())
    sims = [abs((d * np.array(s)).mean()) for s in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean([x >= obs - 1e-15 for x in sims]))


for arm in ("Signed_r4", "Abs_r4", "Pos_r4", "RoPE"):
    R = D[arm]; seeds = sorted(R, key=int)
    it = np.array([R[s]["intact"] for s in seeds])
    print(f"== {arm} (n {len(seeds)}): intact {it.mean():.4f}")
    for cond in ("pos_class", "gain"):
        v = np.array([R[s][cond] for s in seeds]); d = v - it
        print(f"   {cond:9s} {v.mean():.4f} (cost {d.mean():+.4f}; per seed {d.min():+.3f}..{d.max():+.3f}); "
              f"seeds lower {int((d < 0).sum())}/{len(d)}; paired t p {stats.ttest_1samp(d, 0).pvalue:.4f}; "
              f"exact sign-flip p {signflip_p(d):.4f}")
    ps = np.array([np.mean(R[s]["row_pos_share"]) for s in seeds])
    print(f"   SS(P)/SS(S) (row_pos_share, mean over heads and seeds) {ps.mean():.3f}")

x, y = [], []
for arm in ("Abs_r4", "Pos_r4"):
    for s, r in D[arm].items():
        x.append(r["intact"]); y.append(r["pos_class"] / r["intact"])
x, y = np.array(x), np.array(y)
rho, p = stats.spearmanr(x, y)
rng = np.random.default_rng(0)
perm = np.mean([abs(stats.spearmanr(x, rng.permutation(y))[0]) >= abs(rho) - 1e-12 for _ in range(20000)])
print(f"\nSpearman(intact, pos_class / intact) over the {len(x)} non-negative runs (Abs_r4 + Pos_r4): rho {rho:.3f}, "
      f"scipy p {p:.1e}, permutation p (20k) {perm:.1e}")
print("   share kept by the runs with intact > 0.95: " + ", ".join(f"{b:.3f}" for a, b in zip(x, y) if a > 0.95))

txt = open(f"{REPO}/docs/paper/expansion/scripts/language/tw_common_used_out.txt").read()
c = [float(v) for v in re.findall(r"used channels ([0-9.]+)", txt)]
k = [float(v) for v in re.findall(r"steps ([0-9.]+)\n", txt + "\n")]
rho_tw = stats.spearmanr(c, k)[0]
allr = [stats.spearmanr(c, pp)[0] for pp in itertools.permutations(k)]
pex = np.mean([abs(r) >= abs(rho_tw) - 1e-12 for r in allr])
print(f"\nText world (tw_common_used_out.txt, n {len(c)}): Spearman {rho_tw:+.3f}, exact two-sided permutation p {pex:.4f}")

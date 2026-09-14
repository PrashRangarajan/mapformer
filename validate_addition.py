"""Shortcut gates for the addition task (ADDITION_DESIGN.md), run on the task code itself.

Per-digit accuracy of predictors that see only the SUM stream (orders 1-5: previous k sum digits
predict the next one), a majority-digit predictor, and a reference predictor that copies
(a_j + b_j) mod 10 ignoring carries (a partial algorithm, recorded as a reference, not a shortcut).
Exact-match chance is 10^-(n+1) and is reported by length.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from mapformer.environment_addition import AdditionWorld


def sums(env, rng, n, dmax=None, n_digits=None):
    out = []
    for _ in range(n):
        if n_digits is None:
            la, lb = int(rng.randint(1, dmax + 1)), int(rng.randint(1, dmax + 1))
        else:
            la = lb = n_digits
        a, b = env.sample_operand(rng, la), env.sample_operand(rng, lb)
        m = max(la, lb); a = [0] * (m - la) + a; b = [0] * (m - lb) + b
        out.append((a, b, env.add_digits(a, b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmax", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--out", default="/home/prashr/mapformer/ADDITION_GATES.json")
    a = ap.parse_args()
    env = AdditionWorld("shared")
    tr = sums(env, np.random.RandomState(1), a.n_train, dmax=a.dmax)
    te = sums(env, np.random.RandomState(2), a.n_test, dmax=a.dmax)
    res = {}
    for k in range(1, 6):
        table = defaultdict(Counter)
        for _, _, s in tr:
            for j in range(len(s)):
                table[tuple(s[max(0, j - k):j])][s[j]] += 1
        ok = n = 0
        for _, _, s in te:
            for j in range(len(s)):
                c = table.get(tuple(s[max(0, j - k):j]))
                ok += int(c is not None and c.most_common(1)[0][0] == s[j]); n += 1
        res[f"ngram_o{k}"] = ok / n
    maj = Counter(d for _, _, s in tr for d in s).most_common(1)[0][0]
    res["majority_digit"] = sum(d == maj for _, _, s in te for d in s) / sum(len(s) for _, _, s in te)
    ok = n = 0; ex = 0
    for x, y, s in te:
        m = len(x); pred = [(x[m - 1 - j] + y[m - 1 - j]) % 10 for j in range(m)] + [0]
        ok += sum(p == t for p, t in zip(pred, s)); n += len(s); ex += int(pred == s)
    res["reference_no_carry_digit"] = ok / n; res["reference_no_carry_exact"] = ex / len(te)
    res["chance_digit"] = 0.1
    print(json.dumps(res, indent=1)); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

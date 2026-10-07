"""Smoke test (Amendment 2-3 design; the registered accuracy is p, plain-hard): every branch and qualifier of analyze_rank_nowrap.decide is reachable, on synthetic
per-seed data at the registered n (5 cells; h here stands for the registered p, HIT = p >= HIT_P). Run from /home/prashr:
python3 mapformer/docs/audits/2026-10-06/rank_nowrap_branch_smoke.py [n]   (n defaults to the registered N_SEEDS;
Amendment 4: every case is written in terms of n, half = ceil(n/2), low = floor(0.25 n), and passes at n = 10 and 12)"""
import sys

import numpy as np

sys.path.insert(0, "/home/prashr")
from mapformer import analyze_rank_nowrap as A     # noqa: E402

import math

n = int(sys.argv[1]) if len(sys.argv) > 1 else A.N_SEEDS
half, low = math.ceil(n / 2), math.floor(A.LOW * n)


def cell(hits, fail_h=0.3, hit_h=1.0):
    """hits runs at h = hit_h, the others at fail_h (+ a small spread); raw = 0.9 + 0.1 h (any monotone map)."""
    h = np.array([hit_h] * hits + [fail_h + 0.01 * i for i in range(n - hits)])
    return int((h >= A.HIT_P).sum()), h, np.minimum(0.9 + 0.1 * h, 1.0)


def run(spec, expect, qual=None):
    hit, h, raw = {}, {}, {}
    for k, args in spec.items():
        hit[k], h[k], raw[k] = cell(*args)
    br, q, c = A.decide(hit, n, h, raw)
    ok = br == expect and (qual is None or any(qual in x for x in q))
    print(f"{'PASS' if ok else 'FAIL'}  expected {expect:44s} got {br:44s} | " + "; ".join(q)[:140])
    return ok


def S(a32, b32, m32, al, bl):
    return {"A32": a32, "B32": b32, "M32": m32, "AL": al, "BL": bl}


cases = [
    (S((n,), (n,), (n,), (n,), (n,)), "CEILING"),
    (S((n - 1,), (n,), (1,), (2,), (n,)), "CONTROL FAILED (VOID)"),
    (S((1,), (n,), (1,), (1,), (n // 2 - 1,)), "LARGE-GRID CONTROL FAILED", "hits on"),
    (S((1,), (n,), (1,), (1,), (n // 2, 0.0)), "LARGE-GRID CONTROL FAILED", "detectably worse"),
    (S((4, 0.6), (n,), (1,), (0, 0.0), (n,)), "LARGE GRID HURTS RANK 2"),
    (S((1,), (n,), (1,), (n - 1,), (n,)), "PERIODIC CODE IS THE LIMIT"),
    (S((1,), (n,), (1,), (n - 1,), (n,)), "PERIODIC CODE IS THE LIMIT", "[attaches to the verdict]"),
    (S((0, 0.0), (half,), (0, 0.0), (n,), (half,)), "PERIODIC CODE IS THE LIMIT", "REVERSAL"),
    (S((1,), (n,), (n - 1,), (n - 1,), (n,)), "FIXED MAP WAS THE LIMIT"),
    (S((1,), (n,), (5,), (n - 1,), (n,)), "LARGE TORUS RESCUES RANK 2, CAUSE UNRESOLVED"),
    (S((1,), (n,), (low + 1,), (n - 1,), (n,)), "LARGE TORUS RESCUES RANK 2, CAUSE UNRESOLVED", "neither a clear failure"),
    (S((3, 0.6), (n,), (0, 0.0), (n - 1,), (n,)), "PERIODIC CODE IS THE LIMIT", "HARDER"),
    (S((0,), (n,), (1,), (5,), (n,)), "PARTIAL", "rank 3 still detectably above"),
    (S((0,), (n,), (1,), (7, 0.85), (8, 0.85)), "PARTIAL", "AL HIT"),
    (S((1,), (n,), (1,), (1,), (n,)), "RANK LIMIT IS GENERAL"),
    (S((1,), (n,), (n,), (1,), (n,)), "FIXED MAP AT 32, LARGE TORUS FAILS"),
    (S((2, 0.5), (n,), (1,), (4, 0.4), (n,)), "INTERMEDIATE"),
    (S((2, 0.6), (8,), (2, 0.6), (5, 0.6), (7,)), "UNMEASURED"),
]
res = [run(*c) for c in cases]
print(f"\n{sum(res)}/{len(res)} branch cases pass (n = {n})")
sys.exit(0 if all(res) else 1)

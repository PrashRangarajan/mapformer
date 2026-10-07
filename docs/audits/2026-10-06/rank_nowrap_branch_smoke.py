"""Smoke test (Amendment 1 design): every branch and qualifier of analyze_rank_nowrap.decide is reachable, on synthetic
per-seed data at the registered n (5 cells, HIT counts from floor-relative accuracy). Run from /home/prashr:
python3 mapformer/docs/audits/2026-10-06/rank_nowrap_branch_smoke.py"""
import sys

import numpy as np

sys.path.insert(0, "/home/prashr")
from mapformer import analyze_rank_nowrap as A     # noqa: E402

n = A.N_SEEDS
FL = {"A32": 0.75, "B32": 0.75, "M32": 0.75, "AL": 0.872, "BL": 0.872}


def cell(k, hits, fail_rel=-0.2, hit_rel=1.0):
    """hits runs at floor-relative hit_rel; the others at fail_rel (+ a small spread)."""
    rel = np.array([hit_rel] * hits + [fail_rel + 0.01 * i for i in range(n - hits)])
    raw = np.minimum(FL[k] + rel * (1 - FL[k]), 1.0)
    rel = (raw - FL[k]) / (1 - FL[k])
    return int((rel >= A.HIT_REL).sum()), raw, rel


def run(spec, expect, qual=None, hm=None):
    hit, raw, rel = {}, {}, {}
    for k, args in spec.items():
        hit[k], raw[k], rel[k] = cell(k, *args)
    br, q, c = A.decide(hit, n, raw, rel, hm)
    ok = br == expect and (qual is None or any(qual in x for x in q))
    print(f"{'PASS' if ok else 'FAIL'}  expected {expect:44s} got {br:44s} | " + "; ".join(q)[:140])
    return ok


def S(a32, b32, m32, al, bl):
    return {"A32": a32, "B32": b32, "M32": m32, "AL": al, "BL": bl}


trail = {"AL": [0.80] * n, "BL": [0.99 + 0.001 * i for i in range(n)]}
cases = [
    (S((n,), (n,), (n,), (n,), (n,)), "CEILING"),
    (S((n - 1,), (n,), (1,), (2,), (n,)), "CONTROL FAILED (VOID)"),
    (S((1,), (n,), (1,), (1,), (n // 2 - 1,)), "LARGE-GRID CONTROL FAILED", "hits on"),
    (S((1,), (n,), (1,), (1,), (n // 2, -0.9)), "LARGE-GRID CONTROL FAILED", "detectably worse"),
    (S((4, 0.5), (n,), (1,), (0, -0.9), (n,)), "LARGE GRID HURTS RANK 2"),
    (S((1,), (n,), (1,), (n - 1,), (n,)), "PERIODIC CODE IS THE LIMIT"),
    (S((0, -0.9), (5,), (0, -0.9), (n,), (5,)), "PERIODIC CODE IS THE LIMIT", "REVERSAL"),
    (S((1,), (n,), (1,), (n - 1,), (n,)), "PERIODIC CODE IS THE LIMIT", "ON THE HARD TARGETS RANK 2 STILL TRAILS", trail),
    (S((1,), (n,), (n,), (n - 1,), (n,)), "MAP MEMORISATION WAS THE LIMIT"),
    (S((1,), (8,), (5,), (n - 1,), (n,)), "LARGE TORUS RESCUES RANK 2, CAUSE UNRESOLVED"),
    (S((0,), (n,), (1,), (5,), (n,)), "PARTIAL", "rank 3 still detectably above"),
    (S((0,), (n,), (1,), (7, 0.8), (8, 0.8)), "PARTIAL", "AL HIT"),
    (S((1,), (n,), (1,), (1,), (n,)), "RANK LIMIT IS GENERAL"),
    (S((1,), (n,), (n,), (1,), (n,)), "MEMORISATION AT 32, LARGE TORUS FAILS"),
    (S((2, 0.3), (n,), (1,), (4, 0.0), (n,)), "INTERMEDIATE"),
    (S((2, 0.3), (8,), (2, 0.3), (5, 0.3), (7,)), "UNMEASURED"),
]
res = [run(*c) for c in cases]
print(f"\n{sum(res)}/{len(res)} branch cases pass (n = {n})")
sys.exit(0 if all(res) else 1)

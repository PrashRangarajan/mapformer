"""Smoke test: every branch (and the REVERSAL qualifier) of analyze_rank_nowrap.decide is reachable, on synthetic
per-seed data at the registered n. Run from /home/prashr: python3 mapformer/docs/audits/2026-10-06/rank_nowrap_branch_smoke.py"""
import sys

import numpy as np

sys.path.insert(0, "/home/prashr")
from mapformer import analyze_rank_nowrap as A     # noqa: E402

n = A.N_SEEDS
FL = {"A32": 0.75, "B32": 0.75, "AL": 0.872, "BL": 0.872}


def cell(k, solved, fail_rel=-0.2, solved_rel=1.0):
    """k: cell; solved: SOLVED count; failed runs sit at floor-relative fail_rel (+ small spread)."""
    rel = np.array([solved_rel] * solved + [fail_rel + 0.01 * i for i in range(n - solved)])
    raw = np.minimum(FL[k] + rel * (1 - FL[k]), 1.0)
    return solved, raw, (raw - FL[k]) / (1 - FL[k])


def run(spec, expect, qual=None):
    sol, raw, rel = {}, {}, {}
    for k, args in spec.items():
        sol[k], raw[k], rel[k] = cell(k, *args)
    br, q, c = A.decide(sol, n, raw, rel)
    ok = br == expect and (qual is None or any(qual in x for x in q))
    print(f"{'PASS' if ok else 'FAIL'}  expected {expect:28s} got {br:28s} | " + "; ".join(q)[:150])
    return ok


cases = [
    ({"A32": (n,), "B32": (n,), "AL": (n,), "BL": (n,)}, "CEILING", None),
    ({"A32": (n - 1,), "B32": (n,), "AL": (2,), "BL": (n,)}, "CONTROL FAILED (VOID)", None),
    ({"A32": (1,), "B32": (n,), "AL": (1,), "BL": (n // 2 - 1,)}, "LARGE-GRID CONTROL FAILED", "solves on"),
    ({"A32": (1,), "B32": (n,), "AL": (1,), "BL": (n // 2 + 1, -0.9)}, "LARGE-GRID CONTROL FAILED", "detectably worse"),
    ({"A32": (4, 0.5), "B32": (n,), "AL": (0, -0.9), "BL": (n,)}, "LARGE GRID HURTS RANK 2", None),
    ({"A32": (1,), "B32": (n,), "AL": (n - 1,), "BL": (n,)}, "PERIODIC CODE IS THE LIMIT", None),
    ({"A32": (0, -0.9), "B32": (7,), "AL": (n,), "BL": (7,)}, "PERIODIC CODE IS THE LIMIT", "REVERSAL"),
    ({"A32": (0,), "B32": (n,), "AL": (6,), "BL": (n,)}, "PARTIAL", "rank 3 still detectably above"),
    ({"A32": (0,), "B32": (n,), "AL": (8, 0.9), "BL": (9, 0.9)}, "PARTIAL", "AL"),
    ({"A32": (1,), "B32": (n,), "AL": (1,), "BL": (n,)}, "RANK LIMIT IS GENERAL", None),
    ({"A32": (2,), "B32": (n,), "AL": (5,), "BL": (n,)}, "INTERMEDIATE", None),
    ({"A32": (2, 0.3), "B32": (8,), "AL": (6, 0.3), "BL": (8,)}, "UNMEASURED", None),
]
res = [run(*c) for c in cases]
print(f"\n{sum(res)}/{len(res)} branch cases pass (n = {n})")
sys.exit(0 if all(res) else 1)

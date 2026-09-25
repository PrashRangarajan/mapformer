"""Shared test statistics for analysis scripts (audit 2026-09-24, hygiene patch 02 +
efficiency #7, applied 2026-09-24).

WHY. Since stats_guard landed (2026-09-11) sixteen analysis scripts re-implemented
`2.8 * sd / sqrt(n)` inline and none imported stats_guard; analyze_rank_matched.py then
added an exact permutation test and Fisher's exact test privately. This module is the
ONE place those live -- stats_guard's small-n helpers and analyze_rank_matched /
analyze_rank_proj delegate here -- so a new aggregator imports rather than re-derives.

What it adds to stats_guard, and why
------------------------------------
mde_multiplier()  The house MDE multiplier 2.8 = z_.975 + z_.80 is the LARGE-SAMPLE value.
                  With the sd estimated from n seeds the exact multiplier is
                  t_{.975,n-1} + t_{.80,n-1}: 5.36 at n=3, 3.72 at n=5, 3.26 at n=8,
                  3.00 at n=16. So a quoted "MDE" understates what the design can detect
                  by 92% / 33% / 16% / 7%.
paired_p()        The house verdict |delta| > MDE is the test |t| > 2.8. Its actual
                  two-sided alpha is 0.107 at n=3, 0.068 at n=4, 0.049 at n=5, 0.027 at
                  n=8. Report the p-value, not only the verdict.
signflip_p()      Exact paired permutation (sign-flip) test. Distribution-free, so valid
                  for BIMODAL seeds (some runs train, some do not). Its minimum achievable
                  p is 2/2^n: 0.25 at n=3, 0.125 at n=4, 0.0625 at n=5 -- with fewer than
                  six paired seeds NO distribution-free test can reach p < 0.05, and a
                  "DETECTABLE" there rests entirely on normality.
perm2_p/perm2_ci  Exact two-sample permutation (unpaired), vectorised: the C(N, nb)
                  relabelling table is built once and each test is one gather plus row
                  sums. The exact branch recomputes the statistic on the SHIFTED data, the
                  same arithmetic as analyze_rank_matched's original per-combination loop,
                  so its p-values and its fixed-grid CI reproduce the committed
                  RANK_MATCHED_*_ANALYSIS.txt / RANK_PROJ_ANALYSIS.txt byte for byte (checked
                  when this was applied). perm2_ci(grid=None) walks outward from the observed
                  difference instead, so it cannot clip; grid=(lo, hi) is the old fixed grid
                  and reports whether an end was touched (clipped).
fisher_solved()   Fisher's exact test on solved counts, two-sided.
classify_run()    SOLVED / STALLED / DESCENDING (RANK_MATCHED_PREREG Amendment 2), plus
                  RISING, which the registered rule folds into DESCENDING (a run whose loss
                  went UP by >5% is not "stopped by the budget while descending").
bimodality()      Flags an arm whose seeds split between solved and unsolved -- the case
                  where a mean +/- sd and the paired MDE are the wrong summary.

Everything returns plain floats/dicts; nothing prints.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy import stats

MDE_K = 2.8   # the house large-sample multiplier, kept for backward-compatible tables
SMALL_N = 6   # below this no distribution-free paired test can reach p < .05


# ------------------------------------------------------------------------------ MDE
def mde_multiplier(n: int, alpha: float = 0.05, power: float = 0.80,
                   exact_t: bool = True) -> float:
    """t_{1-alpha/2,n-1} + t_{power,n-1} (exact_t) or the normal-theory 2.8."""
    if not exact_t:
        return MDE_K
    if n < 2:
        return float("nan")
    df = n - 1
    return float(stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df))


def mde(sd: float, n: int, exact_t: bool = True, alpha: float = 0.05,
        power: float = 0.80) -> float:
    return mde_multiplier(n, alpha=alpha, power=power, exact_t=exact_t) * sd / math.sqrt(n)


def verdict_alpha(n: int, k: float = MDE_K) -> float:
    """The two-sided alpha the house rule |delta| > k*sd/sqrt(n) actually runs at."""
    return float(2 * stats.t.sf(k, n - 1))


# ------------------------------------------------------------------------------ paired
def paired_p(d: Sequence[float]) -> float:
    """Two-sided one-sample t-test p on per-seed differences (sd = 0: 0 if mean != 0)."""
    d = np.asarray(list(d), float)
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() != 0 else 1.0
    t = d.mean() / (sd / math.sqrt(d.size))
    return float(2 * stats.t.sf(abs(t), d.size - 1))


def signflip_p(d: Sequence[float], max_exact: int = 20, n_mc: int = 200_000,
               seed: int = 0) -> dict:
    """Exact (n <= max_exact) or Monte Carlo sign-flip test of mean(d) = 0, two-sided.
    Returns {p, exact, min_p}. min_p is the smallest p the design can produce."""
    d = np.asarray(list(d), float)
    n = d.size
    obs = abs(d.mean())
    if n <= max_exact:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        null = np.abs((signs * d).mean(1))
        p = float((null >= obs - 1e-12).mean())
        return {"p": p, "exact": True, "min_p": 2.0 / 2 ** n}
    rng = np.random.default_rng(seed)
    signs = rng.choice((1.0, -1.0), size=(n_mc, n))
    null = np.abs((signs * d).mean(1))
    p = float(((null >= obs - 1e-12).sum() + 1) / (n_mc + 1))
    return {"p": p, "exact": False, "min_p": 2.0 / 2 ** n}


# ------------------------------------------------------------------------------ two-sample
@lru_cache(maxsize=16)
def _relabellings(N: int, nb: int, max_exact: int, n_mc: int, seed: int):
    """Index sets for group b: all C(N, nb) in itertools order if small enough, else
    seeded Monte Carlo draws."""
    if math.comb(N, nb) <= max_exact:
        return np.array(list(itertools.combinations(range(N), nb)), dtype=np.intp), True
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((n_mc, N)), axis=1)[:, :nb], False


def _perm_hits(a, b, shift, idx):
    """Two-sided count of relabellings at least as extreme, for mean(b) - mean(a) - shift.
    The same arithmetic as the original per-combination loop: shift b, then per relabelling
    sb = sum of the drawn values and d = sb/nb - (tot - sb)/(N - nb)."""
    a = np.asarray(a, float); b = np.asarray(b, float) - shift
    x = np.concatenate([a, b]); nb = len(b); obs = b.mean() - a.mean()
    tot = x.sum()
    sb = x[idx].sum(axis=1)
    d = sb / nb - (tot - sb) / (len(x) - nb)
    return int((np.abs(d) >= abs(obs) - 1e-12).sum())


def perm2_p(a: Sequence[float], b: Sequence[float], shift: float = 0.0,
            max_exact: int = 250_000, n_mc: int = 200_000, seed: int = 0) -> dict:
    """Two-sided permutation p for mean(b) - mean(a) - shift = 0 (unpaired)."""
    idx, exact = _relabellings(len(a) + len(b), len(b), max_exact, n_mc, seed)
    hits = _perm_hits(a, b, shift, idx)
    p = hits / len(idx) if exact else (hits + 1) / (len(idx) + 1)
    return {"p": float(p), "exact": exact, "min_p": 2.0 / math.comb(len(a) + len(b), len(b))}


def perm2_ci(a: Sequence[float], b: Sequence[float], level: float = 0.95,
             step: float = 0.001, grid: tuple[float, float] | None = None,
             max_exact: int = 250_000, n_mc: int = 200_000, seed: int = 0) -> dict:
    """Test-inversion CI for the shift: every s with perm2_p(a, b, s) >= 1 - level.

    grid=None: walk outward from the observed difference in `step`s until the test rejects
    on each side -- nothing to clip at. grid=(lo, hi): the fixed grid
    np.arange(lo, hi + 1e-9, step), min/max of the accepted points (analyze_rank_matched's
    original rule, reproduced exactly); `clipped` says whether an end of the grid was
    itself accepted, i.e. the true interval may extend past it.
    Returns {lo, hi, clipped}; lo/hi are None if no grid point is accepted."""
    idx, exact = _relabellings(len(a) + len(b), len(b), max_exact, n_mc, seed)
    alpha = 1 - level

    def acc(s):
        hits = _perm_hits(a, b, s, idx)
        p = hits / len(idx) if exact else (hits + 1) / (len(idx) + 1)
        return p >= alpha

    if grid is not None:
        pts = np.arange(grid[0], grid[1] + 1e-9, step)
        ok = [s for s in pts if acc(s)]
        if not ok:
            return {"lo": None, "hi": None, "clipped": False}
        return {"lo": min(ok), "hi": max(ok), "clipped": bool(acc(pts[0]) or acc(pts[-1]))}
    d0 = float(np.mean(b) - np.mean(a))
    span = 4 * (float(np.ptp(np.concatenate([np.asarray(a, float), np.asarray(b, float)]))) or 1.0)
    lo = hi = d0
    while acc(lo - step) and d0 - lo < span:
        lo -= step
    while acc(hi + step) and hi - d0 < span:
        hi += step
    return {"lo": float(lo), "hi": float(hi), "clipped": bool(d0 - lo >= span or hi - d0 >= span)}


def fisher_solved(solved_a: int, n_a: int, solved_b: int, n_b: int) -> float:
    """Two-sided Fisher exact p on the table [[solved_b, n_b - solved_b], [solved_a, ...]]."""
    return float(stats.fisher_exact([[solved_b, n_b - solved_b], [solved_a, n_a - solved_a]])[1])


# ------------------------------------------------------------------------------ run classes
def classify_run(losses: Sequence[float], solved_loss: float = 0.05,
                 stall_tol: float = 0.05) -> dict:
    """Amendment-2 classes from per-epoch loss, with RISING split out of DESCENDING.
    `registered` is the pre-registered label (SOLVED / STALLED / DESCENDING); `cls` is the
    same except that a run whose final-10% mean loss is >5% ABOVE the 10% before it reads
    RISING. tail = mean of the final 5%; ratio = final-10% mean / previous-10% mean."""
    l = np.asarray(losses, float); E = len(l)
    k5, k10 = max(1, round(0.05 * E)), max(1, round(0.10 * E))
    tail = l[-k5:].mean()
    if tail < solved_loss:
        return {"cls": "SOLVED", "registered": "SOLVED", "tail": tail, "ratio": None}
    last, prev = l[-k10:].mean(), l[-2 * k10:-k10].mean()
    ratio = last / prev
    if abs(ratio - 1) < stall_tol:
        return {"cls": "STALLED", "registered": "STALLED", "tail": tail, "ratio": ratio}
    return {"cls": "RISING" if ratio > 1 else "DESCENDING", "registered": "DESCENDING",
            "tail": tail, "ratio": ratio}


def bimodality(x: Sequence[float], solved: Sequence[bool] | None = None,
               threshold: float | None = None) -> dict:
    """Solved fraction and Sarle's bimodality coefficient (> 0.555 suggests bimodal).
    `solved` (from classify_run) is preferred over an accuracy threshold."""
    x = np.asarray(x, float); n = x.size
    if solved is None and threshold is not None:
        solved = x >= threshold
    frac = None if solved is None else float(np.mean(solved))
    bc = float("nan")
    if n > 3 and x.std() > 0:
        g = stats.skew(x, bias=False); k = stats.kurtosis(x, bias=False)
        bc = float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))
    return {"solved_frac": frac, "bimodality_coef": bc,
            "mixed": frac is not None and 0 < frac < 1}


def report(a: Sequence[float], b: Sequence[float], label: str = "b - a",
           paired: bool = True) -> str:
    """One line with every number a results file should quote for a contrast."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = [label]
    if paired:
        d = b - a; n = d.size; sd = d.std(ddof=1)
        sf = signflip_p(d)
        out.append(f"{d.mean():+.4f} (sd {sd:.4f}, MDE {MDE_K*sd/math.sqrt(n):.4f} house / "
                   f"{mde(sd, n):.4f} exact-t, {int((d > 0).sum())}/{n} +) "
                   f"t-test p {paired_p(d):.4f}, sign-flip p {sf['p']:.4f} (min {sf['min_p']:.4f})"
                   + (f" [n<{SMALL_N}: no exact test can reach .05]" if n < SMALL_N else ""))
    pp = perm2_p(a, b)
    out.append(f"two-sample perm p {pp['p']:.4f}{'' if pp['exact'] else ' (MC)'}")
    return "  ".join(out)

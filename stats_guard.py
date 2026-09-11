"""Paired-contrast reporting and the replication guard, as one tested implementation.

WHY THIS EXISTS. The same ~20 lines (paired delta by seed, sd, MDE, seeds-positive,
verdict) were re-implemented by hand in roughly fifteen aggregators, and every
retraction of 2026-08-26..09-11 was a failure of one of the pieces below to be
applied, not a failure of the arithmetic. This module is the arithmetic plus the
guard rails, so a new aggregator imports it instead of re-deriving it.

What each function encodes, and where it was bought
---------------------------------------------------
paired()          rule 11 -- "null" requires power. MDE = 2.8 * sd / sqrt(n) (~80%
                  power, two-sided alpha .05). |delta| > MDE -> "DETECTABLE", else
                  "unmeasured". NEVER "null": at n=9, sd 0.18 the MDE is 0.165,
                  larger than every effect being dismissed (CLAUDE.md rule 11).
                  Seeds-positive counts d > 0 whatever the sign of delta, as every
                  table in the repo does (D5: -0.033 is "9/24").
interaction()     2x2 interactions compare two paired contrasts. Unpaired form:
                  MDE = 2.8 * sqrt(var1/n1 + var2/n2). If both contrasts are on the
                  same seeds, paired=True takes the per-seed difference of
                  differences instead (tighter, and what LOOP_HEADROOM did).
rule9()           rule 9 -- is accuracy just the training loss? r(loss, acc) over all
                  runs. At |r| > 0.98 the held-out eval carries no information the
                  loss does not.
loss_matched()    pooled linear fit acc ~ loss over every run in the pool, residuals
                  paired by seed. Carries rule9() with it, and WARNS at |r| > 0.98
                  that loss-matching conditions on a MEDIATOR: a function-class
                  limit also shows up as worse fit, so a zero residual cannot by
                  itself distinguish optimisation from representation
                  (AUDIT_2026-09-10.md finding 7).
replication_split() rule 6 + D5 E4 -- report the pooled estimate AND first-k seeds vs
                  the remaining (fresh) seeds. Flags a sign flip and flags a fresh
                  estimate that no longer clears its own MDE. D5_RESULTS.md: the n=8
                  +0.120 became -0.109 on seeds 8-23; the n=8 +0.237 of DOF D4 was
                  1.9x high. A same-seed rerun that "matches" is ONE sample twice
                  (rule 27) -- see ckpt_guard.compare_checkpoints for that half.

Seed alignment: inputs are either dicts {seed: value} or sequences. A sequence is
read as index == seed, which is how the repo's _*.json summary files are written
(test_guards.py verifies this against the per-run JSON). Dicts with different seed
sets are paired on the intersection and the dropped seeds are REPORTED, never
silently discarded.

Usage
-----
    from mapformer.stats_guard import paired, loss_matched, replication_split, load_arms
    acc, loss = load_arms("_MAGONLY.json")
    c = paired(acc["EMDoF_alignfree"], acc["EMDoF_magonly"], "AlignFree - MagOnly")
    print(c.row())            # | AlignFree - MagOnly | +0.146 | 0.150 | 0.086 | 22/24 | DETECTABLE |
    lm, r9 = loss_matched(acc, loss, "EMDoF_alignfree", "EMDoF_magonly")
    print(r9)                 # r, and the mediator warning when |r| > 0.98
    print(replication_split(c.diffs_by_seed, first_k=8).report())
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, Union

import numpy as np

REPO = Path("/home/prashr/mapformer")
MDE_K = 2.8                 # z_{.975} + z_{.80} ~ 1.96 + 0.84
MEDIATOR_R = 0.98           # |r(loss, acc)| above this -> the mediator warning

Values = Union[Mapping[int, float], Sequence[float], np.ndarray]


# ----------------------------------------------------------------------------- basics
def mde(sd: float, n: int) -> float:
    """Minimum detectable effect of a paired mean, ~80% power at alpha .05."""
    return MDE_K * sd / math.sqrt(n)


def verdict(delta: float, mde_: float) -> str:
    """'DETECTABLE' iff |delta| > MDE. Otherwise 'unmeasured' -- never 'null'."""
    return "DETECTABLE" if abs(delta) > mde_ else "unmeasured"


def _as_map(x: Values, seeds: Sequence[int] | None = None) -> dict[int, float]:
    if isinstance(x, Mapping):
        return {int(k): float(v) for k, v in x.items()}
    x = list(x)
    if seeds is None:
        seeds = range(len(x))
    seeds = list(seeds)
    if len(seeds) != len(x):
        raise ValueError(f"{len(x)} values but {len(seeds)} seeds")
    return {int(s): float(v) for s, v in zip(seeds, x)}


def _align(a: Values, b: Values, seeds=None, label=""):
    A, B = _as_map(a, seeds), _as_map(b, seeds)
    common = sorted(set(A) & set(B))
    dropped = sorted(set(A) ^ set(B))
    if dropped:
        warnings.warn(f"[stats_guard] {label or 'contrast'}: seeds {dropped} present in only "
                      f"one arm were DROPPED; pairing on {len(common)} common seeds",
                      stacklevel=3)
    return common, np.array([A[s] for s in common]), np.array([B[s] for s in common])


# ----------------------------------------------------------------------------- contrasts
@dataclass
class Contrast:
    """A paired contrast by seed. `diffs_by_seed` keeps the per-seed differences so the
    same object can be handed to replication_split()."""
    label: str
    delta: float
    sd: float
    n: int
    mde: float
    n_pos: int
    diffs_by_seed: dict[int, float] = field(repr=False, default_factory=dict)

    @property
    def verdict(self) -> str:
        return verdict(self.delta, self.mde)

    @property
    def t(self) -> float:
        return self.delta / (self.sd / math.sqrt(self.n)) if self.sd > 0 else float("inf")

    def row(self) -> str:
        """Markdown row in the house column order: contrast | delta | sd | MDE | seeds + | verdict."""
        return (f"| {self.label} | {self.delta:+.3f} | {self.sd:.3f} | {self.mde:.3f} | "
                f"{self.n_pos}/{self.n} | {self.verdict} |")

    def __str__(self):
        return (f"{self.label}: {self.delta:+.3f} (sd {self.sd:.3f}, MDE {self.mde:.3f}, "
                f"{self.n_pos}/{self.n} seeds +, t {self.t:.2f}) {self.verdict}")


TABLE_HEADER = "| contrast | delta | sd | MDE | seeds + | verdict |\n|---|---|---|---|---|---|"


def table(contrasts: Sequence[Contrast]) -> str:
    return "\n".join([TABLE_HEADER] + [c.row() for c in contrasts])


def from_diffs(d: Values, label: str = "", seeds=None) -> Contrast:
    """Contrast from per-seed differences directly."""
    D = _as_map(d, seeds)
    ks = sorted(D)
    arr = np.array([D[s] for s in ks])
    if arr.size < 2:
        raise ValueError(f"{label}: need n >= 2 seeds for an sd, got {arr.size}")
    sd = float(arr.std(ddof=1))
    return Contrast(label, float(arr.mean()), sd, int(arr.size), mde(sd, arr.size),
                    int((arr > 0).sum()), {s: D[s] for s in ks})


def paired(a: Values, b: Values, label: str = "", seeds=None) -> Contrast:
    """`a - b`, paired by seed."""
    ks, A, B = _align(a, b, seeds, label)
    return from_diffs(dict(zip(ks, A - B)), label)


@dataclass
class Interaction:
    label: str
    delta: float
    se: float
    mde: float
    n1: int
    n2: int
    paired: bool

    @property
    def verdict(self) -> str:
        return verdict(self.delta, self.mde)

    def __str__(self):
        form = "paired diff-of-diffs" if self.paired else "unpaired"
        return (f"{self.label}: {self.delta:+.3f} (se {self.se:.3f}, MDE {self.mde:.3f}, "
                f"n {self.n1}/{self.n2}, {form}) {self.verdict}")


def interaction(c1: Contrast | Values, c2: Contrast | Values, label: str = "",
                paired: bool = False) -> Interaction:
    """Interaction = contrast1 - contrast2. Unpaired: MDE = 2.8*sqrt(var1/n1 + var2/n2).
    paired=True (same seeds in both) uses the per-seed difference of differences."""
    d1 = c1.diffs_by_seed if isinstance(c1, Contrast) else _as_map(c1)
    d2 = c2.diffs_by_seed if isinstance(c2, Contrast) else _as_map(c2)
    if paired:
        c = globals()["paired"](d1, d2, label)
        return Interaction(label, c.delta, c.sd / math.sqrt(c.n), c.mde, c.n, c.n, True)
    a1 = np.array(list(d1.values())); a2 = np.array(list(d2.values()))
    se = math.sqrt(a1.var(ddof=1) / a1.size + a2.var(ddof=1) / a2.size)
    return Interaction(label, float(a1.mean() - a2.mean()), se, MDE_K * se,
                       a1.size, a2.size, False)


# ----------------------------------------------------------------------------- rule 9
@dataclass
class Rule9:
    r: float
    n: int
    slope: float
    intercept: float
    resid_sd: float

    @property
    def mediator_warning(self) -> bool:
        return abs(self.r) > MEDIATOR_R

    def __str__(self):
        s = (f"rule 9: r(final loss, acc) = {self.r:+.3f} over {self.n} runs; "
             f"acc = {self.intercept:.3f} {self.slope:+.3f}*loss, resid sd {self.resid_sd:.3f}")
        if self.mediator_warning:
            s += ("\n  WARNING |r| > 0.98: held-out accuracy is an affine readout of training "
                  "loss, so any raw 'effect' is a loss gap. The loss-matched residual is the "
                  "honest contrast -- BUT loss-matching conditions on a MEDIATOR: a "
                  "function-class limit also shows up as worse fit, so a zero residual "
                  "cannot by itself distinguish optimisation from representation "
                  "(AUDIT_2026-09-10.md finding 7). That needs an existence construction "
                  "(rule 29), not this statistic.")
        return s


def rule9(acc: Sequence[float], loss: Sequence[float]) -> Rule9:
    x = np.asarray(loss, float); y = np.asarray(acc, float)
    if x.size < 3:
        raise ValueError("rule9 needs >= 3 runs")
    b1, b0 = np.polyfit(x, y, 1)
    resid = y - (b0 + b1 * x)
    return Rule9(float(np.corrcoef(x, y)[0, 1]), int(x.size), float(b1), float(b0),
                 float(resid.std(ddof=1)))


def loss_matched(acc: Mapping[str, Values], loss: Mapping[str, Values], a: str, b: str,
                 pool: Sequence[str] | None = None, label: str | None = None
                 ) -> tuple[Contrast, Rule9]:
    """Loss-matched residual contrast `a - b`.

    One linear fit acc ~ loss is POOLED over every run of every arm in `pool` (default:
    all arms given), as agg_sign.py and the D5/MagOnly analyses did. Residuals are then
    paired by seed. Returns (contrast on residuals, rule9 on the pool) -- always read the
    second before quoting the first.
    """
    pool = list(pool) if pool is not None else list(acc)
    A = {v: _as_map(acc[v]) for v in pool}
    L = {v: _as_map(loss[v]) for v in pool}
    xs, ys = [], []
    for v in pool:
        for s in A[v]:
            if s in L[v]:
                xs.append(L[v][s]); ys.append(A[v][s])
    r9 = rule9(ys, xs)
    fit = lambda x: r9.intercept + r9.slope * x
    res = {v: {s: A[v][s] - fit(L[v][s]) for s in A[v] if s in L[v]} for v in pool}
    for v in (a, b):
        if v not in res:
            raise KeyError(f"arm {v!r} not in the pool {pool}")
    c = paired(res[a], res[b], label or f"{a} - {b} (loss-matched)")
    return c, r9


# ----------------------------------------------------------------------------- replication
@dataclass
class Replication:
    pooled: Contrast
    first: Contrast
    fresh: Contrast
    first_k: int

    @property
    def sign_flip(self) -> bool:
        return np.sign(self.first.delta) != np.sign(self.fresh.delta)

    @property
    def fresh_unmeasured(self) -> bool:
        return self.fresh.verdict != "DETECTABLE"

    @property
    def replicates(self) -> bool:
        """Same sign on fresh seeds AND the fresh estimate clears its own MDE."""
        return not self.sign_flip and not self.fresh_unmeasured

    def report(self) -> str:
        lab = self.pooled.label
        o = [f"replication guard: {lab}",
             "| seeds | delta | sd | MDE | seeds + | verdict |", "|---|---|---|---|---|---|"]
        for tag, c in ((f"first {self.first_k}", self.first),
                       (f"fresh ({self.fresh.n})", self.fresh),
                       (f"pooled ({self.pooled.n})", self.pooled)):
            o.append(f"| {tag} | {c.delta:+.3f} | {c.sd:.3f} | {c.mde:.3f} | "
                     f"{c.n_pos}/{c.n} | {c.verdict} |")
        if self.sign_flip:
            o.append("  FLAG sign flip: the fresh seeds give the opposite sign. The first-k "
                     "estimate was noise; do not quote the pooled number as an effect "
                     "(D5 E4).")
        elif self.fresh_unmeasured:
            o.append("  FLAG the fresh-seed estimate does not clear its own MDE: the "
                     "phenomenon is NOT independently replicated at detectable size; quote "
                     "pooled WITH the fresh estimate beside it (AUDIT_2026-09-10 finding 3).")
        else:
            o.append("  replicates: same sign on fresh seeds and detectable on them alone.")
        ratio = self.first.delta / self.pooled.delta if self.pooled.delta else float("inf")
        if abs(ratio) > 1.5 and not self.sign_flip:
            o.append(f"  note: first-{self.first_k} estimate is {ratio:.1f}x the pooled one "
                     "(D5: n=8 was 1.9x high).")
        return "\n".join(o)


def replication_split(d: Contrast | Values, first_k: int = 8, label: str | None = None,
                      seeds=None) -> Replication:
    """Pooled vs first-k seeds vs remaining ('fresh') seeds, from per-seed differences.
    `first_k` counts the lowest-numbered seeds -- the ones an earlier batch already saw."""
    if isinstance(d, Contrast):
        label = label or d.label
        D = d.diffs_by_seed
    else:
        D = _as_map(d, seeds)
    label = label or "contrast"
    ks = sorted(D)
    if len(ks) < first_k + 2:
        raise ValueError(f"need >= {first_k + 2} seeds to split at {first_k}, got {len(ks)}")
    first = {s: D[s] for s in ks[:first_k]}
    fresh = {s: D[s] for s in ks[first_k:]}
    return Replication(from_diffs(D, label), from_diffs(first, f"{label} [first {first_k}]"),
                       from_diffs(fresh, f"{label} [fresh]"), first_k)


# ----------------------------------------------------------------------------- IO
def load_arms(path: str | Path, acc_key: str = "acc", loss_key: str = "loss"):
    """Read a {arm: {"acc": [...], "loss": [...]}} summary file (_MAGONLY.json, _D5_N24.json).
    Relative paths resolve against REPO, not the cwd (`python3 -m` runs from the parent)."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    d = json.load(open(p))
    return ({v: x[acc_key] for v, x in d.items()},
            {v: x[loss_key] for v, x in d.items() if loss_key in x})

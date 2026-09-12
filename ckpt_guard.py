"""Checkpoint comparison, checkpoint discovery, and intervention manipulation checks.

Three lessons, each learned by an error in the week of 2026-09-05..11.

1. SAME-SEED "REPRODUCTION" IS DETERMINISM, NOT REPLICATION (CLAUDE.md rule 27).
   DOF_RESULTS D4 matched a published +0.237 "to three decimals" and read that as a
   stable effect size. The `sep` and `P0` checkpoints for seeds 0-7 in runs/dof/recency
   and runs/recency_em were BITWISE IDENTICAL, 16/16 (AUDIT_2026-09-10.md finding 10) --
   the same computation twice. At n=24 the contrast was +0.128. compare_checkpoints()
   reports bitwise equality of weights AND loss curves and LABELS an equal pair as
   DETERMINISM: it licenses pairing a new batch with stored arms; it says nothing about
   whether an effect size replicates (use stats_guard.replication_split for that).

   It is NaN-AWARE. The unfreeze/noleak models register trajectory buffers initialised
   to NaN (model_em_unfreeze.py), and torch.equal(nan, nan) is False, so a naive
   comparison reports "not equal" for two identical models at init -- a false alarm hit
   this session. Here NaN positions must match and all other elements must match
   BIT-FOR-BIT (so -0.0 vs +0.0 counts as a difference, as a real bitwise check should).

   Two checkpoint formats exist and both are handled:
       train_recency  {"model_state", "losses", + flat config keys}
       train_variant  {"model_state_dict", "losses", "config"}

2. THE EVALUATOR LAYOUT TRAP. eval_noise_refine.py expects
   runs_dir/<noise-tag>/<variant>_s<seed>/<variant>.pt and silently `continue`s past a
   missing file, printing a table of dashes (hit twice this week). require_checkpoints()
   FAILS LOUDLY: it lists the paths it tried, the .pt files that actually exist under
   runs_dir, and which known layout WOULD have matched.

3. VERIFY THE MANIPULATION IN CODE. Every intervention in this line (frozen pathway,
   masked w_in columns, NaN-initialised recording buffers that must draw no RNG, a new
   arm that must equal its comparator at init) was checked by an ad-hoc script. These
   asserts are the reusable form:
       snapshot / assert_frozen_unchanged / assert_moved / assert_zero
       assert_no_rng_consumed   (torch CPU + CUDA if initialised, numpy global, python)
       assert_same_function_at_init
   assert_moved is the POSITIVE control: a frozen-check that could not have failed
   proves nothing (the trainable parameters must be shown to move in the same run).

Absolute REPO constant: `python3 -m mapformer.X` runs from the PARENT dir, where relative
paths resolve wrongly and fail silently (four occurrences).
"""
from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO = Path("/home/prashr/mapformer")


def repo_path(p: str | Path) -> Path:
    """Absolute as given. A relative path is tried against REPO, then REPO's parent (the
    `mapformer/runs/...` form used from /home/prashr), then the cwd; the first that exists
    wins. If none exists, REPO/p is returned so the error message names the repo path."""
    p = Path(p)
    if p.is_absolute():
        return p
    for base in (REPO, REPO.parent, Path.cwd()):
        if (base / p).exists():
            return (base / p).resolve()
    return REPO / p


# ============================================================================ loading
@dataclass
class Ckpt:
    path: Path
    fmt: str                       # "recency" | "variant"
    state: dict
    losses: list | None
    config: dict
    raw: dict = field(repr=False)


# Widened 2026-09-11: k_fixed / k_curriculum / k_set are part of the recency checkpoint format
# (SEARCH, SPREAD). They were missing, so `ck.config.get("k_set")` read None on a checkpoint that
# recorded one -- which fired SPREAD's arm-vs-checkpoint guard against a batch that was correct.
# An allow-list silently drops what it does not know: add keys here when the trainer saves them.
_FLAT_CONFIG_KEYS = ("variant", "seed", "vocab_size", "d_model", "n_heads", "n_layers",
                     "grid_size", "k_max", "n_symbols", "min_gap",
                     "k_fixed", "k_curriculum", "k_set")


def load_checkpoint(path: str | Path) -> Ckpt:
    """Load either checkpoint format into one shape."""
    p = repo_path(path)
    raw = torch.load(p, map_location="cpu", weights_only=False)
    if "model_state" in raw:
        fmt, sd = "recency", raw["model_state"]
        cfg = {k: raw[k] for k in _FLAT_CONFIG_KEYS if k in raw}
    elif "model_state_dict" in raw:
        fmt, sd = "variant", raw["model_state_dict"]
        cfg = dict(raw.get("config", {}))
    else:
        raise KeyError(f"{p}: no 'model_state' or 'model_state_dict'; keys {sorted(raw)}")
    return Ckpt(p, fmt, sd, raw.get("losses"), cfg, raw)


# ============================================================================ comparison
_INT_VIEW = {torch.float64: torch.int64, torch.float32: torch.int32,
             torch.float16: torch.int16, torch.bfloat16: torch.int16}


def tensors_equal(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    """NaN-aware BITWISE equality. Returns (equal, reason-if-not)."""
    a = a.detach().cpu(); b = b.detach().cpu()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, f"shape/dtype {tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}"
    if a.dtype in _INT_VIEW:
        fa, fb = a.contiguous().reshape(-1), b.contiguous().reshape(-1)
        na, nb = torch.isnan(fa), torch.isnan(fb)
        if not torch.equal(na, nb):
            return False, f"NaN masks differ ({int(na.sum())} vs {int(nb.sum())} NaN)"
        m = ~na
        ia, ib = fa.view(_INT_VIEW[a.dtype])[m], fb.view(_INT_VIEW[a.dtype])[m]
        if torch.equal(ia, ib):
            return True, ""
        d = (fa[m].double() - fb[m].double()).abs()
        return False, f"{int((ia != ib).sum())}/{ia.numel()} elements differ, max |d| {float(d.max()):.3e}"
    if torch.equal(a, b):
        return True, ""
    return False, "values differ"


def _floats_equal(x, y) -> bool:
    if isinstance(x, float) and isinstance(y, float):
        if x != x or y != y:
            return (x != x) and (y != y)
        return struct.pack("<d", x) == struct.pack("<d", y)
    if torch.is_tensor(x) and torch.is_tensor(y):
        return tensors_equal(x, y)[0]
    return x == y


def losses_equal(la: Sequence | None, lb: Sequence | None) -> tuple[bool, str]:
    if la is None or lb is None:
        return (la is None and lb is None), "loss curve missing on one side"
    if len(la) != len(lb):
        return False, f"lengths {len(la)} vs {len(lb)}"
    bad = [i for i, (x, y) in enumerate(zip(la, lb)) if not _floats_equal(x, y)]
    if bad:
        return False, f"{len(bad)}/{len(la)} steps differ, first at {bad[0]}"
    return True, ""


@dataclass
class Comparison:
    n_compared: int
    differing: dict[str, str]
    only_a: list[str]
    only_b: list[str]
    shared_only: bool
    losses_equal: bool | None = None
    losses_reason: str = ""
    a: str = ""
    b: str = ""

    @property
    def weights_equal(self) -> bool:
        keys_ok = self.shared_only or (not self.only_a and not self.only_b)
        return keys_ok and not self.differing

    @property
    def identical(self) -> bool:
        return self.weights_equal and self.losses_equal is not False

    @property
    def verdict(self) -> str:
        return "DETERMINISM" if self.identical else "DIFFERENT"

    def report(self) -> str:
        o = [f"compare  A={self.a}\n         B={self.b}",
             f"  tensors compared {self.n_compared}"
             + (" (shared keys only)" if self.shared_only else "")
             + f", differing {len(self.differing)}"
             + f", only-in-A {len(self.only_a)}, only-in-B {len(self.only_b)}"]
        if self.losses_equal is not None:
            o.append(f"  loss curves {'bitwise equal' if self.losses_equal else 'DIFFER: ' + self.losses_reason}")
        for k, why in list(self.differing.items())[:10]:
            o.append(f"    {k}: {why}")
        if self.only_a or self.only_b:
            o.append(f"    only A: {self.only_a[:6]}  only B: {self.only_b[:6]}")
        if self.identical:
            o.append("  => DETERMINISM: bitwise identical. This is the same computation run "
                     "twice. It licenses pairing with / reusing the stored arm; it is NOT a "
                     "replication and says nothing about effect-size stability (rule 27).")
        else:
            o.append("  => DIFFERENT: not the same computation. Do not pair a new batch with "
                     "the stored arm; retrain comparators in-batch (rule 3).")
        return "\n".join(o)


def compare_state_dicts(sa: Mapping, sb: Mapping, shared_only: bool = False) -> Comparison:
    ka, kb = set(sa), set(sb)
    diff = {}
    shared = sorted(ka & kb)
    for k in shared:
        ok, why = tensors_equal(sa[k], sb[k])
        if not ok:
            diff[k] = why
    return Comparison(len(shared), diff, sorted(ka - kb), sorted(kb - ka), shared_only)


def compare_checkpoints(pa: str | Path, pb: str | Path, shared_only: bool = False) -> Comparison:
    """Bitwise weights + loss curves, either checkpoint format on either side.
    shared_only=True compares only tensors present in both (e.g. U1: an arm with extra
    recording buffers against the arm it must equal)."""
    A, B = load_checkpoint(pa), load_checkpoint(pb)
    c = compare_state_dicts(A.state, B.state, shared_only)
    c.losses_equal, c.losses_reason = losses_equal(A.losses, B.losses)
    c.a, c.b = str(A.path), str(B.path)
    return c


# ============================================================================ discovery
LAYOUTS = {
    # eval_noise_refine.py
    "noise_refine": "{noise_tag}/{variant}_s{seed}/{variant}.pt",
    # train_recency.py --output-dir <runs>/<variant>_s<seed>
    "recency": "{variant}_s{seed}/{variant}_recency.pt",
    # train_variant.py --output-dir <runs>/<variant>_s<seed>
    "variant": "{variant}_s{seed}/{variant}.pt",
    # train_variant.py with a per-config subdir, e.g. runs/<cfg>/s<seed>/<variant>.pt
    "config_seed": "{config}/s{seed}/{variant}.pt",
}


def noise_tag(p: float) -> str:
    """Exactly eval_noise_refine's tag: 0.0 -> 'p0', 0.1 -> 'p01', 0.25 -> 'p025'."""
    return f"p{p:g}".replace(".", "")


class CheckpointLayoutError(FileNotFoundError):
    pass


@dataclass
class Found:
    found: dict[tuple, Path]
    missing: list[Path]
    layout: str


def _expand(layout: str, variants, seeds, noises=None, configs=None):
    tmpl = LAYOUTS.get(layout, layout)
    axes = []
    for v in variants:
        for s in seeds:
            for p in (noises if "{noise_tag}" in tmpl else [None]):
                for c in (configs if "{config}" in tmpl else [None]):
                    kw = dict(variant=v, seed=s)
                    if p is not None:
                        kw["noise_tag"] = noise_tag(p)
                    if c is not None:
                        kw["config"] = c
                    key = tuple(x for x in (p, c) if x is not None) + (v, s)
                    axes.append((key, tmpl.format(**kw)))
    return tmpl, axes


def find_checkpoints(runs_dir, layout: str, variants: Sequence[str], seeds: Iterable[int],
                     noises: Sequence[float] | None = None,
                     configs: Sequence[str] | None = None) -> Found:
    """Resolve every (noise?, config?, variant, seed) combination against a layout
    template (a key of LAYOUTS or a literal format string)."""
    root = repo_path(runs_dir)
    seeds = list(seeds)
    tmpl = LAYOUTS.get(layout, layout)
    if "{noise_tag}" in tmpl and not noises:
        raise ValueError(f"layout {tmpl!r} needs noises=")
    if "{config}" in tmpl and not configs:
        raise ValueError(f"layout {tmpl!r} needs configs=")
    _, axes = _expand(layout, variants, seeds, noises, configs)
    found, missing = {}, []
    for key, rel in axes:
        p = root / rel
        (found.__setitem__(key, p) if p.exists() else missing.append(p))
    return Found(found, missing, tmpl)


def require_checkpoints(runs_dir, layout: str, variants: Sequence[str], seeds: Iterable[int],
                        noises: Sequence[float] | None = None,
                        configs: Sequence[str] | None = None,
                        require_all: bool = False) -> Found:
    """find_checkpoints, but RAISE when nothing is found (or anything, with require_all).
    The error names the tried paths, what .pt files exist, and which layout would match."""
    seeds = list(seeds)
    f = find_checkpoints(runs_dir, layout, variants, seeds, noises, configs)
    if f.found and not (require_all and f.missing):
        if f.missing:
            print(f"[ckpt_guard] WARNING {len(f.missing)} of {len(f.found) + len(f.missing)} "
                  f"expected checkpoints missing, e.g. {f.missing[0]}")
        return f
    root = repo_path(runs_dir)
    msg = [f"{'no' if not f.found else len(f.missing)} checkpoints "
           f"{'found' if not f.found else 'missing'} under {root} "
           f"({'exists' if root.exists() else 'DOES NOT EXIST'}) with layout {f.layout!r}",
           "tried:"] + [f"  {p}" for p in f.missing[:8]]
    if len(f.missing) > 8:
        msg.append(f"  ... and {len(f.missing) - 8} more")
    if root.exists():
        present = []
        for pat in ("*.pt", "*/*.pt", "*/*/*.pt", "*/*/*/*.pt"):
            present += sorted(root.glob(pat))
            if len(present) > 8:
                break
        msg.append("actually present (first 8):")
        msg += [f"  {p.relative_to(root)}" for p in present[:8]] or ["  (no .pt files)"]
        alt = []
        for name in LAYOUTS:
            if LAYOUTS[name] == f.layout:
                continue
            try:
                g = find_checkpoints(runs_dir, name, variants, seeds, noises, configs)
            except ValueError:
                continue
            if g.found:
                alt.append(f"{name!r} ({LAYOUTS[name]}) finds {len(g.found)}")
        msg.append("layouts that WOULD match: " + (", ".join(alt) if alt else "none of the known ones"))
    raise CheckpointLayoutError("\n".join(msg))


# ============================================================================ manipulation checks
def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Cloned parameters AND buffers (state_dict), for before/after checks."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _resolve_after(after):
    return snapshot(after) if isinstance(after, torch.nn.Module) else after


def assert_frozen_unchanged(before: Mapping, after, names: Sequence[str]) -> None:
    """Every named tensor is bitwise (NaN-aware) unchanged. `after` may be the model."""
    after = _resolve_after(after)
    bad = []
    for n in names:
        if n not in before or n not in after:
            raise KeyError(f"{n!r} not in snapshot; similar: "
                           f"{[k for k in before if n.split('.')[-1] in k][:6]}")
        ok, why = tensors_equal(before[n], after[n])
        if not ok:
            bad.append(f"{n}: {why}")
    if bad:
        raise AssertionError("frozen tensors MOVED:\n  " + "\n  ".join(bad))


def assert_moved(before: Mapping, after, names: Sequence[str]) -> None:
    """Positive control: each named tensor changed. A freeze check is only meaningful if
    the trainable tensors are shown to move in the same run."""
    after = _resolve_after(after)
    still = [n for n in names if tensors_equal(before[n], after[n])[0]]
    if still:
        raise AssertionError(f"expected these to train, but they did not move: {still}")


def assert_zero(t: torch.Tensor, index=(), what: str = "tensor") -> None:
    """t[index] is exactly zero, e.g. assert_zero(w_in, (slice(None), slice(2, None)))."""
    sub = t.detach()[index] if index != () else t.detach()
    nz = int(torch.count_nonzero(sub))
    if nz:
        raise AssertionError(f"{what}: {nz}/{sub.numel()} entries nonzero, "
                             f"max |x| {float(sub.abs().max()):.3e}")


def _rng_snapshot(extra: Sequence) -> dict:
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
         "python": random.getstate()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        s["cuda"] = torch.cuda.get_rng_state_all()
    for i, g in enumerate(extra):
        if isinstance(g, torch.Generator):
            s[f"extra{i}"] = g.get_state()
        elif isinstance(g, np.random.RandomState):
            s[f"extra{i}"] = g.get_state()
        elif isinstance(g, np.random.Generator):
            s[f"extra{i}"] = g.bit_generator.state
        else:
            raise TypeError(f"unsupported generator {type(g)}")
    return s


def _same(x, y) -> bool:
    if torch.is_tensor(x):
        return torch.equal(x, y)
    if isinstance(x, np.ndarray):
        return np.array_equal(x, y)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    return x == y


def assert_no_rng_consumed(fn: Callable, *args, extra_generators: Sequence = (), **kw):
    """Run fn(*args, **kw); raise if it advanced torch (CPU, and CUDA if initialised),
    numpy's global, python's `random`, or any generator in `extra_generators`.
    Returns fn's result."""
    before = _rng_snapshot(extra_generators)
    out = fn(*args, **kw)
    after = _rng_snapshot(extra_generators)
    moved = [k for k in before if not _same(before[k], after[k])]
    if moved:
        raise AssertionError(f"RNG consumed: {moved}")
    return out


@torch.no_grad()
def assert_same_function_at_init(model_a: torch.nn.Module, model_b: torch.nn.Module,
                                 inputs: torch.Tensor, atol: float = 0.0) -> float:
    """Both models in eval mode on `inputs`; outputs must agree (bitwise when atol == 0).
    Returns max |diff|. Training modes are restored."""
    ma, mb = model_a.training, model_b.training
    model_a.eval(); model_b.eval()
    try:
        ya, yb = model_a(inputs), model_b(inputs)
    finally:
        model_a.train(ma); model_b.train(mb)
    d = float((ya.double() - yb.double()).abs().max())
    ok = torch.equal(ya, yb) if atol == 0 else d <= atol
    if not ok:
        raise AssertionError(f"not the same function at init: max |diff| {d:.3e} (atol {atol})")
    return d

"""Rewind probe for MapFormer-EM on the recency task, promoted from ad-hoc scripts.

WHAT IT MEASURES
----------------
AUDIT_2026-09-10.md finding 2: single-p0 EM solves recency exactly once the query token
q_k carries Delta = -(k-1) symbol steps, so the inclusive cumsum REWINDS the count and
the retrieval offset is 0 for every k. Two readouts ask whether a model has that code:

1. Rewind slope. Delta(token) = action_to_lie(token_emb(token)) is a function of the
   token id alone, so the whole code is one (vocab, H*n_b) table. With sym = mean Delta
   over symbol tokens, y_k = <Delta(q_k), sym> / <sym, sym> is q_k's step IN UNITS OF
   THE MEAN SYMBOL STEP, and the rewind slope is the least-squares slope of y_k on k.
   Exact rewind = -1, no rewind = 0. Also per (head, block): y_k^hb = Delta(q_k)_hb /
   sym_hb, over blocks where sym_hb is not ~0 (a ratio against ~0 is meaningless).
   This is the statistic model_em_unfreeze._record() stores as traj_slope.

2. A_P selection. On held-out episodes, does the model's OWN position kernel A_P, at the
   q_k position, put its maximum over prior SYMBOL keys on the answer? (Non-symbol keys
   are excluded: filler has Delta = 0 and ties the answer by construction; the content
   branch must gate them -- the construction never claimed otherwise.) Reported summed
   over heads and per head, beside the uniform-over-candidates floor.

FOR THE WARM-START FAMILY (a `token_emb.latent` code): the latent code has TWO
coordinates (count, cycle) and every token's row trains. UNFREEZE_RESULTS.md's first
"latent-code slope" read coordinate 0 only and so missed coordinate-1 contamination and
symbols drifting apart; the corrected statistic uses the COMPLETE latent pathway --
Delta from both latent coordinates through the current w_in / w_out. This probe reports
    slope_latpath   the full latent pathway (== traj_latpath in model_em_noleak)
    slope_coord0    coordinate 0 only -- LEGACY, kept only to reproduce old tables
    leak            ||Delta from content coords|| / ||Delta from latent coords||, symbols

POSITION ORIGINS, three forms, detected rather than assumed:
    single_p0   `p0_pos` parameter (VanillaEM_P0*, EMWarm*, EMUnf*, EMNoLeak*)
    separate    `q0_pos` and `k0_pos` parameters (VanillaEM*, EMDoF_alignfree)
    property    `k0_pos` is a computed property, not a parameter (EMDoF_alignlock,
                EMDoF_magonly) -- read through the property, so the probe sees what
                forward() sees.

TOKEN LAYOUT is read from environment_recency.RecencyWorld (sym_offset, filler_offset,
query_offset, mask_tok), rebuilt from the checkpoint's own n_symbols / k_max / min_gap.
p_query / n_filler / p_filler are not saved by train_recency and are taken at their
defaults (which every recency batch used); pass them if a run differed.

The probe does not perturb the caller's RNG (model construction runs under forked /
restored RNG state; episodes use a private RandomState). test_guards.py checks that.

Usage
-----
    python3 -m mapformer.probe_rewind --runs-dir runs/dof/recency \
        --arms VanillaEM_P0_r4 EMDoF_alignfree --seeds 0-7
    python3 -m mapformer.probe_rewind --ckpt runs/warm/EMWarm_freeze_s0/EMWarm_freeze_recency.pt

    from mapformer.probe_rewind import probe_checkpoint
    r = probe_checkpoint(REPO / "runs/warm/EMWarm_freeze_s0/EMWarm_freeze_recency.pt")
    r["slope"]      # -1.000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch

from mapformer.ckpt_guard import REPO, load_checkpoint, repo_path, require_checkpoints

BLOCK_REL_MIN = 1e-3        # a block is "valid" if |sym_hb| > this * max|sym|


# ----------------------------------------------------------------------------- loading
def build_env(cfg: dict, **overrides):
    from mapformer.environment_recency import RecencyWorld
    for k in ("n_symbols", "k_max"):
        if k not in cfg and k not in overrides:
            raise KeyError(f"checkpoint config lacks {k!r}; pass it explicitly")
    kw = dict(n_symbols=cfg.get("n_symbols"), k_max=cfg.get("k_max"),
              min_gap=cfg.get("min_gap"))
    kw.update(overrides)
    return RecencyWorld(seed=10000, **kw)


def load_model(pt, variant: str | None = None, **env_overrides):
    """-> (model on CPU in eval mode, Ckpt, RecencyWorld). Caller RNG is left untouched."""
    ck = load_checkpoint(pt)
    cfg = ck.config
    variant = variant or cfg.get("variant") or re.sub(r"_recency$", "", ck.path.stem)
    env = build_env(cfg, **env_overrides)
    vocab = cfg.get("vocab_size", env.unified_vocab_size)
    if vocab != env.unified_vocab_size:
        raise ValueError(f"{ck.path}: checkpoint vocab {vocab} != rebuilt env vocab "
                         f"{env.unified_vocab_size}; the token layout would be wrong")
    from mapformer.train_variant import VARIANT_MAP
    np_state, py_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            m = VARIANT_MAP[variant](vocab_size=vocab, d_model=cfg.get("d_model", 128),
                                     n_heads=cfg.get("n_heads", 2),
                                     n_layers=cfg.get("n_layers", 1),
                                     grid_size=cfg.get("grid_size", 64))
    finally:
        np.random.set_state(np_state); random.setstate(py_state)
    m.load_state_dict(ck.state)
    return m.eval(), ck, env


# ----------------------------------------------------------------------------- the code
@torch.no_grad()
def delta_table(model) -> torch.Tensor:
    """(vocab, H*n_b) Delta per token id. Delta depends on the token id only."""
    V = model.token_emb.base.num_embeddings if hasattr(model.token_emb, "base") \
        else model.token_emb.num_embeddings
    x = model.token_emb(torch.arange(V))
    return model.action_to_lie(x[None]).reshape(V, -1).double()


@torch.no_grad()
def latent_split(model):
    """Warm-start family only: (Delta from latent coords, Delta from content coords, n_lat).
    Relies on base coords 0..n_lat-1 being held at zero, which is CHECKED here."""
    te = model.token_emb
    if not hasattr(te, "latent"):
        return None
    n_lat = te.latent.shape[1]
    if torch.count_nonzero(te.base.weight[:, :n_lat]):
        raise AssertionError("token_emb.base coords 0..n_lat-1 are not zero: the latent / "
                             "content split below would be wrong")
    V = te.base.num_embeddings
    x = te(torch.arange(V)).double()
    wi = model.action_to_lie.w_in.weight.double()
    wo = model.action_to_lie.w_out.weight.double()
    d_lat = (x[:, :n_lat] @ wi[:, :n_lat].T) @ wo.T
    d_con = (x[:, n_lat:] @ wi[:, n_lat:].T) @ wo.T
    return d_lat, d_con, n_lat


def _slope(y: torch.Tensor) -> float:
    k = torch.arange(y.shape[0], dtype=y.dtype)
    return float(((k - k.mean()) * (y - y.mean())).sum() / ((k - k.mean()) ** 2).sum())


def _sym_q(env):
    sym = slice(env.sym_offset, env.sym_offset + env.n_symbols)
    q = slice(env.query_offset, env.query_offset + env.k_max)
    return sym, q


def rewind_slope(delta: torch.Tensor, env) -> float:
    """Pooled slope of q_k's Delta, in units of the mean symbol Delta. Exact rewind -1."""
    sym, q = _sym_q(env)
    s = delta[sym].mean(0)
    return _slope(delta[q] @ s / (s @ s))


def block_slopes(delta: torch.Tensor, env, rel_min: float = BLOCK_REL_MIN):
    """Per-(head, block) slopes over valid blocks (|sym_hb| > rel_min * max|sym|)."""
    sym, q = _sym_q(env)
    s = delta[sym].mean(0)
    valid = s.abs() > rel_min * s.abs().max()
    sl = torch.full_like(s, float("nan"))
    for j in torch.nonzero(valid).flatten().tolist():
        sl[j] = _slope(delta[q, j] / s[j])
    return sl, valid


def coord0_slope(model, env) -> float | None:
    """LEGACY traj_lat statistic: latent coordinate 0 only. NOT the latent pathway."""
    te = model.token_emb
    if not hasattr(te, "latent"):
        return None
    sym, q = _sym_q(env)
    lat = te.latent.detach()[:, 0].double()
    return _slope(lat[q] / lat[sym].mean())


# ----------------------------------------------------------------------------- A_P selection
def position_origins(model):
    """-> (q0, k0, form). Reads k0 through a property when that is how forward() gets it."""
    params = dict(model.named_parameters())
    if "p0_pos" in params:
        p = model.p0_pos.detach()
        return p, p, "single_p0"
    if not hasattr(model, "q0_pos"):
        raise AttributeError(f"{type(model).__name__} has no q0_pos / p0_pos: A_P selection "
                             "is defined for MapFormer-EM only")
    form = "separate" if "k0_pos" in params else "property"
    return model.q0_pos.detach(), model.k0_pos.detach(), form


@torch.no_grad()
def ap_selection(model, env, n_episodes: int = 32, T: int = 1024, seed: int = 5000) -> dict:
    """Fraction of k-back queries whose A_P argmax over prior symbol keys is the answer."""
    from mapformer.model import _apply_rope
    q0, k0, form = position_origins(model)
    H, dh = q0.shape
    rng = np.random.RandomState(seed)
    toks, sps, ans, infos = env.generate_batch(n_episodes, T, rng)
    x = toks[:, :-1]                               # what train_recency feeds the model
    B, L = x.shape
    cos_a, sin_a = model.path_integrator(model.action_to_lie(model.token_emb(x)))
    qp = _apply_rope(q0[None, :, None, :].expand(B, -1, L, -1), cos_a, sin_a).double()
    kp = _apply_rope(k0[None, :, None, :].expand(B, -1, L, -1), cos_a, sin_a).double()
    hit_sum = 0; hit_head = np.zeros(H); n = 0; floor = 0.0
    for b in range(B):
        sympos = np.asarray(infos[b]["sym_positions"])
        for p, a, k in zip(sps[b], ans[b], infos[b]["offsets"]):
            if p >= L:
                continue
            cand = sympos[sympos < p]
            tgt = cand[-k]
            assert int(toks[b, tgt]) == a, "answer position does not hold the answer"
            sc = (qp[b, :, p, None, :] * kp[b, :, cand, :]).sum(-1) / math.sqrt(dh)  # (H, nc)
            hit_sum += int(cand[int(sc.sum(0).argmax())] == tgt)
            hit_head += (cand[sc.argmax(-1).numpy()] == tgt)
            floor += 1.0 / len(cand); n += 1
    return dict(form=form, n_queries=n, sel_sum=hit_sum / max(n, 1),
                sel_per_head=(hit_head / max(n, 1)).tolist(), floor=floor / max(n, 1))


# ----------------------------------------------------------------------------- one checkpoint
def probe_model(model, env, select: bool = True, **sel_kw) -> dict:
    d = delta_table(model)
    bl, valid = block_slopes(d, env)
    vb = bl[valid]
    out = dict(slope=rewind_slope(d, env),
               block_median=float(vb.median()) if vb.numel() else None,
               block_n_valid=int(valid.sum()), block_n_total=int(valid.numel()),
               block_below_m05=int((vb < -0.5).sum()),
               block_slopes=[None if math.isnan(v) else v for v in bl.tolist()])
    ls = latent_split(model)
    if ls is not None:
        d_lat, d_con, _ = ls
        sym, _ = _sym_q(env)
        out["slope_latpath"] = rewind_slope(d_lat, env)
        out["slope_coord0"] = coord0_slope(model, env)
        out["leak"] = float((d_con[sym].norm(dim=-1) /
                             d_lat[sym].norm(dim=-1).clamp_min(1e-12)).mean())
    if select:
        out.update(ap_selection(model, env, **sel_kw))
    else:
        out["form"] = position_origins(model)[2] if hasattr(model, "path_integrator") else None
    return out


def probe_checkpoint(pt, variant: str | None = None, select: bool = True, **sel_kw) -> dict:
    m, ck, env = load_model(pt, variant)
    r = probe_model(m, env, select, **sel_kw)
    r.update(path=str(ck.path), variant=ck.config.get("variant"), seed=ck.config.get("seed"),
             final_loss=float(ck.losses[-1]) if ck.losses else None)
    return r


# ----------------------------------------------------------------------------- CLI
def _seeds(spec):
    out = []
    for tok in spec:
        if "-" in tok:
            a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def _fmt(v, f="+.3f"):
    return "-" if v is None else format(v, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--seeds", nargs="+", default=["0-7"])
    ap.add_argument("--layout", default="recency")
    ap.add_argument("--ckpt", nargs="+", default=[])
    ap.add_argument("--no-select", action="store_true")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    paths = [repo_path(p) for p in a.ckpt]
    if a.runs_dir:
        f = require_checkpoints(a.runs_dir, a.layout, a.arms, _seeds(a.seeds))
        paths += [f.found[k] for k in sorted(f.found)]
    if not paths:
        ap.error("give --ckpt or --runs-dir with --arms")
    rows = []
    print(f"{'variant':18s} {'s':>2} {'form':9s} {'slope':>7} {'latpath':>8} {'coord0':>7} "
          f"{'leak':>6} {'blk med':>7} {'<-0.5':>6} {'A_P sel':>7} {'floor':>6}")
    for p in paths:
        r = probe_checkpoint(p, select=not a.no_select, n_episodes=a.episodes, T=a.T)
        rows.append(r)
        print(f"{str(r['variant']):18s} {str(r['seed']):>2} {str(r.get('form')):9s} "
              f"{r['slope']:+7.3f} {_fmt(r.get('slope_latpath')):>8} "
              f"{_fmt(r.get('slope_coord0')):>7} {_fmt(r.get('leak'), '.3f'):>6} "
              f"{_fmt(r['block_median']):>7} {r['block_below_m05']:>3}/{r['block_n_valid']:<2} "
              f"{_fmt(r.get('sel_sum'), '.3f'):>7} {_fmt(r.get('floor'), '.3f'):>6}")
    print("\nslope: -1 = exact rewind, 0 = none. latpath = complete latent pathway (both "
          "coordinates); coord0 is the legacy coordinate-0-only statistic.")
    if a.out:
        json.dump(rows, open(repo_path(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()

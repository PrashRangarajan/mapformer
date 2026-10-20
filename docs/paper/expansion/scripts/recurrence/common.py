"""Shared helpers for the recurrence analyses (CPU, inference only, stored checkpoints).

Arms (torus, trained and tested at T=1024, 900 epochs, seeds 0-7; LOOP_RANK_PREREG.md):
  A   Vanilla       shared r=2, 1 layer        runs/rank_mi/p0/Vanilla_s*      (-> rank_matched_e900)
  L   Looped        shared r=2, 1 block x4     runs/loop_rank/p0/Looped_s*     (same params + init as A)
  L4  Vanilla_L4    shared r=2, 4 real layers  runs/loop_rank/p0/Vanilla_L4_s*
  C   Vanilla_r4mi  shared r=4, 1 layer        runs/rank_mi/p0/Vanilla_r4mi_s*

Evaluation protocol copied from eval_rank_strata.py / eval_noise_refine.py: GridWorld env seed 10000
(the held-out map), np.random.seed(1234 + s), trajectories of T steps, model in eval mode.
`forward_passes` re-implements WMTransformerLayer.forward (manual path, dropout off) so it can return
every pass's hidden state and attention; `check_forward` asserts it matches the model's own logits.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/prashr")
from mapformer.environment import GridWorld          # noqa: E402
from mapformer.train_variant import VARIANT_MAP      # noqa: E402
from mapformer.model import _apply_rope              # noqa: E402

REPO = "/home/prashr/mapformer"
OUT = f"{REPO}/docs/paper/expansion/scripts/recurrence/out"
ARMS = {"A": ("Vanilla", "rank_mi"), "L": ("Looped", "loop_rank"),
        "L4": ("Vanilla_L4", "loop_rank"), "C": ("Vanilla_r4mi", "rank_mi")}
DELTA = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)  # == ACTION_DELTAS


def load(arm, s):
    v, d = ARMS[arm]
    b = torch.load(f"{REPO}/runs/{d}/p0/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
    c = b["config"]
    m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                       n_layers=c["n_layers"], grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"])
    return m.eval(), c, b


def env_for(c):
    return GridWorld(size=c["grid_size"], n_obs_types=c.get("n_obs_types", 16),
                     p_empty=c.get("p_empty", 0.5), seed=10000)


def blocks_of(m, arm, n_passes=None):
    if arm == "L":
        return [m.layers[0]] * (n_passes or m.n_loops)
    return list(m.layers)


@torch.no_grad()
def phases(m, tok):
    """theta (H, L, n_blocks) for one token sequence, from the model's own step map and omega."""
    x = m.token_emb(tok[None])
    delta = m.action_to_lie(x)
    th = torch.cumsum(delta, 1) * m.path_integrator.omega[None, None]
    return th[0].transpose(0, 1)


@torch.no_grad()
def forward_passes(m, arm, tok, n_passes=None, want_attn=True):
    """Returns (list of per-pass logits (L, V), list of per-pass attention (H, L, L) or None)."""
    x = m.token_emb(tok[None])
    cos_a, sin_a = m.path_integrator(m.action_to_lie(x))
    B, L, D = x.shape
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    logits, attns = [], []
    for blk in blocks_of(m, arm, n_passes):
        h = blk.norm1(x)
        Q = blk.q_proj(h).view(B, L, blk.n_heads, blk.d_head).transpose(1, 2)
        K = blk.k_proj(h).view(B, L, blk.n_heads, blk.d_head).transpose(1, 2)
        V = blk.v_proj(h).view(B, L, blk.n_heads, blk.d_head).transpose(1, 2)
        Q = _apply_rope(Q, cos_a, sin_a); K = _apply_rope(K, cos_a, sin_a)
        sc = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(blk.d_head)
        sc = sc.masked_fill(mask, float("-inf"))
        a = F.softmax(sc, -1)
        out = torch.matmul(a, V).transpose(1, 2).reshape(B, L, D)
        x = x + blk.o_proj(out)
        x = x + blk.ffn(blk.norm2(x))
        logits.append(m.out_proj(m.out_norm(x))[0])
        attns.append(a[0] if want_attn else None)
    return logits, attns


def check_forward(m, arm, tok):
    with torch.no_grad():
        ref = m(tok[None])[0]
    lg, _ = forward_passes(m, arm, tok, want_attn=False)
    err = (lg[-1] - ref).abs().max().item()
    assert err < 1e-4, (arm, err)
    return err


def step_info(tok, size):
    """Per step t: wrapped cell id, unwrapped-position-is-new flag, lag since last visit to the cell."""
    a = tok[0::2].numpy()
    a = a[: (len(tok) + 1) // 2]
    u = np.cumsum(DELTA[a], 0)
    n = len(a)
    off = n + 1
    ku = (u[:, 0] + off) * (2 * off + 1) + (u[:, 1] + off)
    new_u = np.zeros(n, dtype=bool)
    new_u[np.unique(ku, return_index=True)[1]] = True
    kw = (u[:, 0] % size) * size + (u[:, 1] % size)
    order = np.argsort(kw, kind="stable")
    prev = np.full(n, -10**9, dtype=np.int64)
    same = kw[order[1:]] == kw[order[:-1]]
    prev[order[1:][same]] = order[:-1][same]
    return kw, new_u, np.arange(n) - prev


def stratum(new_u, lag):
    """Same strata as eval_rank_strata.py."""
    return np.where(new_u, "wrap", np.where(lag < 128, "short", "long"))


def trajectories(env, T, n, seed):
    np.random.seed(seed)
    out = []
    for _ in range(n):
        tok, _o, rev = env.generate_trajectory(T)
        out.append((tok, rev))
    return out


os.makedirs(OUT, exist_ok=True)

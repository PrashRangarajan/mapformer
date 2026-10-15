"""Shared helpers for the theory-appendix checks (CPU only; eval-only; nothing here trains or edits a run dir)."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/home/prashr")
import numpy as np
import torch

torch.set_num_threads(4)
REPO = "/home/prashr/mapformer"


def ring_batch(n, T=128, size=32, p_plus=0.5, seed=0):
    """n ring trajectories from the registered environment (environment_cancel.GridWorldCancel); returns tokens (n, 2T),
    positions (n, T) and moves (n, T) in {+1, -1}."""
    from mapformer.environment_cancel import GridWorldCancel
    np.random.seed(seed)
    env = GridWorldCancel(size=size, p_plus=p_plus)
    toks, pos = [], []
    for _ in range(n):
        tok, _, _ = env.generate_trajectory(T)
        toks.append(tok)
        pos.append([p[0] for p in env.visited_locations])
    toks = torch.stack(toks)
    mv = np.where(toks[:, 0::2].numpy() == env.action_offset, 1, -1)
    return toks, np.array(pos), mv, env


def reflect_qk(model, chans):
    """T1 reflection J = diag(1, -1) on rotary channel (h, c) for each (h, c) in chans: negate output coordinate
    h * d_head + 2c + 1 of q_proj and k_proj (weight row AND bias entry) in every layer."""
    with torch.no_grad():
        for layer in model.layers:
            for h, c in chans:
                i = h * layer.d_head + 2 * c + 1
                for p in (layer.q_proj, layer.k_proj):
                    p.weight[i].neg_()
                    if p.bias is not None:
                        p.bias[i].neg_()

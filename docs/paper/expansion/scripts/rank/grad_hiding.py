"""ANALYSIS (post hoc, CPU) for RANK_THEORY.md check C9: which way does the READOUT gradient move the frame at
initialisation, on the real task (torus G64, T = 1024, batch 16; RANK_MI's data path)?

Proposition 2 says: at per-head rank D >= 2, a readout step that lowers drift does so for free at an isotropic frame by
moving resolution AWAY from the drift direction beta-hat ("hiding"); at rank >= D + 1 the readout can instead cancel
the clock along the off-plane latent direction ("parking") without changing the frame.

Per head, from the unperturbed model: beta-hat (drift direction, cells), G = (U sw)(U sw)^T (D x D frame Gram).
Step the readout W_out only, by eps along -grad (and along -sign(grad), Adam's first step), recompute G' with the SAME
beta-hat and channel weights, and report
  hide = (d lam_perp - d lam_beta) / (|d lam_perp| + |d lam_beta|),  lam_beta = bhat^T G bhat, lam_perp = tr G - lam_beta
  (+1: all of the readout's frame change moves resolution off the drift axis; 0: none; random-axis null is symmetric
  around 0), and d kappa (visible drift; negative = drift reduced).
Arms: Vanilla (shared r2) and Vanilla_r4mi (shared r4 from r2's base) rebuilt at init exactly as train_variant does
(torch.manual_seed(seed); np.random.seed(seed); GridWorld(seed); model), seeds 0-3. Gradient averaged over 2 batches.
Output grad_hiding_out.txt.
"""
import copy, os, sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
torch.set_num_threads(2)
from rank_geom import head_geometry, latent_tables  # noqa: E402
from mapformer.environment import GridWorld  # noqa: E402
from mapformer.train_variant import VARIANT_MAP  # noqa: E402


@torch.no_grad()
def frame(m, D=2):
    """Per head (U sw (D, nb), beta (D,), kappa) with the head_geometry conventions."""
    g = head_geometry(m, D=D)
    omega = m.path_integrator.omega.detach().numpy()
    V = m.token_emb.weight.shape[0]; L = m.layers[0]; H = m.n_heads; dh = m.d_model // H
    e = m.token_emb.weight.detach(); hN = L.norm1(e)
    Q = L.q_proj(hN).view(V, H, dh); K = L.k_proj(hN).view(V, H, dh)
    qa = torch.sqrt(Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2); ka = torch.sqrt(K[..., 0::2] ** 2 + K[..., 1::2] ** 2)
    w = (qa[:4].mean(0) * ka[4:].mean(0)).numpy()
    out = []
    for h, (Z, F) in enumerate(latent_tables(m)):
        A = (Z @ F.T) * omega[h][None]
        U = np.stack([(A[2 * d] - A[2 * d + 1]) / 2 for d in range(D)])
        sw = np.sqrt(w[h] / w[h].sum())
        out.append((U * sw, np.array(g[h]["beta"]), g[h]["kappa"]))
    return out


def lam_split(Uw, bhat):
    G = Uw @ Uw.T
    lb = float(bhat @ G @ bhat)
    return lb, float(np.trace(G)) - lb


def main():
    L = []
    for arm in ("Vanilla", "Vanilla_r4mi"):
        for seed in range(4):
            torch.manual_seed(seed); np.random.seed(seed)
            env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=seed)
            m = VARIANT_MAP[arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
            m.train(); crit = nn.CrossEntropyLoss()
            m.zero_grad()
            for _ in range(2):
                tok, om, rev, _ = env.generate_batch(16, 1024)
                logits = m(tok[:, :-1]); mask = rev[:, 1:]
                (crit(logits[mask], tok[:, 1:][mask]) / 2).backward()
            gW = m.action_to_lie.w_out.weight.grad.detach().clone()
            m.eval()
            base = frame(m)
            for kind, dirn in (("grad", -gW / gW.norm()), ("sign", -torch.sign(gW) / torch.sign(gW).norm())):
                for eps in (1e-3,):
                    mp = copy.deepcopy(m)
                    with torch.no_grad():
                        mp.action_to_lie.w_out.weight.add_(eps * dirn * m.action_to_lie.w_out.weight.norm())
                    pert = frame(mp)
                    cells = []
                    for (Uw0, b0, k0), (Uw1, _, k1) in zip(base, pert):
                        bh = b0 / max(np.linalg.norm(b0), 1e-12)
                        lb0, lp0 = lam_split(Uw0, bh); lb1, lp1 = lam_split(Uw1, bh)
                        dlb, dlp = lb1 - lb0, lp1 - lp0
                        hide = (dlp - dlb) / max(abs(dlp) + abs(dlb), 1e-30)
                        cells.append(f"hide {hide:+.2f} dkappa {k1 - k0:+.2e} (kappa0 {k0:.2f})")
                    L.append(f"{arm:13s} s{seed} {kind}: " + " | ".join(cells))
    open(f"{HERE}/grad_hiding_out.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()

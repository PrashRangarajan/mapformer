"""SEARCH_THEORY A4 (eval-only, CPU): how much of the change each class NEEDS can a move of the two move latents supply?

At a model's (W_out, omega) and move latents (z+, z-), a latent move (dz+, dz-) changes channel c's angle sum by
  signed: omega_c w_c . (dz+ + dz-)                            -> column space of diag(omega) W              (rank <= r)
  |.|:    omega_c (e_c(+) w_c . dz+ + e_c(-) w_c . dz-)          -> column space of diag(omega) [E+ W, E- W]  (rank <= 2r)
with e_c(.) = sign of the pre-|.| step. Needed change of the angle sums:
  signed (through zero): -sigma_0 (sigma_0 = omega W (z+ + z-)) -- in the column space by construction (Proposition S2)
  |.| (nearest wrap, k_c = max(1, round(sigma_0 / 2 pi))): 2 pi k_c - sigma_0
Reported: the share of the needed change (squared norm, unweighted and usage-weighted with the solved source's usage) that
lies in the latent-reachable column space; the random-subspace baseline is (rank / C). Inits: MECHANISM seeds 800-807, the
default init (signed) and its |.| twin; also the AO_absmag start (balanced gauge, |default| angles).
  python3 latent_reach.py > latent_reach_out.txt
"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, "/home/prashr")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from latent_geometry import build
from sign_clock import usage

R = Path("/home/prashr/mapformer/runs/wrap_mech/p0")


def share(M, v, w=None):
    if w is not None:
        sw = np.sqrt(w); M = M * sw[:, None]; v = v * sw
    Q, _ = np.linalg.qr(M)
    rank = int(np.linalg.matrix_rank(M))
    return float(np.linalg.norm(Q[:, :rank].T @ v) ** 2 / np.linalg.norm(v) ** 2), rank


def main():
    rows = []
    for seed in range(800, 808):
        m = build(True, seed)                                         # default init (identical weights for SO / AO)
        sd_src = torch.load(R / f"SO_s{seed}/SO.pt", map_location="cpu", weights_only=False)["model_state_dict"]
        u = usage(sd_src); w = u / u.sum()
        with torch.no_grad():
            e = m.token_emb.weight.double(); Wi = m.action_to_lie.w_in.weight.double(); W = m.action_to_lie.w_out.weight.double().numpy()
            z = (e[m.act] @ Wi.T).numpy(); om = m.path_integrator.omega.double().reshape(-1).numpy()
        pre = z @ W.T                                                 # (2, C)
        Ms = om[:, None] * W
        need_s = -(om * (pre[0] + pre[1]))
        sig_abs = om * (np.abs(pre[0]) + np.abs(pre[1]))              # |.| angle sums (omega > 0 at init)
        k = np.maximum(1, np.round(sig_abs / (2 * math.pi)))
        need_a = 2 * math.pi * k - sig_abs
        Ma = np.concatenate([om[:, None] * np.sign(pre[0])[:, None] * W, om[:, None] * np.sign(pre[1])[:, None] * W], 1)
        r_s, rk_s = share(Ms, need_s); r_a, rk_a = share(Ma, need_a)
        rw_s, _ = share(Ms, need_s, w); rw_a, _ = share(Ma, need_a, w)
        # the AO_absmag start: balanced gauge, latents c e1, c e2, W_out[:, :2] = |angle| / omega' / c, other columns 0
        ang = np.abs(om * pre); omp = np.maximum(np.abs(om), 1.0); c = float(np.linalg.norm(z, axis=1).mean())
        Wb = np.zeros_like(W); Wb[:, 0] = ang[0] / omp / c; Wb[:, 1] = ang[1] / omp / c
        Mb = np.concatenate([omp[:, None] * Wb, omp[:, None] * Wb], 1)  # all pre-|.| steps >= 0 at this start
        sigb = ang[0] + ang[1]; kb = np.maximum(1, np.round(sigb / (2 * math.pi)))
        rb, rkb = share(Mb, 2 * math.pi * kb - sigb); rbw, _ = share(Mb, 2 * math.pi * kb - sigb, w)
        print(f"seed {seed}: signed need in latent span {r_s:.3f} (usage-w {rw_s:.3f}, rank {rk_s}) | |.| default init: "
              f"{r_a:.3f} (usage-w {rw_a:.3f}, rank {rk_a}; random baseline {rk_a / len(om):.3f}) | |.| absmag start: {rb:.3f} "
              f"(usage-w {rbw:.3f}, rank {rkb})")
        rows.append((r_s, rw_s, r_a, rw_a, rb, rbw))
    a = np.array(rows).mean(0)
    print(f"\nmean: signed {a[0]:.3f} / {a[1]:.3f} (usage-w) | |.| default {a[2]:.3f} / {a[3]:.3f} | |.| absmag start {a[4]:.3f} / {a[5]:.3f}")


if __name__ == "__main__":
    main()

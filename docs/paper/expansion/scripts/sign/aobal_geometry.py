"""SEARCH_THEORY: the constructed and the held wrapped |.| solutions (WRAP_EXISTENCE AO_bal, ring, seeds 600-607; eval-only).
In the balanced construction both move latents are orthogonal (c e1, c e2) and every pre-|.| step is >= 0, so the |.| clock
is k = diag(omega) W u with u = z_x + z_y -- linear in ONE latent vector, as for signed -- but its target is the vector
b = (2 pi n_c / omega_c)_c (n_c = wrap count), not 0. Reports per run: latent cos, the spread of b over the cancelling used
channels, the distinct wrap counts, and how much of b a readout drawn at the default initialisation could carry (share of
b in col(diag(omega_init) W_init): b must be in the readout's column space for a latent move to reach it).
  python3 aobal_geometry.py > aobal_geometry_out.txt"""
import sys, math, json
from pathlib import Path
sys.path.insert(0, "/home/prashr")
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from sign_clock import usage
from latent_geometry import build
R = Path("/home/prashr/mapformer/runs/wrap_existence/p0")
for s in range(600, 608):
    d = torch.load(R / f"AO_bal_s{s}/AO_bal.pt", map_location="cpu", weights_only=False); sd = d["model_state_dict"]
    e = sd["token_emb.weight"].double(); Wi = sd["action_to_lie.w_in.weight"].double(); W = sd["action_to_lie.w_out.weight"].double()
    om = sd["path_integrator.omega"].double().reshape(-1)
    z = e[:2] @ Wi.T; pre = z @ W.T
    ang = pre.abs() * om; k = ang.sum(0); u = usage(sd); w = u / u.sum()
    n = torch.round(k / (2 * math.pi)); canc = ((k - 2 * math.pi * n).abs() < 0.1) & (n != 0)
    cos = float(z[0] @ z[1] / (z[0].norm() * z[1].norm()))
    coh = float((torch.sign(pre[0]) == torch.sign(pre[1])).double().mean())
    b = (k / om)[canc]
    used = canc & (torch.tensor(w) > 1.0 / len(w))
    m0 = build(False, s)
    with torch.no_grad():
        W0 = m0.action_to_lie.w_out.weight.double().numpy(); om0 = m0.path_integrator.omega.double().reshape(-1).numpy()
    B = om0[:, None] * W0; Q, _ = np.linalg.qr(B)
    tgt = np.zeros(len(om0)); tgt[canc.numpy()] = 2 * math.pi * n[canc].numpy()      # the clock target in angle units
    shr = float(np.linalg.norm(Q.T @ tgt) ** 2 / np.linalg.norm(tgt) ** 2)
    print(f"s{s}: latent cos {cos:+.3f}; pre-|.| same-sign share {coh:.2f}; cancelling channels {int(canc.sum())} "
          f"(used {int(used.sum())}), usage on them {float(w[canc.numpy()].sum()):.2f}; wrap counts {sorted(set(n[canc].int().tolist()))}; "
          f"target sum of steps 2 pi n / omega over cancelling channels: {float(b.min()):.2f}..{float(b.max()):.2f} "
          f"(ratio {float(b.max() / b.min()):.1f}); share of the angle target in an initial readout's column space {shr:.3f}")

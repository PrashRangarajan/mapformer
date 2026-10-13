"""Rule 14 existence for the ring learnability claim (CPU). From each SOLVED Signed_r4 ring model (RING_DIAL, p_plus 0.5,
seeds 30-37) build an Abs_r4 model (|W_out W_in x|) whose rotary angles equal the signed model's mod 2 pi:
- w_in rows 0, 1 = the two action embeddings projected orthogonal to every observation / blank embedding (rows 2, 3 = 0), so
  observation steps are exactly 0 (the audit's construction, `docs/audits/2026-10-12/wrap_construction_t2.py`);
- w_out solved per channel so the +1 / -1 steps are (omega * Delta_signed) mod 2 pi / omega >= 0 (abs is then the identity);
- every other weight copied from the signed model.
Checks, on a fresh ring stream (np seed 10^6, 400 trajectories, T = 128): accuracy of (a) the signed model, (b) the signed
model with its observation steps zeroed (isolates that one change), (c) the constructed Abs model (frozen). Writes the Abs
checkpoints for the trainable warm-start: runs/wrap_existence/init/Abs_r4_p0.5_s<seed>/Abs_r4.pt"""
import math, os, numpy as np, torch
from mapformer.train_variant import VARIANT_MAP
from mapformer.environment_cancel import GridWorldCancel
from mapformer.train_cancel import evaluate

R = "/home/prashr/mapformer/runs/ring_dial/p0"; OUT = "/home/prashr/mapformer/runs/wrap_existence/init"
torch.set_num_threads(8)
te = GridWorldCancel(size=32, seed=10000, p_plus=0.5)
for s in range(30, 38):
    b = torch.load(f"{R}/Signed_r4_p0.5_s{s}/Signed_r4.pt", map_location="cpu", weights_only=False)
    V = b["config"]["vocab_size"]; kw = dict(vocab_size=V, d_model=128, n_heads=2, n_layers=1, grid_size=32)
    sig = VARIANT_MAP["Signed_r4"](**kw); sig.load_state_dict(b["model_state_dict"]); sig.eval()
    E = sig.token_emb.weight.detach(); om = sig.path_integrator.omega.detach().reshape(-1)          # (H*nb,)
    with torch.no_grad():
        D = sig.action_to_lie(E[None])[0].reshape(V, -1)                                            # signed steps (V, C)
    tgt_p = torch.remainder(om * D[0], 2 * math.pi) / om; tgt_m = torch.remainder(om * D[1], 2 * math.pi) / om
    Q, _ = torch.linalg.qr(E[2:].T)                                                                  # obs + blank span
    perp = lambda v: v - Q @ (Q.T @ v)
    W_in = torch.zeros(4, 128); W_in[0] = perp(E[0]); W_in[1] = perp(E[1])
    lat = W_in @ E.T                                                                                 # (4, V)
    M = torch.stack([lat[:2, 0], lat[:2, 1]])                                                        # tokens x latent
    sol = torch.linalg.solve(M, torch.stack([tgt_p, tgt_m]))                                         # (2, C)
    W_out = torch.zeros(om.numel(), 4); W_out[:, :2] = sol.T
    sd = {k: v.clone() for k, v in b["model_state_dict"].items()}
    sd["action_to_lie.w_in.weight"] = W_in; sd["action_to_lie.w_out.weight"] = W_out
    ab = VARIANT_MAP["Abs_r4"](**kw); ab.load_state_dict(sd); ab.eval()
    with torch.no_grad():
        Da = ab.action_to_lie(E[None])[0].reshape(V, -1)
        ang_err = float(torch.abs(torch.angle(torch.exp(1j * om * (Da[:2] - D[:2])))).max())
    # (b) signed with obs steps zeroed: same W_in / W_out construction but signed targets (no wrap)
    sol_s = torch.linalg.solve(M, torch.stack([D[0], D[1]])); W_out_s = torch.zeros_like(W_out); W_out_s[:, :2] = sol_s.T
    sd0 = {k: v.clone() for k, v in b["model_state_dict"].items()}
    sd0["action_to_lie.w_in.weight"] = W_in; sd0["action_to_lie.w_out.weight"] = W_out_s
    sig0 = VARIANT_MAP["Signed_r4"](**kw); sig0.load_state_dict(sd0); sig0.eval()
    a_sig = evaluate(sig, te, 128, 400, "cpu", seed=10 ** 6)[0]
    a_sig0 = evaluate(sig0, te, 128, 400, "cpu", seed=10 ** 6)[0]
    a_abs = evaluate(ab, te, 128, 400, "cpu", seed=10 ** 6)[0]
    print(f"s{s}: Abs steps min {float(Da[:2].min()):.2e} (>= 0), obs |step| max {float(Da[2:].abs().max()):.1e}, action angle error "
          f"mod 2pi max {ang_err:.1e} | acc T=128: signed {a_sig:.4f}, signed obs-steps-zeroed {a_sig0:.4f}, Abs constructed (frozen) {a_abs:.4f}",
          flush=True)
    os.makedirs(f"{OUT}/Abs_r4_p0.5_s{s}", exist_ok=True)
    cfg = dict(b["config"]); cfg["constructed_from"] = f"{R}/Signed_r4_p0.5_s{s}/Signed_r4.pt"
    torch.save({"model_state_dict": ab.state_dict(), "losses": [], "variant": "Abs_r4", "seed": s, "config": cfg},
               f"{OUT}/Abs_r4_p0.5_s{s}/Abs_r4.pt")

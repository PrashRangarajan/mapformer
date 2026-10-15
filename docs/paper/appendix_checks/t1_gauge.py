"""T1 (relative-sign gauge), checked on real checkpoints and on a random multi-layer model.
(a) SignedAO_r4 ring checkpoint (runs/wrap_mech/p0/SO_s800): flip the step sign of a random half of the channels by negating
    the W_out rows (h * nb + c), reflect q/k in those channels (rows and biases): logits unchanged. Control: flip without the
    reflection.
(b) The same with the flip carried by omega instead of W_out (the only place a |.| model can carry a channel's sign), on the
    AbsAO_r4 checkpoint (AO_s800).
(c) Random 2-layer MapFormerWM_Signed_r4 (every token steps; biases non-zero), float64: logits equal to ~1e-13.
(d) Corollary (i), constructive: the AbsAO_r4 checkpoint AO_s800 is rewritten as a SignedAO_r4 model whose move steps are
    D |S| for a random channel-sign matrix D (W_out solved exactly on the two move latents) plus the reflection: logits equal.
(e) Corollary (iii), the converse: SO_s800's weights loaded into the |.| class change the logits (its rows are not
    sign-coherent): the excluded codes are exactly the opposite-sign rows.
"""
import copy
import torch
from common import REPO, ring_batch, reflect_qk
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4, action_angles
from mapformer.model_sign import MapFormerWM_Signed_r4

torch.manual_seed(0)
out = []
P = lambda s: (print(s), out.append(s))


def load(arm, s=800, cls=None):
    b = torch.load(f"{REPO}/runs/wrap_mech/p0/{arm}_s{s}/{arm}.pt", map_location="cpu", weights_only=False)
    cls = cls or (MapFormerWM_SignedAO_r4 if arm.startswith("SO") else MapFormerWM_AbsAO_r4)
    m = cls(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(0, 1))
    m.load_state_dict(b["model_state_dict"]); return m.eval()


toks, pos, mv, env = ring_batch(64, seed=12345)
nb = 32
chans = [(h, c) for h in range(2) for c in range(nb) if torch.rand(()) < 0.5]
P(f"channels flipped: {len(chans)} of 64")

with torch.no_grad():
    # (a)
    so = load("SO"); L0 = so(toks)
    g = copy.deepcopy(so)
    with torch.no_grad():
        for h, c in chans:
            g.action_to_lie.w_out.weight[h * nb + c].neg_()
    Lflip = g(toks); reflect_qk(g, chans); L1 = g(toks)
    A0, A1 = action_angles(so), action_angles(g)
    P(f"(a) SO_s800 float32: move angles flipped in {int((A0 * A1 < 0).sum())} entries; max|logit diff| flip+reflect "
      f"{(L1 - L0).abs().max():.2e} (logit scale {L0.abs().max():.1f}); flip only (control) {(Lflip - L0).abs().max():.2e}")
    so64, g64 = so.double(), g.double()
    P(f"    float64: flip+reflect {(g64(toks) - so64(toks)).abs().max():.2e}")
    # (b)
    ao = load("AO").double(); La = ao(toks)
    g = copy.deepcopy(ao)
    for h, c in chans:
        g.path_integrator.omega[h, c].neg_()
    Lf = g(toks); reflect_qk(g, chans); Lr = g(toks)
    P(f"(b) AO_s800 (|.|), sign carried by omega, float64: flip+reflect {(Lr - La).abs().max():.2e}; flip only {(Lf - La).abs().max():.2e}")
    # (c)
    m = MapFormerWM_Signed_r4(vocab_size=19, d_model=128, n_heads=2, n_layers=2, grid_size=32).double().eval()
    for layer in m.layers:
        layer.q_proj.bias.normal_(0, 0.5); layer.k_proj.bias.normal_(0, 0.5)
    m.action_to_lie.w_out.weight.mul_(5.0)
    Lm = m(toks); g = copy.deepcopy(m)
    for h, c in chans:
        g.action_to_lie.w_out.weight[h * nb + c].neg_()
    Lf = g(toks); reflect_qk(g, chans); Lr = g(toks)
    P(f"(c) random 2-layer Signed_r4, all tokens step, q/k biases N(0, 0.5), float64: flip+reflect {(Lr - Lm).abs().max():.2e}; "
      f"flip only {(Lf - Lm).abs().max():.2e}")
    # (d) Corollary (i): |.| model == signed model with sign-coherent rows D|S|
    ao = load("AO").double(); La = ao(toks)
    so_c = MapFormerWM_SignedAO_r4(vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(0, 1)).double().eval()
    so_c.load_state_dict(ao.state_dict())
    e = ao.token_emb.weight[ao.act]; Z = e @ ao.action_to_lie.w_in.weight.T            # (2, r) move latents
    S = Z @ ao.action_to_lie.w_out.weight.T                                            # (2, C) signed pre-|.| table
    D = torch.ones(2 * nb, dtype=torch.float64)
    for h, c in chans:
        D[h * nb + c] = -1
    tgt = S.abs() * D                                                                  # sign-coherent rows
    Wout = (torch.linalg.pinv(Z) @ tgt).T
    so_c.action_to_lie.w_out.weight.copy_(Wout)
    # omega unchanged; the sign of channel c now sits in W_out, so reflect q/k there
    reflect_qk(so_c, chans)
    St = so_c.step_table()[so_c.act]
    P(f"(d) AO_s800 rewritten as SignedAO with steps D|S| (D: {len(chans)} channels negative): table residual "
      f"{(St.reshape(2, -1) - tgt).abs().max():.2e}; rows sign-coherent: {bool(((St[0] * St[1]).reshape(-1) >= 0).all())}; "
      f"logits max diff {(so_c(toks) - La).abs().max():.2e}")
    # (e) converse
    so = load("SO").double(); Ls = so(toks)
    ab = MapFormerWM_AbsAO_r4(vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(0, 1)).double().eval()
    ab.load_state_dict(so.state_dict())
    A = action_angles(so)
    opp = int((A[0] * A[1] < 0).sum())
    P(f"(e) SO_s800: {opp}/64 channels have opposite-signed move angles; the same weights in the |.| class change the logits "
      f"by up to {(ab(toks) - Ls).abs().max():.2e}; no channel-sign gauge can remove an opposite-sign row (it multiplies both entries by the same sign)")
    # (d2) the same model inside the |.| class with the sign carried by omega: no reflection needed
    ab2 = load("AO").double(); ab2.load_state_dict(so_c.state_dict())                    # same W_out (rows D|S|), |.| applied
    reflect_qk(ab2, chans)                                                             # undo (d)'s reflection
    with torch.no_grad():
        for h, c in chans:
            ab2.path_integrator.omega[h, c].neg_()
    so_d = copy.deepcopy(so_c); reflect_qk(so_d, chans)                                # signed: steps D|S|, omega, AO's q/k
    P(f"(d2) signed model with sign-coherent rows (steps D|S|, omega, no reflection) vs the |.| model with |S| and omega' = D omega: "
      f"angle tables max diff {(action_angles(ab2) - action_angles(so_d)).abs().max():.1e}; logits max diff {(ab2(toks) - so_d(toks)).abs().max():.2e} "
      f"(the |.| class contains it exactly: omega carries the channel sign)")
    # (f) MapEM: the reflection acts on q0_pos / k0_pos
    from mapformer.model import MapFormerEM
    em = MapFormerEM(vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=32).double().eval()
    em.action_to_lie.w_out.weight.mul_(5.0); Le = em(toks); g = copy.deepcopy(em)
    for h, c in chans:
        g.action_to_lie.w_out.weight[h * nb + c].neg_()
    Lf = g(toks)
    for h, c in chans:
        g.q0_pos[h, 2 * c + 1].neg_(); g.k0_pos[h, 2 * c + 1].neg_()
    P(f"(f) MapEM (A_P = rotated q0_pos . rotated k0_pos): flip + reflect q0_pos, k0_pos {(g(toks) - Le).abs().max():.2e}; flip only {(Lf - Le).abs().max():.2e}")
open(__file__.replace(".py", "_out.txt"), "w").write("\n".join(out) + "\n")

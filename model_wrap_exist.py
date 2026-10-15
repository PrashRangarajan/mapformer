"""Rule 14 existence for the ring learnability claim (WRAP_EXISTENCE_PREREG.md).

RING_DIAL: a non-negative rotary step (Abs_r4, |W_out W_in x|) learns the ring map on 2/8 seeds vs 8/8 for the signed step,
although a non-negative rotary step can REPRESENT any signed action code (angles wrap mod 2 pi: a step of
2 pi / omega - Delta reproduces -Delta). The first existence attempt (`docs/audits/2026-10-13/wrap_existence_build.py`)
found that trained Signed_r4 ring solutions rely on their observation steps, which the rank-4 bottleneck cannot also wrap.
So this test uses models whose observation and blank steps are 0 BY CONSTRUCTION and whose two action steps are free
(RANK_TESTS' R1AO, at the shared rank 4 of Signed_r4 / Abs_r4):
  SignedAO_r4   step(+1) = W_out W_in e(+1), step(-1) = W_out W_in e(-1), every other token 0
  AbsAO_r4      the same with |.| per channel (the Abs_r4 constraint)
Both are built from MapWM's base at the seed, so at one seed their initial weights are identical (Abs differs only by |.|).
`wrap_construct(signed, mode=...)` returns a model whose action angles equal the signed model's mod 2 pi (modes in its
docstring). The registered primary (Amendment 1) is `balanced` against its signed control `signed_balanced`.
"""
import math

import torch

from mapformer.model_sign import _sign_variant


def _ao_variant(mode):
    base = _sign_variant(mode, 4)

    class _AO(base):
        def __init__(self, vocab_size, *a, action_ids=(0, 1), **kw):
            super().__init__(vocab_size, *a, **kw)
            self.register_buffer("act", torch.tensor(action_ids, dtype=torch.long), persistent=False)

        def step_table(self):
            """(V, H, nb) per-token step before omega: the actions' steps (after the sign constraint), 0 otherwise."""
            e = self.token_emb.weight
            ea = self.frozen_act_emb if getattr(self, "freeze_latent", False) else e[self.act]   # ACCOUNTS T2
            v = self.action_to_lie(ea.view(1, len(self.act), -1))[0]            # (A, H, nb)
            out = torch.zeros(e.shape[0], *v.shape[1:], device=e.device, dtype=v.dtype)
            out[self.act] = v
            return out

        def freeze_move_latents(self):
            """ACCOUNTS T2 (Amendment 1): the move latents z = W_in e(move) become constants -- a buffer copy of the move
            embeddings feeds the step path (the embeddings still train through the content path) and W_in gets no grad (so
            AdamW applies no decay to it either)."""
            self.register_buffer("frozen_act_emb", self.token_emb.weight[self.act].detach().clone(), persistent=True)
            self.freeze_latent = True
            self.action_to_lie.w_in.weight.requires_grad_(False)

        def delta_of(self, tokens):
            return self.step_table()[tokens]

        def get_position_state(self, tokens: torch.Tensor):
            return self.path_integrator(self.step_table()[tokens])

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            B, L = tokens.shape
            x = self.token_emb(tokens)
            cos_a, sin_a = self.path_integrator(self.step_table()[tokens])
            causal_mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
            for layer in self.layers:
                x = layer(x, cos_a, sin_a, causal_mask)
            return self.out_proj(self.out_norm(x))

    _AO.__name__ = f"MapFormerWM_{mode}AO_r4"
    return _AO


MapFormerWM_SignedAO_r4 = _ao_variant("signed")
MapFormerWM_AbsAO_r4 = _ao_variant("abs")


class MapFormerWM_SignedAO4_r4(MapFormerWM_SignedAO_r4):
    """2D torus (environment.GridWorld): the four move tokens 0-3 carry steps, every other token 0 (TORUS_EXIST)."""
    def __init__(self, vocab_size, *a, action_ids=(0, 1, 2, 3), **kw):
        super().__init__(vocab_size, *a, action_ids=action_ids, **kw)


class MapFormerWM_AbsAO4_r4(MapFormerWM_AbsAO_r4):
    """The same with |.| per channel."""
    def __init__(self, vocab_size, *a, action_ids=(0, 1, 2, 3), **kw):
        super().__init__(vocab_size, *a, action_ids=action_ids, **kw)


def action_angles(m):
    """(A, C) per-channel angle step omega * step of each action (ring: +1, -1; torus: 0-3)."""
    with torch.no_grad():
        st = m.step_table()[m.act].reshape(len(m.act), -1)
        return st * m.path_integrator.omega.reshape(-1)


def wrap_share(m, big=0.05, tol=0.1, pairs=((0, 1),)):
    """Over (action, opposite) pairs and channels with |wrap(angle(a))| > big (wrap-invariant selection, Amendment 1): median
    |wrap(angle(a) + angle(opposite))| and the share below tol rad (0 = the pair cancels mod 2 pi; RING_DIAL's wrap-aware
    opposition, on the AO step table). Ring: pairs ((0, 1),); torus: ((0, 1), (2, 3)) (environment.ACTION_DELTAS)."""
    A = action_angles(m).double(); ws = []
    for i, j in pairs:
        sel = torch.angle(torch.exp(1j * A[i])).abs() > big
        ws.append(torch.angle(torch.exp(1j * (A[i] + A[j])))[sel].abs())
    w = torch.cat(ws)
    if w.numel() == 0:
        return float("nan"), float("nan")
    return float(w.median()), float((w < tol).double().mean())


@torch.no_grad()
def usage_weights(m, tokens, action_offset=0, n_actions=2):
    """ACCOUNTS Amendment 1 (eval-only): per rotary channel (head h, block c -> index h * nb + c, the order of action_angles),
    mean |q_c| over move-token positions (the queries that predict observations) times mean |k_c| over observation-token
    positions (the keys), from the first layer's projections in eval mode. A channel whose q or k slice is ~0 cannot
    influence the score, whatever its angle."""
    m.eval(); layer = m.layers[0]
    h = layer.norm1(m.token_emb(tokens)); B, T, _ = h.shape
    Q = layer.q_proj(h).view(B, T, layer.n_heads, layer.d_head)
    K = layer.k_proj(h).view(B, T, layer.n_heads, layer.d_head)
    qn = torch.sqrt(Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2); kn = torch.sqrt(K[..., 0::2] ** 2 + K[..., 1::2] ** 2)  # (B,T,H,nb)
    is_act = (tokens >= action_offset) & (tokens < action_offset + n_actions)
    wq = qn[is_act].mean(0); wk = kn[~is_act].mean(0)                  # (H, nb)
    return (wq * wk).reshape(-1)


def wrap_share_weighted(m, w, big=0.05, tol=0.1, pairs=((0, 1),)):
    """Usage-weighted cancelling share: sum of w over cancelling channels / sum of w, over (pair, channel) with
    |wrap(angle(a))| > big."""
    A = action_angles(m).double(); num = den = 0.0
    for i, j in pairs:
        sel = torch.angle(torch.exp(1j * A[i])).abs() > big
        canc = torch.angle(torch.exp(1j * (A[i] + A[j]))).abs() < tol
        num += float((w.double() * (sel & canc)).sum()); den += float((w.double() * sel).sum())
    return num / den if den > 0 else float("nan")


MODES = ("min", "gauge", "balanced", "signed_balanced")


def wrap_construct(signed, vocab_size, action_ids, mode="min", **kw):
    """A model with the signed model's action angles mod 2 pi. Returns (model, report). Modes:
      min              AbsAO_r4; W_in kept; W_out min-norm for steps Delta mod (2 pi / |omega|); omega kept
      gauge            AbsAO_r4; as min, then per channel (omega, W_out) -> (k omega, W_out / k), k = max(1, max target /
                       max |source step|)
      balanced         AbsAO_r4 (Amendment 1, from the independent check): W_in re-solved so the two action latents are
                       c e1 and c e2 (orthogonal; c = the source's mean action-latent norm; W_in acts only on the action
                       embeddings, so it is a free gauge); omega' = max(|omega|, 1) per channel; steps
                       remainder(omega Delta, 2 pi) / omega' in [0, 2 pi); W_out[:, :2] = steps^T / c, W_out[:, 2:] = 0
      signed_balanced  SignedAO_r4 at the balanced gauge with the UNWRAPPED signed angle (steps omega Delta / omega'): the
                       matched control for balanced (they differ only in the wrap)
    In a solved signed model the action latents are nearly antiparallel (cos ~ -0.99), so min and gauge inflate W_out
    (pilot: 1825 and 8.4 vs 0.39), and Adam's relative step then moves the phase far (rule 15)."""
    assert mode in MODES, mode
    cls = MapFormerWM_SignedAO_r4 if mode == "signed_balanced" else MapFormerWM_AbsAO_r4
    m = cls(vocab_size, action_ids=action_ids, **kw)
    nA = len(action_ids)
    sd = {k: v.detach().clone() for k, v in signed.state_dict().items()}
    with torch.no_grad():
        om = signed.path_integrator.omega.detach().reshape(-1).double()                  # (C,)
        e = signed.token_emb.weight.detach().double()
        W_in = signed.action_to_lie.w_in.weight.detach().double()                        # (r, d)
        lat = e[signed.act] @ W_in.T                                                     # (A, r)
        D = lat @ signed.action_to_lie.w_out.weight.detach().double().T                  # (A, C) signed steps
        if mode in ("min", "gauge"):
            tgt = torch.remainder(D, 2 * math.pi / om.abs())                             # >= 0, same angle mod 2 pi
            k = torch.ones_like(om)
            if mode == "gauge":
                k = (tgt.max(0).values / D.abs().max()).clamp_min(1.0)
            tgt = tgt / k; om_new = om * k
            W_out = (torch.linalg.pinv(lat) @ tgt).T                                     # (C, r) min-norm
        else:
            r = W_in.shape[0]; c = float(lat.norm(dim=1).mean())
            assert nA <= r, (nA, r)
            L = torch.zeros(nA, r, dtype=lat.dtype); L[torch.arange(nA), torch.arange(nA)] = c
            W_in = (torch.linalg.pinv(e[signed.act]) @ L).T                              # (r, d): e_act W_in^T = L
            om_new = om.abs().clamp_min(1.0)
            ang = om * D
            if mode == "balanced":
                ang = torch.remainder(ang, 2 * math.pi)
            steps = ang / om_new                                                         # (A, C)
            W_out = torch.zeros(om.numel(), r, dtype=lat.dtype); W_out[:, :nA] = steps.T / c
            sd["action_to_lie.w_in.weight"] = W_in.to(signed.action_to_lie.w_in.weight.dtype)
    sd["action_to_lie.w_out.weight"] = W_out.to(signed.action_to_lie.w_out.weight.dtype)
    sd["path_integrator.omega"] = om_new.reshape(signed.path_integrator.omega.shape).to(signed.path_integrator.omega.dtype)
    m.load_state_dict(sd)
    with torch.no_grad():
        Da = m.step_table()[m.act].reshape(nA, -1).double()
        om2 = m.path_integrator.omega.detach().reshape(-1).double()
        err = float(torch.angle(torch.exp(1j * (om2 * Da - om * D))).abs().max())
    rep = {"mode": mode, "angle_err_max": err, "step_min": float(Da.min()),
           "w_out_absmax_signed": float(signed.action_to_lie.w_out.weight.detach().abs().max()),
           "w_out_absmax_built": float(m.action_to_lie.w_out.weight.detach().abs().max()),
           "w_in_absmax_signed": float(signed.action_to_lie.w_in.weight.detach().abs().max()),
           "w_in_absmax_built": float(m.action_to_lie.w_in.weight.detach().abs().max()),
           "step_absmax_signed": float(D.abs().max()), "step_absmax_built": float(Da.abs().max()),
           "omega_absmax_signed": float(om.abs().max()), "omega_absmax_built": float(om2.abs().max()),
           "n_omega_negative": int((om < 0).sum()),
           "latent_cos_signed": float(torch.nn.functional.cosine_similarity(lat[0], lat[1], dim=0)),
           "latent_cond_signed": float(torch.linalg.cond(lat))}
    return m, rep

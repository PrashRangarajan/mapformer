"""Per-head geometry of the content-to-angle map, for the rank theory (docs/paper/expansion/RANK_THEORY.md).

ANALYSIS, CPU. For a 1-layer MapWM-family model on a D-torus / ring, per head h:
  A[v]      = omega_h * step_h(v)                       omega-scaled phase step of token v (nb channels)
  U_d       = (A[minus_d] - A[plus_d]) / 2               axis half-steps (sign irrelevant)
  c         = mean_actions A + p_empty A[blank] + (1 - p_empty) mean_objects A   per-move common step ("clock")
  w         = mean_actions |q_pair| * mean_obs |k_pair|  channel weight (exact for layer 1; basins.py / head_stats)
Readouts (channel space, sw = sqrt(w / sum w)):
  kappa     = |wrap(c) sw| / mean_d |U_d sw|             visible drift per move, in units of one move (SCORE_RANK def.)
  indep     = s_min / s_max of (U sw)                    frame conditioning (0 = collinear axes)
  beta      = argmin |sw (wrap(c) - U^T beta)|           phantom displacement per move (cells)
  offmap    = |sw (wrap(c) - U^T beta)| / |sw wrap(c)|   part of the visible clock that is NOT a phantom displacement
                                                         (0 exactly at per-head rank D, up to wrapping)
  r_beta    = (bhat^T G bhat) / (tr G / D), G = (U sw)(U sw)^T   resolution of the head along its drift direction,
                                                         relative to its mean resolution (1 = isotropic / unaligned)
Latent space (native coordinates of the bottleneck the optimiser sees; per head restricted to the read subspace):
  lam       = |c_lat| / mean_d |U_lat,d|                 latent clock, read or not
  lam_perp  = |c_lat - P_U c_lat| / mean_d |U_lat,d|     latent clock outside the map plane (0 at rank D)
For the rank-D+clock-slot model (R2Clock below) the slot is one more latent coordinate that every token steps by 1.
"""
import numpy as np
import torch
import torch.nn as nn

from mapformer.model import MapFormerWM


class MapFormerWM_r2Clock(MapFormerWM):
    """Shared rank 2 (= `Vanilla`, identical draws at the seed) plus a per-channel learned phase rate per token,
    initialised at 0: phase_i(t) = omega_i (<w_i, z_t> + t * clock_i). Equivalent to a third latent coordinate with
    a FIXED step of 1 per token whose readout column (clock) is learned: the map stays rank 2, but every channel gets a
    private, map-free way to cancel its own drift. At init the function equals Vanilla's exactly."""

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 2)
        self.clock = nn.Parameter(torch.zeros(n_heads, self.n_blocks))   # no RNG draw
        inner = self.action_to_lie
        outer = self

        class _WithClock(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x) + outer.clock[None, None]
        self.action_to_lie = _WithClock()


def _base_tables(atl, e, H, nb):
    V = e.shape[0]
    out = []
    if isinstance(atl.w_out, nn.ModuleList):                   # per-head bottleneck
        r = atl.r
        Zall = atl.w_in(e).detach().view(V, H, r)
        for h in range(H):
            out.append((Zall[:, h].numpy(), atl.w_out[h].weight.detach().numpy()))
        return out
    W = atl.w_out.weight.detach()
    if hasattr(atl, "mask"):
        W = W * atl.mask
    Z = atl.w_in(e).detach().numpy()
    for h in range(H):
        Fh = W[h * nb:(h + 1) * nb].numpy()
        # restrict to the head's read subspace (row space of Fh)
        u, s, vt = np.linalg.svd(Fh, full_matrices=False)
        keep = s > 1e-6 * max(s.max(), 1e-30)
        B = vt[keep].T                                          # (r_tot, r_h) orthonormal
        out.append((Z @ B, Fh @ B))
    return out


def latent_tables(m):
    """Per head: (Z (V, r_h) latent step of every token in the head's read subspace, F (nb, r_h) readout).
    Phase step of token v in channel i = omega_i * (F @ Z[v])_i. A clock slot (with_clock) is one more latent
    coordinate that every token steps by 1, read by the learned column `clock`."""
    e = m.token_emb.weight.detach()
    V = e.shape[0]
    atl = m.action_to_lie
    H, nb = m.n_heads, m.n_blocks
    if hasattr(atl, "inner"):
        base = _base_tables(atl.inner, e, H, nb)
        ck = m.clock.detach().numpy()
        return [(np.concatenate([Z, np.ones((V, 1))], 1), np.concatenate([F, ck[h][:, None]], 1))
                for h, (Z, F) in enumerate(base)]
    return _base_tables(atl, e, H, nb)


def with_clock(base_cls):
    """Subclass of base_cls with a per-channel learned phase rate per token (init 0, no RNG draw): the function at
    init equals base_cls's at the same seed."""
    class _C(base_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.clock = nn.Parameter(torch.zeros(self.n_heads, self.n_blocks))
            inner = self.action_to_lie
            outer = self

            class _WithClock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.inner = inner

                def forward(self, x):
                    return self.inner(x) + outer.clock[None, None]
            self.action_to_lie = _WithClock()
    _C.__name__ = base_cls.__name__ + "_Clock"
    return _C


@torch.no_grad()
def head_geometry(m, D=2, n_obs=16, p_empty=0.5, act_pairs=None, obs_offset=None, weighted=True):
    """Return a list (per head) of dicts with the readouts in the module docstring."""
    if act_pairs is None:
        act_pairs = [(2 * d, 2 * d + 1) for d in range(D)]
    nA = 2 * D
    if obs_offset is None:
        obs_offset = nA
    omega = m.path_integrator.omega.detach().numpy()            # (H, nb)
    V = m.token_emb.weight.shape[0]
    L = m.layers[0]; H = m.n_heads; dh = m.d_model // H
    e = m.token_emb.weight.detach()
    hN = L.norm1(e)
    Q = L.q_proj(hN).view(V, H, dh); K = L.k_proj(hN).view(V, H, dh)
    qa = torch.sqrt(Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2); ka = torch.sqrt(K[..., 0::2] ** 2 + K[..., 1::2] ** 2)
    acts = [a for p in act_pairs for a in p]
    obs = list(range(obs_offset, obs_offset + n_obs + 1))
    w = (qa[acts].mean(0) * ka[obs].mean(0)).numpy() if weighted else np.ones((H, m.n_blocks))
    res = []
    if hasattr(m, "step_table"):                                # ring R1DF / R1AO: phase space only
        St = m.step_table().detach().numpy()                    # (V, H, nb)
        tabs = [(St[:, h], None) for h in range(H)]
    else:
        tabs = latent_tables(m)
    for h, (Z, F) in enumerate(tabs):
        A = (Z @ F.T) * omega[h][None] if F is not None else Z * omega[h][None]   # (V, nb)
        U = np.stack([(A[a] - A[b]) / 2 for a, b in act_pairs])  # (D, nb)
        objs = A[obs_offset:obs_offset + n_obs]; blank = A[obs_offset + n_obs]
        c_raw = A[acts].mean(0) + p_empty * blank + (1 - p_empty) * objs.mean(0)
        c = np.angle(np.exp(1j * c_raw))
        sw = np.sqrt(w[h] / w[h].sum())
        Uw = U * sw; cw = c * sw
        nU = np.linalg.norm(Uw, axis=1).mean()
        sv = np.linalg.svd(Uw, compute_uv=False)
        beta, *_ = np.linalg.lstsq(Uw.T, cw, rcond=None)
        resid = cw - Uw.T @ beta
        G = Uw @ Uw.T
        bn = np.linalg.norm(beta)
        r_beta = float((beta @ G @ beta) / bn ** 2 / (np.trace(G) / D)) if bn > 0 else float("nan")
        # latent space
        if F is None:
            res.append(dict(kappa=float(np.linalg.norm(cw) / nU), indep=float(sv[-1] / sv[0]), beta=beta.tolist(),
                            beta_norm=float(bn), offmap=float(np.linalg.norm(resid) / max(np.linalg.norm(cw), 1e-12)),
                            r_beta=r_beta, lam=float("nan"), lam_perp=float("nan"), rank=1, nU=float(nU),
                            cheb_raw=float(np.linalg.norm(c_raw * sw) / nU), readgain=float("nan")))
            continue
        Ul = np.stack([(Z[a] - Z[b]) / 2 for a, b in act_pairs])   # (D, r_h)
        cl = Z[acts].mean(0) + p_empty * Z[obs_offset + n_obs] + (1 - p_empty) * Z[obs_offset:obs_offset + n_obs].mean(0)
        nUl = np.linalg.norm(Ul, axis=1).mean()
        if np.linalg.matrix_rank(Ul) > 0:
            coef, *_ = np.linalg.lstsq(Ul.T, cl, rcond=None)
            cperp = cl - Ul.T @ coef
        else:
            cperp = cl
        # wrap-aware per-channel drift: fraction of weight in channels whose |tau| < 0.05 rad
        res.append(dict(kappa=float(np.linalg.norm(cw) / nU), indep=float(sv[-1] / sv[0]),
                        beta=beta.tolist(), beta_norm=float(bn),
                        offmap=float(np.linalg.norm(resid) / max(np.linalg.norm(cw), 1e-12)),
                        r_beta=r_beta,
                        lam=float(np.linalg.norm(cl) / nUl), lam_perp=float(np.linalg.norm(cperp) / nUl),
                        rank=int(Z.shape[1]), nU=float(nU),
                        cheb_raw=float(np.linalg.norm(c_raw * sw) / nU),
                        readgain=float(np.linalg.norm(c_raw * sw) / nU / max(np.linalg.norm(cl) / nUl, 1e-12))))
    return res


def basin(heads, thr=0.01):
    ok = [h["indep"] for h in heads if h["kappa"] <= thr]
    if not ok:
        return "CLOCK"
    return "CLEAN" if max(ok) >= 0.2 else "COLLAPSE"

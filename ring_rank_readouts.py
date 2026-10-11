"""Readouts for RANK_TESTS_PREREG.md experiment 1 (1D ring). The T1 drift readout of `analyze_score_rank.head_stats`
specialised to D = 1: per head, the omega-scaled step of each token A[v] (radians per channel); the axis half-step
U = (A[+1] - A[-1]) / 2; the per-move common component mm = wrap(mean of the two action steps + the expected observation
step, p_empty * A[blank] + (1 - p_empty) * mean A[objects]); kappa = || mm * sw || / || U * sw ||, sw = sqrt of the
channel weights w_j = mean over actions |q pair j| * mean over observations |k pair j| (normalised), as head_stats.
kappa is the decoded false drift per move in units of one move; a head is CLEAN if kappa <= 0.01 (SCORE_RANK's
threshold), else CLOCK (in 1D there is one axis, so no COLLAPSE class)."""
import numpy as np
import torch


@torch.no_grad()
def step_angles(m):
    """(V, H, nb) omega-scaled step per token."""
    if hasattr(m, "step_table"):
        A = m.step_table()
    else:
        A = m.action_to_lie(m.token_emb.weight[None])[0]
    return (A * m.path_integrator.omega).cpu().numpy()


@torch.no_grad()
def drift_kappa(m, V, n_obs_types=16, p_empty=0.5, weighted=True, action_offset=0, obs_offset=2):
    A = step_angles(m)
    a_plus, a_minus = A[action_offset], A[action_offset + 1]
    objs = A[obs_offset:obs_offset + n_obs_types]; blank = A[obs_offset + n_obs_types]
    U = (a_plus - a_minus) / 2
    mm = np.angle(np.exp(1j * ((a_plus + a_minus) / 2 + p_empty * blank + (1 - p_empty) * objs.mean(0))))
    H = A.shape[1]
    if weighted:
        L = m.layers[0]; e = m.token_emb.weight; h = L.norm1(e); dh = m.d_model // H
        Q = L.q_proj(h).view(V, H, dh); K = L.k_proj(h).view(V, H, dh)
        qa = torch.sqrt(Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2); ka = torch.sqrt(K[..., 0::2] ** 2 + K[..., 1::2] ** 2)
        acts = slice(action_offset, action_offset + 2); obs = slice(obs_offset, obs_offset + n_obs_types + 1)
        w = (qa[acts].mean(0) * ka[obs].mean(0)).cpu().numpy()
    else:
        w = np.ones(U.shape)
    out = []
    for hh in range(H):
        sw = np.sqrt(w[hh] / w[hh].sum())
        nu = float(np.linalg.norm(U[hh] * sw))
        out.append(float(np.linalg.norm(mm[hh] * sw)) / nu if nu > 0 else float("inf"))
    return out


def head_class(kappas, thr=0.01):
    return "CLEAN" if min(kappas) <= thr else "CLOCK"


@torch.no_grad()
def object_spread(m, n_obs_types=16, action_offset=0, obs_offset=2):
    """Per head: the spread of object steps around their mean, || sd over objects || / || U || (unweighted), which kappa
    cannot see (audit M3: a zero-mean object leak reads kappa ~ 0)."""
    A = step_angles(m)
    U = (A[action_offset] - A[action_offset + 1]) / 2
    objs = A[obs_offset:obs_offset + n_obs_types]
    sd = objs.std(0)
    return [float(np.linalg.norm(sd[h]) / max(np.linalg.norm(U[h]), 1e-12)) for h in range(A.shape[1])]

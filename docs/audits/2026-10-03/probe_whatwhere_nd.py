"""What/where separability probe on the RANK_WRAP / RANK_ND per-head checkpoints (GridWorldND). CPU only.

Purpose: a free test of LIT_WHAT_WHERE.md's main alternative explanation ("the paper torus redraws the map
every sequence, so ANY model may learn separation because the data are factorised"). These checkpoints were
trained on ONE fixed map per seed (environment_nd.GridWorldND(seed=args.seed), train_variant.py:507):
  2H  runs/rank_wrap/N10/D2  Vanilla_r3ph  grid 10 (100 cells)  -- memorised its map (own 0.986, unseen 0.273)
  2L  runs/rank_wrap/N32/D2  Vanilla_r3ph  grid 32 (1024 cells) -- generalised (unseen 0.999)
  A2  runs/rank_nd/D2        Vanilla_r2ph  grid 32               -- per-head rank 2, partly failed (0.64-1.00)
  3H  runs/rank_wrap/N10/D3  Vanilla_r4ph  grid 10 (1000 cells)  -- 4/8 solved, unseen 0.60-1.00
  3L  runs/rank_wrap/N18/D3  Vanilla_r4ph  grid 18 (5832 cells)  -- 8/8, unseen 1.000
All 1-layer MapWM (model.py) with the per-head bottleneck (model_rank_perhead.py), T=1024, 900 ep cosine.

Readouts follow docs/WHAT_WHERE_ANALYSIS.md sec. 6 (probe docs/audits/2026-09-27/probe_whatwhere.py), with
these changes, all forced by the environment or by a check below:
 * Actions: GridWorldND D=2 is a=0 +x, 1 -x, 2 +y, 3 -y (2D actions in D dims); observations ids 2D..2D+16
   (16 objects + blank). Content grid = 2D action queries x 17 observation keys (68 pairs in 2D, 102 in 3D).
 * Displacement d = key cell - query cell, taken modulo N with representative per axis in a window:
     w10  : -4..5 per axis (10^D displacements; identical set for every N; the FULL torus for N=10)
     full : -(N/2-1)..N/2 per axis (the full torus for every N: N^D displacements)
 * Phase (PRIMARY, "fixed"): theta_key - theta_query = -(sum of the model's action steps along the minimal
   path from the key cell to the query cell). This is what the model's cumsum gives up to the steps of the
   observation tokens in between (reported as leak). The key observation's OWN step is NOT added: it sits in
   the cumsum of both the key and the query and cancels. The 2026-09-27 probe added it (and used the
   query->key path with forward steps, equal to ours when steps are antisymmetric); that definition is
   computed too ("orig") so the change can be seen.
 * Grid fidelity (new, rule 9): on 4 real T=1024 sequences of the model's own training map, every real
   layer-1 score (action query t, observation key s < t; ~2.1M pairs) is compared with the grid's prediction
   at its torus displacement (R^2). Split by pair type: exact (the walk's action counts equal the minimal
   path's), detour (same net displacement, back-and-forth), wrap (unwrapped displacement != minimal
   representative). Implementation check: the grid formula evaluated at the walk's ACTUAL action counts
   reproduces the real score recomputed with observation steps zeroed (float32 cumsum precision expected).
 * Real-pair decomposition (new; needed where the grid does not describe the model): the same real scores,
   restricted to torus displacements in w10, are decomposed by nested fits into content (68 pairs), position
   (100 displacements), additive, full (content x position cell means), interaction = full - additive,
   residual = 1 - full (score variance that content and torus displacement do not explain: path history).
   inter/pos(real) = interaction / (full - content). Weighted by how often each pair occurs (blank ~50%).
 * Attention on real sequences, per head: share of observation-key attention on same-cell keys (revisits),
   attention-weighted mean lag in steps.
 Readouts per head on the C x P score matrix: pos/content/interaction variance shares, inter/pos,
 shape1, peak0, flip; leak = mean |omega*step(obs)| / mean |omega*step(action)|; opposition
 |step(+e_i)+step(-e_i)| / |step(+e_i)|. Head = the head with the larger position-dependent sd ("where-head").
 Map entanglement (descriptive): I(o_query; o_key | d), averaged over d in w10 minus 0, exact over the cells
 of the training map (in bits; 0 for a map redrawn every sequence, in the limit).

Verification: the full forward rebuilt from the score function reproduces model(tokens) logits (max abs diff).
"""
import json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

torch.set_num_threads(8)
sys.path.insert(0, "/home/prashr")
from mapformer.train_variant import VARIANT_MAP
from mapformer.model import _apply_rope
from mapformer.environment_nd import GridWorldND
from mapformer.stats_core import perm2_p

REPO = "/home/prashr/mapformer"
OUT = os.path.dirname(os.path.abspath(__file__))
N_OBS = 17
T_PROBE = 1024

CELLS = {
    "2H": dict(dir="runs/rank_wrap/N10/D2", arm="Vanilla_r3ph", D=2, N=10, src="wrap"),
    "2L": dict(dir="runs/rank_wrap/N32/D2", arm="Vanilla_r3ph", D=2, N=32, src="wrap"),
    "A2": dict(dir="runs/rank_nd/D2", arm="Vanilla_r2ph", D=2, N=32, src="nd"),
    "3H": dict(dir="runs/rank_wrap/N10/D3", arm="Vanilla_r4ph", D=3, N=10, src="wrap"),
    "3L": dict(dir="runs/rank_wrap/N18/D3", arm="Vanilla_r4ph", D=3, N=18, src="wrap"),
}


def accuracies(cell, c, s):
    if c["src"] == "wrap":
        J = json.load(open(f"{REPO}/RANK_WRAP.json"))
        sec = J["secondary"][cell][s]; assert sec["seed"] == s
        return dict(held=J["acc"][cell][s], own=sec["own"], held_wrap=sec["wrap"], held_other=sec["other"])
    J = json.load(open(f"{REPO}/RANK_ND.json")); S = json.load(open(f"{REPO}/RANK_ND_SECONDARY.json"))
    sec = S[cell][s]; assert sec["seed"] == s
    return dict(held=J["D2"]["acc"]["Vanilla_r2ph"]["1024"][str(s)], own=sec["own_all"],
                held_wrap=sec["held_wrap"], held_other=sec["held_other"])


def build(arm, N, D, seed=None, path=None):
    if path is None:
        torch.manual_seed(seed)
        m = VARIANT_MAP[arm](vocab_size=2 * D + N_OBS, d_model=128, n_heads=2, n_layers=1, grid_size=N)
        return m.eval(), None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    assert ck["variant"] == arm and c["grid_size"] == N and c["n_dims"] == D and c["n_layers"] == 1, c
    m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                         n_layers=c["n_layers"], grid_size=c["grid_size"])
    m.load_state_dict(ck["model_state_dict"])
    return m.eval(), ck


def layer_scores(m, h, cos, sin):
    """Pre-softmax layer-1 scores (B,H,L,L), mirroring WMTransformerLayer.forward."""
    layer = m.layers[0]; B, L, _ = h.shape; H, dh = layer.n_heads, layer.d_head
    Q = layer.q_proj(h).view(B, L, H, dh).transpose(1, 2)
    K = layer.k_proj(h).view(B, L, H, dh).transpose(1, 2)
    Q, K = _apply_rope(Q, cos, sin), _apply_rope(K, cos, sin)
    return Q @ K.transpose(-1, -2) / math.sqrt(dh)


@torch.no_grad()
def verify(m, tokens):
    layer = m.layers[0]
    x = m.token_emb(tokens); B, L, D = x.shape
    c, s = m.path_integrator(m.action_to_lie(x))
    h = layer.norm1(x)
    S = layer_scores(m, h, c, s)
    S = S.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
    V = layer.v_proj(h).view(B, L, layer.n_heads, layer.d_head).transpose(1, 2)
    out = (F.softmax(S, -1) @ V).transpose(1, 2).reshape(B, L, D)
    x2 = x + layer.o_proj(out); x2 = x2 + layer.ffn(layer.norm2(x2))
    mine = m.out_proj(m.out_norm(x2))
    ref = m(tokens)
    # negative control: drop the rotation (cos=1, sin=0); the check must be able to fail
    S0 = layer_scores(m, h, torch.ones_like(c), torch.zeros_like(s))
    S0 = S0.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
    out0 = (F.softmax(S0, -1) @ V).transpose(1, 2).reshape(B, L, D)
    y = x + layer.o_proj(out0); y = y + layer.ffn(layer.norm2(y))
    return (mine - ref).abs().max().item(), (m.out_proj(m.out_norm(y)) - ref).abs().max().item()


@torch.no_grad()
def content_tensors(m, D):
    """A[a,o,h,b] = q1k1+q2k2, Bm = q2k1-q1k2 (pairs of the interleaved rotary layout), / sqrt(dh).
    Score(a, o, phi) = sum_b A cos(phi_b) + Bm sin(phi_b), phi = theta_key - theta_query."""
    layer = m.layers[0]; H, dh = layer.n_heads, layer.d_head; nA = 2 * D
    E = m.token_emb.weight
    Q = layer.q_proj(layer.norm1(E[:nA])).view(nA, H, dh)
    K = layer.k_proj(layer.norm1(E[nA:nA + N_OBS])).view(N_OBS, H, dh)
    q1, q2, k1, k2 = Q[..., 0::2], Q[..., 1::2], K[..., 0::2], K[..., 1::2]
    A = (q1[:, None] * k1[None] + q2[:, None] * k2[None]) / math.sqrt(dh)
    Bm = (q2[:, None] * k1[None] - q1[:, None] * k2[None]) / math.sqrt(dh)
    return A.double(), Bm.double()                            # (nA, O, H, nb)


@torch.no_grad()
def steps(m, D):
    E = m.token_emb.weight
    st = (m.action_to_lie(E[None])[0] * m.path_integrator.omega).double()   # (V, H, nb) omega-scaled
    return st[:2 * D], st[2 * D:2 * D + N_OBS]


def window(N, D, kind):
    lo, hi = (-4, 5) if kind == "w10" else (-(N // 2 - 1), N // 2)
    ax = np.arange(lo, hi + 1)
    g = np.stack(np.meshgrid(*([ax] * D), indexing="ij"), -1).reshape(-1, D)
    return g, lo


def counts_minimal(v, D):
    """Action counts (P, 2D) of the minimal path realising displacement v (P, D)."""
    C = np.zeros((len(v), 2 * D))
    for i in range(D):
        C[:, 2 * i] = np.clip(v[:, i], 0, None); C[:, 2 * i + 1] = np.clip(-v[:, i], 0, None)
    return C


def grid_scores(A, Bm, st_a, st_o, disp, mode):
    """S (H, C, P). mode 'fixed': phi = -(steps key->query). 'orig': phi = steps query->key + step(o)."""
    nA, O, H, nb = A.shape
    if mode == "fixed":
        phi = -torch.einsum("pa,ahb->phb", torch.from_numpy(counts_minimal(-disp, st_a.shape[0] // 2)), st_a)
        phi = phi[:, None].expand(-1, O, -1, -1)               # (P, O, H, nb)
    else:
        phi = torch.einsum("pa,ahb->phb", torch.from_numpy(counts_minimal(disp, st_a.shape[0] // 2)), st_a)
        phi = phi[:, None] + st_o[None]
    S = torch.einsum("aohb,pohb->haop", A, phi.cos()) + torch.einsum("aohb,pohb->haop", Bm, phi.sin())
    return S.reshape(H, nA * O, -1)


def readouts(S, i0):
    res = []
    for h in range(S.shape[0]):
        X = S[h].numpy()
        g = X.mean(); rm = X.mean(1, keepdims=True); cm = X.mean(0, keepdims=True)
        tot = ((X - g) ** 2).mean()
        cont = ((rm - g) ** 2).mean(); pos = ((cm - g) ** 2).mean(); inter = ((X - rm - cm + g) ** 2).mean()
        Xc = X - rm
        u, sig, vt = np.linalg.svd(Xc, full_matrices=False)
        v = vt[0] * (1 if vt[0][i0] >= vt[0].mean() else -1)
        res.append({"cont_share": cont / tot, "pos_share": pos / tot, "inter_share": inter / tot,
                    "inter_of_pos": inter / (pos + inter), "shape1": float(sig[0] ** 2 / (sig ** 2).sum()),
                    "peak0": float((X.argmax(1) == i0).mean()), "flip": float(((Xc @ v) < 0).mean()),
                    "pos_sd": float(np.sqrt(pos + inter))})
    return res


def walk(env, T, seed):
    """Real trajectory plus UNWRAPPED cell coordinates of every step."""
    np.random.seed(seed)
    tok, _, _ = env.generate_trajectory(T)
    D = env.dims
    acts = tok[0::2].numpy()
    unw = np.cumsum(env.action_deltas[acts], 0)                 # unwrapped offset from the start cell
    return tok, acts, unw


@torch.no_grad()
def seq_pairs(m, env, D, N, A, Bm, st_a, lo_full, seed):
    """All (action query, earlier observation key) pairs of one real T_PROBE sequence of the training map."""
    tok, acts, unw = walk(env, T_PROBE, seed=seed)
    L = tok.numel(); nA = 2 * D
    x = m.token_emb(tok[None])
    c, s = m.path_integrator(m.action_to_lie(x))
    h = m.layers[0].norm1(x)
    Sreal = layer_scores(m, h, c, s)[0].double()               # (H, L, L)
    Pattn = F.softmax(Sreal.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf")), -1)
    T = len(acts)
    tq, sk = np.tril_indices(T, -1)                           # query step tq (action token 2tq), key step sk < tq
    qi, ki = 2 * tq, 2 * sk + 1                                # the obs of step sk is token 2sk+1
    a_t = acts[tq]; o_s = tok[ki].numpy() - nA
    oh = np.zeros((T, nA)); oh[np.arange(T), acts] = 1
    cum = np.cumsum(oh, 0)
    cnt = cum[tq] - cum[sk]                                    # actions taken from the key cell to the query cell
    u = unw[sk] - unw[tq]                                      # unwrapped key - query displacement
    rep = ((u - lo_full) % N) + lo_full                        # torus displacement, full-window representative
    wrap = (u != rep).any(1)
    minimal = (cnt == counts_minimal(-rep, D)).all(1)
    cls = np.where(wrap, 2, np.where(minimal, 0, 1))           # 0 exact path, 1 detour, 2 wrap
    idx = np.zeros(len(rep), dtype=np.int64)
    for i in range(D):
        idx = idx * N + (rep[:, i] - lo_full)
    # implementation check: the grid formula at the walk's ACTUAL action counts equals the real score
    # recomputed with observation steps zeroed (both are the same function of the same phase)
    dlt = m.action_to_lie(x)[0].clone(); dlt[1::2] = 0
    ca, sa = m.path_integrator(dlt[None])
    sub = np.random.RandomState(0).choice(len(tq), min(20000, len(tq)), replace=False)
    Sact = layer_scores(m, h, ca, sa)[0].double()
    phi = -torch.einsum("na,ahb->nhb", torch.from_numpy(cnt[sub]), st_a)
    pred = (A[a_t[sub], o_s[sub]] * phi.cos() + Bm[a_t[sub], o_s[sub]] * phi.sin()).sum(-1)
    real_act = Sact[:, qi[sub], ki[sub]].T
    impl = float((pred - real_act).abs().max()); impl_rel = impl / float(real_act.std())
    # attention readouts per head, per query: mass on observation keys, on same-cell observation keys,
    # and the attention-weighted lag (steps) over observation keys
    att = Pattn[:, qi, ki].numpy()                             # (H, n)
    same = (rep == 0).all(1)
    lag = (tq - sk).astype(float)
    attn = []
    for hh in range(att.shape[0]):
        w = att[hh]
        m_obs = np.bincount(tq, w, T); m_same = np.bincount(tq, w * same, T); wl = np.bincount(tq, w * lag, T)
        has = np.bincount(tq, same, T) > 0
        attn.append({"mass_obs": float(m_obs[1:].mean()),
                     "samecell_frac_of_obs": float((m_same[has] / np.maximum(m_obs[has], 1e-30)).mean()),
                     "mean_lag": float((wl[1:] / np.maximum(m_obs[1:], 1e-30)).mean())})
    return dict(real=Sreal[:, qi, ki].numpy(), att=att, c=a_t * N_OBS + o_s, idx=idx, rep=rep, cls=cls,
                impl=impl, impl_rel=impl_rel, attn=attn)


def r2(y, yh, mask=None):
    if mask is not None:
        if mask.sum() < 10: return None
        y, yh = y[mask], yh[mask]
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def real_decomp(y, c, p, nC, nP):
    """Variance decomposition of REAL scores by content pair c and torus displacement p (unbalanced, so by
    nested fits): additive (c + p, backfitted), full (c x p cell means), residual = what neither explains
    (path history: intermediate observations' steps, wraps, detours)."""
    g = y.mean(); sst = ((y - g) ** 2).sum()
    def means(k, n, v):
        s = np.bincount(k, v, n); w = np.bincount(k, None, n); return s / np.maximum(w, 1)
    cell = c * nP + p
    full = means(cell, nC * nP, y)[cell]
    mc = means(c, nC, y)[c]; mp = means(p, nP, y)[p]
    a = np.zeros(nC); b = np.zeros(nP)
    for _ in range(100):
        a = means(c, nC, y - g - b[p]); b = means(p, nP, y - g - a[c])
    add = g + a[c] + b[p]
    R = lambda f: 1 - ((y - f) ** 2).sum() / sst
    Rf, Ra, Rc, Rp = R(full), R(add), R(mc), R(mp)
    counts = np.bincount(cell, None, nC * nP); counts = counts[counts > 0]
    return {"R2_content": Rc, "R2_position": Rp, "R2_additive": Ra, "R2_full": Rf,
            "interaction": Rf - Ra, "residual": 1 - Rf,
            "inter_of_pos": (Rf - Ra) / max(Rf - Rc, 1e-12),
            "cells": int(len(counts)), "cell_n_median": float(np.median(counts))}


@torch.no_grad()
def fidelity(m, env, D, N, A, Bm, st_a, Sfull, lo_full, n_seq=4):
    """Real layer-1 scores of n_seq sequences on the training map vs the grid; real-pair decomposition."""
    P = [seq_pairs(m, env, D, N, A, Bm, st_a, lo_full, seed=1 + i) for i in range(n_seq)]
    cat = lambda k: np.concatenate([p[k] for p in P], -1)
    real, att, c, idx, cls = cat("real"), cat("att"), cat("c"), cat("idx"), cat("cls")
    rep = np.concatenate([p["rep"] for p in P], 0)
    nA = 2 * D
    out = {"n_pairs": int(len(c)), "frac_exact": float((cls == 0).mean()), "frac_detour": float((cls == 1).mean()),
           "frac_wrap": float((cls == 2).mean()), "impl_check_maxabs": max(p["impl"] for p in P),
           "impl_check_rel": max(p["impl_rel"] for p in P), "heads": []}
    # w10 displacement set as indices 0..10^D-1, for the real-pair decomposition
    in10 = ((rep >= -4) & (rep <= 5)).all(1)
    p10 = np.zeros(len(rep), dtype=np.int64)
    for i in range(D):
        p10 = p10 * 10 + (rep[:, i] + 4)
    for hh in range(real.shape[0]):
        y = real[hh]
        pred = Sfull[hh].numpy().reshape(nA * N_OBS, -1)[c, idx]
        w = att[hh]
        r = {"R2_grid_all": r2(y, pred)}
        for k, nm in enumerate(("exact", "detour", "wrap")):
            r[f"R2_grid_{nm}"] = r2(y, pred, cls == k)
            r[f"attn_share_{nm}"] = float(w[cls == k].sum() / w.sum())
        r["attn_rmse_over_sd"] = float(np.sqrt((w * (y - pred) ** 2).sum() / w.sum()) / y.std())
        r["attn"] = {k: float(np.mean([p["attn"][hh][k] for p in P])) for k in P[0]["attn"][hh]}
        r["real_w10"] = real_decomp(y[in10], c[in10], p10[in10], nA * N_OBS, 10 ** D)
        out["heads"].append(r)
    return out


def map_entanglement(env, D, N):
    """Mean over d in w10\\{0} of I(o_at_cell ; o_at_cell+d) in bits, exact over the training map's cells."""
    M = env.obs_map.numpy()
    disp, _ = window(N, D, "w10")
    mis = []
    for d in disp:
        if not d.any(): continue
        Md = np.roll(M, shift=tuple(-d), axis=tuple(range(D)))
        J = np.zeros((N_OBS, N_OBS)); np.add.at(J, (M.ravel(), Md.ravel()), 1); J /= J.sum()
        px, py = J.sum(1), J.sum(0); nz = J > 0
        mis.append(float((J[nz] * np.log2(J[nz] / np.outer(px, py)[nz])).sum()))
    return float(np.mean(mis))


def probe(m, D, N, env):
    A, Bm = content_tensors(m, D)
    st_a, st_o = steps(m, D)
    r = {}
    for wk in ("w10", "full"):
        disp, lo = window(N, D, wk)
        i0 = int(np.where((disp == 0).all(1))[0][0])
        for mode in ("fixed", "orig"):
            if wk == "full" and mode == "orig": continue
            S = grid_scores(A, Bm, st_a, st_o, disp, mode)
            r[f"{wk}_{mode}"] = readouts(S, i0)
            if wk == "full":
                Sfull, lo_full, disp_full = S, lo, disp
    nrm = lambda v: v.norm(dim=-1)
    r["leak"] = (nrm(st_o).mean(0) / nrm(st_a).mean(0)).tolist()
    r["opp"] = [(nrm(st_a[2 * i] + st_a[2 * i + 1]) / nrm(st_a[2 * i])).tolist() for i in range(D)]
    r["fidelity"] = fidelity(m, env, D, N, A, Bm, st_a, Sfull, lo_full)
    return r


def where_head(heads):
    return max(range(len(heads)), key=lambda h: heads[h]["pos_sd"])


def main():
    allres = {}
    for cell, c in CELLS.items():
        D, N = c["D"], c["N"]
        allres[cell] = {"cfg": c, "runs": [], "untrained": []}
        for s in range(8):
            p = f"{REPO}/{c['dir']}/{c['arm']}_s{s}/{c['arm']}.pt"
            m, ck = build(c["arm"], N, D, path=p)
            env = GridWorldND(dims=D, size=N, n_obs_types=16, p_empty=0.5, seed=s)   # the training map
            np.random.seed(2)
            toks = torch.stack([env.generate_trajectory(T_PROBE)[0] for _ in range(2)])
            err, err_norot = verify(m, toks)
            r = probe(m, D, N, env)
            r.update(seed=s, path=os.path.relpath(p, REPO), verify_maxabs=err, verify_norot=err_norot,
                     final_loss=float(ck["losses"][-1]), map_MI_bits=map_entanglement(env, D, N),
                     **accuracies(cell, c, s))
            allres[cell]["runs"].append(r)
            print(f"{cell} s{s} verify {err:.1e} (no-rot {err_norot:.2f}) impl {r['fidelity']['impl_check_maxabs']:.1e} "
                  f"held {r['held']:.3f} own {r['own']:.3f}", flush=True)
        for s in range(3):
            m, _ = build(c["arm"], N, D, seed=s)
            env = GridWorldND(dims=D, size=N, n_obs_types=16, p_empty=0.5, seed=s)
            np.random.seed(2)
            toks = torch.stack([env.generate_trajectory(T_PROBE)[0] for _ in range(2)])
            r = probe(m, D, N, env); r["verify_maxabs"], r["verify_norot"] = verify(m, toks)
            allres[cell]["untrained"].append(r)
    held_out_MI = {f"N{N}D{D}": map_entanglement(GridWorldND(dims=D, size=N, n_obs_types=16, p_empty=0.5,
                                                             seed=10000), D, N)
                   for N, D in ((10, 2), (32, 2), (10, 3), (18, 3))}
    allres["_heldout_map_MI_bits"] = held_out_MI
    json.dump(allres, open(f"{OUT}/probe_whatwhere_nd.json", "w"), indent=1)


if __name__ == "__main__":
    main()

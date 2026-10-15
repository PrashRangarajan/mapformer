"""Probes for CoPE on the ring (COPE_DIAGNOSIS.md). Eval-only; no training.

Token ids (GridWorldCancel, dims=1): 0 = +1 move, 1 = -1 move, 2..17 = observation types, 18 = blank.
Revisit targets are predicted at ACTION-token positions (query = the move just made).
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/prashr")
from mapformer.environment_cancel import GridWorldCancel
from mapformer.train_variant import VARIANT_MAP
import mapformer.model_mono_baselines as mmb

REPO = Path("/home/prashr/mapformer")
OBS = list(range(2, 19))          # 16 types + blank
EVAL_SEED = 10 ** 6


def run_dirs():
    out = []
    for arm in ("CoPEF", "CoPEF_Hi", "CoPEF_Lo", "CoPEF_LoS"):
        for s in range(900, 908):
            out.append(("WRAP_COPE", arm, s, REPO / f"runs/wrap_cope/p0/{arm}_s{s}" / f"{arm}.pt"))
    for s in range(700, 708):
        out.append(("GENERALITY", "CoPEF", s, REPO / f"runs/generality/ring/CoPEF_s{s}" / "CoPEF.pt"))
    return out


def load(path, dev="cpu"):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = VARIANT_MAP[ck["variant"]](vocab_size=ck["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32)
    m.load_state_dict(ck["model_state_dict"]); m.to(dev).eval()
    return m, ck


@torch.no_grad()
def tables(m):
    """Gate logits L[h, a, kappa] (= content logits), gates G, effective position tables Z[h, a, p] = q_{a,h}.e(p)."""
    blk = m.layers[0]; at = blk.attn
    E = m.token_emb.weight                                   # (V, d)
    qkv = at.qkv(blk.norm1(E)).view(E.shape[0], 3, at.h, at.dh)
    q, k = qkv[:, 0], qkv[:, 1]                               # (V, H, dh)
    L = torch.einsum("ahd,khd->hak", q, k) / math.sqrt(at.dh)  # (H, V, V): query type a, key type kappa
    pe = at.cope.pos_emb[0]                                   # (dh, npos)
    Z = torch.einsum("ahd,dp->hap", q, pe)                    # (H, V, npos)
    return L.numpy(), torch.sigmoid(L).numpy(), Z.numpy()


def obs_freq():
    f = np.zeros(19); f[2:18] = 0.5 / 16; f[18] = 0.5           # p_empty 0.5, 16 types uniform
    return f


def increments(G):
    """Per head h and query type a in {0,1}: alpha, beta, sigma, delta, obs-gate mean and sd (frequency weighted)."""
    f = obs_freq()[OBS]; f = f / f.sum()
    out = {}
    for h in range(G.shape[0]):
        for a in (0, 1):
            go = G[h, a, OBS]
            cbar = float((f * go).sum()); tau = float(np.sqrt((f * (go - cbar) ** 2).sum()))
            al = float(G[h, a, 0] + cbar); be = float(G[h, a, 1] + cbar)
            kap = 8 * (al - be)                                       # lattice (P = 2): odd integer
            out[(h, a)] = dict(alpha=al, beta=be, sigma=(al + be) / 2, delta=(al - be) / 2, kappa=kap,
                               kappa_dist_odd=float(abs(kap - (2 * np.floor(kap / 2) + 1))),
                               cbar=cbar, tau=tau, g_plus=float(G[h, a, 0]), g_minus=float(G[h, a, 1]),
                               g_obs=go.tolist())
    return out


def period2(Zrow, lo=0, hi=255):
    """Share of the (mean-removed) table power at the Nyquist (period-2) frequency, and its amplitude.
    Sign: + means even knots high."""
    z = Zrow[lo:hi].astype(np.float64); z = z - z.mean()
    alt = np.where(np.arange(lo, hi) % 2 == 0, 1.0, -1.0)
    a2 = float((z * alt).mean())                                       # amplitude of the period-2 component
    share = float(len(z) * a2 ** 2 / max((z ** 2).sum(), 1e-30))
    return share, a2


def trajectories(n_trials, T=128, seed=EVAL_SEED):
    env = GridWorldCancel(size=32, seed=10000, p_plus=0.5)
    np.random.seed(seed)
    out = []
    for _ in range(n_trials):
        tok, _om, rev = env.generate_trajectory(T)
        out.append((tok, rev))
    return out


def strata(tok):
    """For each step t (query position 2t in the input, target obs at 2t+1): revisit info.
    Returns list of dicts for revisit steps: t, N_local (moves since latest D = 0 visit, or None), N_any (moves since
    latest visit of the cell), wrap_only (no earlier visit at unwrapped D = 0)."""
    acts = tok[0::2].numpy(); d = np.where(acts == 0, 1, -1)
    U = np.cumsum(d)                                           # unwrapped position after move t (up to start)
    res = []
    for t in range(len(U)):
        same = [s for s in range(t) if (U[t] - U[s]) % 32 == 0]
        if not same:
            continue
        loc = [s for s in same if U[t] == U[s]]
        res.append(dict(t=t, N_local=(t - max(loc)) if loc else None, N_any=t - max(same), wrap_only=not loc,
                        D_latest=int(U[t] - U[max(same)])))
    return res


class SnapCtx:
    """Eval-only intervention on CoPE: replace gates by a per-(head, query type, key type) table, keeping
    the attention content logits (and everything else) unchanged."""
    tokens = None
    Gsnap = None                                                # (H, V, V) torch

    @staticmethod
    def forward(self, query, attn_logits):
        if SnapCtx.Gsnap is None:
            gates = torch.sigmoid(attn_logits)
        else:
            x = SnapCtx.tokens                                     # (B, L)
            Gs = SnapCtx.Gsnap                                     # (H, V, V)
            gates = Gs[:, x[0][:, None], x[0][None, :]][None]       # (1, H, L, L)
            gates = gates * torch.ones_like(attn_logits).tril()
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        pos_ceil = pos.ceil().long(); pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil); logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)


_orig_forward = mmb.CoPE.forward


def install_snap():
    mmb.CoPE.forward = SnapCtx.forward


@torch.no_grad()
def score(m, trajs, Gsnap=None):
    """Revisit accuracy (argmax of the model's logits, as train_cancel.evaluate) with per-revisit strata rows."""
    install_snap()
    SnapCtx.Gsnap = None if Gsnap is None else torch.as_tensor(Gsnap, dtype=torch.float32)
    rows = []; ok = tot = 0; nll = 0.0
    for tok, rev in trajs:
        x = tok[:-1].unsqueeze(0); SnapCtx.tokens = x
        lp = F.log_softmax(m(x).float(), -1)[0]
        tgt = tok[1:]; msk = rev[1:]
        st = {r["t"]: r for r in strata(tok)}
        for i in torch.nonzero(msk).flatten().tolist():
            t = i // 2; r = st[t]
            c = int(lp[i].argmax() == tgt[i]); ok += c; tot += 1; li = -float(lp[i, tgt[i]]); nll += li
            rows.append(dict(r, ok=c, nll=li, a=int(x[0, i])))
    SnapCtx.Gsnap = None
    return ok / tot, nll / tot, rows


def strat_table(rows):
    def acc(sel):
        sel = list(sel); return (float(np.mean([r["ok"] for r in sel])), len(sel)) if sel else (float("nan"), 0)
    out = {"all": acc(rows), "local": acc(r for r in rows if not r["wrap_only"]),
           "wrap_only": acc(r for r in rows if r["wrap_only"])}
    for lo, hi in ((1, 8), (8, 32), (32, 64), (64, 129)):
        out[f"local N[{lo},{hi})"] = acc(r for r in rows if not r["wrap_only"] and lo <= r["N_local"] < hi)
    return out

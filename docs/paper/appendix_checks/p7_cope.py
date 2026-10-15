"""P7 (CoPE on the ring: the lattice family). Counts computed by the paper's CoPE module (model_mono_baselines.CoPE, Listing 1)
from constructed gate tables G[query type, key type] (attention logits = logit(G), causal mask), position logit = a periodic
piecewise-linear table z (P = 2: +1 at even knots, -1 at odd; P = 3: +1 at 0 mod 3, -0.5 otherwise) read with a unit query.
Readout per revisit query (move token 2t): READ if every same-cell observation key has z = max over keys and every other key
is lower. Ring 32, T 128, registered walk; the same gates on a non-wrapping line (same walk process, no modulus).
"""
import math
import numpy as np
import torch
from common import ring_batch
from mapformer.model_mono_baselines import CoPE

out = []
P = lambda s: (print(s), out.append(s))
logit = lambda p: math.log(p / (1 - p))
B = 120
toks, pos_ring, mv, env = ring_batch(B, seed=77)
pos_line = np.cumsum(mv, 1)                                                     # same moves, no modulus
L = toks.shape[1]
ismove = (toks < 2)


def counts_via_module(gp, gm, c, Pp, tbl):
    """gp, gm: gates of +1 / -1 keys (same for both query types); c: observation gate. Returns z(p) (B, L, L) from CoPE."""
    G = torch.where(toks[:, None, :] == 0, gp, torch.where(toks[:, None, :] == 1, gm, c)).double().expand(B, L, L).clone()
    lg = torch.log(G / (1 - G))
    lg = lg.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool))[None], float("-inf"))
    cope = CoPE(2048, 2).double()
    with torch.no_grad():
        k = torch.arange(2048)
        cope.pos_emb[0, 0] = torch.tensor([tbl(int(i)) for i in k], dtype=torch.float64); cope.pos_emb[0, 1] = 0
        q = torch.zeros(B, L, 2, dtype=torch.float64); q[..., 0] = 1
        zz = cope(q, lg)
        gates = torch.sigmoid(lg); pcount = gates.flip(-1).cumsum(-1).flip(-1)
    return zz, pcount


def read_share(zz, pos):
    ok = tot = 0; wrap_ok = wrap_tot = 0
    for b in range(B):
        for t in range(1, 128):
            keys = np.arange(t); same = pos[b, keys] == pos[b, t]
            if not same.any():
                continue
            z = zz[b, 2 * t, 2 * keys + 1].numpy()
            good = (z[same].min() > z[~same].max() + 1e-9) if (~same).any() else True
            tot += 1; ok += good
            # wrap-only: every earlier visit at unwrapped |D| >= 32
            Dun = np.abs(np.cumsum(mv[b])[t] - np.cumsum(mv[b])[keys[same]])
            if (Dun >= 32).all():
                wrap_tot += 1; wrap_ok += good
    return ok / tot, wrap_ok / max(wrap_tot, 1), tot, wrap_tot


tri2 = lambda i: 1.0 if i % 2 == 0 else -1.0
tri3 = lambda i: 1.0 if i % 3 == 0 else -0.5
# sanity: module count == formula p = n+ alpha + n- beta (alpha = g+ + c, beta = g- + c), key 2s+1 to query 2t
gp, gm, c = 17 / 16 - 0.5, 15 / 16 - 0.5, 0.5
zz, pc = counts_via_module(gp, gm, c, 2, tri2)
cp = np.concatenate([np.zeros((B, 1)), np.cumsum(mv == 1, 1)], 1); cm = np.concatenate([np.zeros((B, 1)), np.cumsum(mv == -1, 1)], 1)
err = 0.0
for b in range(0, B, 7):
    for t in range(1, 128):
        s = np.arange(t)
        f = (cp[b, t + 1] - cp[b, s + 1]) * (gp + c) + (cm[b, t + 1] - cm[b, s + 1]) * (gm + c)
        err = max(err, float(np.abs(pc[b, 2 * t, 2 * s + 1].numpy() - f).max()))
P(f"count convention: max |CoPE count(key o_s -> query a_t) - (n+ alpha + n- beta)| = {err:.1e} (one move gate + one observation gate per step)")

cases = [("P=2 k=17 (lattice)", 2, 17, 0.5, tri2), ("P=2 k=15 (lattice)", 2, 15, 0.5, tri2), ("P=2 k=13 (lattice)", 2, 13, 0.5, tri2),
         ("P=3 k=11 (lattice)", 3, 11, 0.98, tri3), ("P=3 k=21 (lattice)", 3, 21, 0.98, tri3),
         ("P=2 k=18 (even: aliased)", 2, 18, 0.5, tri2), ("P=2 k=16 (even: kappa 0, parity only)", 2, 16, 0.5, tri2),
         ("P=2 k=17.3 (off lattice)", 2, 17.3, 0.5, tri2)]
Dmax = int(max(np.abs(np.cumsum(mv[b])[t] - np.concatenate([[0], np.cumsum(mv[b])])[:t + 1]).max() for b in range(B) for t in range(128)))
Nmax = 128
P(f"largest |D| between a key and a query in this sample: {Dmax}; sufficient precision (every count within P/64 of its lattice value): "
  f"|dk| < 1/(2 Dmax) = {1 / (2 * Dmax):.5f}; |d sigma| < P/(64 Nmax) = {2 / (64 * Nmax):.6f} (P = 2)")
cases += [(f"P=2 k=17 + 0.95/(2 Dmax)", 2, 17 + 0.95 / (2 * Dmax), 0.5, tri2), (f"P=2 k=17 + 1/64", 2, 17 + 1 / 64, 0.5, tri2),
          (f"P=2 k=17 + 1/16", 2, 17 + 1 / 16, 0.5, tri2)]
SIG = {"sig_ok": 0.95 * 2 / (64 * Nmax), "sig_bad": 4 * 2 / (64 * Nmax)}
cases += [("P=2 k=17, sigma + 0.95 P/(64 Nmax)", 2, 17, ("sig_ok",), tri2), ("P=2 k=17, sigma + 4 P/(64 Nmax)", 2, 17, ("sig_bad",), tri2)]
for name, Pp, k, c, tbl in cases:
    alpha = k * Pp / 32; beta = Pp - alpha
    if isinstance(c, tuple):            # perturb sigma = (alpha + beta) / 2 by d, keeping alpha - beta: each move gate + d
        d = SIG[c[0]]; c = 0.5; gp_, gm_ = alpha - c + d, beta - c + d
    else:
        gp_, gm_ = alpha - c, beta - c
    if not (0 < gp_ < 1 and 0 < gm_ < 1):
        P(f"{name}: gates out of (0, 1)"); continue
    zz, _ = counts_via_module(gp_, gm_, c, Pp, tbl)
    r, w, n, nw = read_share(zz, pos_ring); rl, _, nl, _ = read_share(zz, pos_line)
    P(f"{name:38s} gates (+ {gp_:.4f}, - {gm_:.4f}, obs {c:.3f}): ring returns read {r:.3f} (wrap-only {w:.3f}, n {n}/{nw}); "
      f"line returns read {rl:.3f} (n {nl})")
open(__file__.replace(".py", "_out.txt"), "w").write("\n".join(out) + "\n")

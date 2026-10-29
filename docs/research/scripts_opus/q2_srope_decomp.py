"""POST HOC, CPU, eval-only (Opus scope expansion 2026-10-29). Why does faithful Selective RoPE miss the torus map?
Decompose the trained SROPE_TORUS models' phase into components and ask which one decoheres same-place revisits.

Readout: same-place phase coherence C0 at revisits = usage-weighted mean over rotary channels of cos(theta_query - theta_key),
query = the action token a_t that predicts the revisited observation o_t, key = the observation token o_s at the last visit
to the cell; channel weight = mean |q_c| (over action tokens) x mean |k_c| (over observation tokens) of the trained layer
(content phase ignored: C0 as in the paper's Sec. 7.4). By stratum (short gap < 128, long, wrap-only).

Counterfactual phases for SRoPE / SRoPE_NoSiLU (same trained weights, the angle recomputed with one component changed):
  full        the trained generator
  obs0        the projection of every observation token (objects and blank) zeroed BEFORE the conv (observations do
              not step; what the moves contribute is untouched)
  gate_cls    the sigmoid gate replaced by its mean over the token's class (move / observation): removes token-identity
              (what-is-seen) modulation of the step
  conv1       the conv collapsed to one tap at lag 0 equal to the sum of the taps (removes the in-flight lag: a token's step
              lands on that token); SiLU kept after it
  all3        obs0 + gate_cls + conv1
Also: the 'effective' observation step = the total phase an observation token adds once it has left the conv window,
divided by a move's (NoSiLU: exact; SiLU: by finite difference), against the registered first-difference ratio.
Accuracy under each counterfactual (eval-only; the readout was not trained on it) is reported as a secondary.
  CUDA_VISIBLE_DEVICES= python3 docs/research/scripts_opus/q2_srope_decomp.py [n_walks]
"""
import math, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/prashr")
from mapformer.environment import GridWorld  # noqa: E402
from mapformer.train_variant import VARIANT_MAP  # noqa: E402
from mapformer.eval_rank_strata import kinds_vec  # noqa: E402
from mapformer.model_srope import l2_norm  # noqa: E402

REPO = "/home/prashr/mapformer"
torch.set_num_threads(8)
N_WALKS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SEEDS = range(1700, 1708)
ARMS = {"SRoPE": "SRoPE", "SRoPE_NoSiLU": "SRoPE_NoSiLU", "MapWM": "Signed_r4", "SRoPEGen": "SRoPEGen"}
N_ACT, BLANK = 4, 20


def load(arm, s):
    V = ARMS[arm]
    b = torch.load(f"{REPO}/runs/srope_torus/{arm}/p0/{V}_s{s}/{V}.pt", map_location="cpu", weights_only=False)
    c = b["config"]
    m = VARIANT_MAP[V](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
                       grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"]); m.eval()
    return m


@torch.no_grad()
def srope_steps(gen, h, tok, obs0=False, gate_cls=False, conv1=False, zero_pos=None):
    """Re-implementation of SelectiveRoPEAngle.steps with the counterfactual switches (full-rank config only)."""
    B, T, _ = h.shape
    xh = l2_norm(h.view(B, T, gen.H, gen.d_in))
    W = gen.scalar.view(-1, 1, 1) * F.normalize(gen.weight, dim=-2)
    p = torch.einsum("bthi,hio->btho", xh, W).reshape(B, T, gen.H * gen.pe)
    is_obs = (tok >= N_ACT)
    if obs0:
        p = p * (~is_obs).unsqueeze(-1)
    if zero_pos is not None:
        p = p.clone(); p[:, zero_pos] = 0.0
    w = gen.conv.weight                                          # (C, 1, K)
    if conv1:
        K = w.shape[-1]
        w = torch.zeros_like(w); w[..., -1] = gen.conv.weight.sum(-1)   # lag-0 tap = sum (Conv1d: last tap = lag 0)
    z = F.conv1d(p.transpose(1, 2), w, padding=w.shape[-1] - 1, groups=p.shape[-1])[..., :T].transpose(1, 2)
    if gen.conv_act is not None:
        z = F.silu(z)
    z = z.reshape(B, T, gen.H, gen.pe)
    g = torch.sigmoid(gen.gate(xh.reshape(B, T, -1)))           # (B, T, H)
    if gate_cls:
        gm = g.clone()
        for msk in (is_obs, ~is_obs):
            gm[msk] = g[msk].mean(0)
        g = gm
    return z * g.unsqueeze(-1)


@torch.no_grad()
def angles(m, tok, **cf):
    """(B, H, T, nb) angles of layer 1."""
    if hasattr(m, "srope"):
        gen = m.srope[0]; h = m.layers[0].norm1(m.token_emb(tok))
        return (gen.temperature * torch.cumsum(srope_steps(gen, h, tok, **cf), 1)).transpose(1, 2)
    if hasattr(m, "angle"):
        return m.angle(m.token_emb(tok)).transpose(1, 2)
    c, s = m.get_position_state(tok)
    return torch.atan2(s, c)


@torch.no_grad()
def forward_with(m, tok, th):
    x = m.token_emb(tok); L = tok.shape[1]
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    x = m.layers[0](x, torch.cos(th), torch.sin(th), mask)
    return m.out_proj(m.out_norm(x))


@torch.no_grad()
def usage(m, tok):
    blk = m.layers[0]; h = blk.norm1(m.token_emb(tok)); B, L, _ = h.shape
    Q = blk.q_proj(h).view(B, L, blk.n_heads, blk.d_head); K = blk.k_proj(h).view(B, L, blk.n_heads, blk.d_head)
    qn = (Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2).sqrt(); kn = (K[..., 0::2] ** 2 + K[..., 1::2] ** 2).sqrt()
    act = tok < N_ACT
    return (qn[act].mean(0) * kn[~act].mean(0))                 # (H, nb)


def main():
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    CFS = {"full": {}, "obs0": {"obs0": True}, "gate_cls": {"gate_cls": True}, "conv1": {"conv1": True},
           "all3": {"obs0": True, "gate_cls": True, "conv1": True}}
    res = {}
    for arm in ARMS:
        for s in SEEDS:
            m = load(arm, s)
            np.random.seed(1234 + s)
            walks = [env.generate_trajectory(1024) for _ in range(N_WALKS)]
            tok = torch.stack([w[0] for w in walks])
            inp = tok[:, :-1]
            U = usage(m, inp)
            cfs = CFS if hasattr(m, "srope") else {"full": {}}
            for name, cf in cfs.items():
                th = angles(m, inp, **cf)                        # (B, H, L, nb)
                logits = forward_with(m, inp, th) if (hasattr(m, "srope") or hasattr(m, "path_integrator")) else m(inp)
                if name == "full":
                    dd = float((logits[:1] - m(inp[:1])).abs().max()); assert dd < 1e-3, (arm, dd)
                pred = logits.argmax(-1)
                for b, (tk, _om, rev) in enumerate(walks):
                    new_u, lag = kinds_vec(tk, 64)
                    a = tk.numpy()[0::2]; P = np.cumsum(np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])[a], 0) % 64
                    last = {}
                    for t in range(1024):
                        c = (int(P[t, 0]), int(P[t, 1]))
                        if rev[2 * t + 1] and c in last:
                            sidx = last[c]; qi, ki = 2 * t, 2 * sidx + 1
                            d = th[b, :, qi] - th[b, :, ki]                   # (H, nb)
                            coh = float((U * torch.cos(d)).sum() / U.sum())
                            st = "wrap" if new_u[t] else ("short" if lag[t] < 128 else "long")
                            ok = int(pred[b, qi] == tk[2 * t + 1])
                            res.setdefault((arm, name, st, s), []).append((coh, ok))
                        last[c] = t
            if hasattr(m, "srope"):                                   # effective (post-window) step per token class
                gen = m.srope[0]; h = m.layers[0].norm1(m.token_emb(inp))
                z = srope_steps(gen, h, inp)                           # (B, L, H, pe)
                th_full = gen.temperature * torch.cumsum(z, 1)
                inc = torch.cat([th_full[:, :1], th_full[:, 1:] - th_full[:, :-1]], 1)
                # finite-difference effect of each token, measured 4 tokens later: zero that token's projection
                eff = {"move": [], "obs": []}; fd = {"move": [], "obs": []}
                for pos in range(41, 61):
                    z2 = srope_steps(gen, h[:2], inp[:2], zero_pos=pos)   # the token's projection zeroed, gates untouched
                    dth = (gen.temperature * (z[:2] - z2).sum(1))       # total phase change from removing token pos
                    cls = "move" if int(inp[0, pos]) < N_ACT else "obs"
                    eff[cls].append(dth.abs().mean().item())
                    fd[cls].append((gen.temperature * inc[:2, pos]).abs().mean().item())
                res[(arm, "effstep", s)] = (np.mean(eff["obs"]) / np.mean(eff["move"]), np.mean(fd["obs"]) / np.mean(fd["move"]))
            print(arm, s, "done", flush=True)
    lines = []
    P = lambda *x: (print(*x), lines.append(" ".join(str(y) for y in x)))
    P(f"== same-place phase coherence C0 (usage-weighted cos) and accuracy, {N_WALKS} walks x 8 seeds, T = 1024 ==")
    for arm in ARMS:
        for name in CFS:
            if (arm, name, "short", 1700) not in res:
                continue
            cells = []
            for st in ("short", "long", "wrap"):
                c = [np.mean([x[0] for x in res[(arm, name, st, s)]]) for s in SEEDS]
                a_ = [np.mean([x[1] for x in res[(arm, name, st, s)]]) for s in SEEDS]
                cells.append(f"{st} C0 {np.mean(c):.3f} acc {np.mean(a_):.3f}")
            P(f"{arm:13s} {name:9s} | " + " | ".join(cells))
    for arm in ("SRoPE", "SRoPE_NoSiLU"):
        e = np.array([res[(arm, 'effstep', s)] for s in SEEDS])
        P(f"{arm}: observation/move |phase| -- effective (token removed, whole window) {e[:, 0].mean():.3f}; "
          f"first-difference at the token (the registered readout's kind) {e[:, 1].mean():.3f}")
    open(f"{REPO}/docs/research/scripts_opus/out/q2_srope_decomp_out.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

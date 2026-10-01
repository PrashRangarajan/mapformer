"""What/where separability of the layer-1 attention score, on stored 1-layer torus checkpoints. CPU only.

For a 1-layer model the layer-1 Q and K of a token are functions of that token's identity alone
(x = token_emb, no positional input), so the pre-softmax score between a query token a and a key
token o is EXACTLY a function S(a, o, position) of content and of the phase difference the model
assigns. We build that function on a factorial grid:

  content   c = (query action a in {N,S,W,E}) x (key observation o in 16 objects + blank) = 68 pairs
  position  path models: torus displacement d = (dx, dy) in [-R, R]^2 of the key's cell from the
            query's (query phase 0 after its own step; key phase = omega * (minimal-path sum of the
            model's OWN action steps) + omega * step(o), i.e. the key's own token step is included,
            as in the model's cumsum; the steps of the observation tokens between them are not --
            their size is reported separately as `leak`).
            index models: token lag tau in {1, 3, ..., 2L-1} (action query at even index, obs key odd).

Readouts per head (S is C x P):
  pos_share    var(position main effect) / total var
  cont_share   var(content main effect)  / total var
  inter_share  var(interaction, residual of the additive fit) / total var
  inter_of_pos inter / (pos + inter): share of the position-dependent variance that depends on content.
               0 => additive separation (content cannot change the position kernel at all).
  shape1       energy of the top singular vector of the row-centred matrix: 1 => every content pair
               uses the SAME position kernel up to a (signed) scale (multiplicative separation).
  peak0        (path models) fraction of content pairs whose argmax over displacement is d = 0.
  flip         (path models) fraction of content pairs whose row projects NEGATIVELY on the
               shared kernel oriented to peak at d = 0 (content inverting the kernel: peak -> trough).
Plus, for path models: leak = mean |step(obs)| / mean |step(action)| (omega-scaled), and opposition
|step(N)+step(S)| / |step(N)| (and W/E).

Verification (rule 9): the score function is re-implemented from the layer code; the full forward is
rebuilt from it on a real torus sequence and the logits compared with model(tokens).
"""
import glob, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
sys.path.insert(0, "/home/prashr")
from mapformer.train_variant import VARIANT_MAP
from mapformer.model import _apply_rope
from mapformer.model_pope import DELTA_MIN, DELTA_MAX
from mapformer.environment import GridWorld

REPO = "/home/prashr/mapformer"
R = 16                      # displacement window
N_ACT, N_OBS = 4, 17        # N,S,W,E ; 16 objects + blank (ids 4..20)


def load(path, variant=None, untrained_seed=None):
    if untrained_seed is not None:
        torch.manual_seed(untrained_seed)
        m = VARIANT_MAP[variant](vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64)
        return m.eval(), variant
    ck = torch.load(path, map_location="cpu", weights_only=False)
    v = ck["variant"]
    m = VARIANT_MAP[v](vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    m.load_state_dict(ck["model_state_dict"])
    return m.eval(), v


def kind(m):
    n = type(m).__name__
    if "EM" in n:
        return "em"
    pope = hasattr(m.layers[0], "pope_delta")
    path = hasattr(m, "action_to_lie")
    return ("pope" if pope else "rope") + ("_path" if path else "_index")


def scores(m, layer, hq, hk, cq, sq, ck, sk, qp=None, kp=None):
    """Pre-softmax scores (B,H,Tq,Tk) mirroring the layer code. hq/hk: normed inputs (B,T,d)."""
    B, Tq, _ = hq.shape; Tk = hk.shape[1]; H, dh = layer.n_heads, layer.d_head
    sh = lambda z, T: z.view(B, T, H, dh).transpose(1, 2)
    k = kind(m)
    if k == "em":
        Qc, Kc = sh(layer.q_content(hq), Tq), sh(layer.k_content(hk), Tk)
        AX = Qc @ Kc.transpose(-1, -2) / math.sqrt(dh)
        AP = qp @ kp.transpose(-1, -2) / math.sqrt(dh)
        return AX * AP, AX, AP
    Q, K = sh(layer.q_proj(hq), Tq), sh(layer.k_proj(hk), Tk)
    if k.startswith("pope"):
        mq, mk = F.softplus(Q), F.softplus(K)
        d = layer.pope_delta.clamp(DELTA_MIN, DELTA_MAX).view(1, H, 1, -1)
        cd, sd = torch.cos(d), torch.sin(d)
        cK, sK = ck * cd - sk * sd, sk * cd + ck * sd
        return (torch.matmul(mq * cq, (mk * cK).transpose(-1, -2))
                + torch.matmul(mq * sq, (mk * sK).transpose(-1, -2))) / math.sqrt(dh), None, None
    Qr, Kr = _apply_rope(Q, cq, sq), _apply_rope(K, ck, sk)
    return Qr @ Kr.transpose(-1, -2) / math.sqrt(dh), None, None


def phases_seq(m, tokens):
    """cos/sin (B,H,T,nb) the model uses for a real sequence (and EM's q_pos/k_pos)."""
    B, L = tokens.shape
    x = m.token_emb(tokens)
    k = kind(m)
    if k.endswith("index"):
        if k.startswith("pope"):
            ang = torch.outer(torch.arange(L, dtype=x.dtype), m.theta_c)
            c = ang.cos()[None, None].expand(B, m.n_heads, L, -1); s = ang.sin()[None, None].expand(B, m.n_heads, L, -1)
        else:
            c, s = m._rope_cos_sin(L, x.device, x.dtype); c, s = c.expand(B, -1, -1, -1), s.expand(B, -1, -1, -1)
        return c, s, None, None
    c, s = m.path_integrator(m.action_to_lie(x))
    if k == "em":
        q0 = m.q0_pos[None, :, None, :].expand(B, -1, L, -1); k0 = m.k0_pos[None, :, None, :].expand(B, -1, L, -1)
        return c, s, _apply_rope(q0, c, s), _apply_rope(k0, c, s)
    return c, s, None, None


@torch.no_grad()
def verify(m, tokens):
    """Rebuild the forward from `scores` and compare logits with the model's own."""
    layer = m.layers[0]
    x = m.token_emb(tokens); B, L, D = x.shape
    c, s, qp, kp = phases_seq(m, tokens)
    h = layer.norm1(x)
    S, _, _ = scores(m, layer, h, h, c, s, c, s, qp, kp)
    S = S.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
    V = layer.v_proj(h).view(B, L, layer.n_heads, layer.d_head).transpose(1, 2)
    out = (F.softmax(S, -1) @ V).transpose(1, 2).reshape(B, L, D)
    x2 = x + layer.o_proj(out); x2 = x2 + layer.ffn(layer.norm2(x2))
    mine = m.out_proj(m.out_norm(x2))
    return (mine - m(tokens)).abs().max().item()


@torch.no_grad()
def factorial(m):
    layer = m.layers[0]; H = m.n_heads
    E = m.token_emb.weight                                   # (21, d)
    hq = layer.norm1(E[:N_ACT])[None]                        # queries: actions
    hk = layer.norm1(E[N_ACT:N_ACT + N_OBS])[None]           # keys: obs + blank
    k = kind(m); out = {"kind": k}
    if k.endswith("index"):
        lags = torch.arange(1, 256, 2, dtype=torch.float32)  # T=128 steps -> 256 tokens
        freqs = m.theta_c if k.startswith("pope") else m.inv_freq
        P = len(lags)
        rows = []
        for lag in lags:
            aq = (lag * freqs)[None, None, None, :].expand(1, H, N_ACT, -1)
            ak = torch.zeros(1, H, N_OBS, freqs.numel())
            Sx, _, _ = scores(m, layer, hq, hk, aq.cos(), aq.sin(), ak.cos(), ak.sin())
            rows.append(Sx[0])                               # (H, A, O)
        S = torch.stack(rows, -1)                            # (H, A, O, P)
        S = S.reshape(H, N_ACT * N_OBS, P)
        out["P"] = P
    else:
        delta = m.action_to_lie(E[None])[0]                 # (21, H, nb)
        om = m.path_integrator.omega                         # (H, nb)
        st = delta * om                                      # omega-scaled step per token
        Nn, Ss, Ww, Ee = st[0], st[1], st[2], st[3]
        rng = range(-R, R + 1)
        disp = [(dx, dy) for dx in rng for dy in rng]
        P = len(disp); i0 = disp.index((0, 0))
        ang = torch.stack([(dx * Ee if dx > 0 else -dx * Ww) + (dy * Nn if dy > 0 else -dy * Ss)
                           for dx, dy in disp])              # (P, H, nb) key-cell phase
        keyang = ang[:, None] + st[N_ACT:N_ACT + N_OBS][None]   # (P, O, H, nb) + key's own step
        aq = torch.zeros(1, H, N_ACT, delta.shape[-1])
        rows, axs = [], None
        for p in range(P):
            ak = keyang[p].permute(1, 0, 2)[None]            # (1, H, O, nb)
            if k == "em":
                qp = _apply_rope(m.q0_pos[None, :, None, :].expand(1, H, N_ACT, -1), aq.cos(), aq.sin())
                kp = _apply_rope(m.k0_pos[None, :, None, :].expand(1, H, N_OBS, -1), ak.cos(), ak.sin())
                Sx, axs, _ = scores(m, layer, hq, hk, None, None, None, None, qp, kp)
            else:
                Sx, _, _ = scores(m, layer, hq, hk, aq.cos(), aq.sin(), ak.cos(), ak.sin())
            rows.append(Sx[0])
        S = torch.stack(rows, -1).reshape(H, N_ACT * N_OBS, P)
        out["P"] = P
        nrm = lambda v: v.norm(dim=-1)                       # per head
        act = nrm(st[:N_ACT]).mean(0); obs = nrm(st[N_ACT:N_ACT + N_OBS]).mean(0)
        out["leak"] = (obs / act).tolist()
        out["opp_NS"] = (nrm(Nn + Ss) / nrm(Nn)).tolist()
        out["opp_WE"] = (nrm(Ww + Ee) / nrm(Ee)).tolist()
        if axs is not None:
            out["AX_neg_frac"] = (axs[0] < 0).float().mean(dim=(-1, -2)).tolist()
    res = []
    for h in range(H):
        X = S[h].double().numpy()                            # (C, P)
        g = X.mean(); rm = X.mean(1, keepdims=True); cm = X.mean(0, keepdims=True)
        tot = ((X - g) ** 2).mean()
        cont = ((rm - g) ** 2).mean(); pos = ((cm - g) ** 2).mean()
        inter = ((X - rm - cm + g) ** 2).mean()
        Xc = X - rm
        sv = np.linalg.svd(Xc, compute_uv=True)
        u, sig, vt = sv
        r = {"total_sd": float(np.sqrt(tot)), "cont_share": cont / tot, "pos_share": pos / tot,
             "inter_share": inter / tot, "inter_of_pos": inter / (pos + inter),
             "shape1": float(sig[0] ** 2 / (sig ** 2).sum())}
        if not k.endswith("index"):
            r["peak0"] = float((X.argmax(1) == i0).mean())
            v = vt[0] * (1 if vt[0][i0] >= vt[0].mean() else -1)    # orient shared kernel to peak at d=0
            proj = Xc @ v
            r["flip"] = float((proj < 0).mean())
            r["pos_sd"] = float(np.sqrt(pos + inter))
        res.append(r)
    out["heads"] = res
    return out


def main():
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    np.random.seed(1)
    toks = torch.stack([env.generate_trajectory(128)[0] for _ in range(4)])
    arms = ["RoPE", "PoPE-Flat", "Vanilla", "MapPoPE-Flat", "Vanilla_r4", "MapPoPE_r4"]
    paths = {a: sorted(glob.glob(f"{REPO}/runs/paper2x2/p0/{a}_s[0-7]/{a}.pt")) for a in arms}
    paths["VanillaEM_r4 (dof batch)"] = sorted(glob.glob(f"{REPO}/runs/dof/torus/VanillaEM_r4_s[0-7]/VanillaEM_r4.pt"))
    allres = {}
    for arm, ps in paths.items():
        allres[arm] = []
        for p in ps:
            m, v = load(p)
            err = verify(m, toks)
            r = factorial(m); r["verify_maxabs"] = err; r["path"] = os.path.relpath(p, REPO)
            allres[arm].append(r)
        # untrained control, 3 inits
        v = arm.split()[0]
        allres[arm + " UNTRAINED"] = []
        for s in range(3):
            m, _ = load(None, v, untrained_seed=s)
            r = factorial(m); r["verify_maxabs"] = verify(m, toks)
            allres[arm + " UNTRAINED"].append(r)
    json.dump(allres, open(os.path.join(os.path.dirname(__file__), "probe_whatwhere.json"), "w"), indent=1)
    keys = ["pos_share", "cont_share", "inter_share", "inter_of_pos", "shape1", "peak0", "flip"]
    print(f"{'arm':32s} n  verify   " + " ".join(f"{k:>12s}" for k in keys) + "   leak  oppNS  oppWE")
    for arm, rs in allres.items():
        if not rs: continue
        # per run, take the head with the larger position-dependent variance (the 'where' head),
        # and also report the mean over both heads
        for tag, pick in (("where-head", "max"), ("both-heads", "mean")):
            vals = {k: [] for k in keys}
            for r in rs:
                hs = r["heads"]
                if pick == "max":
                    hsel = [max(hs, key=lambda h: h["pos_share"] + h["inter_share"] if "pos_sd" not in h else h["pos_sd"])]
                else:
                    hsel = hs
                for k in keys:
                    if k in hsel[0]:
                        vals[k].append(np.mean([h[k] for h in hsel]))
            line = f"{arm[:24]+' '+tag[:6]:32s} {len(rs)}  {max(r['verify_maxabs'] for r in rs):.1e} "
            for k in keys:
                line += f" {np.mean(vals[k]):5.3f}+-{np.std(vals[k]):4.3f}" if vals[k] else f" {'--':>12s}"
            if "leak" in rs[0]:
                line += f"  {np.mean([np.mean(r['leak']) for r in rs]):.3f}  {np.mean([np.mean(r['opp_NS']) for r in rs]):.3f}  {np.mean([np.mean(r['opp_WE']) for r in rs]):.3f}"
            if "AX_neg_frac" in rs[0]:
                line += f"  AX<0 {np.mean([np.mean(r['AX_neg_frac']) for r in rs]):.3f}"
            print(line)


if __name__ == "__main__":
    main()

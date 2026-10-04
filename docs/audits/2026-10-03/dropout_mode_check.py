"""Eval-mode vs train-mode accuracy on identical walks (found 2026-10-03 in the TW_NORMSTEP pilot: MapWM s100 scores
0.85 in eval mode, 0.99 in train mode). Per checkpoint: held-out-map revisit accuracy / NLL (map 10000, np seed 10**6,
40 walks, T=1024) with all dropout off (eval), all on (train, mean of 3 dropout seeds), and only one dropout site on:
attention probabilities, attention-output residual, FFN output."""
import sys, glob, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
from mapformer.environment_textworld import TextWorld
from mapformer.analyze_textworld_secondary import load as load_tw
from mapformer.tw_normstep_readouts import load as load_ns


def walks(n=40, T=1024):
    te = TextWorld(size=64, seed=10000); np.random.seed(10**6)
    return [te.generate_trajectory(T) for _ in range(n)]


@torch.no_grad()
def score(m, W, mode, dseed=0):
    m.eval(); torch.manual_seed(dseed); ok = tot = 0; nll = 0.0
    for tok, _o, rev in W:
        lg = fwd(m, tok[None, :-1], mode)[0]; msk = rev[1:]
        lp = F.log_softmax(lg, -1); ok += int((lg.argmax(-1)[msk] == tok[1:][msk]).sum()); tot += int(msk.sum())
        nll += float(-lp[msk].gather(1, tok[1:][msk][:, None]).sum())
    return ok / tot, nll / tot


@torch.no_grad()
def fwd(m, tok, mode):
    # re-implements MapFormerWM / _StepOverride forward + WMTransformerLayer with per-site dropout control
    import math
    x = m.token_emb(tok)
    d = m.step(tok, x) if hasattr(m, "step") else m.action_to_lie(x)
    cos_a, sin_a = m.path_integrator(d)
    from mapformer.model import _apply_rope
    T = tok.shape[1]; cm = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    p = 0.1; drop = lambda z, on: F.dropout(z, p, training=on)
    for L in m.layers:
        B = x.shape[0]; h = L.norm1(x)
        Q = _apply_rope(L.q_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2), cos_a, sin_a)
        K = _apply_rope(L.k_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2), cos_a, sin_a)
        V = L.v_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2)
        s = (Q @ K.transpose(-1, -2) / math.sqrt(L.d_head)).masked_fill(cm, float("-inf"))
        a = drop(F.softmax(s, -1), mode in ("train", "attn"))
        o = L.o_proj((a @ V).transpose(1, 2).reshape(B, T, L.d_model))
        x = x + drop(o, mode in ("train", "resid"))
        f = L.ffn[2](L.ffn[1](L.ffn[0](L.norm2(x))))
        x = x + drop(f, mode in ("train", "ffn"))
    return m.out_proj(m.out_norm(x))


if __name__ == "__main__":
    W = walks()
    cks = sys.argv[1:]
    for ck in cks:
        m = (load_ns(ck)[0] if "tw_normstep" in ck else load_tw(ck, "cpu")).eval()
        with torch.no_grad():                            # the re-implementation must equal the model in eval mode
            tok = W[0][0][None, :-1]; assert (fwd(m, tok, "eval") - m(tok)).abs().max() < 1e-4
        r = {"eval": score(m, W, "eval")}
        tr = [score(m, W, "train", s) for s in range(3)]; r["train"] = tuple(np.mean(tr, 0))
        for site in ("attn", "resid", "ffn"):
            r[site] = score(m, W, site)
        print(ck.split("/runs/")[-1], "  ".join(f"{k} {a:.4f}/{n:.3f}" for k, (a, n) in r.items()), flush=True)

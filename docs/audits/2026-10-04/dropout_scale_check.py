"""Verifies the formal-theory claim (docs/theory/2026-10-04/03_formal.md): the eval/train-mode gap is the 1/(1-p)
scale of inverted attention dropout, not its noise. Same 40 held-out walks as dropout_mode_check (map 10000, np seed
10**6, T=1024): eval mode; eval with attention probabilities x 1/(1-p) = 1.111 (no noise); attention dropout on."""
import math, sys, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-03")
from dropout_mode_check import walks, fwd, score
from mapformer.model import _apply_rope
from mapformer.analyze_textworld_secondary import load as load_tw
from mapformer.tw_normstep_readouts import load as load_ns


@torch.no_grad()
def fwd_scaled(m, tok, scale):
    x = m.token_emb(tok); d = m.step(tok, x) if hasattr(m, "step") else m.action_to_lie(x)
    cos_a, sin_a = m.path_integrator(d); T = tok.shape[1]; cm = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    for L in m.layers:
        B = x.shape[0]; h = L.norm1(x)
        Q = _apply_rope(L.q_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2), cos_a, sin_a)
        K = _apply_rope(L.k_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2), cos_a, sin_a)
        V = L.v_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2)
        a = F.softmax((Q @ K.transpose(-1, -2) / math.sqrt(L.d_head)).masked_fill(cm, float("-inf")), -1) * scale
        x = x + L.o_proj((a @ V).transpose(1, 2).reshape(B, T, L.d_model))
        x = x + L.ffn[2](L.ffn[1](L.ffn[0](L.norm2(x))))
    return m.out_proj(m.out_norm(x))


def acc(m, W, scale):
    ok = tot = 0
    for tok, _o, rev in W:
        lg = fwd_scaled(m, tok[None, :-1], scale)[0]; msk = rev[1:]
        ok += int((lg.argmax(-1)[msk] == tok[1:][msk]).sum()); tot += int(msk.sum())
    return ok / tot


if __name__ == "__main__":
    W = walks()
    for ck in sys.argv[1:]:
        m = (load_ns(ck)[0] if "tw_normstep" in ck else load_tw(ck, "cpu")).eval()
        print(ck.split("/runs/")[-1], f"eval {acc(m, W, 1.0):.4f}  eval x1.111 {acc(m, W, 1 / 0.9):.4f}  "
              f"attn dropout on {score(m, W, 'attn')[0]:.4f}", flush=True)

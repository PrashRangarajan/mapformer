"""Auditor check C1: re-derive the model from the stated recurrence, independently of model_ssm_sign's own
sequential_reference. float64 throughout. Everything (step table, class functions, moves mask, omega, the recurrence,
the RMS norm, the residual/FFN/out head) is recomputed from the raw state dict with numpy loops."""
import sys
import numpy as np
import torch
from mapformer.environment_steps import GridWorldSteps
from mapformer.model_ssm_sign import build

torch.set_default_dtype(torch.float64)
env = GridWorldSteps(size=32, seed=2171, p_first=0.5, deltas=(1, -1))
np.random.seed(2172)
toks = [env.generate_trajectory(40)[0][:-1] for _ in range(2)]
tok = torch.stack(toks)
V = env.unified_vocab_size


def ln(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def my_forward(sd, mode, steps, tokens, act=(0, 1), sign=-1.0):
    P = {k: v.detach().numpy().astype(np.float64) for k, v in sd.items()}
    E = P["token_emb.weight"]
    s = E @ P["action_to_lie.w_in.weight"].T @ P["action_to_lie.w_out.weight"].T      # (V, 64)
    f = {"signed": lambda z: z, "abs": np.abs, "softplus": lambda z: np.log1p(np.exp(z)), "exp": np.exp}
    if mode == "index":
        D = np.ones_like(s)
    else:
        D = f[mode](s)
        if steps == "moves":
            mask = np.zeros(V, bool); mask[list(act)] = True
            D = D * mask[:, None]
    om = P["path_integrator.omega"].reshape(-1)          # (H*N,) head-major
    ang = D * om                                         # per-token angle (V, 64)
    out = []
    for b in range(tokens.shape[0]):
        t_ = tokens[b].numpy()
        x = E[t_]                                        # (T, d)
        h = ln(x, P["layers.0.norm1.weight"], P["layers.0.norm1.bias"])
        q = h @ P["layers.0.q_proj.weight"].T + P["layers.0.q_proj.bias"]
        k = h @ P["layers.0.k_proj.weight"].T + P["layers.0.k_proj.bias"]
        v = h @ P["layers.0.v_proj.weight"].T + P["layers.0.v_proj.bias"]
        T = len(t_)
        y = np.zeros((T, 128))
        for hd in range(2):
            sl = slice(64 * hd, 64 * hd + 64)
            qc = q[:, sl][:, 0::2] + 1j * q[:, sl][:, 1::2]   # (T, 32)
            kc = k[:, sl][:, 0::2] + 1j * k[:, sl][:, 1::2]
            vv = v[:, sl]                                     # (T, 64)
            a = ang[t_][:, 32 * hd:32 * hd + 32]              # (T, 32)
            Hs = np.zeros((32, 64), complex)
            for t in range(T):
                Hs = np.exp(sign * 1j * a[t])[:, None] * Hs + np.outer(kc[t], vv[t])
                yt = (qc[t].conj() @ Hs).real
                y[t, sl] = yt
        # RMS per head
        yy = y.reshape(T, 2, 64)
        yy = yy / np.sqrt((yy ** 2).mean(-1, keepdims=True) + 1e-5)
        y = yy.reshape(T, 128) * P["layers.0.y_norm.weight"]
        o = y @ P["layers.0.o_proj.weight"].T + P["layers.0.o_proj.bias"]
        x = x + o
        h2 = ln(x, P["layers.0.norm2.weight"], P["layers.0.norm2.bias"])
        ff = gelu(h2 @ P["layers.0.ffn.0.weight"].T + P["layers.0.ffn.0.bias"]) @ P["layers.0.ffn.2.weight"].T + P["layers.0.ffn.2.bias"]
        x = x + ff
        x = ln(x, P["out_norm.weight"], P["out_norm.bias"])
        out.append(x @ P["out_proj.weight"].T + P["out_proj.bias"])
    return np.stack(out)


worst = 0.0
rows = []
for mode in ("signed", "abs", "softplus", "exp", "index"):
    for steps in ("moves", "all"):
        torch.manual_seed(2170)
        m = build(mode, steps, V, 32).double().eval()
        g = torch.Generator().manual_seed(2173)
        with torch.no_grad():   # non-trivial steps: angles O(1) rad per move (exp milder)
            sc = 0.5 if mode == "exp" else 2.0
            m.action_to_lie.w_out.weight.copy_(torch.randn(m.action_to_lie.w_out.weight.shape, generator=g) * sc)
            m.action_to_lie.w_in.weight.mul_(4.0)
        with torch.no_grad():
            par = m(tok).numpy()
        mine = my_forward(m.state_dict(), mode, steps, tok)
        rel = np.abs(par - mine).max() / np.abs(mine).max()
        # the model's own step table vs mine
        worst = max(worst, rel)
        rows.append((mode, steps, rel))
        print(f"{mode:9s} {steps:6s} parallel vs auditor recurrence rel {rel:.1e}")
# gauge claim: exp(+i omega D) with omega -> -omega equals the model
torch.manual_seed(2170)
m = build("abs", "moves", V, 32).double().eval()
sd = {k: v.clone() for k, v in m.state_dict().items()}
sd2 = dict(sd); sd2["path_integrator.omega"] = -sd["path_integrator.omega"]
a = my_forward(sd, "abs", "moves", tok, sign=-1.0)
b = my_forward(sd2, "abs", "moves", tok, sign=+1.0)
print(f"gauge: exp(-i om D) == exp(+i (-om) D): {np.abs(a - b).max():.1e}")
# is the sign of the rotation itself a gauge? exp(+i om D) vs exp(-i om D) with the same omega
c = my_forward(sd, "abs", "moves", tok, sign=+1.0)
print(f"exp(+i om D) vs exp(-i om D), same omega (not a gauge in general unless q/k are conjugated): {np.abs(a - c).max():.2e}")
print("WORST", worst)
sys.exit(0 if worst < 1e-10 else 1)

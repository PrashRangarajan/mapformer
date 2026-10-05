"""Minimal 1-layer path-integrated attention on a D-torus (CPU). No FFN. Map redrawn per sequence, interleaved
[a1,o1,a2,o2,...], loss on revisited observations (predicted at the preceding action token), per-head rank r step map
Delta_h = omega_h * W_out^h W_in^h e(token). Arms: plain; --antisym 1 (action steps exactly odd, observation steps 0:
no per-move common component can exist; Lemma 1's obstruction removed by construction)."""
import sys, math, time, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad')
from toy import walks

class Toy(nn.Module):
    def __init__(s, V, D, d, H, r, N, antisym):
        super().__init__(); s.H, s.dh, s.nb, s.r, s.D, s.antisym = H, d // H, d // H // 2, r, D, antisym
        s.emb = nn.Embedding(V, d); s.win = nn.Linear(d, H * r, bias=False)
        s.wout = nn.Parameter(torch.empty(H, s.nb, r).uniform_(-1 / math.sqrt(r), 1 / math.sqrt(r)))
        om = torch.tensor([2 * math.pi * (1 / N) ** (i / max(s.nb - 1, 1)) for i in range(s.nb)])
        s.omega = nn.Parameter(om.repeat(H, 1)); s.ln1 = nn.LayerNorm(d)
        s.q = nn.Linear(d, d); s.k = nn.Linear(d, d); s.v = nn.Linear(d, d); s.o = nn.Linear(d, d)
        s.lno = nn.LayerNorm(d); s.out = nn.Linear(d, V)
        partner = torch.arange(V)
        for i in range(D): partner[2*i] = 2*i+1; partner[2*i+1] = 2*i
        s.register_buffer('partner', partner); s.register_buffer('isact', (torch.arange(V) < 2 * D).float())
    def steps(s, tok):
        z = s.win(s.emb(tok)).view(*tok.shape, s.H, s.r)
        if s.antisym:
            zp = s.win(s.emb(s.partner[tok])).view(*tok.shape, s.H, s.r)
            z = (z - zp) * 0.5 * s.isact[tok][..., None, None]
        return torch.einsum('bthr,hnr->bthn', z, s.wout) * s.omega          # B,T,H,nb (radians)
    def forward(s, tok):
        B, T = tok.shape; x = s.emb(tok); th = torch.cumsum(s.steps(tok), 1).transpose(1, 2)   # B,H,T,nb
        c, sn = torch.cos(th), torch.sin(th); h = s.ln1(x)
        def rot(z):
            z = z.view(B, T, s.H, s.dh).transpose(1, 2); a, b = z[..., 0::2], z[..., 1::2]
            return torch.cat([a * c - b * sn, a * sn + b * c], -1)
        Q, K = rot(s.q(h)), rot(s.k(h)); Vv = s.v(h).view(B, T, s.H, s.dh).transpose(1, 2)
        sc = (Q @ K.transpose(-1, -2)) / math.sqrt(s.dh)
        sc = sc.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf'))
        o = (sc.softmax(-1) @ Vv).transpose(1, 2).reshape(B, T, -1)
        return s.out(s.lno(x + s.o(o)))

def geometry(m, D, N):
    """per head: cond of the phase frame, clock ratio (common per-move phase / frame), winding residual (Sec 1 probe)."""
    with torch.no_grad():
        V = m.emb.num_embeddings; P = m.steps(torch.arange(V)[None])[0].numpy()     # V,H,nb
    out = []
    for h in range(m.H):
        K = np.stack([(P[2*j, h] - P[2*j+1, h]) / 2 for j in range(D)], 1)
        tau = P[:2*D, h].mean(0) + P[2*D:, h].mean(0)
        sv = np.linalg.svd(K, compute_uv=False)
        res = np.abs(N * K / (2 * np.pi) - np.round(N * K / (2 * np.pi))).mean()
        out.append(dict(cond=float(sv[-1] / sv[0]), clk=float(np.linalg.norm(tau) / np.linalg.norm(K)), wind=float(res)))
    return out

def run(a):
    torch.manual_seed(a.seed); rng = np.random.default_rng(1000 + a.seed); D, N, T = a.D, a.N, a.T
    m = Toy(2 * D + 17, D, a.d, a.H, a.r, N, a.antisym)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    hist = []; snaps = []
    for step in range(a.steps):
        tok, rev, _ = walks(rng, a.B, T, D, N)
        lg = m(tok)[:, 0::2]; loss = F.cross_entropy(lg[rev], tok[:, 1::2][rev])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); hist.append(loss.item())
        if (step + 1) % (a.steps // 5) == 0: snaps.append(geometry(m, D, N))
    m.eval(); erng = np.random.default_rng(99)
    with torch.no_grad():
        tok, rev, wrap = walks(erng, 64, T, D, N); ok = m(tok)[:, 0::2].argmax(-1) == tok[:, 1::2]
    return dict(seed=a.seed, r=a.r, H=a.H, antisym=a.antisym, D=D, N=N, T=T, steps=a.steps,
                final_loss=float(np.mean(hist[-a.steps // 20:])), acc=float(ok[rev].float().mean()),
                acc_wrap=float(ok[rev & wrap].float().mean()), acc_other=float(ok[rev & ~wrap].float().mean()),
                geom=snaps, curve=[float(np.mean(hist[i:i + 250])) for i in range(0, a.steps, 250)])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k, v in dict(D=2, N=16, T=192, r=2, H=2, d=32, B=16, steps=10000, lr=3e-3, seed=0, antisym=0).items():
        ap.add_argument('--' + k, type=type(v), default=v)
    a = ap.parse_args(); torch.set_num_threads(4); t0 = time.time(); res = run(a); res['sec'] = time.time() - t0
    print(json.dumps(res), flush=True)

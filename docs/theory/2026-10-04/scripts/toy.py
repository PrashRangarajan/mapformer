"""Small-scale MapWM on a D-torus, CPU. Uses the repo's MapFormerWM layer code with a per-head rank-r step map.
Map redrawn per sequence (in-context map). Loss on revisited observations only (paper)."""
import sys, math, time, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/home/prashr')
from mapformer.model import MapFormerWM
from mapformer.model_rank_perhead import ActionToLieAlgebraPerHead

def walks(rng, B, T, D, N, K=16, p_empty=0.5):
    nA = 2 * D
    deltas = np.zeros((nA, D), int)
    for i in range(D): deltas[2*i, i] = 1; deltas[2*i+1, i] = -1
    acts = np.empty((B, T), int)
    for b in range(B):
        t = 0
        while t < T:
            a = rng.integers(nA); k = rng.integers(1, 11)
            acts[b, t:t+k] = a; t += k
    unw = np.cumsum(deltas[acts], 1) + rng.integers(0, N, (B, 1, D))      # unwrapped positions
    pos = unw % N
    cell = np.ravel_multi_index(tuple(pos[..., i] for i in range(D)), (N,) * D)   # B,T
    maps = np.where(rng.random((B, N ** D)) < p_empty, K, rng.integers(0, K, (B, N ** D)))
    obs = np.take_along_axis(maps, cell, 1)
    tok = np.empty((B, 2 * T), int); tok[:, 0::2] = acts; tok[:, 1::2] = obs + nA
    rev = np.zeros((B, T), bool); wrap = np.zeros((B, T), bool)
    for b in range(B):
        seen_c, seen_u = set(), set()
        for t in range(T):
            c = cell[b, t]; u = tuple(unw[b, t])
            if c in seen_c:
                rev[b, t] = True; wrap[b, t] = u not in seen_u
            seen_c.add(c); seen_u.add(u)
    return torch.tensor(tok), torch.tensor(rev), torch.tensor(wrap)

class AntiSym(nn.Module):
    """Step map with an exact antisymmetric action code and zero observation steps (no clock possible)."""
    def __init__(self, base, D):
        super().__init__(); self.base = base; self.D = D
    def forward(self, x):
        d = self.base(x)                     # B,T,H,nb (x is the embedding sequence)
        return d
def build(args, vocab):
    m = MapFormerWM(vocab, d_model=args.d, n_heads=args.H, n_layers=1, dropout=0.0, grid_size=args.N)
    m.action_to_lie = ActionToLieAlgebraPerHead(args.d, args.H, m.n_blocks, args.r)
    return m

def run(args):
    torch.manual_seed(args.seed); rng = np.random.default_rng(1000 + args.seed)
    D, N, T = args.D, args.N, args.T; nA = 2 * D; vocab = nA + 17
    m = build(args, vocab)
    if args.antisym:
        # step(token) := A(e_token) - A(e_partner) for actions (partner = opposite), 0 for observations
        a2l = m.action_to_lie; partner = torch.arange(vocab)
        for i in range(D): partner[2*i] = 2*i+1; partner[2*i+1] = 2*i
        isact = (torch.arange(vocab) < nA).float()
        emb = m.token_emb
        orig = a2l.forward
        def fwd(x, _o=orig):
            return _o(x)
        m._antisym = (partner, isact)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05,
                                                anneal_strategy='cos')
    def forward(tok):
        if not args.antisym: return m(tok)
        partner, isact = m._antisym
        x = m.token_emb(tok); xp = m.token_emb(partner[tok])
        delta = (m.action_to_lie(x) - m.action_to_lie(xp)) * 0.5 * isact[tok][..., None, None]
        cos_a, sin_a = m.path_integrator(delta)
        L = tok.shape[1]; mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        for layer in m.layers: x = layer(x, cos_a, sin_a, mask)
        return m.out_proj(m.out_norm(x))
    hist = []
    for step in range(args.steps):
        tok, rev, _ = walks(rng, args.B, T, D, N)
        logits = forward(tok)[:, 0::2]                     # at action tokens -> predict next obs
        tgt = tok[:, 1::2]
        loss = F.cross_entropy(logits[rev], tgt[rev])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(loss.item())
    m.eval(); erng = np.random.default_rng(99)
    with torch.no_grad():
        tok, rev, wrap = walks(erng, 64, T, D, N)
        pred = forward(tok)[:, 0::2].argmax(-1); ok = pred == tok[:, 1::2]
    fl = float(np.mean(hist[-max(1, args.steps // 20):]))
    return dict(seed=args.seed, r=args.r, D=D, N=N, T=T, antisym=args.antisym, final_loss=fl,
                acc=float(ok[rev].float().mean()), acc_wrap=float(ok[rev & wrap].float().mean()),
                acc_other=float(ok[rev & ~wrap].float().mean()), wrap_share=float((rev & wrap).sum() / rev.sum()),
                curve=[float(np.mean(hist[i:i+100])) for i in range(0, args.steps, 100)])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k, v in dict(D=2, N=12, T=192, r=2, H=2, d=64, B=32, steps=3000, lr=3e-3, seed=0, antisym=0).items():
        ap.add_argument('--' + k, type=type(v), default=v)
    args = ap.parse_args(); torch.set_num_threads(2)
    t0 = time.time(); res = run(args); res['sec'] = time.time() - t0
    print(json.dumps(res))

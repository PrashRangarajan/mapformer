import sys, numpy as np, torch
from mapformer.environment_textworld_ctx2 import TextWorldCtx2
from mapformer.train_ctxstep2 import ARMS
R = "/home/prashr/mapformer/runs/ctxstep2_pilot"
def load(cue, arm, L, s, V):
    b = torch.load(f"{R}/{cue}_{arm}_L{L}_s{s}/{arm}.pt", map_location="cpu", weights_only=False)
    m = ARMS[arm](vocab_size=V, d_model=128, n_heads=2, n_layers=L, grid_size=64); m.load_state_dict(b["model_state_dict"]); return m.eval()
def theta(m, tok):
    with torch.no_grad():
        if hasattr(m, "angle"): return m.angle(m.token_emb(tok))
        d = m.step(tok)[0] if hasattr(m, "step") else m.action_to_lie(m.token_emb(tok))
        return torch.cumsum(d, 1) * m.path_integrator.omega
for cue in ("lead", "trail"):
    env = TextWorldCtx2(seed=10000, cue=cue); V = env.unified_vocab_size
    np.random.seed(321); data = []
    for _ in range(30):
        t = env.generate_trajectory(1024)[0]; data.append((t, list(env.ctx)))
    N, S = env.idx["north"], env.idx["south"]
    def eff(m, want):
        out = []
        for tok, ctx in data:
            for i, k in ctx:
                if (k == "move") != (want == "move") or i + 8 >= len(tok): continue
                a = tok.clone(); a[i] = N; b = tok.clone(); b[i] = S
                out.append((theta(m, a[None])[0, i + 7] - theta(m, b[None])[0, i + 7]).abs().mean().item())
                if len(out) >= 50: return np.mean(out)
        return np.mean(out)
    for arm, L in (("CF", 1), ("CG", 1), ("SR", 1), ("HS", 2)):
        for s in (0, 1):
            m = load(cue, arm, L, s, V); mv, dc = eff(m, "move"), eff(m, "decoy")
            print(f"[{cue}] {arm} s{s}: angle change move {mv:.3f} decoy {dc:.3f} -> ratio {dc / mv:.2f}", flush=True)

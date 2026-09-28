"""Pre-launch gate for SIGN_MATCHED_PREREG.md, at the T=1024 config."""
import numpy as np, torch
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP


def build(v, s):
    torch.manual_seed(s); np.random.seed(s)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=s)
    return env, VARIANT_MAP[v](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)


s = 3
env, sg = build("Signed_r4", s)
toks = torch.randint(0, env.unified_vocab_size, (4, 1024))
for v in ("Signed_r4", "Abs_r4", "Pos_r4", "RoPE", "Vanilla_r4"):
    m = build(v, s)[1]
    print(f"{v:11s} params {sum(p.numel() for p in m.parameters())}")
    if v in ("Abs_r4", "Pos_r4"):
        with torch.no_grad():
            print(f"   min Delta at init {m.delta_of(toks).min().item():+.4g}")
van = build("Vanilla_r4", s)[1]; van.load_state_dict(sg.state_dict()); sg.eval(); van.eval()
with torch.no_grad():
    print("Signed_r4 vs Vanilla_r4 on Signed's weights, max|logit diff|:", (sg(toks) - van(toks)).abs().max().item())
    t2 = toks.clone(); t2[:, 900:] = torch.randint(0, env.unified_vocab_size, (4, 124))
    a = build("Abs_r4", s)[1].eval()
    print("causal leak Abs_r4 (positions < 900):", (a(toks)[:, :900] - a(t2)[:, :900]).abs().max().item())

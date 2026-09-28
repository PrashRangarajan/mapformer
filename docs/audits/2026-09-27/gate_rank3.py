"""Pre-launch gate for RANK3_PREREG.md: E (Vanilla_r3ph) shares every non-bottleneck weight with
Vanilla (A) at the same seed, its initial angle-increment std sits beside A/B/D, and it is causal."""
import numpy as np, torch
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP


def build(v, s):
    torch.manual_seed(s); np.random.seed(s)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=s)
    return env, VARIANT_MAP[v](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)


for s in (0, 5):
    env, A = build("Vanilla", s)
    ms = {v: build(v, s)[1] for v in ("Vanilla_r2ph", "Vanilla_r3ph", "Vanilla_r4ph")}
    sa = A.state_dict()
    for v, m in ms.items():
        sm = m.state_dict()
        shared = [k for k in sa if k in sm and not k.startswith("action_to_lie")]
        d = max((sa[k].float() - sm[k].float()).abs().max().item() for k in shared)
        nb = sum(p.numel() for n, p in m.named_parameters() if n.startswith("action_to_lie"))
        print(f"seed {s} {v}: {len(shared)} shared keys, max |diff| vs A {d:.1e}; bottleneck params {nb}; "
              f"total {sum(p.numel() for p in m.parameters())}")
    toks = torch.randint(0, env.unified_vocab_size, (8, 256))
    for v, m in [("Vanilla", A)] + list(ms.items()):
        m.eval()
        with torch.no_grad():
            x = m.token_emb(toks) if hasattr(m, "token_emb") else m.embed(toks)
            dl = m.action_to_lie(x)
        print(f"seed {s} {v}: initial angle-increment std {dl.std().item():.3f}")
# causal leak for E
m = ms["Vanilla_r3ph"]; m.eval()
t = torch.randint(0, env.unified_vocab_size, (2, 128)); t2 = t.clone(); t2[:, 100:] = torch.randint(0, env.unified_vocab_size, (2, 28))
with torch.no_grad():
    o1 = m(t); o2 = m(t2)
o1 = o1[0] if isinstance(o1, tuple) else o1; o2 = o2[0] if isinstance(o2, tuple) else o2
print("causal leak E (positions < 100):", (o1[:, :100] - o2[:, :100]).abs().max().item())

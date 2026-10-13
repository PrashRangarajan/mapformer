import math, torch, numpy as np
from mapformer.train_variant import VARIANT_MAP
from mapformer.environment_cancel import GridWorldCancel
env = GridWorldCancel(size=32, seed=30, p_plus=0.5)
print("vocab", env.unified_vocab_size, "n_actions", env.n_actions)
sds = {}
for arm in ("Signed_r4", "Abs_r4", "RoPE"):
    torch.manual_seed(30); np.random.seed(30)
    m = VARIANT_MAP[arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=32)
    sds[arm] = m
    print(arm, type(m).__name__, "params", sum(p.numel() for p in m.parameters()), "layers", len(m.layers),
          "mode", getattr(getattr(m, "action_to_lie", None), "mode", None))
a, b = sds["Signed_r4"].state_dict(), sds["Abs_r4"].state_dict()
print("same keys", list(a) == list(b), "max diff", max((a[k]-b[k]).abs().max().item() for k in a))
# torch RNG after construction (dropout stream)
for arm in ("Signed_r4","Abs_r4"):
    torch.manual_seed(30); VARIANT_MAP[arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=32)
    print(arm, "next rand", torch.rand(1).item())
m = sds["Signed_r4"]
print("omega head0", m.path_integrator.omega[0, :3].tolist(), "...", m.path_integrator.omega[0, -3:].tolist(), "2pi/32", 2*math.pi/32)
# initial Delta scale for action tokens
with torch.no_grad():
    for arm in ("Signed_r4","Abs_r4"):
        mm = sds[arm]; e = mm.token_emb(torch.tensor([[0,1,2,3]]))
        d = mm.action_to_lie(e)
        print(arm, "init |Delta| mean action+", d[0,0].abs().mean().item(), "action-", d[0,1].abs().mean().item(), "obs", d[0,2].abs().mean().item())
        # forward on the same tokens: identical?
    torch.manual_seed(0); tok, _, _ = env.generate_trajectory(64); tok = tok[None, :-1]
    sds["Signed_r4"].eval(); sds["Abs_r4"].eval()
    o1 = sds["Signed_r4"](tok); o2 = sds["Abs_r4"](tok)
    print("init forward max diff Signed vs Abs", (o1-o2).abs().max().item())

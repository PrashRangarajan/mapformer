import sys, numpy as np, torch
import mapformer, mapformer.model as M
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
print("pkg", mapformer.__file__)
def build(V, seed, d=256, L=2):
    torch.manual_seed(seed); np.random.seed(seed)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=seed)
    return env, VARIANT_MAP[V](vocab_size=env.unified_vocab_size, d_model=d, n_heads=2, n_layers=L, grid_size=64)
for seed in (1480, 1481):
    env, so = build("SignedAO4_r4", seed); _, ao = build("AbsAO4_r4", seed)
    sa, sb = so.state_dict(), ao.state_dict()
    print(f"seed {seed}: same keys {sa.keys()==sb.keys()}, tensors differing {[k for k in sa if not torch.equal(sa[k], sb[k])]}",
          f"params SO {sum(p.numel() for p in so.parameters()):,} AO {sum(p.numel() for p in ao.parameters()):,}",
          f"layers {len(so.layers)} d_head {so.layers[0].d_head} n_blocks {so.n_blocks} omega {tuple(so.path_integrator.omega.shape)}",
          f"W_in {tuple(so.action_to_lie.w_in.weight.shape)} W_out {tuple(so.action_to_lie.w_out.weight.shape)} act {so.act.tolist()} vocab {env.unified_vocab_size}")
    with torch.no_grad():
        ts, ta = so.step_table(), ao.step_table()
        nonact = [i for i in range(ts.shape[0]) if i not in so.act.tolist()]
        print("  non-action rows zero:", bool((ts[nonact] == 0).all() and (ta[nonact] == 0).all()),
              "| AO == |SO|:", torch.equal(ta, ts.abs()), "| AO min", float(ta.min()), "| SO has negatives:", bool((ts < 0).any()))
    om = so.path_integrator.omega.detach()
    print("  omega head0 first/last %.4f %.4f, ratio %.2f, monotone dec %s, heads equal %s" % (om[0,0], om[0,-1], om[0,0]/om[0,-1],
          bool((om[0,1:] < om[0,:-1]).all()), torch.equal(om[0], om[1])))
# both layers get the same cos/sin; fast-attn plain layer count
env, so = build("SignedAO4_r4", 1480)
seen = []
for l in so.layers:
    l.register_forward_pre_hook(lambda mod, args: seen.append((args[1].clone(), args[2].clone())))
tok, _, _, _ = env.generate_batch(2, 64)
so.eval(); so(tok[:, :-1])
print("layers called", len(seen), "| cos/sin identical across layers:", torch.equal(seen[0][0], seen[1][0]) and torch.equal(seen[0][1], seen[1][1]))
plain = [m for m in so.modules() if type(m).forward is M.WMTransformerLayer.forward]
print("plain WMTransformerLayer modules (fast-attn applies):", len(plain))
# phase only moves at action tokens
with torch.no_grad():
    d = so.step_table()[tok[:, :-1]]
    print("nonzero step at observation positions:", int((d[:, 1::2] != 0).any(-1).any(-1).sum()), "| action positions with nonzero step:", int((d[:, 0::2] != 0).any(-1).any(-1).float().mean()*100), "%")
# d128 L1 reference count
_, r = build("SignedAO4_r4", 1480, d=128, L=1); print("d128 L1 params", f"{sum(p.numel() for p in r.parameters()):,}")

"""Sensitivity of the per-word clock readouts (TW_NORMSTEP Amendment 1): inject a token-independent step (what an
uncancelled LayerNorm bias gives) of size frac x mean direction step, random direction, into committed MapWM
text-world models (s1, s5 map seeds; s0 clock seed); also the 8 unmodified seeds (baseline). Output: _out.txt."""
import torch, numpy as np; torch.set_num_threads(8)
from mapformer.tw_normstep_readouts import drift
from mapformer.analyze_textworld_secondary import load
from mapformer.environment_textworld import TextWorld
env=TextWorld(size=64,seed=0); g=torch.Generator().manual_seed(0)
R="/home/prashr/mapformer/runs/textworld/p0"
fmt=lambda d: f"full {d['full']:2d} ({d['full_rad']:.3f} rad)  opt {d['opt']:2d} ({d['opt_rad']:.4f} rad)"
for s in range(8):
    m=load(f"{R}/Vanilla_r4_L1_s{s}/Vanilla_r4.pt","cpu"); print(f"baseline s{s}: {fmt(drift(m))}")
for s in (1,5,0):
    m=load(f"{R}/Vanilla_r4_L1_s{s}/Vanilla_r4.pt","cpu")
    with torch.no_grad():
        D=m.action_to_lie(m.token_emb.weight[None])[0]; dirs=[i for a in range(4) for i in env.dir_ids[a]]
        ref=D[dirs].reshape(len(dirs),-1).norm(dim=1).mean()
        u=torch.randn(D.shape[1:],generator=g); u=u/u.norm()
    for frac in (0.005,0.01,0.02,0.05,0.1,0.2):
        h=m.action_to_lie.register_forward_hook(lambda mod,i,o: o+frac*ref*u)
        print(f"s{s} tick {frac:.3f}: {fmt(drift(m))}"); h.remove()
